"""
Спектральный анализ аудио — генерация мягких меток для speaker diarization.

Принцип: GMM-кластеризация по MFCC + спектральным полосам.
Выход — компактные текстовые метки (~3K токенов на 15 мин),
которые LLM использует как подсказки при разметке спикеров.

Зависимости: numpy, scipy, scikit-learn, ffmpeg (системный).
Установка:
    pip install numpy scipy scikit-learn
    apt install ffmpeg
"""

import os
import shutil
import subprocess
import tempfile
import logging
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.ndimage import median_filter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ─── Частотные полосы (аналог радио-мониторинга) ───────────────────
BANDS = [
    (80, 150),    # низкие (мужской F0)
    (150, 250),   # средние (F0 зона)
    (250, 400),   # верхние основные
    (400, 700),   # первая форманта
    (700, 1200),  # вторая форманта
    (1200, 2000), # высокие форманты
    (2000, 3500), # верхние обертоны
]

BAND_LABELS = [f"{lo}-{hi}" for lo, hi in BANDS]


def _merge_short_blocks(blocks: list, min_sec: float) -> list:
    min_sec = float(min_sec or 0.0)
    if min_sec <= 0 or len(blocks) < 2:
        return blocks

    out = [tuple(b) for b in blocks]
    changed = True
    while changed and len(out) > 1:
        changed = False
        next_out = []
        i = 0
        while i < len(out):
            start, end, label, conf = out[i]
            dur = end - start
            if dur >= min_sec:
                next_out.append(out[i])
                i += 1
                continue

            if i > 0 and i + 1 < len(out) and out[i - 1][2] == out[i + 1][2]:
                prev = next_out.pop()
                nxt = out[i + 1]
                prev_d = prev[1] - prev[0]
                cur_d = end - start
                next_d = nxt[1] - nxt[0]
                total = prev_d + cur_d + next_d
                merged_conf = (prev[3] * prev_d + conf * cur_d + nxt[3] * next_d) / total
                next_out.append((prev[0], nxt[1], prev[2], merged_conf))
                i += 2
                changed = True
                continue

            if i == 0:
                nxt = out[i + 1]
                nxt_d = nxt[1] - nxt[0]
                cur_d = end - start
                total = nxt_d + cur_d
                merged_conf = (nxt[3] * nxt_d + conf * cur_d) / total
                next_out.append((start, nxt[1], nxt[2], merged_conf))
                i += 2
                changed = True
                continue

            prev = next_out.pop()
            prev_d = prev[1] - prev[0]
            cur_d = end - start
            if i + 1 < len(out):
                nxt = out[i + 1]
                next_d = nxt[1] - nxt[0]
                if next_d > prev_d:
                    total = next_d + cur_d
                    merged_conf = (nxt[3] * next_d + conf * cur_d) / total
                    next_out.append(prev)
                    next_out.append((start, nxt[1], nxt[2], merged_conf))
                    i += 2
                else:
                    total = prev_d + cur_d
                    merged_conf = (prev[3] * prev_d + conf * cur_d) / total
                    next_out.append((prev[0], end, prev[2], merged_conf))
                    i += 1
                changed = True
                continue

            total = prev_d + cur_d
            merged_conf = (prev[3] * prev_d + conf * cur_d) / total
            next_out.append((prev[0], end, prev[2], merged_conf))
            i += 1
            changed = True

        out = []
        for b in next_out:
            if out and out[-1][2] == b[2]:
                prev = out[-1]
                prev_d = prev[1] - prev[0]
                b_d = b[1] - b[0]
                total = prev_d + b_d
                out[-1] = (prev[0], b[1], prev[2], (prev[3] * prev_d + b[3] * b_d) / total)
            else:
                out.append(b)

    return out


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is not available. Install system ffmpeg or add imageio-ffmpeg."
        ) from exc


def _convert_to_wav(src_path: str) -> str:
    """Конвертирует аудио в WAV mono 8kHz через ffmpeg."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        _ffmpeg_bin(), "-y", "-i", src_path,
        "-ac", "1", "-ar", "8000", "-f", "wav", wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        os.unlink(wav_path)
        raise RuntimeError(f"ffmpeg error: {result.stderr[:500]}")
    return wav_path


def _build_mel_filterbank(n_mels: int, frame_len: int, sr: int):
    """Мел-фильтрбанк для MFCC."""
    def hz2mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel2hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_pts = np.linspace(hz2mel(80), hz2mel(3800), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((frame_len + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, frame_len // 2))
    for m in range(1, n_mels + 1):
        fl, fc, fr = bins[m - 1], bins[m], bins[m + 1]
        for k in range(fl, fc):
            if fc > fl:
                fb[m - 1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc:
                fb[m - 1, k] = (fr - k) / (fr - fc)
    return fb


def _normalize_speech_ranges(
    speech_ranges: Optional[list[tuple[float, float]]],
    duration_sec: float,
) -> Optional[list[tuple[float, float]]]:
    if not speech_ranges:
        return None

    ranges = []
    for start, end in speech_ranges:
        try:
            s = max(0.0, float(start))
            e = min(float(duration_sec), float(end))
        except (TypeError, ValueError):
            continue
        if e > s:
            ranges.append((s, e))
    if not ranges:
        return None

    ranges.sort()
    merged = []
    cur_s, cur_e = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _frame_in_speech_ranges(start: float, end: float, ranges: Optional[list[tuple[float, float]]]) -> bool:
    if not ranges:
        return True
    for range_start, range_end in ranges:
        if end < range_start:
            return False
        if start <= range_end and end >= range_start:
            return True
    return False


def _extract_features(
    data: np.ndarray,
    sr: int,
    frame_len: int = 512,
    hop: int = 512,
    speech_ranges: Optional[list[tuple[float, float]]] = None,
):
    """
    Извлекает MFCC + спектральные полосы + центроид + энергию для каждого фрейма.
    Возвращает (features, energies, times, speech_mask).
    """
    n_frames = len(data) // hop - 1
    duration_sec = len(data) / float(sr or 1)
    guided_ranges = _normalize_speech_ranges(speech_ranges, duration_sec)
    freq_bins = fftfreq(frame_len, 1.0 / sr)[:frame_len // 2]
    filterbank = _build_mel_filterbank(13, frame_len, sr)
    window = np.hanning(frame_len)

    features = []
    energies = np.zeros(n_frames, dtype=np.float32)
    times = np.arange(n_frames, dtype=np.float32) * hop / sr
    speech_mask = np.zeros(n_frames, dtype=bool)
    speech_indices = []

    for i in range(n_frames):
        frame = data[i * hop: i * hop + frame_len]
        energy = float(np.sum(frame ** 2))
        energies[i] = energy
        frame_start = float(i * hop / sr)
        frame_end = float((i * hop + frame_len) / sr)

        if not _frame_in_speech_ranges(frame_start, frame_end, guided_ranges):
            continue

        if energy < 0.0005:
            continue

        speech_mask[i] = True
        speech_indices.append(i)

        windowed = frame * window
        power = np.abs(fft(windowed))[:frame_len // 2] ** 2

        # MFCC
        mel_spec = np.log(np.dot(filterbank, power) + 1e-12)
        mfcc = np.zeros(13)
        arange = np.arange(13)
        for k in range(13):
            mfcc[k] = np.sum(mel_spec * np.cos(np.pi * k * (arange + 0.5) / 13))

        # Энергия по полосам
        total_e = np.sum(power) + 1e-12
        band_feats = []
        for lo, hi in BANDS:
            mask = (freq_bins >= lo) & (freq_bins < hi)
            band_feats.append(np.sum(power[mask]) / total_e)

        # Центроид
        mag = np.sqrt(power)
        centroid = np.sum(freq_bins * mag) / (np.sum(mag) + 1e-12)

        features.append(np.concatenate([mfcc, band_feats, [centroid, energy]]))

    return (
        np.array(features) if features else np.empty((0, 22)),
        energies,
        times,
        speech_mask,
        np.array(speech_indices),
    )


def _estimate_f0_track(
    data: np.ndarray,
    sr: int,
    speech_indices: np.ndarray,
    frame_len: int = 512,
    hop: int = 512,
    fmin: float = 75.0,
    fmax: float = 350.0,
) -> tuple[np.ndarray, np.ndarray]:
    f0s = np.zeros(len(speech_indices), dtype=np.float32)
    clarity = np.zeros(len(speech_indices), dtype=np.float32)
    min_lag = max(1, int(np.ceil(sr / fmax)))
    max_lag = min(frame_len - 1, int(np.floor(sr / fmin)))

    for out_i, frame_i in enumerate(speech_indices):
        start = int(frame_i * hop)
        frame = data[start:start + frame_len]
        if len(frame) < frame_len:
            continue
        frame = frame.astype(np.float32)
        frame = frame - frame.mean()
        frame = frame * np.hanning(frame_len)
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        if ac[0] <= 1e-8:
            continue
        search = ac[min_lag:max_lag]
        if len(search) == 0:
            continue
        scores = search / ac[0]
        if len(scores) >= 3:
            peak_rel = np.where((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:]))[0] + 1
        else:
            peak_rel = np.array([], dtype=int)
        if len(peak_rel):
            best = float(scores[peak_rel].max())
            valid = peak_rel[scores[peak_rel] >= max(0.25, best * 0.6)]
            rel = int(valid[-1] if len(valid) else peak_rel[int(np.argmax(scores[peak_rel]))])
        else:
            rel = int(np.argmax(scores))
        lag = min_lag + rel
        score = float(ac[lag] / ac[0])
        if score >= 0.25:
            f0s[out_i] = float(sr / lag)
            clarity[out_i] = score
    return f0s, clarity


def _cluster_speakers(
    features: np.ndarray,
    speech_times: np.ndarray,
    n_speakers: int = 2,
    ivr_cutoff_sec: float = 0.0,
):
    """
    GMM-кластеризация.
    Если ivr_cutoff_sec > 0 — первые N секунд помечаются как отдельный кластер (IVR / приветствие).
    """
    if len(features) == 0:
        return np.array([]), np.array([]), {}

    scaler = StandardScaler()

    if ivr_cutoff_sec > 0:
        late_mask = speech_times > ivr_cutoff_sec
        early_mask = ~late_mask

        if late_mask.sum() < 20:
            # Слишком мало данных после cutoff — кластеризуем всё
            ivr_cutoff_sec = 0.0
        else:
            X_all = scaler.fit_transform(features)
            X_late = X_all[late_mask]

            gmm = GaussianMixture(
                n_components=n_speakers,
                covariance_type="full",
                random_state=42,
                n_init=5,
                max_iter=300,
            )
            gmm.fit(X_late)

            labels = np.full(len(features), n_speakers, dtype=int)  # IVR = n_speakers
            labels[late_mask] = gmm.predict(X_late)
            probs = np.zeros((len(features), n_speakers))
            probs[late_mask] = gmm.predict_proba(X_late)

            # Профили кластеров
            profiles = {}
            for c in range(n_speakers):
                mask = labels == c
                if mask.sum() > 0:
                    profiles[f"C{c}"] = {
                        "centroid_hz": int(features[mask, -2].mean()),
                        "n_frames": int(mask.sum()),
                    }
            early_count = int(early_mask.sum())
            if early_count > 0:
                profiles[f"C{n_speakers}"] = {
                    "centroid_hz": int(features[early_mask, -2].mean()),
                    "n_frames": early_count,
                    "note": f"IVR/приветствие (0-{ivr_cutoff_sec:.0f}с)",
                }

            return labels, probs, profiles

    # Без IVR cutoff — простая кластеризация
    X = scaler.fit_transform(features)
    gmm = GaussianMixture(
        n_components=n_speakers,
        covariance_type="full",
        random_state=42,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    profiles = {}
    for c in range(n_speakers):
        mask = labels == c
        if mask.sum() > 0:
            profiles[f"C{c}"] = {
                "centroid_hz": int(features[mask, -2].mean()),
                "n_frames": int(mask.sum()),
            }

    return labels, probs, profiles


def _window_embedding(win_features: np.ndarray, win_f0: Optional[np.ndarray] = None) -> np.ndarray:
    x = win_features.copy()
    x[:, -1] = np.log1p(np.maximum(x[:, -1], 0))
    mfcc = x[:, :13]
    bands = x[:, 13:20]
    centroid = x[:, -2]
    log_energy = x[:, -1]
    pitch_features = np.zeros(6, dtype=np.float32)
    if win_f0 is not None:
        voiced = win_f0[win_f0 > 0]
        if len(voiced) >= 2:
            pitch_features = np.array(
                [
                    np.median(voiced),
                    voiced.mean(),
                    voiced.std(),
                    np.percentile(voiced, 25),
                    np.percentile(voiced, 75),
                    len(voiced) / float(len(win_f0) or 1),
                ],
                dtype=np.float32,
            )
    return np.concatenate(
        [
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
            bands.mean(axis=0),
            bands.std(axis=0),
            [
                centroid.mean(),
                centroid.std(),
                np.percentile(centroid, 25),
                np.percentile(centroid, 75),
                log_energy.mean(),
                log_energy.std(),
            ],
            pitch_features,
        ]
    )


def _cluster_speakers_windowed(
    features: np.ndarray,
    speech_times: np.ndarray,
    n_speakers: int = 2,
    ivr_cutoff_sec: float = 0.0,
    window_sec: float = 2.5,
    step_sec: float = 0.5,
    f0_values: Optional[np.ndarray] = None,
):
    if len(features) == 0 or len(speech_times) == 0:
        return np.array([]), np.array([]), {}, []

    duration = float(speech_times[-1])
    window_sec = max(0.8, float(window_sec or 2.5))
    step_sec = max(0.2, float(step_sec or 0.5))
    min_frames = max(8, int(window_sec / 0.064 * 0.25))

    centers = np.arange(window_sec / 2, duration + step_sec, step_sec)
    embeddings = []
    windows = []
    for center in centers:
        start = max(0.0, center - window_sec / 2)
        end = center + window_sec / 2
        mask = (speech_times >= start) & (speech_times <= end)
        if mask.sum() < min_frames:
            continue
        win = features[mask]
        win_f0 = f0_values[mask] if f0_values is not None and len(f0_values) == len(features) else None
        embeddings.append(_window_embedding(win, win_f0))
        windows.append(
            {
                "start": start,
                "end": end,
                "center": center,
                "speech_frames": int(mask.sum()),
            }
        )

    if len(embeddings) < max(2, n_speakers):
        return _cluster_speakers(features, speech_times, n_speakers, ivr_cutoff_sec) + ([],)

    X = StandardScaler().fit_transform(np.vstack(embeddings))

    if ivr_cutoff_sec > 0:
        late_mask = np.array([w["center"] > ivr_cutoff_sec for w in windows])
        if late_mask.sum() >= max(8, n_speakers):
            labels_w = np.full(len(windows), n_speakers, dtype=int)
            probs_w = np.zeros((len(windows), n_speakers + 1), dtype=float)
            gmm = GaussianMixture(
                n_components=n_speakers,
                covariance_type="diag",
                random_state=42,
                n_init=10,
                max_iter=500,
            )
            gmm.fit(X[late_mask])
            late_labels = gmm.predict(X[late_mask])
            late_probs = gmm.predict_proba(X[late_mask])
            labels_w[late_mask] = late_labels
            probs_w[late_mask, :n_speakers] = late_probs
            probs_w[~late_mask, n_speakers] = 1.0
        else:
            ivr_cutoff_sec = 0.0

    if ivr_cutoff_sec <= 0:
        gmm = GaussianMixture(
            n_components=n_speakers,
            covariance_type="diag",
            random_state=42,
            n_init=10,
            max_iter=500,
        )
        gmm.fit(X)
        labels_w = gmm.predict(X)
        probs_w = gmm.predict_proba(X)

    centers_w = np.array([w["center"] for w in windows])
    labels = np.zeros(len(features), dtype=int)
    probs = np.zeros((len(features), probs_w.shape[1]), dtype=float)
    for i, t in enumerate(speech_times):
        nearest = int(np.argmin(np.abs(centers_w - t)))
        labels[i] = labels_w[nearest]
        probs[i] = probs_w[nearest]

    profiles = {}
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        if mask.sum() > 0:
            window_count = int((labels_w == c).sum())
            voiced = f0_values[mask] if f0_values is not None and len(f0_values) == len(features) else np.array([])
            voiced = voiced[voiced > 0]
            profiles[f"C{c}"] = {
                "centroid_hz": int(features[mask, -2].mean()),
                "n_frames": int(mask.sum()),
                "n_windows": window_count,
                "f0_hz": int(np.median(voiced)) if len(voiced) else None,
            }
            if ivr_cutoff_sec > 0 and c == n_speakers:
                profiles[f"C{c}"]["note"] = f"IVR/приветствие (0-{ivr_cutoff_sec:.0f}с)"

    for w, label, prob in zip(windows, labels_w, probs_w):
        w["speaker"] = f"C{label}"
        w["confidence"] = float(prob[label]) if label < len(prob) else float(prob.max())

    return labels, probs, profiles, windows


def _build_compact_output(
    labels: np.ndarray,
    probs: np.ndarray,
    profiles: dict,
    speech_indices: np.ndarray,
    times: np.ndarray,
    energies: np.ndarray,
    speech_mask: np.ndarray,
    n_speakers: int,
    confidence_cutoff: float = 0.55,
    min_block_sec: float = 0.5,
) -> str:
    """
    Формирует компактные текстовые метки для LLM-промпта.
    """
    n_frames = len(times)

    # ── Полный таймлайн ──
    full_labels = np.full(n_frames, -1, dtype=int)
    for idx, frame_idx in enumerate(speech_indices):
        full_labels[frame_idx] = labels[idx]

    # ── Сглаживание ──
    temp = full_labels.copy()
    for i in range(1, len(temp)):
        if temp[i] == -1 and temp[i - 1] != -1:
            temp[i] = temp[i - 1]
    smoothed = median_filter(temp, size=25)
    smoothed[~speech_mask] = -1

    # ── Блоки ──
    blocks = []
    cur_label = smoothed[0]
    seg_start = 0
    for i in range(1, n_frames):
        if smoothed[i] != cur_label:
            if cur_label >= 0:
                dur = times[i] - times[seg_start]
                if dur >= 0.3:
                    # Средняя уверенность для блока
                    block_mask = speech_indices[
                        (speech_indices >= seg_start) & (speech_indices < i)
                    ]
                    if len(block_mask) > 0:
                        block_idx = np.searchsorted(speech_indices, block_mask)
                        block_idx = block_idx[block_idx < len(labels)]
                        if len(block_idx) > 0 and cur_label < probs.shape[1]:
                            conf = float(probs[block_idx, cur_label].mean())
                        else:
                            conf = 0.9
                    else:
                        conf = 0.9
                    blocks.append((times[seg_start], times[i], cur_label, conf))
            cur_label = smoothed[i]
            seg_start = i

    # Последний блок
    if cur_label >= 0:
        dur = times[-1] - times[seg_start]
        if dur >= 0.3:
            blocks.append((times[seg_start], times[-1], cur_label, 0.9))

    # Мержим соседние с одинаковым кластером
    merged = []
    for b in blocks:
        if merged and merged[-1][2] == b[2]:
            merged[-1] = (merged[-1][0], b[1], b[2], (merged[-1][3] + b[3]) / 2)
        else:
            merged.append(list(b))

    # Убираем короткие (<0.5с)
    final_blocks = []
    for b in merged:
        if (b[1] - b[0]) < 0.5 and final_blocks:
            final_blocks[-1] = (final_blocks[-1][0], b[1], final_blocks[-1][2], final_blocks[-1][3])
        else:
            final_blocks.append(tuple(b))
    final_blocks = _merge_short_blocks(final_blocks, min_block_sec)

    # ── Точки смены ──
    change_points = []
    for i in range(1, len(final_blocks)):
        prev = final_blocks[i - 1]
        curr = final_blocks[i]
        if prev[2] != curr[2]:
            conf = curr[3]
            if conf >= confidence_cutoff:
                change_points.append((curr[0], prev[2], curr[2], conf))

    # ── Всплески энергии ──
    ws = 30
    smoothed_e = np.convolve(energies, np.ones(ws) / ws, mode="same")
    threshold = np.mean(smoothed_e) + 3 * np.std(smoothed_e)
    spikes = []
    in_spike = False
    spike_start = 0
    for i in range(len(smoothed_e)):
        if smoothed_e[i] > threshold and not in_spike:
            spike_start = i
            in_spike = True
        elif smoothed_e[i] <= threshold and in_spike:
            intensity = float(smoothed_e[spike_start:i].max() / threshold)
            spikes.append((times[spike_start], times[i], intensity))
            in_spike = False

    # ── Формируем текст ──
    lines = []
    lines.append("<audio_analysis>")
    lines.append("<info>")
    lines.append(f"Длительность: {times[-1]:.0f}с | Метод: GMM(MFCC+спектр)")

    profile_parts = []
    for name, p in sorted(profiles.items()):
        note = p.get("note", "")
        extra = f" ({note})" if note else ""
        profile_parts.append(f"{name}=центроид~{p['centroid_hz']}Hz{extra}")
    lines.append("Кластеры: " + ", ".join(profile_parts))
    lines.append(
        "Уверенность: 1.0=точно, 0.55=сомнительно. Ниже 0.55 отсечено."
    )
    lines.append("</info>")

    lines.append(f'<speaker_blocks count="{len(final_blocks)}">')
    for start, end, cluster, conf in final_blocks:
        lines.append(f"{start:.1f}-{end:.1f}|C{cluster}|{conf:.2f}")
    lines.append("</speaker_blocks>")

    lines.append(f'<change_points count="{len(change_points)}">')
    for t, from_c, to_c, conf in change_points:
        lines.append(f"{t:.1f}|C{from_c}→C{to_c}|{conf:.2f}")
    lines.append("</change_points>")

    if spikes:
        lines.append(f'<energy_spikes count="{len(spikes)}">')
        for start, end, intensity in spikes:
            lines.append(f"{start:.1f}-{end:.1f}|intensity:{intensity:.1f}")
        lines.append("</energy_spikes>")

    lines.append("</audio_analysis>")

    return "\n".join(lines)


def _fmt_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _round_float(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


def _speaker_name(label: int) -> str:
    return "UNK" if int(label) < 0 else f"C{int(label)}"


def _block_indices(
    start: float,
    end: float,
    times: np.ndarray,
    speech_indices: np.ndarray,
    n_features: int,
) -> np.ndarray:
    start_i = np.searchsorted(times, start, side="left")
    end_i = np.searchsorted(times, end, side="right")
    frame_mask = speech_indices[(speech_indices >= start_i) & (speech_indices < end_i)]
    feature_idx = np.searchsorted(speech_indices, frame_mask)
    return feature_idx[feature_idx < n_features]


def _refine_short_blocks_by_anchors(
    final_blocks: list,
    features: np.ndarray,
    speech_indices: np.ndarray,
    times: np.ndarray,
    f0_values: Optional[np.ndarray],
    short_sec: float = 2.5,
    anchor_min_sec: float = 8.0,
    confidence_cutoff: float = 0.55,
) -> tuple[list, dict]:
    if not final_blocks or short_sec <= 0:
        return final_blocks, {"enabled": False, "unknown_blocks": 0, "reassigned_blocks": 0}

    block_embeddings = []
    valid_block_ids = []
    for i, (start, end, label, conf) in enumerate(final_blocks):
        idx = _block_indices(start, end, times, speech_indices, len(features))
        if len(idx) < 8:
            continue
        win_f0 = f0_values[idx] if f0_values is not None and len(f0_values) == len(features) else None
        block_embeddings.append(_window_embedding(features[idx], win_f0))
        valid_block_ids.append(i)

    if len(block_embeddings) < 3:
        return final_blocks, {"enabled": False, "unknown_blocks": 0, "reassigned_blocks": 0}

    embeddings = np.vstack(block_embeddings)
    valid_lookup = {block_i: emb_i for emb_i, block_i in enumerate(valid_block_ids)}

    anchor_ids = []
    for i, (start, end, label, conf) in enumerate(final_blocks):
        if i not in valid_lookup or int(label) < 0:
            continue
        if (end - start) >= anchor_min_sec and conf >= confidence_cutoff:
            anchor_ids.append(i)

    anchor_labels = sorted({int(final_blocks[i][2]) for i in anchor_ids})
    if len(anchor_labels) < 2:
        return final_blocks, {"enabled": False, "unknown_blocks": 0, "reassigned_blocks": 0}

    anchor_emb_indices = [valid_lookup[i] for i in anchor_ids]
    scaler = StandardScaler().fit(embeddings[anchor_emb_indices])
    scaled = scaler.transform(embeddings)

    prototypes = {}
    radii = {}
    for label in anchor_labels:
        ids = [valid_lookup[i] for i in anchor_ids if int(final_blocks[i][2]) == label]
        vals = scaled[ids]
        proto = vals.mean(axis=0)
        prototypes[label] = proto
        if len(vals) > 1:
            d = np.linalg.norm(vals - proto, axis=1)
            radii[label] = max(2.0, float(np.median(d) + 2.0 * np.std(d)))
        else:
            radii[label] = 3.0

    refined = []
    unknown_blocks = 0
    reassigned_blocks = 0
    for i, block in enumerate(final_blocks):
        start, end, label, conf = block
        dur = end - start
        needs_review = dur < short_sec or conf < confidence_cutoff
        if not needs_review or i not in valid_lookup:
            refined.append(tuple(block))
            continue

        emb = scaled[valid_lookup[i]]
        distances = sorted(
            ((float(np.linalg.norm(emb - proto)), label_id) for label_id, proto in prototypes.items()),
            key=lambda x: x[0],
        )
        best_dist, best_label = distances[0]
        second_dist = distances[1][0] if len(distances) > 1 else best_dist + 1.0
        radius = radii.get(best_label, 3.0)
        margin_ok = (second_dist - best_dist) >= 0.35 or (second_dist / max(best_dist, 1e-6)) >= 1.12
        distance_ok = best_dist <= radius * 1.35

        if distance_ok and margin_ok:
            new_conf = max(0.05, min(0.95, 1.0 - best_dist / max(radius * 1.35, 1e-6)))
            if int(best_label) != int(label):
                reassigned_blocks += 1
            refined.append((start, end, int(best_label), new_conf))
        else:
            unknown_blocks += 1
            refined.append((start, end, -1, min(float(conf), 0.2)))

    merged = []
    for b in refined:
        if merged and merged[-1][2] == b[2] and b[2] >= 0:
            prev = merged[-1]
            prev_d = prev[1] - prev[0]
            b_d = b[1] - b[0]
            total = prev_d + b_d
            merged[-1] = (prev[0], b[1], prev[2], (prev[3] * prev_d + b[3] * b_d) / total)
        else:
            merged.append(b)

    return merged, {
        "enabled": True,
        "unknown_blocks": unknown_blocks,
        "reassigned_blocks": reassigned_blocks,
        "anchor_blocks": len(anchor_ids),
        "anchor_speakers": [_speaker_name(x) for x in anchor_labels],
    }


def _build_anchor_model(
    final_blocks: list,
    features: np.ndarray,
    speech_indices: np.ndarray,
    times: np.ndarray,
    f0_values: Optional[np.ndarray],
    exclude_ids: set[int],
    anchor_min_sec: float,
    confidence_cutoff: float,
) -> Optional[dict]:
    anchor_embeddings = []
    anchor_labels = []
    for i, (start, end, label, conf) in enumerate(final_blocks):
        if i in exclude_ids or int(label) < 0:
            continue
        if (end - start) < anchor_min_sec or conf < confidence_cutoff:
            continue
        idx = _block_indices(start, end, times, speech_indices, len(features))
        if len(idx) < 12:
            continue
        win_f0 = f0_values[idx] if f0_values is not None and len(f0_values) == len(features) else None
        anchor_embeddings.append(_window_embedding(features[idx], win_f0))
        anchor_labels.append(int(label))

    labels = sorted(set(anchor_labels))
    if len(labels) < 2:
        return None

    X = np.vstack(anchor_embeddings)
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    prototypes = {}
    radii = {}
    for label in labels:
        ids = [i for i, x in enumerate(anchor_labels) if x == label]
        vals = scaled[ids]
        proto = vals.mean(axis=0)
        prototypes[label] = proto
        if len(vals) > 1:
            d = np.linalg.norm(vals - proto, axis=1)
            radii[label] = max(2.0, float(np.median(d) + 2.0 * np.std(d)))
        else:
            radii[label] = 3.0
    return {"scaler": scaler, "prototypes": prototypes, "radii": radii, "labels": labels}


def _assign_to_anchor_model(embedding: np.ndarray, model: Optional[dict]) -> tuple[int, float]:
    if not model:
        return -1, 0.2
    emb = model["scaler"].transform([embedding])[0]
    distances = sorted(
        ((float(np.linalg.norm(emb - proto)), label) for label, proto in model["prototypes"].items()),
        key=lambda x: x[0],
    )
    if not distances:
        return -1, 0.2
    best_dist, best_label = distances[0]
    second_dist = distances[1][0] if len(distances) > 1 else best_dist + 1.0
    radius = model["radii"].get(best_label, 3.0)
    margin_ok = (second_dist - best_dist) >= 0.25 or (second_dist / max(best_dist, 1e-6)) >= 1.08
    distance_ok = best_dist <= radius * 1.5
    if not (margin_ok and distance_ok):
        return -1, 0.2
    conf = max(0.05, min(0.95, 1.0 - best_dist / max(radius * 1.5, 1e-6)))
    return int(best_label), conf


def _detect_transient_speakers(
    features: np.ndarray,
    speech_times: np.ndarray,
    f0_values: Optional[np.ndarray],
    labels: np.ndarray,
    probs: np.ndarray,
    profiles: dict,
    base_speakers: int,
    search_sec: float = 60.0,
    late_start_sec: float = 75.0,
    min_sec: float = 4.0,
    distance_threshold: float = 9.0,
    local_speakers: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    if search_sec <= 0 or local_speakers < 2 or len(features) == 0:
        return labels, probs, profiles, {"enabled": False, "added": 0}

    early_mask = speech_times <= search_sec
    if early_mask.sum() < 40:
        return labels, probs, profiles, {"enabled": True, "added": 0, "reason": "too_few_early_frames"}

    late_embeddings = []
    late_names = []
    for label in range(base_speakers):
        mask = (speech_times >= late_start_sec) & (labels == label)
        if mask.sum() < 40:
            continue
        late_f0 = f0_values[mask] if f0_values is not None and len(f0_values) == len(features) else None
        late_embeddings.append(_window_embedding(features[mask], late_f0))
        late_names.append(label)

    if len(late_embeddings) < 2:
        return labels, probs, profiles, {"enabled": True, "added": 0, "reason": "too_few_late_prototypes"}

    early_features = features[early_mask]
    early_times = speech_times[early_mask]
    early_f0 = f0_values[early_mask] if f0_values is not None and len(f0_values) == len(features) else None
    local_k = max(2, min(int(local_speakers), max(2, int(early_mask.sum() // 35))))
    local_labels, _local_probs, _local_profiles, _local_windows = _cluster_speakers_windowed(
        early_features,
        early_times,
        local_k,
        0.0,
        window_sec=1.5,
        step_sec=0.25,
        f0_values=early_f0,
    )
    if len(local_labels) == 0:
        return labels, probs, profiles, {"enabled": True, "added": 0, "reason": "local_cluster_empty"}

    candidates = []
    early_indices = np.where(early_mask)[0]
    candidate_embeddings = []
    for local_label in sorted(set(local_labels.tolist())):
        local_mask = local_labels == local_label
        if local_mask.sum() < 20:
            continue
        duration = float(local_mask.sum() * (np.median(np.diff(speech_times)) if len(speech_times) > 1 else 0.064))
        if duration < min_sec:
            continue
        frame_indices = early_indices[local_mask]
        c_times = speech_times[frame_indices]
        c_f0 = f0_values[frame_indices] if f0_values is not None and len(f0_values) == len(features) else None
        emb = _window_embedding(features[frame_indices], c_f0)
        candidate_embeddings.append(emb)
        voiced = c_f0[c_f0 > 0] if c_f0 is not None else np.array([])
        candidates.append(
            {
                "local_label": int(local_label),
                "frame_indices": frame_indices,
                "duration": duration,
                "first": float(c_times[0]),
                "last": float(c_times[-1]),
                "f0_hz": int(np.median(voiced)) if len(voiced) else None,
                "centroid_hz": int(features[frame_indices, -2].mean()),
            }
        )

    if not candidates:
        return labels, probs, profiles, {"enabled": True, "added": 0, "reason": "no_candidate_duration"}

    scaler = StandardScaler().fit(np.vstack(late_embeddings + candidate_embeddings))
    late_scaled = scaler.transform(np.vstack(late_embeddings))
    cand_scaled = scaler.transform(np.vstack(candidate_embeddings))

    accepted = []
    for i, cand in enumerate(candidates):
        distances = np.linalg.norm(late_scaled - cand_scaled[i], axis=1)
        min_dist = float(distances.min())
        nearest = int(late_names[int(np.argmin(distances))])
        cand["distance"] = min_dist
        cand["nearest_main"] = _speaker_name(nearest)
        # A transient speaker should be acoustically far from late main-speaker prototypes
        # and concentrated near the beginning of the call.
        if min_dist >= distance_threshold:
            accepted.append(cand)

    if not accepted:
        return labels, probs, profiles, {
            "enabled": True,
            "added": 0,
            "reason": "no_far_candidate",
            "candidates": [
                {
                    "duration": _round_float(c["duration"], 1),
                    "range": f"{_fmt_mmss(c['first'])}-{_fmt_mmss(c['last'])}",
                    "distance": _round_float(c["distance"], 2),
                    "nearest_main": c["nearest_main"],
                    "f0_hz": c["f0_hz"],
                }
                for c in candidates
            ],
        }

    accepted.sort(key=lambda c: c["distance"], reverse=True)
    new_labels = labels.copy()
    new_probs = np.zeros((len(labels), probs.shape[1] + len(accepted)), dtype=float)
    new_probs[:, :probs.shape[1]] = probs
    new_profiles = dict(profiles)

    added = []
    for offset, cand in enumerate(accepted):
        new_label = base_speakers + offset
        idx = cand["frame_indices"]
        new_labels[idx] = new_label
        new_probs[idx, :] = 0.0
        new_probs[idx, new_label] = min(0.95, 0.55 + (cand["distance"] - distance_threshold) / max(distance_threshold, 1.0) * 0.2)
        key = f"C{new_label}"
        new_profiles[key] = {
            "centroid_hz": cand["centroid_hz"],
            "n_frames": int(len(idx)),
            "f0_hz": cand["f0_hz"],
            "note": f"transient { _fmt_mmss(cand['first'])}-{_fmt_mmss(cand['last'])}",
        }
        added.append(
            {
                "speaker": key,
                "range": f"{_fmt_mmss(cand['first'])}-{_fmt_mmss(cand['last'])}",
                "duration": _round_float(cand["duration"], 1),
                "distance": _round_float(cand["distance"], 2),
                "nearest_main": cand["nearest_main"],
                "f0_hz": cand["f0_hz"],
            }
        )

    return new_labels, new_probs, new_profiles, {
        "enabled": True,
        "added": len(added),
        "search_sec": search_sec,
        "late_start_sec": late_start_sec,
        "distance_threshold": distance_threshold,
        "added_speakers": added,
    }


def _local_cluster_blocks(
    block_features: np.ndarray,
    block_times: np.ndarray,
    block_f0: Optional[np.ndarray],
    window_sec: float,
    step_sec: float,
) -> list:
    if len(block_features) < 20:
        return []
    labels, probs, _profiles, _windows = _cluster_speakers_windowed(
        block_features,
        block_times,
        n_speakers=2,
        ivr_cutoff_sec=0.0,
        window_sec=window_sec,
        step_sec=step_sec,
        f0_values=block_f0,
    )
    if len(labels) == 0:
        return []
    frame_step = float(np.median(np.diff(block_times))) if len(block_times) > 1 else 0.064
    out = []
    start_i = 0
    cur = int(labels[0])
    for i in range(1, len(labels)):
        gap = block_times[i] - block_times[i - 1]
        if int(labels[i]) != cur or gap > 0.8:
            idx = np.arange(start_i, i)
            if len(idx):
                start = float(block_times[start_i])
                end = float(block_times[i - 1] + frame_step)
                conf = float(probs[idx, cur].mean()) if cur < probs.shape[1] else 0.5
                if end - start >= 0.35:
                    out.append((start, end, cur, conf, idx))
            start_i = i
            cur = int(labels[i])
    idx = np.arange(start_i, len(labels))
    if len(idx):
        start = float(block_times[start_i])
        end = float(block_times[-1] + frame_step)
        conf = float(probs[idx, cur].mean()) if cur < probs.shape[1] else 0.5
        if end - start >= 0.35:
            out.append((start, end, cur, conf, idx))

    merged = []
    for b in out:
        if merged and merged[-1][2] == b[2]:
            prev = merged[-1]
            merged[-1] = (prev[0], b[1], prev[2], (prev[3] + b[3]) / 2, np.concatenate([prev[4], b[4]]))
        else:
            merged.append(b)
    return merged


def _split_mixed_blocks(
    final_blocks: list,
    features: np.ndarray,
    speech_indices: np.ndarray,
    times: np.ndarray,
    f0_values: Optional[np.ndarray],
    confidence_cutoff: float,
    anchor_min_sec: float,
    mixed_check_sec: float = 0.0,
    mixed_min_part_sec: float = 1.2,
    micro_window_sec: float = 1.2,
    micro_step_sec: float = 0.25,
) -> tuple[list, dict]:
    if mixed_check_sec <= 0:
        return final_blocks, {"enabled": False, "mixed_blocks": 0, "created_subblocks": 0}

    candidates = {}
    exclude_ids = set()
    for i, (start, end, label, conf) in enumerate(final_blocks):
        if (end - start) < mixed_check_sec:
            continue
        idx = _block_indices(start, end, times, speech_indices, len(features))
        if len(idx) < 30:
            continue
        block_f0 = f0_values[idx] if f0_values is not None and len(f0_values) == len(features) else None
        local = _local_cluster_blocks(features[idx], times[speech_indices[idx]], block_f0, micro_window_sec, micro_step_sec)
        if len(local) < 2:
            continue
        totals = {}
        for s, e, local_label, _local_conf, _local_idx in local:
            totals[local_label] = totals.get(local_label, 0.0) + (e - s)
        meaningful = [dur for dur in totals.values() if dur >= mixed_min_part_sec]
        if len(meaningful) < 2:
            continue
        candidates[i] = (idx, local)
        exclude_ids.add(i)

    if not candidates:
        return final_blocks, {"enabled": True, "mixed_blocks": 0, "created_subblocks": 0}

    model = _build_anchor_model(
        final_blocks,
        features,
        speech_indices,
        times,
        f0_values,
        exclude_ids,
        anchor_min_sec,
        confidence_cutoff,
    )

    split_blocks = []
    created = 0
    unknown = 0
    for i, block in enumerate(final_blocks):
        if i not in candidates:
            split_blocks.append(tuple(block))
            continue
        idx, local = candidates[i]
        for start, end, _local_label, local_conf, local_idx in local:
            feature_idx = idx[local_idx]
            if len(feature_idx) < 8:
                label, conf = -1, 0.2
            else:
                win_f0 = f0_values[feature_idx] if f0_values is not None and len(f0_values) == len(features) else None
                emb = _window_embedding(features[feature_idx], win_f0)
                label, conf = _assign_to_anchor_model(emb, model)
                if label < 0:
                    unknown += 1
                    conf = min(float(local_conf), 0.2)
                else:
                    conf = max(conf, min(float(local_conf), 0.95) * 0.75)
            split_blocks.append((start, end, label, conf))
            created += 1

    merged = []
    for b in sorted(split_blocks, key=lambda x: x[0]):
        if merged and merged[-1][2] == b[2] and b[2] >= 0:
            prev = merged[-1]
            gap = b[0] - prev[1]
            if gap <= 0.35:
                prev_d = prev[1] - prev[0]
                b_d = b[1] - b[0]
                total = prev_d + b_d
                merged[-1] = (prev[0], b[1], prev[2], (prev[3] * prev_d + b[3] * b_d) / total)
                continue
        merged.append(b)

    return merged, {
        "enabled": True,
        "mixed_blocks": len(candidates),
        "created_subblocks": created,
        "unknown_subblocks": unknown,
        "anchor_available": bool(model),
    }


def _build_structured_report(
    labels: np.ndarray,
    probs: np.ndarray,
    profiles: dict,
    speech_indices: np.ndarray,
    times: np.ndarray,
    energies: np.ndarray,
    speech_mask: np.ndarray,
    features: np.ndarray,
    n_speakers: int,
    confidence_cutoff: float = 0.55,
    group_sec: float = 60.0,
    min_block_sec: float = 0.5,
    f0_values: Optional[np.ndarray] = None,
    short_uncertain_sec: float = 2.5,
    anchor_min_sec: float = 8.0,
    mixed_check_sec: float = 6.0,
    mixed_min_part_sec: float = 1.2,
    micro_window_sec: float = 1.2,
    micro_step_sec: float = 0.25,
) -> dict:
    n_frames = len(times)
    if n_frames == 0:
        return {"ok": False, "error": "No audio frames after conversion."}

    full_labels = np.full(n_frames, -1, dtype=int)
    for idx, frame_idx in enumerate(speech_indices):
        full_labels[frame_idx] = labels[idx]

    temp = full_labels.copy()
    for i in range(1, len(temp)):
        if temp[i] == -1 and temp[i - 1] != -1:
            temp[i] = temp[i - 1]
    smoothed = median_filter(temp, size=25)
    smoothed[~speech_mask] = -1

    blocks = []
    cur_label = smoothed[0]
    seg_start = 0
    for i in range(1, n_frames):
        if smoothed[i] != cur_label:
            if cur_label >= 0:
                dur = times[i] - times[seg_start]
                if dur >= 0.3:
                    block_mask = speech_indices[
                        (speech_indices >= seg_start) & (speech_indices < i)
                    ]
                    if len(block_mask) > 0:
                        block_idx = np.searchsorted(speech_indices, block_mask)
                        block_idx = block_idx[block_idx < len(labels)]
                        if len(block_idx) > 0 and cur_label < probs.shape[1]:
                            conf = float(probs[block_idx, cur_label].mean())
                        else:
                            conf = 0.9
                    else:
                        conf = 0.9
                    blocks.append((times[seg_start], times[i], cur_label, conf))
            cur_label = smoothed[i]
            seg_start = i

    if cur_label >= 0:
        dur = times[-1] - times[seg_start]
        if dur >= 0.3:
            blocks.append((times[seg_start], times[-1], cur_label, 0.9))

    merged = []
    for b in blocks:
        if merged and merged[-1][2] == b[2]:
            merged[-1] = (merged[-1][0], b[1], b[2], (merged[-1][3] + b[3]) / 2)
        else:
            merged.append(list(b))

    final_blocks = []
    for b in merged:
        if (b[1] - b[0]) < 0.5 and final_blocks:
            final_blocks[-1] = (
                final_blocks[-1][0],
                b[1],
                final_blocks[-1][2],
                final_blocks[-1][3],
            )
        else:
            final_blocks.append(tuple(b))
    final_blocks = _merge_short_blocks(final_blocks, min_block_sec)
    final_blocks, mixed_policy = _split_mixed_blocks(
        final_blocks,
        features,
        speech_indices,
        times,
        f0_values,
        confidence_cutoff,
        anchor_min_sec,
        mixed_check_sec,
        mixed_min_part_sec,
        micro_window_sec,
        micro_step_sec,
    )
    final_blocks, short_policy = _refine_short_blocks_by_anchors(
        final_blocks,
        features,
        speech_indices,
        times,
        f0_values,
        short_uncertain_sec,
        anchor_min_sec,
        confidence_cutoff,
    )

    change_points = []
    for i in range(1, len(final_blocks)):
        prev = final_blocks[i - 1]
        curr = final_blocks[i]
        if prev[2] != curr[2]:
            conf = curr[3]
            if conf >= confidence_cutoff:
                change_points.append(
                    {
                        "time": _round_float(curr[0], 1),
                        "time_label": _fmt_mmss(curr[0]),
                        "from": _speaker_name(prev[2]),
                        "to": _speaker_name(curr[2]),
                        "confidence": _round_float(conf),
                    }
                )

    ws = 30
    smoothed_e = np.convolve(energies, np.ones(ws) / ws, mode="same")
    threshold = float(np.mean(smoothed_e) + 3 * np.std(smoothed_e))
    spikes = []
    in_spike = False
    spike_start = 0
    for i in range(len(smoothed_e)):
        if threshold > 0 and smoothed_e[i] > threshold and not in_spike:
            spike_start = i
            in_spike = True
        elif threshold <= 0 or (smoothed_e[i] <= threshold and in_spike):
            if in_spike:
                intensity = float(smoothed_e[spike_start:i].max() / threshold) if threshold > 0 else 0.0
                spikes.append(
                    {
                        "start": _round_float(times[spike_start], 1),
                        "end": _round_float(times[i], 1),
                        "start_label": _fmt_mmss(times[spike_start]),
                        "end_label": _fmt_mmss(times[i]),
                        "intensity": _round_float(intensity, 2),
                    }
                )
            in_spike = False

    block_rows = []
    speaker_metrics = {}
    group_map = {}
    group_sec = max(5.0, float(group_sec or 60.0))

    for start, end, cluster, conf in final_blocks:
        start_i = np.searchsorted(times, start, side="left")
        end_i = np.searchsorted(times, end, side="right")
        frame_mask = speech_indices[(speech_indices >= start_i) & (speech_indices < end_i)]
        feature_idx = np.searchsorted(speech_indices, frame_mask)
        feature_idx = feature_idx[feature_idx < len(features)]
        block_features = features[feature_idx] if len(feature_idx) else np.empty((0, features.shape[1]))
        block_f0 = (
            f0_values[feature_idx]
            if f0_values is not None and len(f0_values) == len(features) and len(feature_idx)
            else np.array([])
        )
        voiced_f0 = block_f0[block_f0 > 0]

        if len(block_features):
            centroid_hz = int(block_features[:, -2].mean())
            avg_energy = float(block_features[:, -1].mean())
            band_values = block_features[:, 13:20].mean(axis=0)
            dominant_i = int(np.argmax(band_values))
            band_profile = {
                BAND_LABELS[i]: _round_float(v, 4)
                for i, v in enumerate(band_values)
            }
            dominant_band = BAND_LABELS[dominant_i]
        else:
            centroid_hz = None
            avg_energy = 0.0
            band_profile = {}
            dominant_band = ""
        f0_hz = int(np.median(voiced_f0)) if len(voiced_f0) else None
        voiced_ratio = len(voiced_f0) / float(len(block_f0) or 1) if len(block_f0) else 0.0

        duration = float(end - start)
        row = {
            "start": _round_float(start, 1),
            "end": _round_float(end, 1),
            "start_label": _fmt_mmss(start),
            "end_label": _fmt_mmss(end),
            "duration": _round_float(duration, 1),
            "speaker": _speaker_name(cluster),
            "confidence": _round_float(conf),
            "low_confidence": bool(conf < confidence_cutoff),
            "centroid_hz": centroid_hz,
            "avg_energy": _round_float(avg_energy, 6),
            "f0_hz": f0_hz,
            "voiced_ratio": _round_float(voiced_ratio, 2),
            "dominant_band": dominant_band,
            "band_profile": band_profile,
        }
        block_rows.append(row)

        sp = speaker_metrics.setdefault(
            row["speaker"],
            {
                "speaker": row["speaker"],
                "talk_seconds": 0.0,
                "blocks": 0,
                "confidence_sum": 0.0,
                "duration_weight": 0.0,
                "centroid_hz": profiles.get(row["speaker"], {}).get("centroid_hz"),
                "f0_hz": profiles.get(row["speaker"], {}).get("f0_hz"),
                "n_frames": profiles.get(row["speaker"], {}).get("n_frames", 0),
                "note": profiles.get(row["speaker"], {}).get("note", ""),
                "band_profile_acc": {label: 0.0 for label in BAND_LABELS},
                "band_profile_weight": 0,
            },
        )
        sp["talk_seconds"] += duration
        sp["blocks"] += 1
        sp["confidence_sum"] += conf * duration
        sp["duration_weight"] += duration
        if band_profile:
            for label, value in band_profile.items():
                sp["band_profile_acc"][label] += value
            sp["band_profile_weight"] += 1

        bucket = int(start // group_sec)
        g_start = bucket * group_sec
        g_end = g_start + group_sec
        group = group_map.setdefault(
            bucket,
            {
                "label": f"{_fmt_mmss(g_start)}-{_fmt_mmss(g_end)}",
                "start": _round_float(g_start, 1),
                "end": _round_float(g_end, 1),
                "blocks": [],
            },
        )
        group["blocks"].append(row)

    speech_labeled_seconds = sum(b["duration"] for b in block_rows)
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    detected_speech_seconds = float(speech_mask.sum() * frame_step)
    duration_seconds = float(times[-1])
    low_conf_count = sum(1 for b in block_rows if b["low_confidence"])
    avg_conf = (
        sum(b["confidence"] * b["duration"] for b in block_rows) / speech_labeled_seconds
        if speech_labeled_seconds > 0
        else 0.0
    )

    speakers = []
    for sp in speaker_metrics.values():
        weight = sp.pop("duration_weight") or 1.0
        band_weight = sp.pop("band_profile_weight") or 1
        band_acc = sp.pop("band_profile_acc")
        sp["talk_seconds"] = _round_float(sp["talk_seconds"], 1)
        sp["talk_percent"] = _round_float(
            100 * sp["talk_seconds"] / speech_labeled_seconds if speech_labeled_seconds else 0.0,
            1,
        )
        sp["avg_confidence"] = _round_float(sp.pop("confidence_sum") / weight)
        sp["avg_block_seconds"] = _round_float(
            sp["talk_seconds"] / sp["blocks"] if sp["blocks"] else 0.0,
            1,
        )
        sp["band_profile"] = {
            label: _round_float(value / band_weight, 4)
            for label, value in band_acc.items()
        }
        speakers.append(sp)
    speakers.sort(key=lambda x: x["speaker"])

    real_speaker_count = sum(1 for sp in speakers if sp["speaker"] != "UNK")
    unknown_seconds = sum(sp["talk_seconds"] for sp in speakers if sp["speaker"] == "UNK")

    return {
        "ok": True,
        "method": "GMM(MFCC+spectral bands)",
        "duration_seconds": _round_float(duration_seconds, 1),
        "duration_label": _fmt_mmss(duration_seconds),
        "metrics": {
            "expected_speakers": n_speakers,
            "detected_clusters": real_speaker_count,
            "unknown_seconds": _round_float(unknown_seconds, 1),
            "speaker_blocks": len(block_rows),
            "change_points": len(change_points),
            "energy_spikes": len(spikes),
            "avg_confidence": _round_float(avg_conf),
            "low_confidence_blocks": low_conf_count,
            "detected_speech_seconds": _round_float(detected_speech_seconds, 1),
            "labeled_speech_seconds": _round_float(speech_labeled_seconds, 1),
            "speech_ratio_percent": _round_float(
                100 * detected_speech_seconds / duration_seconds if duration_seconds else 0.0,
                1,
            ),
            "short_unknown_blocks": short_policy.get("unknown_blocks", 0),
            "short_reassigned_blocks": short_policy.get("reassigned_blocks", 0),
            "mixed_blocks": mixed_policy.get("mixed_blocks", 0),
            "mixed_subblocks": mixed_policy.get("created_subblocks", 0),
        },
        "speakers": speakers,
        "blocks": block_rows,
        "time_groups": [group_map[k] for k in sorted(group_map)],
        "change_points_list": change_points,
        "energy_spikes_list": spikes,
        "short_policy": short_policy,
        "mixed_policy": mixed_policy,
    }


# ═══════════════════════════════════════════════════════════════════
#  Публичный API
# ═══════════════════════════════════════════════════════════════════

def analyze_audio(
    audio_path: str,
    n_speakers: int = 2,
    ivr_cutoff_sec: float = 0.0,
    confidence_cutoff: float = 0.55,
    speech_ranges: Optional[list[tuple[float, float]]] = None,
) -> Optional[str]:
    """
    Анализирует аудиофайл и возвращает компактные мягкие метки
    для вставки в LLM-промпт.

    Параметры:
        audio_path        — путь к аудиофайлу (.mp3/.ogg/.opus/.wav)
        n_speakers        — сколько основных спикеров ожидаем (по умолчанию 2)
        ivr_cutoff_sec    — первые N секунд помечаются как отдельный кластер (IVR)
                            0 = не выделять
        confidence_cutoff — отсечка точек смены по уверенности (0.55 по умолчанию)

    Возвращает:
        str — компактные метки (~1-4K токенов) или None при ошибке.
    """
    wav_path = None
    try:
        # Конвертируем в WAV
        wav_path = _convert_to_wav(audio_path)
        sr, raw = wavfile.read(wav_path)
        data = raw.astype(np.float32) / 32768.0

        log.info(
            "spectral: %s — %.0fs, sr=%d", audio_path, len(data) / sr, sr
        )

        # Извлекаем признаки
        features, energies, times, speech_mask, speech_indices = _extract_features(
            data, sr, speech_ranges=speech_ranges
        )
        del data  # освобождаем память

        if len(features) < 50:
            log.warning("spectral: слишком мало речевых фреймов (%d)", len(features))
            return None

        speech_times = times[speech_indices]

        # Кластеризуем
        labels, probs, profiles = _cluster_speakers(
            features, speech_times, n_speakers, ivr_cutoff_sec
        )

        if len(labels) == 0:
            return None

        # Формируем компактный вывод
        return _build_compact_output(
            labels,
            probs,
            profiles,
            speech_indices,
            times,
            energies,
            speech_mask,
            n_speakers,
            confidence_cutoff,
        )

    except Exception:
        log.exception("spectral analysis failed for %s", audio_path)
        return None
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def analyze_audio_report(
    audio_path: str,
    n_speakers: int = 2,
    ivr_cutoff_sec: float = 0.0,
    confidence_cutoff: float = 0.55,
    group_sec: float = 60.0,
    method: str = "windowed",
    window_sec: float = 2.5,
    step_sec: float = 0.5,
    min_block_sec: float = 2.0,
    short_uncertain_sec: float = 2.5,
    anchor_min_sec: float = 8.0,
    mixed_check_sec: float = 6.0,
    mixed_min_part_sec: float = 1.2,
    micro_window_sec: float = 1.2,
    micro_step_sec: float = 0.25,
    transient_enabled: bool = False,
    transient_search_sec: float = 60.0,
    transient_late_start_sec: float = 75.0,
    transient_min_sec: float = 4.0,
    transient_distance_threshold: float = 9.0,
    transient_local_speakers: int = 3,
    speech_ranges: Optional[list[tuple[float, float]]] = None,
) -> dict:
    wav_path = None
    try:
        wav_path = _convert_to_wav(audio_path)
        sr, raw = wavfile.read(wav_path)
        if getattr(raw, "ndim", 1) > 1:
            raw = raw.mean(axis=1)
        data = raw.astype(np.float32) / 32768.0

        features, energies, times, speech_mask, speech_indices = _extract_features(
            data, sr, speech_ranges=speech_ranges
        )
        f0_values, f0_clarity = _estimate_f0_track(data, sr, speech_indices)
        duration_seconds = len(data) / float(sr or 1)
        del data

        if len(features) < 50:
            return {
                "ok": False,
                "error": f"Too few speech frames for analysis: {len(features)}.",
                "duration_seconds": _round_float(duration_seconds, 1),
                "duration_label": _fmt_mmss(duration_seconds),
                "speech_guided": bool(speech_ranges),
            }

        speech_times = times[speech_indices]
        if method == "frame":
            labels, probs, profiles = _cluster_speakers(
                features, speech_times, n_speakers, ivr_cutoff_sec
            )
            for c in range(n_speakers):
                mask = labels == c
                voiced = f0_values[mask]
                voiced = voiced[voiced > 0]
                if f"C{c}" in profiles:
                    profiles[f"C{c}"]["f0_hz"] = int(np.median(voiced)) if len(voiced) else None
            windows = []
            method_label = "Frame GMM(MFCC+spectral bands)"
        else:
            labels, probs, profiles, windows = _cluster_speakers_windowed(
                features,
                speech_times,
                n_speakers,
                ivr_cutoff_sec,
                window_sec,
                step_sec,
                f0_values,
            )
            method_label = f"Windowed GMM({window_sec:.1f}s/{step_sec:.1f}s)"

        if len(labels) == 0:
            return {
                "ok": False,
                "error": "Speaker clustering returned no labels.",
                "duration_seconds": _round_float(duration_seconds, 1),
                "duration_label": _fmt_mmss(duration_seconds),
                "speech_guided": bool(speech_ranges),
            }

        transient_policy = {"enabled": False, "added": 0}
        if transient_enabled and method != "frame":
            labels, probs, profiles, transient_policy = _detect_transient_speakers(
                features,
                speech_times,
                f0_values,
                labels,
                probs,
                profiles,
                n_speakers,
                transient_search_sec,
                transient_late_start_sec,
                transient_min_sec,
                transient_distance_threshold,
                transient_local_speakers,
            )

        report = _build_structured_report(
            labels,
            probs,
            profiles,
            speech_indices,
            times,
            energies,
            speech_mask,
            features,
            n_speakers,
            confidence_cutoff,
            group_sec,
            min_block_sec,
            f0_values,
            short_uncertain_sec,
            anchor_min_sec,
            mixed_check_sec,
            mixed_min_part_sec,
            micro_window_sec,
            micro_step_sec,
        )
        report["compact"] = _build_compact_output(
            labels,
            probs,
            profiles,
            speech_indices,
            times,
            energies,
            speech_mask,
            n_speakers,
            confidence_cutoff,
            min_block_sec,
        )
        report["sample_rate"] = sr
        report["source_name"] = os.path.basename(audio_path)
        report["method"] = method_label
        if speech_ranges:
            report["method"] += " + STT-guided"
        report["speech_guided"] = bool(speech_ranges)
        report["speech_ranges_count"] = len(speech_ranges or [])
        report["windows"] = windows
        report["pitch_frames"] = int((f0_values > 0).sum())
        report["transient_policy"] = transient_policy
        if report.get("ok"):
            report["metrics"]["transient_added"] = transient_policy.get("added", 0)
            report["metrics"]["estimated_total_speakers"] = report["metrics"].get("detected_clusters", 0)
        return report

    except Exception as exc:
        log.exception("spectral report failed for %s", audio_path)
        return {"ok": False, "error": str(exc)}
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def export_audio_clips(
    audio_path: str,
    blocks: list[dict],
    output_dir: str,
    padding_sec: float = 0.15,
    max_clips: int = 250,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    ffmpeg = _ffmpeg_bin()

    for idx, block in enumerate(blocks[:max_clips]):
        try:
            start = max(0.0, float(block.get("start", 0.0)) - padding_sec)
            end = max(start + 0.1, float(block.get("end", start + 0.1)) + padding_sec)
            duration = end - start
            speaker = str(block.get("speaker") or "C")
            safe_speaker = "".join(ch for ch in speaker if ch.isalnum()) or "C"
            filename = f"{idx + 1:04d}_{safe_speaker}_{int(start * 1000):09d}_{int(end * 1000):09d}.wav"
            out_path = os.path.join(output_dir, filename)
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                audio_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                log.warning("clip export failed for block %s: %s", idx, result.stderr[:300])
                continue
            exported.append(
                {
                    "index": idx,
                    "filename": filename,
                    "start": _round_float(start, 1),
                    "end": _round_float(end, 1),
                    "duration": _round_float(duration, 1),
                }
            )
        except Exception:
            log.exception("clip export failed for block %s", idx)

    return exported
