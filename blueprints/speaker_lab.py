from __future__ import annotations

import os
import shutil
import time
import uuid

from flask import Blueprint, abort, flash, render_template, request, send_from_directory, url_for
from werkzeug.datastructures import FileStorage

from config import ALLOWED_EXTS, DEFAULTS, TMP_DIR
from services.spectral import analyze_audio_report, export_audio_clips


speaker_lab_bp = Blueprint("speaker_lab", __name__, url_prefix="/speaker-test")

TEST_EXTS = ALLOWED_EXTS | {".wav"}
CLIPS_DIR = os.path.join(TMP_DIR, "speaker_test_clips")


def _int_form(name: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(request.form.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _float_form(name: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _cleanup_old_clip_runs(max_age_sec: int = 7200) -> None:
    os.makedirs(CLIPS_DIR, exist_ok=True)
    root = os.path.abspath(CLIPS_DIR)
    now = time.time()
    for name in os.listdir(root):
        path = os.path.abspath(os.path.join(root, name))
        if not path.startswith(root):
            continue
        if not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) > max_age_sec:
                shutil.rmtree(path)
        except OSError:
            pass


@speaker_lab_bp.route("/clips/<run_id>/<path:filename>")
def clip(run_id: str, filename: str):
    if not run_id.replace("-", "").isalnum():
        abort(404)
    run_dir = os.path.abspath(os.path.join(CLIPS_DIR, run_id))
    root = os.path.abspath(CLIPS_DIR)
    if not run_dir.startswith(root) or not os.path.isdir(run_dir):
        abort(404)
    return send_from_directory(run_dir, filename, mimetype="audio/wav", as_attachment=False)


@speaker_lab_bp.route("", methods=["GET", "POST"])
def speaker_test():
    _cleanup_old_clip_runs()
    params = {
        "n_speakers": DEFAULTS.get("SPECTRAL_N_SPEAKERS", 2),
        "ivr_cutoff_sec": DEFAULTS.get("SPECTRAL_IVR_CUTOFF_SEC", 0),
        "confidence_cutoff": DEFAULTS.get("SPECTRAL_CONFIDENCE_CUTOFF", 0.55),
        "group_sec": DEFAULTS.get("SPECTRAL_GROUP_SEC", 60),
        "method": DEFAULTS.get("SPECTRAL_METHOD", "windowed"),
        "window_sec": DEFAULTS.get("SPECTRAL_WINDOW_SEC", 3.5),
        "step_sec": DEFAULTS.get("SPECTRAL_STEP_SEC", 0.5),
        "min_block_sec": DEFAULTS.get("SPECTRAL_MIN_BLOCK_SEC", 0.0),
        "short_uncertain_sec": DEFAULTS.get("SPECTRAL_SHORT_UNCERTAIN_SEC", 2.5),
        "anchor_min_sec": DEFAULTS.get("SPECTRAL_ANCHOR_MIN_SEC", 8.0),
        "mixed_check_sec": DEFAULTS.get("SPECTRAL_MIXED_CHECK_SEC", 0.0),
        "mixed_min_part_sec": DEFAULTS.get("SPECTRAL_MIXED_MIN_PART_SEC", 1.2),
        "micro_window_sec": DEFAULTS.get("SPECTRAL_MICRO_WINDOW_SEC", 1.2),
        "micro_step_sec": DEFAULTS.get("SPECTRAL_MICRO_STEP_SEC", 0.25),
        "transient_enabled": DEFAULTS.get("SPECTRAL_TRANSIENT_ENABLED", True),
        "transient_search_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_SEARCH_SEC", 60.0),
        "transient_late_start_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_LATE_START_SEC", 75.0),
        "transient_min_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_MIN_SEC", 4.0),
        "transient_distance_threshold": DEFAULTS.get("SPECTRAL_TRANSIENT_DISTANCE_THRESHOLD", 9.0),
        "transient_local_speakers": DEFAULTS.get("SPECTRAL_TRANSIENT_LOCAL_SPEAKERS", 3),
    }
    if params["method"] not in {"windowed", "frame"}:
        params["method"] = "windowed"
    report = None
    filename = ""

    if request.method == "POST":
        params = {
            "n_speakers": _int_form("n_speakers", DEFAULTS.get("SPECTRAL_N_SPEAKERS", 2), 1, 8),
            "ivr_cutoff_sec": _float_form("ivr_cutoff_sec", DEFAULTS.get("SPECTRAL_IVR_CUTOFF_SEC", 0.0), 0.0, 600.0),
            "confidence_cutoff": _float_form("confidence_cutoff", DEFAULTS.get("SPECTRAL_CONFIDENCE_CUTOFF", 0.55), 0.0, 1.0),
            "group_sec": _float_form("group_sec", DEFAULTS.get("SPECTRAL_GROUP_SEC", 60.0), 5.0, 600.0),
            "method": request.form.get("method", DEFAULTS.get("SPECTRAL_METHOD", "windowed")),
            "window_sec": _float_form("window_sec", DEFAULTS.get("SPECTRAL_WINDOW_SEC", 3.5), 0.8, 10.0),
            "step_sec": _float_form("step_sec", DEFAULTS.get("SPECTRAL_STEP_SEC", 0.5), 0.2, 5.0),
            "min_block_sec": _float_form("min_block_sec", DEFAULTS.get("SPECTRAL_MIN_BLOCK_SEC", 0.0), 0.0, 30.0),
            "short_uncertain_sec": _float_form("short_uncertain_sec", DEFAULTS.get("SPECTRAL_SHORT_UNCERTAIN_SEC", 2.5), 0.0, 20.0),
            "anchor_min_sec": _float_form("anchor_min_sec", DEFAULTS.get("SPECTRAL_ANCHOR_MIN_SEC", 8.0), 2.0, 60.0),
            "mixed_check_sec": _float_form("mixed_check_sec", DEFAULTS.get("SPECTRAL_MIXED_CHECK_SEC", 0.0), 0.0, 60.0),
            "mixed_min_part_sec": _float_form("mixed_min_part_sec", DEFAULTS.get("SPECTRAL_MIXED_MIN_PART_SEC", 1.2), 0.4, 20.0),
            "micro_window_sec": _float_form("micro_window_sec", DEFAULTS.get("SPECTRAL_MICRO_WINDOW_SEC", 1.2), 0.5, 6.0),
            "micro_step_sec": _float_form("micro_step_sec", DEFAULTS.get("SPECTRAL_MICRO_STEP_SEC", 0.25), 0.1, 3.0),
            "transient_enabled": request.form.get("transient_enabled") == "1",
            "transient_search_sec": _float_form("transient_search_sec", DEFAULTS.get("SPECTRAL_TRANSIENT_SEARCH_SEC", 60.0), 0.0, 300.0),
            "transient_late_start_sec": _float_form("transient_late_start_sec", DEFAULTS.get("SPECTRAL_TRANSIENT_LATE_START_SEC", 75.0), 0.0, 600.0),
            "transient_min_sec": _float_form("transient_min_sec", DEFAULTS.get("SPECTRAL_TRANSIENT_MIN_SEC", 4.0), 1.0, 60.0),
            "transient_distance_threshold": _float_form("transient_distance_threshold", DEFAULTS.get("SPECTRAL_TRANSIENT_DISTANCE_THRESHOLD", 9.0), 1.0, 30.0),
            "transient_local_speakers": _int_form("transient_local_speakers", DEFAULTS.get("SPECTRAL_TRANSIENT_LOCAL_SPEAKERS", 3), 2, 6),
        }
        if params["method"] not in {"windowed", "frame"}:
            params["method"] = "windowed"
        audio: FileStorage | None = request.files.get("audio_file")
        if not audio or not audio.filename:
            flash("Выберите аудиофайл для анализа.")
        else:
            filename = audio.filename
            ext = os.path.splitext(filename)[1].lower()
            if ext not in TEST_EXTS:
                flash("Поддерживаются .mp3, .ogg, .opus и .wav.")
            else:
                local_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")
                try:
                    audio.save(local_path)
                    report = analyze_audio_report(
                        local_path,
                        n_speakers=params["n_speakers"],
                        ivr_cutoff_sec=params["ivr_cutoff_sec"],
                        confidence_cutoff=params["confidence_cutoff"],
                        group_sec=params["group_sec"],
                        method=params["method"],
                        window_sec=params["window_sec"],
                        step_sec=params["step_sec"],
                        min_block_sec=params["min_block_sec"],
                        short_uncertain_sec=params["short_uncertain_sec"],
                        anchor_min_sec=params["anchor_min_sec"],
                        mixed_check_sec=params["mixed_check_sec"],
                        mixed_min_part_sec=params["mixed_min_part_sec"],
                        micro_window_sec=params["micro_window_sec"],
                        micro_step_sec=params["micro_step_sec"],
                        transient_enabled=params["transient_enabled"],
                        transient_search_sec=params["transient_search_sec"],
                        transient_late_start_sec=params["transient_late_start_sec"],
                        transient_min_sec=params["transient_min_sec"],
                        transient_distance_threshold=params["transient_distance_threshold"],
                        transient_local_speakers=params["transient_local_speakers"],
                    )
                    report["display_name"] = filename
                    if report.get("ok"):
                        run_id = uuid.uuid4().hex
                        clip_dir = os.path.join(CLIPS_DIR, run_id)
                        clips = export_audio_clips(local_path, report.get("blocks", []), clip_dir)
                        for clip in clips:
                            block = report["blocks"][clip["index"]]
                            block["clip_url"] = url_for(
                                "speaker_lab.clip",
                                run_id=run_id,
                                filename=clip["filename"],
                            )
                            block["clip_duration"] = clip["duration"]
                        report["clip_count"] = len(clips)
                        report["clip_run_id"] = run_id
                    if not report.get("ok"):
                        flash(f"Спектральный анализ не дал результата: {report.get('error', 'unknown error')}")
                finally:
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass

    return render_template(
        "speaker_test.html",
        params=params,
        report=report,
        filename=filename,
        allowed_exts=", ".join(sorted(TEST_EXTS)),
    )
