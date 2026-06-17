from __future__ import annotations
import os, uuid, time, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.datastructures import FileStorage

from config import DEFAULTS, ALLOWED_EXTS, TMP_DIR, TEMPLATES_PATH
from services.storage import upload_local_file, delete_remote
from services.asr import start_long, poll_once
from services.llm import analyze_with_llm, build_system_with_checklist
from services.parsing import build_segments, parse_analysis_status, parse_checklist, annotate_checklist_with_timestamps
from services.spectral import analyze_audio_report
import json
import logging

log = logging.getLogger(__name__)

qa_calls_bp = Blueprint("qa_calls", __name__, url_prefix="/")

BATCHES: Dict[str, Dict[str, Any]] = {}
BATCH_LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(DEFAULTS.get("ASYNC_MAX_WORKERS", 4))),
    thread_name_prefix="qa-batch",
)
ENCODING_BY_EXT = {".mp3": "MP3", ".ogg": "OGG_OPUS", ".opus": "OGG_OPUS"}


def _load_templates():
    if not os.path.exists(TEMPLATES_PATH):
        from blueprints.admin import _ensure_templates_file
        _ensure_templates_file()
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _allowed_mime(mime: str) -> bool:
    if not mime or mime == "application/octet-stream":
        return True
    return any(mime.startswith(x) for x in ("audio/mpeg", "audio/ogg", "audio/opus"))


def _cleanup_old_batches() -> None:
    ttl = int(DEFAULTS.get("JOB_TTL_SEC", 7200) or 7200)
    if ttl <= 0:
        return
    cutoff = time.time() - ttl
    with BATCH_LOCK:
        old_ids = [
            batch_id
            for batch_id, batch in BATCHES.items()
            if batch.get("created", 0) < cutoff and batch.get("done")
        ]
        for batch_id in old_ids:
            BATCHES.pop(batch_id, None)


def _update_item(batch_id: str, item_id: str, **updates: Any) -> None:
    with BATCH_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return
        for item in batch.get("items", []):
            if item.get("id") == item_id:
                item.update(updates)
                return


def _get_item_progress(batch_id: str, item_id: str, default: int = 0) -> int:
    with BATCH_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return default
        for item in batch.get("items", []):
            if item.get("id") == item_id:
                return int(item.get("progress", default) or default)
    return default


def _spectral_kwargs() -> Dict[str, Any]:
    method = str(DEFAULTS.get("SPECTRAL_METHOD", "windowed") or "windowed").lower()
    if method not in {"windowed", "frame"}:
        method = "windowed"
    return {
        "n_speakers": DEFAULTS.get("SPECTRAL_N_SPEAKERS", 2),
        "ivr_cutoff_sec": DEFAULTS.get("SPECTRAL_IVR_CUTOFF_SEC", 0),
        "confidence_cutoff": DEFAULTS.get("SPECTRAL_CONFIDENCE_CUTOFF", 0.55),
        "group_sec": DEFAULTS.get("SPECTRAL_GROUP_SEC", 60),
        "method": method,
        "window_sec": DEFAULTS.get("SPECTRAL_WINDOW_SEC", 3.5),
        "step_sec": DEFAULTS.get("SPECTRAL_STEP_SEC", 0.5),
        "min_block_sec": DEFAULTS.get("SPECTRAL_MIN_BLOCK_SEC", 0),
        "short_uncertain_sec": DEFAULTS.get("SPECTRAL_SHORT_UNCERTAIN_SEC", 2.5),
        "anchor_min_sec": DEFAULTS.get("SPECTRAL_ANCHOR_MIN_SEC", 8),
        "mixed_check_sec": DEFAULTS.get("SPECTRAL_MIXED_CHECK_SEC", 0),
        "mixed_min_part_sec": DEFAULTS.get("SPECTRAL_MIXED_MIN_PART_SEC", 1.2),
        "micro_window_sec": DEFAULTS.get("SPECTRAL_MICRO_WINDOW_SEC", 1.2),
        "micro_step_sec": DEFAULTS.get("SPECTRAL_MICRO_STEP_SEC", 0.25),
        "transient_enabled": DEFAULTS.get("SPECTRAL_TRANSIENT_ENABLED", True),
        "transient_search_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_SEARCH_SEC", 60),
        "transient_late_start_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_LATE_START_SEC", 75),
        "transient_min_sec": DEFAULTS.get("SPECTRAL_TRANSIENT_MIN_SEC", 4),
        "transient_distance_threshold": DEFAULTS.get("SPECTRAL_TRANSIENT_DISTANCE_THRESHOLD", 9),
        "transient_local_speakers": DEFAULTS.get("SPECTRAL_TRANSIENT_LOCAL_SPEAKERS", 3),
    }


def _build_soft_labels(local_path: str, filename: str) -> str | None:
    if not DEFAULTS.get("SPECTRAL_ENABLED"):
        return None
    report = analyze_audio_report(local_path, **_spectral_kwargs())
    if not report.get("ok"):
        log.warning(
            "spectral failed for %s: %s",
            filename,
            report.get("error", "unknown error"),
        )
        return None
    compact = report.get("compact")
    log.info("spectral OK for %s (%d chars)", filename, len(compact or ""))
    return compact or None


def _build_transcript(segs: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"=== Фрагмент {s['mmss']}–{s['e_mmss']} ===\n{s['text']}"
        for s in segs
    ) or "(пусто)"


def _wait_for_asr(batch_id: str, item_id: str, op_id: str) -> Dict[str, Any]:
    interval = max(1.0, float(DEFAULTS.get("ASR_POLL_INTERVAL_SEC", 3) or 3))
    timeout = max(interval, float(DEFAULTS.get("ASR_POLL_TIMEOUT_SEC", 7200) or 7200))
    deadline = time.time() + timeout
    poll_errors = 0

    while True:
        if time.time() > deadline:
            raise TimeoutError("ASR не завершился за отведённое время")
        try:
            js = poll_once(op_id)
            poll_errors = 0
        except Exception as exc:
            poll_errors += 1
            if poll_errors >= 3:
                raise
            _update_item(
                batch_id,
                item_id,
                stage=f"Повторяем опрос распознавания ({poll_errors}/3)…",
                progress=max(_get_item_progress(batch_id, item_id, 35), 35),
            )
            log.warning("ASR poll retry %s for %s: %s", poll_errors, op_id, exc)
            time.sleep(interval)
            continue

        if js.get("done") or js.get("error"):
            return js

        progress = min(78, max(_get_item_progress(batch_id, item_id, 35) + 2, 40))
        _update_item(
            batch_id,
            item_id,
            stage="Распознаём аудио…",
            progress=progress,
        )
        time.sleep(interval)


def _process_item(
    batch_id: str,
    item_id: str,
    filename: str,
    local_path: str,
    ext: str,
    system_prompt: str,
    checklist_items: list[str],
) -> None:
    key = None
    style = None
    op_id = None
    remote_deleted = False
    asr_terminal = False

    try:
        _update_item(batch_id, item_id, status="running", stage="Загружаем аудио…", progress=5)
        uri, key, style = upload_local_file(local_path, ext)
        _update_item(
            batch_id,
            item_id,
            uri=uri,
            s3_key=key,
            s3_style=style,
            stage="Запускаем распознавание…",
            progress=12,
        )

        op_id = start_long(uri, enc=ENCODING_BY_EXT[ext])
        _update_item(batch_id, item_id, op_id=op_id, stage="Распознаём аудио…", progress=20)

        soft_labels = None
        if DEFAULTS.get("SPECTRAL_ENABLED"):
            _update_item(batch_id, item_id, stage="Считаем мягкие метки спикеров…", progress=25)
            soft_labels = _build_soft_labels(local_path, filename)
            _update_item(
                batch_id,
                item_id,
                soft_labels=soft_labels,
                stage="Распознаём аудио…",
                progress=max(_get_item_progress(batch_id, item_id, 25), 35),
            )

        js = _wait_for_asr(batch_id, item_id, op_id)
        asr_terminal = True
        if js.get("error"):
            raise RuntimeError(f"Ошибка STT: {js.get('error')}")

        segs = build_segments(js)
        transcript = _build_transcript(segs)
        _update_item(
            batch_id,
            item_id,
            stage="Анализируем диалог…",
            progress=85,
            segments=segs,
            transcript=transcript,
        )

        llm_answer = analyze_with_llm(
            transcript,
            system_prompt,
            soft_labels=soft_labels,
        )
        analysis_status = parse_analysis_status(llm_answer)
        cl = parse_checklist(llm_answer, checklist_items)
        cl = annotate_checklist_with_timestamps(cl, segs)
        _update_item(
            batch_id,
            item_id,
            status="done",
            stage="Готово",
            progress=100,
            llm=llm_answer,
            analysis_status=analysis_status,
            checklist=cl,
        )

        if key and style and DEFAULTS["DELETE_REMOTE_AFTER"]:
            delete_remote(key, style)
            remote_deleted = True

    except Exception as exc:
        log.exception("batch item failed for %s", filename)
        _update_item(
            batch_id,
            item_id,
            status="error",
            stage=f"Ошибка: {exc}",
            progress=100,
        )
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
        if key and style and not remote_deleted and DEFAULTS["DELETE_REMOTE_AFTER"] and (op_id is None or asr_terminal):
            delete_remote(key, style)


def create_batch_from_uploads(
    files: list[FileStorage],
    prompt_id: str,
    checklist_id: str,
) -> tuple[str | None, str | None]:
    files = [f for f in files if f and f.filename]

    if not prompt_id or not checklist_id:
        return None, "Выберите промт и чек-лист."

    if not files:
        return None, "Файлы не выбраны."

    if len(files) > 10:
        return None, "Максимум 10 записей за раз."

    validated: list[tuple[FileStorage, str, str]] = []
    for f in files:
        name = f.filename or "audio"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            return None, f"Файл {name}: поддерживаются только .mp3/.ogg/.opus"
        mime = getattr(f, "mimetype", "") or ""
        if not _allowed_mime(mime):
            return None, f"Файл {name}: неподдерживаемый MIME-тип {mime}"
        validated.append((f, name, ext))

    data = _load_templates()
    prompt = next((p for p in data.get("prompts", []) if p.get("id") == prompt_id), None)
    checklist = next((c for c in data.get("checklists", []) if c.get("id") == checklist_id), None)
    if not prompt or not checklist:
        return None, "Неверный выбор промта/чек-листа."

    batch_id = uuid.uuid4().hex
    checklist_items = checklist.get("items") or []
    system_prompt = build_system_with_checklist(prompt.get("system") or "", checklist_items)

    with BATCH_LOCK:
        BATCHES[batch_id] = {
            "created": time.time(),
            "prompt_id": prompt_id,
            "checklist_id": checklist_id,
            "prompt_title": prompt.get("title"),
            "checklist_title": checklist.get("title"),
            "items": [],
            "done": False,
            "error": None,
        }

    for f, name, ext in validated:
        local_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")
        item_id = uuid.uuid4().hex
        item = {
            "id": item_id,
            "filename": name,
            "uri": None,
            "s3_key": None,
            "s3_style": None,
            "op_id": None,
            "stage": "В очереди…",
            "progress": 0,
            "status": "queued",
            "segments": [],
            "transcript": None,
            "soft_labels": None,
            "llm": None,
            "analysis_status": None,
            "checklist": None,
        }
        with BATCH_LOCK:
            BATCHES[batch_id]["items"].append(item)
        try:
            f.save(local_path)
        except Exception as exc:
            log.exception("failed to save upload %s", name)
            _update_item(
                batch_id,
                item_id,
                status="error",
                stage=f"Ошибка сохранения файла: {exc}",
                progress=100,
            )
            continue
        EXECUTOR.submit(
            _process_item,
            batch_id,
            item_id,
            name,
            local_path,
            ext,
            system_prompt,
            checklist_items,
        )

    return batch_id, None


@qa_calls_bp.route("", methods=["GET"])
def index():
    data = _load_templates()
    return render_template("upload_batch.html", prompts=data.get("prompts", []), checklists=data.get("checklists", []))


@qa_calls_bp.route("/start_batch", methods=["POST"])
def start_batch():
    files: list[FileStorage] = [f for f in request.files.getlist("audio_files") if f and f.filename]
    prompt_id = (request.form.get("prompt_id") or "").strip()
    checklist_id = (request.form.get("checklist_id") or "").strip()

    batch_id, error = create_batch_from_uploads(files, prompt_id, checklist_id)
    if error or not batch_id:
        flash(error or "Не удалось запустить пакет.")
        return redirect(url_for("qa_calls.index"))

    session["last_batch_id"] = batch_id
    return render_template("progress_batch.html", batch_id=batch_id)


@qa_calls_bp.route("/status_batch/<batch_id>")
def status_batch(batch_id: str):
    _cleanup_old_batches()
    with BATCH_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return jsonify({"done": True, "redirect": url_for("qa_calls.index"), "stage": "Не найдено"})

        items = list(batch.get("items", []))
        total = len(items) or 1
        finished = sum(1 for it in items if it.get("status") in ("done", "error"))
        avg_pct = int(sum(int(i.get("progress", 0) or 0) for i in items) / total)
        all_done = finished == total
        if all_done:
            batch["done"] = True

        payload_items = [
            {
                "filename": it.get("filename"),
                "stage": it.get("stage"),
                "progress": int(it.get("progress", 0) or 0),
                "status": it.get("status"),
            }
            for it in items
        ]

    if all_done:
        return jsonify({
            "done": True,
            "redirect": url_for("qa_calls.results_batch", batch_id=batch_id),
            "items": payload_items,
            "progress": 100,
            "stage": f"Готово {finished}/{total}",
        })

    return jsonify({
        "done": False,
        "progress": avg_pct,
        "stage": f"Готово {finished}/{total}",
        "items": payload_items,
    })


@qa_calls_bp.route("/results/<batch_id>")
def results_batch(batch_id: str):
    with BATCH_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            flash("Пакет не найден.")
            return redirect(url_for("qa_calls.index"))
    return render_template("results_batch.html", batch=batch)
