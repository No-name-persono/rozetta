from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, jsonify, request, url_for

from blueprints.qa_calls import (
    BATCHES,
    BATCH_LOCK,
    _cleanup_old_batches,
    _load_templates,
    create_batch_from_uploads,
)
from services.api_keys import verify_api_key

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def _extract_api_key() -> str | None:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("X-API-Key") or "").strip() or None


def require_api_key(fn: Callable[..., Any]):
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        key_info = verify_api_key(_extract_api_key())
        if not key_info:
            return _error("Invalid or missing API key.", 401)
        request.api_key = key_info
        return fn(*args, **kwargs)

    return wrapper


def _item_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": item.get("filename"),
        "status": item.get("status"),
        "stage": item.get("stage"),
        "progress": int(item.get("progress", 0) or 0),
        "analysis_status": item.get("analysis_status"),
    }


def _item_result(item: dict[str, Any]) -> dict[str, Any]:
    row = _item_summary(item)
    row.update(
        {
            "segments": item.get("segments") or [],
            "transcript": item.get("transcript"),
            "llm": item.get("llm"),
            "checklist": item.get("checklist"),
            "soft_labels": item.get("soft_labels"),
        }
    )
    return row


def _batch_payload(batch_id: str, api_key_id: int, include_results: bool = False) -> tuple[dict[str, Any] | None, int]:
    _cleanup_old_batches()
    with BATCH_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return None, 404
        if batch.get("api_key_id") != api_key_id:
            return None, 404

        items = list(batch.get("items", []))
        total = len(items) or 1
        finished = sum(1 for item in items if item.get("status") in {"done", "error"})
        done = finished == total
        if done:
            batch["done"] = True
        progress = int(sum(int(item.get("progress", 0) or 0) for item in items) / total)
        mapper = _item_result if include_results else _item_summary

        payload = {
            "ok": True,
            "batch_id": batch_id,
            "done": done,
            "status": "done" if done else "running",
            "progress": 100 if done else progress,
            "finished": finished,
            "total": total,
            "prompt": {
                "id": batch.get("prompt_id"),
                "title": batch.get("prompt_title"),
            },
            "checklist": {
                "id": batch.get("checklist_id"),
                "title": batch.get("checklist_title"),
            },
            "items": [mapper(item) for item in items],
        }
    return payload, 200


@api_bp.get("/health")
def health():
    return jsonify({"ok": True, "service": "rozeeta"})


@api_bp.get("/templates")
@require_api_key
def templates():
    data = _load_templates()
    return jsonify(
        {
            "ok": True,
            "prompts": [
                {
                    "id": prompt.get("id"),
                    "title": prompt.get("title"),
                    "locked": bool(prompt.get("locked")),
                }
                for prompt in data.get("prompts", [])
            ],
            "checklists": [
                {
                    "id": checklist.get("id"),
                    "title": checklist.get("title"),
                    "locked": bool(checklist.get("locked")),
                    "items": checklist.get("items") or [],
                }
                for checklist in data.get("checklists", [])
            ],
        }
    )


@api_bp.post("/batches")
@require_api_key
def create_batch():
    prompt_id = (request.form.get("prompt_id") or "").strip()
    checklist_id = (request.form.get("checklist_id") or "").strip()
    files = request.files.getlist("audio_files")

    batch_id, error = create_batch_from_uploads(files, prompt_id, checklist_id)
    if error or not batch_id:
        return _error(error or "Could not create batch.", 400)
    with BATCH_LOCK:
        if batch_id in BATCHES:
            BATCHES[batch_id]["source"] = "api"
            BATCHES[batch_id]["api_key_id"] = request.api_key["id"]

    return (
        jsonify(
            {
                "ok": True,
                "batch_id": batch_id,
                "status_url": url_for("api.batch_status", batch_id=batch_id, _external=True),
                "results_url": url_for("api.batch_results", batch_id=batch_id, _external=True),
            }
        ),
        202,
    )


@api_bp.get("/batches/<batch_id>")
@require_api_key
def batch_status(batch_id: str):
    payload, status = _batch_payload(batch_id, request.api_key["id"], include_results=False)
    if not payload:
        return _error("Batch not found.", status)
    return jsonify(payload), status


@api_bp.get("/batches/<batch_id>/results")
@require_api_key
def batch_results(batch_id: str):
    payload, status = _batch_payload(batch_id, request.api_key["id"], include_results=True)
    if not payload:
        return _error("Batch not found.", status)
    return jsonify(payload), status
