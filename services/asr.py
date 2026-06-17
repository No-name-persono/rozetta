import json
from typing import Any, Dict, Iterable

import requests

from config import DEFAULTS


def _auth_headers(api_key: str, iam_token: str, folder_id: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if iam_token:
        headers["Authorization"] = f"Bearer {iam_token}"
    elif api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    else:
        raise RuntimeError("[STT] Не указан ни IAM_TOKEN, ни API_KEY")
    if folder_id:
        headers["x-folder-id"] = folder_id.strip()
    return headers


def _headers() -> Dict[str, str]:
    return _auth_headers(
        DEFAULTS["YANDEX_API_KEY"],
        DEFAULTS["YANDEX_IAM_TOKEN"],
        DEFAULTS["YANDEX_FOLDER_ID"],
    )


def _asr_version() -> str:
    value = str(DEFAULTS.get("ASR_API_VERSION", "v3") or "v3").lower()
    return "v2" if value == "v2" else "v3"


def start_long(
    uri: str,
    enc: str,
    lang: str | None = None,
    model: str | None = None,
    literature: bool | None = None,
) -> str:
    if _asr_version() == "v2":
        return _start_v2(uri, enc, lang, model, literature)
    return _start_v3(uri, enc, lang, model, literature)


def _start_v2(
    uri: str,
    enc: str,
    lang: str | None,
    model: str | None,
    literature: bool | None,
) -> str:
    payload = {
        "config": {
            "specification": {
                "languageCode": lang or DEFAULTS.get("ASR_LANGUAGE", "ru-RU"),
                "model": model or DEFAULTS.get("ASR_MODEL", "general"),
                "profanityFilter": False,
                "literature_text": DEFAULTS.get("ASR_LITERATURE_TEXT", True) if literature is None else bool(literature),
                "audioEncoding": enc,
            }
        },
        "audio": {"uri": uri},
    }
    r = requests.post(DEFAULTS["ASR_ENDPOINT"], headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    op_id = r.json().get("id")
    if not op_id:
        raise RuntimeError(f"STT v2 не вернул id: {r.text[:500]}")
    return op_id


def _start_v3(
    uri: str,
    enc: str,
    lang: str | None,
    model: str | None,
    literature: bool | None,
) -> str:
    language = lang or DEFAULTS.get("ASR_LANGUAGE", "ru-RU")
    payload = {
        "uri": uri,
        "recognition_model": {
            "model": model or DEFAULTS.get("ASR_MODEL", "general"),
            "audio_format": {
                "container_audio": {
                    "container_audio_type": enc,
                },
            },
            "text_normalization": {
                "profanity_filter": False,
                "literature_text": DEFAULTS.get("ASR_LITERATURE_TEXT", True) if literature is None else bool(literature),
            },
            "language_restriction": {
                "restriction_type": "WHITELIST",
                "language_code": [language],
            },
        },
    }
    r = requests.post(DEFAULTS["ASR_V3_ENDPOINT"], headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    op_id = r.json().get("id")
    if not op_id:
        raise RuntimeError(f"STT v3 не вернул id: {r.text[:500]}")
    return op_id


def poll_once(op_id: str) -> Dict[str, Any]:
    r = requests.get(DEFAULTS["OPS_ENDPOINT"].format(op_id), headers=_headers(), timeout=30)
    r.raise_for_status()
    operation = r.json()
    if _asr_version() == "v3" and operation.get("done") and not operation.get("error"):
        operation["response"] = {"chunks": _fetch_v3_chunks(op_id)}
    return operation


def _fetch_v3_chunks(op_id: str) -> list[dict[str, Any]]:
    r = requests.get(
        DEFAULTS["ASR_V3_RESULT_ENDPOINT"],
        headers=_headers(),
        params={"operation_id": op_id},
        timeout=120,
    )
    r.raise_for_status()
    events = [_unwrap_result(obj) for obj in _parse_json_stream(r.text)]
    return _v3_events_to_v2_chunks(events)


def _parse_json_stream(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [obj for obj in parsed if isinstance(obj, dict)]
    if isinstance(parsed, dict):
        return [parsed]

    decoder = json.JSONDecoder()
    pos = 0
    out = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
    return out


def _unwrap_result(obj: dict[str, Any]) -> dict[str, Any]:
    result = obj.get("result")
    return result if isinstance(result, dict) else obj


def _v3_events_to_v2_chunks(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    finals: dict[int, dict[str, Any]] = {}
    next_index = 0

    for event in events:
        final_refinement = event.get("finalRefinement") or event.get("final_refinement")
        if isinstance(final_refinement, dict):
            final_index = _safe_int(final_refinement.get("finalIndex") or final_refinement.get("final_index"), next_index)
            normalized = final_refinement.get("normalizedText") or final_refinement.get("normalized_text") or {}
            chunk = _alternative_update_to_chunk(normalized)
            if chunk:
                finals[final_index] = chunk
            continue

        final = event.get("final")
        if isinstance(final, dict):
            cursors = event.get("audioCursors") or event.get("audio_cursors") or {}
            final_index = _safe_int(cursors.get("finalIndex") or cursors.get("final_index"), next_index)
            chunk = _alternative_update_to_chunk(final)
            if chunk:
                finals[final_index] = chunk
                next_index = max(next_index, final_index + 1)

    return [finals[i] for i in sorted(finals)]


def _alternative_update_to_chunk(update: dict[str, Any]) -> dict[str, Any] | None:
    alternatives = update.get("alternatives") or []
    normalized = []
    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        text = (alt.get("text") or "").strip()
        words = [_v3_word_to_v2_word(w) for w in (alt.get("words") or []) if isinstance(w, dict)]
        if text or words:
            normalized.append({"text": text, "words": words})
    if not normalized:
        return None
    return {"alternatives": normalized}


def _v3_word_to_v2_word(word: dict[str, Any]) -> dict[str, str]:
    return {
        "word": str(word.get("text") or ""),
        "startTime": _ms_to_seconds(word.get("startTimeMs") or word.get("start_time_ms")),
        "endTime": _ms_to_seconds(word.get("endTimeMs") or word.get("end_time_ms")),
    }


def _ms_to_seconds(value: Any) -> str:
    try:
        return f"{float(value) / 1000.0:.3f}s"
    except (TypeError, ValueError):
        return "0.000s"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
