import re
import difflib
from typing import Dict, List, Any, Optional, Tuple

_STATUS_RX = re.compile(r"\b(да|нет|не выявлено)\b", re.IGNORECASE)
_ANALYSIS_STATUS_RX = re.compile(r"^\s*СТАТУС\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ANALYSIS_REASON_RX = re.compile(r"^\s*ПРИЧИНА\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_word_rx = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _norm_text(t: str) -> str:
    return " ".join((t or "").split()).lower()


def _tokens(s: str) -> List[str]:
    return [w for w in _word_rx.findall(_norm_text(s)) if len(w) >= 3]


def _bigrams(tokens: List[str]) -> set:
    return set(zip(tokens, tokens[1:]))


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / float(len(a | b))


def build_segments(op_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = (op_json.get("response") or {}).get("chunks") or []
    segments, last = [], None
    def _t(ts):
        try: return float(str(ts).rstrip("s"))
        except: return None
    def _mmss(x):
        if x is None: return "--:--"
        s = float(x); m = int(s // 60); sec = int(round(s - m*60))
        return f"{m:02d}:{sec:02d}"

    for ch in chunks:
        best = (ch.get("alternatives") or [{}])[0]
        text = (best.get("text") or "").strip()
        if not text: continue
        words = best.get("words") or []
        s = _t(words[0].get("startTime")) if words else None
        e = _t(words[-1].get("endTime")) if words else None
        norm = _norm_text(text)
        # анти-дубликаты
        is_dup = False
        if last is not None:
            s_close = (s is None or last['s'] is None or abs(s - last['s']) <= 0.35)
            e_close = (e is None or last['e'] is None or abs(e - last['e']) <= 0.35)
            sim_ok = _similar(norm, last['norm']) >= 0.96
            if s_close and e_close and sim_ok:
                is_dup = True
        if is_dup: continue
        seg_id = f"seg-{len(segments)+1}"
        segments.append({
            "i": len(segments)+1,
            "s": s, "e": e,
            "text": text,
            "id": seg_id,
            "mmss": _mmss(s),
            "e_mmss": _mmss(e),
            "norm": norm,
        })
        last = {"s": s, "e": e, "norm": norm}
    return segments


def parse_analysis_status(llm_text: str) -> Dict[str, str]:
    """
    Pulls the business status from model output into a stable API object.
    Processing status remains in item["status"]; this is the call-analysis status.
    """
    out = {"value": "unknown", "label": "", "reason": "", "raw": ""}
    if not llm_text:
        return out

    status_match = _ANALYSIS_STATUS_RX.search(llm_text)
    if not status_match:
        return out

    raw = " ".join(status_match.group(1).strip().split())
    if not raw:
        return out

    parts = re.split(r"\s+[—–-]\s+", raw, maxsplit=1)
    label = parts[0].strip(" .:;")
    reason = parts[1].strip(" .:;") if len(parts) > 1 else ""

    if not reason:
        reason_match = _ANALYSIS_REASON_RX.search(llm_text)
        if reason_match:
            reason = " ".join(reason_match.group(1).strip().split()).strip(" .:;")

    out.update({
        "value": _analysis_status_value(label),
        "label": label,
        "reason": reason,
        "raw": raw,
    })
    return out


def _analysis_status_value(label: str) -> str:
    text = _norm_text(label).replace("ё", "е")
    if "нецелев" in text:
        return "not_target"
    if "целев" in text:
        return "target"
    if "заинтересован" in text:
        return "interested"
    if "не определ" in text or "неопредел" in text:
        return "uncertain"
    if "негатив" in text:
        return "not_target"
    return "unknown"


def parse_checklist(llm_text: str, items: List[str]) -> Dict[str, Dict[str, str]]:
    out = {i: {"status": "не выявлено", "quote": ""} for i in items}
    if not llm_text:
        return out
    lower = llm_text.lower()
    start_idx = lower.find("чек-лист")
    block = llm_text[start_idx:] if start_idx >= 0 else llm_text
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    num_line_rx = re.compile(r"^(\d+)[\).\-–—]*\s*(.+?)\s*[:\-–—]", re.IGNORECASE)
    for ln in lines:
        m = num_line_rx.match(ln)
        if not m: continue
        text_after = ln[m.end():].strip()
        status_m = _STATUS_RX.search(text_after)
        status = status_m.group(1).lower() if status_m else "не выявлено"
        quote = text_after[status_m.end():].strip(" :—-–\t\u00A0") if status_m else ""
        title_raw = m.group(2).strip().rstrip(".")
        best_item, best_sim = None, 0.0
        for item in items:
            sim = difflib.SequenceMatcher(None, title_raw.lower(), item.lower().rstrip(".")).ratio()
            if sim > best_sim:
                best_item, best_sim = item, sim
        if best_item and best_sim >= 0.6:
            out[best_item] = {"status": status, "quote": quote}
    # эвристика
    lower_full = llm_text.lower()
    for item in items:
        if out[item]["status"] != "не выявлено":
            continue
        idx = lower_full.find(item.lower().rstrip("."))
        if idx >= 0:
            tail = llm_text[idx: idx + 200]
            sm = _STATUS_RX.search(tail)
            status = sm.group(1).lower() if sm else "не выявлено"
            quote = tail[sm.end():].split("\n")[0].strip(" :—-–") if sm else ""
            out[item] = {"status": status, "quote": quote}
    return out


def annotate_checklist_with_timestamps(checklist: Dict[str, Dict[str, str]], segments: list) -> Dict[str, Dict[str, str]]:
    ts_pattern = re.compile(r"\[TS\s+(\d{1,2}:\d{2}(?::\d{2})?)\]")
    def _best_segment_for_quote(quote: str, item: str):
        qtoks = set(_tokens(quote))
        qbi = _bigrams(list(qtoks))
        best = (0.0, None)
        # простая стратегия: fuzzy + биграммы
        for seg in segments:
            stoks_list = _tokens(seg["text"])
            stoks = set(stoks_list)
            cover = len(qtoks & stoks) / float(len(qtoks) or 1)
            if cover > best[0]:
                best = (cover, seg)
        if best[1] and best[0] >= 0.5:
            seg = best[1]
            return seg["mmss"], seg["id"]
        # фолбэк
        target = _norm_text(quote)
        best2 = (0.0, None)
        for seg in segments:
            sim = _similar(target, seg["norm"])
            if sim > best2[0]:
                best2 = (sim, seg)
        if best2[1] and best2[0] >= 0.45:
            seg = best2[1]
            return seg["mmss"], seg["id"]
        return None

    # основной проход
    for item, row in checklist.items():
        quote_text = row.get("quote", "") or ""
        ts_match = ts_pattern.search(quote_text)
        if ts_match:
            ts_str = ts_match.group(1)
            row["ts"] = ts_str
            row["quote"] = ts_pattern.sub("", quote_text).strip()
            # seg_id подберём эвристикой по цитате
            info = _best_segment_for_quote(row["quote"], item)
            if info:
                row["seg_id"] = info[1]
            else:
                row["seg_id"] = ""
            continue
        info = _best_segment_for_quote(row.get("quote", ""), item)
        if info:
            row["ts"], row["seg_id"] = info
        else:
            row["ts"] = row["seg_id"] = ""
    return checklist
