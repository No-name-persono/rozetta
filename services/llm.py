import requests
from config import DEFAULTS


def _auth_headers_llm() -> dict:
    headers = {"Content-Type": "application/json"}
    if DEFAULTS["LLM_IAM_TOKEN"]:
        headers["Authorization"] = f"Bearer {DEFAULTS['LLM_IAM_TOKEN']}"
    elif DEFAULTS["LLM_API_KEY"]:
        headers["Authorization"] = f"Api-Key {DEFAULTS['LLM_API_KEY']}"
    else:
        raise RuntimeError("[LLM] Не указан ни IAM_TOKEN, ни API_KEY")
    return headers


def build_system_with_checklist(base_system: str, checklist_items: list[str]) -> str:
    # «Сшиваем» чек-лист в system-подсказку, чтобы модель гарантированно держала формат
    numbered = "\n".join(f"{i+1}. {t.rstrip('.')}" for i, t in enumerate(checklist_items))
    tail = (
        "\n\nЧЕК-ЛИСТ (строгий формат ответа):\n"
        "Для каждого пункта укажи статус <да | нет | не выявлено> и короткую цитату.\n"
        "Обязательно добавляй таймкоды в виде [TS мм:сс] или [TS ч:мм:сс] в цитату.\n"
        f"Пункты чек-листа:\n{numbered}\n"
    )
    return base_system.rstrip() + tail


def analyze_with_llm(transcript_text: str, system_message: str, soft_labels: str = None) -> str:
    headers = _auth_headers_llm()

    # Собираем user message
    user_parts = []
    user_parts.append("Есть ли негативные факторы в диалоге? Ответь строго по формату выше.")

    if soft_labels:
        user_parts.append(
            "\n\nСПЕКТРАЛЬНЫЕ ПОДСКАЗКИ (мягкие метки из аудио-анализа):\n"
            "Используй их как вероятностные подсказки для определения спикеров. "
            "Приоритет отдавай смыслу текста диалога. "
            "Кластеры (C0, C1, …) — спектральные группы, не подтверждённые спикеры.\n"
            + soft_labels
        )

    user_parts.append("\n\nРАСШИФРОВКА:\n" + (transcript_text or ""))

    data = {
        "modelUri": DEFAULTS["LLM_MODEL_URI"],
        "completionOptions": {
            "stream": False,
            "temperature": DEFAULTS["LLM_TEMPERATURE"],
            "maxTokens": DEFAULTS["LLM_MAX_TOKENS"],
        },
        "messages": [
            {"role": "system", "text": system_message},
            {"role": "user", "text": "".join(user_parts)},
        ],
    }
    r = requests.post(DEFAULTS["LLM_API_URL"], headers=headers, json=data, timeout=180)
    if r.status_code != 200:
        return f"Ошибка LLM: статус {r.status_code}, ответ: {r.text[:700]}"
    js = r.json()
    return (
        js.get("result", {})
        .get("alternatives", [{}])[0]
        .get("message", {})
        .get("text", "")
        or "Ошибка: пустой ответ от модели."
    )
