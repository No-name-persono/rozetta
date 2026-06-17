from __future__ import annotations
import os, json, uuid
from typing import Dict, Any
from flask import Blueprint, render_template, request, redirect, url_for, flash
from config import TEMPLATES_PATH

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _ensure_templates_file():
    if not os.path.exists(TEMPLATES_PATH):
        data = {
            "prompts": [
                {"id": str(uuid.uuid4()), "title": "Default", "system": (
                    "Ты — аналитик качества звонков в недвижимости. По расшифровке разговора определи, "
                    "есть ли негативные факторы и почему застройщик мог посчитать звонок НЕЦЕЛЕВЫМ.\\n"
                    "Строгий формат ответа:\\n"
                    "СТАТУС: <целевой | нецелевой — краткая причина>\\n"
                    "РАЗБОР:\\n"
                    "- Список негативных факторов (каждый пункт с краткой цитатой)\\n"
                    "ПРИМЕРЫ:\\n"
                    "- 1–3 короткие цитаты, подтверждающие выводы.\\n"
                    "Если оснований недостаточно — напиши: СТАТУС: целевой — оснований для нецелевого не выявлено.\\n"
                ), "locked": True}
            ],
            "checklists": [
                {"id": str(uuid.uuid4()), "title": "Стандарт", "items": [
                    "Представился оператор.",
                    "Спросил имя клиента.",
                    "Уточнил направление (пожелания / локацию).",
                    "Озвучил количество комнат.",
                    "Озвучил площадь.",
                    "Озвучил цену.",
                    "Озвучил сроки сдачи.",
                    "Спросил про сроки покупки (актуальность в ближайшие 3 месяца).",
                    "Спросил про обращение к данному застройщику за последние 3 месяца.",
                    "Зафиксировал ответ клиента на этот вопрос.",
                    "Повторил название ЖК перед переводом на менеджера застройщика.",
                ], "locked": True}
            ]
        }
        with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load() -> Dict[str, Any]:
    _ensure_templates_file()
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: Dict[str, Any]):
    with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@admin_bp.route("/templates", methods=["GET"])
def templates_page():
    data = _load()
    return render_template("admin_templates.html", prompts=data.get("prompts", []), checklists=data.get("checklists", []))


@admin_bp.route("/templates/prompt/add", methods=["POST"])
def add_prompt():
    data = _load()
    title = (request.form.get("title") or "").strip()
    system = (request.form.get("system") or "").strip()
    if not title or not system:
        flash("Укажите название и текст промта.")
        return redirect(url_for("admin.templates_page"))
    data.setdefault("prompts", []).append({"id": str(uuid.uuid4()), "title": title, "system": system, "locked": False})
    _save(data)
    flash("Промт добавлен.")
    return redirect(url_for("admin.templates_page"))


@admin_bp.route("/templates/prompt/update", methods=["POST"])
def update_prompt():
    data = _load()
    pid = request.form.get("id")
    title = (request.form.get("title") or "").strip()
    system = (request.form.get("system") or "").strip()
    for p in data.get("prompts", []):
        if p.get("id") == pid:
            if p.get("locked"):
                flash("Базовый промт заблокирован для изменений.")
                break
            if title: p["title"] = title
            if system: p["system"] = system
            _save(data)
            flash("Промт сохранён.")
            break
    return redirect(url_for("admin.templates_page"))


@admin_bp.route("/templates/prompt/delete", methods=["POST"])
def delete_prompt():
    data = _load()
    pid = request.form.get("id")
    new_prompts = []
    deleted = False
    for p in data.get("prompts", []):
        if p.get("id") == pid:
            if p.get("locked"):
                flash("Базовый промт нельзя удалить.")
                new_prompts.append(p)
            else:
                deleted = True
        else:
            new_prompts.append(p)
    data["prompts"] = new_prompts
    _save(data)
    flash("Промт удалён." if deleted else "Не найдено или заблокировано.")
    return redirect(url_for("admin.templates_page"))


@admin_bp.route("/templates/checklist/add", methods=["POST"])
def add_checklist():
    data = _load()
    title = (request.form.get("title") or "").strip()
    items = [v.strip() for k, v in sorted(request.form.items()) if k.startswith("item_") and v.strip()]
    if not title or not items:
        flash("Укажите название и хотя бы один пункт чек-листа.")
        return redirect(url_for("admin.templates_page"))
    data.setdefault("checklists", []).append({"id": str(uuid.uuid4()), "title": title, "items": items, "locked": False})
    _save(data)
    flash("Чек-лист добавлен.")
    return redirect(url_for("admin.templates_page"))


@admin_bp.route("/templates/checklist/update", methods=["POST"])
def update_checklist():
    data = _load()
    cid = request.form.get("id")
    title = (request.form.get("title") or "").strip()
    items = [v.strip() for k, v in sorted(request.form.items()) if k.startswith("item_") and v.strip()]
    for c in data.get("checklists", []):
        if c.get("id") == cid:
            if c.get("locked"):
                flash("Базовый чек-лист заблокирован для изменений.")
                break
            if title: c["title"] = title
            if items: c["items"] = items
            _save(data)
            flash("Чек-лист сохранён.")
            break
    return redirect(url_for("admin.templates_page"))


@admin_bp.route("/templates/checklist/delete", methods=["POST"])
def delete_checklist():
    data = _load()
    cid = request.form.get("id")
    new_checklists = []
    deleted = False
    for c in data.get("checklists", []):
        if c.get("id") == cid:
            if c.get("locked"):
                flash("Базовый чек-лист нельзя удалить.")
                new_checklists.append(c)
            else:
                deleted = True
        else:
            new_checklists.append(c)
    data["checklists"] = new_checklists
    _save(data)
    flash("Чек-лист удалён." if deleted else "Не найдено или заблокировано.")
    return redirect(url_for("admin.templates_page"))