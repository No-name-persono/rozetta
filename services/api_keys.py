from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from config import DEFAULTS

KEY_PREFIX = "rz_"


def _db_path() -> str:
    return DEFAULTS["API_KEYS_DB"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")


def create_api_key(name: str) -> dict[str, Any]:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("name is required")

    init_db()
    raw_key = KEY_PREFIX + secrets.token_urlsafe(32)
    row = {
        "name": clean_name,
        "key": raw_key,
        "key_prefix": raw_key[:12],
        "created_at": _now(),
    }
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO api_keys (name, key_hash, key_prefix, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (row["name"], _hash_key(raw_key), row["key_prefix"], row["created_at"]),
        )
        row["id"] = cur.lastrowid
    return row


def verify_api_key(key: str | None) -> dict[str, Any] | None:
    raw_key = (key or "").strip()
    if not raw_key:
        return None

    init_db()
    key_hash = _hash_key(raw_key)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, key_prefix, created_at, last_used_at, revoked_at
            FROM api_keys
            WHERE key_hash = ?
            """,
            (key_hash,),
        ).fetchone()
        if not row or row["revoked_at"]:
            return None
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["id"]))
        return dict(row)


def list_api_keys(include_revoked: bool = False) -> list[dict[str, Any]]:
    init_db()
    where = "" if include_revoked else "WHERE revoked_at IS NULL"
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, name, key_prefix, created_at, last_used_at, revoked_at
            FROM api_keys
            {where}
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def revoke_api_key(identifier: str) -> bool:
    value = (identifier or "").strip()
    if not value:
        raise ValueError("identifier is required")

    init_db()
    with _connect() as conn:
        if value.isdigit():
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (_now(), int(value)),
            )
        else:
            cur = conn.execute(
                """
                UPDATE api_keys
                SET revoked_at = ?
                WHERE (name = ? OR key_prefix = ?) AND revoked_at IS NULL
                """,
                (_now(), value, value),
            )
        return cur.rowcount > 0
