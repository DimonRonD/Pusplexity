"""
Хранение данных пользователей в SQLite.
Память чата сохраняется между сессиями и перезапусками.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Путь к БД — из env или рядом с модулем
_DB_DIR = Path(os.environ.get("USER_DB_PATH", Path(__file__).parent))
DB_PATH = _DB_DIR / "user_data.db"

# Схема
SCHEMA = """
CREATE TABLE IF NOT EXISTS user_data (
    email TEXT PRIMARY KEY,
    model TEXT NOT NULL DEFAULT 'gpt-5.2',
    text_chat_history TEXT NOT NULL DEFAULT '[]',
    rag_chat_history TEXT NOT NULL DEFAULT '[]',
    text_context TEXT,
    text_context_filename TEXT,
    updated_at TEXT NOT NULL
)
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def get_user_data(email: str) -> dict:
    """Загружает данные пользователя из БД."""
    _init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_data WHERE email = ?", (email,)
        ).fetchone()
    if not row:
        return {
            "model": "gpt-5.2",
            "pending_images": [],
            "text_chat_history": [],
            "rag_chat_history": [],
            "text_context": None,
            "text_context_filename": None,
            "rag_add_mode": False,
        }
    return {
        "model": row["model"] or "gpt-5.2",
        "pending_images": [],  # не храним — временные
        "text_chat_history": json.loads(row["text_chat_history"] or "[]"),
        "rag_chat_history": json.loads(row["rag_chat_history"] or "[]"),
        "text_context": row["text_context"],
        "text_context_filename": row["text_context_filename"],
        "rag_add_mode": False,  # сессионное состояние
    }


def save_user_data(email: str, data: dict) -> None:
    """Сохраняет данные пользователя в БД (без pending_images и rag_add_mode)."""
    import datetime
    _init_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_data (email, model, text_chat_history, rag_chat_history,
                                  text_context, text_context_filename, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                model = excluded.model,
                text_chat_history = excluded.text_chat_history,
                rag_chat_history = excluded.rag_chat_history,
                text_context = excluded.text_context,
                text_context_filename = excluded.text_context_filename,
                updated_at = excluded.updated_at
            """,
            (
                email,
                data.get("model", "gpt-5.2"),
                json.dumps(data.get("text_chat_history", []), ensure_ascii=False),
                json.dumps(data.get("rag_chat_history", []), ensure_ascii=False),
                data.get("text_context"),
                data.get("text_context_filename"),
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
