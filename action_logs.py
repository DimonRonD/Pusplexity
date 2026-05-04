"""
SQLite-логирование действий пользователей.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_DIR = Path(os.environ.get("LOG_DB_PATH", Path(__file__).parent))
DB_PATH = _DB_DIR / "action_logs.db"
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS action_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id TEXT NOT NULL,
    source TEXT NOT NULL,        -- telegram | web
    action TEXT NOT NULL,        -- command, upload, request, response, etc.
    request_text TEXT,
    response_text TEXT,
    created_at TEXT NOT NULL
);
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


def log_action(
    actor_id: str,
    source: str,
    action: str,
    request_text: str | None = None,
    response_text: str | None = None,
) -> None:
    """
    Безопасно записывает событие в БД логов, не падая при ошибках.
    """
    try:
        with _LOCK:
            _init_db()
            with _get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO action_logs (
                        actor_id, source, action, request_text, response_text, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(actor_id),
                        source,
                        action,
                        request_text,
                        response_text,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
    except Exception as e:
        logger.exception("Не удалось записать action log: %s", e)

