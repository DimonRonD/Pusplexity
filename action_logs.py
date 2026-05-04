"""
SQLite-логирование действий пользователей.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import sqlite3
import threading
from datetime import date, datetime, time, timezone
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
    component TEXT,
    tokens_total INTEGER,
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
        # Мягкая миграция для уже существующей таблицы.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(action_logs)").fetchall()}
        if "component" not in columns:
            conn.execute("ALTER TABLE action_logs ADD COLUMN component TEXT")
        if "tokens_total" not in columns:
            conn.execute("ALTER TABLE action_logs ADD COLUMN tokens_total INTEGER")
        conn.commit()


def log_action(
    actor_id: str,
    source: str,
    action: str,
    request_text: str | None = None,
    response_text: str | None = None,
    component: str | None = None,
    tokens_total: int | None = None,
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
                        actor_id, source, action, request_text, response_text, component, tokens_total, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(actor_id),
                        source,
                        action,
                        request_text,
                        response_text,
                        component,
                        tokens_total,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
    except Exception as e:
        logger.exception("Не удалось записать action log: %s", e)


def extract_total_tokens(text: str | None) -> int | None:
    """
    Извлекает общее число токенов из строки вида 'Токены: 123 ...'.
    """
    if not text:
        return None
    m = re.search(r"Токены:\s*(\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _day_bounds(target_day: date) -> tuple[str, str]:
    start_dt = datetime.combine(target_day, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(target_day, time.max, tzinfo=timezone.utc)
    return start_dt.isoformat(), end_dt.isoformat()


def build_daily_stats(target_day: date) -> dict:
    start_iso, end_iso = _day_bounds(target_day)
    with _LOCK:
        _init_db()
        with _get_conn() as conn:
            unique_tg = conn.execute(
                """
                SELECT COUNT(DISTINCT actor_id) AS c
                FROM action_logs
                WHERE source='telegram' AND created_at BETWEEN ? AND ?
                """,
                (start_iso, end_iso),
            ).fetchone()["c"]
            unique_web = conn.execute(
                """
                SELECT COUNT(DISTINCT actor_id) AS c
                FROM action_logs
                WHERE source='web' AND created_at BETWEEN ? AND ?
                """,
                (start_iso, end_iso),
            ).fetchone()["c"]
            total_requests = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM action_logs
                WHERE action='user_request' AND created_at BETWEEN ? AND ?
                """,
                (start_iso, end_iso),
            ).fetchone()["c"]
            total_tokens = conn.execute(
                """
                SELECT COALESCE(SUM(tokens_total), 0) AS c
                FROM action_logs
                WHERE action='ai_response' AND created_at BETWEEN ? AND ?
                """,
                (start_iso, end_iso),
            ).fetchone()["c"]
            component_rows = conn.execute(
                """
                SELECT COALESCE(component, 'unknown') AS component, COUNT(*) AS c
                FROM action_logs
                WHERE action='ai_request' AND created_at BETWEEN ? AND ?
                GROUP BY COALESCE(component, 'unknown')
                ORDER BY c DESC, component ASC
                """,
                (start_iso, end_iso),
            ).fetchall()
    return {
        "date": target_day.isoformat(),
        "unique_telegram_users": int(unique_tg or 0),
        "unique_web_users": int(unique_web or 0),
        "total_user_requests": int(total_requests or 0),
        "total_tokens": int(total_tokens or 0),
        "components": {row["component"]: int(row["c"] or 0) for row in component_rows},
    }


def build_daily_csv(target_day: date) -> bytes:
    start_iso, end_iso = _day_bounds(target_day)
    with _LOCK:
        _init_db()
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT actor_id, source, action, component, tokens_total, request_text, response_text, created_at
                FROM action_logs
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at ASC, id ASC
                """,
                (start_iso, end_iso),
            ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["actor_id", "source", "action", "component", "tokens_total", "request_text", "response_text", "created_at"]
    )
    for row in rows:
        writer.writerow(
            [
                row["actor_id"],
                row["source"],
                row["action"],
                row["component"],
                row["tokens_total"],
                row["request_text"] or "",
                row["response_text"] or "",
                row["created_at"],
            ]
        )
    return buf.getvalue().encode("utf-8-sig")

