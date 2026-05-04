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
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_DIR = Path(os.environ.get("LOG_DB_PATH", Path(__file__).parent))
DB_PATH = _DB_DIR / "action_logs.db"
_LOCK = threading.Lock()
_LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "30"))
_PURGE_MIN_INTERVAL_SEC = 24 * 60 * 60
_last_purge_ts = 0.0

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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_logs_created_at ON action_logs(created_at)"
        )
        conn.commit()


def _purge_old_logs_locked(retention_days: int) -> int:
    if retention_days <= 0:
        return 0
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff_dt.isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM action_logs WHERE created_at < ?",
            (cutoff_iso,),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def maybe_purge_old_logs() -> int:
    """
    Периодически удаляет старые логи (по умолчанию старше 30 дней).
    Запускается с троттлингом: не чаще 1 раза в сутки.
    """
    global _last_purge_ts
    now_ts = time_module.time()
    if now_ts - _last_purge_ts < _PURGE_MIN_INTERVAL_SEC:
        return 0
    with _LOCK:
        # Повторная проверка после захвата lock.
        now_ts = time_module.time()
        if now_ts - _last_purge_ts < _PURGE_MIN_INTERVAL_SEC:
            return 0
        _init_db()
        deleted = _purge_old_logs_locked(_LOG_RETENTION_DAYS)
        _last_purge_ts = now_ts
        if deleted > 0:
            logger.info(
                "Удалено старых записей логов: %d (retention=%d days)",
                deleted,
                _LOG_RETENTION_DAYS,
            )
        return deleted


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
        maybe_purge_old_logs()
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


def _query_stats(
    *,
    target_day: date,
    actor_id: str | None = None,
    source: str | None = None,
) -> dict:
    start_iso, end_iso = _day_bounds(target_day)
    filters = ["created_at BETWEEN ? AND ?"]
    params: list[object] = [start_iso, end_iso]
    if actor_id is not None:
        filters.append("actor_id = ?")
        params.append(str(actor_id))
    if source is not None:
        filters.append("source = ?")
        params.append(source)
    where_sql = " AND ".join(filters)

    with _LOCK:
        _init_db()
        with _get_conn() as conn:
            total_requests = conn.execute(
                f"""
                SELECT COUNT(*) AS c
                FROM action_logs
                WHERE action='user_request' AND {where_sql}
                """,
                params,
            ).fetchone()["c"]
            total_tokens = conn.execute(
                f"""
                SELECT COALESCE(SUM(tokens_total), 0) AS c
                FROM action_logs
                WHERE action='ai_response' AND {where_sql}
                """,
                params,
            ).fetchone()["c"]
            component_rows = conn.execute(
                f"""
                SELECT COALESCE(component, 'unknown') AS component, COUNT(*) AS c
                FROM action_logs
                WHERE action='ai_request' AND {where_sql}
                GROUP BY COALESCE(component, 'unknown')
                ORDER BY c DESC, component ASC
                """,
                params,
            ).fetchall()
    return {
        "total_user_requests": int(total_requests or 0),
        "total_tokens": int(total_tokens or 0),
        "components": {row["component"]: int(row["c"] or 0) for row in component_rows},
    }


def build_daily_stats(target_day: date, actor_id: str | None = None, source: str | None = None) -> dict:
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
    scoped = _query_stats(target_day=target_day, actor_id=actor_id, source=source)
    return {
        "date": target_day.isoformat(),
        "unique_telegram_users": int(unique_tg or 0),
        "unique_web_users": int(unique_web or 0),
        "total_user_requests": scoped["total_user_requests"],
        "total_tokens": scoped["total_tokens"],
        "components": scoped["components"],
    }


def build_daily_dual_stats(target_day: date, actor_id: str, source: str) -> dict:
    total = build_daily_stats(target_day)
    mine = _query_stats(target_day=target_day, actor_id=actor_id, source=source)
    return {
        "date": target_day.isoformat(),
        "unique_telegram_users": total["unique_telegram_users"],
        "unique_web_users": total["unique_web_users"],
        "mine": mine,
        "total": {
            "total_user_requests": total["total_user_requests"],
            "total_tokens": total["total_tokens"],
            "components": total["components"],
        },
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

