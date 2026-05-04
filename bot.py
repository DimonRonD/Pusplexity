#!/usr/bin/env python3
"""
Pusplexity — чат-бот для обработки изображений через OpenAI GPT Image.
Поддерживает Telegram и CLI-режим.
"""

import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from action_logs import (
    build_daily_csv,
    build_daily_stats,
    extract_total_tokens,
    log_action,
)
from processor import ImageProcessor
from rag_store import DATA_DIR, RAGStore, load_document

load_dotenv()

# ANSI-цвета для уровней логирования
_COLORS = {
    logging.DEBUG: "\033[36m",    # cyan
    logging.INFO: "\033[32m",    # green
    logging.WARNING: "\033[33m", # yellow
    logging.ERROR: "\033[31m",   # red
    logging.CRITICAL: "\033[35m\033[1m",  # bold magenta
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Форматтер с выделением уровней цветом (только для TTY)."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if sys.stderr.isatty() and record.levelno in _COLORS:
            color = _COLORS[record.levelno]
            levelname = record.levelname
            spaces = 8 - len(levelname)  # padding из %(levelname)-8s
            old = f"| {levelname}{' ' * spaces}"
            new = f"| {color}{levelname}{_RESET}{' ' * spaces}"
            msg = msg.replace(old, new, 1)
        return msg


# Настройка логирования
# По умолчанию логи в консоль отключены (для запуска как сервис).
# Флаг --log или -v включает вывод в консоль.
_LOG_CONSOLE = "--log" in sys.argv or "-v" in sys.argv
if "--log" in sys.argv:
    sys.argv.remove("--log")
if "-v" in sys.argv:
    sys.argv.remove("-v")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

root = logging.getLogger()
root.handlers.clear()
root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if _LOG_CONSOLE:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    root.addHandler(_handler)
else:
    root.addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)
# Снижаем уровень логов python-telegram-bot
logging.getLogger("telegram").setLevel(logging.WARNING)


def run_telegram_bot():
    """Запуск Telegram-бота."""
    logger.info("Запуск Telegram-бота...")

    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
            PicklePersistence,
        )
    except ImportError:
        logger.error("Модуль python-telegram-bot не установлен")
        print(
            "Для Telegram-режима установите: pip install python-telegram-bot"
        )
        sys.exit(1)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        print(
            "Задайте TELEGRAM_BOT_TOKEN в .env (скопируйте из .env.example)"
        )
        sys.exit(1)

    logger.info("Токен загружен, инициализация процессора изображений")
    processor = ImageProcessor()

    def _log_tg_action(
        user_id: int | str,
        action: str,
        request_text: str | None = None,
        response_text: str | None = None,
        component: str | None = None,
        tokens_total: int | None = None,
    ) -> None:
        log_action(
            actor_id=str(user_id),
            source="telegram",
            action=action,
            request_text=request_text,
            response_text=response_text,
            component=component,
            tokens_total=tokens_total,
        )

    def _log_tg_error(
        user_id: int | str,
        exc: Exception | str,
        *,
        component: str | None = None,
        request_text: str | None = None,
    ) -> None:
        _log_tg_action(
            user_id=user_id,
            action="error",
            request_text=request_text,
            response_text=str(exc),
            component=component or "telegram",
        )

    def _parse_logs_date(raw: str | None):
        if not raw:
            return datetime.now(timezone.utc).date()
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None

    def _format_logs_report(stats: dict) -> str:
        components = stats.get("components", {})
        if components:
            comp_lines = "\n".join(f"  • {k}: {v}" for k, v in components.items())
        else:
            comp_lines = "  • нет данных"
        return (
            f"📊 Статистика за {stats['date']}\n\n"
            f"• Уникальные пользователи Telegram: {stats['unique_telegram_users']}\n"
            f"• Уникальные пользователи Web: {stats['unique_web_users']}\n"
            f"• Всего запросов пользователей: {stats['total_user_requests']}\n"
            f"• Израсходовано токенов: {stats['total_tokens']}\n\n"
            f"Компоненты:\n{comp_lines}"
        )

    LEGACY_TEXT_MODEL = "gpt-5.2"
    DEFAULT_TEXT_MODEL = "gpt-5.5"

    def _resolve_text_model(raw: str | None) -> str:
        model = (raw or "").strip()
        if not model or model == "latest":
            return DEFAULT_TEXT_MODEL
        return model

    TEXT_MODEL = _resolve_text_model(os.environ.get("OPENAI_TEXT_MODEL"))
    DEFAULT_MODEL = "gpt-image-1.5"
    MODELS = {
        TEXT_MODEL: TEXT_MODEL,
        "gpt-image-1": "gpt-image-1",
        "gpt-image-1.5": "gpt-image-1.5",
        "dall-e-2": "dall-e-2",
        "create": "create",  # text-to-image, gpt-image-1.5
        "dalle_create": "dalle_create",  # text-to-image, DALL-E 2
        "rag_text": "rag_text",  # RAG: ответы с контекстом из документов
    }

    def set_model(context: ContextTypes.DEFAULT_TYPE, model: str) -> str:
        """Устанавливает модель для пользователя. Возвращает имя модели."""
        context.user_data["model"] = model
        return model

    def get_model(context: ContextTypes.DEFAULT_TYPE) -> str:
        """Возвращает выбранную модель пользователя."""
        model = context.user_data.get("model", DEFAULT_MODEL)
        if model == LEGACY_TEXT_MODEL:
            # Мягкая миграция старого значения из persistence.
            context.user_data["model"] = TEXT_MODEL
            return TEXT_MODEL
        return model

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info("Команда /start от user_id=%s, username=%s", user.id, user.username)
        set_model(context, TEXT_MODEL)
        await update.message.reply_text(
            "🖼 Pusplexity\n\n"
            f"Режим по умолчанию: {TEXT_MODEL} (чат)\n\n"
            "◾ /text — чат, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD\n"
            "◾ /image1, /image15, /dalle — редактирование фото\n"
            "◾ /create, /dalle_gen — генерация по тексту\n"
            "◾ /rag_add, /rag_index, /rag_text — RAG: база знаний по документам\n\n"
            "/help — полная справка по всем командам"
        )

    MODEL_LABELS = {
        TEXT_MODEL: TEXT_MODEL,
        "gpt-image-1": "gpt-image-1",
        "gpt-image-1.5": "gpt-image-1.5",
        "dall-e-2": "DALL-E 2",
        "create": "gpt-image-1.5 (create)",
        "dalle_create": "DALL-E 2 (create)",
        "rag_text": "RAG",
    }

    def _format_image_error(exc: Exception) -> str:
        """Форматирует ошибки генерации изображений для пользователя."""
        code = None
        if hasattr(exc, "body") and isinstance(exc.body, dict):
            err = exc.body.get("error") or exc.body
            code = err.get("code") if isinstance(err, dict) else None
        if hasattr(exc, "code"):
            code = code or getattr(exc, "code", None)
        if not code and "moderation_blocked" in str(exc):
            code = "moderation_blocked"
        if code == "moderation_blocked":
            return (
                "⚠️ Запрос отклонён системой безопасности OpenAI.\n\n"
                "Попробуйте переформулировать описание. "
                "Если считаете это ошибкой — обратитесь в support: help.openai.com"
            )
        return str(exc)

    async def cmd_image1(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "gpt-image-1")
        await update.message.reply_text(
            "✅ Модель: gpt-image-1 (режим сохранён)\n\n"
            "Можно загружать 1–10 фото (альбомом или по одному). "
            "Фото объединяются для обработки."
        )

    async def cmd_image15(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "gpt-image-1.5")
        await update.message.reply_text(
            "✅ Модель: gpt-image-1.5 (режим сохранён)\n\n"
            "Можно загружать 1–10 фото (альбомом или по одному). "
            "Фото объединяются для обработки."
        )

    async def cmd_dalle(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "dall-e-2")
        await update.message.reply_text(
            "✅ Модель: DALL-E 2 (режим сохранён)\n\n"
            "⚠️ DALL-E 2 поддерживает только 1 изображение"
        )

    async def cmd_create(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "create")
        await update.message.reply_text(
            "✅ Режим: Create (режим сохранён)\n\n"
            "Генерация по тексту без фото. Отправьте текстовое описание — получите изображение.\n"
            "Модель: gpt-image-1.5"
        )

    async def cmd_dalle_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "dalle_create")
        await update.message.reply_text(
            "✅ Режим: DALL-E 2 Gen (режим сохранён)\n\n"
            "Генерация по тексту без фото. Отправьте текстовое описание — получите изображение.\n"
            "Модель: DALL-E 2 (до 1000 символов)"
        )

    TELEGRAM_MAX_MESSAGE = 4000  # лимит Telegram 4096, 4000 для совместимости
    CHAT_HISTORY_SIZE = 20  # последних сообщений (user+assistant) для контекста

    def _update_chat_history(
        user_data: dict, key: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Добавляет пару user/assistant в историю, обрезает до CHAT_HISTORY_SIZE."""
        history = list(user_data.get(key, []))
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        user_data[key] = history[-CHAT_HISTORY_SIZE:]

    async def cmd_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, TEXT_MODEL)
        await update.message.reply_text(
            f"✅ Переключено на текстовый режим ({TEXT_MODEL})\n\n"
            "• Чат только текстом\n"
            "• Анализ 1 фото по подписи\n"
            "• Загрузите DOCX, PDF, XLSX, TXT, MD как контекст — затем задавайте вопросы\n\n"
            "Длинные ответы разбиваются на несколько сообщений."
        )

    async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data["text_chat_history"] = []
        context.user_data.pop("text_context", None)
        context.user_data.pop("text_context_filename", None)
        await update.message.reply_text("✅ История сеанса /text очищена (включая загруженный контекст).")

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 Pusplexity — Справка по командам\n\n"
            "◾ Режимы работы\n"
            f"/start — Начало работы ({TEXT_MODEL} по умолчанию).\n"
            f"/text — Чат {TEXT_MODEL}: текст, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD. Память 20 сообщений.\n"
            "/image1 — gpt-image-1: редактирование 1–10 фото.\n"
            "/image15 — gpt-image-1.5: редактирование 1–10 фото.\n"
            "/dalle — DALL-E 2: редактирование 1 фото.\n"
            "/create — Генерация по тексту (gpt-image-1.5).\n"
            "/dalle_gen — Генерация по тексту (DALL-E 2, до 1000 символов).\n\n"
            "◾ RAG — база знаний\n"
            "/rag_add — Включить режим загрузки. Отправьте TXT, PDF, XLSX, DOCX, MD.\n"
            "/rag_index — Индексировать файлы из data/ в ChromaDB.\n"
            "/rag_list — Список источников в хранилище.\n"
            "/rag_delete <источник> — Удалить источник и его данные из ChromaDB.\n"
            "/rag_text — Режим RAG. Задавайте вопросы, ответы по документам. До смены режима.\n"
            "/rag_clear — Очистить историю сеанса /rag_text.\n\n"
            "/logs [YYYY-MM-DD] — статистика и CSV-лог за дату (по умолчанию сегодня).\n\n"
            "/clear — Очистить историю и контекст документа в /text.\n"
            "/help — Эта справка."
        )

    async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
        raw_date = " ".join(context.args or []).strip() if context.args else ""
        target_date = _parse_logs_date(raw_date)
        if target_date is None:
            await update.message.reply_text("Неверный формат даты. Используйте: /logs 2026-05-03")
            return
        stats = build_daily_stats(target_date)
        report = _format_logs_report(stats)
        await update.message.reply_text(report)
        csv_bytes = build_daily_csv(target_date)
        filename = f"logs_{target_date.isoformat()}.csv"
        await update.message.reply_document(
            document=io.BytesIO(csv_bytes),
            filename=filename,
            caption=f"CSV-лог за {target_date.isoformat()}",
        )

    RAG_ALLOWED_EXTENSIONS = (".txt", ".pdf", ".xlsx", ".xls", ".docx", ".md", ".text")
    TEXT_CONTEXT_EXTENSIONS = (".txt", ".pdf", ".xlsx", ".xls", ".docx", ".md", ".text")

    def _get_rag_store():
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан (нужен для эмбеддингов)")
        return RAGStore(api_key=api_key)

    async def cmd_rag_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data["rag_add_mode"] = True
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        await update.message.reply_text(
            "📂 Режим загрузки RAG включён.\n\n"
            "Отправьте документы (TXT, PDF, XLSX, DOCX). Можно несколько подряд.\n"
            "Чтобы выйти из режима — отправьте /rag_index или любую другую команду."
        )

    async def cmd_rag_index(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data["rag_add_mode"] = False
        msg = await update.message.reply_text("⏳ Индексация документов…")
        try:
            store = _get_rag_store()
            counts = await asyncio.to_thread(store.index_documents, DATA_DIR)
        except ValueError as e:
            await msg.edit_text(str(e))
            return
        except Exception as e:
            logger.exception("Ошибка индексации RAG: %s", e)
            await msg.edit_text(f"Ошибка индексации: {e}")
            return
        if not counts:
            await msg.edit_text(
                "Нет документов для индексации. Используйте /rag_add и отправьте файлы (TXT, PDF, XLSX, DOCX)."
            )
            return
        total = sum(counts.values())
        lines = [f"✅ Проиндексировано {total} чанков из {len(counts)} файлов:\n"]
        for src, cnt in sorted(counts.items()):
            lines.append(f"  • {src}: {cnt} чанков")
        await msg.edit_text("\n".join(lines))

    async def cmd_rag_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            store = _get_rag_store()
            sources = await asyncio.to_thread(store.list_sources)
        except ValueError as e:
            await update.message.reply_text(str(e))
            return
        except Exception as e:
            logger.exception("Ошибка list_sources: %s", e)
            await update.message.reply_text(f"Ошибка: {e}")
            return
        if not sources:
            await update.message.reply_text(
                "Хранилище RAG пусто. Используйте /rag_add и /rag_index."
            )
            return
        text = "📚 Источники в RAG:\n\n" + "\n".join(f"• {s}" for s in sources)
        await update.message.reply_text(text)

    async def cmd_rag_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data["rag_chat_history"] = []
        await update.message.reply_text("✅ История сеанса /rag_text очищена.")

    async def cmd_rag_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
        source = " ".join(context.args or []).strip() if context.args else ""
        if not source:
            await update.message.reply_text(
                "Использование: /rag_delete <источник>\n\n"
                "Укажите точное имя источника из /rag_list, например:\n"
                "/rag_delete doc.pdf"
            )
            return
        try:
            store = _get_rag_store()
            count = await asyncio.to_thread(
                store.delete_source, source, DATA_DIR
            )
        except ValueError as e:
            await update.message.reply_text(str(e))
            return
        except Exception as e:
            logger.exception("Ошибка удаления источника: %s", e)
            await update.message.reply_text(f"Ошибка: {e}")
            return
        await update.message.reply_text(
            f"✅ Источник «{source}» удалён ({count} чанков, файл из data/ при наличии)."
        )

    async def _process_rag_query(
        update: Update, context: ContextTypes.DEFAULT_TYPE, query: str
    ) -> None:
        """Обрабатывает RAG-запрос (поиск + ответ)."""
        _log_tg_action(update.effective_user.id, "user_request", request_text=query)
        msg = await update.message.reply_text("⏳ Ищу в базе и формирую ответ…")
        try:
            store = _get_rag_store()
            results = await asyncio.to_thread(store.query, query, 5)
        except ValueError as e:
            _log_tg_error(update.effective_user.id, e, component="rag")
            await msg.edit_text(str(e))
            return
        except Exception as e:
            logger.exception("Ошибка RAG query: %s", e)
            _log_tg_error(update.effective_user.id, e, component="rag")
            await msg.edit_text(f"Ошибка: {e}")
            return
        if not results:
            await msg.edit_text(
                "Хранилище RAG пусто или по запросу ничего не найдено. "
                "Используйте /rag_add и /rag_index."
            )
            return
        context_parts = [f"[{src}]\n{doc}" for doc, src, _ in results]
        rag_context = "\n\n---\n\n".join(context_parts)
        # Собираем источники с лучшим score: 1/(1+distance), выше = релевантнее
        source_scores: dict[str, float] = {}
        for doc, src, dist in results:
            if src:
                score = round(1 / (1 + dist), 3)
                if src not in source_scores or score > source_scores[src]:
                    source_scores[src] = score
        sources_line = ", ".join(f"{s} ({d})" for s, d in sorted(source_scores.items()))

        rag_history = list(context.user_data.get("rag_chat_history", []))
        _log_tg_action(
            update.effective_user.id,
            "ai_request",
            request_text=json.dumps(
                {
                    "mode": "rag_text",
                    "model": TEXT_MODEL,
                    "query": query,
                    "history": rag_history,
                    "cache_context": rag_context,
                },
                ensure_ascii=False,
            ),
            component="rag",
        )
        try:
            result_text, used_tokens = await asyncio.to_thread(
                processor.process_text_with_rag_context,
                query,
                rag_context,
                model=TEXT_MODEL,
                history=rag_history if rag_history else None,
            )
        except Exception as e:
            logger.exception("Ошибка OpenAI для /rag_text: %s", e)
            _log_tg_error(update.effective_user.id, e, component="rag", request_text=query)
            await msg.edit_text(f"Ошибка: {e}")
            return
        _update_chat_history(
            context.user_data, "rag_chat_history", query, result_text
        )
        await msg.delete()
        parts = chunk_text(result_text)
        if not parts:
            await update.message.reply_text("(Пустой ответ)")
        else:
            for part in parts:
                await update.message.reply_text(part)
        await update.message.reply_text(f"🤖 Модель ответа: {TEXT_MODEL}")
        await update.message.reply_text(f"📎 Источники (score): {sources_line}")
        _log_tg_action(update.effective_user.id, "ai_response", response_text=result_text, component="rag", tokens_total=used_tokens)

    async def cmd_rag_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "rag_text")
        query = (update.message.text or "").replace("/rag_text", "", 1).strip()
        if not query:
            await update.message.reply_text(
                "✅ Режим RAG включён.\n\n"
                "Задавайте вопросы — ответы будут сформированы на основе документов из хранилища.\n"
                "Для смены режима: /text, /image15 и другие команды."
            )
            return
        await _process_rag_query(update, context, query)

    async def handle_rag_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        if not doc or not doc.file_name:
            return
        ext = Path(doc.file_name).suffix.lower()

        # RAG add mode: сохраняем в data/
        if context.user_data.get("rag_add_mode") and ext in RAG_ALLOWED_EXTENSIONS:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            dest = DATA_DIR / doc.file_name
            try:
                tg_file = await context.bot.get_file(doc.file_id)
                await tg_file.download_to_drive(dest)
            except Exception as e:
                logger.exception("Ошибка загрузки документа: %s", e)
                await update.message.reply_text(f"Ошибка сохранения: {e}")
                return
            await update.message.reply_text(f"✅ Сохранён: {doc.file_name}")
            return

        # Режим /text: загрузка документа как контекст для вопросов
        if get_model(context) == TEXT_MODEL and ext in TEXT_CONTEXT_EXTENSIONS:
            msg = await update.message.reply_text("⏳ Загружаю документ как контекст…")
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    dest = Path(tmpdir) / doc.file_name
                    tg_file = await context.bot.get_file(doc.file_id)
                    await tg_file.download_to_drive(dest)
                    content = await asyncio.to_thread(load_document, dest)
                if not content or not content.strip():
                    await msg.edit_text("Не удалось извлечь текст из документа.")
                    return
                context.user_data["text_context"] = content.strip()
                context.user_data["text_context_filename"] = doc.file_name
                await msg.edit_text(
                    f"✅ Документ «{doc.file_name}» загружен как контекст.\n\n"
                    "Задайте вопрос — ответ будет с учётом содержимого документа.\n"
                    "/clear — сбросить контекст и историю."
                )
            except Exception as e:
                logger.exception("Ошибка загрузки документа для контекста: %s", e)
                await msg.edit_text(f"Ошибка: {e}")
            return

        if ext not in RAG_ALLOWED_EXTENSIONS:
            if context.user_data.get("rag_add_mode"):
                await update.message.reply_text(
                    f"Формат {ext} не поддерживается. Используйте TXT, PDF, XLSX или DOCX."
                )
            return
        await update.message.reply_text(
            "Для загрузки в RAG отправьте /rag_add. "
            "В режиме /text документ станет контекстом для вопросов."
        )

    MEDIA_GROUP_DELAY = 2  # сек — ждём все фото альбома
    media_groups: dict[str, dict] = {}  # media_group_id -> {file_ids, caption, first_update, user_id}

    async def process_media_group_after_delay(
        group_id: str, bot, application
    ):
        """Обработка собранного альбома после задержки (без JobQueue)."""
        await asyncio.sleep(MEDIA_GROUP_DELAY)
        data = media_groups.pop(group_id, None)
        if not data:
            return

        file_ids = data["file_ids"]
        caption = data.get("caption", "")
        first_update = data["first_update"]
        user_id = data["user_id"]
        user_data = application.user_data[user_id]
        model = user_data.get("model", DEFAULT_MODEL)

        if model == "rag_text":
            user_data["pending_images"] = []
            await first_update.message.reply_text("В режиме RAG отправляйте только текстовые вопросы.")
            return

        # Режимы create/dalle_create: только caption, фото не скачиваем
        if model in ("create", "dalle_create"):
            user_data["pending_images"] = []
            if caption:
                class Ctx:
                    pass
                ctx = Ctx()
                ctx.bot = bot
                ctx.application = application
                ctx.user_data = user_data
                await process_and_reply(first_update, ctx, [], caption)
            else:
                mode_name = "DALL-E Gen" if model == "dalle_create" else "Create"
                await first_update.message.reply_text(
                    f"В режиме {mode_name} отправьте текстовое описание (подпись к фото)."
                )
            return

        logger.info(
            "Собран альбом: %d фото от user_id=%s",
            len(file_ids),
            user_id,
        )

        images: list[bytes] = []
        for fid in file_ids:
            f = await bot.get_file(fid)
            images.append(bytes(await f.download_as_bytearray()))

        existing = list(user_data.get("pending_images", []))
        if model == TEXT_MODEL:
            images = images[:1]  # Текстовый режим: только 1 фото
        else:
            images = existing + images

        if len(images) > 10:
            images = images[-10:]
            await first_update.message.reply_text("Максимум 10 изображений. Использую последние 10.")

        class Ctx:
            pass
        ctx = Ctx()
        ctx.bot = bot
        ctx.application = application
        ctx.user_data = application.user_data[user_id]
        if caption:
            user_data["pending_images"] = []
            await process_and_reply(first_update, ctx, images, caption)
        else:
            user_data["pending_images"] = images
            model_label = MODEL_LABELS.get(model, model)
            await first_update.message.reply_text(
                f"Получено {len(images)} изображений. Модель: {model_label}\n"
                "Отправьте текстовую команду или добавьте подпись."
            )

    async def handle_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        caption = (message.caption or "").strip()

        # Режим RAG: только текстовые вопросы
        if get_model(context) == "rag_text":
            await message.reply_text("В режиме RAG отправляйте только текстовые вопросы.")
            return

        # Режимы create/dalle_create: только текст. Одиночное фото — обрабатываем caption
        if get_model(context) in ("create", "dalle_create") and not message.media_group_id:
            if caption:
                context.user_data["pending_images"] = []
                await process_and_reply(update, context, [], caption)
            else:
                mode_name = "DALL-E Gen" if get_model(context) == "dalle_create" else "Create"
                await message.reply_text(
                    f"В режиме {mode_name} отправьте только текстовое описание изображения "
                    "(без фото)."
                )
            return

        # Альбом: каждое фото приходит отдельным сообщением с общим media_group_id
        if message.media_group_id:
            file_id = None
            if message.photo:
                file_id = max(message.photo, key=lambda p: p.file_size).file_id
            elif message.document and (message.document.mime_type or "").startswith("image/"):
                file_id = message.document.file_id

            if not file_id:
                if message.document:
                    await message.reply_text("Поддерживаются только изображения (PNG, JPEG).")
                return

            group_id = str(message.media_group_id)
            if group_id in media_groups:
                media_groups[group_id]["file_ids"].append(file_id)
                logger.debug(
                    "Добавлено фото в альбом %s, всего %d",
                    group_id,
                    len(media_groups[group_id]["file_ids"]),
                )
            else:
                media_groups[group_id] = {
                    "file_ids": [file_id],
                    "caption": caption,
                    "user_id": user.id,
                    "first_update": update,
                }
                asyncio.create_task(
                    process_media_group_after_delay(
                        group_id, context.bot, context.application
                    )
                )
                logger.info("Начат сбор альбома %s от user_id=%s", group_id, user.id)
            return

        # Одиночное фото или документ
        images: list[bytes] = list(context.user_data.get("pending_images", []))
        if get_model(context) == TEXT_MODEL:
            images = []  # Текстовый режим: только 1 новое фото

        logger.info(
            "Получены изображения от user_id=%s: photo=%s, document=%s, caption_len=%d",
            user.id,
            bool(message.photo),
            bool(message.document),
            len(caption or ""),
        )

        if message.photo:
            largest = max(message.photo, key=lambda p: p.file_size)
            file = await context.bot.get_file(largest.file_id)
            images.append(bytes(await file.download_as_bytearray()))
        elif message.document:
            doc = message.document
            mime = doc.mime_type or ""
            logger.debug("Документ: file_id=%s, mime=%s", doc.file_id, mime)
            if not mime.startswith("image/"):
                logger.warning("Отклонён не-изображение от user_id=%s: mime=%s", user.id, mime)
                await message.reply_text("Поддерживаются только изображения (PNG, JPEG).")
                return
            file = await context.bot.get_file(doc.file_id)
            images.append(bytes(await file.download_as_bytearray()))

        if len(images) > 10:
            logger.info("Обрезка до 10 изображений (получено %d)", len(images))
            await message.reply_text("Максимум 10 изображений. Используйте последние 10.")
            images = images[-10:]

        if caption:
            logger.info(
                "Обработка запроса user_id=%s: %d изображений, prompt_len=%d",
                user.id,
                len(images),
                len(caption or ""),
            )
            context.user_data["pending_images"] = []
            await process_and_reply(update, context, images, caption)
        else:
            context.user_data["pending_images"] = images
            logger.debug("Сохранено %d изображений в pending для user_id=%s", len(images), user.id)
            model = get_model(context)
            model_label = MODEL_LABELS.get(model, model)
            await message.reply_text(
                f"Получено {len(images)} изображений. Модель: {model_label}\n"
                "Отправьте текстовую команду или добавьте подпись."
            )

    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        text = (message.text or "").strip()
        images = context.user_data.get("pending_images", [])

        logger.info(
            "Текстовое сообщение от user_id=%s: text_len=%d, pending_images=%d",
            user.id,
            len(text),
            len(images),
        )
        if text:
            _log_tg_action(user.id, "user_request", request_text=text)

        # Режим RAG: текст как вопрос по документам
        if get_model(context) == "rag_text":
            if not text:
                await message.reply_text("Введите вопрос для поиска в RAG.")
                return
            for text_part in chunk_text(text, TELEGRAM_MAX_MESSAGE):
                await _process_rag_query(update, context, text_part)
            return

        # Режим create: только текст, без изображений
        if get_model(context) == "create":
            if not text:
                await message.reply_text("Введите текстовое описание изображения.")
                return
            if len(text) > 4000:
                text = text[:4000] + "\n\n[... обрезано]"
                await message.reply_text("Описание обрезано до 4000 символов.")
            context.user_data["pending_images"] = []
            await process_and_reply(update, context, [], text)
            return

        # Режим dalle_create: только текст (DALL-E 2, лимит 1000 символов)
        if get_model(context) == "dalle_create":
            if not text:
                await message.reply_text("Введите текстовое описание изображения.")
                return
            if len(text) > 1000:
                text = text[:1000] + "\n\n[... обрезано]"
                await message.reply_text("DALL-E 2: описание обрезано до 1000 символов.")
            context.user_data["pending_images"] = []
            await process_and_reply(update, context, [], text)
            return

        # Режим text (latest): можно только текст ИЛИ текст + 1 фото
        if get_model(context) == TEXT_MODEL:
            if not text:
                await message.reply_text("Введите сообщение или отправьте фото с подписью.")
                return
            context.user_data["pending_images"] = []
            text_parts = chunk_text(text, TELEGRAM_MAX_MESSAGE)
            for idx, text_part in enumerate(text_parts):
                # В текстовом режиме фото допустимо только в первом запросе.
                part_images = images if idx == 0 else []
                await process_and_reply(update, context, part_images, text_part)
            return

        if not images:
            logger.debug("user_id=%s: нет отложенных изображений", user.id)
            await message.reply_text(
                "Сначала отправьте изображение, затем команду текстом."
            )
            return

        # Ограничение длины промпта для совместимости с API
        if len(text) > 4000:
            text = text[:4000] + "\n\n[... обрезано]"
            await message.reply_text("Промпт обрезан до 4000 символов.")

        context.user_data["pending_images"] = []
        await process_and_reply(update, context, images, text)

    def chunk_text(text: str, max_len: int = TELEGRAM_MAX_MESSAGE) -> list[str]:
        """Разбивает текст на части не более max_len символов."""
        if len(text) <= max_len:
            return [text] if text else []
        chunks = []
        while text:
            chunk = text[:max_len]
            # Пытаемся разбить по переносу строки
            last_nl = chunk.rfind("\n")
            if last_nl > max_len // 2:
                chunk = text[: last_nl + 1]
            chunks.append(chunk)
            text = text[len(chunk) :]
        return chunks

    async def process_and_reply(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        images: list,
        prompt: str,
    ):
        user = update.effective_user
        model = get_model(context)
        logger.info(
            "Начало обработки для user_id=%s: %d изображений, model=%s, prompt_len=%d",
            user.id,
            len(images),
            model,
            len(prompt or ""),
        )

        # Режим create: только текст → изображение (images.generate, gpt-image-1.5)
        if model == "create":
            model_label = MODEL_LABELS.get(model, model)
            message = await update.message.reply_text(
                f"Генерирую изображение ({model_label})…"
            )
            _log_tg_action(
                user.id,
                "ai_request",
                request_text=json.dumps(
                    {"mode": "create", "model": "gpt-image-1.5", "prompt": prompt},
                    ensure_ascii=False,
                ),
                component="create",
            )
            try:
                result_bytes, usage_str = await asyncio.to_thread(
                    processor.process_create,
                    prompt,
                    model="gpt-image-1.5",
                )
            except ValueError as e:
                _log_tg_error(user.id, e, component="create", request_text=prompt)
                await message.edit_text(str(e))
                return
            except Exception as e:
                err_msg = _format_image_error(e)
                if "moderation_blocked" in str(e):
                    logger.info("Create: moderation blocked для user_id=%s", user.id)
                else:
                    logger.exception(
                        "Ошибка create для user_id=%s: %s",
                        user.id,
                        e,
                    )
                _log_tg_error(user.id, e, component="create", request_text=prompt)
                await message.edit_text(err_msg)
                return
            output = Path("temp_output.png")
            output.write_bytes(result_bytes)
            caption = f"Модель: {model_label}"
            if usage_str:
                caption += f"\n{usage_str}"
            try:
                await update.message.reply_photo(
                    photo=output.open("rb"),
                    caption=caption,
                )
            finally:
                output.unlink(missing_ok=True)
            await message.delete()
            _log_tg_action(
                user.id,
                "ai_response",
                response_text=usage_str or "image_generated",
                component="create",
                tokens_total=extract_total_tokens(usage_str),
            )
            return

        # Режим dalle_create: только текст → изображение (images.generate, DALL-E 2)
        if model == "dalle_create":
            model_label = MODEL_LABELS.get(model, model)
            message = await update.message.reply_text(
                f"Генерирую изображение ({model_label})…"
            )
            _log_tg_action(
                user.id,
                "ai_request",
                request_text=json.dumps(
                    {"mode": "dalle_create", "model": "dall-e-2", "prompt": prompt},
                    ensure_ascii=False,
                ),
                component="dalle_create",
            )
            try:
                result_bytes, usage_str = await asyncio.to_thread(
                    processor.process_create,
                    prompt,
                    model="dall-e-2",
                )
            except ValueError as e:
                _log_tg_error(user.id, e, component="dalle_create", request_text=prompt)
                await message.edit_text(str(e))
                return
            except Exception as e:
                err_msg = _format_image_error(e)
                if "moderation_blocked" in str(e):
                    logger.info("DALL-E create: moderation blocked для user_id=%s", user.id)
                else:
                    logger.exception(
                        "Ошибка dalle_create для user_id=%s: %s",
                        user.id,
                        e,
                    )
                _log_tg_error(user.id, e, component="dalle_create", request_text=prompt)
                await message.edit_text(err_msg)
                return
            output = Path("temp_output.png")
            output.write_bytes(result_bytes)
            caption = f"Модель: {model_label}"
            if usage_str:
                caption += f"\n{usage_str}"
            try:
                await update.message.reply_photo(
                    photo=output.open("rb"),
                    caption=caption,
                )
            finally:
                output.unlink(missing_ok=True)
            await message.delete()
            _log_tg_action(
                user.id,
                "ai_response",
                response_text=usage_str or "image_generated",
                component="dalle_create",
                tokens_total=extract_total_tokens(usage_str),
            )
            return

        # Текстовый режим (latest): только текст ИЛИ 1 изображение + текст ИЛИ текст с контекстом документа
        if model == TEXT_MODEL:
            message = await update.message.reply_text("Обрабатываю…")
            text_history = list(context.user_data.get("text_chat_history", []))
            text_context = context.user_data.get("text_context")
            _log_tg_action(
                user.id,
                "ai_request",
                request_text=json.dumps(
                    {
                        "mode": "text",
                        "model": model,
                        "prompt": prompt,
                        "history": text_history,
                        "cache_context": text_context,
                        "images_count": len(images),
                    },
                    ensure_ascii=False,
                ),
                component="text",
            )
            try:
                if images:
                    if len(images) > 1:
                        images = images[:1]
                        logger.info("Текстовый режим: берём 1 изображение")
                    result_text, used_tokens = await asyncio.to_thread(
                        processor.process_text_with_image,
                        images[0],
                        prompt,
                        model=model,
                        history=text_history if text_history else None,
                    )
                elif text_context:
                    result_text, used_tokens = await asyncio.to_thread(
                        processor.process_text_with_rag_context,
                        prompt,
                        text_context,
                        model=model,
                        history=text_history if text_history else None,
                    )
                else:
                    result_text, used_tokens = await asyncio.to_thread(
                        processor.process_text_only,
                        prompt,
                        model=model,
                        history=text_history if text_history else None,
                    )
            except (ValueError, IndexError) as e:
                _log_tg_error(user.id, e, component="text", request_text=prompt)
                await message.edit_text(str(e))
                return
            except Exception as e:
                logger.exception("Ошибка текстового режима для user_id=%s: %s", user.id, e)
                _log_tg_error(user.id, e, component="text", request_text=prompt)
                await message.edit_text(f"Ошибка: {e}")
                return

            user_msg_for_history = f"[Изображение] {prompt}" if images else prompt
            _update_chat_history(
                context.user_data, "text_chat_history", user_msg_for_history, result_text
            )
            await message.delete()
            parts = chunk_text(result_text)
            if not parts:
                await update.message.reply_text("(Пустой ответ)")
            else:
                for part in parts:
                    await update.message.reply_text(part)
            await update.message.reply_text(f"🤖 Модель ответа: {model}")
            _log_tg_action(
                user.id,
                "ai_response",
                response_text=result_text,
                component="text",
                tokens_total=used_tokens,
            )
            return

        # Режим генерации изображений
        model_label = MODEL_LABELS.get(model, model)
        message = await update.message.reply_text(
            f"Обрабатываю изображения ({model_label})…"
        )
        _log_tg_action(
            user.id,
            "ai_request",
            request_text=json.dumps(
                {
                    "mode": "image_edit",
                    "model": model,
                    "prompt": prompt,
                    "images_count": len(images),
                },
                ensure_ascii=False,
            ),
            component="image_edit",
        )
        try:
            result_bytes, usage_str = await asyncio.to_thread(
                processor.process,
                images,
                prompt,
                model=model,
            )
        except ValueError as e:
            logger.warning("Ошибка валидации для user_id=%s: %s", user.id, e)
            _log_tg_error(user.id, e, component="image_edit", request_text=prompt)
            await message.edit_text(str(e))
            return
        except Exception as e:
            err_msg = _format_image_error(e)
            if "moderation_blocked" in str(e):
                logger.info("Обработка: moderation blocked для user_id=%s", user.id)
            else:
                logger.exception(
                    "Ошибка при обработке для user_id=%s: %s",
                    user.id,
                    e,
                )
            _log_tg_error(user.id, e, component="image_edit", request_text=prompt)
            await message.edit_text(err_msg)
            return

        logger.info(
            "Успешная обработка для user_id=%s: результат %d байт",
            user.id,
            len(result_bytes),
        )

        output = Path("temp_output.png")
        output.write_bytes(result_bytes)
        caption = f"Модель: {model_label}"
        if usage_str:
            caption += f"\n{usage_str}"
        try:
            await update.message.reply_photo(
                photo=output.open("rb"),
                caption=caption,
            )
            logger.debug("Фото отправлено user_id=%s", user.id)
        finally:
            output.unlink(missing_ok=True)
        await message.delete()
        _log_tg_action(
            user.id,
            "ai_response",
            response_text=usage_str or "image_generated",
            component="image_edit",
            tokens_total=extract_total_tokens(usage_str),
        )

    def main():
        persistence_path = os.environ.get("BOT_DATA_PATH", "bot_data.pickle")
        persistence = PicklePersistence(persistence_path)
        app = Application.builder().token(token).persistence(persistence).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("text", cmd_text))
        app.add_handler(CommandHandler("image1", cmd_image1))
        app.add_handler(CommandHandler("image15", cmd_image15))
        app.add_handler(CommandHandler("dalle", cmd_dalle))
        app.add_handler(CommandHandler("create", cmd_create))
        app.add_handler(CommandHandler("dalle_gen", cmd_dalle_gen))
        app.add_handler(CommandHandler("help", cmd_help))
        app.add_handler(CommandHandler("clear", cmd_clear))
        app.add_handler(CommandHandler("rag_add", cmd_rag_add))
        app.add_handler(CommandHandler("rag_index", cmd_rag_index))
        app.add_handler(CommandHandler("rag_list", cmd_rag_list))
        app.add_handler(CommandHandler("rag_delete", cmd_rag_delete))
        app.add_handler(CommandHandler("rag_text", cmd_rag_text))
        app.add_handler(CommandHandler("rag_clear", cmd_rag_clear))
        app.add_handler(CommandHandler("logs", cmd_logs))
        app.add_handler(
            MessageHandler(
                filters.Document.ALL & ~filters.Document.IMAGE,
                handle_rag_document,
            )
        )
        app.add_handler(MessageHandler(filters.PHOTO, handle_images))
        app.add_handler(MessageHandler(filters.Document.IMAGE, handle_images))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

        logger.info("Telegram-бот запущен, ожидание сообщений...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram-бот остановлен")

    main()


def run_cli():
    """Интерактивный CLI-режим."""
    logger.info("Запуск CLI-режима")

    processor = ImageProcessor()

    print("Pusplexity — обработка изображений через OpenAI\n")
    print("Введите пути к изображениям (1-10 шт.), через пробел:")
    paths_str = input("> ").strip()
    paths = [Path(p.strip()) for p in paths_str.split() if p.strip()]

    if not paths:
        print("Изображения не указаны.")
        return

    for p in paths:
        if not p.exists():
            logger.error("Файл не найден: %s", p)
            print(f"Файл не найден: {p}")
            return

    logger.info("CLI: введено %d путей: %s", len(paths), paths)

    print("\nВведите текстовую команду:")
    prompt = input("> ").strip()
    if not prompt:
        print("Команда не может быть пустой.")
        return

    logger.info("CLI: обработка %d изображений, prompt_len=%d", len(paths), len(prompt))
    print("Обрабатываю…")
    try:
        result_bytes, usage_str = processor.process(paths, prompt)
    except Exception as e:
        logger.exception("CLI: ошибка при обработке: %s", e)
        print(f"Ошибка: {e}")
        return

    output_path = Path("output.png")
    output_path.write_bytes(result_bytes)
    logger.info("CLI: результат сохранён в %s (%d байт)", output_path, len(result_bytes))
    print(f"Готово. Результат сохранён в {output_path.absolute()}")
    if usage_str:
        print(usage_str)


if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if mode == "cli":
        run_cli()
    elif mode == "telegram" or mode == "tg":
        run_telegram_bot()
    else:
        print("Pusplexity")
        print("Использование:")
        print("  python bot.py telegram    — запуск Telegram-бота")
        print("  python bot.py cli         — интерактивный режим в консоли")
        print("  python bot.py telegram -v — с выводом логов в консоль (--log)")
        print()
        print("По умолчанию логи в консоль отключены (для работы как сервис).")
        print("Флаг --log или -v включает вывод логов.")
        print()
        print("Настройте .env (скопируйте .env.example):")
        print("  OPENAI_API_KEY — ключ OpenAI")
        print("  TELEGRAM_BOT_TOKEN — токен бота (для Telegram)")
