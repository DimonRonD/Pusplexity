#!/usr/bin/env python3
"""
ImageBot — чат-бот для обработки изображений через OpenAI GPT Image.
Поддерживает Telegram и CLI-режим.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from processor import ImageProcessor

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

    DEFAULT_MODEL = "gpt-image-1.5"
    MODELS = {
        "gpt-5.2": "gpt-5.2",
        "gpt-image-1": "gpt-image-1",
        "gpt-image-1.5": "gpt-image-1.5",
        "dall-e-2": "dall-e-2",
        "create": "create",  # text-to-image, gpt-image-1.5
        "dalle_create": "dalle_create",  # text-to-image, DALL-E 2
    }

    def set_model(context: ContextTypes.DEFAULT_TYPE, model: str) -> str:
        """Устанавливает модель для пользователя. Возвращает имя модели."""
        context.user_data["model"] = model
        return model

    def get_model(context: ContextTypes.DEFAULT_TYPE) -> str:
        """Возвращает выбранную модель пользователя."""
        return context.user_data.get("model", DEFAULT_MODEL)

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info("Команда /start от user_id=%s, username=%s", user.id, user.username)
        set_model(context, "gpt-5.2")
        await update.message.reply_text(
            "🖼 ImageBot\n\n"
            "Модель: gpt-5.2 (по умолчанию для нового пользователя)\n\n"
            "Режимы: чат текстом (/text), редактирование фото (/image1, /image15, /dalle), генерация по тексту (/create).\n\n"
            "Команды:\n"
            "/help — справка по всем командам\n"
            "/text — текстовый режим (gpt-5.2), 1 фото для распознавания\n"
            "/image1 — gpt-image-1\n"
            "/image15 — gpt-image-1.5\n"
            "/dalle — DALL-E 2 (редактирование 1 фото)\n"
            "/create — генерация по тексту (gpt-image-1.5)\n"
            "/dalle_gen — генерация по тексту (DALL-E 2)"
        )

    MODEL_LABELS = {
        "gpt-image-1": "gpt-image-1",
        "gpt-image-1.5": "gpt-image-1.5",
        "dall-e-2": "DALL-E 2",
        "create": "gpt-image-1.5 (create)",
        "dalle_create": "DALL-E 2 (create)",
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

    async def cmd_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        set_model(context, "gpt-5.2")
        await update.message.reply_text(
            "✅ Переключено на текстовый режим (gpt-5.2)\n\n"
            "Чат с OpenAI только текстом (без фото) или анализ 1 фото по текстовой команде.\n"
            "Длинные ответы (>4000 символов) разбиваются на несколько сообщений."
        )

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 ImageBot — Справка по командам\n\n"
            "/start — Начало работы. Устанавливает режим gpt-5.2 по умолчанию.\n\n"
            "/text — Текстовый режим (gpt-5.2). Чат только текстом или анализ 1 фото. Длинные ответы разбиваются на несколько сообщений.\n\n"
            "/image1 — Модель gpt-image-1. Редактирование 1–10 фото по текстовой команде.\n\n"
            "/image15 — Модель gpt-image-1.5. Редактирование 1–10 фото по текстовой команде.\n\n"
            "/dalle — DALL-E 2. Редактирование только 1 фото по текстовой команде.\n\n"
            "/create — Генерация по тексту без фото (gpt-image-1.5).\n\n"
            "/dalle_gen — Генерация по тексту без фото (DALL-E 2, до 1000 символов)."
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
        if model == "gpt-5.2":
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
        if get_model(context) == "gpt-5.2":
            images = []  # Текстовый режим: только 1 новое фото

        logger.info(
            "Получены изображения от user_id=%s: photo=%s, document=%s, caption=%r",
            user.id,
            bool(message.photo),
            bool(message.document),
            caption or None,
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
                "Обработка запроса user_id=%s: %d изображений, prompt=%r",
                user.id,
                len(images),
                caption,
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
            "Текстовое сообщение от user_id=%s: %r, pending_images=%d",
            user.id,
            text[:100],
            len(images),
        )

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

        # Режим text (gpt-5.2): можно только текст ИЛИ текст + 1 фото
        if get_model(context) == "gpt-5.2":
            if not text:
                await message.reply_text("Введите сообщение или отправьте фото с подписью.")
                return
            if len(text) > 4000:
                text = text[:4000] + "\n\n[... обрезано]"
                await message.reply_text("Промпт обрезан до 4000 символов.")
            context.user_data["pending_images"] = []
            await process_and_reply(update, context, images, text)
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
            "Начало обработки для user_id=%s: %d изображений, model=%s, prompt=%r",
            user.id,
            len(images),
            model,
            prompt,
        )

        # Режим create: только текст → изображение (images.generate, gpt-image-1.5)
        if model == "create":
            model_label = MODEL_LABELS.get(model, model)
            message = await update.message.reply_text(
                f"Генерирую изображение ({model_label})…"
            )
            try:
                result_bytes, usage_str = await asyncio.to_thread(
                    processor.process_create,
                    prompt,
                    model="gpt-image-1.5",
                )
            except ValueError as e:
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
            return

        # Режим dalle_create: только текст → изображение (images.generate, DALL-E 2)
        if model == "dalle_create":
            model_label = MODEL_LABELS.get(model, model)
            message = await update.message.reply_text(
                f"Генерирую изображение ({model_label})…"
            )
            try:
                result_bytes, usage_str = await asyncio.to_thread(
                    processor.process_create,
                    prompt,
                    model="dall-e-2",
                )
            except ValueError as e:
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
            return

        # Текстовый режим (gpt-5.2): только текст ИЛИ 1 изображение + текст
        if model == "gpt-5.2":
            message = await update.message.reply_text("Обрабатываю…")
            try:
                if images:
                    if len(images) > 1:
                        images = images[:1]
                        logger.info("Текстовый режим: берём 1 изображение")
                    result_text = await asyncio.to_thread(
                        processor.process_text_with_image,
                        images[0],
                        prompt,
                        model=model,
                    )
                else:
                    result_text = await asyncio.to_thread(
                        processor.process_text_only,
                        prompt,
                        model=model,
                    )
            except (ValueError, IndexError) as e:
                await message.edit_text(str(e))
                return
            except Exception as e:
                logger.exception("Ошибка текстового режима для user_id=%s: %s", user.id, e)
                await message.edit_text(f"Ошибка: {e}")
                return

            await message.delete()
            parts = chunk_text(result_text)
            if not parts:
                await update.message.reply_text("(Пустой ответ)")
            else:
                for part in parts:
                    await update.message.reply_text(part)
            return

        # Режим генерации изображений
        model_label = MODEL_LABELS.get(model, model)
        message = await update.message.reply_text(
            f"Обрабатываю изображения ({model_label})…"
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

    print("ImageBot — обработка изображений через OpenAI\n")
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

    logger.info("CLI: обработка %d изображений с prompt=%r", len(paths), prompt)
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
        print("ImageBot")
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
