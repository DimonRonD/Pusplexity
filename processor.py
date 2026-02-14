"""
Ядро обработки изображений через OpenAI GPT Image API.
Принимает 1-10 изображений и текстовую команду, возвращает изображение в высоком качестве.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


def _format_usage(usage) -> str | None:
    """Форматирует usage из ответа API в строку для отображения."""
    if not usage or not hasattr(usage, "total_tokens"):
        return None
    total = getattr(usage, "total_tokens", 0) or 0
    if total <= 0:
        return None
    input_tok = getattr(usage, "input_tokens", None)
    output_tok = getattr(usage, "output_tokens", None)
    if input_tok is not None and output_tok is not None:
        return f"Токены: {total} (вход: {input_tok}, выход: {output_tok})"
    return f"Токены: {total}"


class ImageProcessor:
    """Обработчик изображений через OpenAI gpt-image-1.5."""

    MODEL = "gpt-image-1.5"
    QUALITY = "high"
    SIZE = "1536x1024"
    OUTPUT_FORMAT = "png"

    SYSTEM_PROMPT = (
        "Критически важно: используй ТОЛЬКО приложенные изображения. "
        "Не добавляй дополнительных людей, лиц или персонажей. "
        "Строго следуй инструкциям пользователя без отклонений.\n\n"
    )

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)

    def process(
        self,
        images: list[Path | BinaryIO | bytes],
        prompt: str,
        *,
        model: str | None = None,
        quality: str = QUALITY,
        size: str = SIZE,
        output_format: str = OUTPUT_FORMAT,
    ) -> tuple[bytes, str | None]:
        """
        Обрабатывает изображения по текстовой команде.

        Args:
            images: Список из 1-10 изображений (путь к файлу, файловый объект или bytes).
            prompt: Текстовая команда для обработки.
            quality: Качество выходного изображения (low, medium, high, auto).
            size: Размер (1024x1024, 1536x1024, 1024x1536, auto).
            output_format: Формат (png, webp, jpeg).

    Returns:
        Кортеж (байты изображения, строка с использованием токенов или None).
        """
        logger.info(
            "Начало обработки: %d изображений, prompt=%r",
            len(images),
            prompt,
        )

        model = model or self.MODEL

        if not 1 <= len(images) <= 10:
            logger.warning("Недопустимое количество изображений: %d", len(images))
            raise ValueError("Количество изображений должно быть от 1 до 10")

        if model == "dall-e-2" and len(images) > 1:
            logger.info("DALL-E 2 поддерживает только 1 изображение, берём первое")
            images = images[:1]
            size = "1024x1024"

        if not prompt or not prompt.strip():
            logger.warning("Пустая текстовая команда")
            raise ValueError("Текстовая команда не может быть пустой")

        logger.debug(
            "Параметры: quality=%s, size=%s, output_format=%s",
            quality,
            size,
            output_format,
        )

        image_files = self._prepare_images(images)
        logger.debug("Подготовлено %d файлов для отправки", len(image_files))

        start_time = time.perf_counter()
        try:
            # OpenAI SDK требует (filename, fileobj, mimetype) для bytes/Stream,
            # иначе ставит application/octet-stream
            api_images = [
                self._to_api_format(item, i, []) for i, item in enumerate(image_files)
            ]

            full_prompt = self.SYSTEM_PROMPT + prompt.strip()
            edit_kwargs = dict(
                model=model,
                image=api_images,
                prompt=full_prompt,
                quality=quality,
                size=size,
                output_format=output_format,
            )
            result = self.client.images.edit(**edit_kwargs)
        finally:
            for f in image_files:
                if hasattr(f, "close"):
                    try:
                        f.close()
                    except Exception:
                        pass

        elapsed = time.perf_counter() - start_time
        logger.info("OpenAI ответ получен за %.2f сек", elapsed)

        if not result.data or not result.data[0].b64_json:
            logger.error("OpenAI вернул пустой ответ: data=%r", result.data)
            raise RuntimeError("OpenAI не вернул изображение")

        decoded = base64.b64decode(result.data[0].b64_json)
        logger.info("Обработка завершена: результат %.1f КБ", len(decoded) / 1024)
        usage_str = _format_usage(result.usage)
        return decoded, usage_str

    def _to_api_format(self, fileobj, index: int, _temp_files: list = None):
        """
        Преобразует файл в формат (filename, fileobj, mimetype) для OpenAI API.
        bytes и BytesIO без расширения приводят к application/octet-stream,
        поэтому нормализуем через Pillow в PNG/JPEG с явным mimetype.
        """
        from PIL import Image

        if hasattr(fileobj, "seek"):
            fileobj.seek(0)
        data = fileobj.read() if hasattr(fileobj, "read") else fileobj

        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt = (im.format or "").upper()
                # OpenAI поддерживает: image/png, image/jpeg, image/webp
                if fmt == "PNG":
                    ext, save_fmt, mime = "png", "PNG", "image/png"
                elif fmt in ("JPEG", "JPG"):
                    ext, save_fmt, mime = "jpg", "JPEG", "image/jpeg"
                elif fmt == "WEBP":
                    ext, save_fmt, mime = "webp", "WEBP", "image/webp"
                else:
                    ext, save_fmt, mime = "png", "PNG", "image/png"

                if im.mode not in ("RGB", "RGBA", "L"):
                    im = im.convert("RGB")

                buf = io.BytesIO()
                if save_fmt == "JPEG":
                    if im.mode in ("RGBA", "LA", "P"):
                        im = im.convert("RGB")
                    im.save(buf, format="JPEG", quality=95)
                else:
                    im.save(buf, format=save_fmt)
                buf.seek(0)

                filename = f"image_{index}.{ext}"
                logger.debug("Изображение %d: %s, mimetype=%s", index + 1, filename, mime)
                return (filename, buf, mime)
        except Exception as e:
            logger.exception("Не удалось обработать изображение %d: %s", index + 1, e)
            raise ValueError(f"Неподдерживаемый или повреждённый файл изображения: {e}") from e

    def _prepare_images(
        self, images: list[Path | BinaryIO | bytes]
    ) -> list[BinaryIO | tuple]:
        """Подготавливает изображения для отправки в API."""
        prepared = []
        for i, img in enumerate(images):
            if isinstance(img, Path):
                suf = img.suffix.lower()
                if suf in (".png", ".jpg", ".jpeg", ".webp"):
                    prepared.append(open(img, "rb"))
                    logger.debug("Изображение %d: Path %s", i + 1, img)
                else:
                    # Расширение не из белого списка — нормализуем через Pillow
                    with open(img, "rb") as f:
                        data = f.read()
                    buf = io.BytesIO(data)
                    prepared.append(buf)
                    logger.debug("Изображение %d: Path %s (нормализация)", i + 1, img)
            elif isinstance(img, bytes):
                prepared.append(io.BytesIO(img))
                logger.debug("Изображение %d: bytes, размер %d", i + 1, len(img))
            elif hasattr(img, "read"):
                if hasattr(img, "seek"):
                    img.seek(0)
                prepared.append(img)
                logger.debug("Изображение %d: файловый объект", i + 1)
            else:
                logger.error("Неподдерживаемый тип изображения: %s", type(img))
                raise TypeError(
                    f"Неподдерживаемый тип изображения: {type(img)}. "
                    "Используйте Path, bytes или файловый объект."
                )
        return prepared


    def process_text_with_image(
        self,
        image: Path | BinaryIO | bytes,
        prompt: str,
        *,
        model: str = "gpt-5.2",
        history: list[dict] | None = None,
    ) -> str:
        """
        Текстовый режим: распознавание/анализ изображения, возврат текста.
        Поддерживает 1 изображение. history — контекст диалога (без изображений).
        """
        if hasattr(image, "seek"):
            image.seek(0)
            data = image.read()
        elif isinstance(image, Path):
            data = image.read_bytes()
        else:
            data = image

        b64 = base64.b64encode(data).decode("utf-8")
        mime = "image/png"  # или определить по magic bytes
        if data[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif data[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            mime = "image/webp"

        content = [
            {"type": "text", "text": prompt.strip()},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        logger.info("Текстовый режим: ответ %d символов", len(text))
        return text

    def process_text_only(
        self,
        prompt: str,
        *,
        model: str = "gpt-5.2",
        history: list[dict] | None = None,
    ) -> str:
        """
        Чат только по тексту, без изображений.
        history: список {"role": "user"|"assistant", "content": "..."} для контекста.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Сообщение не может быть пустым")
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt.strip()})
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        logger.info("Текстовый чат: ответ %d символов", len(text))
        return text

    def process_text_with_rag_context(
        self,
        prompt: str,
        context: str,
        *,
        model: str = "gpt-5.2",
        history: list[dict] | None = None,
    ) -> str:
        """
        Ответ на вопрос с контекстом из RAG.
        history: список {"role": "user"|"assistant", "content": "..."} для контекста диалога.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Вопрос не может быть пустым")
        system = (
            "Ты помощник с двумя источниками информации:\n"
            "1) Контекст из корпоративной базы знаний (документы) — для вопросов о содержимом.\n"
            "2) История диалога (предыдущие вопросы и ответы) — для вопросов о самом разговоре: "
            "«какие были предыдущие вопросы/команды», «что мы обсуждали», «повтори ответ» и т.п.\n\n"
            "Для вопросов о документах отвечай только на основе контекста. "
            "Для вопросов о диалоге используй историю. Если информации нет, честно скажи."
        )
        user_content = f"Контекст:\n\n{context}\n\n---\nВопрос: {prompt.strip()}"
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        logger.info("RAG чат: ответ %d символов", len(text))
        return text

    def process_create(
        self,
        prompt: str,
        *,
        model: str | None = None,
        quality: str = QUALITY,
        size: str = SIZE,
        output_format: str = OUTPUT_FORMAT,
    ) -> tuple[bytes, str | None]:
        """
        Генерирует изображение по текстовому описанию (text-to-image).
        Не требует входных изображений. Использует images.generate.

        Args:
            prompt: Текстовое описание желаемого изображения.
            model: Модель (gpt-image-1.5, dall-e-2, dall-e-3 и т.д.).
            quality: Качество (low, medium, high, auto).
            size: Размер (1024x1024, 1536x1024, 1024x1536, auto).
            output_format: Формат (png, webp, jpeg).

        Returns:
            Кортеж (байты изображения, строка с использованием токенов или None).
        """
        model = model or self.MODEL

        if not prompt or not prompt.strip():
            raise ValueError("Текстовое описание не может быть пустым")

        # DALL-E 2: только 1024x1024, quality=standard
        if model == "dall-e-2":
            size = "1024x1024"
            quality = "standard"

        logger.info("Генерация изображения по тексту: model=%s, prompt=%r", model, prompt[:100])

        start_time = time.perf_counter()
        if model.startswith("dall-e"):
            kwargs = dict(
                prompt=prompt.strip(),
                model=model,
                size=size,
                n=1,
                response_format="b64_json",
            )
        else:
            kwargs = dict(
                prompt=prompt.strip(),
                model=model,
                quality=quality,
                size=size,
                output_format=output_format,
                n=1,
            )

        result = self.client.images.generate(**kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info("OpenAI generate ответ за %.2f сек", elapsed)

        if not result.data or not result.data[0].b64_json:
            logger.error("OpenAI вернул пустой ответ: data=%r", result.data)
            raise RuntimeError("OpenAI не вернул изображение")

        decoded = base64.b64decode(result.data[0].b64_json)
        logger.info("Генерация завершена: результат %.1f КБ", len(decoded) / 1024)
        usage_str = _format_usage(result.usage)
        return decoded, usage_str


def process_images(
    images: list[Path | BinaryIO | bytes],
    prompt: str,
    api_key: str | None = None,
    **kwargs,
) -> bytes:
    """
    Удобная функция для обработки изображений.

    Args:
        images: 1-10 изображений.
        prompt: Текстовая команда.
        api_key: Ключ OpenAI (если не задан — берётся из OPENAI_API_KEY).
        **kwargs: Доп. параметры (quality, size, output_format).

    Returns:
        Байты результирующего изображения.
    """
    processor = ImageProcessor(api_key=api_key)
    result_bytes, _ = processor.process(images, prompt, **kwargs)
    return result_bytes
