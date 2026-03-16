#!/usr/bin/env python3
"""
ImageBot — веб-интерфейс на Flask.
Полностью повторяет функционал Telegram-бота.
Вход по email и паролю (users.txt).
"""

import base64
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from auth import verify_credentials
from dotenv import load_dotenv
from processor import ImageProcessor
from rag_store import DATA_DIR, RAGStore, load_document

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

logger = logging.getLogger(__name__)

# Состояние пользователей: sid -> {model, pending_images, text_chat_history, ...}
_user_data: dict[str, dict] = {}

DEFAULT_MODEL = "gpt-image-1.5"
MODELS = {
    "gpt-5.2": "gpt-5.2",
    "gpt-image-1": "gpt-image-1",
    "gpt-image-1.5": "gpt-image-1.5",
    "dall-e-2": "dall-e-2",
    "create": "create",
    "dalle_create": "dalle_create",
    "rag_text": "rag_text",
}
MODEL_LABELS = {
    "gpt-image-1": "gpt-image-1",
    "gpt-image-1.5": "gpt-image-1.5",
    "dall-e-2": "DALL-E 2",
    "create": "gpt-image-1.5 (create)",
    "dalle_create": "DALL-E 2 (create)",
    "rag_text": "RAG",
}
RAG_ALLOWED_EXTENSIONS = (".txt", ".pdf", ".xlsx", ".xls", ".docx", ".md", ".text")
TEXT_CONTEXT_EXTENSIONS = (".txt", ".pdf", ".xlsx", ".xls", ".docx", ".md", ".text")
CHAT_HISTORY_SIZE = 20


def _get_sid() -> str:
    if "_sid" not in session:
        session["_sid"] = str(uuid.uuid4())
    return session["_sid"]


def _get_user_data() -> dict:
    sid = _get_sid()
    if sid not in _user_data:
        _user_data[sid] = {
            "model": "gpt-5.2",
            "pending_images": [],
            "text_chat_history": [],
            "rag_chat_history": [],
            "text_context": None,
            "text_context_filename": None,
            "rag_add_mode": False,
        }
    return _user_data[sid]


def _set_model(model: str) -> None:
    _get_user_data()["model"] = model


def _get_model() -> str:
    return _get_user_data().get("model", DEFAULT_MODEL)


def _update_chat_history(key: str, user_msg: str, assistant_msg: str) -> None:
    ud = _get_user_data()
    history = list(ud.get(key, []))
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    ud[key] = history[-CHAT_HISTORY_SIZE:]


def _get_rag_store() -> RAGStore:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан")
    return RAGStore(api_key=api_key)


def _format_image_error(exc: Exception) -> str:
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
            "⚠️ Запрос отклонён системой безопасности OpenAI. "
            "Попробуйте переформулировать описание."
        )
    return str(exc)


def _chunk_text(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text] if text else []
    chunks = []
    while text:
        chunk = text[:max_len]
        last_nl = chunk.rfind("\n")
        if last_nl > max_len // 2:
            chunk = text[: last_nl + 1]
        chunks.append(chunk)
        text = text[len(chunk) :]
    return chunks


# --- Маршруты ---


@app.route("/")
def index():
    if "email" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("chat"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "email" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    if not email or not password:
        return render_template("login.html", error="Введите email и пароль")
    if not verify_credentials(email, password):
        return render_template("login.html", error="Неверный email или пароль")
    session["email"] = email
    return redirect(url_for("chat"))


@app.route("/logout")
def logout():
    sid = session.get("_sid")
    if sid and sid in _user_data:
        del _user_data[sid]
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat")
def chat():
    if "email" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html")


def _require_auth():
    if "email" not in session:
        return {"ok": False, "error": "Требуется авторизация"}, 401


# API-эндпоинты для команд (AJAX)


@app.route("/api/command", methods=["POST"])
def api_command():
    if "email" not in session:
        return {"ok": False, "error": "Требуется авторизация"}, 401
    cmd = (request.form.get("command") or "").strip().lower()
    ud = _get_user_data()
    processor = ImageProcessor()

    # Команды переключения режима
    if cmd == "start":
        _set_model("gpt-5.2")
        return {"ok": True, "message": (
            "🖼 ImageBot\n\nРежим по умолчанию: gpt-5.2 (чат)\n\n"
            "◾ /text — чат, анализ 1 фото, контекст из документов\n"
            "◾ /image1, /image15, /dalle — редактирование фото\n"
            "◾ /create, /dalle_gen — генерация по тексту\n"
            "◾ /rag_add, /rag_index, /rag_text — RAG: база знаний\n\n"
            "/help — справка"
        )}
    if cmd == "text":
        _set_model("gpt-5.2")
        return {"ok": True, "message": (
            "✅ Текстовый режим (gpt-5.2)\n\n"
            "• Чат, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD"
        )}
    if cmd == "image1":
        _set_model("gpt-image-1")
        return {"ok": True, "message": "✅ Модель: gpt-image-1. Можно загружать 1–10 фото."}
    if cmd == "image15":
        _set_model("gpt-image-1.5")
        return {"ok": True, "message": "✅ Модель: gpt-image-1.5. Можно загружать 1–10 фото."}
    if cmd == "dalle":
        _set_model("dall-e-2")
        return {"ok": True, "message": "✅ Модель: DALL-E 2. Поддерживает только 1 изображение."}
    if cmd == "create":
        _set_model("create")
        return {"ok": True, "message": "✅ Режим Create. Отправьте текстовое описание — получите изображение."}
    if cmd == "dalle_gen":
        _set_model("dalle_create")
        return {"ok": True, "message": "✅ Режим DALL-E 2 Gen. Отправьте текстовое описание."}
    if cmd == "clear":
        ud["text_chat_history"] = []
        ud.pop("text_context", None)
        ud.pop("text_context_filename", None)
        return {"ok": True, "message": "✅ История /text очищена."}
    if cmd == "help":
        return {"ok": True, "message": (
            "📖 ImageBot — Справка\n\n"
            "/text — чат gpt-5.2\n"
            "/image1, /image15, /dalle — редактирование фото\n"
            "/create, /dalle_gen — генерация по тексту\n"
            "/rag_add, /rag_index, /rag_list, /rag_delete, /rag_text, /rag_clear\n"
            "/clear — очистить историю /text"
        )}

    # RAG
    if cmd == "rag_add":
        ud["rag_add_mode"] = True
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "message": (
            "📂 Режим загрузки RAG включён. Загрузите документы (TXT, PDF, XLSX, DOCX)."
        )}
    if cmd == "rag_index":
        ud["rag_add_mode"] = False
        try:
            store = _get_rag_store()
            counts = store.index_documents(DATA_DIR)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception("Ошибка индексации RAG: %s", e)
            return {"ok": False, "error": str(e)}
        if not counts:
            return {"ok": True, "message": "Нет документов для индексации. Используйте /rag_add."}
        total = sum(counts.values())
        lines = [f"✅ Проиндексировано {total} чанков из {len(counts)} файлов:\n"]
        for src, cnt in sorted(counts.items()):
            lines.append(f"  • {src}: {cnt} чанков")
        return {"ok": True, "message": "\n".join(lines)}
    if cmd == "rag_list":
        try:
            store = _get_rag_store()
            sources = store.list_sources()
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if not sources:
            return {"ok": True, "message": "Хранилище RAG пусто. Используйте /rag_add и /rag_index."}
        return {"ok": True, "message": "📚 Источники:\n\n" + "\n".join(f"• {s}" for s in sources)}
    if cmd == "rag_clear":
        ud["rag_chat_history"] = []
        return {"ok": True, "message": "✅ История /rag_text очищена."}

    return {"ok": False, "error": f"Неизвестная команда: {cmd}"}


@app.route("/api/rag_delete", methods=["POST"])
def api_rag_delete():
    if "email" not in session:
        return {"ok": False, "error": "Требуется авторизация"}, 401
    source = (request.form.get("source") or "").strip()
    if not source:
        return {"ok": False, "error": "Укажите источник: /rag_delete <имя>"}
    try:
        store = _get_rag_store()
        count = store.delete_source(source, DATA_DIR)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "message": f"✅ Источник «{source}» удалён ({count} чанков)."}


@app.route("/api/send", methods=["POST"])
def api_send():
    if "email" not in session:
        return {"ok": False, "error": "Требуется авторизация"}, 401
    text = (request.form.get("text") or "").strip()
    ud = _get_user_data()
    model = _get_model()
    processor = ImageProcessor()

    # Загрузка изображений
    images: list[bytes] = list(ud.get("pending_images", []))
    if "images[]" in request.files or "image" in request.files:
        files = request.files.getlist("images[]") or request.files.getlist("image") or []
        if not files and "image" in request.files:
            files = [request.files["image"]]
        for f in files:
            if f and f.filename:
                ext = Path(f.filename).suffix.lower()
                if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif") or (f.content_type or "").startswith("image/"):
                    images.append(f.read())
    if len(images) > 10:
        images = images[-10:]

    # RAG режим
    if model == "rag_text":
        if not text:
            return {"ok": False, "error": "Введите вопрос для RAG."}
        if len(text) > 2000:
            text = text[:2000] + "\n\n[... обрезано]"
        try:
            store = _get_rag_store()
            results = store.query(text, 5)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if not results:
            return {"ok": False, "error": "Хранилище RAG пусто. Используйте /rag_add и /rag_index."}
        context_parts = [f"[{src}]\n{doc}" for doc, src, _ in results]
        rag_context = "\n\n---\n\n".join(context_parts)
        rag_history = list(ud.get("rag_chat_history", []))
        try:
            result_text = processor.process_text_with_rag_context(
                text, rag_context, model="gpt-5.2", history=rag_history or None
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        _update_chat_history("rag_chat_history", text, result_text)
        source_scores = {}
        for doc, src, dist in results:
            if src:
                score = round(1 / (1 + dist), 3)
                if src not in source_scores or score > source_scores[src]:
                    source_scores[src] = score
        sources_line = ", ".join(f"{s} ({d})" for s, d in sorted(source_scores.items()))
        return {"ok": True, "type": "text", "message": result_text, "sources": sources_line}

    # Create / dalle_create
    if model in ("create", "dalle_create"):
        if not text:
            return {"ok": False, "error": "Введите текстовое описание изображения."}
        if model == "dalle_create" and len(text) > 1000:
            text = text[:1000] + "\n\n[... обрезано]"
        elif len(text) > 4000:
            text = text[:4000] + "\n\n[... обрезано]"
        try:
            result_bytes, usage_str = processor.process_create(
                text, model="dall-e-2" if model == "dalle_create" else "gpt-image-1.5"
            )
        except Exception as e:
            return {"ok": False, "error": _format_image_error(e)}
        b64 = base64.b64encode(result_bytes).decode("utf-8")
        return {"ok": True, "type": "image", "image_b64": b64, "usage": usage_str}

    # gpt-5.2 текстовый режим
    if model == "gpt-5.2":
        if not text:
            return {"ok": False, "error": "Введите сообщение или отправьте фото с подписью."}
        if len(text) > 4000:
            text = text[:4000] + "\n\n[... обрезано]"
        text_history = list(ud.get("text_chat_history", []))
        text_context = ud.get("text_context")
        try:
            if images:
                images = images[:1]
                result_text = processor.process_text_with_image(
                    images[0], text, model=model, history=text_history or None
                )
            elif text_context:
                result_text = processor.process_text_with_rag_context(
                    text, text_context, model=model, history=text_history or None
                )
            else:
                result_text = processor.process_text_only(
                    text, model=model, history=text_history or None
                )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        user_msg = f"[Изображение] {text}" if images else text
        _update_chat_history("text_chat_history", user_msg, result_text)
        ud["pending_images"] = []
        return {"ok": True, "type": "text", "message": result_text}

    # Режимы редактирования изображений
    if model in ("gpt-image-1", "gpt-image-1.5", "dall-e-2") and images and not text:
        ud["pending_images"] = images
        return {"ok": True, "type": "text", "message": (
            f"Получено {len(images)} изображений. Модель: {MODEL_LABELS.get(model, model)}. "
            "Введите текстовую команду."
        )}
    if not images:
        return {"ok": False, "error": "Сначала загрузите изображение, затем введите команду."}
    if len(text) > 4000:
        text = text[:4000] + "\n\n[... обрезано]"
    try:
        result_bytes, usage_str = processor.process(images, text, model=model)
    except Exception as e:
        return {"ok": False, "error": _format_image_error(e)}
    ud["pending_images"] = []
    b64 = base64.b64encode(result_bytes).decode("utf-8")
    return {"ok": True, "type": "image", "image_b64": b64, "usage": usage_str}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "email" not in session:
        return {"ok": False, "error": "Требуется авторизация"}, 401
    ud = _get_user_data()
    model = _get_model()

    # Документ для RAG add
    if ud.get("rag_add_mode"):
        f = request.files.get("file")
        if not f or not f.filename:
            return {"ok": False, "error": "Выберите файл"}
        ext = Path(f.filename).suffix.lower()
        if ext not in RAG_ALLOWED_EXTENSIONS:
            return {"ok": False, "error": f"Формат {ext} не поддерживается. TXT, PDF, XLSX, DOCX."}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_DIR / secure_filename(f.filename)
        f.save(dest)
        return {"ok": True, "message": f"✅ Сохранён: {f.filename}"}

    # Документ для контекста /text
    if model == "gpt-5.2":
        f = request.files.get("file")
        if not f or not f.filename:
            return {"ok": False, "error": "Выберите файл"}
        ext = Path(f.filename).suffix.lower()
        if ext not in TEXT_CONTEXT_EXTENSIONS:
            return {"ok": False, "error": f"Формат {ext} не поддерживается."}
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                f.save(tmp.name)
                content = load_document(Path(tmp.name))
            if not content or not content.strip():
                return {"ok": False, "error": "Не удалось извлечь текст."}
            ud["text_context"] = content.strip()
            ud["text_context_filename"] = f.filename
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "message": f"✅ Документ «{f.filename}» загружен как контекст."}

    # Изображения для редактирования
    files = request.files.getlist("file") or request.files.getlist("images[]") or []
    if not files and "file" in request.files:
        files = [request.files["file"]]
    images = list(ud.get("pending_images", []))
    for f in files:
        if f and f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif") or (f.content_type or "").startswith("image/"):
                images.append(f.read())
    if len(images) > 10:
        images = images[-10:]
    ud["pending_images"] = images
    return {"ok": True, "message": f"Получено {len(images)} изображений. Модель: {MODEL_LABELS.get(model, model)}. Введите команду."}


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Задайте OPENAI_API_KEY в .env")
        return
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
