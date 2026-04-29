#!/usr/bin/env python3
"""
Pusplexity — веб-интерфейс на Flask.
Полностью повторяет функционал Telegram-бота.
Вход по email и паролю (users.txt).
"""

import base64
import logging
import os
import secrets
import tempfile
import time
import uuid
from pathlib import Path

import user_db

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    Response,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from auth import verify_credentials
from dotenv import load_dotenv
from processor import ImageProcessor
from rag_store import DATA_DIR, RAGStore, load_document

load_dotenv()

app = Flask(__name__)
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be set")
app.secret_key = _secret_key
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

logger = logging.getLogger(__name__)


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_e):
    """Для API всегда возвращаем JSON, чтобы фронтенд не падал на HTML-ошибке."""
    if request.path.startswith("/api/"):
        return _api_error("Файл слишком большой. Максимум 50 МБ.", 413)
    return "Файл слишком большой. Максимум 50 МБ.", 413

# Кэш в памяти (pending_images, rag_add_mode — сессионные)
_user_cache: dict[str, dict] = {}

DEFAULT_MODEL = "gpt-image-1.5"
LEGACY_TEXT_MODEL = "gpt-5.2"
DEFAULT_TEXT_MODEL = "gpt-5.5"


def _resolve_text_model(raw: str | None) -> str:
    model = (raw or "").strip()
    if not model or model == "latest":
        return DEFAULT_TEXT_MODEL
    return model


TEXT_MODEL = _resolve_text_model(os.environ.get("OPENAI_TEXT_MODEL"))
MODELS = {
    TEXT_MODEL: TEXT_MODEL,
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
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
CSRF_SESSION_KEY = "_csrf_token"
_LOGIN_ATTEMPTS_WINDOW_SEC = 10 * 60
_LOGIN_ATTEMPTS_LIMIT = 8
_login_attempts: dict[str, list[float]] = {}


def _api_error(message: str, status: int = 400) -> tuple[dict, int]:
    return {"ok": False, "error": message}, status


def _api_internal_error(log_message: str, exc: Exception) -> tuple[dict, int]:
    request_id = uuid.uuid4().hex[:10]
    logger.exception("%s (request_id=%s): %s", log_message, request_id, exc)
    return _api_error(f"Внутренняя ошибка (id={request_id})", 500)


def _guess_doc_type(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        return "zip-office"
    if data.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "ole-office"
    try:
        data[:2048].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return None


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith((b"GIF87a", b"GIF89a")):
        return True
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return True
    return False


def _validate_uploaded_document(filename: str, data: bytes) -> bool:
    ext = Path(filename).suffix.lower()
    doc_type = _guess_doc_type(data)
    if ext in (".txt", ".md", ".text"):
        return doc_type == "text"
    if ext == ".pdf":
        return doc_type == "pdf"
    if ext in (".xlsx", ".docx"):
        return doc_type == "zip-office"
    if ext == ".xls":
        return doc_type == "ole-office"
    return False


def _issue_csrf_token() -> str:
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


@app.context_processor
def inject_template_globals():
    return {"csrf_token": _issue_csrf_token}


@app.before_request
def protect_post_requests():
    if request.method != "POST":
        return None
    token = session.get(CSRF_SESSION_KEY)
    provided = (
        request.headers.get("X-CSRF-Token")
        or request.form.get("csrf_token")
        or ""
    )
    if not token or not secrets.compare_digest(str(token), str(provided)):
        if request.path.startswith("/api/"):
            return _api_error("Неверный CSRF-токен", 403)
        return render_template("login.html", error="Сессия обновилась. Повторите вход."), 403
    return None


def _is_login_rate_limited(identity: str) -> bool:
    now = time.time()
    attempts = [t for t in _login_attempts.get(identity, []) if now - t <= _LOGIN_ATTEMPTS_WINDOW_SEC]
    _login_attempts[identity] = attempts
    return len(attempts) >= _LOGIN_ATTEMPTS_LIMIT


def _register_login_attempt(identity: str) -> None:
    now = time.time()
    attempts = [t for t in _login_attempts.get(identity, []) if now - t <= _LOGIN_ATTEMPTS_WINDOW_SEC]
    attempts.append(now)
    _login_attempts[identity] = attempts


def _get_user_key() -> str | None:
    """Email пользователя или None."""
    return session.get("email")


def _get_user_data() -> dict:
    """Данные пользователя: из SQLite + кэш (pending_images, rag_add_mode)."""
    key = _get_user_key()
    if not key:
        key = str(uuid.uuid4())
    if key not in _user_cache:
        ud = user_db.get_user_data(key) if "@" in key else {
            "model": TEXT_MODEL,
            "pending_images": [],
            "text_chat_history": [],
            "rag_chat_history": [],
            "text_context": None,
            "text_context_filename": None,
            "rag_add_mode": False,
        }
        _user_cache[key] = ud
    return _user_cache[key]


def _save_user_data() -> None:
    """Сохраняет данные в SQLite (для авторизованных по email)."""
    key = _get_user_key()
    if key and "@" in key and key in _user_cache:
        user_db.save_user_data(key, _user_cache[key])


def _set_model(model: str) -> None:
    _get_user_data()["model"] = model
    _save_user_data()


def _get_model() -> str:
    ud = _get_user_data()
    model = ud.get("model", DEFAULT_MODEL)
    if model == LEGACY_TEXT_MODEL:
        # Мягкая миграция старого значения из SQLite/кэша.
        ud["model"] = TEXT_MODEL
        _save_user_data()
        return TEXT_MODEL
    return model


def _update_chat_history(key: str, user_msg: str, assistant_msg: str) -> None:
    ud = _get_user_data()
    history = list(ud.get(key, []))
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    ud[key] = history[-CHAT_HISTORY_SIZE:]
    _save_user_data()


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


@app.route("/health")
def health():
    """Проверка доступности (для диагностики)."""
    return "OK", 200


@app.route("/")
def index():
    # Редирект без session — /login сам перенаправит в /chat при авторизации
    return Response(status=302, headers={"Location": "/login"})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "email" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    identity = f"{request.remote_addr}:{email.lower()}"
    if _is_login_rate_limited(identity):
        return render_template("login.html", error="Слишком много попыток входа. Повторите позже.")
    if not email or not password:
        return render_template("login.html", error="Введите email и пароль")
    if not verify_credentials(email, password):
        _register_login_attempt(identity)
        return render_template("login.html", error="Неверный email или пароль")
    session["email"] = email
    _login_attempts.pop(identity, None)
    return redirect(url_for("chat"))


@app.route("/logout")
def logout():
    # Не удаляем _user_data — память сохраняется для аккаунта
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat")
def chat():
    if "email" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html")


def _require_auth():
    if "email" not in session:
        return _api_error("Требуется авторизация", 401)


# API-эндпоинты для команд (AJAX)


@app.route("/api/command", methods=["POST"])
def api_command():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    cmd = (request.form.get("command") or "").strip().lower()
    ud = _get_user_data()
    processor = ImageProcessor()

    # Команды переключения режима
    if cmd == "start":
        _set_model(TEXT_MODEL)
        return {"ok": True, "message": (
            f"🖼 Pusplexity\n\nРежим по умолчанию: {TEXT_MODEL} (чат)\n\n"
            "◾ /text — чат, анализ 1 фото, контекст из документов\n"
            "◾ /image1, /image15, /dalle — редактирование фото\n"
            "◾ /create, /dalle_gen — генерация по тексту\n"
            "◾ /rag_add, /rag_index, /rag_text — RAG: база знаний\n\n"
            "/help — справка"
        )}
    if cmd == "text":
        _set_model(TEXT_MODEL)
        return {"ok": True, "message": (
            f"✅ Текстовый режим ({TEXT_MODEL})\n\n"
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
        _save_user_data()
        return {"ok": True, "message": "✅ История /text очищена."}
    if cmd == "help":
        return {"ok": True, "message": (
            "📖 Pusplexity — Справка по кнопкам\n\n"
            "◾ Режимы работы\n"
            f"Старт — Начало работы ({TEXT_MODEL} по умолчанию).\n"
            f"Текст — Чат {TEXT_MODEL}: текст, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD. Память 20 сообщений.\n"
            "Image1 — gpt-image-1: редактирование 1–10 фото.\n"
            "Image15 — gpt-image-1.5: редактирование 1–10 фото.\n"
            "DALL-E — DALL-E 2: редактирование 1 фото.\n"
            "Create — Генерация изображения по тексту (gpt-image-1.5).\n"
            "DALL-E Gen — Генерация по тексту (DALL-E 2, до 1000 символов).\n\n"
            "◾ RAG — база знаний\n"
            "RAG Add — Включить режим загрузки. Загрузите TXT, PDF, XLSX, DOCX, MD.\n"
            "RAG Index — Индексировать файлы из data/ в ChromaDB.\n"
            "RAG List — Список источников в хранилище.\n"
            "RAG Delete — Удалить источник и его данные из ChromaDB (укажите имя).\n"
            "RAG — Режим RAG. Задавайте вопросы, ответы по документам. До смены режима.\n"
            "RAG Clear — Очистить историю сеанса RAG.\n\n"
            "Clear — Очистить историю и контекст документа в режиме Текст.\n"
            "Help — Эта справка."
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
            return _api_error(str(e))
        except Exception as e:
            return _api_internal_error("Ошибка индексации RAG", e)
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
            return _api_error(str(e))
        except Exception as e:
            return _api_internal_error("Ошибка list_sources", e)
        if not sources:
            return {"ok": True, "message": "Хранилище RAG пусто. Используйте /rag_add и /rag_index."}
        return {"ok": True, "message": "📚 Источники:\n\n" + "\n".join(f"• {s}" for s in sources)}
    if cmd == "rag_text":
        _set_model("rag_text")
        ud["rag_add_mode"] = False
        _save_user_data()
        return {"ok": True, "message": (
            "✅ Режим RAG включён.\n\n"
            "Задавайте вопросы — ответы будут сформированы на основе документов из хранилища.\n"
            "Для смены режима выберите другой режим в панели."
        )}
    if cmd == "rag_clear":
        ud["rag_chat_history"] = []
        _save_user_data()
        return {"ok": True, "message": "✅ История /rag_text очищена."}

    return _api_error(f"Неизвестная команда: {cmd}")


@app.route("/api/rag_delete", methods=["POST"])
def api_rag_delete():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    source = (request.form.get("source") or "").strip()
    if not source:
        return _api_error("Укажите источник: /rag_delete <имя>")
    try:
        store = _get_rag_store()
        count = store.delete_source(source, DATA_DIR)
    except ValueError as e:
        return _api_error(str(e))
    except Exception as e:
        return _api_internal_error("Ошибка удаления источника RAG", e)
    return {"ok": True, "message": f"✅ Источник «{source}» удалён ({count} чанков)."}


@app.route("/api/send", methods=["POST"])
def api_send():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
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
                raw = f.read()
                if ext in ALLOWED_IMAGE_EXTENSIONS and _looks_like_image(raw):
                    images.append(raw)
    if len(images) > 10:
        images = images[-10:]

    # RAG режим
    if model == "rag_text":
        if not text:
            return _api_error("Введите вопрос для RAG.")
        try:
            store = _get_rag_store()
            results = store.query(text, 5)
        except Exception as e:
            return _api_internal_error("Ошибка RAG query", e)
        if not results:
            return _api_error("Хранилище RAG пусто. Используйте /rag_add и /rag_index.")
        context_parts = [f"[{src}]\n{doc}" for doc, src, _ in results]
        rag_context = "\n\n---\n\n".join(context_parts)
        rag_history = list(ud.get("rag_chat_history", []))
        try:
            result_text = processor.process_text_with_rag_context(
                text, rag_context, model=TEXT_MODEL, history=rag_history or None
            )
        except Exception as e:
            return _api_internal_error("Ошибка OpenAI для /rag_text", e)
        _update_chat_history("rag_chat_history", text, result_text)
        source_scores = {}
        for doc, src, dist in results:
            if src:
                score = round(1 / (1 + dist), 3)
                if src not in source_scores or score > source_scores[src]:
                    source_scores[src] = score
        sources_line = ", ".join(f"{s} ({d})" for s, d in sorted(source_scores.items()))
        return {
            "ok": True,
            "type": "text",
            "message": result_text,
            "sources": sources_line,
            "model": TEXT_MODEL,
        }

    # Create / dalle_create
    if model in ("create", "dalle_create"):
        if not text:
            return _api_error("Введите текстовое описание изображения.")
        if model == "dalle_create" and len(text) > 1000:
            text = text[:1000] + "\n\n[... обрезано]"
        elif len(text) > 4000:
            text = text[:4000] + "\n\n[... обрезано]"
        try:
            result_bytes, usage_str = processor.process_create(
                text, model="dall-e-2" if model == "dalle_create" else "gpt-image-1.5"
            )
        except Exception as e:
            return _api_error(_format_image_error(e))
        b64 = base64.b64encode(result_bytes).decode("utf-8")
        return {
            "ok": True,
            "type": "image",
            "image_b64": b64,
            "usage": usage_str,
            "model": "dall-e-2" if model == "dalle_create" else "gpt-image-1.5",
        }

    # latest текстовый режим
    if model == TEXT_MODEL:
        if not text:
            return _api_error("Введите сообщение или отправьте фото с подписью.")
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
            return _api_internal_error("Ошибка текстового режима", e)
        user_msg = f"[Изображение] {text}" if images else text
        _update_chat_history("text_chat_history", user_msg, result_text)
        ud["pending_images"] = []
        return {"ok": True, "type": "text", "message": result_text, "model": model}

    # Режимы редактирования изображений
    if model in ("gpt-image-1", "gpt-image-1.5", "dall-e-2") and images and not text:
        ud["pending_images"] = images
        return {"ok": True, "type": "text", "message": (
            f"Получено {len(images)} изображений. Модель: {MODEL_LABELS.get(model, model)}. "
            "Введите текстовую команду."
        )}
    if not images:
        return _api_error("Сначала загрузите изображение, затем введите команду.")
    if len(text) > 4000:
        text = text[:4000] + "\n\n[... обрезано]"
    try:
        result_bytes, usage_str = processor.process(images, text, model=model)
    except Exception as e:
        return _api_error(_format_image_error(e))
    ud["pending_images"] = []
    b64 = base64.b64encode(result_bytes).decode("utf-8")
    return {"ok": True, "type": "image", "image_b64": b64, "usage": usage_str, "model": model}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    ud = _get_user_data()
    model = _get_model()

    # Документ для RAG add
    if ud.get("rag_add_mode"):
        f = request.files.get("file")
        if not f or not f.filename:
            return _api_error("Выберите файл")
        ext = Path(f.filename).suffix.lower()
        if ext not in RAG_ALLOWED_EXTENSIONS:
            return _api_error(f"Формат {ext} не поддерживается. TXT, PDF, XLSX, DOCX.")
        raw = f.read()
        if not _validate_uploaded_document(f.filename, raw):
            return _api_error("Файл не прошёл проверку типа содержимого.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_DIR / secure_filename(f.filename or f"upload-{uuid.uuid4().hex}")
        dest.write_bytes(raw)
        return {"ok": True, "message": f"✅ Сохранён: {f.filename}"}

    # Режим /text (latest): изображение для анализа ИЛИ документ для контекста
    if model == TEXT_MODEL:
        f = request.files.get("file")
        if not f or not f.filename:
            return _api_error("Выберите файл")
        ext = Path(f.filename).suffix.lower()
        raw = f.read()
        is_image = ext in ALLOWED_IMAGE_EXTENSIONS and _looks_like_image(raw)
        if is_image:
            ud["pending_images"] = [raw]
            return {"ok": True, "message": f"✅ Изображение «{f.filename}» загружено. Введите вопрос или описание для анализа."}
        if ext not in TEXT_CONTEXT_EXTENSIONS:
            return _api_error(f"Формат {ext} не поддерживается. Изображения: PNG, JPG, JPEG, WEBP, GIF. Документы: TXT, PDF, XLSX, DOCX, MD.")
        if not _validate_uploaded_document(f.filename, raw):
            return _api_error("Файл не прошёл проверку типа содержимого.")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / secure_filename(f.filename)
                tmp_path.write_bytes(raw)
                content = load_document(tmp_path)
            if not content or not content.strip():
                return _api_error("Не удалось извлечь текст.")
            ud["text_context"] = content.strip()
            ud["text_context_filename"] = f.filename
            _save_user_data()
        except Exception as e:
            return _api_internal_error("Ошибка загрузки документа для контекста", e)
        return {"ok": True, "message": f"✅ Документ «{f.filename}» загружен как контекст."}

    # Изображения для редактирования
    files = request.files.getlist("file") or request.files.getlist("images[]") or []
    if not files and "file" in request.files:
        files = [request.files["file"]]
    images = list(ud.get("pending_images", []))
    for f in files:
        if f and f.filename:
            ext = Path(f.filename).suffix.lower()
            raw = f.read()
            if ext in ALLOWED_IMAGE_EXTENSIONS and _looks_like_image(raw):
                images.append(raw)
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
