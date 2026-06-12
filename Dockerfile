# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
# Кэш pip ускоряет повторные сборки (DOCKER_BUILDKIT=1).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY bot.py processor.py rag_store.py auth.py web_app.py user_db.py action_logs.py telegram_format.py ./
COPY templates/ ./templates/
COPY static/ ./static/

RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# По умолчанию — Telegram-бот (для обратной совместимости)
CMD ["python", "bot.py", "telegram"]
