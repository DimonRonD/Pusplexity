FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py processor.py rag_store.py auth.py web_app.py user_db.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# По умолчанию — Telegram-бот (для обратной совместимости)
CMD ["python", "bot.py", "telegram"]
