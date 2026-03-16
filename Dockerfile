FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py processor.py rag_store.py auth.py web_app.py ./
COPY templates/ ./templates/

# По умолчанию — Telegram-бот (для обратной совместимости)
CMD ["python", "bot.py", "telegram"]
