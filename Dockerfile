FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py processor.py ./

# По умолчанию без логов (для сервиса). Добавьте -v в command для отладки
CMD ["python", "bot.py", "telegram"]
