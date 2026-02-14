# ImageBot

Чат-бот для обработки изображений через OpenAI GPT Image API. Принимает 1–10 изображений и текстовую команду, возвращает изображение в высоком качестве.

## Возможности

- **1–10 изображений на вход** — объединение, коллажи, стилизация
- **Текстовые команды** — «сделай в стиле акварели», «объедини в коллаж», «убери фон» и т.п.
- **Высокое качество** — модель `gpt-image-1`, качество `high`, размер до 1536×1024, PNG
- **Режимы**: Telegram-бот или интерактивный CLI

## Требования

- Python 3.10+
- OpenAI API-ключ с доступом к GPT Image (gpt-image-1)
- Telegram Bot Token (для режима Telegram)

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

1. Скопируйте `.env.example` в `.env`:
   ```bash
   cp .env.example .env
   ```

2. Заполните переменные:
   - `OPENAI_API_KEY` — ключ из [OpenAI Platform](https://platform.openai.com/api-keys)
   - `TELEGRAM_BOT_TOKEN` — токен от [@BotFather](https://t.me/BotFather)

## Запуск

### Docker Compose (рекомендуется для сервера)

```bash
# Создайте .env с OPENAI_API_KEY и TELEGRAM_BOT_TOKEN
cp .env.example .env

docker compose up -d
```

Для отладки (с выводом логов в консоль):

```bash
docker compose run --rm imagebot python bot.py telegram -v
```

### Telegram-бот (локально)

```bash
python bot.py telegram
```

Отправьте боту изображения (с подписью-командой или без), затем текстовую команду.

### CLI

```bash
python bot.py cli
```

Введите пути к файлам и текст команды по запросу. Результат сохраняется в `output.png`.

## Примеры команд

- «Сделай это фото в стиле импрессионизма»
- «Объедини все изображения в один коллаж»
- «Преобразуй в пиксель-арт»
- «Убери фон и оставь прозрачным»
- «Добавь закат на задний план»

## API

Можно использовать `processor` напрямую:

```python
from pathlib import Path
from processor import process_images

result_bytes = process_images(
    images=[Path("photo1.jpg"), Path("photo2.png")],
    prompt="Объедини в коллаж",
    quality="high",
    size="1536x1024",
)
with open("result.png", "wb") as f:
    f.write(result_bytes)
```

## Лицензия

MIT
