# ImageBot

Telegram-бот для работы с изображениями через OpenAI GPT Image API и RAG-базы знаний на ChromaDB. Поддерживает редактирование и генерацию изображений, а также семантический поиск по документам с ответами в контексте.

## Возможности

### Работа с изображениями

- **1–10 изображений на вход** — объединение, коллажи, стилизация
- **Текстовые команды** — «сделай в стиле акварели», «объедини в коллаж», «убери фон» и т.п.
- **Модели**: gpt-image-1, gpt-image-1.5, DALL-E 2
- **Генерация по тексту** — создание изображений без исходных фото (create, dalle_gen)
- **Текстовый режим** — чат gpt-5.2, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD, память 20 сообщений

### RAG — база знаний

- **Загрузка документов** — TXT, PDF, XLSX, DOCX, MD (папка `data/`)
- **Индексация в ChromaDB** — чанкинг, эмбеддинги OpenAI, векторный поиск
- **Семантический поиск** — режим `/rag_text`, задавайте вопросы с контекстом из документов
- **Управление источниками** — список, удаление с очисткой ChromaDB
- **Память диалога** — 20 последних сообщений в /text и /rag_text

## Требования

- Python 3.10+
- OpenAI API-ключ (GPT Image, Chat, Embeddings)
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

## Структура проекта

```
ImageBot/
├── bot.py              # Telegram-бот и CLI
├── processor.py       # Обработка изображений и чат OpenAI
├── rag_store.py       # RAG на ChromaDB (индексация, поиск, удаление)
├── rag_view_chunks.py # Просмотр чанков в консоли (отдельно от бота)
├── requirements.txt
├── .env.example
├── README.md
├── Dockerfile
├── docker-compose.yml
├── data/              # Загруженные документы для RAG (создаётся по запросу)
└── chroma_db/         # Хранилище ChromaDB (создаётся при индексации)
```

## Запуск

### Docker Compose (рекомендуется для сервера)

```bash
cp .env.example .env
docker compose up -d
```

Отладка с выводом логов:
```bash
docker compose run --rm imagebot python bot.py telegram -v
```

**Dockerfile** — образ на `python:3.13-slim`, копирует `bot.py`, `processor.py`, `rag_store.py`. Точка входа: `python bot.py telegram`.

**Структура volumes (docker-compose):**

| Volume | Путь в контейнере | Назначение |
|--------|-------------------|------------|
| `bot_data` | `/data` | Состояние бота (persistence), `bot_data.pickle` |
| `bot_documents` | `/app/data` | Загруженные документы для RAG (`/rag_add`) |
| `chroma_data` | `/app/chroma_db` | Векторная база ChromaDB (индексы для `/rag_text`) |

Данные в volumes сохраняются между перезапусками контейнера.

### Telegram-бот (локально)

```bash
python bot.py telegram
```

Опционально: `python bot.py telegram --log` или `-v` для логов в консоль.

### CLI

```bash
python bot.py cli
```

Введите пути к изображениям и текстовую команду. Результат сохраняется в `output.png`.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Начало работы, выбор режима по умолчанию |
| `/text` | Текстовый режим: чат, анализ 1 фото, контекст из DOCX/PDF/XLSX/TXT/MD. Память 20 сообщений, /clear — сброс |
| `/image1` | Редактирование 1–10 фото (gpt-image-1) |
| `/image15` | Редактирование 1–10 фото (gpt-image-1.5) |
| `/dalle` | Редактирование 1 фото (DALL-E 2) |
| `/create` | Генерация изображения по тексту (gpt-image-1.5) |
| `/dalle_gen` | Генерация по тексту (DALL-E 2) |
| `/rag_add` | Включить загрузку документов → TXT, PDF, XLSX, DOCX, MD |
| `/rag_index` | Индексировать файлы из `data/` в ChromaDB |
| `/rag_list` | Список источников в RAG |
| `/rag_delete <источник>` | Удалить источник из RAG и ChromaDB |
| `/rag_text` | Режим RAG — задавайте вопросы, ответы по документам (до смены режима) |
| `/rag_clear` | Очистить историю сеанса /rag_text |
| `/clear` | Очистить историю сеанса /text |
| `/help` | Справка по командам |

## Примеры

### Текст с контекстом документа (/text)

1. `/text` → переключение в текстовый режим
2. Отправить `report.docx` или `data.pdf`
3. «Какие основные выводы?» → ответ с учётом документа
4. `/clear` → сброс контекста и истории

### Изображения

- «Сделай это фото в стиле импрессионизма»
- «Объедини все изображения в один коллаж»
- «Преобразуй в пиксель-арт»
- «Убери фон и оставь прозрачным»

### RAG

1. `/rag_add` → отправить `policy.pdf` или `report.docx`
2. `/rag_index` → «Проиндексировано 25 чанков из 1 файла»
3. `/rag_text` → режим RAG включён
4. Какие условия отпуска? → ответ с цитатами из документа (можно задавать новые вопросы без команды)
5. `/text` или другая команда → выход из режима RAG

## API

### Обработка изображений

```python
from pathlib import Path
from processor import ImageProcessor

processor = ImageProcessor()
result_bytes, usage = processor.process(
    images=[Path("photo1.jpg"), Path("photo2.png")],
    prompt="Объедини в коллаж",
    quality="high",
    size="1536x1024",
)
Path("result.png").write_bytes(result_bytes)
```

### RAG

```python
from rag_store import RAGStore, DATA_DIR

store = RAGStore()
store.index_documents(DATA_DIR)  # Индексация из data/
store.list_sources()             # Список источников
results = store.query("вопрос")  # Семантический поиск
store.delete_source("doc.pdf")   # Удаление источника
```

### Просмотр чанков (консоль)

```bash
python rag_view_chunks.py              # все чанки
python rag_view_chunks.py --list       # список источников
python rag_view_chunks.py --source doc.pdf --limit 5
```

## Лицензия

MIT
