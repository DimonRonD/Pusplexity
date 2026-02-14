#!/usr/bin/env python3
"""
Скрипт для просмотра чанков в RAG-хранилище ChromaDB.
Работает отдельно от бота, только читает данные.
Использование:
    python rag_view_chunks.py              # все источники и чанки
    python rag_view_chunks.py --list       # только список источников
    python rag_view_chunks.py --source doc.pdf  # чанки по источнику
    python rag_view_chunks.py --limit 10   # не более N чанков
"""

import argparse
import sys
from pathlib import Path


def get_collection(persist_dir: Path):
    """Подключается к ChromaDB и возвращает коллекцию."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("Ошибка: установите chromadb: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        print(f"Хранилище не найдено: {persist_dir}", file=sys.stderr)
        print("Сначала запустите бота и выполните /rag_add + /rag_index.", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="imagebot_rag",
        metadata={"description": "RAG collection for ImageBot"},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Просмотр чанков RAG-хранилища ChromaDB"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("chroma_db"),
        help="Путь к папке chroma_db (по умолчанию: chroma_db)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Только показать список источников",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Показать чанки только по указанному источнику (имя из /rag_list)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Максимальное количество чанков для вывода (0 = без ограничений)",
    )
    args = parser.parse_args()

    collection = get_collection(args.path)

    if args.source:
        res = collection.get(
            where={"source": args.source},
            include=["documents", "metadatas"],
        )
    else:
        res = collection.get(
            include=["documents", "metadatas"],
        )

    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", []) or [{}] * len(ids)

    if not ids:
        print("Хранилище пусто или источник не найден.")
        return

    sources = sorted({m.get("source", "?") for m in metas if m})
    print(f"Источники: {', '.join(sources)}\n")

    if args.list:
        for s in sources:
            count = sum(1 for m in metas if m and m.get("source") == s)
            print(f"  • {s}: {count} чанков")
        return

    print("=" * 60)
    for i, (doc_id, doc_text, meta) in enumerate(zip(ids, docs, metas)):
        if args.limit and i >= args.limit:
            print(f"\n... показано {args.limit} из {len(ids)} чанков")
            break
        source = meta.get("source", "?")
        print(f"\n[{doc_id}] {source}")
        print("-" * 40)
        print(doc_text[:500] + "..." if len(doc_text) > 500 else doc_text)
        print()

    print("=" * 60)
    print(f"Всего чанков: {len(ids)}")


if __name__ == "__main__":
    main()
