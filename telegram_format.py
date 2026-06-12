"""
Разметка ответов для Telegram.

Основной путь (Bot API 10.1+): sendRichMessage + Rich Markdown (GFM-таблицы, списки).
Fallback: parse_mode=HTML через format_chat_markup (таблицы -> блоки «ключ: значение»).

Конструкции проекта:
  **жирный**   __курсив__   ~~зачёркнутый~~   `моноширный`   ||скрытый||
"""

from __future__ import annotations

import html
import os
import re

# Лимиты Telegram: Rich Message до 32768 символов; обычное сообщение ~4096.
RICH_MESSAGE_MAX_LEN = 32000
LEGACY_HTML_MAX_LEN = 4000

# В Rich Markdown __ = жирный; в проекте __ = курсив -> конвертируем в _..._
_PROJECT_ITALIC = re.compile(r"__(.+?)__", re.DOTALL)


def use_rich_messages() -> bool:
    """Rich Message (sendRichMessage) включён по умолчанию."""
    return os.environ.get("TELEGRAM_USE_RICH_MESSAGES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def prepare_telegram_rich_markdown(text: str) -> str:
    """
    Подготовка Markdown для sendRichMessage (Bot API 10.1+).
    Таблицы GFM (| col |) передаются как есть — клиент рендерит нативно.
    """
    if not text:
        return ""
    return _PROJECT_ITALIC.sub(r"_\1_", text)

_FENCED_CODE = re.compile(r"```[^\n]*\n?([\s\S]*?)```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_SPOILER = re.compile(r"\|\|(.+?)\|\|", re.DOTALL)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"__(.+?)__", re.DOTALL)
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_HEADER = re.compile(r"^(#{1,3})\s+(.+)$")
_LIST_ITEM = re.compile(r"^(\s*)[-*•]\s+(.+)$")

_INLINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_LINK, "a"),
    (_SPOILER, "tg-spoiler"),
    (_BOLD, "b"),
    (_ITALIC, "i"),
    (_STRIKE, "s"),
]


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _esc_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(_SEPARATOR_CELL.match(cell.strip()) for cell in cells if cell.strip())


def _table_block_to_html(lines: list[str]) -> str:
    """Markdown-таблица -> блоки «<b>Колонка</b>: значение» по строкам."""
    if len(lines) < 2:
        return _format_text_block("\n".join(lines))

    header = _split_table_row(lines[0])
    separator = _split_table_row(lines[1])
    if not _is_table_separator_row(separator):
        return _format_text_block("\n".join(lines))

    row_blocks: list[str] = []
    for line in lines[2:]:
        if "|" not in line or not line.strip():
            break
        cells = _split_table_row(line)
        pairs: list[str] = []
        for idx, head in enumerate(header):
            value = cells[idx] if idx < len(cells) else ""
            if not head and not value:
                continue
            pairs.append(f"<b>{_format_inline(head)}</b>: {_format_inline(value)}")
        if pairs:
            row_blocks.append("\n".join(pairs))

    return "\n\n".join(row_blocks)


def _split_markdown_blocks(text: str) -> list[tuple[str, str]]:
    """
    Делит текст на блоки: text | table | pre.
    Таблицы и ``` не разрываются при последующей нарезке на сообщения.
    """
    if not text:
        return []

    blocks: list[tuple[str, str]] = []
    lines = text.split("\n")
    i = 0
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            blocks.append(("text", "\n".join(text_buf)))
            text_buf.clear()

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            flush_text()
            fence_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence_lines.append(lines[i])
                i += 1
            if i < len(lines):
                fence_lines.append(lines[i])
                i += 1
            blocks.append(("fence", "\n".join(fence_lines)))
            continue

        if "|" in line and i + 1 < len(lines):
            next_cells = _split_table_row(lines[i + 1])
            if _is_table_separator_row(next_cells):
                flush_text()
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(("table", "\n".join(table_lines)))
                continue

        text_buf.append(line)
        i += 1

    flush_text()
    return blocks


def _format_styles(raw: str) -> str:
    """Рекурсивно применяет inline-разметку к фрагменту без `code`."""
    if not raw:
        return ""
    best_match = None
    best_tag = ""
    for pattern, tag in _INLINE_PATTERNS:
        match = pattern.search(raw)
        if match and (best_match is None or match.start() < best_match.start()):
            best_match = match
            best_tag = tag
    if best_match is None:
        return _esc(raw)

    before = _format_styles(raw[: best_match.start()])
    inner = _format_styles(best_match.group(1))
    after = _format_styles(raw[best_match.end() :])
    if best_tag == "tg-spoiler":
        return f"{before}<tg-spoiler>{inner}</tg-spoiler>{after}"
    if best_tag == "a":
        href = _esc_attr(best_match.group(2))
        return f'{before}<a href="{href}">{inner}</a>{after}'
    return f"{before}<{best_tag}>{inner}</{best_tag}>{after}"


def _format_inline(raw: str) -> str:
    """Inline: `code` и стили (** __ ~~ ||, ссылки)."""
    if not raw:
        return ""
    parts: list[str] = []
    pos = 0
    for match in _INLINE_CODE.finditer(raw):
        parts.append(_format_styles(raw[pos : match.start()]))
        parts.append(f"<code>{_esc(match.group(1))}</code>")
        pos = match.end()
    parts.append(_format_styles(raw[pos:]))
    return "".join(parts)


def _format_text_block(raw: str) -> str:
    """Абзацы: заголовки #, списки -, обычные строки."""
    if not raw:
        return ""
    out: list[str] = []
    for line in raw.split("\n"):
        if not line.strip():
            out.append("")
            continue
        header = _HEADER.match(line)
        if header:
            out.append(f"<b>{_format_inline(header.group(2).strip())}</b>")
            continue
        item = _LIST_ITEM.match(line)
        if item:
            out.append(f"• {_format_inline(item.group(2))}")
            continue
        out.append(_format_inline(line))
    return "\n".join(out)


def format_chat_markup(text: str) -> str:
    """Преобразует Markdown-подобный текст в Telegram HTML."""
    if not text:
        return ""

    parts: list[str] = []
    for kind, content in _split_markdown_blocks(text):
        if kind == "fence":
            match = _FENCED_CODE.match(content)
            body = match.group(1).rstrip() if match else content
            parts.append(f"<pre>{_esc(body)}</pre>")
        elif kind == "table":
            parts.append(_table_block_to_html(content.split("\n")))
        else:
            parts.append(_format_text_block(content))
    return "\n".join(part for part in parts if part)


def chunk_markdown_for_telegram(text: str, max_len: int = 4000) -> list[str]:
    """
    Нарезка длинного ответа на части ≤ max_len без разрыва таблиц и ``` блоков.
  """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    blocks = _split_markdown_blocks(text)
    chunks: list[str] = []
    current = ""

    def append_chunk(value: str) -> None:
        nonlocal current
        if value:
            chunks.append(value)

    for kind, content in blocks:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{content}" if current else content
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            append_chunk(current)
            current = ""
        if len(content) <= max_len:
            current = content
            continue
        if kind == "text":
            for line in content.split("\n"):
                sep = "\n" if current else ""
                cand = f"{current}{sep}{line}" if current else line
                if len(cand) <= max_len:
                    current = cand
                else:
                    if current:
                        append_chunk(current)
                    while len(line) > max_len:
                        append_chunk(line[:max_len])
                        line = line[max_len:]
                    current = line
        else:
            append_chunk(content[:max_len])
            rest = content[max_len:]
            while rest:
                append_chunk(rest[:max_len])
                rest = rest[max_len:]

    if current:
        append_chunk(current)
    return chunks
