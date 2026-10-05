"""Разбиение текста на фрагменты по границам абзацев и предложений."""

from typing import List

import textwrap
import re


# Разделитель блоков внутри фрагмента
SEPARATOR = "\n\n"

# Абзацы разделяются одной или несколькими пустыми строками
_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")

# Конец предложения: точка/восклицательный/вопросительный знак или многоточие + пробел
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

# Три и более перевода строки сжимаются до разделителя абзацев
_EXTRA_NEWLINES = re.compile(r"\n{3,}")

# Повторяющиеся пробелы и табуляции внутри строки
_EXTRA_SPACES = re.compile(r"[ \t ]+")


def normalize_text(text: str) -> str:
    """
    Приводит текст к единому виду: убирает лишние пробелы/переводы строк.
    Границы абзацев сохраняются - на них опирается разбиение на фрагменты.
    """
    text = _EXTRA_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())

    return _EXTRA_NEWLINES.sub(SEPARATOR, text).strip()


def _overlap_tail(text: str, size: int) -> str:
    """
    Возвращает хвост текста длиной до size символов для перекрытия со следующим фрагментом.
    Срез выравнивается по границе слова, чтобы фрагмент не начинался с обрубка.
    """
    if size <= 0:
        return ""

    # split(None, 1) отбрасывает обрезанное слово в начале среза
    parts = text[-size:].split(None, 1)

    return parts[-1].strip() if parts else ""


def _split_oversized(unit: str, limit: int) -> List[str]:
    """
    Делит слишком длинный абзац: сначала по предложениям, затем, если нужно, по словам.
    Гарантирует, что каждая часть не длиннее limit.
    """
    parts: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush() -> None:
        # Изменение переменной из внешней области видимости
        nonlocal buffer_len
        if buffer:
            parts.append(" ".join(buffer))
            buffer.clear()
            buffer_len = 0

    for sentence in _SENTENCE_BOUNDARY.split(unit):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Предложение не помещается целиком: режем по словам, а слова длиннее лимита - по символам
        if len(sentence) > limit:
            flush()
            parts.extend(textwrap.wrap(sentence, limit, break_on_hyphens=False))
            continue

        if buffer and buffer_len + 1 + len(sentence) > limit:
            flush()

        buffer_len += len(sentence) + (1 if buffer else 0)
        buffer.append(sentence)

    flush()

    return parts


def split_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """
    Разбивает текст на фрагменты размером около chunk_size символов с перекрытием overlap.
    Границы фрагментов проходят по абзацам и предложениям = слова и мысли не рвутся посередине.
    """
    text = normalize_text(text)
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным.")

    # Перекрытие больше половины фрагмента = сильное дублирование текста
    overlap = max(0, min(overlap, chunk_size // 2))

    # Абзацы - минимальные неделимые блоки, слишком длинные дробим заранее
    units: List[str] = []
    for paragraph in _PARAGRAPH_SEPARATOR.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_oversized(paragraph, chunk_size))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        if current and current_len + len(SEPARATOR) + len(unit) > chunk_size:
            chunk = SEPARATOR.join(current)
            chunks.append(chunk)

            tail = _overlap_tail(chunk, overlap)
            current = [tail] if tail else []
            current_len = len(tail)

        current_len += len(unit) + (len(SEPARATOR) if current else 0)
        current.append(unit)

    if current:
        chunks.append(SEPARATOR.join(current))

    return chunks
