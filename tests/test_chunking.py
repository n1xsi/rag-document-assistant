"""Тесты разбиения текста на фрагменты."""

from src.chunking import normalize_text, split_text

import pytest


def test_empty_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\n  \t ") == []


def test_normalize_collapses_whitespace_and_newlines():
    text = "Первая   строка  \n\n\n\n  Вторая строка   "

    assert normalize_text(text) == "Первая строка\n\nВторая строка"


def test_normalize_handles_windows_line_endings():
    assert normalize_text("Строка\r\nещё\r\n") == "Строка\nещё"


def test_short_text_stays_single_chunk():
    text = "Короткий абзац про защиту информации."

    assert split_text(text, chunk_size=700, overlap=100) == [text]


def test_chunks_do_not_exceed_size_plus_overlap():
    paragraphs = [f"Абзац номер {i}. " + "Текст про защиту информации. " * 8 for i in range(20)]
    text = "\n\n".join(paragraphs)

    chunks = split_text(text, chunk_size=300, overlap=50)

    assert chunks
    # Перекрытие добавляется к началу фрагмента, поэтому допуск - chunk_size + overlap
    assert all(len(chunk) <= 350 for chunk in chunks)


def test_every_paragraph_survives_in_some_chunk():
    paragraphs = [f"Пункт {i} политики защиты информации." for i in range(1, 15)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, chunk_size=120, overlap=20)

    for paragraph in paragraphs:
        assert any(paragraph in chunk for chunk in chunks), paragraph


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Предложение номер {i} в документе." for i in range(1, 30))
    chunks = split_text(text, chunk_size=200, overlap=60)

    assert len(chunks) > 1
    # Начало каждого следующего фрагмента должно встречаться в предыдущем
    for previous, current in zip(chunks, chunks[1:]):
        head = current[:20]
        assert head in previous


def test_long_paragraph_split_by_sentences():
    sentence = "Это предложение средней длины про обработку документов. "
    text = sentence * 12
    chunks = split_text(text, chunk_size=200, overlap=0)

    assert len(chunks) > 1
    # Разбиение по предложениям: фрагменты заканчиваются знаком конца предложения
    assert all(chunk.rstrip().endswith(".") for chunk in chunks)


def test_word_longer_than_chunk_size_is_hard_split():
    text = "а" * 500
    chunks = split_text(text, chunk_size=100, overlap=0)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks) == text


def test_overlap_is_capped_at_half_of_chunk_size():
    text = "\n\n".join(f"Абзац {i} с текстом." for i in range(1, 20))
    # overlap больше chunk_size привёл бы к бесконечному дублированию
    chunks = split_text(text, chunk_size=100, overlap=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 150 for chunk in chunks)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        split_text("Текст", chunk_size=0)
