"""Тесты загрузки документов."""

from src.document_loader import load_documents_from_dir, load_text_from_file

import pytest


def test_reads_utf8_with_bom(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Политика защиты информации", encoding="utf-8-sig")
    text = load_text_from_file(path)

    assert text == "Политика защиты информации"
    assert "﻿" not in text


def test_reads_cp1251(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("Пользовательское соглашение".encode("cp1251"))

    assert load_text_from_file(path) == "Пользовательское соглашение"


def test_reads_markdown(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Заголовок\n\nТекст.", encoding="utf-8")

    assert load_text_from_file(path).startswith("# Заголовок")


def test_unsupported_extension_returns_empty(tmp_path, capsys):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")

    assert load_text_from_file(path) == ""
    assert "не поддерживается" in capsys.readouterr().out


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_text_from_file("nope/missing.txt")


def test_broken_pdf_returns_empty_instead_of_raising(tmp_path, capsys):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not really a pdf")

    assert load_text_from_file(path) == ""
    assert "Ошибка при чтении" in capsys.readouterr().out


def test_directory_scan_is_sorted_and_skips_unsupported(tmp_path):
    (tmp_path / "b.txt").write_text("Второй документ", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Первый документ", encoding="utf-8")
    (tmp_path / "skip.xlsx").write_bytes(b"binary")
    documents = load_documents_from_dir(tmp_path)

    assert [doc.source for doc in documents] == ["a.txt", "b.txt"]
    assert documents[0].text == "Первый документ"


def test_directory_scan_is_recursive_with_relative_source(tmp_path):
    nested = tmp_path / "hr" / "2026"
    nested.mkdir(parents=True)
    (nested / "rules.txt").write_text("Правила", encoding="utf-8")
    documents = load_documents_from_dir(tmp_path)

    assert [doc.source for doc in documents] == ["hr/2026/rules.txt"]


def test_non_recursive_scan_is_not_offered(tmp_path):
    """Обход всегда рекурсивный: отдельного режима нет, чтобы не плодить настроек."""
    (tmp_path / "top.txt").write_text("Верхний уровень", encoding="utf-8")
    nested = tmp_path / "inner"
    nested.mkdir()
    (nested / "deep.txt").write_text("Вложенный", encoding="utf-8")
    documents = load_documents_from_dir(tmp_path)

    assert [doc.source for doc in documents] == ["inner/deep.txt", "top.txt"]


def test_empty_file_is_skipped(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n  ", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("Содержимое", encoding="utf-8")
    documents = load_documents_from_dir(tmp_path)

    assert [doc.source for doc in documents] == ["ok.txt"]


def test_missing_directory_raises():
    with pytest.raises(NotADirectoryError):
        load_documents_from_dir("nope/missing-dir")


def test_empty_directory_returns_empty_list(tmp_path):
    assert load_documents_from_dir(tmp_path) == []
