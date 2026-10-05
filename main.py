"""Точка входа: индексирование документов и ответы на вопросы."""

from typing import List, Optional
from pathlib import Path

import argparse
import json
import sys

from src.config import ConfigError, LLMConfig, RAGConfig
from src.document_loader import load_documents_from_dir
from src.assistant import Answer, DocumentAssistant
from src.llm import LLMClient


OUTPUT_FILE = Path("assistant_results.json")

# Вопросы для батч-режима
DEFAULT_QUESTIONS = [
    "О чём пользовательское соглашение?",
    "Что такое бизнес-модель?",  # Русский вопрос к англоязычному документу - на нём видно кросс-языковой поиск
    "Какая цель политики в области защиты информации?",
    "Какая погода сегодня в Москве?",  # Вопрос вне тематики документов - на нём видно, что модель не выдумывает ответ
]

EXIT_COMMANDS = {"exit", "quit", "q", "выход", "выйти"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Document Assistant: ответы на вопросы по документам из каталога.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="каталог с документами (PDF/DOCX/TXT/MD), обходится рекурсивно",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="интерактивный режим: вопросы вводятся с клавиатуры",
    )
    parser.add_argument(
        "-q", "--question", action="append", dest="questions", metavar="ТЕКСТ",
        help="вопрос для батч-режима, можно указать несколько раз",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="офлайн-режим: только поиск фрагментов, без обращения к API",
    )
    parser.add_argument(
        "--show-context", action="store_true",
        help="печатать найденные фрагменты и их оценки сходства",
    )

    return parser.parse_args(argv)


def build_assistant(no_llm: bool) -> Optional[DocumentAssistant]:
    """Собирает ассистента. Возвращает None, если не задан доступ к модели."""
    llm = None

    if no_llm:
        print("LLM отключена (--no-llm): будет выполнен только семантический поиск.")
    else:
        try:
            llm_config = LLMConfig.from_env()
        except ConfigError as e:
            print(f"Ошибка конфигурации: {e}")
            print("Подсказка: для проверки только поиска запустите с флагом --no-llm.")
            return None

        print(f"LLM: {llm_config.model} через {llm_config.base_url}")
        llm = LLMClient(llm_config)

    return DocumentAssistant(config=RAGConfig(), llm=llm)


def print_answer(answer: Answer, show_context: bool) -> None:
    """Печатает ответ, источники и, по желанию, найденный контекст."""
    print(f"\nВопрос: {answer.question}")

    if answer.error:
        print(f"Ошибка обращения к модели: {answer.error}")
    elif answer.text:
        print(f"Ответ: {answer.text}")

    if answer.sources:
        print(f"Источники: {', '.join(answer.sources)}")

    if show_context and answer.retrieved:
        print("Контекст:")
        for i, chunk in enumerate(answer.retrieved, 1):
            preview = chunk.text.replace("\n", " ")[:200]
            print(f"  {i}. [{chunk.source}] сходство {chunk.score:.3f}: {preview}...")


def run_batch(assistant: DocumentAssistant, questions: List[str], show_context: bool) -> int:
    """Отвечает на список вопросов, сохраняет результат в JSON и возвращает число ошибок."""
    print(f"\n---------- ГЕНЕРАЦИЯ ОТВЕТОВ ({len(questions)}) ----------")

    answers = []
    for question in questions:
        answer = assistant.answer_query(question)
        print_answer(answer, show_context)
        answers.append(answer)

    try:
        OUTPUT_FILE.write_text(
            json.dumps([a.to_dict() for a in answers], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nРезультаты сохранены в {OUTPUT_FILE}")

    except OSError as e:
        print(f"\nНе удалось сохранить результаты в {OUTPUT_FILE}: {e}")

    return sum(1 for answer in answers if answer.error)


def run_interactive(assistant: DocumentAssistant, show_context: bool) -> None:
    """Цикл вопрос-ответ с клавиатуры."""
    print("\n---------- ИНТЕРАКТИВНЫЙ РЕЖИМ ----------")
    print(f"Задайте вопрос по документам. Для выхода: {', '.join(sorted(EXIT_COMMANDS))}.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Выход.")
            return

        print_answer(assistant.answer_query(question), show_context)


def main(argv: Optional[List[str]] = None) -> int:
    # В Windows потоки по умолчанию не в UTF-8: без этого русский текст в выводе может упасть/превратиться в мусор
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)

    # Без LLM показывать нечего, кроме найденных фрагментов
    show_context = args.show_context or args.no_llm

    print("---------- ЗАПУСК RAG DOCUMENT ASSISTANT ----------")

    assistant = build_assistant(args.no_llm)
    if assistant is None:
        return 1

    print(f"\n---------- ЗАГРУЗКА ДОКУМЕНТОВ ({args.data_dir}) ----------")
    try:
        documents = load_documents_from_dir(args.data_dir)
    except NotADirectoryError as e:
        print(f"Ошибка: {e}")
        return 1

    if not documents:
        print("Ошибка: нет документов для обработки.")
        return 1

    print("\n---------- ИНДЕКСАЦИЯ ----------")
    assistant.index_documents(documents)

    if not assistant.chunks:
        print("Ошибка: из документов не удалось получить ни одного фрагмента.")
        return 1

    if args.interactive:
        run_interactive(assistant, show_context)
        return 0

    # Ненулевой код возврата, если хотя бы один ответ не получен - чтобы это было видно скрипту
    return 1 if run_batch(assistant, args.questions or DEFAULT_QUESTIONS, show_context) else 0


if __name__ == "__main__":
    sys.exit(main())
