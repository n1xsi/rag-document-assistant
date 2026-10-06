"""Тесты RAG-логики: индексация, поиск, формирование ответа."""

from dataclasses import replace

from src.document_loader import LoadedDocument
from src.assistant import DocumentAssistant
from src.llm import NO_CONTEXT_ANSWER
from src.config import RAGConfig

import pytest


BASE_CONFIG = RAGConfig(chunk_size=200, chunk_overlap=20, top_k=2, min_similarity=0.1)

DOCUMENTS = [
    LoadedDocument(source="agreement.txt", text="Пользовательское соглашение регулирует доступ. "
                                                "Соглашение вступает в силу с момента приёма."),
    LoadedDocument(source="university.txt", text="Университет оценивает успех студентов. "
                                                 "Университет требует посещаемости."),
    LoadedDocument(source="policy.txt", text="Политика защиты информации описывает режим. "
                                             "Политика обязательна для всех."),
]


def build_assistant(fake_encoder, llm=None, **config_overrides) -> DocumentAssistant:
    """Собирает проиндексированного ассистента на заглушках."""
    assistant = DocumentAssistant(
        config=replace(BASE_CONFIG, **config_overrides),
        llm=llm,
        encoder=fake_encoder,
    )
    assistant.index_documents(DOCUMENTS)

    return assistant


@pytest.fixture
def assistant(fake_encoder) -> DocumentAssistant:
    return build_assistant(fake_encoder)


def test_index_creates_chunks_with_sources(assistant):
    assert assistant.chunks
    assert assistant.embeddings is not None
    assert len(assistant.embeddings) == len(assistant.chunks)
    assert {chunk.source for chunk in assistant.chunks} == {
        "agreement.txt", "university.txt", "policy.txt"
    }


def test_retrieve_ranks_relevant_source_first(assistant):
    retrieved = assistant.retrieve("Что говорит университет?")

    assert retrieved
    assert retrieved[0].source == "university.txt"


def test_retrieve_respects_top_k(assistant):
    assert len(assistant.retrieve("соглашение университет политика", top_k=1)) == 1


def test_irrelevant_query_is_filtered_by_threshold(fake_encoder):
    assistant = build_assistant(fake_encoder, min_similarity=0.5)

    # В словаре FakeEncoder нет слов запроса - сходство нулевое
    assert assistant.retrieve("Курс доллара к евро") == []


def test_answer_without_context_does_not_call_llm(fake_encoder, fake_llm):
    assistant = build_assistant(fake_encoder, llm=fake_llm, min_similarity=0.5)

    answer = assistant.answer_query("Курс доллара к евро")

    assert answer.text == NO_CONTEXT_ANSWER
    assert fake_llm.received_questions == []


def test_answer_passes_context_with_sources_to_llm(fake_encoder, fake_llm):
    assistant = build_assistant(fake_encoder, llm=fake_llm)

    answer = assistant.answer_query("Что говорит университет?")

    assert answer.text == fake_llm.reply
    assert answer.error is None
    assert "university.txt" in answer.sources
    assert fake_llm.received_questions == ["Что говорит университет?"]
    # Источник передаётся рядом с текстом, чтобы модель могла на него ссылаться
    assert any(source == "university.txt" for source, _ in fake_llm.received_chunks[0])


def test_offline_mode_returns_retrieved_without_answer(assistant):
    answer = assistant.answer_query("Что говорит университет?")

    assert answer.text == ""
    assert answer.error is None
    assert answer.retrieved
    assert "university.txt" in answer.sources


def test_llm_error_is_reported_separately_from_answer(fake_encoder):
    from src.llm import LLMError

    class FailingLLM:
        def generate(self, question, chunks):
            raise LLMError("401 неверный ключ")

    assistant = build_assistant(fake_encoder, llm=FailingLLM())
    answer = assistant.answer_query("Что говорит университет?")

    # Ошибка не подменяет собой ответ: вызывающий код отличит сбой от результата
    assert answer.text == ""
    assert "401" in answer.error


def test_query_on_empty_index_returns_no_context(fake_encoder):
    assistant = DocumentAssistant(config=BASE_CONFIG, encoder=fake_encoder)

    assert assistant.answer_query("Любой вопрос").text == NO_CONTEXT_ANSWER


def test_index_documents_with_empty_input(fake_encoder):
    assistant = DocumentAssistant(config=BASE_CONFIG, encoder=fake_encoder)
    assistant.index_documents([])

    assert assistant.chunks == []
    assert assistant.embeddings is None


def test_answer_to_dict_shape(fake_encoder, fake_llm):
    assistant = build_assistant(fake_encoder, llm=fake_llm)
    payload = assistant.answer_query("Что говорит университет?").to_dict()

    assert set(payload) == {"question", "answer", "error", "sources", "retrieved"}
    assert payload["retrieved"][0].keys() == {"source", "score", "text"}
