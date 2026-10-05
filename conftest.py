"""
Общие фикстуры для тестов.

Наличие conftest.py в корне проекта добавляет корень в sys.path,
поэтому тесты импортируют пакет src без установки проекта.
"""

from typing import List, Sequence, Tuple

import numpy as np
import pytest


class FakeEncoder:
    """
    Детерминированная замена SentenceTransformer: вектор - частоты слов из словаря.
    Позволяет проверять логику поиска без загрузки модели и обращений к сети.
    """

    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary = list(vocabulary)

    def encode(self, texts, show_progress_bar: bool = False, normalize_embeddings: bool = False):
        vectors = []

        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [float(lowered.count(word)) for word in self.vocabulary],
                dtype=np.float32,
            )

            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm

            vectors.append(vector)

        return np.vstack(vectors)


class FakeLLM:
    """Заглушка LLMClient: запоминает переданный контекст и возвращает фиксированный ответ."""

    def __init__(self, reply: str = "Ответ модели."):
        self.reply = reply
        self.received_questions: List[str] = []
        self.received_chunks: List[List[Tuple[str, str]]] = []

    def generate(self, question: str, chunks: Sequence[Tuple[str, str]]) -> str:
        self.received_questions.append(question)
        self.received_chunks.append(list(chunks))

        return self.reply


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder(["соглашение", "университет", "информация", "политика", "погода"])


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
