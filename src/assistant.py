"""RAG-ассистент: индексация документов, семантический поиск, генерация ответа."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .llm import NO_CONTEXT_ANSWER, LLMClient, LLMError
from .document_loader import LoadedDocument
from .chunking import split_text
from .config import RAGConfig


@dataclass
class Chunk:
    """Фрагмент документа вместе с источником, из которого он взят."""
    text: str
    source: str


@dataclass
class RetrievedChunk(Chunk):
    """Найденный фрагмент с оценкой близости к запросу."""
    score: float = 0.0


@dataclass
class Answer:
    """
    Результат работы ассистента по одному вопросу.
    Ассистент возвращает данные; как их показать, решает вызывающий код.
    """
    question: str
    text: str = ""
    error: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Представление для сохранения в JSON."""
        payload = asdict(self)
        payload["answer"] = payload.pop("text")

        for chunk in payload["retrieved"]:
            chunk["score"] = round(chunk["score"], 4)

        return payload


class DocumentAssistant:
    """
    Полный цикл RAG: документы -> фрагменты -> эмбеддинги -> поиск -> промпт -> ответ LLM.

    llm=None отключает модель (поиск работает).
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[LLMClient] = None,
        encoder=None,
    ):
        self.config = config or RAGConfig()
        self.llm = llm
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.encoder = encoder or self._load_encoder()

    def _load_encoder(self):
        """Загружает модель эмбеддингов, предпочитая уже скачанные файлы сетевой проверке."""
        # Lazy Import
        from sentence_transformers import SentenceTransformer

        print(f"Загрузка модели эмбеддингов {self.config.embedding_model}...")

        try:
            # local_files_only=True чтобы не проверять наличие файлов в сети, если модель уже скачана
            return SentenceTransformer(self.config.embedding_model, local_files_only=True)

        except Exception:
            print("Модель не найдена локально, скачиваю...")
            return SentenceTransformer(self.config.embedding_model)

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Векторизует тексты и возвращает массив формы (len(texts), dim)."""
        vectors = self.encoder.encode(
            list(texts),
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        return np.asarray(vectors, dtype=np.float32)

    def index_documents(self, documents: Sequence[LoadedDocument]) -> None:
        """Разбивает документы на фрагменты и считает эмбеддинги."""
        self.chunks = []
        for doc in documents:
            self.chunks.extend(
                Chunk(text=text, source=doc.source)
                for text in split_text(doc.text, self.config.chunk_size, self.config.chunk_overlap)
            )

        if not self.chunks:
            self.embeddings = None
            return

        print(f"Вычисление эмбеддингов для {len(self.chunks)} фрагментов...")
        self.embeddings = self._encode([c.text for c in self.chunks], show_progress=True)
        print(f"Индексация завершена: {len(self.chunks)} фрагментов из {len(documents)} документов.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Находит фрагменты, наиболее близкие к запросу по косинусному сходству.
        Фрагменты со сходством ниже config.min_similarity отбрасываются.
        """
        if not self.chunks or self.embeddings is None:
            return []

        top_k = top_k or self.config.top_k

        # Векторы УЖЕ нормализованы => косинусное сходство = скалярное произведение
        scores = self.embeddings @ self._encode([query])[0]
        best_indices = scores.argsort()[-top_k:][::-1]

        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                source=self.chunks[i].source,
                score=float(scores[i]),
            )
            for i in best_indices
            if scores[i] >= self.config.min_similarity
        ]

    def answer_query(self, query: str) -> Answer:
        """Отвечает на вопрос: поиск фрагментов -> промпт -> генерация."""
        retrieved = self.retrieve(query)
        sources = list(dict.fromkeys(chunk.source for chunk in retrieved))

        # Ни один фрагмент не прошёл порог релевантности - к модели не обращаемся
        if not retrieved:
            return Answer(question=query, text=NO_CONTEXT_ANSWER)

        # Офлайн-режим: найденные фрагменты уже лежат в retrieved, ответ модели не нужен
        if self.llm is None:
            return Answer(question=query, sources=sources, retrieved=retrieved)

        try:
            text = self.llm.generate(query, [(c.source, c.text) for c in retrieved])
        except LLMError as e:
            return Answer(question=query, error=str(e), sources=sources, retrieved=retrieved)

        return Answer(question=query, text=text, sources=sources, retrieved=retrieved)
