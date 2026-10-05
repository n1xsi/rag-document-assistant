"""Конфигурация проекта: доступ к модели читается из .env, остальное - значения по умолчанию."""

from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

import os


# Загрузка .env из корня проекта (существующие переменные окружения - приоритет)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Некорректная или неполная конфигурация."""


@dataclass
class LLMConfig:
    """Параметры подключения к LLM через OpenAI-compatible API."""
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Чтение из окружения доступа к модели - три параметра, которые у каждого свои.
        Остальные значения правятся прямо в полях датакласса.
        """
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "Не задан LLM_API_KEY. Скопируйте .env.example в .env и укажите ключ API."
            )

        return cls(
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip(),
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
        )


@dataclass
class RAGConfig:
    """Параметры индексации и поиска."""
    # Мультиязычная модель: документы и запросы на русском языке
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chunk_size: int = 700     # Символов в одном фрагменте
    chunk_overlap: int = 100  # Перекрытие между фрагментами
    top_k: int = 4            # Сколько фрагментов уходит в контекст

    # Порог отсечения: на документах из 'data/' релевантные фрагменты дают 0.33-0.73, вопросы вне тематики <=0.26
    min_similarity: float = 0.30
