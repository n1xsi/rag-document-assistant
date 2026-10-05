"""Клиент генерации ответов через OpenAI-compatible API."""

from typing import Sequence, Tuple

from .config import LLMConfig


NO_CONTEXT_ANSWER = "В предоставленных документах нет информации по этому вопросу."

SYSTEM_PROMPT = (
    "Ты - ассистент по корпоративным документам. Отвечай на вопрос пользователя, "
    "опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты документов.\n"
    "Правила:\n"
    "1. Не используй знания вне приведённых фрагментов и ничего не додумывай.\n"
    f"2. Если фрагментов недостаточно для ответа, прямо скажи: \"{NO_CONTEXT_ANSWER}\"\n"
    "3. Ссылайся на источник в формате [название файла], откуда взята информация.\n"
    "4. Отвечай по-русски, кратко и по делу - без вступлений и повторения вопроса."
)


class LLMError(RuntimeError):
    """Ошибка обращения к API модели."""


class LLMClient:
    """Обёртка над OpenAI SDK: он совместим с любым endpoint'ом /v1/chat/completions."""

    def __init__(self, config: LLMConfig):
        # Lazy Import
        from openai import OpenAI

        self.config = config
        # Если что - SDK сам повторяет запрос при ошибках и таймаутах
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def generate(self, question: str, chunks: Sequence[Tuple[str, str]]) -> str:
        """
        Отправляет вопрос с найденным контекстом в модель и возвращает текст ответа.

        chunks: источник + текст фрагмента.
        """
        from openai import APIStatusError, APITimeoutError, OpenAIError

        context = "\n\n---\n\n".join(f"[{source}]\n{text}" for source, text in chunks)
        user_prompt = f"Фрагменты документов:\n\n{context}\n\nВопрос: {question}"

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        except APITimeoutError as e:
            raise LLMError(
                f"Модель не ответила за {self.config.timeout:.0f} с (попыток: {self.config.max_retries + 1})."
            ) from e

        except APIStatusError as e:
            raise LLMError(
                f"API вернул ошибку {e.status_code} для модели '{self.config.model}': {e.message}"
            ) from e

        except OpenAIError as e:
            raise LLMError(f"Не удалось обратиться к {self.config.base_url}: {e}") from e

        if not response.choices:
            raise LLMError("API вернул ответ без вариантов генерации.")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("Модель вернула пустой ответ.")

        return content.strip()
