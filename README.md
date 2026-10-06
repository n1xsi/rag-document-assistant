<h1 align="center">

  rag-document-assistant

  [![Python](https://custom-icon-badges.demolab.com/badge/Python-3.14-gray?style=for-the-badge&logo=pythonn&labelColor=white)](#)
  [![NumPy](https://img.shields.io/badge/numpy-2.5.3-gray?style=for-the-badge&logo=numpy&logoColor=013243&labelColor=white)](#)
  [![pytest](https://img.shields.io/badge/pytest-9.1.1-gray?style=for-the-badge&logo=pytest&logoColor=0A9EDC&labelColor=white)](#) <br>
  [![Hugging Face](https://img.shields.io/badge/Sentence--Transformers-white?style=for-the-badge&logo=huggingface&logoColor=FFD21E)](#)
  [![OpenAI](https://custom-icon-badges.demolab.com/badge/OpenAI--compatible-white?style=for-the-badge&logo=openaiblack)](#)

</h1>

Базовая реализация задачи №8 (DocumentAssistant) с использованием подхода RAG. Тестовое задание выполнялось для IT Hub «Северстали».

Запуск LLM локально затруднена, поэтому в проекте использована mock-функция, имитирующая вызов модели (*в коде есть комментарий с примером того, как заменить модель на реальную*).

## Стек технологий
* Python 3.10+
* Sentence-Transformers (Embeddings)
* NumPy/Scikit-Learn (vector search)
* PyPDF & Python-Docx (parsing)

## Запуск
```bash
pip install -r requirements.txt
python main.py
```

В первый раз запуск программы будет скачиваться модель **all-MiniLM-L6-v2** (около 80 МБ). В конце работы в корне проекта будет создан файл `assistant_results.json`.

## Особенности
* Извлечение текста из PDF/DOCX/TXT;
* Семантический поиск с использованием косинусоидального сходства (`cosine_similarity`);
* Настраиваемая стратегия разбиения на фрагменты.

## Status
Approved :shipit: :heavy_check_mark:
