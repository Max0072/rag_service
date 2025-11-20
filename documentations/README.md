# RAG Service

Сервис для семантического поиска с использованием эмбеддингов и векторного индекса.

## Основные компоненты

### 1. **SearchEngine** (`search_engine.py`)
Ядро системы для векторного поиска:
- Эмбеддинг модель: `all-MiniLM-L6-v2` (384 измерения)
- Векторный индекс: FAISS IndexFlatIP
- Поиск по одному запросу или батчу запросов

### 2. **API** (`API.py`)
FastAPI сервер для работы с поисковым движком:

**Endpoints:**
- `POST /upload_json` - загрузка данных (JSON/JSONL)
- `POST /search` - поиск по одному запросу
- `POST /search_batch` - батч поиск
- `DELETE /delete_all` - очистка индекса

**Запуск:**
```bash
python API.py
```

**Пример использования:**
```bash
# Загрузка данных
curl -X POST "http://0.0.0.0:8000/upload_json" \
  -H "Content-Type: application/json" \
  -d '[{"text": "sample text", "metadata": "..."}]'

# Поиск
curl -X POST "http://0.0.0.0:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"text": "query text", "k": 5}'
```

### 3. **LLMs** (`LLMs.py`)
Интеграция с языковыми моделями:
- `LLMopenAI` - работа с OpenAI API (gpt-4o-mini)
- Требует `OPENAI_API_KEY` в `.env`

### 4. **Chunkers** (`chunkers.py`)
Разбиение текста на чанки:
- `SemanticChunker` - семантическое разбиение
- `RecursiveCharacterTextChunker` - рекурсивное по символам

### 5. **Datasets** (`datasets_for_evaluation.py`)
Загрузка BeIR/NQ датасета для оценки:
- `corpus` - документы Wikipedia (~2.7M)
- `queries` - вопросы пользователей (~3.5K)
- `qrels` - метки релевантности

## Установка

```bash
# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Настроить переменные окружения
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Требования

- Python 3.13+
- PyTorch (с поддержкой MPS/CUDA опционально)
- OpenAI API key

## Структура проекта

```
RAG_service/
├── API.py                          # FastAPI сервер
├── search_engine.py                # Векторный поисковый движок
├── LLMs.py                         # Интеграция с LLM
├── chunkers.py                     # Чанкеры текста
├── prompt_builders.py              # Билдеры промптов
├── datasets_for_evaluation.py      # Загрузка датасетов
├── main.py                         # Основной скрипт
├── requirements.txt                # Зависимости
└── .env                            # Переменные окружения
```

## Формат данных

**Загрузка корпуса:**
```json
[
  {"text": "текст документа", "metadata": "дополнительная информация"},
  ...
]
```

**Поиск:**
```json
{"text": "поисковый запрос", "k": 5}
```

**Ответ:**
```json
{
  "ids": [0, 15, 42, ...],
  "scores": [0.95, 0.87, 0.82, ...]
}
```