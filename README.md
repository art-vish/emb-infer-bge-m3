# emb-infer-bge-m3

> **BGE-M3 Embedding Inference API**: Высокопроизводительный API для получения эмбеддингов с использованием модели BGE-M3 от BAAI, поддерживающий три типа векторных представлений: dense, sparse и colbert для максимальной гибкости в задачах поиска и RAG.

Контейнеризированное FastAPI приложение для продакшн развертывания BGE-M3 модели с поддержкой всех типов эмбеддингов и селективной генерации векторов.

## Функциональность

- **Три типа эмбеддингов**: dense, sparse и colbert векторы
- **Селективная генерация**: выбор конкретных типов векторов через параметры запроса
- **Батчинг и очереди**: интеллектуальная обработка параллельных запросов
- **GPU-ускорение**: автоматическое использование CUDA при наличии
- **Аутентификация**: защита API с помощью Bearer токенов
- **Контейнеризация**: готовые Docker и docker-compose конфигурации
- **Мониторинг**: эндпоинты для проверки здоровья и статистики

## Требования

- Docker и Docker Compose
- NVIDIA GPU (опционально, для ускорения)
- 8+ GB RAM (для загрузки BGE-M3 модели)

## 🚀 Быстрый старт

### Вариант 1: Готовый образ (Рекомендуется)

1. **Скачайте production конфигурацию**:
   ```bash
   curl -O https://raw.githubusercontent.com/art-vish/emb-infer-bge-m3/main/docker-compose.prod.yaml
   ```

2. **Настройте API токен** (опционально):
   ```bash
   export API_TOKEN="your_secure_api_token_here"
   ```

3. **Запустите сервис**:
   ```bash
   docker-compose -f docker-compose.prod.yaml up -d
   ```

### Вариант 2: Сборка из исходного кода

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/art-vish/emb-infer-bge-m3.git
   cd emb-infer-bge-m3
   ```

2. **Настройте окружение**:
   ```bash
   cp .env.example .env
   # Отредактируйте .env файл для установки API_TOKEN
   ```

3. **Запустите сервис**:
   ```bash
   docker-compose up -d
   ```

### Проверка работы
```bash
curl -H "Authorization: Bearer your_api_token_here" http://localhost:8000/health
```

## API Документация

### Аутентификация

Все эндпоинты требуют Bearer токен в заголовке:
```
Authorization: Bearer your_api_token_here
```

### Основной эндпоинт

**`POST /v1/embeddings`** - Генерация BGE-M3 эмбеддингов

Поддерживает селективную генерацию векторов через параметры:
- `return_dense`: генерировать dense векторы (по умолчанию: true)
- `return_sparse`: генерировать sparse векторы (по умолчанию: true) 
- `return_colbert`: генерировать colbert векторы (по умолчанию: true)

### Служебные эндпоинты

- **`GET /health`** - Проверка состояния сервиса
- **`GET /stats`** - Статистика батчинга и производительности
- **`GET /v1/models`** - Информация о доступных моделях
- **`GET /`** - Информация об API

## Примеры использования

### Все типы векторов (по умолчанию)

```python
import requests

headers = {
    "Authorization": "Bearer your_api_token_here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": ["Текст для векторизации", "Еще один текст"]
    }
)

result = response.json()
for emb_data in result['data']:
    print(f"Dense dimension: {len(emb_data['dense_embedding'])}")        # 1024
    print(f"Sparse tokens: {len(emb_data['sparse_embedding'])}")         # Переменное
    print(f"ColBERT vectors: {len(emb_data['colbert_embedding'])}")      # Переменное
```

### Селективная генерация векторов

```python
# Только dense векторы для семантического поиска
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": ["Запрос пользователя"],
        "return_dense": True,
        "return_sparse": False,
        "return_colbert": False
    }
)

# Только sparse векторы для лексического поиска
response = requests.post(
    "http://localhost:8000/v1/embeddings", 
    headers=headers,
    json={
        "input": ["Ключевые слова поиска"],
        "return_dense": False,
        "return_sparse": True,
        "return_colbert": False
    }
)

# Dense + ColBERT для гибридного поиска
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": ["Сложный поисковый запрос"],
        "return_dense": True,
        "return_sparse": False,
        "return_colbert": True
    }
)
```

### Обработка массивов текстов

```python
# Батчевая обработка нескольких текстов
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": [
            "Первый документ для индексации",
            "Второй документ для индексации", 
            "Третий документ для индексации"
        ],
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": False
    }
)

# Результат содержит векторы для каждого входного текста
result = response.json()
print(f"Обработано {len(result['data'])} текстов")
print(f"Типы векторов: {result['embedding_types']}")
```

## Конфигурация

### Переменные окружения

#### Безопасность
- `API_TOKEN`: Токен для аутентификации (обязательно)

#### Конфигурация модели  
- `MODEL_PATH`: Путь к локальной модели (по умолчанию: ./BGE-M3)
- `MODEL_NAME`: Имя модели в API (по умолчанию: BAAI/bge-m3)

#### Настройки производительности
- `BATCH_SIZE`: Размер батча (по умолчанию: 8)
- `BATCH_TIMEOUT_MS`: Таймаут батчинга в мс (по умолчанию: 100)
- `PROCESSING_CONCURRENCY`: Параллельных потоков (по умолчанию: 2)
- `MAX_QUEUE_SIZE`: Максимальный размер очереди (по умолчанию: 50)

### Продакшн конфигурация

Создайте `.env` файл для продакшн развертывания:

```bash
# SECURITY SETTINGS
API_TOKEN=your_api_token_here

# MODEL CONFIGURATION  
MODEL_PATH=/app/BGE-M3
MODEL_NAME=BAAI/bge-m3

# PERFORMANCE TUNING
BATCH_SIZE=16
BATCH_TIMEOUT_MS=50
PROCESSING_CONCURRENCY=4
MAX_QUEUE_SIZE=100
```

Запуск в продакшн:
```bash
docker-compose up -d
```

## Производительность и масштабирование

### Батчинг
Система автоматически объединяет запросы в батчи для повышения пропускной способности:
- Запросы группируются по `BATCH_SIZE`
- Максимальное время ожидания: `BATCH_TIMEOUT_MS`
- Поддержка разных комбинаций векторов в одном батче

### Мониторинг производительности

```bash
# Проверка здоровья
curl -H "Authorization: Bearer your_api_token_here" http://localhost:8000/health

# Статистика батчинга
curl -H "Authorization: Bearer your_api_token_here" http://localhost:8000/stats

# Мониторинг ресурсов контейнера
docker stats
```

Пример ответа `/stats`:
```json
{
  "batching": {
    "queue_size": 0,
    "processed_requests": 1247,
    "average_batch_size": 3.2,
    "average_processing_time": 0.15
  },
  "config": {
    "batch_size": 16,
    "timeout_ms": 50,
    "max_queue_size": 100
  }
}
```

## BGE-M3 Типы векторов

### Dense векторы (1024 измерения)
- **Применение**: Семантический поиск, RAG, кластеризация
- **Особенности**: Непрерывные векторы, хорошо работают с косинусным сходством
- **Формат**: `List[float]` длиной 1024

### Sparse векторы
- **Применение**: Лексический поиск, точное совпадение терминов
- **Особенности**: Разреженные векторы на уровне токенов
- **Формат**: `Dict[int, float]` (token_id -> weight)

### ColBERT векторы
- **Применение**: Детальное сопоставление, fine-grained поиск
- **Особенности**: Множественные векторы для каждого токена
- **Формат**: `List[List[float]]` (список векторов по 1024 измерения)

### Гибридный поиск

Комбинирование типов векторов для оптимальных результатов:

```python
# Пример гибридного поиска
def hybrid_search(query, documents):
    # Получаем все типы векторов для запроса
    query_vectors = get_embeddings(query, all_types=True)
    
    # Семантический поиск (dense)
    semantic_scores = cosine_similarity(
        query_vectors['dense'], 
        doc_dense_vectors
    )
    
    # Лексический поиск (sparse)  
    lexical_scores = sparse_dot_product(
        query_vectors['sparse'],
        doc_sparse_vectors  
    )
    
    # Детальное сопоставление (colbert)
    colbert_scores = max_similarity(
        query_vectors['colbert'],
        doc_colbert_vectors
    )
    
    # Объединение скоров
    final_scores = 0.5 * semantic_scores + 0.3 * lexical_scores + 0.2 * colbert_scores
    return final_scores
```

## Загрузка модели

### Автоматическая загрузка
Модель BGE-M3 загружается автоматически с Hugging Face Hub при первом запуске.

### Предварительная загрузка
```bash
# Установка Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Загрузка полной модели
huggingface-cli download BAAI/bge-m3 --local-dir ./BGE-M3

# Загрузка основных файлов
huggingface-cli download BAAI/bge-m3 model.safetensors --local-dir ./BGE-M3
huggingface-cli download BAAI/bge-m3 config.json --local-dir ./BGE-M3
huggingface-cli download BAAI/bge-m3 tokenizer.json --local-dir ./BGE-M3
```

## Команды Docker

### Разработка
```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

### Продакшн
```bash
# Полная пересборка
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Обновление без даунтайма
docker-compose pull
docker-compose up -d --force-recreate
```

## Устранение неполадок

### Проблемы с памятью
```bash
# Проверка использования памяти
docker stats

# Увеличение лимитов памяти в docker-compose.yaml
services:
  emb-infer-bge-m3:
    deploy:
      resources:
        limits:
          memory: 12G
```

### Проблемы с GPU
```bash
# Проверка доступности NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Проверка логов загрузки модели
docker-compose logs | grep "Using device"
```

### Проблемы с производительностью
- Увеличьте `BATCH_SIZE` для большей пропускной способности
- Уменьшите `BATCH_TIMEOUT_MS` для меньшей задержки
- Настройте `PROCESSING_CONCURRENCY` под ваше железо

## Лицензия

Этот проект использует модель BGE-M3 от BAAI. См. их лицензию для коммерческого использования.

---

## English Documentation

# emb-infer-bge-m3

> **BGE-M3 Embedding Inference API**: High-performance API for BGE-M3 embeddings from BAAI, supporting three types of vector representations: dense, sparse, and colbert for maximum flexibility in search and RAG tasks.

Containerized FastAPI application for production deployment of BGE-M3 model with support for all embedding types and selective vector generation.

## Features

- **Three embedding types**: dense, sparse, and colbert vectors
- **Selective generation**: choose specific vector types via request parameters  
- **Batching & queues**: intelligent concurrent request processing
- **GPU acceleration**: automatic CUDA usage when available
- **Authentication**: API protection with Bearer tokens
- **Containerization**: ready-to-use Docker and docker-compose configurations
- **Monitoring**: health check and statistics endpoints

## Quick Start

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd emb-infer-bge-m3
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env file to set API_TOKEN
   ```

3. **Start service**:
   ```bash
   docker-compose up -d
   ```

4. **Test API**:
   ```bash
   curl -H "Authorization: Bearer your_api_token_here" http://localhost:8000/health
   ```

## API Reference

### Main Endpoint

**`POST /v1/embeddings`** - Generate BGE-M3 embeddings

Supports selective vector generation via parameters:
- `return_dense`: generate dense vectors (default: true)
- `return_sparse`: generate sparse vectors (default: true)
- `return_colbert`: generate colbert vectors (default: true)

### Example Usage

```python
import requests

headers = {
    "Authorization": "Bearer your_api_token_here",
    "Content-Type": "application/json"
}

# All vector types (default)
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": ["Text to embed", "Another text"]
    }
)

# Dense vectors only
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers=headers,
    json={
        "input": ["Search query"],
        "return_dense": True,
        "return_sparse": False,
        "return_colbert": False
    }
)
```

## Configuration

### Environment Variables

- `API_TOKEN`: Authentication token (required)
- `BATCH_SIZE`: Batch size for processing (default: 8)
- `PROCESSING_CONCURRENCY`: Parallel processing threads (default: 2)
- `MAX_QUEUE_SIZE`: Maximum queue size (default: 50)

### Production Setup

```bash
# Set production environment
export API_TOKEN="your_api_token_here"
export BATCH_SIZE=16
export PROCESSING_CONCURRENCY=4
export MAX_QUEUE_SIZE=100

# Start with production settings
docker-compose up -d
```

## Vector Types

- **Dense vectors**: 1024-dimensional continuous embeddings for semantic search
- **Sparse vectors**: Token-level weights for lexical matching
- **ColBERT vectors**: Multi-vector representations for fine-grained matching

Perfect for hybrid search combining semantic and lexical approaches!

## 🐳 Docker Hub

Готовый образ доступен на Docker Hub:

- **Repository**: [`asvishnya/emb-infer-bge-m3`](https://hub.docker.com/r/asvishnya/emb-infer-bge-m3)
- **Latest**: `asvishnya/emb-infer-bge-m3:latest`
- **Stable**: `asvishnya/emb-infer-bge-m3:v1.0.0`

### Быстрый запуск с Docker Hub:

```bash
# Скачать и запустить одной командой
docker run -d \
  -p 8000:8000 \
  -e API_TOKEN=your_api_token_here \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  asvishnya/emb-infer-bge-m3:latest
```

### Или с docker-compose:

```bash
# Скачать production конфигурацию
curl -O https://raw.githubusercontent.com/art-vish/emb-infer-bge-m3/main/docker-compose.prod.yaml

# Запустить
docker-compose -f docker-compose.prod.yaml up -d
```

**Преимущества готового образа:**
- ✅ Быстрый запуск без сборки
- ✅ Проверенная стабильная версия  
- ✅ Автоматические обновления
- ✅ Меньше требований к системе

---

*Built with ❤️ using FastAPI, BGE-M3, and Docker*