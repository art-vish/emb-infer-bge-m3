# Тесты emb-infer-bge-m3

Тесты для BGE-M3 Embedding Inference API с селективной генерацией векторов.

## Быстрый тест

```bash
# Убедитесь, что сервер запущен
docker-compose up -d

# Запустите функциональный тест
python tests/test_api_quick.py

# Или через pytest
python -m pytest tests/ -v
```

## Доступные тесты

### Основной тест
- **`test_api_quick.py`** - Полный функциональный тест API
  - Проверка health endpoint
  - Тестирование всех типов векторов (dense, sparse, colbert)
  - Селективная генерация векторов
  - Батчевая обработка

## Конфигурация

Тесты автоматически читают токен из `.env` файла:

```bash
# .env
API_TOKEN=prod_bge_m3_secure_token_2024
```

## Поддерживаемые комбинации векторов

- `return_dense=true|false` - Dense vectors (1024 dimensions)
- `return_sparse=true|false` - Sparse vectors (token-level weights)  
- `return_colbert=true|false` - ColBERT vectors (multi-vector representations)

По умолчанию все типы включены.

## Примеры тестирования

```python
# Все типы векторов
{
    "input": "Test text"
}

# Только dense векторы
{
    "input": "Test text",
    "return_dense": true,
    "return_sparse": false,
    "return_colbert": false
}

# Batch processing
{
    "input": ["Text 1", "Text 2", "Text 3"],
    "return_dense": true,
    "return_sparse": true,
    "return_colbert": false
}
```