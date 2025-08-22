import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
dotenv_path = Path('.env')
if dotenv_path.exists():
    try:
        load_dotenv(dotenv_path=dotenv_path)
        print("Loaded .env file successfully")
    except UnicodeDecodeError:
        print("Warning: .env file has encoding issues, skipping and using system environment variables")
else:
    # Попробуем загрузить из env в корне проекта
    alt_dotenv_path = Path('env')
    if alt_dotenv_path.exists():
        try:
            load_dotenv(dotenv_path=alt_dotenv_path)
            print("Loaded env file successfully")
        except UnicodeDecodeError:
            print("Warning: env file has encoding issues, using system environment variables")
    else:
        print("Info: No .env file found, using system environment variables")

# Constants - using environment variables
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./BGE-M3")  # Local path to BGE-M3 model
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")  # BGE-M3 model name
API_TOKEN = os.environ.get("API_TOKEN", "default_token_change_me")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 50))
PROCESSING_CONCURRENCY = int(os.environ.get("PROCESSING_CONCURRENCY", 2))

# Batching configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))  # Max requests per batch
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 100))  # Max wait time for batch formation (ms)

# Для отладки
print(f"Loaded configuration:")
print(f"MODEL_PATH: {LOCAL_MODEL_PATH}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"API_TOKEN: {'***' + API_TOKEN[-4:] if len(API_TOKEN) > 4 else 'Not set properly'}")
print(f"MAX_QUEUE_SIZE: {MAX_QUEUE_SIZE}")
print(f"PROCESSING_CONCURRENCY: {PROCESSING_CONCURRENCY}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"BATCH_TIMEOUT_MS: {BATCH_TIMEOUT_MS}ms") 