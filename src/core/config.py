import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
dotenv_path = Path('.env')
if dotenv_path.exists():
    try:
        load_dotenv(dotenv_path=dotenv_path)
        # Will be logged after logging setup
    except UnicodeDecodeError:
        pass  # Will be logged after logging setup
else:
    # Попробуем загрузить из env в корне проекта
    alt_dotenv_path = Path('env')
    if alt_dotenv_path.exists():
        try:
            load_dotenv(dotenv_path=alt_dotenv_path)
            # Will be logged after logging setup
        except UnicodeDecodeError:
            pass  # Will be logged after logging setup
    else:
        pass  # Will be logged after logging setup

# Constants - using environment variables
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./BGE-M3")  # Local path to BGE-M3 model
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")  # BGE-M3 model name
API_TOKEN = os.environ.get("API_TOKEN", "default_token_change_me")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 50))
PROCESSING_CONCURRENCY = int(os.environ.get("PROCESSING_CONCURRENCY", 2))

# Batching configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))  # Max requests per batch
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 100))  # Max wait time for batch formation (ms)

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json")  # json or text
USE_JSON_LOGGING = LOG_FORMAT.lower() == "json"

# Initialize structured logging
from src.core.logging_config import setup_logging, get_logger

setup_logging(log_level=LOG_LEVEL, use_json=USE_JSON_LOGGING)
logger = get_logger("config")

# Log configuration (replacing print statements)
logger.info("Configuration loaded successfully", extra={
    "model_path": LOCAL_MODEL_PATH,
    "model_name": MODEL_NAME,
    "api_token_set": API_TOKEN != "default_token_change_me",
    "max_queue_size": MAX_QUEUE_SIZE,
    "processing_concurrency": PROCESSING_CONCURRENCY,
    "batch_size": BATCH_SIZE,
    "batch_timeout_ms": BATCH_TIMEOUT_MS,
    "log_level": LOG_LEVEL,
    "log_format": LOG_FORMAT
}) 