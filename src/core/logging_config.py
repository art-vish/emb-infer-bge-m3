import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        extra_fields = ['request_id', 'user_id', 'duration', 'batch_size', 'model_name',
                       'test_field', 'count', 'warning_type', 'error_code', 'device',
                       'path', 'model', 'load_time', 'error', 'vector_types', 'compute_time',
                       'api_token_set', 'max_queue_size', 'processing_concurrency', 
                       'batch_timeout_ms', 'log_level', 'log_format', 'timeout_ms']
        
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_level: str = "INFO", use_json: bool = True) -> None:
    """Setup structured logging configuration"""
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Our application logger
    app_logger = logging.getLogger("emb_infer_bge_m3")
    app_logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str = "emb_infer_bge_m3") -> logging.Logger:
    """Get application logger with structured logging"""
    return logging.getLogger(name)


# Context manager for request logging
class RequestContext:
    """Context manager for adding request-specific info to logs"""
    
    def __init__(self, request_id: str, user_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.logger = get_logger()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def log_info(self, message: str, **kwargs):
        extra = {"request_id": self.request_id}
        if self.user_id:
            extra["user_id"] = self.user_id
        extra.update(kwargs)
        self.logger.info(message, extra=extra)
        
    def log_error(self, message: str, **kwargs):
        extra = {"request_id": self.request_id}
        if self.user_id:
            extra["user_id"] = self.user_id
        extra.update(kwargs)
        self.logger.error(message, extra=extra)
        
    def log_warning(self, message: str, **kwargs):
        extra = {"request_id": self.request_id}
        if self.user_id:
            extra["user_id"] = self.user_id
        extra.update(kwargs)
        self.logger.warning(message, extra=extra)
