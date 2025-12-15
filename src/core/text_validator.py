"""
Text validation utilities for BGE-M3 model
"""
import os
import re
from typing import List, Union
from fastapi import HTTPException, status
from src.core.logging_config import get_logger

logger = get_logger("text_validator")

# BGE-M3 model limits
MAX_TOKEN_LENGTH = 8192  # BGE-M3 maximum token length
MAX_CHAR_LENGTH = 32768  # Rough estimate: ~4 chars per token
MIN_CHAR_LENGTH = 1      # Minimum text length
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 100))  # Maximum texts in one request

# Rough token estimation (more accurate would require tokenizer)
CHARS_PER_TOKEN_ESTIMATE = 4


class TextValidator:
    """Validator for text inputs to BGE-M3 model"""
    
    def __init__(self):
        self.max_tokens = MAX_TOKEN_LENGTH
        self.max_chars = MAX_CHAR_LENGTH
        self.min_chars = MIN_CHAR_LENGTH
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Remove extra whitespace and count characters
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        char_count = len(cleaned_text)
        
        # Rough estimation: average 4 characters per token
        estimated_tokens = max(1, char_count // CHARS_PER_TOKEN_ESTIMATE)
        return estimated_tokens
    
    def validate_single_text(self, text: str, text_index: int = 0) -> None:
        """Validate a single text input"""
        if not isinstance(text, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {text_index} must be a string, got {type(text).__name__}"
            )
        
        # Check minimum length
        if len(text.strip()) < self.min_chars:
            logger.warning("Text too short", extra={
                "text_index": text_index,
                "text_length": len(text),
                "min_length": self.min_chars
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {text_index} is too short (minimum {self.min_chars} characters)"
            )
        
        # Check maximum character length
        if len(text) > self.max_chars:
            logger.warning("Text too long by characters", extra={
                "text_index": text_index,
                "text_length": len(text),
                "max_chars": self.max_chars
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {text_index} is too long ({len(text)} chars, maximum {self.max_chars})"
            )
        
        # Estimate token count
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens > self.max_tokens:
            logger.warning("Text too long by estimated tokens", extra={
                "text_index": text_index,
                "estimated_tokens": estimated_tokens,
                "max_tokens": self.max_tokens,
                "text_length": len(text)
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {text_index} is too long (~{estimated_tokens} tokens, maximum {self.max_tokens}). "
                       f"Consider splitting into smaller chunks."
            )
        
        logger.debug("Text validation passed", extra={
            "text_index": text_index,
            "text_length": len(text),
            "estimated_tokens": estimated_tokens
        })
    
    def validate_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """Validate input texts and return normalized list"""
        # Normalize to list
        if isinstance(texts, str):
            text_list = [texts]
        elif isinstance(texts, list):
            if not texts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input cannot be empty"
                )
            text_list = texts
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input must be string or list of strings, got {type(texts).__name__}"
            )
        
        # Check batch size limit
        if len(text_list) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many texts in batch: {len(text_list)} (maximum: {MAX_BATCH_SIZE})"
            )
        
        # Validate each text
        for i, text in enumerate(text_list):
            self.validate_single_text(text, i)
        
        total_estimated_tokens = sum(self.estimate_tokens(text) for text in text_list)
        
        logger.info("Batch text validation completed", extra={
            "text_count": len(text_list),
            "total_estimated_tokens": total_estimated_tokens,
            "avg_tokens_per_text": round(total_estimated_tokens / len(text_list), 1)
        })
        
        return text_list
    
    def get_text_stats(self, texts: List[str]) -> dict:
        """Get statistics for text batch"""
        if not texts:
            return {"count": 0, "total_chars": 0, "total_estimated_tokens": 0}
        
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        
        return {
            "count": len(texts),
            "total_chars": total_chars,
            "avg_chars": round(total_chars / len(texts), 1),
            "total_estimated_tokens": total_tokens,
            "avg_estimated_tokens": round(total_tokens / len(texts), 1),
            "max_chars": max(len(text) for text in texts),
            "min_chars": min(len(text) for text in texts)
        }


# Global validator instance
text_validator = TextValidator()


def validate_embedding_input(texts: Union[str, List[str]]) -> List[str]:
    """Convenience function to validate embedding input"""
    return text_validator.validate_texts(texts)


def get_input_stats(texts: Union[str, List[str]]) -> dict:
    """Get statistics for input texts"""
    normalized_texts = text_validator.validate_texts(texts)
    return text_validator.get_text_stats(normalized_texts)
