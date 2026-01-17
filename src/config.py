from typing import Optional
import os
from dotenv import load_dotenv


class AppConfig:
    """Centralized configuration management for the application."""

    def __init__(self) -> None:
        """Load and validate all configuration from environment variables."""
        load_dotenv()

        self.nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")

        self.faiss_persist_directory: str = os.getenv(
            "FAISS_PERSIST_DIR", "./faiss_index"
        )

        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

        self.chat_cache_ttl: int = int(os.getenv("CHAT_CACHE_TTL", "3600"))  # 1 hour
        self.embedding_cache_ttl: int = int(
            os.getenv("EMBEDDING_CACHE_TTL", "7200")
        )  # 2 hours
        self.max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "500"))

        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_backoff_min: int = int(os.getenv("RETRY_BACKOFF_MIN", "4"))
        self.retry_backoff_max: int = int(os.getenv("RETRY_BACKOFF_MAX", "10"))

    @property
    def faiss_config(self) -> dict:
        """Get FAISS configuration."""
        return {"persist_directory": self.faiss_persist_directory}


# Lazy-loaded global configuration instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance (lazy-loaded)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


# For backward compatibility
config = get_config()
