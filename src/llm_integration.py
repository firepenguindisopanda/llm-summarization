import logging
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from cachetools import TTLCache

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
except ImportError:
    ChatNVIDIA = None
    NVIDIAEmbeddings = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config

logger = logging.getLogger(__name__)


class CachedNVIDIAClient:
    """NVIDIA AI endpoints client with caching and rate limiting."""

    def __init__(self):
        """Initialize the client with caching and model setup."""
        if not ChatNVIDIA or not NVIDIAEmbeddings:
            raise ImportError("langchain-nvidia-ai-endpoints is required")

        self.chat_cache = TTLCache(
            maxsize=config.max_cache_size, ttl=config.chat_cache_ttl
        )
        self.embedding_cache = TTLCache(
            maxsize=config.max_cache_size, ttl=config.embedding_cache_ttl
        )

        self.available_models = self._get_available_models()
        self.chat_model = self._initialize_chat_model()
        self.embeddings = self._initialize_embeddings()

        logger.info(
            f"Initialized CachedNVIDIAClient with {len(self.available_models)} available models"
        )

    def _get_available_models(self) -> List[str]:
        """Get list of available NVIDIA models."""
        try:
            models = ChatNVIDIA.get_available_models() or []
            processed = []
            for m in models:
                if isinstance(m, str):
                    processed.append(m)
                elif isinstance(m, dict):
                    for k in ("id", "model", "name", "model_id"):
                        v = m.get(k)
                        if v:
                            processed.append(str(v))
                            break
                else:
                    v = getattr(m, "id", None) or getattr(m, "name", None)
                    if v:
                        processed.append(str(v))
                    else:
                        processed.append(str(m))

            seen = set()
            result = [x for x in processed if not (x in seen or seen.add(x))]
            return result
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            return []

    def _initialize_chat_model(self):
        """Initialize the chat model with first available model or fallback."""
        preferred_models = [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.2-3b-instruct",
            "mistralai/mistral-7b-instruct-v0.2",
            "microsoft/phi-3-mini-4k-instruct",
        ]

        for preferred in preferred_models:
            if preferred in self.available_models:
                model_name = preferred
                logger.info(f"Using preferred model: {model_name}")
                break
        else:
            model_name = (
                self.available_models[0]
                if self.available_models
                else "meta/llama-3.1-8b-instruct"
            )
            logger.info(f"Using fallback model: {model_name}")

        try:
            return ChatNVIDIA(model=model_name, temperature=0.7, max_tokens=2048)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            try:
                return ChatNVIDIA(
                    model="meta/llama-3.1-8b-instruct", temperature=0.7, max_tokens=2048
                )
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {fallback_error}")
                raise RuntimeError(
                    f"Unable to initialize any chat model: {fallback_error}"
                ) from fallback_error

    def _initialize_embeddings(self):
        """Initialize the embeddings model with fallback."""
        logger.info("Initializing NVIDIAEmbeddings with NV-Embed-QA")

        if NVIDIAEmbeddings:
            try:
                embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")
                test_result = embeddings.embed_query("test")
                if test_result and len(test_result) > 0:
                    logger.info("NVIDIA embeddings initialized successfully")
                    return embeddings
                else:
                    logger.warning("NVIDIA embeddings test failed - using fallback")
            except Exception as e:
                logger.warning(f"NVIDIA embeddings failed: {e} - using fallback")

        logger.info("Using fallback embeddings (no actual similarity search)")
        from .vector_store import FallbackEmbeddings

        return FallbackEmbeddings()

    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(
            multiplier=1, min=config.retry_backoff_min, max=config.retry_backoff_max
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke_chat_model(
        self, messages: List[Union[SystemMessage, HumanMessage]], **kwargs
    ) -> str:
        """Invoke chat model with retry logic."""
        try:
            response = self.chat_model.invoke(messages, **kwargs)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Chat model invocation failed: {e}")
            raise

    def summarize_with_cache(
        self,
        text: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Summarize text with caching and rate limiting."""
        cache_key = hash((text, system_prompt or "", temperature, max_tokens))

        if cache_key in self.chat_cache:
            logger.info("Returning cached summary")
            return self.chat_cache[cache_key]

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(
            HumanMessage(content=f"Please summarize the following text:\n\n{text}")
        )

        try:
            summary = self._invoke_chat_model(
                messages, temperature=temperature, max_tokens=max_tokens
            )

            self.chat_cache[cache_key] = summary
            logger.info("Generated and cached new summary")

            return summary
        except Exception as e:
            error_msg = f"Summarization failed: {e}"
            logger.error(error_msg)
            current_model = getattr(self.chat_model, "model", "unknown")
            detailed_error = f"{error_msg} (using model: {current_model})"
            raise RuntimeError(detailed_error) from e

    def embed_with_cache(self, text: str) -> List[float]:
        """Generate embeddings with caching."""
        cache_key = hash(text)

        if cache_key in self.embedding_cache:
            logger.debug("Returning cached embedding")
            return self.embedding_cache[cache_key]

        try:
            embedding = self.embeddings.embed_query(text)
            self.embedding_cache[cache_key] = embedding
            logger.debug("Generated and cached new embedding")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def get_available_models(self) -> List[str]:
        """Get list of available models (public interface)."""
        return self.available_models.copy()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.chat_cache.clear()
        self.embedding_cache.clear()
        logger.info("Cleared all caches")

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "chat_cache_size": len(self.chat_cache),
            "chat_cache_maxsize": self.chat_cache.maxsize,
            "embedding_cache_size": len(self.embedding_cache),
            "embedding_cache_maxsize": self.embedding_cache.maxsize,
        }

llm_client = CachedNVIDIAClient()
