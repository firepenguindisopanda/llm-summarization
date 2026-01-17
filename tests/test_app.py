"""Tests for the LLM Summarization application."""

import pytest
from unittest.mock import Mock, patch
from src.config import config


class TestConfig:
    """Test configuration management."""

    @patch.dict(
        "os.environ", {"NVIDIA_API_KEY": "test_key", "VECTOR_STORE_TYPE": "chroma"}
    )
    def test_config_initialization(self):
        """Test config loads correctly."""
        from src.config import AppConfig

        config = AppConfig()

        assert config.nvidia_api_key == "test_key"
        assert config.vector_store_type == "chroma"
        assert not config.is_production

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.config.load_dotenv")
    def test_config_validation_missing_api_key(self, mock_load_dotenv):
        """Test config fails without NVIDIA API key."""
        mock_load_dotenv.return_value = None

        # Reset the global config instance
        import src.config

        src.config._config_instance = None

        from src.config import AppConfig

        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            AppConfig()


class TestPDFProcessor:
    """Test PDF processing functionality."""

    def test_pdf_processor_initialization(self):
        """Test PDF processor initializes correctly."""
        try:
            from src.pdf_processor import PDFProcessor

            processor = PDFProcessor()

            assert processor.chunk_size == config.chunk_size
            assert processor.chunk_overlap == config.chunk_overlap
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_chunk_text(self):
        """Test text chunking."""
        try:
            from src.pdf_processor import pdf_processor

            text = "This is a test document. " * 100  # Long text
            chunks = pdf_processor.chunk_text(text)

            assert len(chunks) > 1
            assert all(len(chunk) <= config.chunk_size for chunk in chunks)
        except ImportError:
            pytest.skip("pypdf not installed")


class TestLLMIntegration:
    """Test LLM integration (mocked)."""

    def test_llm_module_import(self):
        """Test that LLM integration module can be imported."""
        try:
            from src import llm_integration
            assert hasattr(llm_integration, "CachedNVIDIAClient")
            assert hasattr(llm_integration, "llm_client")
        except ImportError as e:
            if "langchain-nvidia-ai-endpoints" in str(e):
                pytest.skip("NVIDIA endpoints package not installed")
            else:
                raise


class TestVectorStore:
    """Test vector store functionality."""

    @patch("src.vector_store.Chroma")
    def test_chroma_vector_store(self, mock_chroma):
        """Test Chroma vector store initialization."""
        from src.vector_store import ChromaVectorStore
        from unittest.mock import Mock

        mock_embeddings = Mock()
        config_dict = {"persist_directory": "./test_db", "collection_name": "test"}

        store = ChromaVectorStore(mock_embeddings, config_dict)

        assert store.embeddings == mock_embeddings
        mock_chroma.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
