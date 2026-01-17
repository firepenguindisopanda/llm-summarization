import logging
from pathlib import Path
from typing import List, Iterator, Optional, Union, Dict, Any
import io

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF text extraction and processing utilities."""

    def __init__(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ):
        """Initialize PDF processor with chunking configuration."""
        if not PdfReader:
            raise ImportError("pypdf is required for PDF processing")

        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_from_file(self, file_path: Union[str, Path]) -> str:
        """Extract text from PDF file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {file_path}")

        logger.info(f"Extracting text from {file_path}")
        with open(file_path, "rb") as file:
            file_content = io.BytesIO(file.read())
            return self.extract_text_from_stream(file_content)

    def extract_text_from_stream(self, stream: io.BytesIO) -> str:
        """Extract text from PDF stream (for uploaded files)."""
        if not PdfReader:
            raise ImportError("pypdf is required for PDF processing")

        try:
            pdf_reader = PdfReader(stream)
            text_parts = []

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_parts.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(
                        f"Failed to extract text from page {page_num + 1}: {e}"
                    )
                    continue

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF")

            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    def extract_text_from_uploaded_file(self, uploaded_file) -> str:
        """Extract text from Streamlit uploaded file object."""
        try:
            # Create a BytesIO object from the uploaded file
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            file_bytes.seek(0)  # Reset to beginning

            logger.info(f"Processing uploaded PDF: {uploaded_file.name}")
            return self.extract_text_from_stream(file_bytes)

        except Exception as e:
            logger.error(f"Failed to process uploaded PDF {uploaded_file.name}: {e}")
            raise RuntimeError(f"Failed to process uploaded PDF: {e}") from e

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        if not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def chunk_text_with_metadata(self, text: str, source: str = "unknown") -> List[Any]:
        """Split text into chunks with metadata."""
        from langchain_core.documents import Document

        chunks = self.chunk_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                },
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} documents with metadata")
        return documents

    def validate_pdf(self, file) -> bool:
        """Validate if file is a readable PDF."""
        if not PdfReader:
            return False

        # Try to read the PDF
        try:
            if hasattr(file, "type") and file.type != "application/pdf":
                return False

            file_bytes = (
                io.BytesIO(file.getvalue()) if hasattr(file, "getvalue") else file
            )
            pdf_reader = PdfReader(file_bytes)

            return len(pdf_reader.pages) > 0

        except Exception as e:
            logger.warning(f"PDF validation failed: {e}")
            return False

    def get_pdf_info(self, file) -> Dict[str, Any]:
        """Get basic information about a PDF file."""
        if not PdfReader:
            return {"pages": 0, "metadata": {}, "error": "pypdf not available"}

        try:
            file_bytes = (
                io.BytesIO(file.getvalue()) if hasattr(file, "getvalue") else file
            )
            pdf_reader = PdfReader(file_bytes)

            info = {"pages": len(pdf_reader.pages), "metadata": {}}

            # Extract metadata if available
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if value: 
                        info["metadata"][key] = str(value)

            return info

        except Exception as e:
            logger.warning(f"Failed to get PDF info: {e}")
            return {"pages": 0, "metadata": {}, "error": str(e)}


pdf_processor = PDFProcessor()
