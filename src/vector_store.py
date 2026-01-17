import os
import logging
from typing import List, Any, Optional


from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

try:
    from pinecone import Pinecone as PineconeClient
    from pinecone.grpc import PineconeGRPC as PineconeGRPCClient
except ImportError:
    PineconeClient = None
    PineconeGRPCClient = None

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
except ImportError:
    NVIDIAEmbeddings = None


class FallbackEmbeddings:
    """Simple fallback embeddings that don't require API calls."""

    def __init__(self):
        self.dimension = 384

    def embed_documents(self, texts):
        """Return dummy embeddings for documents."""
        import numpy as np

        return [np.random.rand(self.dimension).tolist() for _ in texts]

    def embed_query(self, text):
        """Return dummy embedding for query."""
        import numpy as np

        return np.random.rand(self.dimension).tolist()



from .config import config

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Pinecone-based vector store for document storage and retrieval."""

    def __init__(self, embeddings: Any):
        if not PineconeClient:
            raise ImportError("pinecone package is required for Pinecone vector store")
        self.embeddings = embeddings
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "__default__")
        if not self.api_key or not self.index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set in environment")
        self.client = PineconeClient(api_key=self.api_key)
        self.index = self.client.Index(self.index_name)

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        if not documents:
            logger.warning("No documents provided to add to Pinecone")
            return

        vectors = []
        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids and i < len(ids) else f"doc-{i}"
            embedding = self.embeddings.embed_query(doc.page_content)
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": doc.metadata or {},
            })
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors, namespace=self.namespace)
        logger.info(f"Upserted {len(vectors)} documents to Pinecone index {self.index_name}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            namespace=self.namespace,
            vector=embedding,
            top_k=k,
            include_metadata=True,
            include_values=False,
        )
        matches = results.get("matches", [])
        docs = []
        for match in matches:
            page_content = match.get("metadata", {}).get("chunk_text") or ""
            metadata = match.get("metadata", {})
            docs.append(Document(page_content=page_content, metadata=metadata))
        logger.info(f"Pinecone returned {len(docs)} matches for query")
        return docs

    def as_retriever(self, **kwargs) -> Any:
        class Retriever:
            def __init__(self, store, k):
                self.store = store
                self.k = k
            def get_relevant_documents(self, query):
                return self.store.similarity_search(query, k=self.k)
        k = kwargs.get("search_kwargs", {}).get("k", 4) if "search_kwargs" in kwargs else 4
        return Retriever(self, k)

    def delete(self, ids: List[str]) -> None:
        self.index.delete(ids=ids, namespace=self.namespace)
        logger.info(f"Deleted {len(ids)} documents from Pinecone index {self.index_name}")

    def clear_index(self):
        # Pinecone does not support clearing all docs in namespace directly; delete by filter or recreate index
        logger.warning("Clear index not implemented for Pinecone. Delete by filter or recreate index manually.")

    @property
    def document_count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats.get("namespaces", {}).get(self.namespace, {}).get("vector_count", 0)

class FAISSVectorStore:
    """FAISS-based vector store for document storage and retrieval."""

    def __init__(self, embeddings: Any):
        """Initialize FAISS vector store with embeddings."""
        if not embeddings:
            raise ValueError("Embeddings model is required")

        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.persist_directory = "./faiss_index"

        os.makedirs(self.persist_directory, exist_ok=True)

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new empty one."""
        try:
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Loaded existing FAISS index from {self.persist_directory}")
        except Exception as e:
            logger.info(f"No existing FAISS index found, creating new one: {e}")
            # Create empty index with proper dimensionality
            # We'll initialize it when first documents are added
            self.vector_store = None

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the FAISS vector store."""
        if not documents:
            logger.warning("No documents provided to add")
            return

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created new FAISS index with {len(documents)} documents")
            else:
                self.vector_store.add_documents(documents, ids=ids)
                logger.info(f"Added {len(documents)} documents to existing FAISS index")

            self._save_index()

        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise

    def _save_index(self):
        """Save the FAISS index to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(self.persist_directory)
                logger.debug(f"Saved FAISS index to {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents using FAISS."""
        if self.vector_store is None:
            logger.warning("No documents in FAISS index yet")
            return []

        if isinstance(self.embeddings, FallbackEmbeddings):
            logger.info(
                "Using fallback embeddings - returning empty similarity results"
            )
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(
                f"Found {len(results)} similar documents for query: {query[:50]}..."
            )
            return results
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            return []

    def as_retriever(self, **kwargs) -> Any:
        """Return FAISS retriever interface."""
        if self.vector_store is None:
            logger.warning("No FAISS index available for retriever")
            return None

        try:
            return self.vector_store.as_retriever(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create FAISS retriever: {e}")
            return None

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs from FAISS (limited support)."""
        logger.warning(
            "FAISS delete operation not fully supported - recreating index recommended"
        )
        # FAISS doesn't easily support deletion, so we'll log a warning
        # In production, recreate the index

    def clear_index(self):
        """Clear the FAISS index completely."""
        try:
            import shutil

            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.vector_store = None
            logger.info("Cleared FAISS index")
        except Exception as e:
            logger.error(f"Failed to clear FAISS index: {e}")

    @property
    def document_count(self) -> int:
        """Get the number of documents in the index."""
        if self.vector_store and hasattr(self.vector_store, "index"):
            try:
                return self.vector_store.index.ntotal
            except:
                return 0
        return 0

_vector_store_instance: Optional[Any] = None

def get_vector_store(embeddings: Any) -> Any:
    """Get or create the global vector store instance (FAISS or Pinecone)."""
    global _vector_store_instance
    store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
    if _vector_store_instance is not None:
        return _vector_store_instance
    if store_type == "pinecone":
        _vector_store_instance = PineconeVectorStore(embeddings)
    else:
        _vector_store_instance = FAISSVectorStore(embeddings)
    return _vector_store_instance
