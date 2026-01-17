"""Main Streamlit application for PDF document summarization.

Provides a user interface for uploading PDFs and generating summaries using
both direct LLM calls and RAG (Retrieval-Augmented Generation) with vector stores.
"""

import logging
import time
from typing import Optional, Dict, Any

import streamlit as st
from langchain_core.documents import Document

from src.config import config
from src.llm_integration import llm_client
from src.pdf_processor import pdf_processor
from src.vector_store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="PDF Summarization with NVIDIA AI", page_icon="📄", layout="wide"
    )

    st.title("📄 PDF Document Summarization")
    st.markdown("*Powered by NVIDIA AI Endpoints*")


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar configuration and return settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        available_models = llm_client.get_available_models()
        current_model = getattr(llm_client.chat_model, "model", "unknown")

        if available_models:
            # Show current model being used
            st.info(f"🔧 **Current Model:** {current_model}")

            selected_model = st.selectbox(
                "Chat Model",
                available_models,
                index=available_models.index(current_model)
                if current_model in available_models
                else 0,
                help="Select the NVIDIA chat model to use for summarization",
            )

            # Allow user to switch model
            if selected_model != current_model:
                if st.button(f"🔄 Switch to {selected_model}"):
                    try:
                        llm_client.chat_model = llm_client.chat_model.__class__(
                            model=selected_model, temperature=0.7, max_tokens=2048
                        )
                        st.success(f"✅ Switched to model: {selected_model}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to switch model: {e}")
        else:
            st.warning("⚠️ No models available. Check your NVIDIA API key.")
            selected_model = None

        # Processing mode
        # Check if embeddings are working (not fallback)
        from src.vector_store import FallbackEmbeddings

        embeddings_available = not isinstance(llm_client.embeddings, FallbackEmbeddings)

        if embeddings_available:
            mode = st.radio(
                "Processing Mode",
                ["Direct", "RAG"],
                help="Direct: Simple LLM summarization | RAG: Retrieval-augmented generation",
            )
        else:
            mode = "Direct"
            st.info(
                "ℹ️ **RAG mode disabled** - NVIDIA embeddings not available. Using Direct mode only."
            )
            st.radio(
                "Processing Mode",
                ["Direct", "RAG"],
                index=0,
                disabled=True,
                help="RAG requires NVIDIA embeddings which are not available for this account",
            )

        # Vector store info
        import os
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        if vector_store_type == "pinecone":
            st.info("🗄️ Using **Pinecone** (Cloud Vector Store)")
        else:
            st.info("🗄️ Using **FAISS** (Local Vector Store)")

        # Parameters
        with st.expander("Advanced Parameters", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in generation (0.0 = deterministic, 1.0 = creative)",
            )

            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                help="Maximum length of the summary",
            )

            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=config.chunk_size,
                step=100,
                help="Size of text chunks for processing",
            )

        # Cache management
        with st.expander("Cache Management", expanded=False):
            cache_stats = llm_client.cache_stats
            st.write("**Cache Statistics:**")
            st.json(cache_stats)

            if st.button("Clear All Caches"):
                llm_client.clear_cache()
                st.success("Caches cleared!")
                st.rerun()

        return {
            "selected_model": selected_model,
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chunk_size": chunk_size,
        }


def handle_file_upload() -> Optional[Any]:
    """Handle PDF file upload and validation."""
    uploaded_file = st.file_uploader(
        "Upload PDF Document", type=["pdf"], help="Select a PDF file to summarize"
    )

    if uploaded_file:
        # Validate PDF
        if not pdf_processor.validate_pdf(uploaded_file):
            st.error("Invalid PDF file. Please upload a valid PDF document.")
            return None

        # Show PDF info
        pdf_info = pdf_processor.get_pdf_info(uploaded_file)
        with st.expander("📊 PDF Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", pdf_info.get("pages", 0))
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            if pdf_info.get("metadata"):
                st.write("**Metadata:**")
                st.json(pdf_info["metadata"])

        return uploaded_file

    return None


def process_direct_mode(text: str, settings: Dict[str, Any]) -> Optional[str]:
    """Process document using direct LLM summarization."""
    with st.spinner("Generating summary with direct LLM call..."):
        try:
            summary = llm_client.summarize_with_cache(
                text=text,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
            return summary
        except Exception as e:
            st.error(f"Direct summarization failed: {e}")
            return None


def process_rag_mode(text: str, settings: Dict[str, Any]) -> Optional[str]:
    """Process document using RAG (Retrieval-Augmented Generation)."""
    with st.spinner("Processing document with RAG..."):
        try:
            # Chunk the text
            documents = pdf_processor.chunk_text_with_metadata(text, "uploaded_pdf")

            # Get vector store
            vector_store = get_vector_store(llm_client.embeddings)

            # Add documents to vector store
            vector_store.add_documents(documents)

            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})

            # Perform RAG query
            # For now, use a simple approach - get relevant chunks and summarize
            try:
                if retriever and hasattr(retriever, "get_relevant_documents"):
                    relevant_docs = retriever.get_relevant_documents(text[:500])
                elif retriever and hasattr(retriever, "invoke"):
                    relevant_docs = retriever.invoke(text[:500])
                else:
                    # Fallback to similarity search on the vector store directly
                    relevant_docs = vector_store.similarity_search(text[:500], k=4)
            except Exception as retriever_error:
                logger.warning(
                    f"Retriever failed, falling back to similarity search: {retriever_error}"
                )
                relevant_docs = vector_store.similarity_search(text[:500], k=4)

            # Combine relevant content
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Generate summary with context
            system_prompt = (
                "You are an expert at synthesizing information from document chunks. "
                "Create a coherent summary that captures the main ideas from the provided context."
            )

            summary = llm_client.summarize_with_cache(
                text=context,
                system_prompt=system_prompt,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )

            return summary

        except Exception as e:
            st.error(f"RAG processing failed: {e}")
            return None


def display_results(summary: str, processing_time: float, mode: str):
    """Display the summarization results."""
    if summary:
        st.success(
            f"✅ Summary generated in {processing_time:.2f} seconds using **{mode}** mode"
        )

        st.subheader("📝 Summary")
        st.write(summary)

        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="document_summary.txt",
            mime="text/plain",
        )

        # Copy to clipboard (using text area for easy copying)
        st.text_area(
            "Copy summary",
            value=summary,
            height=200,
            help="Copy the summary text from here",
        )
    else:
        st.error("Failed to generate summary. Please try again.")


def main():
    """Main application entry point."""
    setup_page()

    # Render sidebar and get settings
    settings = render_sidebar()

    # File upload
    uploaded_file = handle_file_upload()

    if uploaded_file and settings["selected_model"]:
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            try:
                text = pdf_processor.extract_text_from_uploaded_file(uploaded_file)
                st.success(f"✅ Extracted {len(text)} characters from PDF")
            except Exception as e:
                st.error(f"Failed to extract text: {e}")
                return

        # Show text preview
        with st.expander("📄 Extracted Text Preview", expanded=False):
            st.text_area(
                "Text Content",
                value=text[:2000] + ("..." if len(text) > 2000 else ""),
                height=200,
                disabled=True,
            )

        # Process button
        if st.button("🚀 Generate Summary", type="primary"):
            start_time = time.time()

            if settings["mode"] == "Direct":
                summary = process_direct_mode(text, settings)
            else:  # RAG mode
                summary = process_rag_mode(text, settings)

            processing_time = time.time() - start_time

            # Display results
            display_results(summary, processing_time, settings["mode"])

    elif not settings["selected_model"]:
        st.warning(
            "⚠️ No chat models available. Please check your NVIDIA API key configuration."
        )

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and NVIDIA AI Endpoints*")


if __name__ == "__main__":
    main()
