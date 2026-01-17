# PDF Summarization App with NVIDIA AI

This app lets you upload a PDF and get a summary using NVIDIA's large language models. You can use a direct LLM call or retrieval-augmented generation (RAG) with a vector store. The app runs in your browser with Streamlit.

## How It Works

- Upload a PDF file.
- Choose a summarization mode: Direct (fast, simple) or RAG (uses document search).
- Select a model and adjust settings if needed.
- Get a summary and download or copy the result.

## Features

- Summarize PDFs with NVIDIA LLMs.
- RAG mode uses FAISS for local vector search.
- Adjustable model, temperature, max tokens, and chunk size.
- Caching and rate limiting to control API usage.
- Handles large files with chunking and streaming.

## Requirements

- uv installed
- Python 3.10 or newer
- NVIDIA API key (get one from NVIDIA)

## Install

1. Clone this repository.
2. Install dependencies:
	- Recommended: `uv pip install -e .`
	- Or: `pip install -e .`
3. For development tools: `uv pip install -e .[dev]` or `pip install -e .[dev]`

## Setup

1. Create a `.env` file in the project folder.
2. Add your NVIDIA API key:
	NVIDIA_API_KEY=your_nvidia_api_key
3. (Optional) For Pinecone vector store, add:
	PINECONE_API_KEY=your_pinecone_api_key
	PINECONE_INDEX=your_pinecone_index_name
4. (Optional) Set VECTOR_STORE_TYPE=chroma or pinecone

## Run the App

Run this command in your terminal:
	 `streamlit run streamlit_app.py`
To use a custom port:
	 `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --reload`

## Usage

1. Open the app in your browser.
2. Upload a PDF.
3. Pick a model and mode.
4. Click Generate Summary.
5. Download or copy the summary.

## Testing and Linting

- Run all tests: `pytest`
- Lint code: `ruff check .`
- Format code: `black .`
- Type check: `mypy . --ignore-missing-imports`

## Project Structure

src/ # Core modules (config, LLM, PDF, vector store)
streamlit_app.py # Main app
tests/ # Tests
examples/ # Example files

## Security

- Only PDF files are accepted.
- File size and type are checked.
- API keys are loaded from environment variables.

## Troubleshooting

- If no models appear, check your NVIDIA API key.
- For RAG mode, make sure embeddings are available.
- Clear caches in the sidebar if results seem stale.

## License

See LICENSE file if present.


## Limitations

- large pdfs take a long time to summarize and sometimes the application restarts, so it begins extracting process all over again, it extracts then summarizes
