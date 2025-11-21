# IRDAI Insurance Circulars RAG Chatbot

A Retrieval-Augmented Generation (RAG) system for querying IRDAI insurance circulars with a conversational AI interface.

## Features

- **Semantic Search**: Find relevant insurance regulations using natural language queries
- **Answer Generation**: Get direct answers with source citations using OpenAI/Groq
- **Conversation Context**: Ask follow-up questions that reference previous exchanges
- **Web Interface**: User-friendly Streamlit chat interface
- **REST API**: FastAPI backend for programmatic access
- **Near-Zero Cost**: ~$0.12/year with OpenAI embeddings + Groq LLM | 100% Free: Use local embeddings alternative

## Quick Start

### Prerequisites

- Python 3.12+
- Tesseract OCR (for PDF processing)
- OpenAI or Groq API key (for answer generation)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd rag-irdai-chatbot

# 2. Create virtual environment
python -m venv anaira312
anaira312\Scripts\activate  # Windows
source anaira312/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Data Pipeline (One-Time Setup)

#### Full Pipeline from Scratch

**Step 1: Fetch PDFs from IRDAI Website**

```bash
python src/downloader/fetch_pdfs.py
```

This downloads ~93 IRDAI circulars to `data/raw_downloaded_pdfs/`

**Step 2: Preprocess PDFs**

Use the modular preprocessing pipeline:

```bash
cd src/preprocessing

# Run each module in sequence
python pdf_extractor.py    # Step 1: Extract text from PDFs
python text_cleaner.py     # Step 2: Clean and filter text
python text_chunker.py     # Step 3: Create semantic chunks
```

This performs:
- **pdf_extractor.py**: Text extraction using pdfplumber + Tesseract OCR
- **text_cleaner.py**: Advanced cleaning (remove Hindi, metadata extraction)
- **text_chunker.py**: Semantic chunking (800-token chunks with overlap)

Output: `data/chunks/` directory with JSON files containing text chunks

> **Note**: Each module can also be imported and used programmatically. See [src/preprocessing/README.md](src/preprocessing/README.md) for details.

**Step 3: Build FAISS Index**

```bash
python src/embeddings/build_index.py
```

This creates embeddings and builds the vector store.


### Run the Application

#### Web Interface

```bash
# Start API server (Terminal 1)
cd src/api
uvicorn main:app --reload

# Start Streamlit UI (Terminal 2)
streamlit run app/streamlit_app.py
```

Access at: http://localhost:8501


## System Architecture

```
User Query
    ↓
Streamlit UI (conversation tracking)
    ↓
FastAPI Backend
    ↓
RAG Pipeline
    ├→ Query Enhancement (for follow-ups)
    ├→ FAISS Vector Search
    ├→ Context Aggregation
    └→ LLM Answer Generation (OpenAI/Groq)
        ↓
    Answer + Citations
```

## Configuration

### Environment Variables (.env)

```env
# LLM Provider (openai or groq)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
# Or for OpenAI
OPENAI_API_KEY=your_openai_key_here

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-large

# Paths
VECTOR_STORE_DIR=data/vector_store
```

### LLM Providers

**Groq (Recommended - Free)**
- Get API key: https://console.groq.com
- Model: llama-3.3-70b-versatile
- Fast inference, generous free tier

**OpenAI (Paid)**
- Get API key: https://platform.openai.com
- Model: gpt-4o-mini
- High quality, requires credits

## Project Structure

```
rag-irdai-chatbot/
├── app/
│   └── streamlit_app.py         # Web UI
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   ├── embeddings/
│   │   ├── embedder.py          # Multi-provider embeddings
│   │   └── build_index.py       # FAISS index builder
│   ├── llm/
│   │   └── answer_generator.py  # LLM integration (Groq/OpenAI)
│   ├── preprocessing/           # Modular preprocessing
│   │   ├── pdf_extractor.py     # PDF → Text extraction
│   │   ├── text_cleaner.py      # Advanced text cleaning
│   │   └── text_chunker.py      # Semantic chunking
│   └── rag_pipeline/
│       └── pipeline.py          # Main RAG logic
├── data/
│   ├── raw_downloaded_pdfs/     # Source PDFs
│   ├── chunks/                  # Processed chunks
│   └── vector_store/            # FAISS index
└── docs/                        # Documentation
```

## Key Features

### Follow-up Questions
The system maintains conversation context to handle follow-up questions:

```
User: "What is the threshold for premium revision for senior citizens?"
Bot: "The threshold is 10% per annum..." [with sources]

User: "When was this decision made?"
Bot: "This decision was passed on 30th January, 2025..." [same sources]
```

### Query Enhancement
Vague follow-up questions are automatically enhanced with context from previous queries for better retrieval.

### Source Boosting
Follow-up questions prioritize documents from the previous answer to maintain conversation continuity.

## Performance

- **Query Time**: ~100-200ms (OpenAI embedding) + <1ms (FAISS search) + 1-3s (LLM)
- **Total Response Time**: ~1-3 seconds end-to-end
- **Index Size**: 501 chunks from 93 PDF documents
- **Storage**: ~6MB (FAISS index for 3072-dim vectors) | ~59MB total
- **Cost**: ~$0.12/year with OpenAI + Groq | $0/year with free alternative

## Documentation

- [Setup Instructions](docs/setup_instructions.md) - Detailed installation guide
- [Architecture](docs/architecture.md) - System design and components
- [System Updates](docs/SYSTEM_UPDATES.md) - Recent changes and improvements
- [Preprocessing Module](src/preprocessing/README.md) - Modular preprocessing guide
- [Evaluation](docs/evaluation.md) - Performance metrics and analysis

## Common Tasks

### Add New PDFs
```bash
# 1. Place PDFs in data/raw_downloaded_pdfs/

# 2. Process new PDFs
cd src/preprocessing
python pdf_extractor.py    # Extract text
python text_cleaner.py     # Clean text
python text_chunker.py     # Create chunks

# 3. Rebuild index
cd ../embeddings
python build_index.py
```

### Fetch Latest IRDAI Circulars
```bash
# Download new circulars from IRDAI website
python src/downloader/fetch_pdfs.py

# Then preprocess and rebuild index (see "Add New PDFs" above)
```

### Change LLM Provider
```bash
# Update .env
LLM_PROVIDER=groq  # or openai
GROQ_API_KEY=your_key

# Restart API server
```

### Test the System
```bash
# Test RAG pipeline
python src/rag_pipeline/pipeline.py

# Test preprocessing modules individually
cd src/preprocessing
python pdf_extractor.py    # Test PDF extraction
python text_cleaner.py     # Test text cleaning
python text_chunker.py     # Test chunking
```

## Troubleshooting

### API won't start
- Ensure virtual environment is activated
- Check `pip install uvicorn fastapi`

### FAISS index not found
- Run: `python src/embeddings/build_index.py`

### LLM errors (429 rate limit)
- Switch to Groq (higher free tier)
- Or add retry delays (already implemented)

### Follow-up questions fail
- Ensure "Use conversation context" is enabled in Streamlit
- Check that conversation_history is being sent to API

Built with:
- **Embeddings**: OpenAI API (text-embedding-3-large) | Alternative: sentence-transformers (local)
- **Vector Search**: FAISS (Facebook AI)
- **Text Processing**: LangChain, pdfplumber, Tesseract OCR
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **LLM**: Groq (free) or OpenAI (paid)
