# IRDAI Insurance Circulars RAG Chatbot

A Retrieval-Augmented Generation (RAG) system for querying IRDAI insurance circulars with a conversational AI interface.

## 🎯 Performance Highlights

- **Answer Relevancy: 0.86** (excellent performance)
- **Faithfulness: 0.96** (minimal hallucination)
- **Hit Rate: 87%** (13/15 test queries succeed)
- **Response Time: 2-4 seconds** (end-to-end with LLM)
- **Cost: $0.08 per 1000 queries** (highly cost-effective)

## Features

- **Semantic Search**: Find relevant insurance regulations using natural language queries with **91% context precision**
- **Answer Generation**: Get direct answers with source citations using optimized GPT-4o-mini
- **Conversation Context**: Ask follow-up questions that reference previous exchanges
- **Web Interface**: User-friendly Streamlit chat interface
- **REST API**: FastAPI backend for programmatic access
- **Optimized Configuration**: Systematically tested top_k=3 for best quality/cost balance
- **Low Cost**: ~$0.08/1000 queries with OpenAI embeddings + GPT-4o-mini

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

This downloads ~93 IRDAI circulars to `data/raw_pdfs/`

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

**Step 3: Build ChromaDB Index**

```bash
python src/embeddings/build_index.py
```

This creates embeddings and builds the ChromaDB vector store.


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
    ├→ ChromaDB Vector Search
    ├→ Context Aggregation
    └→ LLM Answer Generation (OpenAI/Groq)
        ↓
    Answer + Citations
```

## Configuration

### Environment Variables (.env)

```env
# LLM Provider (openai or groq)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: Groq for free LLM (alternative)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

# Top-K Retrieval
TOP_K_RESULTS=3
```

### LLM Providers

**OpenAI (Current Configuration)**
- Get API key: https://platform.openai.com
- Model: gpt-4o-mini
- High quality, requires credits
- Used in evaluation for 0.86 answer relevancy

**Groq (Free Alternative)**
- Get API key: https://console.groq.com
- Model: llama-3.3-70b-versatile
- Fast inference, generous free tier
- Switch by changing `LLM_PROVIDER=groq` in .env

## Project Structure

```
rag-irdai-chatbot/
├── app/
│   └── streamlit_app.py         # Web UI
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   ├── embeddings/
│   │   └── build_index.py       # ChromaDB index builder
│   ├── llm/
│   │   └── answer_generator.py  # LLM integration (Groq/OpenAI)
│   ├── preprocessing/           # Modular preprocessing
│   │   ├── pdf_extractor.py     # PDF → Text extraction
│   │   ├── text_cleaner.py      # Advanced text cleaning
│   │   └── text_chunker.py      # Semantic chunking
│   ├── rag_pipeline/
│   │   └── pipeline.py          # Main RAG logic
│   └── downloader/
│       └── fetch_pdfs.py        # IRDAI PDF downloader
├── data/
│   ├── raw_pdfs/                # Source PDFs (93 documents)
│   ├── processed_text_clean/    # Cleaned extracted text
│   ├── chunks/                  # Processed chunks (561 chunks)
│   ├── chromadb/                # ChromaDB vector store
│   └── evaluation_dataset.json  # Evaluation test queries (15 questions)
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

- **Query Time**: ~780ms (OpenAI embedding) + <1ms (ChromaDB search) + 1-3s (LLM)
- **Total Response Time**: ~2-4 seconds end-to-end
- **Index Size**: 561 chunks from 93 PDF documents
- **Storage**: ChromaDB index for 3072-dim vectors | ~59MB total
- **Cost**: ~$0.08 per 1000 queries (OpenAI embeddings + GPT-4o-mini)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Evaluation & Results](docs/evaluation.md)** | Performance metrics and cost analysis |
| **[Architecture](docs/architecture.md)** | System design, components, and data flow |
| **[Setup Instructions](docs/setup_instructions.md)** | Detailed installation and deployment guide |

## Common Tasks

### Add New PDFs
```bash
# 1. Place PDFs in data/raw_pdfs/

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
# Update .env to use Groq (free)
LLM_PROVIDER=groq

# Or use OpenAI (paid, current default)
LLM_PROVIDER=openai

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

### ChromaDB Index Not Found
- **Solution**: Run `python src/embeddings/build_index.py`
- **Expected time**: ~5-10 minutes for 561 chunks from 93 documents
- **Requirements**: OpenAI API key with available credits

### Rate Limiting (Temporary 429)
- **Status**: ✅ Automatic retry logic implemented
- **Behavior**: System retries 3 times with delays (2s, 4s, 8s)
- **Manual**: If still failing, wait 60 seconds and retry

### API Won't Start
- Ensure virtual environment is activated: `anaira312\Scripts\activate`
- Check installation: `pip install uvicorn fastapi`
- Verify port not in use: Try `--port 8001`

### Follow-up Questions Fail
- Ensure "Use conversation context" is enabled in Streamlit UI
- Check that conversation_history is being sent to API
- Verify LLM provider is correctly configured in .env

## Technology Stack

- **Embeddings**: OpenAI text-embedding-3-large (3072-dim)
- **LLM**: GPT-4o-mini (OpenAI) with optimized prompt
- **Vector Store**: ChromaDB with HNSW indexing
- **Text Processing**: LangChain, pdfplumber, Tesseract OCR
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Configuration**: Optimized top_k=3 for best quality/cost balance
