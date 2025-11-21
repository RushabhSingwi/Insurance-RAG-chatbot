# Setup Instructions

Quick guide to setting up and running the IRDAI Insurance Circulars RAG system.

## Prerequisites

- **Python**: 3.12 or higher
- **Tesseract OCR**: For processing scanned PDFs
- **Internet**: Required for initial setup only

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag-irdai-chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv anaira312
anaira312\Scripts\activate

# Linux/Mac
python3.12 -m venv anaira312
source anaira312/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 5. Configure Environment

Create `.env` file:

```env
# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
# Or use OpenAI
# OPENAI_API_KEY=your_openai_key_here

# Embedding Provider and Model (current: OpenAI)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=your_openai_key_here

# Alternative: Free Local Embeddings (uncomment to use)
# EMBEDDING_PROVIDER=sentence-transformers
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store Location
VECTOR_STORE_DIR=data/vector_store
```

**Get API Keys:**
- Groq (Free): https://console.groq.com
- OpenAI (Paid): https://platform.openai.com

## Building the Index

### Quick Start (Using Existing Chunks)

If `data/chunks/` already exists:

```bash
python src/embeddings/build_index.py
```

Expected time: ~1-2 minutes (with OpenAI API)

### Full Pipeline (From Scratch)

If starting from scratch:

**Step 1: Fetch PDFs (~5-10 min)**

```bash
python src/downloader/fetch_pdfs.py
```

Downloads ~93 IRDAI circulars to `data/raw_downloaded_pdfs/`

**Step 2: Preprocess PDFs (~10-15 min)**

Run the modular preprocessing pipeline:

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

**Step 3: Build Index (~1-2 min)**

```bash
cd ../..
python src/embeddings/build_index.py
```

**Total Time**: ~12-18 minutes for complete pipeline

## Running the Application

### Option 1: Web Interface (Recommended)

**Terminal 1 - Start API:**
```bash
cd src/api
uvicorn main:app --reload
```

**Terminal 2 - Start Streamlit:**
```bash
streamlit run app/streamlit_app.py
```

Access at: http://localhost:8501

### Option 2: Command Line

```bash
python src/rag_pipeline/pipeline.py
```

This will run example queries and enter interactive mode.

## Common Issues

### "FAISS index not found"

**Solution:**
```bash
python src/embeddings/build_index.py
```

### "Tesseract not found"

**Solution (Windows):**
Add to PATH or set in .env:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Port already in use

**Streamlit:**
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

**FastAPI:**
```bash
uvicorn main:app --port 8001
```

### LLM Rate Limits (OpenAI 429 error)

**Solutions:**
1. Switch to Groq (higher free tier)
2. Wait 60 seconds between queries
3. Upgrade OpenAI tier ($5 → 500 RPM)

## Verification

Test the installation:

```bash
# Test imports
python -c "import faiss; import openai; print('✅ All packages installed')"

# Test RAG pipeline
python src/rag_pipeline/pipeline.py

# Test API (if running)
curl http://localhost:8000/
```

## Next Steps

1. **Add Documents**: Place PDFs in `data/raw_downloaded_pdfs/` and reprocess
2. **Customize Settings**: Edit `.env` for different models or providers
3. **Deploy**: Set up production server with gunicorn/nginx

## Performance

- **Build Time**: ~12-18 minutes (full pipeline)
- **Query Time**: ~100-200ms (OpenAI embedding) + <1ms (FAISS) + 1-3s (LLM)
- **Storage**: ~59MB total (6MB FAISS index for 3072-dim vectors)
- **Memory**: ~400MB peak (no local model loading with OpenAI API)
- **Cost**: ~$0.02 one-time (embeddings) + $0/session (Groq LLM)
