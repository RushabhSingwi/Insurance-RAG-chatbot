# System Architecture

## Overview

The IRDAI Insurance Circulars RAG system is a conversational AI chatbot built using a modular, pipeline-based architecture. The system processes bilingual PDF documents, extracts English text, generates embeddings, and provides semantic search with LLM-powered answer generation. Users interact through a Streamlit web interface that maintains conversation context for follow-up questions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG CHATBOT SYSTEM                           │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Data      │───→│  Processing  │───→│  Embeddings  │
  │ Collection  │    │   Pipeline   │    │ & Indexing   │
  └─────────────┘    └──────────────┘    └──────┬───────┘
                                                 │
                                                 ▼
                                        ┌──────────────┐
                                        │   ChromaDB   │
                                        │ Vector Store │
                                        └──────┬───────┘
                                               │
                                               ▼
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Streamlit  │───→│   FastAPI    │───→│     RAG      │
  │     UI      │    │   Backend    │    │   Pipeline   │
  └─────────────┘    └──────────────┘    └──────┬───────┘
                                                 │
                                                 ▼
                                        ┌──────────────┐
                                        │ LLM Provider │
                                        │ Groq/OpenAI  │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │    Answer    │
                                        │  + Sources   │
                                        └──────────────┘
```

## Component Architecture

### 1. Data Collection Layer

**Component**: `src/downloader/fetch_pdfs.py`

```
┌─────────────────────────────────────────────────────────┐
│              Data Collection Layer                      │
└─────────────────────────────────────────────────────────┘

  IRDAI Website
       │
       ▼
  [Web Scraper]
   • BeautifulSoup4
   • Requests
       │
       ▼
  [Filename Cleaner]
   • Removes Hindi characters
   • Extracts English names
       │
       ▼
  PDF Downloads (93 files)
  → data/raw_pdfs/
```

**Key Features**:
- Paginated scraping (10 pages, ~200 documents)
- Bilingual filename handling (Hindi/English)
- Duplicate detection
- Rate limiting (0.5s delay)

**Output**: 93 PDF files (2023-2025 circulars)

---

### 2. Text Processing Pipeline

**Components** (Modular Design):
- `src/preprocessing/pdf_extractor.py` - PDF to text extraction
- `src/preprocessing/text_cleaner.py` - Advanced text cleaning
- `src/preprocessing/text_chunker.py` - Semantic chunking

```
┌─────────────────────────────────────────────────────────┐
│           Text Processing Pipeline (Modular)            │
└─────────────────────────────────────────────────────────┘

  Raw PDFs (93 files)
       │
       ▼
  ┌────────────────────────────┐
  │   pdf_extractor.py         │
  ├────────────────────────────┤
  │ • pdfplumber (text-based)  │
  │ • Tesseract OCR (scanned)  │
  │ • Devanagari removal       │
  │ • Unicode cleanup          │
  │ • Ref-based trimming       │
  └──────────┬─────────────────┘
             │
             ▼
  Extracted Text (93 .txt files)
  → data/processed_text/
             │
             ▼
  ┌────────────────────────────┐
  │   text_cleaner.py          │
  ├────────────────────────────┤
  │ • Metadata extraction      │
  │ • Ultra-aggressive cleaning│
  │ • English line filtering   │
  │ • Remove transliterations  │
  │ • Whitespace normalization │
  └──────────┬─────────────────┘
             │
             ▼
  Cleaned Text (93 .txt files)
  → data/processed_text/ (updated)
             │
             ▼
  ┌────────────────────────────┐
  │   text_chunker.py          │
  ├────────────────────────────┤
  │ • LangChain Splitter       │
  │ • 800 tokens per chunk     │
  │ • 200 char overlap         │
  │ • Semantic boundaries      │
  │ • Fallback to simple split │
  └──────────┬─────────────────┘
             │
             ▼
  Text Chunks (561 chunks)
  → data/chunks/ (93 .json files)
```

**Modular Design Benefits**:
- **Reusable**: Each module can be imported and used independently
- **Testable**: Individual components are easier to test
- **Maintainable**: Clear separation of concerns
- **Flexible**: Easy to swap or enhance individual steps

**Text Extraction Strategy** (`pdf_extractor.py`):
1. Try pdfplumber first (fast, for text-based PDFs)
2. Fall back to Tesseract OCR (for scanned PDFs with `eng+hin` support)
3. Apply basic Devanagari and Unicode cleaning
4. Return extracted text with method used

**Text Cleaning Strategy** (`text_cleaner.py`):
1. Extract metadata (Ref, Date, Subject)
2. Ultra-aggressive character filtering (ASCII only)
3. Filter lines by English content ratio (min 50%)
4. Remove consonant clusters (Hindi transliterations)
5. Prepend formatted metadata to cleaned text

**Chunking Strategy** (`text_chunker.py`):
- **Chunk size**: 800 tokens (~3200 characters)
- **Overlap**: 200 characters (preserves context)
- **Separators**: Paragraphs → Sentences → Punctuation → Words → Characters
- **Fallback**: Simple character-based chunking if LangChain unavailable
- **Result**: 501 semantically coherent chunks

---

### 3. Embedding & Indexing Layer

**Components**:
- `src/embeddings/embedder.py`
- `src/embeddings/build_index.py`

```
┌─────────────────────────────────────────────────────────┐
│         Embedding & Indexing Layer                      │
└─────────────────────────────────────────────────────────┘

  Text Chunks (561)
       │
       ▼
  ┌────────────────────────────┐
  │   OpenAI Embeddings        │
  ├────────────────────────────┤
  │ Model: text-embedding-     │
  │        3-large              │
  │ Dimension: 3072            │
  │ Batch size: 32             │
  │ Cost: ~$0.13 per million   │
  │       tokens               │
  └──────────┬─────────────────┘
             │
             ▼
  Embeddings (561 × 3072)
             │
             ▼
  ┌────────────────────────────┐
  │     ChromaDB Index         │
  ├────────────────────────────┤
  │ Algorithm: HNSW            │
  │ Distance: L2 (Euclidean)   │
  │ Vectors: 501               │
  │ Persistent: SQLite + HNSW  │
  └──────────┬─────────────────┘
             │
             ▼
  ┌────────────────────────────┐
  │    Persistent Storage      │
  ├────────────────────────────┤
  │ chromadb/  (collection)    │
  │   ├─ chroma.sqlite3        │
  │   └─ index/ (HNSW)         │
  └────────────────────────────┘
             │
             ▼
  data/chromadb/
```

**Embedding Model** (Current Configuration):
- **Provider**: OpenAI API
- **Model**: `text-embedding-3-large`
- **Dimension**: 3072 (8x larger than free models)
- **Quality**: State-of-the-art embedding performance
- **Cost**: ~$0.13 per million tokens (~$0.01-0.02 for this dataset)
- **Speed**: API-dependent (~780ms per query average)

**Alternative: Free Local Embeddings**:
- **Provider**: Sentence Transformers (local)
- **Model**: `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Cost**: FREE (no API calls)
- **Speed**: ~500 sentences/second on CPU
- **Configuration**: Set `EMBEDDING_PROVIDER=sentence-transformers` in `.env`

**ChromaDB Index**:
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Distance**: L2 (Euclidean distance)
- **Search time**: <1ms for 561 vectors
- **Storage**: Persistent SQLite database with HNSW index
- **Benefits**: Built-in persistence, metadata filtering, easy scalability

---

### 4. RAG Query Pipeline

**Components**:
- `src/rag_pipeline/pipeline.py` - Main RAG orchestration
- `src/llm/answer_generator.py` - LLM integration (Groq/OpenAI)

```
┌─────────────────────────────────────────────────────────┐
│              RAG Query Pipeline                         │
└─────────────────────────────────────────────────────────┘

  User Query + Conversation History
       │
       ▼
  ┌────────────────────────────┐
  │   Query Enhancement        │
  ├────────────────────────────┤
  │ • Check for follow-ups     │
  │ • Add context from history │
  │ • Boost previous sources   │
  └──────────┬─────────────────┘
             │
             ▼
  Enhanced Query
             │
             ▼
  ┌────────────────────────────┐
  │   Query Embedding          │
  ├────────────────────────────┤
  │ Same model as indexing     │
  │ OpenAI text-embedding-     │
  │ 3-large                    │
  └──────────┬─────────────────┘
             │
             ▼
  Query Vector (3072-dim)
             │
             ▼
  ┌────────────────────────────┐
  │  ChromaDB Similarity Search│
  ├────────────────────────────┤
  │ • HNSW approximate search  │
  │ • Compute L2 distances     │
  │ • Rank by similarity       │
  │ • Return top-K results     │
  └──────────┬─────────────────┘
             │
             ▼
  Top-K Similar Chunks
   • Chunks with distances
   • Source metadata
   • Chunk indices
             │
             ▼
  ┌────────────────────────────┐
  │   Context Formatting       │
  ├────────────────────────────┤
  │ • Aggregate chunks         │
  │ • Add source citations     │
  │ • Limit context length     │
  │ • Format for LLM prompt    │
  └──────────┬─────────────────┘
             │
             ▼
  Retrieved Context + Sources
             │
             ▼
  ┌────────────────────────────┐
  │   LLM Answer Generation    │
  ├────────────────────────────┤
  │ Provider: Groq/OpenAI      │
  │ Model: llama-3.3/gpt-4o    │
  │ • System prompt            │
  │ • Context injection        │
  │ • Source citations         │
  │ • Error handling & retry   │
  └──────────┬─────────────────┘
             │
             ▼
  ┌────────────────────────────┐
  │      Final Response        │
  ├────────────────────────────┤
  │ • Generated answer         │
  │ • Retrieved context        │
  │ • Source documents         │
  │ • Similarity scores        │
  │ • LLM metadata             │
  └────────────────────────────┘
             │
             ▼
  FastAPI → Streamlit UI
```

**Pipeline Steps**:
1. **Query Enhancement**: Add context from conversation history for follow-up questions
2. **Query Embedding**: Convert enhanced question to 3072-dim vector using OpenAI API
3. **Similarity Search**: Find top-K most similar chunks using ChromaDB
4. **Source Boosting**: Prioritize documents from previous answer (for follow-ups)
5. **Context Aggregation**: Combine chunks with source citations
6. **LLM Generation**: Generate natural language answer using retrieved context
7. **Response Formatting**: Structure with answer, sources, and metadata

**Performance**:
- Query embedding: ~780ms (OpenAI API call)
- ChromaDB search: <1ms
- Context formatting: <5ms
- LLM generation: ~1-3s (Groq/OpenAI)
- **Total latency**: ~1-3 seconds

---

### 5. Application Layer

**Components**:
- `app/streamlit_app.py` - Web interface
- `src/api/main.py` - FastAPI REST API

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer                          │
└─────────────────────────────────────────────────────────┘

  User Browser
       │
       ▼
  ┌────────────────────────────┐
  │   Streamlit Web UI         │
  ├────────────────────────────┤
  │ • Chat interface           │
  │ • Conversation history     │
  │ • Context toggle           │
  │ • Source display           │
  │ • Session management       │
  └──────────┬─────────────────┘
             │
             ▼ HTTP POST
  ┌────────────────────────────┐
  │   FastAPI Backend          │
  ├────────────────────────────┤
  │ Endpoint: POST /query      │
  │ • Request validation       │
  │ • CORS handling            │
  │ • Error handling           │
  │ • Response formatting      │
  └──────────┬─────────────────┘
             │
             ▼
  ┌────────────────────────────┐
  │   RAG Pipeline             │
  ├────────────────────────────┤
  │ • Query processing         │
  │ • Vector search            │
  │ • LLM generation           │
  └──────────┬─────────────────┘
             │
             ▼
  JSON Response
   {
     "answer": "...",
     "context": "...",
     "sources": [...],
     "llm_provider": "groq/openai",
     "llm_model": "llama-3.3-70b/gpt-4o-mini"
   }
             │
             ▼
  Streamlit UI → User
```

**Streamlit Features**:
- **Chat Interface**: Messages displayed in conversation format
- **Conversation Context**: Toggle to enable/disable follow-up question support
- **Source Citations**: Expandable sections showing retrieved documents
- **Session Management**: Maintains conversation history per user session
- **Clear Chat**: Button to reset conversation history

**FastAPI Features**:
- **REST API**: Single `/query` endpoint for querying
- **Request Model**: Validates `query` and optional `conversation_history`
- **Response Model**: Structured JSON with answer, context, sources, metadata
- **CORS**: Configured for local development
- **Error Handling**: Graceful failures with informative messages

**Communication Flow**:
1. User types question in Streamlit chat
2. Streamlit sends POST request to FastAPI with query + history
3. FastAPI validates request and calls RAG pipeline
4. RAG pipeline processes query and generates answer
5. FastAPI returns structured JSON response
6. Streamlit displays answer and sources in chat

---

## Data Flow

### Index Building Flow

```
┌────────────────┐
│  Raw PDFs (93) │
└────────┬───────┘
         │
         ▼
┌────────────────────────┐
│  pdf_extractor.py      │
│  • pdfplumber          │
│  • Tesseract OCR       │
│  • Basic cleaning      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Extracted Text (93 .txt)│
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  text_cleaner.py       │
│  • Metadata extraction │
│  • Ultra cleaning      │
│  • English filtering   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Cleaned Text (93 .txt) │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  text_chunker.py       │
│  • LangChain splitter  │
│  • 800 token chunks    │
│  • 200 char overlap    │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Chunks (501 in JSON)  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│     embedder.py        │
│  • OpenAI API call     │
│  • Batch encode        │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Embeddings (561 × 3072)│
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   build_index.py       │
│  • Create ChromaDB     │
│  • Add vectors         │
│  • Auto-persist        │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│    Vector Store        │
│  • chromadb/           │
│    ├─ chroma.sqlite3   │
│    └─ index/           │
└────────────────────────┘
```

### Query Flow

```
┌───────────────┐
│   User Query  │
└───────┬───────┘
        │
        ▼
┌────────────────────────┐
│    pipeline.py         │
│  • Load index          │
│  • Load embedder       │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  embedder.embed_text() │
│  • Generate query vec  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  ChromaDB.query()      │
│  • Find top-K similar  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  format_context()      │
│  • Aggregate results   │
│  • Add citations       │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   Response Dict        │
│  • context             │
│  • sources             │
│  • scores              │
└────────────────────────┘
```

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Data Collection** | BeautifulSoup4 | 4.12.3 | HTML parsing |
| | Requests | 2.32.5 | HTTP requests |
| **PDF Processing** | pdfplumber | 0.11.0 | Text extraction |
| | Tesseract OCR | Latest | OCR for scanned PDFs |
| | pdf2image | 1.17.0 | PDF to image conversion |
| **Text Processing** | LangChain | 1.0.8 | Text chunking |
| | langchain-text-splitters | 1.0.0 | Recursive splitter |
| **Embeddings** | OpenAI API | Latest | text-embedding-3-large (current) |
| | sentence-transformers | 2.7.0 | Local embeddings (alternative) |
| **Vector Store** | ChromaDB | 0.4.22 | Similarity search |
| **LLM Integration** | Groq SDK | Latest | Free LLM API (recommended) |
| | OpenAI SDK | Latest | Paid LLM API |
| **Web Backend** | FastAPI | 0.115.12 | REST API server |
| | Uvicorn | Latest | ASGI server |
| **Web Frontend** | Streamlit | 1.41.1 | Interactive web UI |
| **Data Handling** | NumPy | 1.26.4 | Array operations |
| | Pandas | 2.2.1 | Data manipulation |
| **Runtime** | Python | 3.12 | Language runtime |

### Why These Technologies?

1. **OpenAI Embeddings** (Current):
   - ✅ State-of-the-art quality (3072 dimensions)
   - ✅ Best semantic understanding
   - ✅ API response time (~780ms average)
   - ✅ Low cost (~$0.02 for this dataset)
   - ✅ No local model management needed

2. **sentence-transformers** (Alternative):
   - ✅ Free and open-source
   - ✅ Runs locally (no API calls)
   - ✅ Good quality embeddings (384-dim)
   - ✅ Fast on CPU (~500 sentences/sec)
   - ✅ Zero ongoing costs

3. **ChromaDB**:
   - ✅ Open-source vector database
   - ✅ Built-in persistence (no manual save/load)
   - ✅ HNSW indexing for fast search (<1ms)
   - ✅ Metadata filtering capabilities
   - ✅ Easy to scale and maintain

4. **LangChain**:
   - ✅ Smart text splitting
   - ✅ Respects semantic boundaries
   - ✅ Configurable separators
   - ✅ Maintains context with overlap

5. **Groq**:
   - ✅ Free tier with generous limits
   - ✅ Fast inference (~100 tokens/sec)
   - ✅ High-quality LLM (Llama 3.3 70B)
   - ✅ Simple API
   - ✅ No credit card required

6. **FastAPI**:
   - ✅ Modern Python web framework
   - ✅ Automatic API documentation
   - ✅ Type validation with Pydantic
   - ✅ High performance (async)
   - ✅ Easy to deploy

7. **Streamlit**:
   - ✅ Rapid UI development
   - ✅ Native chat interface
   - ✅ Session state management
   - ✅ No HTML/CSS/JS required
   - ✅ Built-in widgets

## Design Principles

### 1. Low Cost with Quality Focus
- **Current Setup**: OpenAI embeddings + Groq LLM
  - Embeddings: ~$0.02 one-time cost (for indexing)
  - LLM: Free with Groq (or ~$0.04/session with OpenAI)
- **Free Alternative**: Sentence Transformers + Groq LLM
  - Change `EMBEDDING_PROVIDER=sentence-transformers` in `.env`
  - Slightly lower embedding quality but zero cost
- All other components are free and open-source

### 2. Conversational UX
- Natural chat interface for easy interaction
- Conversation context for follow-up questions
- Source citations for transparency
- Clear session management

### 3. Modularity
- Each component is independent
- Easy to swap or upgrade components
- Clear interfaces between layers
- Preprocessing pipeline is fully modular (3 reusable components)

### 4. Scalability
- ChromaDB can handle millions of vectors with HNSW
- Batch processing for efficiency
- Optimized data structures
- FastAPI supports async for concurrent requests

### 5. Production-Ready
- Battle-tested libraries
- Comprehensive error handling
- Persistent storage
- Fast performance (<3s query time)
- Retry logic for LLM API calls
- Graceful degradation

## Performance Characteristics

### Build Time
- PDF processing: ~5-10 minutes (93 PDFs)
- Chunking: ~30 seconds (561 chunks)
- Embedding generation: ~1-2 minutes (561 chunks, OpenAI API)
- Index building: <1 second
- **Total**: ~7-13 minutes

### Query Time (End-to-End)
- Query embedding: ~780ms (OpenAI API call)
- ChromaDB search: <1ms
- Context formatting: ~5ms
- LLM generation: ~1-3s (Groq/OpenAI)
- Network latency: ~50-100ms
- **Total**: ~1-3 seconds (user-facing)

### Storage
- Raw PDFs: ~50MB
- Processed text: ~2MB
- Chunks (JSON): ~1MB
- ChromaDB index: Persistent SQLite with HNSW
- **Total**: ~59MB

### Memory
- ChromaDB index: Efficient in-memory caching
- Runtime (Python + FastAPI): ~200MB
- Streamlit UI: ~150MB
- **Peak**: ~400MB
- **Note**: No local model loading required (using OpenAI API)

## Extensibility

### Adding New Documents
1. Place PDFs in `data/raw_pdfs/`
2. Run modular preprocessing pipeline:
   ```bash
   cd src/preprocessing
   python pdf_extractor.py
   python text_cleaner.py
   python text_chunker.py
   ```
3. Rebuild index: `python src/embeddings/build_index.py`

### Changing Embedding Model

**Current**: OpenAI `text-embedding-3-large` (3072-dim, paid)

**Switch to Free Local Embeddings**:
1. Update `.env`:
   ```env
   EMBEDDING_PROVIDER=sentence-transformers
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
2. Rebuild index: `python src/embeddings/build_index.py`

**Other OpenAI Models**:
- `text-embedding-3-small` (1536-dim, cheaper, faster)
- `text-embedding-ada-002` (1536-dim, older model)

**Other Free Models**:
- `all-mpnet-base-v2` (768-dim, better quality)
- `multi-qa-mpnet-base-dot-v1` (768-dim, QA-optimized)

### Switching LLM Provider
**Already Implemented** - Toggle in `.env`:
```env
# Free option (recommended)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Or paid option
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

### Customizing Preprocessing
Each module is independently configurable:
- **pdf_extractor.py**: Toggle OCR, adjust DPI
- **text_cleaner.py**: Adjust English ratio threshold
- **text_chunker.py**: Change chunk size, overlap

### UI Customization
- **Streamlit**: Edit `app/streamlit_app.py` for UI changes
- **FastAPI**: Add endpoints in `src/api/main.py`
- **Alternative UIs**: Gradio, Flask, or custom frontend

## Security Considerations

### Data Security
- All document data stays local (stored in `data/` folder)
- External API calls:
  - **OpenAI API**: For embeddings (query time) and LLM generation
  - Only retrieved context chunks are sent (not full documents)
- Vector store is saved locally (ChromaDB)
- API keys stored securely in `.env` file

### Input Validation
- FastAPI validates request schemas with Pydantic
- Query length limits enforced
- Top-K bounds checked
- File type validation during upload

### API Security
- CORS configured for development
- API keys stored in `.env` (not committed)
- Error messages don't leak sensitive info
- Rate limiting can be added to FastAPI

### Error Handling
- Graceful LLM API failures with retry logic
- Informative error messages without stack traces
- Fallback to retrieval-only if LLM fails
- Session isolation in Streamlit

## Future Enhancements

### Planned
1. **Hybrid Search**: Combine semantic + keyword (BM25) search
2. **Re-ranking**: Use cross-encoder to improve result quality
3. **Multi-language**: Add Hindi language support in UI
4. **Streaming**: Real-time LLM response streaming
5. **Cache**: Query result caching with Redis

### Possible
1. **GPU Support**: Faster embedding generation with CUDA
2. **Distributed**: Scale across multiple machines
3. **Real-time Updates**: Incremental index updates without rebuild
4. **Advanced Analytics**: Query analytics dashboard
5. **Export Features**: Download conversations, generate reports
6. **Multi-modal**: Image/table extraction from PDFs
7. **Authentication**: User login and role-based access
8. **Document Upload**: UI for uploading new PDFs

### Already Implemented ✅
- ✅ **Web UI**: Streamlit interface (done)
- ✅ **REST API**: FastAPI backend (done)
- ✅ **LLM Integration**: Groq/OpenAI support (done)
- ✅ **Conversation Context**: Follow-up question support (done)
- ✅ **Modular Preprocessing**: 3-module pipeline (done)
