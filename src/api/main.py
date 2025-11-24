"""
FastAPI Backend for IRDAI Insurance Circulars RAG System

This API provides endpoints for querying the RAG system.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_pipeline.pipeline import RAGPipeline

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"

# CRITICAL FIX: Clear any system-wide environment variables that may be wrong
# This ensures .env file values take precedence
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

load_dotenv(dotenv_path=env_path, override=True)  # Force override of existing env vars

# Initialize FastAPI app
app = FastAPI(
    title="IRDAI Insurance Circulars RAG API",
    description="API for querying IRDAI insurance circulars using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline (singleton)
rag_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


# Request/Response Models
class ConversationEntry(BaseModel):
    """Model for a conversation history entry."""
    question: str
    answer: str
    sources: Optional[List[str]] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="User's question", min_length=1, max_length=500)
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)
    conversation_history: Optional[List[ConversationEntry]] = Field(None, description="Previous Q&A pairs for context")
    llm_provider: Optional[str] = Field(None, description="LLM provider to use (openai or groq)")


class ChunkResult(BaseModel):
    """Model for a single chunk result."""
    chunk: str
    source_file: str
    chunk_index: int
    distance: float
    similarity: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    question: str
    answer: Optional[str] = None
    context: str
    sources: List[str]
    results: List[ChunkResult]
    total_results: int
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    tokens_used: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_size: int
    embedding_model: str
    embedding_dimension: int


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "IRDAI Insurance Circulars RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        # Get collection count from ChromaDB
        index_size = pipeline.vector_store._collection.count()

        # Get embedding dimension based on provider
        if os.getenv("EMBEDDING_PROVIDER", "huggingface") == "huggingface":
            embedding_dim = 384  # sentence-transformers/all-MiniLM-L6-v2
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            embedding_dim = 1536  # OpenAI text-embedding-3-small
            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        return HealthResponse(
            status="healthy",
            index_size=index_size,
            embedding_model=model_name,
            embedding_dimension=embedding_dim
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.

    Args:
        request: QueryRequest containing question and top_k

    Returns:
        QueryResponse with context, sources, and results
    """
    try:
        pipeline = get_pipeline()

        # Execute query with optional LLM provider
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            llm_provider=request.llm_provider
        )

        # Format results
        chunk_results = []
        for r in result['results']:
            # Calculate similarity score from distance
            similarity = 1 / (1 + r['distance'])

            chunk_results.append(ChunkResult(
                chunk=r['chunk'],
                source_file=r['metadata']['source_file'],
                chunk_index=int(r['metadata']['chunk_index']),
                distance=r['distance'],
                similarity=similarity
            ))

        return QueryResponse(
            question=result['question'],
            answer=result.get('answer'),
            context=result['context'],
            sources=result['sources'],
            results=chunk_results,
            total_results=len(chunk_results),
            llm_provider=result.get('llm_provider'),
            llm_model=result.get('llm_model'),
            tokens_used=result.get('tokens_used')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get statistics about the RAG system."""
    try:
        pipeline = get_pipeline()

        # Get collection data from ChromaDB
        total_documents = pipeline.vector_store._collection.count()

        # Get embedding config
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_dim = 384 if embedding_provider == "huggingface" else 1536

        # Get all metadata to calculate unique sources
        if total_documents > 0:
            all_data = pipeline.vector_store._collection.get(limit=total_documents)
            unique_sources = len(set([m['source_file'] for m in all_data['metadatas']])) if all_data['metadatas'] else 0
        else:
            unique_sources = 0

        return {
            "total_documents": total_documents,
            "embedding_dimension": embedding_dim,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "total_chunks": total_documents,
            "unique_sources": unique_sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    print("Starting IRDAI RAG API...")
    try:
        get_pipeline()
        print("RAG Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("API will start but queries will fail until pipeline is initialized.")


if __name__ == "__main__":
    import uvicorn

    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
