"""
FastAPI Backend for IRDAI Insurance Circulars RAG System

This API provides endpoints for querying the RAG system.
"""

import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from rag_pipeline.pipeline import RAGPipeline

# Load configuration
config = get_config()

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
    question: str = Field(..., description="User's question (may be rephrased for retrieval)", min_length=1, max_length=500)
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)
    conversation_history: Optional[List[ConversationEntry]] = Field(None, description="Previous Q&A pairs for context")
    llm_provider: Optional[str] = Field(None, description="LLM provider to use (openai or groq)")
    original_question: Optional[str] = Field(None, description="Original question before rephrasing (used for LLM prompt)")


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

        return HealthResponse(
            status="healthy",
            index_size=index_size,
            embedding_model=config.EMBEDDING_MODEL,
            embedding_dimension=config.EMBEDDING_DIMENSION
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

        # Convert Pydantic models to dicts for conversation history
        conversation_history_dicts = None
        if request.conversation_history:
            conversation_history_dicts = [
                entry.model_dump() for entry in request.conversation_history
            ]

        # Execute query with optional LLM provider, conversation history, and original question
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            llm_provider=request.llm_provider,
            conversation_history=conversation_history_dicts,
            original_question=request.original_question
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

        # Get all metadata to calculate unique sources
        if total_documents > 0:
            all_data = pipeline.vector_store._collection.get(limit=total_documents)
            unique_sources = len(set([m['source_file'] for m in all_data['metadatas']])) if all_data['metadatas'] else 0
        else:
            unique_sources = 0

        return {
            "total_documents": total_documents,
            "embedding_dimension": config.EMBEDDING_DIMENSION,
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_model": config.EMBEDDING_MODEL,
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
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )
