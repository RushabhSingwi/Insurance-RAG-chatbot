"""
Embedding Index Builder Module

Builds a vector database index from text chunks using OpenAI or HuggingFace embeddings.
Includes token tracking, rate limiting, and retry logic for API calls.
"""

import json
import sys
import time
from pathlib import Path
from uuid import uuid4
from typing import List

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from utils.common import get_files_with_extension

# Fix hnswlib compatibility issue
try:
    import hnswlib
    if not hasattr(hnswlib.Index, 'file_handle_count'):
        hnswlib.Index.file_handle_count = 1
except ImportError:
    pass

# Load configuration
config = get_config()

# Constants
OPENAI_CHUNK_SIZE = 100
REQUEST_TIMEOUT = 60
TOKEN_ESTIMATION_RATIO = 0.75  # ~0.75 tokens per word
OPENAI_EMBEDDING_COST_PER_MILLION = 0.02

# Global token tracking for OpenAI embeddings
total_embedding_tokens = 0


class TokenTrackingOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Wrapper around OpenAI embeddings to track token usage.

    Provides token estimation for cost tracking since actual token usage
    is not exposed by the OpenAI API.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents and track estimated token usage.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        global total_embedding_tokens

        # Estimate tokens (actual usage not exposed by API)
        estimated_tokens = sum(len(text.split()) for text in texts) * TOKEN_ESTIMATION_RATIO

        try:
            result = super().embed_documents(texts)

            # Track cumulative token usage
            total_embedding_tokens += int(estimated_tokens)
            print(f"  [Token Usage] Estimated ~{int(estimated_tokens)} tokens for {len(texts)} documents (Total: ~{total_embedding_tokens})")

            return result
        except Exception as e:
            raise


def load_chunks_as_documents(chunks_dir: Path) -> List[Document]:
    """
    Load all chunks from JSON files and convert to LangChain Documents.

    Args:
        chunks_dir: Directory containing JSON files with text chunks

    Returns:
        List of Document objects with page_content and metadata
    """
    json_files = get_files_with_extension(chunks_dir, '.json')

    documents = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source_file": json_file.stem,
                    "chunk_index": str(idx),
                    "filename": json_file.name
                }
            )
            documents.append(doc)

    return documents


def main() -> None:
    """
    Build vector database index from text chunks.

    Loads chunks from JSON files, creates embeddings, and stores them
    in a ChromaDB vector database with retry logic for rate limiting.
    """
    # Load chunks as documents
    print("Loading chunks...")
    chunks = load_chunks_as_documents(config.CHUNKS_DIR)
    print(f"Loaded {len(chunks)} chunks")

    # Create embeddings based on provider
    print(f"Creating embeddings with {config.EMBEDDING_PROVIDER}: {config.EMBEDDING_MODEL}...")

    if config.EMBEDDING_PROVIDER == "openai":
        embeddings_model = TokenTrackingOpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            chunk_size=OPENAI_CHUNK_SIZE,
            max_retries=config.MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT
        )
        print(f"[INFO] Using OpenAI embeddings with token tracking")
        print(f"[INFO] Rate limit handling: Smaller batches with delays")
    elif config.EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {config.EMBEDDING_PROVIDER}")

    # Create vector store with persist_directory
    print("Creating vector store...")
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=str(config.CHROMADB_VECTOR_DB)
    )

    # Add documents in batches with retry logic to handle rate limiting
    print("Adding documents to vector store in batches...")

    batch_size = config.EMBEDDING_BATCH_SIZE
    inter_batch_delay = config.EMBEDDING_BATCH_DELAY

    if config.EMBEDDING_PROVIDER == "openai":
        print(f"[INFO] Using batch_size={batch_size} with {inter_batch_delay}s delay for OpenAI")

    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_uuids = [str(uuid4()) for _ in range(len(batch_chunks))]

        max_retries = config.MAX_RETRIES
        retry_delay = config.RETRY_BASE_DELAY

        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)
                print(f"  [OK] Added batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({batch_end}/{total_chunks} documents)")

                # Add delay between batches to avoid rate limiting
                if batch_end < total_chunks:
                    time.sleep(inter_batch_delay)
                break

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if "insufficient_quota" in error_str.lower():
                        print(f"  [ERROR] OpenAI API quota exceeded. Please check billing.")
                        raise
                    elif attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"  [WARN] Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  [ERROR] Failed after {max_retries} attempts: {e}")
                        raise
                else:
                    print(f"  [ERROR] {e}")
                    raise

    print(f"\n[OK] Done! Added {total_chunks} documents to vector store at {config.CHROMADB_VECTOR_DB}")

    # Show token usage summary for OpenAI
    if config.EMBEDDING_PROVIDER == "openai" and total_embedding_tokens > 0:
        print(f"\n[Token Usage Summary]")
        print(f"  Total estimated tokens: ~{total_embedding_tokens:,}")
        cost = (total_embedding_tokens / 1_000_000) * OPENAI_EMBEDDING_COST_PER_MILLION
        print(f"  Estimated cost ({config.EMBEDDING_MODEL}): ~${cost:.4f}")
        print(f"  Note: Actual token count may vary slightly")


if __name__ == "__main__":
    main()
