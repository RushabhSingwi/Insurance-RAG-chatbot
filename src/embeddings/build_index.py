import os
import json
import time
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Fix hnswlib compatibility issue
try:
    import hnswlib
    if not hasattr(hnswlib.Index, 'file_handle_count'):
        hnswlib.Index.file_handle_count = 1
except ImportError:
    pass

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Global token tracking for OpenAI embeddings
total_embedding_tokens = 0


class TokenTrackingOpenAIEmbeddings(OpenAIEmbeddings):
    """Wrapper around OpenAI embeddings to track token usage."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents and track token usage."""
        global total_embedding_tokens

        # OpenAI embeddings: estimate ~0.75 tokens per word
        estimated_tokens = sum(len(text.split()) for text in texts) * 0.75

        try:
            result = super().embed_documents(texts)

            # Log estimated token usage (actual usage not exposed by API)
            total_embedding_tokens += int(estimated_tokens)
            print(f"  [Token Usage] Estimated ~{int(estimated_tokens)} tokens for {len(texts)} documents (Total: ~{total_embedding_tokens})")

            return result
        except Exception as e:
            raise e


def load_chunks_as_documents(chunks_dir: str):
    """Load all chunks from JSON files and convert to LangChain Documents."""
    chunks_path = Path(chunks_dir)
    json_files = list(chunks_path.glob("*.json"))

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


def main():
    chunks_dir = "data/chunks"
    output_dir = "data/chromadb"

    # Load chunks as documents
    print("Loading chunks...")
    chunks = load_chunks_as_documents(chunks_dir)
    print(f"Loaded {len(chunks)} chunks")

    # Create embeddings based on provider
    print(f"Creating embeddings with {EMBEDDING_PROVIDER}: {EMBEDDING_MODEL}...")

    if EMBEDDING_PROVIDER == "openai":
        embeddings_model = TokenTrackingOpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            chunk_size=100,  # Process 100 texts per API call
            max_retries=3,
            request_timeout=60
        )
        print(f"[INFO] Using OpenAI embeddings with token tracking")
        print(f"[INFO] Rate limit handling: Smaller batches with delays")
    elif EMBEDDING_PROVIDER == "huggingface":
        # Use local HuggingFace embeddings (free, no API key needed)
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

    # Create vector store with persist_directory (modern ChromaDB API)
    print("Creating vector store...")
    vector_store = Chroma(
        collection_name="irdai_documents",
        embedding_function=embeddings_model,
        persist_directory=output_dir
    )

    # Add documents in batches with retry logic to handle rate limiting
    print("Adding documents to vector store in batches...")

    # Use smaller batch size for OpenAI to avoid rate limits
    if EMBEDDING_PROVIDER == "openai":
        batch_size = 50  # Smaller batches for OpenAI
        inter_batch_delay = 3  # Longer delay between batches
        print(f"[INFO] Using batch_size={batch_size} with {inter_batch_delay}s delay for OpenAI")
    else:
        batch_size = 100  # Larger batches for HuggingFace (local)
        inter_batch_delay = 1

    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_uuids = [str(uuid4()) for _ in range(len(batch_chunks))]

        max_retries = 3
        retry_delay = 2  # Start with 2 second delay

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
                        print(f"  [ERROR] OpenAI API quota exceeded. Please check billing or use GROQ_API_KEY instead.")
                        print(f"  To use Groq: Set EMBEDDING_MODEL to a Groq-compatible model in .env")
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

    print(f"\n[OK] Done! Added {total_chunks} documents to vector store at {output_dir}")

    # Show token usage summary for OpenAI
    if EMBEDDING_PROVIDER == "openai" and total_embedding_tokens > 0:
        print(f"\n[Token Usage Summary]")
        print(f"  Total estimated tokens: ~{total_embedding_tokens}")
        print(f"  Estimated cost ({EMBEDDING_MODEL}): ~${total_embedding_tokens / 1_000_000 * 0.02:.4f}")
        print(f"  Note: Actual token count may vary slightly")


if __name__ == "__main__":
    main()
