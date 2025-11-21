"""
Enhanced Embedding Generator supporting both OpenAI and Free models.

Supports:
1. OpenAI API (text-embedding-3-small, text-embedding-3-large, ada-002)
2. Sentence Transformers (free, local)
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load environment variables (override=True to ensure .env takes precedence)
load_dotenv(override=True)

# Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")  # or "openai"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class EmbeddingGenerator:
    """
    Generate embeddings using either OpenAI API or Sentence Transformers.
    """

    def __init__(self, provider: str = EMBEDDING_PROVIDER, model: str = None):
        """
        Initialize the embedding generator.

        Args:
            provider: "openai" or "sentence-transformers"
            model: Model name (optional, uses env var if not provided)
        """
        self.provider = provider.lower()
        self.model_name = model
        self.model = None
        self.client = None

        print(f"Initializing embedding generator with provider: {self.provider}")

        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "sentence-transformers":
            self._init_sentence_transformers()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'sentence-transformers'")

    def _init_openai(self):
        """Initialize OpenAI embeddings."""
        try:
            from openai import OpenAI

            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Add it to your .env file: OPENAI_API_KEY=your_key_here"
                )

            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = self.model_name or EMBEDDING_MODEL

            # Get embedding dimension
            embedding_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self.embedding_dim = embedding_dims.get(self.model_name, 1536)

            print(f"OpenAI embeddings initialized")
            print(f"  Model: {self.model_name}")
            print(f"  Embedding dimension: {self.embedding_dim}")

        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _init_sentence_transformers(self):
        """Initialize Sentence Transformers (free, local)."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model_name = self.model_name or EMBEDDING_MODEL
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            print(f"Sentence Transformers initialized")
            print(f"  Model: {self.model_name}")
            print(f"  Embedding dimension: {self.embedding_dim}")

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self.provider == "openai":
            return self._embed_text_openai(text)
        else:
            return self._embed_text_sentence_transformers(text)

    def _embed_text_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return None

    def _embed_text_sentence_transformers(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in one batch

        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return self._embed_batch_openai(texts, batch_size)
        else:
            return self._embed_batch_sentence_transformers(texts, batch_size)

    def _embed_batch_openai(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Generate embeddings using OpenAI API (with batching)."""
        embeddings = []

        print(f"Generating embeddings for {len(texts)} texts using OpenAI...")
        print(f"  Model: {self.model_name}")
        print(f"  Estimated cost: ${(sum(len(t.split()) for t in texts) * 1.3 / 1_000_000) * 0.020:.4f}")

        # OpenAI API can handle large batches
        # Split into batches to avoid hitting limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}...")

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )

                for item in response.data:
                    embeddings.append(item.embedding)

            except Exception as e:
                print(f"Error in batch {i // batch_size + 1}: {e}")
                # Fallback to individual encoding
                for text in batch:
                    emb = self.embed_text(text)
                    embeddings.append(emb if emb else [0.0] * self.embedding_dim)

        print(f"✓ Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def _embed_batch_sentence_transformers(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers."""
        embeddings = []

        print(f"Generating embeddings for {len(texts)} texts...")

        try:
            # Encode all texts in batches with progress bar
            all_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Convert to list of lists
            embeddings = [emb.tolist() for emb in all_embeddings]

            print(f"✓ Successfully generated {len(embeddings)} embeddings")

        except Exception as e:
            print(f"Error in batch encoding: {e}")
            # Fallback to individual encoding
            for i, text in enumerate(texts):
                if (i + 1) % 50 == 0:
                    print(f"  Processing {i + 1}/{len(texts)}...")

                embedding = self.embed_text(text)
                embeddings.append(embedding)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator."""
        return self.embedding_dim


def main():
    """Test the embedder with sample texts."""
    print("="*80)
    print("Testing Enhanced Embedding Generator")
    print("="*80)

    # Test both providers if available
    providers_to_test = ["sentence-transformers"]

    # Check if OpenAI key is available
    if OPENAI_API_KEY:
        providers_to_test.append("openai")

    for provider in providers_to_test:
        print(f"\n\nTesting provider: {provider}")
        print("="*80)

        try:
            embedder = EmbeddingGenerator(provider=provider)

            # Test with sample texts
            sample_texts = [
                "This is a test sentence for embedding generation.",
                "Insurance policies provide financial protection.",
                "The IRDAI regulates insurance companies in India."
            ]

            print("\nGenerating embeddings for sample texts...")
            embeddings = embedder.embed_batch(sample_texts)

            print(f"\nResults:")
            print(f"  Number of embeddings: {len(embeddings)}")
            print(f"  Embedding dimension: {len(embeddings[0]) if embeddings and embeddings[0] else 0}")
            if embeddings and embeddings[0]:
                print(f"  First embedding (first 10 values): {embeddings[0][:10]}")

        except Exception as e:
            print(f"Error testing {provider}: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
