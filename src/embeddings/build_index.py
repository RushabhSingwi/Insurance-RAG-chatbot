import os
import json
import pickle
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import faiss

# Import embedder
try:
    from .embedder import EmbeddingGenerator
except ImportError:
    from embedder import EmbeddingGenerator

# Load environment variables
load_dotenv()

# Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


class FAISSIndexBuilder:
    """
    Build and manage FAISS index for efficient similarity search.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index builder.

        Args:
            embedding_dim: Dimension of embedding vectors (384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.metadata = []

        if DEBUG:
            print(f"Initialized FAISSIndexBuilder with dimension: {embedding_dim}")

    def create_index(self, index_type: str = "flat"):
        """
        Create a FAISS index.

        Args:
            index_type: Type of index ('flat' for exact search, 'ivf' for faster approximate search)
        """
        if index_type == "flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if DEBUG:
                print("Created Flat L2 index (exact search)")

        elif index_type == "ivf":
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            if DEBUG:
                print("Created IVF index (approximate search)")

        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict]):
        """
        Add embeddings to the index.

        Args:
            embeddings: Numpy array of embeddings (shape: [num_vectors, embedding_dim])
            chunks: List of text chunks corresponding to embeddings
            metadata: List of metadata dictionaries for each chunk
        """
        if self.index is None:
            self.create_index()

        # Train index if needed (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(embeddings)
            if DEBUG:
                print("Index trained")

        # Add vectors to index
        self.index.add(embeddings)

        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

        if DEBUG:
            print(f"Added {len(embeddings)} vectors to index")

    def save_index(self, index_path: str, chunks_path: str, metadata_path: str):
        """
        Save index and associated data to disk.

        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks
            metadata_path: Path to save metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        if DEBUG:
            print(f"Saved index to: {index_path}")
            print(f"Saved chunks to: {chunks_path}")
            print(f"Saved metadata to: {metadata_path}")

    def load_index(self, index_path: str, chunks_path: str, metadata_path: str):
        """
        Load index and associated data from disk.

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks
            metadata_path: Path to metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        if DEBUG:
            print(f"Loaded index with {self.index.ntotal} vectors")
            print(f"Loaded {len(self.chunks)} chunks")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Search for similar vectors in the index.

        Args:
            query_embedding: Query vector (shape: [1, embedding_dim])
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices, chunks, metadata)
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

        # Get corresponding chunks and metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(dist)
                })

        return results


def load_all_chunks(chunks_dir: str) -> tuple:
    """
    Load all text chunks from the chunks directory.

    Args:
        chunks_dir: Directory containing chunk JSON files

    Returns:
        Tuple of (chunks list, metadata list)
    """
    chunks_path = Path(chunks_dir)
    json_files = list(chunks_path.glob("*.json"))

    if not json_files:
        print(f"No chunk files found in {chunks_dir}")
        return None, None

    print(f"Found {len(json_files)} chunk files\n")

    all_chunks = []
    all_metadata = []

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Loading: {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Add chunks with metadata
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source_file": json_file.stem,
                "chunk_index": idx,
                "filename": json_file.name
            })

        print(f"  [OK] Loaded {len(chunks)} chunks")

    print(f"\n{'='*60}")
    print(f"Total chunks loaded: {len(all_chunks)}")
    print(f"{'='*60}\n")

    return all_chunks, all_metadata


def build_faiss_index(chunks_dir: str, output_dir: str):
    """
    Build FAISS index from all chunks using free HuggingFace embeddings.
    This function will:
    1. Load all chunks from the chunks directory
    2. Generate embeddings using Sentence Transformers (free, local)
    3. Store everything in FAISS index

    Args:
        chunks_dir: Directory containing chunk JSON files
        output_dir: Directory to save the FAISS index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Building FAISS Index with Free HuggingFace Embeddings")
    print("="*60 + "\n")

    # Load all chunks
    print("Step 1: Loading chunks...")
    chunks, metadata = load_all_chunks(chunks_dir)

    if chunks is None:
        print("No chunks to process!")
        return

    # Initialize embedder
    print("\nStep 2: Initializing embedding generator...")
    embedder = EmbeddingGenerator()
    embedding_dim = embedder.get_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}\n")

    # Generate embeddings
    print("Step 3: Generating embeddings...")
    embeddings = embedder.embed_batch(chunks, batch_size=32)

    # Filter out failed embeddings
    valid_data = []
    failed_count = 0
    for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
        if embedding is not None and len(embedding) > 0:
            valid_data.append((chunk, embedding, meta))
        else:
            failed_count += 1
            if DEBUG:
                print(f"  [WARNING] Failed to generate embedding for chunk {i}")

    if failed_count > 0:
        print(f"\n[WARNING] {failed_count} embeddings failed")

    if not valid_data:
        print("No valid embeddings generated!")
        return

    valid_chunks, valid_embeddings, valid_metadata = zip(*valid_data)

    print(f"\n[OK] Successfully generated {len(valid_embeddings)} embeddings")

    # Convert embeddings to numpy array
    embeddings_array = np.array(valid_embeddings, dtype=np.float32)

    # Build FAISS index
    print("\nStep 4: Building FAISS index...")
    embedding_dim = embeddings_array.shape[1]
    builder = FAISSIndexBuilder(embedding_dim=embedding_dim)
    builder.add_embeddings(embeddings_array, list(valid_chunks), list(valid_metadata))

    # Save index
    index_path = output_path / "faiss_index.bin"
    chunks_path = output_path / "chunks.json"
    metadata_path = output_path / "metadata.pkl"

    print("\nStep 5: Saving index...")
    builder.save_index(str(index_path), str(chunks_path), str(metadata_path))

    print(f"\n{'='*60}")
    print(f"Index building complete!")
    print(f"  Total documents in index: {builder.index.ntotal}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Index file: {index_path}")
    print(f"  Chunks file: {chunks_path}")
    print(f"  Metadata file: {metadata_path}")
    print(f"{'='*60}")


def main():
    """Main function to build the FAISS index."""
    chunks_dir = "data/chunks"
    output_dir = "data/vector_store"

    build_faiss_index(chunks_dir, output_dir)


if __name__ == "__main__":
    main()
