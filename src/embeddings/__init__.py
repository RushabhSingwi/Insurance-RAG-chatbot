"""
Embeddings module for generating and managing text embeddings.
"""

from .embedder import EmbeddingGenerator
from .build_index import FAISSIndexBuilder

__all__ = ['EmbeddingGenerator', 'FAISSIndexBuilder']
