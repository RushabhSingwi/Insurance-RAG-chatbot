"""
Centralized Configuration Module

This module provides a single source of truth for all configuration settings
across the RAG IRDAI chatbot application. It handles environment variable loading,
path management, and default values.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Centralized configuration class for the RAG IRDAI chatbot."""

    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialize configuration by loading environment variables.

        Args:
            env_path: Optional path to .env file. If None, uses project root.
        """
        # Determine project root (rag-irdai-chatbot directory)
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        # Load environment variables
        if env_path is None:
            env_path = self.PROJECT_ROOT / ".env"

        # Load with override=True to ensure .env values take precedence
        load_dotenv(dotenv_path=env_path, override=True)

        # Initialize all configuration settings
        self._init_paths()
        self._init_embedding_config()
        self._init_llm_config()
        self._init_retrieval_config()
        self._init_debug_config()

    def _init_paths(self):
        """Initialize all path configurations."""
        # Data directories
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_PDFS_DIR = self.DATA_DIR / "raw_pdfs"
        self.PROCESSED_TEXT_DIR = self.DATA_DIR / "processed_text"
        self.PROCESSED_TEXT_CLEAN_DIR = self.DATA_DIR / "processed_text_clean"
        self.CHUNKS_DIR = self.DATA_DIR / "chunks"
        self.EVALUATION_DIR = self.DATA_DIR / "evaluation"

        # Vector database path (allow override from env)
        chromadb_path = os.getenv("CHROMADB_VECTOR_DB")
        if chromadb_path:
            self.CHROMADB_VECTOR_DB = Path(chromadb_path)
        else:
            self.CHROMADB_VECTOR_DB = self.DATA_DIR / "chromadb"

        # Evaluation dataset
        self.EVALUATION_DATASET_PATH = self.EVALUATION_DIR / "evaluation_dataset.json"

    def _init_embedding_config(self):
        """Initialize embedding provider configuration."""
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Embedding dimensions based on provider
        if self.EMBEDDING_PROVIDER == "huggingface":
            self.EMBEDDING_DIMENSION = 384  # sentence-transformers/all-MiniLM-L6-v2
        else:  # openai
            self.EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

        # Batch processing settings
        if self.EMBEDDING_PROVIDER == "openai":
            self.EMBEDDING_BATCH_SIZE = 50
            self.EMBEDDING_BATCH_DELAY = 3  # seconds
        else:  # huggingface
            self.EMBEDDING_BATCH_SIZE = 100
            self.EMBEDDING_BATCH_DELAY = 1  # seconds

    def _init_llm_config(self):
        """Initialize LLM provider configuration."""
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

        # LLM Provider and Models
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # Answer generation settings
        self.ENABLE_ANSWER_GENERATION = (
            os.getenv("ENABLE_ANSWER_GENERATION", "true").lower() == "true"
        )

    def _init_retrieval_config(self):
        """Initialize retrieval configuration."""
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
        self.COLLECTION_NAME = "irdai_documents"

        # Chunking settings
        self.MAX_TOKENS_PER_CHUNK = 800
        self.CHUNK_OVERLAP = 200

        # Retry settings for rate limiting
        self.MAX_RETRIES = 3
        self.RETRY_BASE_DELAY = 2  # seconds

    def _init_debug_config(self):
        """Initialize debug configuration."""
        debug_value = os.getenv("DEBUG", "false").lower()
        self.DEBUG = debug_value in ("true", "1", "yes")

    def get_model_for_provider(self, provider: Optional[str] = None) -> str:
        """
        Get the appropriate model name for a given provider.

        Args:
            provider: LLM provider name ('openai' or 'groq'). If None, uses default.

        Returns:
            Model name string
        """
        provider = provider or self.LLM_PROVIDER

        if provider == "openai":
            return self.OPENAI_MODEL
        elif provider == "groq":
            return self.GROQ_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_api_key_for_provider(self, provider: Optional[str] = None) -> str:
        """
        Get the API key for a given provider.

        Args:
            provider: LLM provider name ('openai' or 'groq'). If None, uses default.

        Returns:
            API key string
        """
        provider = provider or self.LLM_PROVIDER

        if provider == "openai":
            return self.OPENAI_API_KEY
        elif provider == "groq":
            return self.GROQ_API_KEY
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  PROJECT_ROOT={self.PROJECT_ROOT}\n"
            f"  EMBEDDING_PROVIDER={self.EMBEDDING_PROVIDER}\n"
            f"  EMBEDDING_MODEL={self.EMBEDDING_MODEL}\n"
            f"  LLM_PROVIDER={self.LLM_PROVIDER}\n"
            f"  TOP_K_RESULTS={self.TOP_K_RESULTS}\n"
            f"  DEBUG={self.DEBUG}\n"
            f")"
        )


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get the global configuration instance (singleton pattern).

    Args:
        reload: If True, reload configuration from environment variables.

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config()

    return _config_instance
