"""
Common Utilities Module

This module provides commonly used utility functions across the codebase
to avoid code duplication and ensure consistent behavior.
"""

import io
import sys
from pathlib import Path
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def setup_windows_console_encoding() -> None:
    """
    Fix encoding for Windows console to handle UTF-8 properly.
    Safe to call on any platform - only applies changes on Windows.
    """
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'
        )


def create_embedding_model(provider: str, model_name: str, api_key: Optional[str] = None):
    """
    Create an embedding model instance based on provider type.

    Args:
        provider: Embedding provider ('openai' or 'huggingface')
        model_name: Model identifier
        api_key: API key (required for OpenAI, ignored for HuggingFace)

    Returns:
        Embedding model instance

    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")

        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )

    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def get_files_with_extension(directory: Path, extension: str) -> List[Path]:
    """
    Get all files with a specific extension in a directory.

    Args:
        directory: Directory to search
        extension: File extension (e.g., '.json', '.pdf')

    Returns:
        List of Path objects matching the extension
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'

    return sorted(directory.glob(f'*{extension}'))
