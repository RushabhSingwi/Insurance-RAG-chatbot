"""
Text Chunking Module
Handles intelligent text chunking for embeddings
"""

import sys
from typing import List

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Try to import LangChain text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain_text_splitters not installed.")
    print("Install with: pip install langchain-text-splitters")


def split_into_chunks(text: str, max_tokens: int = 800, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into intelligent chunks using LangChain's RecursiveCharacterTextSplitter.
    Falls back to simple chunking if LangChain is not available.

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (~4 chars per token)
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not LANGCHAIN_AVAILABLE:
        # Fallback: simple chunking by characters
        chunk_size = max_tokens * 4
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    chunk_size = max_tokens * 4  # 1 token ≈ 4 characters

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",      # Paragraph breaks
            "\n",        # Line breaks
            ". ",        # Sentences
            "? ",        # Questions
            "! ",        # Exclamations
            "; ",        # Semicolons
            ", ",        # Commas
            " ",         # Words
            "",          # Characters
        ],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)
    return chunks


def main():
    """Chunk all text files in the input directory."""
    from pathlib import Path
    import json

    input_dir = Path("../../data/processed_text")
    output_dir = Path("../../data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No text files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} text files")
    print("="*60)

    total_chunks = 0

    for i, txt_file in enumerate(txt_files, 1):
        try:
            # Read
            text = txt_file.read_text(encoding="utf-8")

            if not text.strip():
                print(f"[{i}/{len(txt_files)}] {txt_file.name[:50]:<50} SKIPPED (empty)")
                continue

            # Chunk
            chunks = split_into_chunks(text, max_tokens=800, chunk_overlap=200)

            # Save as JSON
            output_file = output_dir / f"{txt_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            print(f"[{i}/{len(txt_files)}] {txt_file.name[:50]:<50} {len(chunks):4d} chunks")
            total_chunks += len(chunks)

        except Exception as e:
            print(f"[{i}/{len(txt_files)}] {txt_file.name[:50]:<50} ERROR: {e}")

    print()
    print("="*60)
    print(f"Complete!")
    print(f"  Files: {len(txt_files)}")
    print(f"  Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
