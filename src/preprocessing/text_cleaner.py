"""
Advanced Text Cleaning Module
Handles aggressive text cleaning and metadata extraction
"""

import re
import unicodedata
from typing import Dict


def remove_hindi_ultra_aggressive(text):
    """
    Ultra-aggressive cleaning: keeps ONLY English letters, numbers, and basic punctuation.
    """
    # Normalize Unicode
    text = unicodedata.normalize('NFKD', text)

    # Remove combining characters
    text = ''.join(char for char in text if not unicodedata.combining(char))

    # Keep only safe ASCII characters
    safe_chars = []
    for char in text:
        if char.isascii() and (char.isalnum() or char.isspace() or char in '.,;:!?()[]{}\'"-/&@#%$\n'):
            safe_chars.append(char)

    return ''.join(safe_chars)


def filter_lines_with_english(text, min_english_ratio=0.5):
    """
    Keep only lines with sufficient English content.

    Args:
        text: Input text
        min_english_ratio: Minimum ratio of English characters to total characters

    Returns:
        Filtered text with only English-heavy lines
    """
    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        english_count = len(re.findall(r'[a-zA-Z]', line))
        total_count = len(re.sub(r'\s', '', line))

        if total_count == 0:
            continue

        # Check for English words (3+ letters)
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', line)

        # Keep if metadata or has good English content
        is_metadata = bool(re.match(r'^(Ref|Date|Subject|Circular|To|From|Re):', line, re.IGNORECASE))

        if total_count > 0:
            english_ratio = english_count / total_count

            if english_ratio >= min_english_ratio or len(english_words) >= 3 or is_metadata:
                filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def extract_metadata(text) -> Dict:
    """
    Extract key metadata from document.

    Returns:
        Dictionary with 'ref', 'date', and 'subject' keys
    """
    metadata = {}

    # Circular/Ref number
    ref_match = re.search(r'(IRDAI[\/A-Z0-9\-]+)', text, re.IGNORECASE)
    if ref_match:
        metadata['ref'] = ref_match.group(1)

    # Date
    date_match = re.search(
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*\d{4})',
        text,
        re.IGNORECASE
    )
    if date_match:
        metadata['date'] = date_match.group(1)

    # Subject
    subject_match = re.search(r'(?:Re|Sub|Subject):\s*([A-Za-z0-9\s\-,/]+?)(?:\n\n|\.(?:\s|$))', text, re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()
        if re.match(r'^[a-zA-Z0-9\s\-,/\.]+$', subject):
            metadata['subject'] = subject[:200]

    return metadata


def clean_text_advanced(text):
    """
    Advanced cleaning pipeline:
    1. Extract metadata
    2. Remove Hindi/special characters
    3. Filter lines with insufficient English
    4. Clean up whitespace

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text with metadata prepended
    """
    # Extract metadata first
    metadata = extract_metadata(text)

    # Remove Hindi and special characters
    text = remove_hindi_ultra_aggressive(text)

    # Filter lines
    text = filter_lines_with_english(text, min_english_ratio=0.5)

    # Additional cleanup
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        single_char_count = len(re.findall(r'\s[a-zA-Z]\s', line))
        if single_char_count < 8:
            clean_lines.append(line)

    text = '\n'.join(clean_lines)

    # Remove Hindi transliterations (consonant clusters)
    text = re.sub(r'\b[bcdfghjklmnpqrstvwxyz]{5,}\b', '', text, flags=re.IGNORECASE)

    # Remove empty containers
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    # Clean whitespace
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    # Format output with metadata
    output_parts = []

    if metadata.get('ref'):
        output_parts.append(f"Ref: {metadata['ref']}")
    if metadata.get('date'):
        output_parts.append(f"Date: {metadata['date']}")
    if metadata.get('subject'):
        output_parts.append(f"Subject: {metadata['subject']}")

    if output_parts:
        output_parts.append('')

    output_parts.append(text)

    return '\n'.join(output_parts)


def main():
    """Clean all text files in the input directory."""
    from pathlib import Path

    input_dir = Path("../../data/processed_text")

    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        return

    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No text files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} text files")
    print("="*60)

    total_original = 0
    total_cleaned = 0

    for i, txt_file in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] {txt_file.name[:50]:<50}", end=" ")

        try:
            # Read
            original = txt_file.read_text(encoding='utf-8', errors='ignore')

            # Clean
            cleaned = clean_text_advanced(original)

            # Write back
            txt_file.write_text(cleaned, encoding='utf-8')

            # Stats
            orig_len = len(original)
            clean_len = len(cleaned)
            total_original += orig_len
            total_cleaned += clean_len

            reduction = ((orig_len - clean_len) / orig_len * 100) if orig_len > 0 else 0
            print(f"{orig_len:6,} → {clean_len:6,} ({reduction:5.1f}%)")

        except Exception as e:
            print(f"ERROR: {e}")

    print()
    print("="*60)
    print(f"Complete!")
    print(f"  Original: {total_original:,} chars")
    print(f"  Cleaned:  {total_cleaned:,} chars")
    print(f"  Reduction: {((total_original - total_cleaned) / total_original * 100):.1f}%")


if __name__ == "__main__":
    main()
