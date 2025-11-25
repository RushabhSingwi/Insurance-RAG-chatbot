"""
PDF Text Extraction Module

Handles PDF to text conversion with OCR support.
Provides functions to extract text from both text-based and scanned PDFs.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from utils.common import (
    setup_windows_console_encoding,
    ensure_directory_exists,
    get_files_with_extension
)

# Fix encoding for Windows console
setup_windows_console_encoding()

# Load configuration
config = get_config()

# Import PDF libraries
try:
    import pdfplumber
    from pdf2image import convert_from_path
    import pytesseract
    PDF_LIBS_AVAILABLE = True
except ImportError:
    PDF_LIBS_AVAILABLE = False
    print("Warning: PDF processing libraries not installed.")
    print("Install with: pip install pdfplumber pdf2image pytesseract")

# Constants
USE_OCR = True
OCR_DPI = 300
OCR_IMAGE_FORMAT = 'png'
OCR_PSM_MODE = '--psm 6'
OCR_LANGUAGES = 'eng+hin'
OCR_LANGUAGE_FALLBACK = 'eng'
MIN_TEXT_LENGTH_FOR_SUCCESS = 50


def remove_devanagari(text: str) -> str:
    """
    Remove Devanagari (Hindi) and other non-English characters from text.

    Args:
        text: Input text containing mixed scripts

    Returns:
        Cleaned text with only English characters and basic punctuation
    """
    # Remove symbols between Hindi characters
    text = re.sub(r'([\u0900-\u097F])\s*[&/\\]\s*([\u0900-\u097F])', r'\1 \2', text)
    text = re.sub(r'([\u0900-\u097F])\s*[^a-zA-Z\s\u0900-\u097F]+\s*([\u0900-\u097F])', r'\1 \2', text)

    # Remove punctuation between Hindi characters
    text = re.sub(r'([\u0900-\u097F])[,.\s]+([\u0900-\u097F])', r'\1\2', text)

    # Remove parentheses containing only Hindi/numbers
    text = re.sub(r'\([\u0900-\u097F0-9\s]+\)', '', text)

    # Remove Devanagari characters
    text = re.sub(r'[\u0900-\u097F]+', '', text)

    # Remove other Indian scripts
    text = re.sub(r'[\u0980-\u0DFF]+', '', text)
    text = re.sub(r'[\u4E00-\u9FFF]+', '', text)  # Chinese
    text = re.sub(r'[\u3040-\u30FF]+', '', text)  # Japanese

    # Remove Unicode replacement characters
    text = re.sub(r'[\uFFFD\uFEFF]+', '', text)
    text = text.replace('�', '')

    # Remove empty containers
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\(\s*["\'\u201C\u201D\u2018\u2019]*\s*\)', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_with_pdfplumber(pdf_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text using pdfplumber (for text-based PDFs).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, method_name) or (None, None) if extraction failed
    """
    if not PDF_LIBS_AVAILABLE:
        return None, None

    try:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = remove_devanagari(page_text)
                    if cleaned_text.strip():
                        text_content.append(cleaned_text)

        if not text_content:
            return None, None

        full_text = "\n\n".join(text_content)

        # Remove text before "Ref " identifier (IRDAI-specific cleanup)
        if "Ref " in full_text:
            ref_index = full_text.find("Ref ")
            full_text = full_text[ref_index:]

        # Remove empty parentheses
        full_text = re.sub(r'\(\s*\)', '', full_text)
        return full_text, "pdfplumber"

    except Exception as e:
        print(f"  pdfplumber extraction failed: {type(e).__name__}: {e}")
        return None, None


def extract_text_with_ocr(pdf_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text from scanned PDFs using OCR (Optical Character Recognition).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, method_name) or (None, None) if extraction failed
    """
    if not PDF_LIBS_AVAILABLE or not USE_OCR:
        return None, None

    try:
        print(f"  Using OCR to extract text...")

        images = convert_from_path(
            pdf_path,
            dpi=OCR_DPI,
            fmt=OCR_IMAGE_FORMAT
        )
        text_content = []

        for i, image in enumerate(images, 1):
            print(f"    OCR processing page {i}/{len(images)}...")

            # Try with bilingual support first, fall back to English only
            try:
                page_text = pytesseract.image_to_string(
                    image,
                    lang=OCR_LANGUAGES,
                    config=OCR_PSM_MODE
                )
            except Exception:
                page_text = pytesseract.image_to_string(
                    image,
                    lang=OCR_LANGUAGE_FALLBACK,
                    config=OCR_PSM_MODE
                )

            cleaned_text = remove_devanagari(page_text)
            if cleaned_text.strip():
                text_content.append(cleaned_text)

        if not text_content:
            return None, None

        full_text = "\n\n".join(text_content)
        print(f"  OCR extracted {len(full_text):,} characters")
        return full_text, "OCR"

    except Exception as e:
        print(f"  OCR extraction failed: {type(e).__name__}: {e}")
        return None, None


def extract_text_from_pdf(
    pdf_path: Path,
    use_ocr: bool = USE_OCR
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text from PDF using the best available method.

    Tries pdfplumber first for text-based PDFs, then falls back to OCR
    for scanned PDFs if the initial extraction yields insufficient text.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR as fallback (default: True)

    Returns:
        Tuple of (extracted_text, method_name) or (None, None) if all methods failed
    """
    # Try pdfplumber first (faster for text-based PDFs)
    extracted_text, method = extract_text_with_pdfplumber(pdf_path)

    # Fall back to OCR if extraction failed or yielded too little text
    if use_ocr and (not extracted_text or len(extracted_text.strip()) < MIN_TEXT_LENGTH_FOR_SUCCESS):
        extracted_text, method = extract_text_with_ocr(pdf_path)

    return extracted_text, method


def main() -> None:
    """
    Process all PDFs in the configured input directory.

    Extracts text from each PDF file and saves it to the output directory.
    Prints progress and summary statistics.
    """
    input_dir = config.RAW_PDFS_DIR
    output_dir = config.PROCESSED_TEXT_DIR
    ensure_directory_exists(output_dir)

    pdf_files = get_files_with_extension(input_dir, '.pdf')

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)

    success_count = 0
    failed_files = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name[:50]}")

        text, method = extract_text_from_pdf(pdf_file)

        if text and len(text.strip()) > 0:
            output_file = output_dir / f"{pdf_file.stem}.txt"
            output_file.write_text(text, encoding='utf-8')
            print(f"  ✓ Extracted {len(text):,} chars using {method}\n")
            success_count += 1
        else:
            print(f"  ✗ Failed to extract text\n")
            failed_files.append(pdf_file.name)

    print("=" * 60)
    print(f"Complete: {success_count}/{len(pdf_files)} PDFs processed successfully")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename in failed_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
