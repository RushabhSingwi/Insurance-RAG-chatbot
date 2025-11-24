"""
PDF Text Extraction Module
Handles PDF to text conversion with OCR support
"""

import os
import re
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

# Configuration
USE_OCR = True  # Set to False to skip OCR for scanned PDFs


def remove_devanagari(text):
    """
    Remove Devanagari (Hindi) characters and keep only English text.
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


def extract_text_with_pdfplumber(pdf_path):
    """
    Extract text using pdfplumber (for text-based PDFs).

    Returns:
        tuple: (extracted_text, method_name) or (None, None) if failed
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

        if text_content:
            full_text = "\n\n".join(text_content)

            # Remove text before "Ref " identifier
            if "Ref " in full_text:
                ref_index = full_text.find("Ref ")
                full_text = full_text[ref_index:]

            full_text = re.sub(r'\(\s*\)', '', full_text)
            return full_text, "pdfplumber"
        return None, None
    except Exception as e:
        print(f"  pdfplumber failed: {e}")
        return None, None


def extract_text_with_ocr(pdf_path):
    """
    Extract text from scanned PDFs using OCR.

    Returns:
        tuple: (extracted_text, method_name) or (None, None) if failed
    """
    if not PDF_LIBS_AVAILABLE or not USE_OCR:
        return None, None

    try:
        print(f"  Using OCR to extract text...")

        images = convert_from_path(pdf_path, dpi=300, fmt='png')
        text_content = []

        for i, image in enumerate(images, 1):
            print(f"    OCR processing page {i}/{len(images)}...")

            try:
                page_text = pytesseract.image_to_string(
                    image,
                    lang='eng+hin',
                    config='--psm 6'
                )
            except:
                page_text = pytesseract.image_to_string(
                    image,
                    lang='eng',
                    config='--psm 6'
                )

            cleaned_text = remove_devanagari(page_text)
            if cleaned_text.strip():
                text_content.append(cleaned_text)

        if text_content:
            full_text = "\n\n".join(text_content)
            print(f"  OCR extracted {len(full_text)} characters")
            return full_text, "OCR"

        return None, None
    except Exception as e:
        print(f"  OCR failed: {type(e).__name__}: {e}")
        return None, None


def extract_text_from_pdf(pdf_path, use_ocr=USE_OCR):
    """
    Extract text from PDF using best available method.
    Tries pdfplumber first, falls back to OCR if needed.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR as fallback

    Returns:
        tuple: (extracted_text, method_name) or (None, None) if failed
    """
    # Try pdfplumber first
    extracted_text, method = extract_text_with_pdfplumber(pdf_path)

    # Fall back to OCR if needed
    if (not extracted_text or len(extracted_text.strip()) < 50) and use_ocr:
        extracted_text, method = extract_text_with_ocr(pdf_path)

    return extracted_text, method


def main():
    """Process all PDFs in the input directory."""
    from pathlib import Path
    import json

    input_dir = Path("../../data/raw_pdfs")
    output_dir = Path("../../data/processed_text")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print("="*60)

    success_count = 0
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

    print("="*60)
    print(f"Complete: {success_count}/{len(pdf_files)} PDFs processed")


if __name__ == "__main__":
    main()
