"""
Unit tests for the pdf_to_text_ocr module.
Tests text cleaning and extraction functions.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Add the project root (which contains `src`) to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Save original stdout before importing pdf_to_text_ocr (which modifies it on Windows)
_original_stdout = sys.stdout

from src.preprocessing.pdf_to_text_ocr import (
    remove_devanagari,
    clean_ocr_text,
    extract_text_from_pdf,
    extract_text_with_pdfplumber,
    extract_text_with_ocr
)

# Restore stdout for pytest to work properly
sys.stdout = _original_stdout


class TestRemoveDevanagari:
    """Test cases for the remove_devanagari function."""

    def test_removes_devanagari_characters(self):
        """Test that Devanagari (Hindi) characters are effectively removed."""
        # Test with Hindi text
        text = "यह हिंदी टेक्स्ट है"
        result = remove_devanagari(text)
        assert result == ""
        
        # Test with mixed Hindi and English
        text = "बीमा Insurance परिपत्र Circular"
        result = remove_devanagari(text)
        assert "बीमा" not in result
        assert "परिपत्र" not in result
        assert "Insurance" in result
        assert "Circular" in result

    def test_preserves_english_text(self):
        """Test that English text is preserved while removing Devanagari."""
        # Test with pure English
        text = "This is an insurance policy document"
        result = remove_devanagari(text)
        assert result == "This is an insurance policy document"
        
        # Test with English and numbers
        text = "Policy 2024 Guidelines Section 10"
        result = remove_devanagari(text)
        assert result == "Policy 2024 Guidelines Section 10"

    def test_cleans_up_whitespace(self):
        """Test that extra whitespace is cleaned up."""
        # Multiple spaces from removed text
        text = "Word1    बीमा    Word2"
        result = remove_devanagari(text)
        assert result == "Word1 Word2"
        
        # Leading/trailing whitespace
        text = "   परिपत्र   English text   "
        result = remove_devanagari(text)
        assert result == "English text"

    def test_handles_empty_and_whitespace_only(self):
        """Test edge cases with empty or whitespace-only input."""
        assert remove_devanagari("") == ""
        assert remove_devanagari("   ") == ""
        assert remove_devanagari("\n\t  ") == ""

    def test_preserves_special_characters_and_punctuation(self):
        """Test that special characters in English text are preserved."""
        text = "Policy (2024-25), Section 10.5: Guidelines & Rules"
        result = remove_devanagari(text)
        assert result == text


class TestCleanOcrText:
    """Test cases for the clean_ocr_text function."""

    def test_filters_gibberish_lines(self):
        """Test that gibberish and low-quality lines are filtered out."""
        # Gibberish with high special character ratio
        text = "Good English line\n@#$%^&*()_+{}|:\n<>?~`\nAnother good line"
        result = clean_ocr_text(text)
        assert "Good English line" in result
        assert "Another good line" in result
        assert "@#$%^&*" not in result

    def test_removes_lines_with_low_alpha_ratio(self):
        """Test that lines with low alphabetic character ratio are removed."""
        text = "Valid insurance document\n!!!###$$$\n***&&&%%%\nAnother valid line"
        result = clean_ocr_text(text)
        assert "Valid insurance document" in result
        assert "Another valid line" in result
        # Lines with mostly special characters should be removed
        assert "!!!" not in result
        assert "***" not in result

    def test_keeps_lines_with_numbers_and_letters(self):
        """Test that lines with numbers and sufficient letters are kept."""
        text = "Policy Number: 12345\nSection 10.5 Guidelines\n1234567890"
        result = clean_ocr_text(text)
        assert "Policy Number: 12345" in result
        assert "Section 10.5 Guidelines" in result
        # Line with only numbers should be removed (below 30% alpha threshold)
        assert "1234567890" not in result
    def test_removes_devanagari_first(self):
        """Test that Devanagari characters are removed before quality filtering."""
        text = "English text\nबीमा हिंदी टेक्स्ट\nMore English text"
        result = clean_ocr_text(text)
        assert "बीमा" not in result
        assert "हिंदी" not in result
        assert "English text" in result
        assert "More English text" in result

    def test_cleans_excessive_special_characters(self):
        """Test that excessive special characters from OCR errors are cleaned."""
        text = "Insurance....policy---document,,,guidelines"
        result = clean_ocr_text(text)
        # Should reduce consecutive special chars
        assert result.count("....") == 0
        assert result.count("---") == 0
        assert result.count(",,,") == 0

    def test_preserves_valid_punctuation(self):
        """Test that valid punctuation is preserved."""
        text = "Section 10.5: Insurance Guidelines (2024-25), Rules & Regulations."
        result = clean_ocr_text(text)
        # Valid punctuation should be preserved
        assert ":" in result
        assert "(" in result
        assert ")" in result
        assert "," in result
        assert "." in result

    def test_removes_empty_lines(self):
        """Test that empty lines are removed."""
        text = "Line 1\n\n\nLine 2\n  \n\t\nLine 3"
        result = clean_ocr_text(text)
        # Should not have excessive blank lines
        assert "\n\n\n" not in result

    def test_handles_mixed_quality_ocr_output(self):
        """Test with realistic mixed-quality OCR output."""
        text = """
        Insurance Regulatory Authority
        बीमा नियामक
        ~!@#$%^&*()_+
        Chapter 1: Introduction
        Policy Number: ABC-2024-001
        ###$$$%%%
        Section 10: Guidelines and Rules
        """.strip()
        
        result = clean_ocr_text(text)
        assert "Insurance Regulatory Authority" in result
        assert "Chapter 1: Introduction" in result
        assert "Policy Number: ABC-2024-001" in result
        assert "Section 10: Guidelines and Rules" in result
        assert "बीमा" not in result
        assert "~!@#$%^&*" not in result

    def test_minimum_alpha_threshold(self):
        """Test the 30% alphabetic character threshold."""
        text = "abc12345678"  # 3 alpha, 7 digits, total 10 -> 30%
        result = clean_ocr_text(text)
        assert result == ""

    def test_preserves_numbers_with_context(self):
        """Test that numbers are preserved when they have contextual letters."""
        text = "Policy 2024\nCircular 123\nGuideline 456 Section A"
        result = clean_ocr_text(text)
        assert "2024" in result
        assert "123" in result
        assert "456" in result


class TestExtractTextFromPdf:
    """Test cases for the extract_text_from_pdf function."""

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_uses_pdfplumber_first(self, mock_ocr, mock_pdfplumber):
        """Test that pdfplumber is tried first for text extraction."""
        mock_pdfplumber.return_value = "Extracted text from pdfplumber" * 10
        mock_ocr.return_value = "OCR text"
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        # Should call pdfplumber first
        mock_pdfplumber.assert_called_once_with("dummy.pdf")
        # Should not call OCR since pdfplumber succeeded
        mock_ocr.assert_not_called()
        assert result == "Extracted text from pdfplumber" * 10
        assert method == "pdfplumber"

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_falls_back_to_ocr_when_pdfplumber_fails(self, mock_ocr, mock_pdfplumber):
        """Test that OCR is used as fallback when pdfplumber fails."""
        mock_pdfplumber.return_value = None
        mock_ocr.return_value = "OCR extracted text"
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        # Should try pdfplumber first
        mock_pdfplumber.assert_called_once_with("dummy.pdf")
        # Should fall back to OCR
        mock_ocr.assert_called_once_with("dummy.pdf")
        assert result == "OCR extracted text"
        assert method == "OCR"

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_falls_back_to_ocr_when_insufficient_text(self, mock_ocr, mock_pdfplumber):
        """Test that OCR is used when pdfplumber returns insufficient text."""
        # Return text below the 50-character threshold
        mock_pdfplumber.return_value = "Short"
        mock_ocr.return_value = "Much longer OCR extracted text from scanned document"
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        # Should try pdfplumber first
        mock_pdfplumber.assert_called_once_with("dummy.pdf")
        # Should fall back to OCR due to insufficient text
        mock_ocr.assert_called_once_with("dummy.pdf")
        assert result == "Much longer OCR extracted text from scanned document"
        assert method == "OCR"

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_skips_pdfplumber_when_use_ocr_true(self, mock_ocr, mock_pdfplumber):
        """Test that pdfplumber is skipped when use_ocr=True."""
        mock_ocr.return_value = "OCR extracted text"
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=True)
        
        # Should not call pdfplumber
        mock_pdfplumber.assert_not_called()
        # Should go straight to OCR
        mock_ocr.assert_called_once_with("dummy.pdf")
        assert result == "OCR extracted text"
        assert method == "OCR"

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_returns_none_when_both_methods_fail(self, mock_ocr, mock_pdfplumber):
        """Test that None is returned when both extraction methods fail."""
        mock_pdfplumber.return_value = None
        mock_ocr.return_value = None
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        assert result is None
        assert method is None

    @patch('src.preprocessing.pdf_to_text_ocr.pdfplumber')
    def test_extract_text_with_pdfplumber_success(self, mock_pdfplumber):
        """Test successful text extraction using pdfplumber."""
        # Mock pdfplumber behavior
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content with English text"
        mock_page2.extract_text.return_value = "Page 2 content बीमा with Hindi"
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = extract_text_with_pdfplumber("dummy.pdf")
        
        assert result is not None
        assert "Page 1 content with English text" in result
        assert "Page 2 content" in result
        # Hindi should be removed
        assert "बीमा" not in result

    @patch('src.preprocessing.pdf_to_text_ocr.pdfplumber')
    def test_extract_text_with_pdfplumber_no_text(self, mock_pdfplumber):
        """Test pdfplumber returns None when no text is found."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = extract_text_with_pdfplumber("dummy.pdf")
        
        assert result is None

    @patch('src.preprocessing.pdf_to_text_ocr.pdfplumber')
    def test_extract_text_with_pdfplumber_exception(self, mock_pdfplumber):
        """Test pdfplumber handles exceptions gracefully."""
        mock_pdfplumber.open.side_effect = Exception("PDF corrupted")
        
        result = extract_text_with_pdfplumber("dummy.pdf")
        
        assert result is None

    @patch('src.preprocessing.pdf_to_text_ocr.convert_from_path')
    @patch('src.preprocessing.pdf_to_text_ocr.pytesseract')
    def test_extract_text_with_ocr_success(self, mock_pytesseract, mock_convert):
        """Test successful OCR extraction."""
        # Mock image conversion
        mock_image1 = MagicMock()
        mock_image2 = MagicMock()
        mock_convert.return_value = [mock_image1, mock_image2]
        
        # Mock OCR extraction
        mock_pytesseract.image_to_string.side_effect = [
            "Page 1 OCR text with बीमा Hindi",
            "Page 2 OCR text English only"
        ]
        
        result = extract_text_with_ocr("dummy.pdf")
        
        assert result is not None
        # Hindi should be filtered by clean_ocr_text
        assert "Page 1 OCR text" in result or "OCR text" in result
        assert "Page 2 OCR text English only" in result

    @patch('src.preprocessing.pdf_to_text_ocr.convert_from_path')
    @patch('src.preprocessing.pdf_to_text_ocr.pytesseract')
    def test_extract_text_with_ocr_exception(self, mock_pytesseract, mock_convert):
        """Test OCR handles exceptions gracefully."""
        mock_convert.side_effect = Exception("PDF conversion failed")
        
        result = extract_text_with_ocr("dummy.pdf")
        
        assert result is None

    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_pdfplumber')
    @patch('src.preprocessing.pdf_to_text_ocr.extract_text_with_ocr')
    def test_text_length_threshold(self, mock_ocr, mock_pdfplumber):
        """Test the 50-character minimum threshold for pdfplumber."""
        # Exactly 50 characters (should not trigger OCR)
        text_50 = "a" * 50
        mock_pdfplumber.return_value = text_50
        mock_ocr.return_value = "OCR text"
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        # Should not call OCR since we have exactly 50 chars
        mock_ocr.assert_not_called()
        assert method == "pdfplumber"
        
        # Reset mocks
        mock_pdfplumber.reset_mock()
        mock_ocr.reset_mock()
        
        # Less than 50 characters (should trigger OCR)
        text_49 = "a" * 49
        mock_pdfplumber.return_value = text_49
        
        result, method = extract_text_from_pdf("dummy.pdf", use_ocr=False)
        
        # Should fall back to OCR
        mock_ocr.assert_called_once()
        assert method == "OCR"
