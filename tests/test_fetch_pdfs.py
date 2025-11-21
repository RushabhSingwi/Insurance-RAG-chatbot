"""
Unit tests for the fetch_pdfs module.
Tests the clean_english_filename function with various inputs.
"""
import pytest
import sys
import os

# Add the project root (which contains `src`) to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.downloader.fetch_pdfs import clean_english_filename


class TestCleanEnglishFilename:
    """Test cases for the clean_english_filename function."""

    def test_removes_hindi_characters(self):
        """Test that Hindi characters are removed and English portions are extracted."""
        # Test with Hindi text followed by English
        input_text = "2001-02 में जारी किये गये श्रेणीकरण पत्र के आधार पर सर्वेक्षकों और हानि निर्धारकों को व्यवसाय_विभाग की अतिरिक्त व्यवस्था की अनुमति की प्रथा का समापन_Cessation of the practice of allowing additional line of business_department to SLA based on categoriz.pdf"
        result = clean_english_filename(input_text)
        assert result == "2001-02 _ _Cessation of the practice of allowing additional line of business_department to SLA based on categoriz.pdf"
        
        # Test with only Hindi characters
        input_text = "बीमा परिपत्र"
        result = clean_english_filename(input_text)
        assert result == ".pdf"
        
        # Test with mixed Hindi and English without separators
        input_text = "बीमा Insurance Document 2024"
        result = clean_english_filename(input_text)
        assert result == "Insurance Document 2024.pdf"

    def test_handles_underscore_separator(self):
        """Test that filenames with ' _ ' separator are handled correctly."""
        # Test underscore separator with English after separator
        input_text = "हिंदी टेक्स्ट _ Guidelines for Health Insurance"
        result = clean_english_filename(input_text)
        assert result == "Guidelines for Health Insurance.pdf"
        
        # Test underscore separator with English before separator
        input_text = "Guidelines for Health Insurance _ हिंदी टेक्स्ट"
        result = clean_english_filename(input_text)
        assert result == "Guidelines for Health Insurance.pdf"
        
        # Test multiple underscore separators - takes first part with >5 English chars
        input_text = "Part1 _ हिंदी _ Final Document"
        result = clean_english_filename(input_text)
        assert result == "Final Document.pdf"  # "Part1" has only 5 chars, "Final Document" has more

    def test_handles_slash_separator(self):
        """Test that filenames with ' / ' separator are handled correctly."""
        # Test slash separator with English after separator
        input_text = "नीति / Policy Document 2024"
        result = clean_english_filename(input_text)
        assert result == "Policy Document 2024.pdf"
        
        # Test slash separator with English before separator
        input_text = "Regulatory Circular / विनियामक"
        result = clean_english_filename(input_text)
        assert result == "Regulatory Circular.pdf"
        
        # Test multiple slash separators
        input_text = "Part A / खंड बी / Section C Document"
        result = clean_english_filename(input_text)
        assert result == "Section C Document.pdf"

    def test_removes_existing_pdf_extension(self):
        """Test that existing .pdf extensions are removed before processing."""
        input_text = "Insurance Guidelines.pdf"
        result = clean_english_filename(input_text)
        assert result == "Insurance Guidelines.pdf"
        assert result.count(".pdf") == 1  # Should only have one .pdf at the end
        
        # Test with .pdf in the middle - function removes .pdf first, then "Document" has 8 chars (>5)
        input_text = "Document.pdf _ Final Version"
        result = clean_english_filename(input_text)
        # After removing .pdf: "Document _ Final Version" -> "Document" has >5 chars, so it's selected
        assert result == "Document.pdf"

    def test_cleans_special_characters(self):
        """Test that non-English special characters are removed."""
        # Test with various special characters and symbols
        input_text = "Insurance@#$%^&* Document"
        result = clean_english_filename(input_text)
        assert result == "Insurance Document.pdf"
        
        # Test with preserved special characters (hyphens, parentheses, commas, periods)
        input_text = "Life-Insurance (Policy-2024), Guidelines."
        result = clean_english_filename(input_text)
        assert result == "Life-Insurance (Policy-2024), Guidelines..pdf"

    def test_handles_multiple_spaces(self):
        """Test that multiple consecutive spaces are cleaned up."""
        input_text = "Insurance    Document     2024"
        result = clean_english_filename(input_text)
        assert result == "Insurance Document 2024.pdf"
        
        # Test with tabs and multiple spaces
        input_text = "Policy  \t  Guidelines   Document"
        result = clean_english_filename(input_text)
        assert result == "Policy Guidelines Document.pdf"

    def test_handles_english_only_filenames(self):
        """Test that pure English filenames pass through correctly."""
        input_text = "Health Insurance Guidelines 2024"
        result = clean_english_filename(input_text)
        assert result == "Health Insurance Guidelines 2024.pdf"

    def test_handles_filenames_with_numbers(self):
        """Test that filenames with numbers are handled correctly."""
        input_text = "Circular-123-2024"
        result = clean_english_filename(input_text)
        assert result == "Circular-123-2024.pdf"
        
        # Test with Hindi and numbers
        input_text = "परिपत्र 123 _ Circular 456 of 2024"
        result = clean_english_filename(input_text)
        assert result == "Circular 456 of 2024.pdf"

    def test_handles_empty_or_minimal_english_content(self):
        """Test edge cases with minimal or no English content."""
        # Test with very short English content (below threshold of >5)
        input_text = "हिंदी टेक्स्ट _ ABC"
        result = clean_english_filename(input_text)
        # "ABC" has only 3 English chars, not >5, so no part is selected
        # Function then cleans the whole original string
        assert result == "_ ABC.pdf"  # Hindi removed, underscore and ABC remain
        
        # Test with just enough English content (>5 chars)
        input_text = "हिंदी _ ABCDEF"
        result = clean_english_filename(input_text)
        assert result == "ABCDEF.pdf"  # 6 chars, which is >5

    def test_strips_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        input_text = "   Insurance Document   "
        result = clean_english_filename(input_text)
        assert result == "Insurance Document.pdf"
        assert not result.startswith(" ")
        assert not result.endswith(" .pdf")

    def test_handles_both_separators_in_same_filename(self):
        """Test filenames that contain both ' _ ' and ' / ' separators."""
        # Underscore appears first
        input_text = "हिंदी _ English Text / और हिंदी"
        result = clean_english_filename(input_text)
        assert result == "English Text.pdf"  # Underscore is checked first
        
        # Slash appears first (in string, but underscore is checked first in code)
        input_text = "हिंदी / खंड _ Important Guidelines"
        result = clean_english_filename(input_text)
        assert result == "Important Guidelines.pdf"

    def test_real_world_examples(self):
        """Test with real-world-like filename examples."""
        # Example 1: IRDAI circular
        input_text = "विनियामक परिपत्र _ Guidelines on Corporate Governance"
        result = clean_english_filename(input_text)
        assert result == "Guidelines on Corporate Governance.pdf"
        
        # Example 2: Policy document
        input_text = "जीवन बीमा / Life Insurance Regulations (Amendment) 2024"
        result = clean_english_filename(input_text)
        assert result == "Life Insurance Regulations (Amendment) 2024.pdf"
        
        # Example 3: Complex mixed content
        # After removing .pdf: "IRDAI-हिंदी-REG-2024 _ Master Circular on Investment"
        # First part "IRDAI-हिंदी-REG-2024" has 10 English chars (IRDAIREG2024), which is >5
        # So it takes the first part, then removes Hindi
        input_text = "IRDAI-हिंदी-REG-2024 _ Master Circular on Investment.pdf"
        result = clean_english_filename(input_text)
        assert result == "IRDAI--REG-2024.pdf"  # Hindi removed from first part
