"""
tests/test_v3.py — Unit tests for V3 ChromaDB RAG
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestV3ChunkId:
    """Test deduplication ID generation."""

    def test_chunk_id_is_deterministic(self):
        import v3_chroma
        id1 = v3_chroma._chunk_id("Same text", 5, "handbook.pdf")
        id2 = v3_chroma._chunk_id("Same text", 5, "handbook.pdf")
        assert id1 == id2

    def test_different_content_different_id(self):
        import v3_chroma
        id1 = v3_chroma._chunk_id("Content A", 1, "handbook.pdf")
        id2 = v3_chroma._chunk_id("Content B", 1, "handbook.pdf")
        assert id1 != id2

    def test_different_page_different_id(self):
        import v3_chroma
        id1 = v3_chroma._chunk_id("Same content", 1, "handbook.pdf")
        id2 = v3_chroma._chunk_id("Same content", 2, "handbook.pdf")
        assert id1 != id2


class TestV3AskFunction:
    """Test V3 ask() with mocks."""

    def test_ask_no_api_returns_error(self):
        with patch("v3_chroma.is_api_configured", return_value=False):
            import v3_chroma
            result = v3_chroma.ask("Test question")
        assert result["version"] == "V3"
        assert result["error"] is not None

    def test_ask_returns_required_keys(self):
        with patch("v3_chroma.is_api_configured", return_value=False):
            import v3_chroma
            result = v3_chroma.ask("Test question")

        required_keys = [
            "version", "question", "answer", "sources",
            "context_docs", "latency_seconds", "error",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_format_docs_includes_chapter(self):
        import v3_chroma
        mock_doc = MagicMock()
        mock_doc.page_content = "Library borrowing rules."
        mock_doc.metadata = {
            "page": 8,
            "source": "handbook.pdf",
            "chapter": "library_policy",
        }
        result = v3_chroma.format_docs([mock_doc])
        assert "library_policy" in result
        assert "Library borrowing rules." in result


class TestV3FAISSComparison:
    """Test that the FAISS vs ChromaDB comparison text is present."""

    def test_comparison_text_exists(self):
        import v3_chroma
        assert hasattr(v3_chroma, "FAISS_VS_CHROMA")
        assert "FAISS" in v3_chroma.FAISS_VS_CHROMA
        assert "ChromaDB" in v3_chroma.FAISS_VS_CHROMA

    def test_no_accuracy_claim(self):
        """Verify we don't claim ChromaDB is inherently more accurate."""
        import v3_chroma
        text = v3_chroma.FAISS_VS_CHROMA.lower()
        # Should NOT claim ChromaDB is "inherently more accurate"
        assert "inherently more accurate" not in text
