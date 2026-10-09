"""
tests/test_v2.py — Unit tests for V2 FAISS RAG
All tests use mocks for API calls; PDF loading tested with temp file.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestV2DocumentLoading:
    """Test PDF loading and chunking."""

    def test_load_and_split_raises_without_pdf(self):
        import v2_faiss
        with pytest.raises(FileNotFoundError):
            v2_faiss.load_and_split(pdf_path="/nonexistent/path.pdf")

    def test_format_docs_with_empty_list(self):
        import v2_faiss
        result = v2_faiss.format_docs([])
        assert result == ""

    def test_format_docs_with_mock_doc(self):
        import v2_faiss
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content about attendance."
        mock_doc.metadata = {"page": 5, "source_file": "handbook.pdf"}

        result = v2_faiss.format_docs([mock_doc])
        assert "Test content about attendance." in result
        assert "Page 5" in result
        assert "handbook.pdf" in result


class TestV2AskFunction:
    """Test the ask() function with mocks."""

    def test_ask_no_api(self):
        with patch("v2_faiss.is_api_configured", return_value=False):
            import v2_faiss
            result = v2_faiss.ask("Test question")
        assert result["version"] == "V2"
        assert result["error"] is not None

    def test_ask_returns_required_keys(self):
        with patch("v2_faiss.is_api_configured", return_value=False):
            import v2_faiss
            result = v2_faiss.ask("Test question")

        required_keys = [
            "version", "question", "answer", "sources",
            "context_docs", "latency_seconds",
            "retrieval_latency", "generation_latency", "error",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_ask_with_mock_db(self):
        """Test full ask() flow with fully mocked db and LLM."""
        mock_doc = MagicMock()
        mock_doc.page_content = "75% attendance required."
        mock_doc.metadata = {"page": 3, "source_file": "handbook.pdf"}

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_parser = MagicMock()
        mock_parser.invoke.return_value = "The minimum attendance is 75%."
        mock_llm.__or__ = MagicMock(return_value=mock_parser)

        with patch("v2_faiss.is_api_configured", return_value=True), \
             patch("v2_faiss.get_llm", return_value=mock_llm), \
             patch("v2_faiss.StrOutputParser") as mock_sop:

            mock_sop_instance = MagicMock()
            mock_sop_instance.invoke.return_value = "The minimum attendance is 75%."
            mock_sop.return_value = mock_sop_instance

            import v2_faiss

            # Manually call with mock db
            with patch.object(mock_llm, "__or__", return_value=mock_sop_instance):
                result = v2_faiss.ask("What is the attendance requirement?", db=mock_db)

        assert result["version"] == "V2"
        assert isinstance(result["context_docs"], list)


class TestV2EmbeddingsNoAPI:
    """Test that embeddings fail gracefully without API key."""

    def test_get_embeddings_raises_without_api(self):
        with patch("v2_faiss.is_api_configured", return_value=False):
            import v2_faiss
            with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
                v2_faiss.get_embeddings()
