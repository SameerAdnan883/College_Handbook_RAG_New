"""
tests/test_v4.py — Unit tests for V4 Reranking
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker class."""

    def test_rerank_empty_docs_returns_empty(self):
        """Reranking with no docs should return empty list."""
        from v4_reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker()

        with patch.object(reranker, "_load"):
            result = reranker.rerank("test query", docs=[], top_n=3)
        assert result == []

    def test_rerank_returns_top_n(self):
        """Reranker should return at most top_n docs."""
        from v4_reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker()

        mock_docs = []
        for i in range(10):
            doc = MagicMock()
            doc.page_content = f"Document {i} about attendance policy."
            mock_docs.append(doc)

        with patch.object(reranker, "_load"), \
             patch.object(reranker, "_model") as mock_model:
            mock_model.predict.return_value = [float(i) for i in range(10)]
            reranker._model = mock_model
            result = reranker.rerank("attendance", mock_docs, top_n=3)

        assert len(result) == 3

    def test_rerank_result_structure(self):
        """Reranked items should have required fields."""
        from v4_reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker()

        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"

        with patch.object(reranker, "_load"), \
             patch.object(reranker, "_model") as mock_model:
            mock_model.predict.return_value = [0.75]
            reranker._model = mock_model
            result = reranker.rerank("test", [mock_doc], top_n=1)

        assert len(result) == 1
        item = result[0]
        assert "doc" in item
        assert "score" in item
        assert "original_rank" in item
        assert "new_rank" in item

    def test_rerank_orders_by_score_descending(self):
        """Highest-scored docs should be ranked first."""
        from v4_reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker()

        docs = [MagicMock() for _ in range(3)]
        for i, d in enumerate(docs):
            d.page_content = f"Doc {i}"

        scores = [0.1, 0.9, 0.5]  # doc at index 1 should be rank 0

        with patch.object(reranker, "_load"), \
             patch.object(reranker, "_model") as mock_model:
            mock_model.predict.return_value = scores
            reranker._model = mock_model
            result = reranker.rerank("query", docs, top_n=3)

        assert result[0]["original_rank"] == 1  # highest score was index 1
        assert result[0]["score"] == 0.9


class TestV4AskFunction:
    """Test V4 ask() with mocks."""

    def test_ask_no_api_returns_error(self):
        with patch("v4_reranking.is_api_configured", return_value=False):
            import v4_reranking
            result = v4_reranking.ask("Test question")
        assert result["version"] == "V4"
        assert result["error"] is not None

    def test_ask_returns_required_keys(self):
        with patch("v4_reranking.is_api_configured", return_value=False):
            import v4_reranking
            result = v4_reranking.ask("Test question")

        required_keys = [
            "version", "question", "answer", "reranked_docs",
            "sources", "latency_seconds", "retrieval_latency",
            "reranking_latency", "generation_latency", "error",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_ask_accepts_cached_reranker(self):
        """Ensure ask() uses the passed reranker instance."""
        with patch("v4_reranking.is_api_configured", return_value=True), \
             patch("v4_reranking.get_or_build_faiss"), \
             patch("v4_reranking.get_llm"):
            import v4_reranking
            
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = []
            
            # Should not raise any errors, and should use mock_reranker
            v4_reranking.ask("Test question", reranker=mock_reranker)
            mock_reranker.rerank.assert_called_once()


class TestV4ScoreDisclaimer:
    """Ensure scores are not described as probabilities."""

    def test_reranking_explanation_disclaims_logits(self):
        import v4_reranking
        explanation = v4_reranking.RERANKING_EXPLANATION.lower()
        assert "logit" in explanation or "not probabilit" in explanation

    def test_format_reranked_docs_mentions_logit(self):
        import v4_reranking
        mock_item = {
            "doc": MagicMock(
                page_content="Test content",
                metadata={"page": 1, "source_file": "handbook.pdf"},
            ),
            "score": 0.85,
            "new_rank": 0,
            "original_rank": 2,
        }
        result = v4_reranking.format_reranked_docs([mock_item])
        assert "logit" in result.lower()
