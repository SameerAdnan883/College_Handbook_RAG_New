"""
tests/test_v1.py — Unit tests for V1 Basic LangChain
All tests are mock-based and do NOT require a live API key.
"""
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestV1NoApiKey:
    """Tests for behaviour when API key is not configured."""

    def test_ask_returns_dict_structure(self):
        with patch("v1_basic.is_api_configured", return_value=False):
            import v1_basic
            result = v1_basic.ask("What is the attendance policy?")
        assert isinstance(result, dict)
        assert "version" in result
        assert "answer" in result
        assert "error" in result

    def test_ask_returns_v1_version(self):
        with patch("v1_basic.is_api_configured", return_value=False):
            import v1_basic
            result = v1_basic.ask("Test question")
        assert result["version"] == "V1"

    def test_ask_no_api_returns_error(self):
        with patch("v1_basic.is_api_configured", return_value=False):
            import v1_basic
            result = v1_basic.ask("Test question")
        assert result["error"] is not None
        assert "API key" in result["error"] or "not configured" in result["error"]

    def test_build_chain_returns_none_without_api(self):
        with patch("v1_basic.is_api_configured", return_value=False):
            import v1_basic
            chain = v1_basic.build_chain()
        assert chain is None


class TestV1WithMockLLM:
    """Tests for behaviour with mocked LLM."""

    def test_ask_returns_answer(self):
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=MagicMock(
            invoke=MagicMock(return_value="Mocked answer")
        ))

        with patch("v1_basic.is_api_configured", return_value=True), \
             patch("v1_basic.get_llm", return_value=mock_llm), \
             patch("v1_basic.build_chain") as mock_chain:

            mock_chain_obj = MagicMock()
            mock_chain_obj.invoke.return_value = "Mocked answer about attendance"
            mock_chain.return_value = mock_chain_obj

            import v1_basic
            result = v1_basic.ask("What is the attendance policy?", llm=mock_llm)

        assert isinstance(result, dict)

    def test_ask_handles_exception(self):
        with patch("v1_basic.is_api_configured", return_value=True), \
             patch("v1_basic.build_chain") as mock_chain:
            mock_chain_obj = MagicMock()
            mock_chain_obj.invoke.side_effect = Exception("Network error")
            mock_chain.return_value = mock_chain_obj

            import v1_basic
            result = v1_basic.ask("Test question")

        assert result["error"] is not None
        assert "Network error" in result["error"]

    def test_ask_records_latency(self):
        with patch("v1_basic.is_api_configured", return_value=False):
            import v1_basic
            result = v1_basic.ask("test")
        assert "latency_seconds" in result
        assert isinstance(result["latency_seconds"], float)


class TestV1Prompt:
    """Tests for prompt template structure."""

    def test_prompt_has_system_and_human(self):
        import v1_basic
        messages = v1_basic.PROMPT.messages
        assert len(messages) == 2

    def test_prompt_accepts_question_variable(self):
        import v1_basic
        # Should not raise
        value = v1_basic.PROMPT.invoke({"question": "Test?"})
        assert value is not None
