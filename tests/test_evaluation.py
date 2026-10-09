"""
tests/test_evaluation.py — Unit tests for evaluation module
"""
import os
import sys
import pytest
import tempfile
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestKeywordScore:
    """Test keyword_score function."""

    def test_full_match(self):
        from evaluation import keyword_score
        # Both keywords present in the answer
        score = keyword_score(
            "The minimum attendance is 75% which equals 75 percent of all classes.",
            ["75%", "75 percent"],
        )
        assert score == 1.0

    def test_partial_match(self):
        from evaluation import keyword_score
        # Only one of three keywords matches → partial score
        score = keyword_score(
            "Students need 75% attendance to appear in exams.",
            ["75%", "automatic f grade", "next academic year"],
        )
        assert 0 < score < 1.0

    def test_no_match(self):
        from evaluation import keyword_score
        score = keyword_score(
            "The university has great facilities.",
            ["75%", "attendance requirement"],
        )
        assert score == 0.0

    def test_api_failure_returns_none(self):
        from evaluation import keyword_score
        score = keyword_score("[Error: quota exceeded]", ["75%"])
        assert score is None

    def test_empty_answer_returns_none(self):
        from evaluation import keyword_score
        score = keyword_score("", ["keyword"])
        assert score is None

    def test_case_insensitive(self):
        from evaluation import keyword_score
        score = keyword_score(
            "ATTENDANCE REQUIREMENT IS 75 PERCENT",
            ["75 percent"],
        )
        assert score == 1.0


class TestRecallAtK:
    """Test recall_at_k function."""

    def test_perfect_recall(self):
        from evaluation import recall_at_k
        score = recall_at_k(
            ["Chapter 2 attendance page 5", "Chapter 3 exam page 10"],
            ["Chapter 2"],
            k=5,
        )
        assert score == 1.0

    def test_zero_recall(self):
        from evaluation import recall_at_k
        score = recall_at_k(
            ["Chapter 5 library page 8"],
            ["Chapter 2"],
            k=5,
        )
        assert score == 0.0

    def test_empty_relevant_chapters_returns_none(self):
        from evaluation import recall_at_k
        score = recall_at_k(["source1"], [], k=5)
        assert score is None


class TestReciprocalRank:
    """Test reciprocal_rank function."""

    def test_first_rank(self):
        from evaluation import reciprocal_rank
        rr = reciprocal_rank(["chapter 2 attendance", "chapter 3"], "chapter 2")
        assert rr == 1.0

    def test_second_rank(self):
        from evaluation import reciprocal_rank
        rr = reciprocal_rank(["chapter 3 exam", "chapter 2 attendance"], "chapter 2")
        assert rr == pytest.approx(0.5, abs=0.001)

    def test_not_found(self):
        from evaluation import reciprocal_rank
        rr = reciprocal_rank(["chapter 3", "chapter 5"], "chapter 2")
        assert rr == 0.0


class TestQuotaErrorDetection:
    """Test quota error detection."""

    def test_detects_quota_error(self):
        from evaluation import is_quota_error
        assert is_quota_error("429 resource exhausted: quota exceeded") is True

    def test_detects_rate_limit(self):
        from evaluation import is_quota_error
        assert is_quota_error("Rate limit reached") is True

    def test_normal_error_not_quota(self):
        from evaluation import is_quota_error
        assert is_quota_error("Connection timeout") is False

    def test_none_not_quota(self):
        from evaluation import is_quota_error
        assert is_quota_error(None) is False


class TestLoadQuestions:
    """Test question loading from CSV."""

    def test_loads_questions_from_csv(self):
        # Write a temp CSV
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["question_id", "question", "expected_answer_keywords",
                            "category", "chapter"]
            )
            writer.writeheader()
            writer.writerow({
                "question_id": "Q01",
                "question": "What is the attendance requirement?",
                "expected_answer_keywords": "75%, 75 percent",
                "category": "attendance",
                "chapter": "Chapter 2",
            })
            temp_path = f.name

        try:
            from evaluation import load_questions
            questions = load_questions(temp_path)
            assert len(questions) == 1
            assert questions[0]["question_id"] == "Q01"
            assert "75%" in questions[0]["expected_keywords"]
        finally:
            os.unlink(temp_path)
