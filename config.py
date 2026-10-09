"""
config.py — Centralized configuration for College Handbook RAG System
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Fix for macOS OpenMP library conflict (faiss/sentence-transformers)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# ─── Gemini / Google AI ──────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")  # configurable; verify before use
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# ─── LangSmith ────────────────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "college-handbook-rag")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangSmith env vars immediately so LangChain picks them up
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

# ─── RAG Parameters ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
PDF_PATH = os.path.join(BASE_DIR, "sample_college_handbook.pdf")
EVAL_QUESTIONS_CSV = os.path.join(DATA_DIR, "evaluation_questions.csv")
EVAL_RESULTS_CSV = os.path.join(DATA_DIR, "evaluation_results.csv")

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "college_handbook")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_api_configured() -> bool:
    """Return True only when a non-empty API key is present."""
    return bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())


def is_langsmith_configured() -> bool:
    """Return True when LangSmith tracing is enabled and a key is present."""
    return bool(LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY and LANGCHAIN_API_KEY.strip())


def validate_config() -> dict:
    """
    Validate the configuration and return a dict of status flags.
    Does NOT raise — callers decide how to handle missing config.
    """
    return {
        "api_configured": is_api_configured(),
        "langsmith_configured": is_langsmith_configured(),
        "pdf_exists": os.path.exists(PDF_PATH),
        "faiss_index_exists": os.path.exists(FAISS_INDEX_DIR),
        "chroma_db_exists": os.path.exists(CHROMA_PERSIST_DIR),
        "eval_questions_exist": os.path.exists(EVAL_QUESTIONS_CSV),
    }


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for d in [DATA_DIR, ARTIFACTS_DIR, FAISS_INDEX_DIR]:
        os.makedirs(d, exist_ok=True)


ensure_dirs()
