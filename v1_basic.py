"""
v1_basic.py — V1: LangChain Basics
Pipeline: Input → Prompt Template → LLM → Output Parser → Chain → Output

No vector database or retrieval is used in V1.
The LLM answers based on its own knowledge (limited by training data).
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import GEMINI_MODEL, is_api_configured


# ─── Prompt Template ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful academic assistant for Greenfield University 
(a fictional educational institution used for demonstration purposes).

Answer the student's question clearly and concisely. If you do not know the answer, 
say so honestly — do not fabricate information.

Important: This system is running in V1 (Basic LangChain) mode.
No document retrieval is used. Answers are based on the language model's knowledge only.
"""

HUMAN_TEMPLATE = """{question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def get_llm(model: str = GEMINI_MODEL):
    """
    Initialise the Gemini LLM via LangChain.
    Returns None if the API is not configured.
    """
    if not is_api_configured():
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from config import GOOGLE_API_KEY
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=1024,
            max_retries=3,
            timeout=60,
        )
        return llm
    except Exception as exc:
        print(f"[V1] Failed to initialise LLM: {exc}", file=sys.stderr)
        return None


def build_chain(llm=None):
    """Build the LangChain LCEL chain: Prompt → LLM → OutputParser."""
    if llm is None:
        llm = get_llm()
    if llm is None:
        return None

    chain = (
        RunnablePassthrough()
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, llm=None) -> dict:
    """
    Ask a question using the V1 chain.

    Returns a dict with:
      - answer: str
      - model: str
      - latency_seconds: float
      - error: Optional[str]
      - version: "V1"
    """
    result = {
        "version": "V1",
        "question": question,
        "answer": "",
        "model": GEMINI_MODEL,
        "latency_seconds": 0.0,
        "error": None,
    }

    if not is_api_configured():
        result["error"] = (
            "API key not configured. "
            "Copy .env.example → .env and set GOOGLE_API_KEY."
        )
        result["answer"] = "[API key not configured — cannot generate answer]"
        return result

    chain = build_chain(llm)
    if chain is None:
        result["error"] = "Failed to build chain. Check API key and model availability."
        result["answer"] = "[Chain initialisation failed]"
        return result

    start = time.perf_counter()
    try:
        answer = chain.invoke({"question": question})
        result["answer"] = answer
    except Exception as exc:
        error_str = str(exc)
        result["error"] = error_str
        result["answer"] = f"[Error generating answer: {error_str}]"
    finally:
        result["latency_seconds"] = round(time.perf_counter() - start, 3)

    return result


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V1 — LangChain Basics (No Retrieval)")
    print("Model:", GEMINI_MODEL)
    print("=" * 60)

    if not is_api_configured():
        print("⚠  GOOGLE_API_KEY is not set.")
        print("   Create a .env file from .env.example and add your key.")
        return

    while True:
        question = input("\nEnter your question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        res = ask(question)
        print(f"\n📝 Answer:\n{res['answer']}")
        print(f"⏱  Latency: {res['latency_seconds']}s")
        if res["error"]:
            print(f"⚠  Error: {res['error']}")


if __name__ == "__main__":
    main()
