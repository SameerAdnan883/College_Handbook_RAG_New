"""
v4_reranking.py — V4: RAG with Cross-Encoder Reranking
Pipeline:
  User Query → Retriever → Top 10–20 Candidates
  → Cross-Encoder Reranker → Top 3–5 Documents
  → LLM → Answer

Cross-Encoder Notes:
  - Scores are raw model logits (not probabilities, not percentages).
  - Higher scores mean higher predicted relevance.
  - Reranking improvements must be validated through evaluation, not assumed.
  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (configurable via config.py)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    RERANK_CANDIDATES, RERANK_TOP_N, RERANKER_MODEL,
    PDF_PATH, FAISS_INDEX_DIR,
    is_api_configured,
)


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic assistant for Greenfield University 
(a fictional educational institution created for RAG demonstration purposes).

The following context has been carefully selected and RERANKED for relevance to 
your question using a cross-encoder model.

Use ONLY this context to answer. If the context does not contain the answer, say:
"The handbook does not appear to contain information on this topic."

Reranked context:
{context}
"""

HUMAN_TEMPLATE = "Question: {question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# ─── Cross-Encoder Reranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for document reranking.

    Important: Scores from ms-marco-MiniLM-L-6-v2 are raw logits.
    They are NOT probabilities; do not interpret them as percentages.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                print(f"[V4] Cross-encoder loaded: {self.model_name}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load cross-encoder '{self.model_name}': {exc}\n"
                    "Install: pip install sentence-transformers"
                ) from exc

    def rerank(self, query: str, docs: list, top_n: int = RERANK_TOP_N) -> list[dict]:
        """
        Rerank documents and return top_n with scores.

        Returns list of dicts:
          {doc, score, original_rank, new_rank}
        Note: Scores are raw logits, not probabilities.
        """
        self._load()
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RuntimeError(f"Reranker prediction failed: {exc}") from exc

        ranked = sorted(
            [{"doc": doc, "score": float(score), "original_rank": i}
             for i, (doc, score) in enumerate(zip(docs, scores))],
            key=lambda x: x["score"],
            reverse=True,
        )

        for new_rank, item in enumerate(ranked):
            item["new_rank"] = new_rank

        return ranked[:top_n]


# Module-level singleton (lazy-loaded)
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ─── Embeddings & index (reuse V2 FAISS logic) ───────────────────────────────

def get_embeddings():
    if not is_api_configured():
        raise RuntimeError("GOOGLE_API_KEY not configured.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # Adding max_retries and timeout for API reliability
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_or_build_faiss(force_rebuild: bool = False):
    """Load or build FAISS index (delegates to V2)."""
    from v2_faiss import build_faiss_index
    return build_faiss_index(force_rebuild=force_rebuild)


def get_llm():
    if not is_api_configured():
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=1024,
        max_retries=3,
        timeout=60,
    )


# ─── Format reranked docs ─────────────────────────────────────────────────────

def format_reranked_docs(reranked_items: list[dict]) -> str:
    parts = []
    for item in reranked_items:
        doc = item["doc"]
        score = item["score"]
        new_rank = item["new_rank"] + 1
        orig_rank = item["original_rank"] + 1
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
        parts.append(
            f"[Rank {new_rank} (was {orig_rank}) | Score: {score:.4f} (raw logit) "
            f"| {source} | Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ─── Ask function ─────────────────────────────────────────────────────────────

def ask(
    question: str,
    db=None,
    candidates: int = RERANK_CANDIDATES,
    top_n: int = RERANK_TOP_N,
    reranker=None,
) -> dict:
    """
    Full V4 pipeline with reranking.

    Returns:
      version, question, answer, reranked_docs (with scores),
      sources, latency breakdown, error
    """
    result = {
        "version": "V4",
        "question": question,
        "answer": "",
        "reranked_docs": [],
        "sources": [],
        "candidates_retrieved": candidates,
        "top_n_after_rerank": top_n,
        "latency_seconds": 0.0,
        "retrieval_latency": 0.0,
        "reranking_latency": 0.0,
        "generation_latency": 0.0,
        "reranker_model": RERANKER_MODEL,
        "error": None,
    }

    if not is_api_configured():
        result["error"] = "API key not configured."
        result["answer"] = "[API key not configured]"
        return result

    total_start = time.perf_counter()

    try:
        # Step 1: Load FAISS index
        if db is None:
            db = get_or_build_faiss()

        # Step 2: Retrieve candidates (more than final top_n)
        retriever = db.as_retriever(search_kwargs={"k": candidates})
        t0 = time.perf_counter()
        candidate_docs = retriever.invoke(question)
        result["retrieval_latency"] = round(time.perf_counter() - t0, 3)
        print(f"[V4] Retrieved {len(candidate_docs)} candidates")

        if not candidate_docs:
            result["answer"] = "No relevant documents found in the knowledge base."
            return result

        # Step 3: Cross-encoder reranking
        reranker_instance = reranker if reranker is not None else get_reranker()
        t1 = time.perf_counter()
        reranked = reranker_instance.rerank(question, candidate_docs, top_n=top_n)
        result["reranking_latency"] = round(time.perf_counter() - t1, 3)
        print(f"[V4] Reranked → top {len(reranked)} documents selected")

        result["reranked_docs"] = [
            {
                "content": item["doc"].page_content,
                "score": item["score"],
                "original_rank": item["original_rank"] + 1,
                "new_rank": item["new_rank"] + 1,
                "page": item["doc"].metadata.get("page", "?"),
                "source": item["doc"].metadata.get(
                    "source_file",
                    item["doc"].metadata.get("source", "?")
                ),
                "score_note": "Raw logit — not a probability.",
            }
            for item in reranked
        ]

        result["sources"] = list({
            f"{item['doc'].metadata.get('source_file', item['doc'].metadata.get('source', '?'))} "
            f"(Page {item['doc'].metadata.get('page', '?')})"
            for item in reranked
        })

        # Step 4: Generate answer from reranked context
        context_str = format_reranked_docs(reranked)
        llm = get_llm()
        if llm is None:
            raise RuntimeError("LLM unavailable.")

        t2 = time.perf_counter()
        prompt_value = PROMPT.invoke({"context": context_str, "question": question})
        answer = (llm | StrOutputParser()).invoke(prompt_value)
        result["generation_latency"] = round(time.perf_counter() - t2, 3)
        result["answer"] = answer

    except Exception as exc:
        result["error"] = str(exc)
        result["answer"] = f"[Error: {exc}]"
    finally:
        result["latency_seconds"] = round(time.perf_counter() - total_start, 3)

    return result


# ─── Reranking explanation ────────────────────────────────────────────────────

RERANKING_EXPLANATION = """
Reranking Pipeline (V4)
------------------------
1. Retrieval (bi-encoder / FAISS):
   Fast approximate nearest-neighbour search retrieves the top-N candidates
   (default 15). Bi-encoder embeddings trade some precision for speed.

2. Cross-Encoder Reranking:
   The cross-encoder (ms-marco-MiniLM-L-6-v2) jointly encodes the query and
   each candidate document together to produce a relevance score.
   - Cross-encoders are slower but more precise than bi-encoders.
   - Scores are raw model logits — NOT probabilities or percentages.
   - A higher score simply means the model considers that pair more relevant.

3. Top-N Selection:
   The top-3 to top-5 highest-scoring documents are passed to the LLM.

4. Answer Generation:
   Gemini generates an answer grounded exclusively in the reranked context.

Important Caveat:
   Reranking does not automatically improve accuracy. Improvements must be
   demonstrated through evaluation (see evaluation.py). On small corpora
   like this handbook, reranking may provide marginal gains.
"""


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V4 — RAG with Cross-Encoder Reranking")
    print(RERANKING_EXPLANATION)
    print("=" * 60)

    if not is_api_configured():
        print("⚠  GOOGLE_API_KEY not set.")
        return

    try:
        db = get_or_build_faiss()
    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}")
        return

    while True:
        question = input("\nEnter question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        res = ask(question, db=db)
        print(f"\n📝 Answer:\n{res['answer']}")
        print(f"\n🔢 Reranked documents:")
        for doc in res["reranked_docs"]:
            print(f"  Rank {doc['new_rank']} (was {doc['original_rank']}) | "
                  f"Score: {doc['score']:.4f} | Page {doc['page']}")
        print(f"\n📄 Sources: {res['sources']}")
        print(f"⏱  Total: {res['latency_seconds']}s "
              f"(retrieval: {res['retrieval_latency']}s, "
              f"rerank: {res['reranking_latency']}s, "
              f"gen: {res['generation_latency']}s)")


if __name__ == "__main__":
    main()
