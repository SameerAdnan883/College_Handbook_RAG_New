"""
v2_faiss.py — V2: FAISS RAG
Pipeline:
  PDF → Document Loader → Text Splitter → Embeddings → FAISS
  → Retriever → Prompt + Context → LLM → Answer

Features:
  - PDF ingestion with PyPDF
  - RecursiveCharacterTextSplitter (configurable chunk size/overlap)
  - Google Generative AI Embeddings
  - FAISS vector store with local persistence
  - Similarity search with configurable top-k
  - Source document & page metadata
  - Fallback when context doesn't support the answer
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    PDF_PATH, FAISS_INDEX_DIR,
    is_api_configured,
)


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic assistant for Greenfield University 
(a fictional educational institution created for RAG demonstration purposes).

Use ONLY the provided context excerpts from the student handbook to answer the question.
If the context does not contain sufficient information, explicitly state:
"The handbook does not appear to contain information on this topic."
Do NOT fabricate information or draw on outside knowledge.

Context from the handbook:
{context}
"""

HUMAN_TEMPLATE = "Question: {question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# ─── Document loading & chunking ──────────────────────────────────────────────

def load_and_split(
    pdf_path: str = PDF_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """Load PDF and split into chunks. Returns list of Document objects."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Run: python generate_handbook_pdf.py"
        )

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["doc_type"] = "college_handbook"
        chunk.metadata["source_file"] = os.path.basename(pdf_path)

    print(f"[V2] Loaded {len(pages)} pages → {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings():
    """Return Google Generative AI Embeddings object."""
    if not is_api_configured():
        raise RuntimeError("GOOGLE_API_KEY not configured.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # Adding max_retries and timeout for API reliability
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# ─── FAISS vector store ───────────────────────────────────────────────────────

def build_faiss_index(
    chunks: list | None = None,
    index_dir: str = FAISS_INDEX_DIR,
    force_rebuild: bool = False,
) -> "FAISS":
    """
    Build or load FAISS index.
    - Loads existing index unless force_rebuild=True.
    - Builds from chunks (re-reads PDF if chunks is None).
    """
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    index_file = os.path.join(index_dir, "index.faiss")

    if not force_rebuild and os.path.exists(index_file):
        print(f"[V2] Loading existing FAISS index from {index_dir}")
        db = FAISS.load_local(index_dir, embeddings,
                              allow_dangerous_deserialization=True)
        return db

    if chunks is None:
        chunks = load_and_split()

    print("[V2] Building FAISS index…")
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    db.save_local(index_dir)
    print(f"[V2] FAISS index saved → {index_dir}")
    return db


# ─── LLM ─────────────────────────────────────────────────────────────────────

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


# ─── Chain ────────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
        parts.append(f"[Excerpt {i} | {source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(db=None, top_k: int = TOP_K):
    """Build LCEL chain: Retriever → Prompt → LLM → Parser."""
    if db is None:
        db = build_faiss_index()
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    llm = get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable — GOOGLE_API_KEY not set.")

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ─── Ask function ─────────────────────────────────────────────────────────────

def ask(question: str, db=None, top_k: int = TOP_K) -> dict:
    """
    Ask a question via FAISS RAG.

    Returns dict:
      - version, question, answer, sources, context_docs,
        latency_seconds, retrieval_latency, generation_latency, error
    """
    result = {
        "version": "V2",
        "question": question,
        "answer": "",
        "sources": [],
        "context_docs": [],
        "latency_seconds": 0.0,
        "retrieval_latency": 0.0,
        "generation_latency": 0.0,
        "error": None,
    }

    if not is_api_configured():
        result["error"] = "API key not configured."
        result["answer"] = "[API key not configured]"
        return result

    total_start = time.perf_counter()

    try:
        # Load/build index
        if db is None:
            db = build_faiss_index()

        retriever = db.as_retriever(search_kwargs={"k": top_k})

        # Retrieval
        t0 = time.perf_counter()
        docs = retriever.invoke(question)
        result["retrieval_latency"] = round(time.perf_counter() - t0, 3)

        result["context_docs"] = [
            {
                "content": d.page_content,
                "page": d.metadata.get("page", "?"),
                "source": d.metadata.get("source_file", d.metadata.get("source", "?")),
            }
            for d in docs
        ]
        result["sources"] = list({
            f"{d.metadata.get('source_file', d.metadata.get('source', '?'))} "
            f"(Page {d.metadata.get('page', '?')})"
            for d in docs
        })

        # Generation
        context_str = format_docs(docs)
        llm = get_llm()
        t1 = time.perf_counter()
        prompt_value = PROMPT.invoke({"context": context_str, "question": question})
        answer = (llm | StrOutputParser()).invoke(prompt_value)
        result["generation_latency"] = round(time.perf_counter() - t1, 3)
        result["answer"] = answer

    except Exception as exc:
        result["error"] = str(exc)
        result["answer"] = f"[Error: {exc}]"
    finally:
        result["latency_seconds"] = round(time.perf_counter() - total_start, 3)

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V2 — FAISS RAG")
    print("=" * 60)

    if not is_api_configured():
        print("⚠  GOOGLE_API_KEY not set — cannot run V2.")
        return

    try:
        db = build_faiss_index()
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        return

    while True:
        question = input("\nEnter question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        res = ask(question, db=db)
        print(f"\n📝 Answer:\n{res['answer']}")
        print(f"\n📄 Sources: {res['sources']}")
        print(f"⏱  Total: {res['latency_seconds']}s "
              f"(retrieval: {res['retrieval_latency']}s, "
              f"generation: {res['generation_latency']}s)")
        if res["error"]:
            print(f"⚠  Error: {res['error']}")


if __name__ == "__main__":
    main()
