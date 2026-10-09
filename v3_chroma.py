"""
v3_chroma.py — V3: ChromaDB RAG
Pipeline:
  PDF → Loader → Chunking → Embeddings → ChromaDB
  → Retriever → Prompt + Context → LLM → Answer

Differences from V2 (FAISS):
  - FAISS is an in-memory flat index (excellent for small-to-medium corpora).
    ChromaDB is a persistent, server-capable document store with native metadata
    filtering and a SQL-backed metadata layer.
  - ChromaDB natively avoids duplicate ingestion when the same document chunk
    (by ID) is already present in the collection.
  - ChromaDB supports metadata filtering at query time (e.g. filter by chapter).
  - ChromaDB does NOT guarantee higher retrieval accuracy than FAISS.
    Retrieval quality depends on the embedding model and chunk quality.
"""

from __future__ import annotations

import hashlib
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
    PDF_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _chunk_id(chunk_text: str, page: int, source: str) -> str:
    """Deterministic ID for deduplication in ChromaDB."""
    content = f"{source}:page{page}:{chunk_text[:120]}"
    return hashlib.md5(content.encode()).hexdigest()


def load_and_split(
    pdf_path: str = PDF_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """Load and split PDF into annotated chunks."""
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
    )
    chunks = splitter.split_documents(pages)

    source_name = os.path.basename(pdf_path)
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        chunk.metadata["source"] = source_name
        chunk.metadata["doc_type"] = "college_handbook"
        chunk.metadata["institution"] = "Greenfield University"
        # Infer chapter from content heuristic
        text = chunk.page_content
        chapter_map = {
            "attendance": "attendance_policy",
            "examination": "examination_policy",
            "library": "library_policy",
            "fee": "fee_structure",
            "computer lab": "computer_lab_policy",
            "hostel": "hostel_policy",
            "student support": "student_support",
            "conduct": "code_of_conduct",
        }
        detected = "general"
        for keyword, chapter in chapter_map.items():
            if keyword.lower() in text.lower():
                detected = chapter
                break
        chunk.metadata["chapter"] = detected
        # Unique ID for deduplication
        chunk.metadata["chunk_id"] = _chunk_id(text, page, source_name)

    print(f"[V3] Loaded {len(pages)} pages → {len(chunks)} chunks")
    return chunks


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings():
    if not is_api_configured():
        raise RuntimeError("GOOGLE_API_KEY not configured.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # Adding max_retries and timeout for API reliability
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

def build_chroma_db(
    chunks: list | None = None,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_rebuild: bool = False,
) -> "Chroma":
    """
    Build or load ChromaDB collection.
    Avoids duplicate ingestion by checking existing chunk_ids.
    """
    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings()

    # If DB exists and force_rebuild not set, load it
    if not force_rebuild and os.path.exists(persist_dir):
        print(f"[V3] Loading existing ChromaDB from {persist_dir}")
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        existing_count = db._collection.count()
        print(f"[V3] Collection '{collection_name}' has {existing_count} documents.")
        if existing_count > 0:
            return db

    if chunks is None:
        chunks = load_and_split()

    print(f"[V3] Building ChromaDB collection '{collection_name}'…")

    # Check for existing IDs to avoid duplicates
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    existing_ids = set()
    try:
        existing_data = db._collection.get(include=[])
        existing_ids = set(existing_data.get("ids", []))
    except Exception:
        pass

    new_chunks = [c for c in chunks
                  if c.metadata.get("chunk_id", "") not in existing_ids]
    new_ids = [c.metadata["chunk_id"] for c in new_chunks]

    if not new_chunks:
        print("[V3] All chunks already present — skipping ingestion.")
        return db

    print(f"[V3] Adding {len(new_chunks)} new chunks "
          f"({len(chunks) - len(new_chunks)} already exist)…")
          
    # Batch processing to prevent SSL timeouts during embedding
    batch_size = 10
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i:i + batch_size]
        batch_ids = new_ids[i:i + batch_size]
        print(f"[V3] Adding batch {i//batch_size + 1} ({len(batch_chunks)} chunks)…")
        db.add_documents(batch_chunks, ids=batch_ids)
        
    print(f"[V3] ChromaDB saved → {persist_dir}")
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


# ─── Format docs ─────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        chapter = doc.metadata.get("chapter", "?")
        parts.append(
            f"[Excerpt {i} | {source} | Page {page} | Chapter: {chapter}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ─── Ask function ─────────────────────────────────────────────────────────────

def ask(
    question: str,
    db=None,
    top_k: int = TOP_K,
    filter_metadata: Optional[dict] = None,
) -> dict:
    """
    Ask a question via ChromaDB RAG.
    Optional filter_metadata (e.g. {"chapter": "attendance_policy"}).

    Returns dict with version, answer, sources, context_docs, latencies, error.
    """
    result = {
        "version": "V3",
        "question": question,
        "answer": "",
        "sources": [],
        "context_docs": [],
        "latency_seconds": 0.0,
        "retrieval_latency": 0.0,
        "generation_latency": 0.0,
        "filter_used": filter_metadata,
        "error": None,
    }

    if not is_api_configured():
        result["error"] = "API key not configured."
        result["answer"] = "[API key not configured]"
        return result

    total_start = time.perf_counter()

    try:
        if db is None:
            db = build_chroma_db()

        search_kwargs = {"k": top_k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        retriever = db.as_retriever(search_kwargs=search_kwargs)

        # Retrieval
        t0 = time.perf_counter()
        docs = retriever.invoke(question)
        result["retrieval_latency"] = round(time.perf_counter() - t0, 3)

        result["context_docs"] = [
            {
                "content": d.page_content,
                "page": d.metadata.get("page", "?"),
                "source": d.metadata.get("source", "?"),
                "chapter": d.metadata.get("chapter", "?"),
                "doc_type": d.metadata.get("doc_type", "?"),
            }
            for d in docs
        ]
        result["sources"] = list({
            f"{d.metadata.get('source', '?')} "
            f"(Page {d.metadata.get('page', '?')}, "
            f"Chapter: {d.metadata.get('chapter', '?')})"
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


# ─── Technical comparison note ────────────────────────────────────────────────

FAISS_VS_CHROMA = """
Technical Comparison: FAISS vs. ChromaDB
-----------------------------------------
FAISS (V2):
  + Extremely fast for similarity search on millions of vectors
  + Runs entirely in memory; no external dependencies
  + Ideal for read-heavy, static knowledge bases
  - Flat index; no metadata filtering during vector search
  - Persistence requires manual save/load
  - No built-in deduplication

ChromaDB (V3):
  + Persistent by default (SQLite + parquet backend)
  + Native metadata filtering at query time
  + Built-in deduplication via document IDs
  + Suitable for dynamic, growing collections
  - Slightly slower than FAISS on pure ANN search for large corpora
  - Requires disk I/O; heavier footprint

Accuracy Claim:
  ChromaDB does NOT inherently produce more accurate results than FAISS.
  Both use the same embedding model (models/gemini-embedding-001).
  Differences in answer quality arise from:
    1. Metadata filtering reducing noise (ChromaDB advantage)
    2. Chunk quality and size (same for both here)
  Ground-truth accuracy is measured in evaluation.py.
"""


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V3 — ChromaDB RAG")
    print(FAISS_VS_CHROMA)
    print("=" * 60)

    if not is_api_configured():
        print("⚠  GOOGLE_API_KEY not set.")
        return

    try:
        db = build_chroma_db()
    except Exception as e:
        print(f"❌ Failed to build ChromaDB: {e}")
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


if __name__ == "__main__":
    main()
