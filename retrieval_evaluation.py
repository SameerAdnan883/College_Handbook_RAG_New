"""
retrieval_evaluation.py — Dedicated retrieval quality evaluation
Measures Recall@k and MRR separately from answer quality.

Note on relevance labels:
  We use chapter labels from the evaluation CSV as WEAK relevance signals.
  These indicate which chapter contains the expected answer.
  True relevance requires human annotation — these labels are heuristic.
  Results are clearly labeled as heuristic throughout.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

from config import EVAL_QUESTIONS_CSV, ARTIFACTS_DIR, is_api_configured
from evaluation import load_questions, recall_at_k, reciprocal_rank


CHAPTER_TO_KEYWORD = {
    "Chapter 2": "attendance",
    "Chapter 3": "examination",
    "Chapter 4": "fee",
    "Chapter 5": "library",
    "Chapter 6": "computer",
    "Chapter 7": "student support",
    "Chapter 8": "hostel",
    "Chapter 9": "conduct",
    "Chapter 10": "contact",
}


def evaluate_retrieval(
    version: str,
    retrieve_fn,
    questions: list[dict],
    k_values: list[int] = None,
    retrieve_kwargs: dict = None,
) -> pd.DataFrame:
    """
    Evaluate retrieval quality at multiple k values.

    Returns DataFrame with per-question metrics.
    Note: Relevance labels are heuristic (chapter-based).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    if retrieve_kwargs is None:
        retrieve_kwargs = {}

    rows = []
    for q in questions:
        row = {
            "version": version,
            "question_id": q["question_id"],
            "question": q["question"],
            "chapter": q["chapter"],
            "category": q["category"],
        }

        if not is_api_configured():
            row["status"] = "no_api"
            row["latency"] = None
            for k in k_values:
                row[f"recall@{k}"] = None
                row[f"rr@{k}"] = None
            rows.append(row)
            continue

        try:
            t0 = time.perf_counter()
            docs = retrieve_fn(q["question"], **retrieve_kwargs)
            latency = round(time.perf_counter() - t0, 3)

            sources = []
            for doc in docs:
                if hasattr(doc, "metadata"):
                    meta = doc.metadata
                    src = meta.get("source_file", meta.get("source", ""))
                    page = meta.get("page", "")
                    chapter_meta = meta.get("chapter", "")
                    sources.append(f"{src} page={page} chapter={chapter_meta}")
                else:
                    sources.append(str(doc))

            row["status"] = "completed"
            row["latency"] = latency
            row["num_retrieved"] = len(docs)

            # Relevance label = chapter keyword from CSV
            chapter = q["chapter"]
            chapter_kw = CHAPTER_TO_KEYWORD.get(chapter, chapter.lower())

            for k in k_values:
                top_k_sources = sources[:k]
                # Check if chapter keyword appears in any top-k source
                hit = any(chapter_kw.lower() in s.lower() for s in top_k_sources)
                row[f"recall@{k}"] = 1.0 if hit else 0.0
                # Reciprocal rank
                rr = 0.0
                for rank, src in enumerate(sources[:k], 1):
                    if chapter_kw.lower() in src.lower():
                        rr = round(1.0 / rank, 4)
                        break
                row[f"rr@{k}"] = rr

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["latency"] = None
            for k in k_values:
                row[f"recall@{k}"] = None
                row[f"rr@{k}"] = None

        rows.append(row)
        print(f"  [{version}] {q['question_id']}: R@5={row.get('recall@5')} | "
              f"RR@5={row.get('rr@5')} | Latency={row.get('latency')}s")

    return pd.DataFrame(rows)


def run_retrieval_evaluation(
    versions: list[str] = None,
    k_values: list[int] = None,
) -> dict:
    """Run retrieval evaluation for all specified versions."""
    if versions is None:
        versions = ["V2", "V3", "V4"]
    if k_values is None:
        k_values = [1, 3, 5, 10]

    questions = load_questions()
    all_dfs = {}

    for version in versions:
        print(f"\n{'='*50}")
        print(f"Retrieval evaluation: {version} (heuristic labels)")
        print("⚠  Recall/MRR labels are chapter-based heuristics, not ground truth.")
        print(f"{'='*50}")

        try:
            if version == "V2":
                import v2_faiss as v2
                db = v2.build_faiss_index() if is_api_configured() else None
                if db:
                    retriever = db.as_retriever(
                        search_kwargs={"k": max(k_values)}
                    )
                    retrieve_fn = retriever.invoke
                    kwargs = {}
                else:
                    retrieve_fn = lambda q: []
                    kwargs = {}

            elif version == "V3":
                import v3_chroma as v3
                db = v3.build_chroma_db() if is_api_configured() else None
                if db:
                    retriever = db.as_retriever(
                        search_kwargs={"k": max(k_values)}
                    )
                    retrieve_fn = retriever.invoke
                    kwargs = {}
                else:
                    retrieve_fn = lambda q: []
                    kwargs = {}

            elif version == "V4":
                import v2_faiss as v2
                db = v2.build_faiss_index() if is_api_configured() else None
                if db:
                    retriever = db.as_retriever(
                        search_kwargs={"k": max(k_values)}
                    )
                    retrieve_fn = retriever.invoke
                    kwargs = {}
                else:
                    retrieve_fn = lambda q: []
                    kwargs = {}

            else:
                continue

        except Exception as e:
            print(f"⚠  Could not load {version}: {e}")
            continue

        df = evaluate_retrieval(
            version=version,
            retrieve_fn=retrieve_fn,
            questions=questions,
            k_values=k_values,
            retrieve_kwargs=kwargs,
        )
        all_dfs[version] = df

        # Print summary
        completed = df[df["status"] == "completed"]
        if not completed.empty:
            print(f"\n📊 {version} Retrieval Summary (heuristic labels):")
            for k in k_values:
                col = f"recall@{k}"
                if col in completed.columns:
                    avg = completed[col].mean()
                    print(f"  Recall@{k}: {avg:.3f}")
            if "rr@5" in completed.columns:
                print(f"  MRR@5: {completed['rr@5'].mean():.3f}")

    # Save
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    combined = pd.concat(all_dfs.values(), ignore_index=True) if all_dfs else pd.DataFrame()
    if not combined.empty:
        out_path = os.path.join(ARTIFACTS_DIR, "retrieval_evaluation.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n✅ Retrieval evaluation saved → {out_path}")

    return all_dfs


if __name__ == "__main__":
    run_retrieval_evaluation()
