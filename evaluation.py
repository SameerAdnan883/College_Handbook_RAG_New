"""
evaluation.py — Comprehensive RAG Evaluation Module
Evaluates V2 (FAISS), V3 (ChromaDB), and V4 (Reranking) on 25 questions.

Metrics:
  - Answer correctness (keyword-based — documented and transparent)
  - Retrieval quality (Recall@k, MRR — requires relevance labels)
  - Response latency (retrieval and generation separately)

Rules:
  - Results are saved after each question (resumable).
  - API failures are NOT counted as incorrect answers.
  - Quota errors stop execution; results so far are preserved.
  - Never fabricate results.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from config import (
    EVAL_QUESTIONS_CSV,
    EVAL_RESULTS_CSV,
    ARTIFACTS_DIR,
    is_api_configured,
)

RESULTS_INTERMEDIATE_JSON = os.path.join(
    os.path.dirname(EVAL_RESULTS_CSV), "eval_intermediate.json"
)


# ─── Question loading ─────────────────────────────────────────────────────────

def load_questions(csv_path: str = EVAL_QUESTIONS_CSV) -> list[dict]:
    """Load evaluation questions from CSV."""
    questions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_keywords": [
                    kw.strip().lower()
                    for kw in row["expected_answer_keywords"].split(",")
                ],
                "category": row["category"],
                "chapter": row["chapter"],
            })
    return questions


# ─── Scoring ──────────────────────────────────────────────────────────────────

def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """
    Keyword-based correctness score.

    Method (transparent):
      Score = (# of keyword groups that appear in answer) / (# of keyword groups)
      Each CSV cell contains comma-separated alternative keywords grouped by |
      For simplicity here: we check if ANY expected keyword appears in the answer.
      Score = 1.0 if at least one expected keyword is found, else 0.0.
      
    Limitations (documented):
      - Does not measure answer completeness or fluency.
      - Cannot detect paraphrased correct answers.
      - A keyword match is necessary but not sufficient for full correctness.
    This method is chosen for reproducibility and transparency over opaque LLM judging.
    """
    if not answer or answer.startswith("["):
        return None  # Not scored — API failure or missing answer

    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw in answer_lower)
    return round(matched / max(len(expected_keywords), 1), 3)


# ─── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_sources: list[str], relevant_chapters: list[str], k: int) -> float:
    """
    Recall@k: fraction of relevant chapters represented in top-k retrieved sources.
    
    Note: We use chapter labels from evaluation_questions.csv as weak relevance labels.
    These are heuristic — they indicate which chapter contains the answer,
    not ground-truth retrieval judgements.
    """
    if not relevant_chapters:
        return None
    top_k = retrieved_sources[:k]
    hits = sum(1 for ch in relevant_chapters
               if any(ch.lower() in src.lower() for src in top_k))
    return round(hits / len(relevant_chapters), 3)


def reciprocal_rank(retrieved_sources: list[str], relevant_chapter: str) -> float:
    """
    Reciprocal Rank: 1/rank of first relevant source found.
    Returns 0 if relevant chapter not found in retrieved list.
    """
    for rank, src in enumerate(retrieved_sources, 1):
        if relevant_chapter.lower() in src.lower():
            return round(1.0 / rank, 3)
    return 0.0


# ─── Result persistence ───────────────────────────────────────────────────────

def load_existing_results(json_path: str = RESULTS_INTERMEDIATE_JSON) -> dict:
    """Load intermediate results (for resumability). Returns {question_id → result}."""
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_intermediate(results: dict, json_path: str = RESULTS_INTERMEDIATE_JSON):
    """Save intermediate results after each question."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def export_csv(results: dict, csv_path: str = EVAL_RESULTS_CSV):
    """Export all results to CSV."""
    rows = []
    for key, res in results.items():
        rows.append(res)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"✅ Results exported → {csv_path}")
    return rows


# ─── Per-version evaluation ───────────────────────────────────────────────────

QUOTA_ERROR_MARKERS = [
    "quota", "rate limit", "429", "resource exhausted",
    "too many requests", "billing", "insufficient quota",
]


def is_quota_error(error_str: str) -> bool:
    if not error_str:
        return False
    return any(marker in error_str.lower() for marker in QUOTA_ERROR_MARKERS)


def evaluate_version(
    version: str,
    questions: list[dict],
    ask_fn,
    existing_results: dict,
    ask_kwargs: dict = None,
) -> tuple[dict, bool]:
    """
    Run evaluation for one version.

    Returns (results_dict, quota_hit).
    Skips questions already completed for this version.
    Stops on quota error and returns partial results.
    """
    if ask_kwargs is None:
        ask_kwargs = {}

    results = {}
    quota_hit = False

    for q in questions:
        key = f"{version}_{q['question_id']}"

        if key in existing_results:
            print(f"  [{version}] {q['question_id']}: ⏭  skipped (already done)")
            results[key] = existing_results[key]
            continue

        print(f"  [{version}] {q['question_id']}: {q['question'][:60]}…")

        row = {
            "version": version,
            "question_id": q["question_id"],
            "question": q["question"],
            "category": q["category"],
            "chapter": q["chapter"],
            "expected_keywords": ", ".join(q["expected_keywords"]),
            "answer": "",
            "keyword_score": None,
            "recall_at_5": None,
            "reciprocal_rank": None,
            "latency_total": None,
            "latency_retrieval": None,
            "latency_generation": None,
            "sources": "",
            "error": None,
            "status": "not_run",
            "timestamp": datetime.utcnow().isoformat(),
        }

        if not is_api_configured():
            row["status"] = "skipped_no_api"
            row["error"] = "GOOGLE_API_KEY not configured."
            results[key] = row
            existing_results[key] = row
            save_intermediate(existing_results)
            continue

        try:
            res = ask_fn(q["question"], **ask_kwargs)

            row["answer"] = res.get("answer", "")
            row["error"] = res.get("error")
            row["latency_total"] = res.get("latency_seconds")
            row["latency_retrieval"] = res.get("retrieval_latency")
            row["latency_generation"] = res.get("generation_latency")

            # Sources
            sources = res.get("sources", [])
            row["sources"] = " | ".join(sources)

            # Keyword score (None if API error)
            if res.get("error") and is_quota_error(res.get("error", "")):
                row["status"] = "quota_error"
                row["keyword_score"] = None
                results[key] = row
                existing_results[key] = row
                save_intermediate(existing_results)
                quota_hit = True
                print(f"    ⛔ Quota error — stopping evaluation for {version}.")
                break
            elif res.get("error"):
                row["status"] = "api_error"
                # API errors NOT counted as wrong answers
                row["keyword_score"] = None
            else:
                row["status"] = "completed"
                row["keyword_score"] = keyword_score(
                    row["answer"], q["expected_keywords"]
                )
                # Retrieval metrics (heuristic labels)
                row["recall_at_5"] = recall_at_k(sources, [q["chapter"]], k=5)
                row["reciprocal_rank"] = reciprocal_rank(sources, q["chapter"])

        except Exception as exc:
            error_str = str(exc)
            row["error"] = error_str
            row["status"] = "exception"
            row["keyword_score"] = None
            if is_quota_error(error_str):
                results[key] = row
                existing_results[key] = row
                save_intermediate(existing_results)
                quota_hit = True
                print(f"    ⛔ Quota error (exception) — stopping {version}.")
                break

        results[key] = row
        existing_results[key] = row
        save_intermediate(existing_results)

        score_str = (f"{row['keyword_score']:.2f}"
                     if row["keyword_score"] is not None else "N/A")
        print(f"    ✅ Score: {score_str} | Latency: {row['latency_total']}s")

    return results, quota_hit


# ─── Summary stats ────────────────────────────────────────────────────────────

def compute_summary(results: dict) -> pd.DataFrame:
    """Compute per-version summary statistics."""
    rows = list(results.values())
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = []

    for version in df["version"].unique():
        vdf = df[df["version"] == version]
        completed = vdf[vdf["status"] == "completed"]
        scored = completed[completed["keyword_score"].notna()]

        summary.append({
            "Version": version,
            "Total_Questions": len(vdf),
            "Completed": len(completed),
            "Scored": len(scored),
            "Avg_Keyword_Score": round(scored["keyword_score"].mean(), 3) if len(scored) > 0 else None,
            "Avg_Recall@5": round(completed["recall_at_5"].dropna().mean(), 3) if len(completed) > 0 else None,
            "Avg_MRR": round(completed["reciprocal_rank"].dropna().mean(), 3) if len(completed) > 0 else None,
            "Avg_Latency_Total": round(completed["latency_total"].dropna().mean(), 3) if len(completed) > 0 else None,
            "Avg_Latency_Retrieval": round(completed["latency_retrieval"].dropna().mean(), 3) if len(completed) > 0 else None,
            "Avg_Latency_Generation": round(completed["latency_generation"].dropna().mean(), 3) if len(completed) > 0 else None,
            "API_Errors": int((vdf["status"] == "api_error").sum()),
            "Quota_Errors": int((vdf["status"] == "quota_error").sum()),
            "Coverage": f"{len(completed)}/{len(vdf)}",
        })

    return pd.DataFrame(summary)


# ─── Charts ───────────────────────────────────────────────────────────────────

def generate_charts(results: dict, output_dir: str = ARTIFACTS_DIR):
    """Generate comparison charts and save to artifacts/."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠  plotly not installed — skipping charts.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(list(results.values()))
    summary = compute_summary(results)

    if summary.empty:
        return []

    chart_files = []

    # Chart 1: Keyword Score by Version
    fig1 = px.bar(
        summary[summary["Avg_Keyword_Score"].notna()],
        x="Version", y="Avg_Keyword_Score",
        title="Average Keyword Score by Version (0–1)",
        color="Version",
        labels={"Avg_Keyword_Score": "Avg Keyword Score"},
    )
    fig1.update_layout(yaxis_range=[0, 1])
    f1 = os.path.join(output_dir, "chart_keyword_score.html")
    fig1.write_html(f1)
    chart_files.append(f1)

    # Chart 2: Latency Comparison
    latency_data = summary[summary["Avg_Latency_Total"].notna()]
    if not latency_data.empty:
        fig2 = go.Figure()
        for col, label in [
            ("Avg_Latency_Retrieval", "Retrieval"),
            ("Avg_Latency_Generation", "Generation"),
        ]:
            fig2.add_trace(go.Bar(
                name=label,
                x=latency_data["Version"],
                y=latency_data[col],
            ))
        fig2.update_layout(
            title="Average Latency by Version (seconds)",
            barmode="stack",
            yaxis_title="Seconds",
        )
        f2 = os.path.join(output_dir, "chart_latency.html")
        fig2.write_html(f2)
        chart_files.append(f2)

    # Chart 3: Recall@5 and MRR
    retrieval_data = summary[summary["Avg_Recall@5"].notna()]
    if not retrieval_data.empty:
        fig3 = go.Figure()
        for col, label in [("Avg_Recall@5", "Recall@5"), ("Avg_MRR", "MRR")]:
            fig3.add_trace(go.Bar(
                name=label, x=retrieval_data["Version"],
                y=retrieval_data[col],
            ))
        fig3.update_layout(
            title="Retrieval Quality by Version (heuristic labels)",
            barmode="group",
            yaxis_range=[0, 1],
        )
        f3 = os.path.join(output_dir, "chart_retrieval_quality.html")
        fig3.write_html(f3)
        chart_files.append(f3)

    # Chart 4: Per-question score heatmap
    completed = df[df["status"] == "completed"].copy()
    if not completed.empty:
        pivot = completed.pivot_table(
            values="keyword_score",
            index="question_id",
            columns="version",
            aggfunc="mean",
        )
        fig4 = px.imshow(
            pivot,
            title="Per-Question Keyword Score Heatmap",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
        )
        f4 = os.path.join(output_dir, "chart_heatmap.html")
        fig4.write_html(f4)
        chart_files.append(f4)

    print(f"✅ {len(chart_files)} charts saved to {output_dir}")
    return chart_files


# ─── Main evaluation runner ───────────────────────────────────────────────────

def run_full_evaluation(
    versions: list[str] = None,
    force_rerun: bool = False,
) -> dict:
    """
    Run evaluation for specified versions (default: V2, V3, V4).
    Resumable — skips already-completed questions.
    Stops on quota errors.

    Returns aggregated results dict.
    """
    if versions is None:
        versions = ["V2", "V3", "V4"]

    questions = load_questions()
    print(f"📋 Loaded {len(questions)} evaluation questions.")

    existing = {} if force_rerun else load_existing_results()
    all_results = dict(existing)

    for version in versions:
        print(f"\n{'='*50}")
        print(f"Evaluating {version}…")
        print(f"{'='*50}")

        try:
            if version == "V2":
                import v2_faiss as v2
                db = v2.build_faiss_index() if is_api_configured() else None
                ask_fn = v2.ask
                kwargs = {"db": db} if db else {}

            elif version == "V3":
                import v3_chroma as v3
                db = v3.build_chroma_db() if is_api_configured() else None
                ask_fn = v3.ask
                kwargs = {"db": db} if db else {}

            elif version == "V4":
                import v4_reranking as v4
                db = v4.get_or_build_faiss() if is_api_configured() else None
                ask_fn = v4.ask
                kwargs = {"db": db} if db else {}

            else:
                print(f"⚠  Unknown version: {version}")
                continue

        except Exception as e:
            print(f"⚠  Could not initialise {version}: {e}")
            continue

        ver_results, quota_hit = evaluate_version(
            version=version,
            questions=questions,
            ask_fn=ask_fn,
            existing_results=all_results,
            ask_kwargs=kwargs,
        )
        all_results.update(ver_results)

        if quota_hit:
            print(f"\n⛔ Quota hit during {version}. "
                  "Results so far are saved. Re-run later to continue.")
            break

    # Export
    export_csv(all_results)
    summary = compute_summary(all_results)
    if not summary.empty:
        print("\n📊 Summary:")
        print(summary.to_string(index=False))
        summary_path = os.path.join(
            os.path.dirname(EVAL_RESULTS_CSV), "eval_summary.csv"
        )
        summary.to_csv(summary_path, index=False)
        print(f"✅ Summary saved → {summary_path}")

    charts = generate_charts(all_results)

    return {
        "results": all_results,
        "summary": summary,
        "charts": charts,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument(
        "--versions", nargs="+", default=["V2", "V3", "V4"],
        choices=["V2", "V3", "V4"],
        help="Versions to evaluate",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore cached results and rerun everything",
    )
    args = parser.parse_args()

    run_full_evaluation(versions=args.versions, force_rerun=args.force_rerun)
