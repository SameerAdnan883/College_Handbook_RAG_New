"""
app.py — Streamlit Web Application
College Handbook Intelligence: End-to-End RAG System

Versions:
  V1 — LangChain Basics (no retrieval)
  V2 — FAISS RAG
  V3 — ChromaDB RAG
  V4 — RAG with Reranking

Features:
  - Version selector
  - Question input & answer display
  - Source document & page info
  - Reranking scores (V4)
  - Loading indicators
  - Evaluation dashboard
  - Comparison tables & charts
  - LangSmith integration (configurable)
"""

import os
import sys
import time

import streamlit as st
import pandas as pd

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="College Handbook Intelligence | RAG System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inline CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Cards */
.rag-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Version badge */
.version-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 8px;
}
.v1 { background: #6c5ce7; color: white; }
.v2 { background: #0984e3; color: white; }
.v3 { background: #00b894; color: white; }
.v4 { background: #e17055; color: white; }

/* Score bar */
.score-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.score-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Source chip */
.source-chip {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    margin: 2px;
    color: #dfe6e9;
}

/* Header gradient text */
.gradient-text {
    background: linear-gradient(90deg, #a29bfe, #74b9ff, #55efc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Timeline */
.timeline-step {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-size: 13px;
    color: #b2bec3;
}
.timeline-icon { margin-right: 10px; font-size: 16px; }

/* Rerank table */
.rerank-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-bottom: 6px;
    font-size: 12px;
}
.rank-1 { border-color: #fdcb6e; }
.rank-2 { border-color: #74b9ff; }
.rank-3 { border-color: #55efc4; }
.rank-other { border-color: #636e72; }

/* Warning box */
.warning-box {
    background: rgba(253, 203, 110, 0.1);
    border: 1px solid rgba(253, 203, 110, 0.4);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 13px;
    color: #fdcb6e;
    margin-bottom: 1rem;
}

/* Answer box */
.answer-box {
    background: rgba(0, 184, 148, 0.08);
    border: 1px solid rgba(0, 184, 148, 0.25);
    border-radius: 12px;
    padding: 1.2rem;
    line-height: 1.7;
    color: #dfe6e9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ─── Imports (after page config) ─────────────────────────────────────────────
from config import (
    validate_config, is_api_configured, is_langsmith_configured,
    GEMINI_MODEL, LANGCHAIN_PROJECT, RERANKER_MODEL,
    EVAL_RESULTS_CSV, ARTIFACTS_DIR,
    LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT,
)


# ─── Cached resource loaders ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="📄 Loading PDF and building FAISS index…")
def load_v2_db():
    if not is_api_configured():
        return None
    try:
        import v2_faiss as v2
        return v2.build_faiss_index()
    except Exception as e:
        st.warning(f"V2 FAISS load failed: {e}")
        return None


@st.cache_resource(show_spinner="🗄️ Loading ChromaDB…")
def load_v3_db():
    if not is_api_configured():
        return None
    try:
        import v3_chroma as v3
        return v3.build_chroma_db()
    except Exception as e:
        st.warning(f"V3 ChromaDB load failed: {e}")
        return None


@st.cache_resource(show_spinner="🔧 Downloading & Loading Reranker Model (~90MB)... This may take several minutes on slow connections.")
def load_reranker():
    try:
        # Fix macOS OpenMP conflicts and HF warnings locally before loading sentence-transformers
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        from v4_reranking import CrossEncoderReranker
        r = CrossEncoderReranker()
        
        print("[app] Starting reranker model load...")
        t0 = time.perf_counter()
        r._load()
        print(f"[app] Reranker model loaded successfully in {time.perf_counter() - t0:.2f}s")
        return r
    except Exception as e:
        print(f"[app] Failed to load reranker: {e}")
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0.5rem 0 1.5rem;'>
            <div style='font-size: 2.5rem;'>🎓</div>
            <div class='gradient-text' style='font-size: 1.3rem; font-weight: 700;'>
                Greenfield University
            </div>
            <div style='font-size: 0.75rem; color: #636e72; margin-top: 4px;'>
                Handbook Intelligence System
            </div>
            <div style='font-size: 0.65rem; color: #e17055; margin-top: 4px;'>
                ⚠ Fictional Educational Data
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔌 System Status")
        cfg = validate_config()

        def status_icon(ok): return "✅" if ok else "❌"

        st.markdown(f"""
        - {status_icon(cfg['api_configured'])} **Gemini API** — `{GEMINI_MODEL}`
        - {status_icon(cfg['pdf_exists'])} **Handbook PDF**
        - {status_icon(cfg['faiss_index_exists'])} **FAISS Index**
        - {status_icon(cfg['chroma_db_exists'])} **ChromaDB**
        - {status_icon(is_langsmith_configured())} **LangSmith** ({LANGCHAIN_PROJECT})
        - {status_icon(cfg['eval_questions_exist'])} **Evaluation Questions**
        """)

        if not cfg["api_configured"]:
            st.markdown("""
            <div class='warning-box'>
            ⚠ <strong>API Key Missing</strong><br>
            Copy <code>.env.example</code> → <code>.env</code><br>
            Add your <code>GOOGLE_API_KEY</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 Version")
        version = st.radio(
            "Select RAG version:",
            ["V1 — Basic LangChain", "V2 — FAISS RAG",
             "V3 — ChromaDB RAG", "V4 — Reranking"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style='font-size:11px; color:#636e72;'>
        <strong>Project:</strong> College Handbook Intelligence<br>
        <strong>Stack:</strong> LangChain · Gemini · FAISS · ChromaDB<br>
        <strong>Reranker:</strong> cross-encoder/ms-marco-MiniLM-L-6-v2<br>
        <strong>Embeddings:</strong> models/gemini-embedding-001
        </div>
        """, unsafe_allow_html=True)

    return version


# ─── Version explanation cards ────────────────────────────────────────────────

VERSION_EXPLANATIONS = {
    "V1": {
        "color": "#6c5ce7", "emoji": "🔤",
        "title": "V1 — LangChain Basics",
        "description": "Direct LLM query without any document retrieval. "
                       "Uses Prompt Templates, Gemini LLM, and Output Parsers "
                       "chained together with LangChain's LCEL syntax.",
        "pipeline": ["Input Question", "Prompt Template", "Gemini LLM", "Output Parser", "Answer"],
        "limitations": "No grounding in the handbook — answers may be inaccurate.",
    },
    "V2": {
        "color": "#0984e3", "emoji": "⚡",
        "title": "V2 — FAISS RAG",
        "description": "PDF → chunks → Google embeddings → FAISS index. "
                       "Retrieves top-k chunks by cosine similarity, then grounds "
                       "the LLM answer in those chunks.",
        "pipeline": ["PDF Ingestion", "Text Splitting", "Google Embeddings",
                     "FAISS Index", "Similarity Search", "LLM + Context", "Answer"],
        "limitations": "Fast flat index; no metadata filtering.",
    },
    "V3": {
        "color": "#00b894", "emoji": "🗄️",
        "title": "V3 — ChromaDB RAG",
        "description": "Same pipeline as V2 but replaces FAISS with ChromaDB — "
                       "a persistent document store with metadata filtering and "
                       "built-in deduplication.",
        "pipeline": ["PDF Ingestion", "Text Splitting", "Google Embeddings",
                     "ChromaDB (persistent)", "Metadata Filter + Search",
                     "LLM + Context", "Answer"],
        "limitations": "Slightly slower than FAISS for pure ANN search.",
    },
    "V4": {
        "color": "#e17055", "emoji": "🎯",
        "title": "V4 — RAG with Reranking",
        "description": "Retrieves 15 candidates via FAISS, then uses a "
                       "cross-encoder (ms-marco-MiniLM-L-6-v2) to score and "
                       "reorder them. Top-4 documents go to the LLM.",
        "pipeline": ["Query", "FAISS (15 candidates)", "Cross-Encoder Reranker",
                     "Top-4 Documents", "LLM + Reranked Context", "Answer"],
        "limitations": "Scores are raw logits, not probabilities. Slower due to reranking.",
    },
}


def render_version_explanation(version_key: str):
    info = VERSION_EXPLANATIONS.get(version_key, {})
    if not info:
        return

    pipeline_html = " → ".join(
        f'<span style="background:rgba(255,255,255,0.1);'
        f'border-radius:6px;padding:2px 8px;font-size:11px;">{step}</span>'
        for step in info["pipeline"]
    )

    st.markdown(f"""
    <div class='rag-card'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:0.8rem;'>
            <span style='font-size:1.8rem;'>{info['emoji']}</span>
            <div>
                <div style='font-size:1rem;font-weight:600;color:#dfe6e9;'>
                    {info['title']}
                </div>
                <div style='font-size:12px;color:#b2bec3;'>
                    {info['description']}
                </div>
            </div>
        </div>
        <div style='margin-bottom:6px;font-size:11px;color:#74b9ff;font-weight:600;'>
            PIPELINE
        </div>
        <div style='color:#dfe6e9;line-height:2;'>{pipeline_html}</div>
        <div style='margin-top:8px;font-size:11px;color:#fdcb6e;'>
            ⚠ {info['limitations']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Q&A interface ────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What is the minimum attendance required to appear in examinations?",
    "How many books can an undergraduate student borrow?",
    "What is the annual tuition fee at Greenfield University?",
    "What happens if a student gets below 60% attendance?",
    "What is the free printing quota per semester?",
    "What is the fine for overdue library books?",
    "How does the reranking pipeline improve retrieval quality?",
    "What is the refund policy for fee withdrawal?",
]


def render_qa_tab(version_key: str):
    version_map = {"V1": "V1 — Basic LangChain", "V2": "V2 — FAISS RAG",
                   "V3": "V3 — ChromaDB RAG", "V4": "V4 — Reranking"}

    render_version_explanation(version_key)

    st.markdown("### 💬 Ask a Question")
    col1, col2 = st.columns([3, 1])

    with col2:
        sample = st.selectbox(
            "Sample questions:", [""] + SAMPLE_QUESTIONS,
            label_visibility="visible",
        )

    with col1:
        question = st.text_input(
            "Your question:",
            value=sample if sample else "",
            placeholder="e.g. What is the minimum attendance requirement?",
            key=f"q_input_{version_key}",
        )

    ask_btn = st.button(
        f"🔍 Ask {version_key}", type="primary",
        key=f"ask_btn_{version_key}",
        use_container_width=True,
    )

    if ask_btn and question:
        _run_query(question, version_key)
    elif ask_btn and not question:
        st.warning("Please enter a question.")


def _run_query(question: str, version_key: str):
    """Run the appropriate pipeline and render results."""

    if not is_api_configured():
        st.markdown("""
        <div class='warning-box'>
        ❌ <strong>API Key Required</strong><br>
        Add <code>GOOGLE_API_KEY</code> to your <code>.env</code> file to run live queries.
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner(f"Running {version_key} pipeline…"):
        start = time.perf_counter()

        try:
            if version_key == "V1":
                import v1_basic as v1
                result = v1.ask(question)

            elif version_key == "V2":
                import v2_faiss as v2
                db = load_v2_db()
                result = v2.ask(question, db=db)

            elif version_key == "V3":
                import v3_chroma as v3
                db = load_v3_db()
                result = v3.ask(question, db=db)

            elif version_key == "V4":
                import v4_reranking as v4
                db = load_v2_db()  # V4 uses FAISS index
                reranker = load_reranker()
                if reranker is None:
                    st.error("❌ Failed to load the Cross-Encoder model. Please check the terminal logs. It may be a network timeout downloading from Hugging Face.")
                    return
                result = v4.ask(question, db=db, reranker=reranker)

            else:
                st.error(f"Unknown version: {version_key}")
                return

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # ── Answer ──
    st.markdown("---")
    st.markdown("#### 📝 Answer")

    if result.get("error") and not result.get("answer", "").startswith("["):
        st.error(f"⚠ Error: {result['error']}")

    answer_text = result.get("answer", "(no answer generated)")
    st.markdown(f"""
    <div class='answer-box'>{answer_text}</div>
    """, unsafe_allow_html=True)

    # ── Latency metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱ Total", f"{result.get('latency_seconds', '—')}s")
    with col2:
        st.metric("📡 Retrieval", f"{result.get('retrieval_latency', '—')}s")
    with col3:
        st.metric("🤖 Generation", f"{result.get('generation_latency', '—')}s")
    with col4:
        rerank_lat = result.get("reranking_latency")
        st.metric("🎯 Reranking", f"{rerank_lat}s" if rerank_lat is not None else "N/A")

    # ── Sources ──
    sources = result.get("sources", [])
    if sources:
        st.markdown("#### 📄 Sources")
        chips = " ".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
        st.markdown(chips, unsafe_allow_html=True)

    # ── Context documents ──
    context_docs = result.get("context_docs", [])
    if context_docs:
        with st.expander(f"📚 Retrieved Context ({len(context_docs)} chunks)", expanded=False):
            for i, doc in enumerate(context_docs, 1):
                st.markdown(f"**Chunk {i}** | Page {doc.get('page', '?')} | "
                            f"{doc.get('source', '?')}")
                if doc.get("chapter"):
                    st.caption(f"Chapter: {doc['chapter']}")
                st.text_area(
                    f"Content {i}", doc.get("content", ""),
                    height=80, key=f"ctx_{version_key}_{i}",
                    label_visibility="collapsed",
                )

    # ── Reranking details (V4 only) ──
    reranked = result.get("reranked_docs", [])
    if reranked:
        st.markdown("#### 🎯 Reranking Details")
        st.caption(
            f"ℹ Cross-encoder: `{RERANKER_MODEL}` — "
            "Scores are raw logits, **not** probabilities."
        )
        for doc in reranked:
            rank_class = f"rank-{min(doc['new_rank'], 4)}" if doc["new_rank"] <= 3 else "rank-other"
            score = doc["score"]
            st.markdown(f"""
            <div class='rerank-item {rank_class}'>
                <strong>Rank {doc['new_rank']}</strong> 
                (was {doc['original_rank']}) &nbsp;|&nbsp;
                <strong>Score: {score:.4f}</strong> &nbsp;|&nbsp;
                Page {doc['page']} &nbsp;|&nbsp; {doc['source']}
                <div style='font-size:10px;color:#636e72;margin-top:2px;'>
                    {doc['content'][:120]}…
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Evaluation dashboard ─────────────────────────────────────────────────────

def render_evaluation_tab():
    st.markdown("## 📊 Evaluation Dashboard")
    st.caption(
        "Evaluation runs V2, V3, and V4 on 25 questions from the fictional handbook. "
        "**Do not start evaluation unless you have API quota available.** "
        "Results are saved after each question and are resumable."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Scoring Method (Keyword-Based):**
        - Score = fraction of expected keywords found in the answer.
        - This method is transparent and reproducible.
        - Paraphrased correct answers may receive lower scores (known limitation).
        - API failures are **not** counted as wrong answers; they are reported separately.
        
        **Retrieval Metrics (Heuristic Labels):**
        - Recall@k and MRR use chapter labels from the evaluation CSV as *weak* relevance signals.
        - These are NOT ground-truth relevance judgements.
        """)

    with col2:
        st.markdown("""
        **Coverage:**
        - Evaluation coverage (% of questions answered) is reported separately from accuracy.
        - Questions skipped due to quota/API errors are clearly labeled.
        """)

    st.markdown("---")

    # Run evaluation button
    if st.button("🚀 Run Full Evaluation (V2 + V3 + V4)", type="primary"):
        if not is_api_configured():
            st.error("❌ API key required to run evaluation.")
        else:
            with st.spinner("Running evaluation — this may take several minutes…"):
                try:
                    from evaluation import run_full_evaluation
                    eval_output = run_full_evaluation()
                    st.success("✅ Evaluation complete!")
                    st.session_state["eval_output"] = eval_output
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

    # Load & display existing results
    if os.path.exists(EVAL_RESULTS_CSV):
        st.markdown("### 📋 Existing Results")
        try:
            df = pd.read_csv(EVAL_RESULTS_CSV)
            st.dataframe(df, use_container_width=True)

            # Summary
            from evaluation import compute_summary
            results_dict = {}
            for _, row in df.iterrows():
                key = f"{row.get('version', '')}_{row.get('question_id', '')}"
                results_dict[key] = row.to_dict()
            summary = compute_summary(results_dict)
            if not summary.empty:
                st.markdown("### 📈 Summary")
                st.dataframe(summary, use_container_width=True)

            # Charts
            _render_charts_from_results(results_dict)

        except Exception as e:
            st.warning(f"Could not load results: {e}")
    else:
        st.info("No evaluation results yet. Run evaluation above.")

    # Download
    if os.path.exists(EVAL_RESULTS_CSV):
        with open(EVAL_RESULTS_CSV, "rb") as f:
            st.download_button(
                "⬇ Download Results CSV",
                f,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )


def _render_charts_from_results(results_dict: dict):
    """Render plotly charts inside Streamlit."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from evaluation import compute_summary

        summary = compute_summary(results_dict)
        if summary.empty:
            return

        st.markdown("### 📊 Comparison Charts")

        col1, col2 = st.columns(2)
        with col1:
            scored = summary[summary["Avg_Keyword_Score"].notna()]
            if not scored.empty:
                fig = px.bar(
                    scored, x="Version", y="Avg_Keyword_Score",
                    title="Average Keyword Score",
                    color="Version",
                    color_discrete_sequence=["#6c5ce7", "#0984e3", "#00b894", "#e17055"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            latency_data = summary[summary["Avg_Latency_Total"].notna()]
            if not latency_data.empty:
                fig2 = go.Figure()
                for col, label, color in [
                    ("Avg_Latency_Retrieval", "Retrieval", "#74b9ff"),
                    ("Avg_Latency_Generation", "Generation", "#55efc4"),
                ]:
                    if col in latency_data.columns:
                        fig2.add_trace(go.Bar(
                            name=label, x=latency_data["Version"],
                            y=latency_data[col], marker_color=color,
                        ))
                fig2.update_layout(
                    title="Latency Breakdown (seconds)",
                    barmode="stack",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Recall@5 and MRR
        retrieval_data = summary[summary["Avg_Recall@5"].notna()]
        if not retrieval_data.empty:
            fig3 = go.Figure()
            for col, label, color in [
                ("Avg_Recall@5", "Recall@5 (heuristic)", "#a29bfe"),
                ("Avg_MRR", "MRR (heuristic)", "#fd79a8"),
            ]:
                if col in retrieval_data.columns:
                    fig3.add_trace(go.Bar(
                        name=label, x=retrieval_data["Version"],
                        y=retrieval_data[col], marker_color=color,
                    ))
            fig3.update_layout(
                title="Retrieval Quality (heuristic labels — not ground truth)",
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        st.info("Install plotly for charts: pip install plotly")
    except Exception as e:
        st.warning(f"Could not render charts: {e}")


# ─── Architecture tab ─────────────────────────────────────────────────────────

def render_architecture_tab():
    st.markdown("## 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │              College Handbook Intelligence System             │
    │                  (Greenfield University — FICTIONAL)         │
    └──────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │ INPUT LAYER                                                  │
    │  sample_college_handbook.pdf  (fictional handbook)          │
    │  data/evaluation_questions.csv  (25 Q&A pairs)              │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ DOCUMENT PROCESSING  (shared across V2/V3/V4)               │
    │  PyPDFLoader → RecursiveCharacterTextSplitter               │
    │  chunk_size=800, overlap=150                                 │
    └──────────┬─────────────────────────────────────────────────┘
               │
    ┌──────────▼────────────────────────┐
    │ EMBEDDING                          │
    │ models/gemini-embedding-001        │
    └──────────┬────────────────────────┘
               │
    ┌──────────▼──────────┐  ┌─────────────────────┐
    │ V2: FAISS Index     │  │ V3: ChromaDB         │
    │ (in-memory, fast)   │  │ (persistent, filter) │
    └──────────┬──────────┘  └──────────┬──────────┘
               │                        │
    ┌──────────▼────────────────────────▼──────────┐
    │ RETRIEVAL                                     │
    │ V2/V3: Similarity search → top-k chunks       │
    │ V4: FAISS → 15 candidates                     │
    │      → cross-encoder/ms-marco-MiniLM-L-6-v2  │
    │      → top-4 reranked documents               │
    └──────────────────────────────────────────────┘
               │
    ┌──────────▼────────────────────────────────────┐
    │ GENERATION                                     │
    │ Prompt Template + Context → Gemini (gemini-X) │
    │ StrOutputParser → Final Answer                 │
    └──────────────────────────────────────────────┘
               │
    ┌──────────▼───────────────────────────────────────┐
    │ MONITORING (optional)                             │
    │ LangSmith: traces, latency, tokens, errors        │
    └──────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### 📁 Project File Structure")
    st.code("""
    College_Handbook_RAG_New/
    ├── app.py                      # Streamlit web application
    ├── v1_basic.py                 # V1: LangChain basics
    ├── v2_faiss.py                 # V2: FAISS RAG
    ├── v3_chroma.py                # V3: ChromaDB RAG
    ├── v4_reranking.py             # V4: Reranking pipeline
    ├── evaluation.py               # Comprehensive evaluation
    ├── retrieval_evaluation.py     # Retrieval-specific metrics
    ├── config.py                   # Centralized configuration
    ├── generate_handbook_pdf.py    # PDF generator (run once)
    ├── requirements.txt
    ├── .env.example                # Template for credentials
    ├── .gitignore
    ├── README.md
    ├── sample_college_handbook.pdf # Generated fictional handbook
    ├── faiss_index/                # FAISS vector index
    ├── chroma_db/                  # ChromaDB persistent store
    ├── data/
    │   ├── evaluation_questions.csv
    │   ├── evaluation_results.csv  # (generated after eval)
    │   └── eval_summary.csv        # (generated after eval)
    ├── artifacts/                  # Charts & outputs
    └── tests/                      # Unit tests
        ├── test_v1.py
        ├── test_v2.py
        ├── test_v3.py
        ├── test_v4.py
        └── test_evaluation.py
    """, language="text")


# ─── LangSmith tab ────────────────────────────────────────────────────────────

def render_langsmith_tab():
    st.markdown("## 🔭 LangSmith Monitoring")

    configured = is_langsmith_configured()
    if configured:
        st.success(f"✅ LangSmith tracing is **enabled** — project: `{LANGCHAIN_PROJECT}`")
    else:
        st.warning("⚠ LangSmith tracing is **disabled** or not configured.")

    st.markdown("""
    ### Setup Instructions
    
    1. Create an account at [smith.langchain.com](https://smith.langchain.com)
    2. Create an API key in your account settings
    3. Add to your `.env`:
    
    ```env
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langsmith_key_here
    LANGCHAIN_PROJECT=college-handbook-rag
    ```
    
    4. Restart the Streamlit app
    
    ### What Gets Traced
    
    | Component | Trace Data |
    |-----------|-----------|
    | User Query | Input text |
    | Retriever | Query, retrieved docs, latency |
    | Reranker | Candidates, scores, latency |
    | LLM Call | Prompt, response, tokens, latency |
    | Errors | Exception details |
    | Evaluation | Per-question runs |
    
    ### Notes
    - Tracing is optional — the app runs without it.
    - Token usage is tracked where supported by the model.
    - Verification: Traces are confirmed only when a live API call succeeds.
    """)

    st.markdown("---")
    st.markdown("### Tracing Configuration Status")

    # Imports moved to global scope to prevent UnboundLocalError
    status_df = pd.DataFrame([
        {"Setting": "LANGCHAIN_TRACING_V2", "Value": str(LANGCHAIN_TRACING_V2),
         "Status": "✅" if LANGCHAIN_TRACING_V2 else "❌"},
        {"Setting": "LANGCHAIN_PROJECT", "Value": LANGCHAIN_PROJECT,
         "Status": "✅"},
        {"Setting": "LANGCHAIN_ENDPOINT", "Value": LANGCHAIN_ENDPOINT,
         "Status": "✅"},
        {"Setting": "LANGCHAIN_API_KEY", "Value": "***" if configured else "(not set)",
         "Status": "✅" if configured else "❌"},
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)


# ─── API Setup guide ──────────────────────────────────────────────────────────

def render_setup_tab():
    st.markdown("## 🔑 API Setup Guide")

    st.markdown("""
    ### Step 1 — Create a Google AI Studio Project & API Key
    
    > ⚠ I cannot create a Google project or API key on your behalf. 
    > Follow these steps in your browser:
    
    1. Go to **[Google AI Studio](https://aistudio.google.com)**
    2. Sign in with your Google account
    3. Click **"Get API key"** in the top navigation
    4. Click **"Create API key in new project"** 
       *(Choose "new project" for separate quota — subject to Google's eligibility)*
    5. Copy the generated API key
    
    > **Important:** A new key in the **same** Google project does NOT create extra quota.  
    > A **separate project** may have its own free tier quota, subject to Google's eligibility.  
    > Do not assume a new project will automatically provide extra free requests.
    
    ---
    
    ### Step 2 — Configure Your Local Environment
    
    ```bash
    # 1. Copy the example env file
    cp .env.example .env
    
    # 2. Open .env and set your key
    # GOOGLE_API_KEY=AIza...your_actual_key_here...
    
    # 3. (Optional) Set LangSmith key
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_API_KEY=lsv2_...your_langsmith_key...
    ```
    
    > 🔒 **Never commit `.env` to Git.** It is listed in `.gitignore`.
    
    ---
    
    ### Step 3 — Verify API Access
    
    ```bash
    python -c "
    from config import validate_config, GEMINI_MODEL
    cfg = validate_config()
    print('API configured:', cfg['api_configured'])
    print('Model:', '""" + GEMINI_MODEL + r"""')
    "
    ```
    
    ---
    
    ### Step 4 — Generate the Handbook PDF
    
    ```bash
    python generate_handbook_pdf.py
    ```
    
    ---
    
    ### Step 5 — Launch the App
    
    ```bash
    streamlit run app.py
    ```
    
    ---
    
    ### Available Gemini Models (as of 2025)
    
    | Model | Notes |
    |-------|-------|
    | `gemini-1.5-flash` | Fast, cost-effective; recommended |
    | `gemini-1.5-pro` | Higher quality; uses more quota |
    | `gemini-2.0-flash` | Latest flash; verify availability |
    
    > Set your preferred model in `.env` as `GEMINI_MODEL=gemini-1.5-flash`
    
    ---
    
    ### Quota Notes
    
    - Google AI Studio free tier: typically 15 requests/minute, 1500/day (subject to change).
    - Evaluation runs ~75 API calls (25 questions × 3 versions). Allow time or spread across sessions.
    - If you hit quota: evaluation saves progress and resumes where it left off.
    - To enable billing: visit [Google Cloud Console](https://console.cloud.google.com) — 
      **do not enable billing without reviewing costs**.
    """)


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 class='gradient-text' style='font-size: 2.2rem; font-weight: 700; margin: 0;'>
            🎓 College Handbook Intelligence
        </h1>
        <p style='color: #b2bec3; font-size: 1rem; margin-top: 0.5rem;'>
            An End-to-End RAG System with FAISS · ChromaDB · Reranking · LangSmith
        </p>
        <div style='font-size:11px;color:#e17055;margin-top:4px;'>
            ⚠ Fictional educational data — Greenfield University — For demonstration purposes only
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    version_label = render_sidebar()
    version_key = version_label.split(" — ")[0]

    # Main tabs
    tabs = st.tabs([
        "💬 Ask Questions",
        "📊 Evaluation",
        "🏗️ Architecture",
        "🔭 LangSmith",
        "🔑 API Setup",
    ])

    with tabs[0]:
        render_qa_tab(version_key)

    with tabs[1]:
        render_evaluation_tab()

    with tabs[2]:
        render_architecture_tab()

    with tabs[3]:
        render_langsmith_tab()

    with tabs[4]:
        render_setup_tab()


if __name__ == "__main__":
    main()
