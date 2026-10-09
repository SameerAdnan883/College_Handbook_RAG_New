# 🎓 College Handbook Intelligence — End-to-End RAG System

> ⚠️ **FICTIONAL EDUCATIONAL DATA** — Greenfield University is a fictional institution created for this RAG demonstration. Not affiliated with any real institution.

An end-to-end Retrieval-Augmented Generation (RAG) system built with LangChain, Gemini, FAISS, ChromaDB, cross-encoder reranking, and LangSmith monitoring.

---

## 🏗️ Architecture Overview

```
V1 → Basic LangChain (Prompt → LLM → Output Parser)
V2 → FAISS RAG (PDF → Embeddings → FAISS → LLM)
V3 → ChromaDB RAG (PDF → Embeddings → ChromaDB → LLM)
V4 → Reranking (FAISS → Cross-Encoder → Top-N → LLM)
```

---

## 📁 Project Structure

```
College_Handbook_RAG_New/
├── app.py                        # Streamlit web application
├── v1_basic.py                   # V1: LangChain basics
├── v2_faiss.py                   # V2: FAISS RAG
├── v3_chroma.py                  # V3: ChromaDB RAG
├── v4_reranking.py               # V4: Reranking pipeline
├── evaluation.py                 # Comprehensive evaluation (25 questions)
├── retrieval_evaluation.py       # Retrieval-specific metrics
├── config.py                     # Centralized configuration
├── generate_handbook_pdf.py      # Fictional handbook PDF generator
├── requirements.txt
├── .env.example                  # Credential template
├── .gitignore
├── sample_college_handbook.pdf   # Generated fictional handbook
├── data/
│   ├── evaluation_questions.csv  # 25 Q&A evaluation pairs
│   ├── evaluation_results.csv    # Generated after running evaluation
│   └── eval_summary.csv
├── artifacts/                    # Charts and exports
├── faiss_index/                  # FAISS vector index (auto-created)
├── chroma_db/                    # ChromaDB persistence (auto-created)
└── tests/
    ├── conftest.py
    ├── test_v1.py
    ├── test_v2.py
    ├── test_v3.py
    ├── test_v4.py
    └── test_evaluation.py
```

---

## ⚡ Quick Start

### Step 1 — Clone and Install

```bash
git clone <your-repo-url>
cd College_Handbook_RAG_New

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=AIza...your_actual_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001

# Optional: LangSmith monitoring
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=lsv2_...your_langsmith_key
LANGCHAIN_PROJECT=college-handbook-rag
```

> 🔒 `.env` is listed in `.gitignore`. **Never commit it.**

### Step 3 — Get a Google AI API Key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Click **"Get API key"** → **"Create API key in new project"**
3. Copy the key into `.env`

> **Quota Note:** A new key in the same Google project does NOT create extra quota.  
> A separate project may have its own free-tier quota (subject to Google's eligibility).  
> Free tier is typically 15 req/min, 1500/day — verify at [ai.google.dev](https://ai.google.dev).

### Step 4 — Generate the Handbook PDF

```bash
python3 generate_handbook_pdf.py
```

### Step 5 — Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Tests

```bash
# All tests (no API key required)
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/test_v4.py -v
```

**Test coverage:** 49 tests across V1, V2, V3, V4, and evaluation — all mock-based, no API calls required.

---

## 📊 Running Evaluation

> ⚠️ Requires a valid API key with sufficient quota. ~75 API calls (25 × 3 versions).

```bash
# All versions
python3 -m python evaluation.py

# Specific versions
python3 evaluation.py --versions V2 V3

# Force rerun (ignore cached results)
python3 evaluation.py --force-rerun
```

Results are saved after **each question** and are fully resumable on quota errors.

---

## 🔭 LangSmith Monitoring

1. Create account at [smith.langchain.com](https://smith.langchain.com)
2. Generate API key
3. Set in `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=college-handbook-rag
   ```
4. Restart app — traces appear automatically in LangSmith dashboard

The app runs normally without LangSmith (tracing is optional).

---

## 🚀 Deployment (Hugging Face Spaces) — Recommended

Hugging Face Spaces provides a generous free tier (16GB RAM) which is highly recommended due to the PyTorch memory requirements in the V4 Reranking pipeline.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space name, select **Streamlit** as the Space SDK, and create the space.
3. Push this repository to your Space:
   ```bash
   git remote add huggingface https://huggingface.co/spaces/your-username/your-space-name
   git push huggingface main
   ```
4. In your Space's **Settings**, go to **Variables and secrets**.
5. Add your API keys as **Secrets**:
   - `GOOGLE_API_KEY`: "your_google_key"
   - `LANGCHAIN_API_KEY`: "your_langsmith_key"
   - `LANGCHAIN_TRACING_V2`: "true"
   - `LANGCHAIN_PROJECT`: "college-handbook-rag"

> **Note:** The FAISS and ChromaDB indexes will automatically rebuild from `sample_college_handbook.pdf` on the first query after a Space restart.

---

## 🚀 Deployment (Streamlit Community Cloud)

1. Push code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Secrets** in the Streamlit dashboard:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   GEMINI_MODEL = "gemini-1.5-flash"
   EMBEDDING_MODEL = "models/gemini-embedding-001"
   ```
5. Deploy — the app builds and runs automatically

> **Warning:** Streamlit Cloud limits memory to ~1GB. V4 (PyTorch Cross-Encoder) may cause Out-Of-Memory (OOM) crashes on this platform. Hugging Face Spaces is strongly preferred.

---

## 📐 Technical Details

### Chunking Strategy
- Splitter: `RecursiveCharacterTextSplitter`
- Chunk size: 800 characters
- Overlap: 150 characters
- Separators: `["\n\n", "\n", ".", " ", ""]`

### Embedding Model
- `models/gemini-embedding-001` via Google AI

### FAISS vs ChromaDB
| Feature | FAISS (V2) | ChromaDB (V3) |
|---------|-----------|---------------|
| Speed | Very fast | Slightly slower |
| Persistence | Manual save/load | Automatic |
| Metadata filtering | No | Yes |
| Deduplication | No | Yes (by ID) |
| Accuracy | Same embedding | Same embedding |

> ChromaDB does NOT produce inherently more accurate results than FAISS. Both use the same embeddings. Differences arise from metadata filtering and deduplication.

### Reranking (V4)
- Retrieve 15 candidates via FAISS
- Score with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Select top-4 for generation
- **Scores are raw logits** — not probabilities

### Evaluation Methodology
- **Keyword scoring**: fraction of expected keywords found in answer (transparent, reproducible)
- **Recall@k**: fraction of relevant chapters in top-k (heuristic chapter labels)
- **MRR**: reciprocal rank of first relevant source (heuristic)
- API failures are **not** counted as wrong answers
- Coverage reported separately from accuracy

---

## 🔑 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | LLM model name |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Embedding model |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `college-handbook-rag` | LangSmith project |
| `CHUNK_SIZE` | No | `800` | Text chunk size |
| `CHUNK_OVERLAP` | No | `150` | Chunk overlap |
| `TOP_K` | No | `5` | Retrieved documents |
| `RERANK_CANDIDATES` | No | `15` | V4 candidates |
| `RERANK_TOP_N` | No | `4` | After reranking |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `GOOGLE_API_KEY not configured` | Copy `.env.example` → `.env` and add key |
| `PDF not found` | Run `python3 generate_handbook_pdf.py` |
| `Quota exceeded` | Wait for quota reset; evaluation resumes automatically |
| `FAISS index not found` | Triggers rebuild automatically on first query |
| `ChromaDB error` | Delete `chroma_db/` folder and restart |
| Streamlit import errors | Check Python version (3.10+ required) |

---

## 📜 License

This is a demonstration project using fictional data. Not for production use.

---

*Built with LangChain · Google Gemini · FAISS · ChromaDB · Sentence-Transformers · Streamlit*
