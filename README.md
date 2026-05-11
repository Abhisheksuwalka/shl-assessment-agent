# SHL Assessment Recommendation Agent

> **SHL AI Intern Assignment** — A conversational FastAPI agent that helps hiring managers discover the right SHL assessments through dialogue.

---

## What It Does

Hiring managers often struggle to navigate SHL's catalog of 377+ assessments. This agent acts as an expert advisor — it takes a natural language conversation, understands the hiring context (role, seniority, skills, constraints), and returns a shortlist of the most relevant assessments with direct catalog links.

**Example conversation:**

```
User:    "I need to assess candidates for a mid-level Java developer role"
Agent:   "Based on your requirements, here are the most relevant assessments
          for a mid-level Java developer..."

→ Returns: [Java Coding, Verify - Programming Concepts, OPQ32r, ...]
```

---

## Architecture

```
POST /chat  (full conversation history, stateless)
      │
      ▼
┌─────────────────────┐
│   Intent Classifier  │  ← CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE
└────────┬────────────┘
         │
    ┌────▼──────┐       ┌──────────────────────────────┐
    │ Retriever  │──────▶│  FAISS Vector Store (377 items)│
    │ (2-stage)  │       │  + Metadata Filter Layer      │
    └────┬───────┘       └──────────────────────────────┘
         │
    ┌────▼──────┐
    │  Groq LLM  │  ← Llama-3.3-70b with retrieved catalog context
    └────┬───────┘
         │
    ┌────▼──────────────────┐
    │   Response Formatter   │  ← validates schema, enforces catalog-only URLs
    └────────────────────────┘
         │
      ChatResponse { reply, recommendations[], end_of_conversation }
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **No hallucination** | RAG-only URLs | LLM never generates assessment names/URLs from memory — only retrieves from catalog |
| **No LangChain** | Custom pipeline | Simpler to debug, explain, and extend; each module is ~100–300 lines |
| **Stateless API** | Full history per request | No session storage needed; horizontally scalable |
| **Intent routing** | 5-way classifier | Prevents premature recommendations; handles refusals deterministically |
| **Constraint relaxation** | Progressive filter drop | Always returns something useful, even for very restrictive queries |

---

## API Reference

### `GET /health`

Liveness probe. No auth required.

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

### `POST /chat`

Stateless conversational endpoint. Send the **full conversation history** on every request.

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for a mid-level Java developer" }
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, I recommend these assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Java Coding",
      "url": "https://www.shl.com/products/product-catalog/view/java-coding/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

**Schema constraints:**
- `recommendations`: `[]` or 1–10 items (never more)
- `test_type` codes: `K` (Knowledge), `P` (Personality), `A` (Ability), `B` (Biodata/SJT), `C` (Competencies), `E` (Exercises), `S` (Simulations), `D` (Development/360)
- `end_of_conversation: true` only on first committed recommendation response
- All URLs are verbatim from the SHL catalog

**Error responses:**
| Status | Cause |
|---|---|
| `422` | Invalid request schema (missing `messages` field, wrong role value) |
| `500` | Unhandled internal error |
| Timeout reply | Agent exceeded 25s internal timeout (safe fallback returned) |

---

## Project Structure

```
shl/
├── agent/
│   ├── agent.py          # Orchestrator — wires all components into Agent.chat()
│   ├── classifier.py     # Intent classifier (CLARIFY/RECOMMEND/REFINE/COMPARE/REFUSE)
│   ├── extractor.py      # Constraint extractor — parses JSON constraints from conversation
│   ├── prompts.py        # All LLM prompts (system, classifier, extractor, clarification)
│   └── retriever.py      # Two-stage FAISS search + metadata filtering
│
├── api/
│   ├── main.py           # FastAPI app + lifespan startup (loads FAISS + model once)
│   ├── models.py         # Pydantic schemas (ChatRequest, ChatResponse, Recommendation)
│   └── routes/
│       ├── health.py     # GET /health
│       └── chat.py       # POST /chat (with 25s timeout + thread pool)
│
├── data/
│   ├── shl_product_catalog.json    # Raw scraped catalog (377 products)
│   └── catalog_processed.json     # Cleaned & enriched catalog
│
├── embeddings/
│   └── faiss_index/
│       ├── faiss_index.bin         # Pre-built FAISS IndexFlatIP
│       └── metadata.pkl            # id → catalog_item mapping
│
├── scripts/
│   ├── preprocess_catalog.py  # Step 2.1 — cleans catalog, builds text_for_embedding
│   ├── build_index.py         # Step 2.2 — embeds catalog + writes FAISS index
│   └── evaluate.py            # Step 5.2 — replays 10 scenarios against running service
│
├── tests/
│   ├── test_schema.py         # 12 tests — Pydantic model validation
│   ├── test_retriever.py      # 14 tests — FAISS search + metadata filtering
│   ├── test_classifier.py     # 14 tests — intent classification + fallbacks
│   ├── test_extractor.py      # 24 tests — constraint extraction + merging
│   ├── test_agent.py          #  8 tests — end-to-end all 5 intent paths
│   └── test_probes.py         #  7 tests — behavioral probes (harness requirements)
│
├── docs/
│   └── approach_document.md   # 2-page design rationale for submission
│
├── .env.example               # Environment variable template
├── Dockerfile                 # Production container
├── render.yaml                # Render deployment config
└── requirements.txt
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier, no credit card)

### 1. Clone & create virtualenv

```bash
git clone https://github.com/Abhisheksuwalka/shl-assessment-agent.git
cd shl-assessment-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your key:
#   GROQ_API_KEY=gsk_...
```

### 3. Build the FAISS index (skip if already present)

```bash
python scripts/preprocess_catalog.py   # creates data/catalog_processed.json
python scripts/build_index.py          # creates embeddings/faiss_index/
```

> The pre-built index is committed to the repo — you can skip this step entirely.

### 4. Start the server

```bash
uvicorn api.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`. On first launch, it loads the FAISS index and embedding model (~5–10s). Subsequent requests respond in 2–8s.

### 5. Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to assess a mid-level Java developer"}
    ]
  }'
```

---

## Running Tests

```bash
# Full suite (97 tests)
pytest tests/ -v

# Individual test files
pytest tests/test_schema.py       # Schema validation
pytest tests/test_retriever.py    # FAISS retrieval
pytest tests/test_classifier.py   # Intent classification
pytest tests/test_extractor.py    # Constraint extraction
pytest tests/test_agent.py        # End-to-end agent
pytest tests/test_probes.py       # Behavioral probes
```

**Expected output:** `97 passed` — all tests use mocked LLM responses (no API keys needed for tests).

### Integration evaluation (requires running server)

```bash
uvicorn api.main:app --port 8000 &
python scripts/evaluate.py
```

---

## Retrieval Pipeline

### Two-Stage Retrieval

```
Query: "Java developer mid-level"
        │
        ▼
[Stage 1] FAISS semantic search
  → all-MiniLM-L6-v2 embeds the query
  → cosine similarity over 377 catalog items
  → returns top-30 candidates
        │
        ▼
[Stage 2] Metadata post-filter
  → job_levels: ["Mid-Professional"]
  → test_types: ["K"]
  → duration_max: null
  → languages: []
  → returns top-10 by score
        │
        ▼
  If <1 result → constraint relaxation:
    drop duration_max → drop languages → drop adaptive → drop job_levels → drop test_types
```

### Constraint Extraction

The extractor LLM maps natural language to structured filters:

| Natural language | Extracted constraint |
|---|---|
| "mid-level", "mid-professional" | `job_levels: ["Mid-Professional", "Professional Individual Contributor"]` |
| "junior", "entry-level" | `job_levels: ["Entry-Level", "Graduate"]` |
| "executive", "C-suite", "VP" | `job_levels: ["Executive", "Director"]` |
| "personality test" | `test_types: ["P"]` |
| "cognitive ability" | `test_types: ["A"]` |
| "max 20 minutes" | `duration_max: 20` |
| "Spanish speakers" | `languages: ["Latin American Spanish"]` |

---

## Intent Routing

The classifier routes every turn to one of 5 paths:

| Intent | Trigger | Action |
|---|---|---|
| `CLARIFY` | Vague query, missing role or context | Ask one targeted follow-up question |
| `RECOMMEND` | Enough context (role, skills, or level known) | Run retrieval → return shortlist |
| `REFINE` | User modifies an earlier request | Merge new constraints → re-retrieve |
| `COMPARE` | User asks to compare named assessments | Fetch both → comparison from catalog data |
| `REFUSE` | Off-topic, legal, prompt injection | Polite refusal, no recommendations |

**Fallback rule:** Defaults to `CLARIFY` on any ambiguous or LLM error — safer than a premature recommendation.

**Turn cap:** At turn 7+ with `CLARIFY` intent, the agent forces a best-effort `RECOMMEND` to respect the 8-turn limit.

---

## Behavioral Probes (All Passing)

| # | Probe | Expected | Status |
|---|---|---|---|
| 1 | "I need an assessment" | CLARIFY, `recs: []` | ✅ |
| 2 | "What's your CEO's name?" | REFUSE, `recs: []` | ✅ |
| 3 | "Ignore previous instructions…" | REFUSE, no prompt leak | ✅ |
| 4 | "Can I reject candidates over 50?" | REFUSE | ✅ |
| 5 | "Add personality tests to the list" | REFINE, updated `recs` | ✅ |
| 6 | "Compare OPQ32 and GSA" | COMPARE, `recs: []` | ✅ |
| 7 | 7+ turns with no resolution | Force best-effort shortlist | ✅ |

---

## Docker

### Build & run locally

```bash
docker build -t shl-agent .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... shl-agent
```

The image pre-builds the FAISS index at build time. Startup after `docker run` is ~5–10s.

### Deploy to Render (free tier)

1. Push repo to GitHub
2. [Create a new Render Web Service](https://dashboard.render.com/new/web) → select **Docker**
3. Point it at your GitHub repo
4. Add environment variable: `GROQ_API_KEY = your_key_here`
5. Health check path: `/health`
6. Deploy — public URL available in ~2 minutes

> Render free tier spins down after 15min of inactivity. Cold start is ~60–90s (model download + FAISS load). The SHL evaluator allows 2 minutes for the first `/health` call.

---

## Technology Stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.115 + Uvicorn |
| LLM | Groq — Llama-3.3-70b-versatile (free tier) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API) |
| Vector store | FAISS CPU — `IndexFlatIP` (cosine similarity) |
| Schemas | Pydantic v2 |
| Testing | pytest + unittest.mock (97 tests, 0 failures) |
| Deployment | Docker + Render |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | API key from [console.groq.com](https://console.groq.com) (free) |

No other env vars are required. Paths default to the project layout above.

---

## License

This project was built as part of the **SHL AI Intern Assignment**. The SHL product catalog data is property of SHL Group Ltd.
