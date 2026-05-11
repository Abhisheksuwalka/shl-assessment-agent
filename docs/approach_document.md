# SHL Assessment Recommendation Agent — Approach Document

## 1. Design Choices

### Architecture: Stateless RAG + Intent Routing

I chose a **custom RAG (Retrieval-Augmented Generation) pipeline with intent routing** over a pure LLM approach or framework-heavy solution (LangChain/LlamaIndex). The key reasons:

- **No hallucination risk**: The LLM never generates assessment names or URLs from memory. All recommendations come directly from FAISS-retrieved catalog entries, with URLs copied verbatim.
- **Deterministic routing**: A lightweight intent classifier (CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE) controls agent behavior, preventing premature recommendations or off-topic responses.
- **Simplicity & debuggability**: Every component is a plain Python module (~100–300 lines each) with clear inputs/outputs. No framework abstractions to debug through.

### Stateless Design

The API is fully stateless — the client sends the complete conversation history on every `POST /chat` request. This simplifies deployment (no session storage, no Redis) and makes the service horizontally scalable.

---

## 2. Retrieval Setup

### Embedding Model: `all-MiniLM-L6-v2`

- **Free, local, fast**: Runs on CPU in ~200ms per query, no API key needed
- **384-dim vectors**: Small enough for the full 377-item catalog to fit in memory (~580KB FAISS index)
- **Good semantic quality**: Captures role/skill/domain similarity well for short technical descriptions

### FAISS Configuration

- **Index type**: `IndexFlatIP` (inner product with L2-normalized vectors = cosine similarity)
- **Two-stage retrieval**: Stage 1 retrieves top-30 semantic candidates; Stage 2 post-filters by metadata constraints (job level, duration, language, test type, adaptive)
- **Constraint relaxation**: If filtering yields zero results, constraints are progressively dropped in priority order (duration → language → adaptive → job level → test type) to always return something useful

### Catalog Preprocessing

Each of the 377 catalog items is enriched with:
- Parsed `duration_minutes` (integer)
- `type_codes` (single-letter: K, P, A, B, C, E, S, D)
- A `text_for_embedding` field concatenating name, description, categories, job levels, duration, and languages

---

## 3. Prompt Design

### System Prompt Structure

The system prompt establishes:
1. **Role**: "Expert SHL Assessment Advisor" — scoped exclusively to SHL assessments
2. **Behavioral rules**: Clarify before recommending on vague queries; one question per turn; never hallucinate URLs; stay grounded in catalog data
3. **Output format**: Conversational reply only — structured recommendations are handled separately by the code, not by the LLM

### Intent-Specific Prompts

- **Classifier prompt**: Few-shot examples for all 5 intents, with explicit rules (e.g., "default to CLARIFY when in doubt")
- **Extractor prompt**: Outputs a JSON schema with validated fields (job_levels from a fixed enum, test_types as single-letter codes)
- **Clarification prompt**: Tracks previously asked questions to avoid repetition; prioritizes high-impact missing info (role → seniority → test type → duration)
- **Recommendation prompt**: Injects the retrieved catalog data and instructs the LLM to explain fit — never to list URLs

### Hallucination Prevention

- URLs and assessment names only come from the retriever's catalog data, never from LLM generation
- The LLM receives a context block labeled "use ONLY these for recommendations"
- A `field_validator` on the Pydantic response schema enforces ≤ 10 recommendations

---

## 4. Evaluation Approach

### Unit Tests (97 tests, all passing)

| Test File | Tests | Coverage |
|---|---|---|
| `test_schema.py` | 12 | Pydantic schemas, serialization, validation errors |
| `test_retriever.py` | 14 | Semantic search, metadata filtering, constraint relaxation, URL validation |
| `test_classifier.py` | 14 | All 5 intents, edge cases, fallback behavior, noisy LLM output |
| `test_extractor.py` | 24 | JSON extraction, constraint parsing/merging, search query building |
| `test_agent.py` | 8 | End-to-end all 5 intent paths with mocked LLM, schema compliance |
| `test_probes.py` | 7 | All 7 behavioral probes (vague, off-topic, injection, legal, refine, compare, turn cap) |

### Integration Evaluation

`scripts/evaluate.py` replays 10 conversation scenarios against the running service, validating:
- Schema compliance on every response
- Correct recommendation presence/absence
- Response time < 30 seconds
- Multi-turn conversation handling

### Behavioral Probes Verified

All 7 probes pass: vague first message → clarify; off-topic → refuse; prompt injection → no leak; legal question → refuse; refinement → updated list; comparison → grounded answer; turn cap → best-effort shortlist.

---

## 5. What Didn't Work

- **Pure LLM (no retrieval)**: The LLM hallucinated assessment names and URLs, even with explicit instructions not to. RAG solved this completely.
- **Keyword-only search**: Failed on semantic queries like "communication skills test" because the catalog uses different terminology ("Verbal Reasoning", "Spoken English"). Semantic embeddings bridge this gap.
- **LangChain/LlamaIndex**: Added complexity without improving retrieval quality. The custom pipeline is easier to debug, explain, and modify.

---

## 6. AI Tool Usage

- **Groq (Llama-3.3-70b-versatile)**: Used as the LLM for intent classification, constraint extraction, and response generation. Free tier, fast inference (~2-5s per call).
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Local embedding model for FAISS index building and query embedding. No API calls needed.
- **AI coding assistant**: Used for code generation, test writing, and documentation. All generated code was reviewed and tested.

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| API Framework | FastAPI | Required by assignment; async support |
| LLM | Groq (Llama-3.3-70b) | Free tier, fast, good quality |
| Embeddings | all-MiniLM-L6-v2 | Free, local, fast |
| Vector Store | FAISS (CPU) | Free, no server needed |
| Deployment | Render (Docker) | Free tier, Docker support |
