# Final Backend Stabilization

## Latest Grounding Recovery Verification

The citation renderer now accepts supported marker-less answer text and malformed
source IDs, attaching citations only from the bounded context actually supplied
to Ollama. Recovery requires at least 60% normalized claim-term overlap and all
numeric values in the supporting passage. Citation marker digits are removed
before value checks. Opposite negation/change direction uses exact source wording.
Empty claims, unrelated evidence, invented values and fabricated quotes still fail
closed. Infrastructure errors remain separate from evidence abstention.

The Ollama prompt numbers sources and requests matching `[1]`/`[2]` markers and
`source_id`. Raw structured-parser output is retained for grounding checks rather
than lost on a formatting error. Optional `DEBUG_LLM_OUTPUT=true` logs bounded raw
answer text; normal logs include validation/recovery counts without source context.
The API also accepts `document_id`/`model` aliases and rejects conflicting scopes.

- Full suite: 213 passed in 72.36 seconds.
- Requested live evaluation: 20/20 passed on the fixed labeled regression corpus;
  report: `docs/evaluation-results.json`.
- Real MiniLM/Ollama RBI PDF check: 3/3 passed; 11 chunks indexed.
- Cumulative reduction: 125 basis points, citation to PDF page 1.
- MCLR: declined by 20 basis points, citation to PDF page 2.
- Today's IPL score: abstained with no LLM invocation.
- Live RBI answers/report: `docs/rbi-grounding-results.json`.
- CPU timings during concurrent verification: about 35/50 seconds for the two
  answers, 838 ms for unrelated abstention; no under-10-second claim is made.

Grounding changes: `app/generation/grounded_answer.py`,
`app/generation/model_gateway.py`, `app/rag_pipeline.py`, `app/api/schemas.py`,
`app/core/config.py`, `.env.example`, `README.md`,
`tests/unit/test_citation_recovery.py`, and
`tests/integration/test_rbi_citation_recovery.py`.

Evaluation labels previously referred to missing KiwiDesk demo files. The default
evaluator now uses the explicitly labeled fictional regression corpus in
`tests/fixtures/evaluation_corpus/`, without modifying editable demo documents.
`scripts/evaluate.py` and the five regression corpus files were updated accordingly.
This is separate from the real RBI PDF verification and is not a universal
semantic-faithfulness guarantee.

The existing FastAPI, Streamlit, SQLite/SQLAlchemy, ChromaDB, authentication,
uploads, and saved conversations are retained. No new RAG frameworks are added;
the existing transitive PostHog dependency is constrained below version 6 to
match Chroma 0.6.3's API and eliminate spurious telemetry errors.

## Final Pipeline

Upload -> extract -> validate text -> sentence-aware adaptive chunking -> real
384-dimensional MiniLM embeddings -> ChromaDB -> vector top 8 + BM25 top 8 ->
RRF -> lightweight rerank -> 4 quick / 6 deep or overview context chunks -> context-only Ollama
claims -> exact source snippets and numbered citations -> abstention.

`app/rag_pipeline.py` is the production orchestrator. Factual queries do not use
the previous exact-sentence/qualifier gate or retrieval retries. A broad lexical
or cosine relevance check rejects clearly unrelated queries; inverse-frequency
term weights prevent a repeated product name alone from passing. Missing information
within related context is handled by the model's empty claims response.

Overviews add the first two chunks of the latest completed selected document
before hybrid matches, deduplicate, and request 3-5 bullet points. Short overviews
can be completed with exact sentences from retrieved context only. This samples
a document; it is not an exhaustive long-document summarizer.

The backend selects source snippets using claim-term overlap and numeric support,
correcting in-range model source numbers only against stronger retrieved evidence. Unknown sources,
invented quotes, and numeric values absent from the snippet are omitted.
Negation mismatches fall back to exact source wording, duplicate claims are
removed, and weak overview generalizations use source-only key points instead. Exact
citation validity is not a guarantee of semantic entailment. Faithfulness is
explicitly a lexical proxy; answer relevancy uses labeled supporting-fact checks.

## Models And Chunking

- Default LLM: `qwen2.5:1.5b`; optional: `qwen2.5:3b`.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`, normalized, 384 dimensions.
- Default chunk target: 240 tokenizer tokens, maximum 248; overlap up to 60.
- MiniLM's input limit is 256 tokens. Whole sentences are grouped when possible.
- Metadata retains document ID, filename, physical PDF page/section, chunk ID,
  source and offsets. Page offsets recover full source sentences for citations.
- New Chroma collections use cosine distance. Existing normalized 384-vector
  collections remain readable; query relevance uses recomputed true cosine.

Text-based PDF/TXT/MD are required and supported; existing DOCX/CSV/JSON remain.
OCR is disabled. Scanned PDFs receive a clear OCR-not-enabled error. The isolated
PDF subprocess avoids importing embedding/LLM libraries, limits BLAS threads,
and enforces the extraction deadline. Failed/pending documents cannot be queried.

## Files Changed

- API: `app/api/routes/chat.py`, `app/main.py`.
- Production pipeline: `app/rag_pipeline.py` (new).
- Generation: `app/generation/grounded_answer.py` (new),
  `app/generation/model_gateway.py`, `app/generation/overview.py`.
- Retrieval: `app/retrieval/service.py`, `app/retrieval/fusion.py`,
  `app/retrieval/vector_store.py`.
- Ingestion: `app/ingestion/pipeline.py`, `app/ingestion/loaders.py`,
  `app/ingestion/pdf_timeout.py`, `app/ingestion/pdf_worker.py` (new).
- Verification: `scripts/evaluate.py`, `scripts/check_document.py`,
  `scripts/submission_smoke.py` (new).
- Tests: `tests/conftest.py`, `tests/fakes.py`,
  `tests/integration/test_rag_api.py`, `tests/integration/test_overview.py`,
  `tests/integration/test_final_submission.py` (new),
  `tests/unit/test_latency_settings.py`, `tests/unit/test_document_variants.py`,
  `tests/unit/test_upload_timeout.py`, `tests/unit/test_overview_intent.py`,
  `tests/unit/test_final_pipeline.py` (new).
- Packaging/documentation: `requirements.txt`, `README.md`, this file.
- Generated reports: `docs/final-smoke-results.json`,
  `docs/final-evaluation-results.json`.
  Baseline/intermediate evaluation reports retain earlier failed runs.

## Final Verification Results

- Full automated suite: 177 passed in 29.20 seconds.
- Dependency validation: `python -m pip check` reports no broken requirements.
- Live Ollama evaluation: 20/20 passed, including answer polarity checks.
- Mean labeled answer relevancy: 1.0; abstention accuracy: 1.0.
- Mean citation groundedness: 1.0; lexical faithfulness proxy: 0.9522.
- Maximum live evaluation latency: 20,765 ms on this CPU machine.
- Fresh PDF/TXT/MD smoke check passed with real MiniLM and Ollama: factual
  answer with PDF page citation, three overview bullets citing three chunks,
  unrelated-question abstention, health, document listing, and deletion.
- Smoke timings: factual answer 16,054 ms; overview 20,210 ms; unrelated
  abstention 578 ms. The under-10-second generation target is not met on this CPU.

These results validate the tested fixtures, not all possible documents or full
semantic faithfulness. Inspect the generated reports for individual answers and
supporting passages. No existing user documents were deleted by these checks.

## Local Commands

Run from the project folder with Python 3.11:

```powershell
cd "C:\Users\Admin\Documents\ChatGPT\rag chatbot\rag-knowledge-assistant"
.\.venv\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
python -m uvicorn app.main:app --port 8000
```

Use the existing Ollama service; run `ollama serve` only if it is not running.
Do not start another backend while port 8000 is occupied. Stop the existing
backend with Ctrl+C before restarting it. In another activated terminal:

```powershell
python -m streamlit run app/ui/streamlit_app.py --server.port 8501
```

For a fresh checkout, first run `py -3.11 -m venv .venv`, activate it,
`python -m pip install -r requirements.txt`, and copy `.env.example` to `.env`.
Set a private JWT secret. Never commit `.env`, uploads, SQLite, model caches,
or Chroma data; existing `.gitignore` excludes them.

Verification (with the numerical-library environment variables above):

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/submission_smoke.py
python scripts/evaluate.py --output docs/final-evaluation-results.json
```

The smoke/evaluation scripts use temporary databases/indexes, not saved user
documents. CPU latency varies; under 10 seconds is not guaranteed. Keep one
backend and one frontend running to avoid memory/page-file exhaustion.
