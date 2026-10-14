# LangChain Refactor

The existing FastAPI, Streamlit, SQLite/SQLAlchemy, ChromaDB, auth, upload,
source citations, and conversation storage are retained. LangChain components
implement the loading interface, sentence-aware splitter with recursive fallback,
Hugging Face embedding wrapper, Chroma vector store/retriever, and ChatOllama.
BM25/RRF, source validation, rendering, and abstention remain custom.

Final model choices: qwen2.5:1.5b (default, faster CPU inference), qwen2.5:3b
(optional quality model). Missing models are
reported, never silently substituted. Both models must be installed on the
machine running Ollama, not necessarily the evaluator's browser machine.

## Changed Files

- requirements.txt
- .env.example
- README.md
- docs/evaluation.md
- docs/langchain-refactor.md
- app/core/config.py
- app/core/errors.py
- app/main.py
- app/api/schemas.py
- app/api/routes/chat.py
- app/ingestion/langchain_documents.py
- app/ingestion/chunker.py
- app/ingestion/profile.py
- app/ingestion/pipeline.py
- app/retrieval/embeddings.py
- app/retrieval/langchain_embeddings.py
- app/retrieval/service.py
- app/retrieval/vector_store.py
- app/generation/grounding.py
- app/generation/model_gateway.py
- app/ui/streamlit_app.py
- scripts/check_document.py
- scripts/evaluate.py
- scripts/reindex_documents.py
- tests/conftest.py
- tests/fakes.py
- tests/integration/test_ingestion_variants.py
- tests/integration/test_query_api.py
- tests/unit/test_latency_settings.py
- tests/unit/test_langchain_integrations.py

## Compatibility And Limits

- POST /chat/query is preferred; POST /chat remains compatible.
- Production embeddings are always normalized 384-dimensional MiniLM vectors.
  Test doubles live only under tests; there is no runtime hash/fake fallback.
- The encoder accepts 256 tokens. Chunks target 240 tokens with up to 60 tokens
  of complete-sentence overlap, rather than truncating a 350-token chunk.
- Current answers are extractive selected passages, not free-form paraphrases.
  Exact source containment does not prove perfect relevance or completeness.
- OCR, ambiguous tables, charts, cross-language questions, and conversational
  follow-ups can remain limited. Existing history is not rewritten.
- ChromaDB stays at 0.6.3; LangChain integrations are pinned for compatibility.
- Compatible existing indexes remain readable without rewriting uploaded files.
  Optional reindex upgrades chunk profiles; --force rebuilds a new collection.
- CPU inference is not guaranteed to finish in 10 seconds.
  A live existing-PDF check with qwen2.5:3b produced a correct page-48 citation
  and took about 27 seconds for generation. The final default is the faster
  qwen2.5:1.5b model; both supported models have README pull commands.

Setup, backend/frontend commands, evaluation commands, and model pull commands
are in README.md. No original uploads, auth records, or history were deleted.

## Verification

- Full regression suite after submission cleanup: 118 passed.
- Dependency check: no broken requirements.
- Real Hugging Face wrapper: 384 dimensions, 256-token model input limit.
- Real Qwen 2.5 3B + existing PDF index: supported answer with filename,
  page number, chunk ID, and exact source excerpt.
- Running backend: health returned 200; /models exposed only the two allowed
  models; OpenAPI included /chat/query. Running Streamlit health returned 200.
- API tests exercised TXT/MD uploads, list/query/delete, unrelated abstention,
  invalid models/files, and explicit infrastructure/model/embedding errors.
