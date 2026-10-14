# Submission Cleanup

This is an in-place cleanup, not a rebuild. FastAPI, Streamlit, LangChain,
SQLite/SQLAlchemy, ChromaDB, auth, uploaded documents, and history remain intact.

## Final Settings

- Models: qwen2.5:1.5b (default for CPU), qwen2.5:3b (optional quality option).
- Embeddings: real normalized all-MiniLM-L6-v2 vectors, 384 dimensions.
- Chunk target: 240 tokenizer-measured tokens; complete-sentence overlap up to 60.
- Oversized units: LangChain recursive fallback; source offsets retained.
- Citations: filename, document/chunk IDs, physical PDF page or available Markdown
  section, exact source excerpt, and source link. Overlapping contained quotes
  from the same document/page are deduplicated.
- Unsupported evidence: one exact abstention sentence, separate from service errors.

## Fresh External Upload Verification

The automated API test creates new TXT/MD bytes at runtime, uploads them through
the file endpoint, verifies multiple Chroma chunks and metadata, queries their
new fact, checks the exact citation, asks an unrelated question, and deletes the
document. It does not depend on files in sample_documents. Unit/API tests use
explicit doubles for model isolation, not a production embedding fallback.

A separate live check used real MiniLM and Qwen 2.5 1.5B with fresh Markdown in
an isolated temporary database: 8 Chroma vectors, correct 29-month answer,
Service Warranty section, exact source excerpt, and correct unrelated abstention.
See external-upload-results.json. No original user uploads/history were modified.

## Verification And Evaluation

- Automated suite: 118 passed.
- Python compilation passed.
- Running /health and /docs returned 200; Swagger HTML was present.
- Running /models exposed only the final two models, defaulting to 1.5B.
- OpenAPI includes /chat/query.
- Lightweight local evaluation reports exact-source faithfulness, an independently
  labeled supporting-fact relevance proxy, semantic/BM25/hybrid precision and
  recall, and abstention accuracy. This is not a RAGAS/semantic-entailment score.
- Evaluation can upload evaluator-provided corpora with question/gold files.
- Saved live evaluation: 20/20 passed, exact-quote faithfulness 1.0, labeled
  relevancy proxy 1.0, abstention accuracy 1.0, hybrid precision@6 0.444,
  hybrid recall@6 1.0, and maximum warmed response 5579 ms. All 20 warmed
  requests finished below 10 seconds in this measured sample run.

## Changed Files

- .env.example
- README.md
- app/core/config.py
- app/api/schemas.py
- app/generation/model_gateway.py
- app/generation/grounding.py
- app/ingestion/chunker.py
- app/ingestion/pipeline.py
- app/retrieval/vector_store.py
- app/ui/streamlit_app.py
- scripts/check_document.py
- scripts/evaluate.py
- tests/conftest.py
- tests/integration/test_query_api.py
- tests/unit/test_grounding.py
- tests/unit/test_langchain_integrations.py
- docs/evaluation.md
- docs/langchain-refactor.md
- docs/external-upload-results.json
- docs/evaluation-results.json
- docs/submission-cleanup.md

## Limits

Answers remain strict source excerpts rather than unrestricted paraphrases.
Exact containment and gold-fact matching are useful checks, not proof of universal
semantic correctness. OCR, ambiguous tables, cross-language questions, charts,
and conversational follow-ups can remain limited. CPU latency is measured rather
than guaranteed. Existing compatible indexes/history remain readable; reindex
with --force only when upgrading older chunk profiles is desired.
