# Evaluation

Run deterministic tests with:

```powershell
python -m pytest -q -p no:cacheprovider
```

The suite covers auth, ownership, retrieval, citations, token budgets, source offsets, sentence boundaries, reindex recovery, CSV/JSON/DOCX, empty/corrupt files, encrypted/scanned PDFs, column layouts, and table labels. Unit/API model selection is mocked; this does not prove LLM quality.

Run live evaluation with Sentence Transformers and Ollama:

```powershell
python scripts/evaluate.py
python scripts/evaluate.py --documents tests/fixtures/general_corpus --questions tests/general_questions.json --gold tests/general_gold.json
```

The evaluation runner uses an isolated temporary index and searches the entire corpus for each question. The varied corpus includes manuals, policies, contracts, shipping terms, CSV fees, and nested JSON. Independently labeled gold passages determine relevant chunks. Passing requires:

- expected answerability
- expected citation source
- expected supporting fact inside a selected citation (not just the right filename)
- abstention for unanswerable questions

Real LangChain Hugging Face embeddings and Ollama are the default. Add `--offline-selector` for deterministic pipeline selection tests; this mode does not measure LLM quality. There is no production or evaluation hash-embedding switch. API/unit tests explicitly inject doubles from `tests/fakes.py`.

The default model is `qwen2.5:1.5b` for faster CPU inference; `--model qwen2.5:3b` selects the optional quality model. Failure tests cover embedding load/dimension errors, model allowlisting, missing models, unavailable Ollama, generation errors, and the `/chat/query` API. An infrastructure error is not counted as an evidence abstention.

Reports include semantic/BM25/hybrid precision@6 and recall@6, citation precision, exact-quote faithfulness, supporting-fact accuracy, retrieval/generation/total latency, and counts under 10 seconds. Precision uses six as its denominator; recall is undefined for unsupported questions. Model warm-up is reported separately.

`mean_answer_relevancy` is a local proxy: the expected source and independently labeled gold supporting fact must occur in a selected citation. It is not a RAGAS/LLM-judge score or proof of complete semantic relevance. `abstention_accuracy` measures the expected refuse/answer decision across all labeled questions. RAGAS is not installed or required; this lightweight evaluation uses local models only.

Save a live report:

```powershell
python scripts/evaluate.py --output docs/evaluation-results.json
```

For evaluator-provided documents, use `--documents PATH --questions QUESTIONS_JSON --gold GOLD_JSON`. Questions use the existing evaluation-question schema, and gold maps each question to an independently verified exact supporting source span. Files are uploaded through the API into an isolated temporary index, not read directly by the answer generator. The fresh-upload API regression independently constructs new TXT/MD bytes and verifies chunking, Chroma storage, source citations, and unrelated abstention.

For a read-only check of an indexed PDF:

```powershell
python scripts/check_document.py --document-id YOUR_DOCUMENT_ID --questions tests/fixtures/rbi_questions.json
```

The PDF regression has ten questions, not a universal acceptance test. Add independently labeled questions for new documents. Exact-quote faithfulness of 1.0 proves parsed-source containment only, not relevance, completeness, or OCR correctness. Cold models, large corpora, and CPU load can exceed 10 seconds. Review extraction warnings and original pages for complex tables or scanned documents.

