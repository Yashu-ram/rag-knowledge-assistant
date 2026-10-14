# Grounded Answer Synthesis

## Root Cause

The previous quick prompt encouraged one claim and capped most answers at two
points. There was no question-shaped completeness check, so a valid citation to
one fact could mask an omitted comparison side. Evaluation checked provenance
and polarity but not every expected answer item.

## Changes

- Generic factual/explanation/comparison/multipart/summary answer planning.
- Numbered context containing filename, page/section and chunk ID.
- Natural synthesis rules, exact values/IDs/dates, separate source-bound claims.
- Supported requested-part coverage checked against cited answer lines.
- At most one synthesis retry, with the original question and identical context.
- More complete validated answers retained; a bad retry cannot erase a valid
  first answer. All attempts contribute to reported model latency.
- Partial answers identify absent context separately from generation omissions.
- Existing numeric validation, source-bound citation recovery and unrelated
  evidence abstention are retained. No extra retrieval or embedding operation.
- Split multipart evidence can establish relevance through actual requested-part
  source passages, avoiding a false rejection by the whole-question score.
  Statements merely declaring information absent do not count as such evidence.
- Raw answer/debug previews limited to 150 characters, and off by default.
- Evaluation checks answer content, expected numbers/keywords, each labeled
  multipart item, and omitted supported-part metadata, as well as citations.

## Files Changed

- `app/generation/synthesis.py` (new)
- `app/generation/model_gateway.py`
- `app/rag_pipeline.py`
- `scripts/evaluate.py`
- `tests/fakes.py`
- `tests/unit/test_synthesis.py` (new)
- `tests/integration/test_synthesis_api.py` (new)
- `tests/evaluation_questions.json`
- `tests/retrieval_gold.json`
- `tests/synthesis_recheck_questions.json` (new)
- `tests/fixtures/evaluation_corpus/shipping.txt` (new)
- `tests/fixtures/evaluation_corpus/warranty.txt` (new)
- `README.md` and this document
- Generated evaluation report: `docs/evaluation-results.json`

## Verification

Final full suite: 231 passed in 106.42 seconds. Generic same/separate-chunk
comparisons, multipart retry, genuine missing evidence, failed-retry preservation,
summary bullets, IPL abstention, RBI fresh/outstanding WALR (5/26 basis points),
and median one-year MCLR (declined by 20 basis points) are covered by automated
tests. API tests use real PDF extraction and explicit test model doubles; live
model quality is measured separately by the evaluation command.

Live evaluation: 23/23 verified incrementally. The initial full run passed 20/23
and exposed two overly strict incidental endpoint-version checks plus one false
multipart relevance rejection. After corrections, all three cases were generated
and evaluated again with real Ollama and passed. The final combined report keeps
the 20 passing original answers and the three fresh rechecks, explicitly recording
that strategy. No rerun is represented as a fresh full 23-question execution.
Mean lexical faithfulness proxy: 0.9655; abstention accuracy: 1.0.
Reports: `docs/evaluation-results.json`, `docs/synthesis-initial-results.json`,
and `docs/synthesis-recheck-results.json`.

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/evaluate.py --output docs/evaluation-results.json
```

## Limits

Requested-part extraction and completeness are conservative lexical/numeric
proxies, not a full semantic parser or independent entailment model. Unusual
phrasing can still yield partial answers. Retrieved context is bounded and does
not prove absence from an entire long document. A retry may increase CPU latency.
Upload, extraction, chunking, MiniLM embeddings, ChromaDB, BM25, RRF and reranking
are unchanged.
