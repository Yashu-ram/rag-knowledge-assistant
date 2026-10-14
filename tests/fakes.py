"""Explicit test doubles, never imported by application code."""
import hashlib
import math
import json
from types import SimpleNamespace
from langchain_core.embeddings import Embeddings
from app.retrieval.keyword_search import tokenize


class TestEmbeddings(Embeddings):
    def _load_model(self):
        return SimpleNamespace(max_seq_length=256, tokenizer=SimpleNamespace(
            encode=lambda text, **kwargs: tokenize(text)))

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            for word in tokenize(text):
                digest = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % 384] += 1
            norm = math.sqrt(sum(x * x for x in vector)) or 1
            vectors.append([x / norm for x in vector])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed(self, texts):
        return self.embed_documents(texts)


def grounded_generator(question, evidence, model_name, response_mode, synthesis_feedback=None):
    """Deterministic source-only answer double for API wiring tests, not LLM quality."""
    from app.generation.grounded_answer import context_records
    from app.generation.grounding import terms
    from app.generation.overview import is_overview_question
    from app.ingestion.structure import source_units

    candidates = []
    for record in context_records(evidence):
        units = source_units(record["text"], pdf=record["page"] is not None)
        for unit in units:
            quote = record["text"][unit.start:unit.end].strip()
            if quote and not quote.startswith("#") and not quote.endswith("?"):
                candidates.append({"text": quote, "quote": quote, "source_id": record["source_id"]})
    if is_overview_question(question):
        unique = {}
        for claim in candidates:
            unique.setdefault(claim["source_id"], claim)
        chosen = list(unique.values())[:5]
        for claim in candidates:
            if len(chosen) >= min(3, len(candidates)):
                break
            if claim not in chosen:
                chosen.append(claim)
    else:
        from app.generation.synthesis import answer_plan, anchors
        plan = answer_plan(question)
        chosen = []
        if plan.parts:
            for part in plan.parts:
                matches = [c for c in candidates if anchors(part) <= terms(c["quote"])]
                if matches:
                    claim = max(matches, key=lambda c: len(terms(question) & terms(c["quote"])))
                    if claim not in chosen:
                        chosen.append(claim)
        else:
            chosen = sorted(candidates, key=lambda c: len(terms(question) & terms(c["quote"])), reverse=True)[:1]
    return json.dumps({"claims": chosen}), "test-grounded-generator", 0
