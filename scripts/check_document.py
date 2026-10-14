"""Check labeled questions against an indexed document without saving messages."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_pipeline import RAGPipeline
from app.core.config import get_settings
from app.generation.model_gateway import generate_answer, warm_ollama_model
from app.retrieval.service import shared_embedding_model
from app.storage.database import SessionLocal
from app.storage.models import Document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen2.5:1.5b", "qwen2.5:3b"], default="qwen2.5:1.5b")
    args = parser.parse_args()
    settings = get_settings()
    startup = time.perf_counter()
    vectors = shared_embedding_model(settings.embedding_model).embed(["Warm up retrieval."])
    startup_ms = round((time.perf_counter() - startup) * 1000)
    warm_started = time.perf_counter()
    model_warmed = warm_ollama_model(args.model)
    warmup_ms = round((time.perf_counter() - warm_started) * 1000)
    results = []
    with SessionLocal() as db:
        document = db.get(Document, args.document_id)
        if not document or document.status != "indexed":
            raise ValueError("Document is missing or not indexed")
        for item in json.loads(args.questions.read_text(encoding="utf-8")):
            started = time.perf_counter()
            question = item["question"]
            result = RAGPipeline(generate_answer).run(db, document.user_id, question,
                [document.document_id], args.model, "quick")
            answer, citations, metrics = result.answer, result.citations, result.metadata["grounding"]
            retrieval_ms = result.metadata["retrieval_latency_ms"]
            generation_ms, model = result.metadata["model_latency_ms"], result.model_used
            total_ms = round((time.perf_counter() - started) * 1000)
            joined = " ".join(" ".join(c.excerpt.split()) for c in citations)
            accepted = item.get("accepted_excerpts", [item["expected_excerpt"]])
            pages = item.get("accepted_pages", [item["expected_page"]])
            passed = (any(value in joined for value in accepted)
                      and any(c.page_number in pages for c in citations))
            results.append({"question": question, "answer": answer,
                            "pages": [c.page_number for c in citations], "model": model,
                            "grounding": metrics, "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms, "total_ms": total_ms,
                            "under_10_seconds": total_ms < 10000, "passed": passed})
    print(json.dumps({"embedding_dimensions": len(vectors[0]), "startup_ms": startup_ms,
                      "model_warmed": model_warmed, "model_warmup_ms": warmup_ms,
                      "passed": sum(r["passed"] for r in results), "total": len(results),
                      "results": results}, indent=2, ensure_ascii=True))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
