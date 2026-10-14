import json
import sys
import tempfile
import argparse
import time
import re
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.main import app
from app.storage.database import Base, get_db
from app.generation.grounding import ABSTENTION
from app.generation.grounded_answer import has_negation
from app.generation.grounding import terms
from app.ingestion.loaders import load_document
from app.storage.models import Chunk
from app.retrieval.evaluation import retrieval_scores
from app.ingestion.pipeline import ALLOWED_TYPES
from app.generation.model_gateway import warm_ollama_model


# Keep labeled regression data stable when users replace demo/sample documents.
SAMPLE_DOCS = ROOT / "tests" / "fixtures" / "evaluation_corpus"
QUESTIONS = ROOT / "tests" / "evaluation_questions.json"
GOLD = json.loads((ROOT / "tests" / "retrieval_gold.json").read_text(encoding="utf-8"))


def main() -> None:
    global SAMPLE_DOCS, QUESTIONS, GOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-selector", action="store_true", help="Test the pipeline with a deterministic selector, without measuring LLM quality")
    parser.add_argument("--model", choices=["qwen2.5:1.5b", "qwen2.5:3b"], default="qwen2.5:1.5b")
    parser.add_argument("--output", type=Path, help="Save the JSON evaluation report")
    parser.add_argument("--documents", type=Path, default=SAMPLE_DOCS)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--gold", type=Path, default=ROOT / "tests/retrieval_gold.json")
    args = parser.parse_args()
    SAMPLE_DOCS, QUESTIONS = args.documents, args.questions
    GOLD = json.loads(args.gold.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        client = build_client(temp_path)
        try:
            token = signup(client)
            uploaded = upload_sample_documents(client, token)
            questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
            if not args.offline_selector:
                started = time.perf_counter()
                warmed = warm_ollama_model(args.model)
                args.model_warmup_ms = round((time.perf_counter() - started) * 1000)
                args.model_warmed = warmed
            def select_passages(question, evidence, model_name, response_mode, synthesis_feedback=None):
                from tests.fakes import grounded_generator
                return grounded_generator(question, evidence, model_name, response_mode, synthesis_feedback)

            selector = patch("app.api.routes.chat.generate_answer", select_passages) if args.offline_selector else nullcontext()
            readiness = patch("app.api.routes.chat.ensure_ollama_model", lambda model=None: args.model) if args.offline_selector else nullcontext()
            with selector, readiness:
                results = [evaluate_question(client, token, question, uploaded, args.model) for question in questions]
            print_report(results, args)
        finally:
            client.close()
            app.dependency_overrides.clear()


def build_client(temp_path: Path) -> TestClient:
    settings = get_settings()
    settings.upload_dir = temp_path / "uploads"
    settings.chroma_dir = temp_path / "chroma"
    settings.jwt_secret_key = "evaluation-secret-key-with-at-least-32-bytes"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def signup(client: TestClient) -> str:
    response = client.post(
        "/auth/signup",
        json={
            "full_name": "Evaluation User",
            "email": "evaluation@example.com",
            "password": "strong-password-123",
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def upload_sample_documents(client: TestClient, token: str) -> dict[str, str]:
    uploaded = {}
    for path in sorted(SAMPLE_DOCS.iterdir()):
        if path.suffix.lower() not in ALLOWED_TYPES:
            continue
        response = client.post(
            "/documents/upload",
            headers=auth(token),
            files={"file": (path.name, path.read_bytes())},
        )
        response.raise_for_status()
        uploaded[path.name] = response.json()["document_id"]
    return uploaded


def answer_polarity_matches_gold(data, gold_text):
    if not gold_text:
        return False
    for citation in data["citations"]:
        if gold_text not in citation["excerpt"]:
            continue
        link = f"[{citation['citation_id']}](#{citation['anchor']})"
        for line in data["answer"].splitlines():
            if link in line and has_negation(line) == has_negation(gold_text):
                return True
    return False


def answer_content_matches(data, item, gold_text):
    """Content/completeness regression proxy, separate from citation provenance."""
    answer = re.sub(r"\[\d+\]\([^)]*\)", "", data["answer"])
    numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    expected_numbers = item.get("expected_numbers")
    if expected_numbers is None:
        quantitative = re.search(r"\b(?:release|version|limit|size|fee|rate|period|duration|how many|how much|how long|base path)\b", item["question"], re.I)
        expected_numbers = (set(re.findall(r"\d+(?:\.\d+)?", gold_text or "")) - set(re.findall(r"\d+(?:\.\d+)?", item["question"]))) if quantitative else set()
    keywords = item.get("expected_keywords", [])
    items = item.get("expected_items", [])
    content_terms = terms(answer)
    gold_terms = terms(gold_text or "")
    return (len(answer.split()) >= 4 and bool(data["citations"])
            and all(str(value) in numbers for value in expected_numbers)
            and all(word.casefold() in answer.casefold() for word in keywords)
            and all(all(word.casefold() in answer.casefold() for word in group) for group in items)
            and (not gold_terms or len(content_terms & gold_terms) / len(gold_terms) >= 0.2)
            and not data["retrieval_metadata"].get("completeness", {}).get("missing_supported_parts"))


def evaluate_question(client: TestClient, token: str, item: dict, uploaded: dict[str, str], model: str) -> dict:
    expected_source = item.get("expected_source")
    gold_text = GOLD.get(item["question"])
    db_provider = app.dependency_overrides[get_db]()
    db = next(db_provider)
    try:
        relevant_ids = {chunk.chunk_id for chunk in db.query(Chunk).all()
                        if chunk.source_filename == expected_source and gold_text and gold_text in chunk.text}
    finally:
        db_provider.close()
    if expected_source and not relevant_ids:
        raise ValueError(f"Missing gold chunk for: {item['question']}")
    response = client.post(
        "/chat/query",
        headers=auth(token),
        json={"question": item["question"], "model_name": model},
    )
    response.raise_for_status()
    data = response.json()
    metadata = data["retrieval_metadata"]
    k = metadata["top_k"]
    final_ids = [row["chunk_id"] for row in metadata["retrieved_chunks"]]
    diagnostics = next((call["diagnostics"] for call in metadata["tool_calls"]
                        if call["tool"] == "hybrid_search"), {})
    retrieval = {
        "semantic": retrieval_scores(diagnostics.get("semantic_chunk_ids", []), relevant_ids, metadata.get("vector_top_k", k)),
        "keyword": retrieval_scores(diagnostics.get("keyword_chunk_ids", []), relevant_ids, metadata.get("bm25_top_k", k)),
        "hybrid_reranked": retrieval_scores(final_ids, relevant_ids, k),
    }
    citation_sources = {citation["source_filename"] for citation in data["citations"]}
    source_ok = expected_source is None or expected_source in citation_sources
    answerability_ok = data["answerable"] == item["expected_answerability"]
    citation_checks = []
    for citation in data["citations"]:
        source = SAMPLE_DOCS / citation["source_filename"]
        pages = load_document(source, source.suffix) if source.is_file() else []
        citation_checks.append(any(citation["excerpt"] in page.text for page in pages))
    precision = sum(citation_checks) / len(citation_checks) if citation_checks else None
    grounding = data["retrieval_metadata"]["grounding"]
    refused = data["answer"] == ABSTENTION and not data["citations"]
    grounded_ok = (refused if item["expected_answerability"] == "unanswerable" else
                   bool(citation_checks) and all(citation_checks) and grounding.get("citation_validity") == 1.0)
    answer_support_ok = refused if expected_source is None else bool(gold_text) and any(
        gold_text in citation["excerpt"] for citation in data["citations"])
    polarity_ok = refused if expected_source is None else answer_polarity_matches_gold(data, gold_text)
    content_ok = refused if expected_source is None else answer_content_matches(data, item, gold_text)
    return {
        "question": item["question"],
        "answer": data["answer"],
        "citations": data["citations"],
        "expected_answerability": item["expected_answerability"],
        "actual_answerability": data["answerable"],
        "expected_source": expected_source,
        "citation_sources": sorted(citation_sources),
        "answerability_ok": answerability_ok,
        "source_ok": source_ok,
        "answer_support_ok": answer_support_ok,
        "answer_polarity_ok": polarity_ok,
        "answer_content_ok": content_ok,
        "completeness": metadata.get("completeness", {}),
        "synthesis_retries": metadata.get("retry_count", 0),
        "answer_relevancy": float(source_ok and answer_support_ok) if expected_source else None,
        "abstention_correct": (item["expected_answerability"] == "unanswerable") == refused,
        "faithfulness": grounding.get("faithfulness"),
        "faithfulness_method": grounding["method"],
        "citation_groundedness": grounding.get("citation_validity"),
        "citation_precision": precision,
        "one_line_refusal": refused,
        "generation_model": data["model_used"],
        "retrieval": retrieval,
        "retrieval_latency_ms": metadata["retrieval_latency_ms"],
        "generation_latency_ms": metadata["model_latency_ms"],
        "total_latency_ms": data["latency_ms"],
        "under_10_seconds": data["latency_ms"] < 10000,
        "passed": answerability_ok and source_ok and grounded_ok and answer_support_ok and polarity_ok and content_ok,
    }


def print_report(results: list[dict], args) -> None:
    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    retrieval_summary = {}
    for method in ("semantic", "keyword", "hybrid_reranked"):
        rows = [r["retrieval"][method] for r in results if r["retrieval"][method]["relevant_total"]]
        retrieval_summary[method] = {
            "mean_precision_at_k": sum(row["precision_at_k"] for row in rows) / len(rows) if rows else None,
            "mean_recall_at_k": sum(row["recall_at_k"] for row in rows) / len(rows) if rows else None,
        }
    relevancy = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]
    report = {"passed": passed, "total": total,
                      "answer_relevancy_method": "labeled_supporting_fact_proxy",
                      "mean_answer_relevancy": sum(relevancy) / len(relevancy) if relevancy else None,
                      "abstention_accuracy": sum(r["abstention_correct"] for r in results) / total if total else None,
                      "mean_lexical_faithfulness_proxy": sum(scores) / len(scores) if scores else None,
                      "faithfulness_limitation": "Lexical overlap and exact citation validity are proxies, not independent semantic entailment evaluation.",
                      "mean_citation_groundedness": sum(r["citation_groundedness"] for r in results if r["citation_groundedness"] is not None) / len(scores) if scores else None,
                      "embedding_mode": "langchain_huggingface",
                      "selector_mode": "offline_test" if args.offline_selector else "live_llm",
                      "model_warmup_ms": getattr(args, "model_warmup_ms", None),
                      "model_warmed": getattr(args, "model_warmed", None),
                      "retrieval_summary": retrieval_summary,
                      "under_10_seconds_count": sum(r["under_10_seconds"] for r in results),
                      "max_total_latency_ms": max(r["total_latency_ms"] for r in results),
                      "scored_answers": len(scores), "results": results}
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
