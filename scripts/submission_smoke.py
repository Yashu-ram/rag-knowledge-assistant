"""Real external-upload smoke test, isolated from the application's saved data."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf
from scripts.evaluate import build_client, signup, auth
from app.generation.grounding import ABSTENTION
from app.main import app
from app.retrieval.service import shared_embedding_model
from app.core.config import get_settings


def main():
    report = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        client = build_client(Path(folder))
        try:
            headers = auth(signup(client))
            vectors = shared_embedding_model(get_settings().embedding_model).embed(["Verify real MiniLM embeddings."])
            assert len(vectors[0]) == 384
            report["embedding_model"] = get_settings().embedding_model
            report["embedding_dimensions"] = len(vectors[0])
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((45, 65), "The Orion press warranty lasts 31 months. Delivery takes six days.")
                pdf_bytes = pdf.tobytes()
            uploaded = []
            for name, data in [
                ("external-orion.pdf", pdf_bytes),
                ("external-returns.txt", b"Unused equipment can be returned within thirty days."),
                ("external-service.md", b"# Service policy\nThe support desk is open Monday through Friday.\n\n# Repairs\nRepairs require a service ticket.\n\n# Delivery\nStandard delivery takes six days."),
            ]:
                response = client.post("/documents/upload", headers=headers, files={"file": (name, data)})
                assert response.status_code == 200, response.text
                doc = response.json()
                assert doc["status"] == "indexed" and doc["analysis"]["chunk_count"]
                uploaded.append(doc)
            report["uploads"] = [{"filename": d["filename"], "chunks": d["analysis"]["chunk_count"],
                                  "status": d["status"]} for d in uploaded]
            report["checks"] = []
            for question, doc, expected in [
                ("What is the Orion press warranty period?", uploaded[0], "31"),
                ("What is this document about?", uploaded[2], None),
                ("What is today's IPL score?", uploaded[0], ABSTENTION),
            ]:
                response = client.post("/chat/query", headers=headers, json={
                    "question": question, "document_ids": [doc["document_id"]], "model_name": "qwen2.5:1.5b",
                })
                assert response.status_code == 200, response.text
                result = response.json()
                if expected == ABSTENTION:
                    assert result["answer"] == ABSTENTION and not result["citations"], result
                    assert result["retrieval_metadata"]["model_latency_ms"] == 0
                else:
                    assert result["citations"] and result["answerable"] == "answerable", result
                    if expected:
                        assert expected in result["answer"], result
                    else:
                        assert 3 <= len(result["answer"].splitlines()) <= 5, result
                report["checks"].append({"question": question, "answer": result["answer"],
                    "citations": result["citations"], "latency_ms": result["latency_ms"],
                    "grounding": result["retrieval_metadata"]["grounding"]})
            assert client.get("/health").status_code == 200
            assert len(client.get("/documents", headers=headers).json()) == 3
            assert client.delete("/documents/" + uploaded[0]["document_id"], headers=headers).status_code == 200
            report["passed"] = True
        finally:
            client.close()
            app.dependency_overrides.clear()
    destination = ROOT / "docs/final-smoke-results.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
