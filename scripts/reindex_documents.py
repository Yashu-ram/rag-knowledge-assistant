"""Upgrade stored source files to the current index, preserving document IDs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app
from app.core.config import get_settings
from app.ingestion.pipeline import index_document
from app.ingestion.profile import INDEX_VERSION
from app.storage.database import SessionLocal
from app.storage.models import Document


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--document-id")
    parser.add_argument("--force", action="store_true", help="Rebuild even current indexes, for example after changing collections")
    args = parser.parse_args()
    settings = get_settings()
    failures = 0
    with SessionLocal() as db:
        documents = db.query(Document).all() if args.all else [db.get(Document, args.document_id)]
        for document in documents:
            if document is None:
                raise ValueError("Document not found")
            analysis = json.loads(document.index_info.analysis_json) if document.index_info else {}
            if not args.force and analysis.get("index_version") == INDEX_VERSION and document.status == "indexed":
                print(f"Already current: {document.filename}", flush=True)
                continue
            path = settings.upload_dir / document.user_id / f"{document.document_id}{document.file_type}"
            try:
                index_document(db, document, path)
                analysis = json.loads(document.index_info.analysis_json)
                print(json.dumps({"filename": document.filename, "status": document.status,
                                  "chunks": analysis["chunk_count"], "target_tokens": analysis["target_tokens"],
                                  "overlap_tokens": analysis["overlap_tokens"], "warnings": len(analysis["warnings"])}), flush=True)
            except Exception as exc:
                failures += 1
                print(f"Reindex failed for {document.filename}: {exc}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
