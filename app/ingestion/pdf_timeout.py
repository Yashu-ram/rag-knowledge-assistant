"""Run native PDF parsing in a lightweight, deadline-bound subprocess."""
import json
import os
import subprocess
import sys
from pathlib import Path

from app.core.errors import RAGServiceError


def load_pdf_with_timeout(path, timeout_seconds):
    from app.ingestion.loaders import PageText

    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        result = subprocess.run([sys.executable, "-m", "app.ingestion.pdf_worker", str(Path(path).resolve())],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for the worker before raising TimeoutExpired.
        raise RAGServiceError("PDF_EXTRACTION_TIMEOUT",
            f"PDF extraction exceeded {timeout_seconds} seconds. Try a smaller text-based PDF.", 504) from exc
    except OSError as exc:
        raise ValueError("PDF extraction worker could not start.") from exc
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("PDF text extraction failed: the parser process stopped unexpectedly.") from exc
    if result.returncode or "error" in payload:
        raise ValueError("PDF text extraction failed: " + payload.get("error", "Parser failed."))
    return [PageText(**page) for page in payload["pages"]]
