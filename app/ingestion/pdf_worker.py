"""PDF worker entry point: deliberately avoids importing API/LLM/embedding modules."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from contextlib import redirect_stdout

from app.ingestion.loaders import load_pdf


def main():
    try:
        with redirect_stdout(sys.stderr):
            pages = load_pdf(Path(sys.argv[1]))
        output = {"pages": [asdict(page) for page in pages]}
        status = 0
    except Exception as exc:
        output = {"error": str(exc)}
        status = 1
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
