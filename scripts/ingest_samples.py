"""Upload demo files through FastAPI into the same account used by Streamlit."""

import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ui.api_client import request_backend


def find_sample_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Sample directory not found: {directory}")
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})


def upload_samples(base_url, token, directory):
    results = []
    for path in find_sample_files(directory):
        try:
            with path.open("rb") as handle:
                with request_backend("POST", base_url, "/documents/upload", token=token,
                                     files={"file": (path.name, handle)}) as response:
                    if response.status_code >= 400:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.json().get('detail', 'Upload failed')}")
                    document = response.json()
            count = document.get("analysis", {}).get("chunk_count", 0)
            print(f"{path.name}\t{document['status']}\t{document['document_id']}\tchunks={count}")
            results.append(document)
        except Exception as exc:
            print(f"{path.name}\tfailed\tdocument_id=unavailable\tchunks=0\t{exc}")
            results.append({"filename": path.name, "status": "failed"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--directory", type=Path, default=ROOT / "sample_documents")
    parser.add_argument("--email", help="Existing account email, same account as Streamlit")
    args = parser.parse_args()
    try:
        files = find_sample_files(args.directory)
        if not files:
            print("No PDF/TXT/MD sample files found.")
            return 1
        token = os.getenv("RAG_ACCESS_TOKEN")
        if not token:
            email = args.email or input("Account email: ").strip()
            password = getpass.getpass("Account password: ")
            with request_backend("POST", args.api_url, "/auth/login", timeout=30,
                                 json={"email": email, "password": password}) as response:
                if response.status_code >= 400:
                    raise RuntimeError(f"Login failed (HTTP {response.status_code}). Sign up in Streamlit first.")
                token = response.json()["access_token"]
        results = upload_samples(args.api_url, token, args.directory)
        return 0 if all(item["status"] == "indexed" for item in results) else 1
    except Exception as exc:
        print(f"Sample ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
