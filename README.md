# HireKiwi RAG Knowledge Assistant

A Python knowledge assistant that accepts external PDF, TXT and Markdown files,
answers from retrieved document content, links claims to supporting passages,
and acknowledges missing information. FastAPI is the API; Streamlit is an optional
interactive frontend. Pretrained models are used; nothing is trained from scratch.

## Assessment Checklist

| Requirement | Implementation / evidence |
| --- | --- |
| PDF/TXT/MD ingestion and chunks | Dynamic parsing and sentence-aware splitting; fresh real uploads verified |
| Semantic retrieval and vector store | Normalized MiniLM embeddings and persistent ChromaDB |
| LLM-grounded answers | Ollama through LangChain; bounded, numbered document context |
| Source names and relevant excerpts | Citations with filename, physical PDF page/section, chunk ID and exact snippet |
| Missing-information handling | One-line abstention; supported partial answers identify missing parts |
| Python API | FastAPI ingestion, document management and question-answering endpoints |
| At least 10 evaluation questions | 23 labeled questions: 21 answerable, 2 unanswerable, including 3 comparison/multi-evidence cases |
| Modular Python architecture | Separate API, ingestion, retrieval, generation, auth, storage and UI modules |
| Environment configuration | `.env.example`; no paid API key needed for local models |
| Automated reliability tests | 231 tests passed in the latest end-to-end verification |
| Docker support | Dockerfile and Compose stack included; validation status below |
| Setup, sample files, results, limitations | This README, `sample_documents/`, `tests/`, and `docs/evaluation-results.json` |

The local functionality is verified, not a prediction of the interviewer's score.
Publishing a repository/archive and submitting its link are still manual steps.

## Quick Start: Python 3.11

Run from the repository root. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```

Set `JWT_SECRET_KEY` in `.env` to the generated value, not the example placeholder.
Never commit/share `.env`. If activation is blocked, use
`Set-ExecutionPolicy -Scope Process RemoteSigned` for that terminal only.
Linux/macOS: use `python3.11 -m venv .venv`, `source .venv/bin/activate`, and
`cp .env.example .env`; the remaining Python commands are the same.

Install Ollama separately, then download the default pretrained LLM:

```powershell
ollama pull qwen2.5:1.5b
# Optional larger model:
ollama pull qwen2.5:3b
```

Ollama must be running. Use `ollama serve` only if its service is not already
running. An occupied port 11434 usually means Ollama is already active.
The first embedding operation downloads `sentence-transformers/all-MiniLM-L6-v2`
from Hugging Face. Internet access and disk space are needed initially; model
weights are external prerequisites, not bundled with the source.

Start the backend in one activated terminal:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another activated terminal:

```powershell
python -m streamlit run app/ui/streamlit_app.py --server.port 8501
```

UI: http://localhost:8501. Swagger: http://localhost:8000/docs.
Health: http://localhost:8000/health. Startup can wait for model warmup/downloads.
Run only one backend/frontend. Stop the old process with Ctrl+C before restarting.

## Evaluator Walkthrough

1. Sign up, then log in to Streamlit.
2. Upload your own text-based PDF/TXT/MD through Documents or the chat attachment control.
3. Wait for extraction, chunking, embedding and Chroma indexing to finish.
4. The document list refreshes automatically. Click **Refresh Documents** for uploads made by another client.
5. Select indexed documents. Processing/failed files cannot be queried.
6. Ask a factual question, then a comparison requiring multiple facts.
7. Open citations and inspect the filename, physical page/section, chunk ID and excerpt.
8. Ask an unrelated question. Expected: "I could not find enough information in the uploaded document to answer this."

Documents persist and are isolated by account. Log back into the same account
after a browser refresh and select its indexed files. Copying a file into a folder
does not ingest it automatically. Streamlit only calls FastAPI; it does not
implement its own RAG pipeline.

Upload bundled demo files into the same existing account:

```powershell
python scripts/ingest_samples.py --email your-account@example.com
```

The helper prompts securely for the password, uploads PDF/TXT/MD recursively,
and prints filename, status, document ID and chunk count. Click **Refresh Documents**
afterwards. Demo documents are fictional/test material, not official policy advice.

## API Examples

Document/chat endpoints require a bearer token. For an existing account:

```powershell
$base = 'http://localhost:8000'
$credential = Get-Credential -Message 'Enter your application email and password'
$loginBody = @{email=$credential.UserName; password=$credential.GetNetworkCredential().Password} | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri "$base/auth/login" -ContentType 'application/json' -Body $loginBody
$headers = @{Authorization="Bearer $($login.access_token)"}

# curl.exe avoids PowerShell's curl alias. Replace the file path.
$uploadJson = curl.exe --fail-with-body -sS -X POST "$base/documents/upload" -H "Authorization: Bearer $($login.access_token)" -F 'file=@C:\path\to\your-document.pdf'
$document = $uploadJson | ConvertFrom-Json
Invoke-RestMethod -Uri "$base/documents" -Headers $headers

$body = @{question='What are the main points?'; document_ids=@($document.document_id); model_name='qwen2.5:1.5b'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/chat/query" -Headers $headers -ContentType 'application/json' -Body $body
```

For a new account use `POST /auth/signup` with `full_name`, `email` and `password`
in Swagger first. Click **Authorize** and enter the returned token. Canonical query
fields are `document_ids` (list) and `model_name`; `document_id` and `model`
aliases also work, with conflicting values rejected.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | API liveness, not proof that a model is installed |
| `GET /models` | Allowed/installed model status |
| `POST /auth/signup`, `/auth/login`, `/auth/logout` | Account/session management |
| `GET /auth/me` | Current user |
| `POST /documents/upload` | Extract, chunk, embed and index one file |
| `GET /documents`, `/documents/{document_id}` | Metadata, status and failure details |
| `DELETE /documents/{document_id}` | Remove document and indexed chunks |
| `POST /documents/{document_id}/reindex` | Reindex a preserved original |
| `POST /chat/query` | Answer, citations, answerability, timings and diagnostics |
| `/conversations` | Saved chats; see Swagger for available methods |

`POST /chat` is a compatibility alias. Upload success means status `indexed`.
Default upload limit: 20 MB. Streamlit upload timeout: 300 seconds.
Optional `/documents/upload?stream=true` returns progress followed by an indexed
result or error; HTTP 200 alone does not confirm streaming-ingestion success.
Invalid files fail clearly; selected processing/failed documents return 409.
Missing models, unavailable Ollama and embedding failures are infrastructure
errors, never disguised as missing document information.

## Architecture And Decisions

```text
Upload -> extract -> sentence-aware chunks -> MiniLM embeddings
       -> SQLite metadata + persistent Chroma vectors
Query  -> MiniLM query vector -> vector top 8 + BM25 top 8
       -> RRF (k=60) -> lightweight rerank -> final context
       -> relevance/intent checks -> Ollama -> citations or abstention
```

- **FastAPI / Streamlit:** the API owns processing; the UI is a client.
- **SQLite/SQLAlchemy:** users, sessions, document/page/chunk metadata and chat history.
- **LangChain:** loader/document interfaces, splitting, HuggingFace embeddings, Chroma retriever and ChatOllama. Application-specific logic remains modular/custom.
- **MiniLM:** fixed `sentence-transformers/all-MiniLM-L6-v2`, normalized 384-dimensional vectors. No production fake/hash fallback.
- **Chunking:** adaptive sentence groups; default target 240 tokenizer tokens, maximum 248, overlap up to 60. This respects MiniLM's 256-token input limit. Oversized units split recursively.
- **Retrieval:** BM25 preserves exact terms/IDs; RRF combines ranks without mixing score scales. Reranking uses lexical coverage and actual cosine scores, not a learned cross-encoder.
- **Distance:** new collections request cosine. Existing normalized L2 indexes remain ranking-compatible; the service recomputes cosine instead of mislabeling distance as similarity.
- **Context:** 4 final candidates for quick mode, 6 for deep/overview mode, bounded to 8,000 context characters. Only supplied context supports citations; full stored pages are not silently sent.
- **Models:** default `qwen2.5:1.5b`; optional `qwen2.5:3b`. Both are pretrained Ollama models; no arbitrary installed models are exposed.
- **Synthesis:** factual sentences, comparisons covering both sides, multi-part claims and 3-5 summary bullets. One correction retry can cover omitted supported parts using identical context, without another retrieval.
- **Grounding:** source IDs, exact excerpts, numeric values and basic polarity/direction checks. Imperfect citations are recovered only from strong supplied-context support. Partial answers identify missing parts.

Environment settings are documented in `.env.example`: upload/extraction limits,
model names, database/index paths, Ollama URL and JWT configuration.
Optional `DEBUG_LLM_OUTPUT=true` logs 150-character model previews; leave it off
for private documents.

## Tests And Evaluation

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/evaluate.py --output docs/evaluation-results.json
```

Automated tests use explicit model doubles and require neither paid API keys nor
Ollama. Coverage includes parsing/chunking, retrieval/fusion, API/auth/isolation,
citations, invalid files, model/embedding failures, summaries, multi-evidence
answers and Streamlit client/document behavior.

Live evaluation uses real MiniLM and Ollama. Its fixed fictional corpus is
`tests/fixtures/evaluation_corpus/`; labels are `tests/evaluation_questions.json`
and `tests/retrieval_gold.json`. Questions search the corpus, not a gold-restricted
document. Temporary databases/indexes leave your saved uploads untouched and do
not populate your account. For another corpus pass matching `--documents`,
`--questions` and `--gold` files. Optional `--offline-selector` uses a test
generator with real embeddings, not live LLM quality. Do not reuse unrelated labels.

### Latest Verified Results

Fresh end-to-end verification on 14 October 2026 (Python 3.11, CPU inference):

| Check | Result |
| --- | --- |
| Automated suite | 231 passed in 61.36 seconds (latest rerun) |
| Fresh full live evaluation | 23/23 passed, without merged/cached answers |
| Mean citation groundedness / validity | 1.0 |
| Abstention accuracy | 1.0 |
| Mean labeled answer-relevancy proxy | 1.0 |
| Mean lexical faithfulness proxy | 0.9774 |
| Final hybrid precision@4 / recall@4 | 0.5714 / 0.9905 |
| Maximum total query latency | 44.338 seconds |

`docs/evaluation-results.json` includes actual answers, excerpts, per-question
checks, retrieval metrics and latency. Fresh real PDF/TXT/MD ingestion also
verified vectors and matching Chroma IDs/metadata. Earlier reports in `docs/`
are historical, not current scores. Precision@k uses configured k even if fewer
chunks exist; recall is undefined without relevant chunks. Fixture results do
not guarantee arbitrary-document accuracy or independent semantic entailment.

## Docker

Prerequisite: Docker Engine running with Linux containers and Docker Compose.
Copy `.env.example` to `.env` and replace `JWT_SECRET_KEY` with a private generated
secret first. The backend/frontend/Ollama stack uses named volumes for SQLite,
uploads, Chroma and model caches. Private local files are excluded from the image.

```powershell
docker compose config --quiet
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:1.5b
# Optional:
docker compose exec ollama ollama pull qwen2.5:3b
docker compose up --build -d backend frontend
docker compose ps
docker compose logs --tail 100 backend
```

Docker UI: http://localhost:18501. Swagger: http://localhost:18000/docs.
These ports avoid the local Python servers on 8501/8000. Containers keep separate
accounts/documents in named volumes, not host data folders. Initial build/model
downloads can take several minutes. No host Ollama installation is required.

Stop with `docker compose down`; named volumes remain. Do not use `down -v`
unless you deliberately want to erase Docker data/models. Container tests:

```powershell
docker compose run --rm --no-deps backend python -m pytest -q -p no:cacheprovider
docker compose exec backend python scripts/evaluate.py --output docs/evaluation-results.json
```

The container evaluation report stays inside that container unless exported.
Validation status: Compose configuration passed with
`docker compose --env-file .env.example config --no-env-resolution --quiet`.
After approval to test containers, Docker Desktop failed to start twice with
Windows socket access errors (`dockerInference` and `sailor-ingest.sock`: "The
file cannot be accessed by the system"). No project image build or container
smoke test could run. This local Docker startup failure must be resolved before
container runtime verification; the Python results above do not establish
Docker runtime correctness. Docker data was not reset or deleted.

## Limitations And Trade-Offs

- OCR is disabled. Scanned PDFs fail clearly: "No extractable text found. This PDF may be scanned and OCR is not enabled." Charts, handwriting and complex tables/reading order are not universally supported.
- Small local models trade quality for CPU/memory cost. Responses can exceed 10 seconds and a retry adds latency. Latest maximum: 44.338 seconds; no sub-10-second guarantee.
- Bounded retrieval/context can omit relevant passages. Summaries sample chunks, not every page. Part detection and faithfulness checks are lexical/numeric proxies, not an entailment model.
- Exact excerpts prove provenance, not every inference. Parsed text can contain extraction mistakes; review citations and warnings for important decisions.
- Local assignment scope: no production HTTPS, rate limiting, background-job queue or full tenant/secret-management hardening. Auth, owner isolation and saved history are included.
- Model downloads need network access initially. Missing models produce clear errors/pull commands rather than silently switching models.

## Submission

Submit a GitHub URL or clean source archive containing `app/`, `scripts/`,
`tests/`, `sample_documents/`, `docs/evaluation-results.json`, `requirements.txt`,
`pytest.ini`, `.env.example`, `.streamlit/config.toml`, `Dockerfile`,
`docker-compose.yml`, `.dockerignore`, `.gitignore` and this README.
Exclude `.venv`, `.env`, private uploads, SQLite/Chroma databases, model caches,
logs and `.streamlit/secrets.toml`. Git ignore rules do not protect a ZIP of the
entire working directory; exclude these paths explicitly.

Live demo/video and hours spent are optional. Report actual hours honestly, not
a fabricated value matching the 4-6 hour target. Confirm the ownership/interview
declaration only when you can truthfully explain the implementation and decisions.
The interviewer decides the score; passing cannot be guaranteed.

