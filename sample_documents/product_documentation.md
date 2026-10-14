# Product Documentation — DocuQuery API

> **Purpose:** This is sample product documentation for testing a retrieval-augmented generation (RAG) chatbot.
>
> **Important:** DocuQuery is a fictional product created for demonstration and evaluation. The limits and requirements below are illustrative, not specifications for a real service.

## 1. Product Overview

DocuQuery API lets applications upload documents, extract their text, and ask questions about the uploaded content. Responses may include source references so a client can inspect the passages used to answer a question.

### Core features

- Upload supported documents for processing.
- Extract text from supported file types.
- Split extracted text into searchable chunks.
- Retrieve relevant passages for a user question.
- Generate an answer grounded in retrieved passages.
- Return source references with answers when relevant evidence is found.
- Return an explicit “insufficient information” response when the documents do not contain enough evidence.
- Delete uploaded documents and their indexed content.
- Check document-processing status.

## 2. Supported File Formats

The API accepts the following file formats:

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Text-based PDFs are supported. Scanned PDFs may require OCR. |
| Markdown | `.md` | Headings, paragraphs, lists, and tables are supported. |
| Plain text | `.txt` | UTF-8 encoding is recommended. |
| Microsoft Word | `.docx` | Text and basic document structure are extracted. |
| CSV | `.csv` | Rows are converted into searchable text records. |
| JSON | `.json` | Valid JSON objects and arrays are supported. |

The API does **not** accept `.doc`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, image files such as `.png` or `.jpg`, audio files, or video files in this sample specification.

### Example question

**Q: Which file formats does the API support?**

**A:** PDF, Markdown (`.md`), plain text (`.txt`), Microsoft Word (`.docx`), CSV (`.csv`), and JSON (`.json`). Scanned PDFs may require OCR.

## 3. Upload Requirements and Limits

| Requirement | Limit |
|---|---|
| Maximum file size | 20 MB per file |
| Maximum files per upload request | 10 |
| Maximum extracted text | 500,000 characters per file |
| Maximum document filename length | 255 characters |
| Supported text encoding | UTF-8 recommended |
| Empty files | Rejected |
| Password-protected files | Rejected unless supplied in an accessible, unencrypted form |

A file with an unsupported extension, invalid content, or a size above the limit is rejected with a validation error. A successful upload does not necessarily mean processing has finished; clients should check the document status.

## 4. API Authentication

Requests must include a bearer token in the HTTP `Authorization` header.

```http
Authorization: Bearer YOUR_API_TOKEN
```

Do not place API tokens in query parameters, public client-side code, or application logs. Requests without a valid token return `401 Unauthorized`.

## 5. API Endpoints

The base URL in this sample specification is:

```text
https://api.docuquery.example/v1
```

The `.example` domain is reserved for documentation and is not a live service.

### 5.1 Upload a document

`POST /documents`

Uploads a document for processing. The request uses `multipart/form-data` with a `file` field.

**Successful response — `202 Accepted`:**

```json
{
  "document_id": "doc_12345",
  "filename": "product-guide.pdf",
  "status": "processing"
}
```

The response includes a document ID that is used to check status or delete the document.

### 5.2 Check processing status

`GET /documents/{document_id}`

Returns the current status of a document.

Possible status values:

- `processing` — the file is being parsed and indexed.
- `ready` — the document can be searched.
- `failed` — processing did not complete successfully.

A question request that references a document still being processed may return `409 Conflict`.

### 5.3 Ask a question

`POST /questions`

Example request:

```json
{
  "question": "Which file formats does the API support?",
  "document_ids": ["doc_12345"]
}
```

Example response:

```json
{
  "answer": "The API supports PDF, Markdown, TXT, DOCX, CSV, and JSON.",
  "insufficient_information": false,
  "sources": [
    {
      "document_id": "doc_12345",
      "filename": "product_documentation.md",
      "section": "Supported File Formats",
      "excerpt": "The API accepts PDF, Markdown, plain text, Microsoft Word, CSV, and JSON."
    }
  ]
}
```

The `sources` array contains the document and passage information used to support the answer. The exact number of sources may vary by query.

### 5.4 Delete a document

`DELETE /documents/{document_id}`

Deletes the document and its searchable index entries.

**Successful response:** `204 No Content`.

After deletion, the document should no longer be available for retrieval. Deletion may not be reversible.

## 6. Question Answering Behavior

- Answers should be based on retrieved document content.
- Source references should be returned when evidence is available.
- The API must not invent product specifications that are absent from the indexed documents.
- If the documents do not contain enough information, the response should say that the information is not available in the provided documentation.
- If the question is ambiguous, the API may ask the user to clarify.
- If no relevant passage is found, the response should set `insufficient_information` to `true` and return an empty `sources` array.

Example response when the answer is unavailable:

```json
{
  "answer": "The provided documentation does not specify the data-center location.",
  "insufficient_information": true,
  "sources": []
}
```

## 7. Rate Limits and API Limits

The illustrative default rate limit is **60 requests per minute per API token**.

| Limit | Default |
|---|---|
| Requests per minute | 60 per API token |
| Concurrent document-processing jobs | 3 per API token |
| Files per upload request | 10 |
| File size | 20 MB per file |
| Question length | 2,000 characters |
| Document IDs per question | 20 |

When a rate limit is exceeded, the API returns `429 Too Many Requests`. The client should wait before retrying. If the response includes a `Retry-After` header, the client should follow it.

These limits are illustrative and may be adjusted by an administrator in a real implementation.

## 8. Error Responses

Errors use a JSON response body.

```json
{
  "error": {
    "code": "UNSUPPORTED_FILE_TYPE",
    "message": "The uploaded file type is not supported."
  }
}
```

| HTTP status | Error code example | Meaning |
|---|---|---|
| `400 Bad Request` | `INVALID_REQUEST` | Required fields are missing or invalid. |
| `401 Unauthorized` | `INVALID_TOKEN` | Authentication is missing or invalid. |
| `404 Not Found` | `DOCUMENT_NOT_FOUND` | The document ID does not exist. |
| `409 Conflict` | `DOCUMENT_NOT_READY` | The document is still processing. |
| `413 Payload Too Large` | `FILE_TOO_LARGE` | The upload exceeds the file-size limit. |
| `415 Unsupported Media Type` | `UNSUPPORTED_FILE_TYPE` | The file format is not supported. |
| `422 Unprocessable Entity` | `EMPTY_DOCUMENT` | The document contains no usable content. |
| `429 Too Many Requests` | `RATE_LIMIT_EXCEEDED` | The request rate limit has been exceeded. |
| `500 Internal Server Error` | `INTERNAL_ERROR` | An unexpected server error occurred. |

## 9. Technical Requirements

### Client requirements

- Send requests over HTTPS.
- Include a valid bearer token.
- Send upload requests as `multipart/form-data`.
- Send question requests as valid JSON with the `Content-Type: application/json` header.
- Handle non-2xx HTTP status codes.
- Use document IDs returned by the upload endpoint.
- Check that a document is `ready` before asking questions about it.
- Implement retry handling for transient failures and rate limits.

### Processing requirements

- Validate file type and size before processing.
- Extract text from supported files.
- Split extracted text into chunks suitable for retrieval.
- Preserve document IDs and useful section or page metadata.
- Retrieve relevant chunks before generating an answer.
- Return citations or source references that point to retrieved evidence.
- Avoid unsupported claims when the retrieved evidence is insufficient.
- Log operational errors without exposing API tokens or other secrets.

## 10. Security and Privacy

- Treat uploaded files as potentially sensitive.
- Transmit files and API requests over HTTPS.
- Store API tokens securely and never commit them to source control.
- Restrict document access to authorized users.
- Do not expose one user's documents to another user.
- Redact secrets from logs where practical.
- Apply the product's retention and deletion policy to uploaded files and extracted text.

This sample specification does not define a legal retention period or guarantee any particular compliance certification.

## 11. Operational Troubleshooting

| Symptom | Likely cause | Suggested action |
|---|---|---|
| Upload returns `415` | Unsupported file type | Convert the file to a supported format. |
| Upload returns `413` | File exceeds 20 MB | Reduce the file size or split the document. |
| Status remains `processing` | Parsing or indexing is still running | Check the status again after a short wait. |
| Question returns no sources | No relevant evidence was retrieved | Check that the document is `ready` and that the question matches its content. |
| Question returns `409` | Document is not ready | Wait for processing to finish. |
| Request returns `429` | Rate limit exceeded | Wait and retry, respecting `Retry-After` if provided. |
| Request returns `401` | Token is missing or invalid | Verify the bearer token. |

## 12. Suggested Evaluation Questions

Use these questions to test whether a RAG system retrieves the right passages and cites them accurately.

### Answerable questions

1. Which file formats does the API support?
2. What is the maximum file size?
3. How many files can be uploaded in one request?
4. What authentication method does the API use?
5. Which endpoint is used to ask a question?
6. What does the `ready` status mean?
7. What happens when a request exceeds the rate limit?
8. What HTTP status is returned for an unsupported file type?
9. What should the API do when it cannot find evidence for an answer?
10. What is the maximum question length?

### Unanswerable questions

1. Where are the production servers physically located?
2. What is the monthly subscription price?
3. Who is the company's CEO?
4. Does the product have SOC 2 certification?
5. What is the guaranteed uptime percentage?

The documentation intentionally does not answer these questions. A well-behaved RAG system should state that the provided documentation does not contain the information and should not invent an answer.

## 13. Versioning

- Documentation version: `1.0`
- Status: Sample specification for RAG development and evaluation
- Last updated: `2026-10-13`

Changes to supported formats, endpoint behavior, authentication, limits, or error codes should be reflected in this document and covered by regression tests.
