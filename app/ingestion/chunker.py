from dataclasses import dataclass
from hashlib import sha256
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

from app.ingestion.loaders import PageText
from app.ingestion.structure import source_units


@dataclass
class TextChunk:
    chunk_id: str
    page_number: int | None
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    token_count: int = 0
    section: str = ""


def chunk_pages(
    document_id: str,
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 160,
    *,
    target_tokens: int | None = None,
    overlap_tokens: int = 40,
    token_counter=None,
) -> list[TextChunk]:
    if target_tokens is not None:
        if not token_counter or target_tokens < 2 or not 0 <= overlap_tokens < target_tokens:
            raise ValueError("Token chunking requires a counter and valid target/overlap")
        return token_chunks(document_id, pages, target_tokens, overlap_tokens, token_counter)
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[TextChunk] = []
    chunk_index = 0
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = _choose_boundary(text, start, chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                raw_id = f"{document_id}:{chunk_index}:{start}:{end}"
                chunks.append(
                    TextChunk(
                        chunk_id=sha256(raw_id.encode("utf-8")).hexdigest()[:24],
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def token_chunks(document_id, pages, target, overlap, counter):
    output = []
    recursive = RecursiveCharacterTextSplitter(chunk_size=target, chunk_overlap=0,
        length_function=counter, separators=["\n\n", "\n", ". ", " ", ""], strip_whitespace=True)
    for page in pages:
        units = source_units(page.text, pdf=page.page_number is not None)
        spans = []
        for unit in units:
            if counter(page.text[unit.start:unit.end]) <= target:
                spans.append((unit.start, unit.end))
            else:
                cursor = unit.start
                for part in recursive.split_text(page.text[unit.start:unit.end]):
                    start = page.text.find(part, cursor, unit.end)
                    if start < 0:
                        raise ValueError("Cannot recover recursive split source offsets")
                    spans.append((start, start + len(part)))
                    cursor = start + len(part)
        index = 0
        while index < len(spans):
            start, end = spans[index]
            stop = index + 1
            while stop < len(spans) and counter(page.text[start:spans[stop][1]]) <= target:
                end = spans[stop][1]
                stop += 1
            raw = page.text[start:end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start, end = start + leading, start + trailing
            text = page.text[start:end]
            raw_id = f"{document_id}:token-v5:{page.page_number}:{len(output)}:{start}:{end}"
            output.append(TextChunk(sha256(raw_id.encode()).hexdigest()[:24], page.page_number,
                                    len(output), start, end, text, counter(text)))
            next_index = stop
            while next_index > index + 1 and counter(page.text[spans[next_index - 1][0]:end]) <= overlap:
                next_index -= 1
            index = next_index
    return output


class SentenceAwareSplitter(TextSplitter):
    """Token-bounded sentence groups; recursive splitting only for oversized units."""

    def __init__(self, target_tokens, overlap_tokens, token_counter):
        super().__init__(chunk_size=target_tokens, chunk_overlap=overlap_tokens, length_function=token_counter)

    def split_text(self, text):
        return [d.page_content for d in self.split_documents([Document(page_content=text)])]

    def split_documents(self, documents):
        result = []
        for document in documents:
            metadata = document.metadata
            chunks = token_chunks(metadata.get("document_id", "text"),
                                  [PageText(document.page_content, metadata.get("page_number"))],
                                  self._chunk_size, self._chunk_overlap, self._length_function)
            units = source_units(document.page_content, pdf=metadata.get("page_number") is not None)
            for chunk in chunks:
                section = next((u.section for u in units
                                if u.section and u.start < chunk.end_offset and u.end > chunk.start_offset), "")
                result.append(Document(page_content=chunk.text, metadata={**metadata,
                    "section": section,
                    "chunk_id": chunk.chunk_id, "chunk_index": len(result),
                    "start_offset": chunk.start_offset, "end_offset": chunk.end_offset,
                    "token_count": chunk.token_count}))
        return result


def _choose_boundary(text: str, start: int, chunk_size: int) -> int:
    hard_end = min(start + chunk_size, len(text))
    if hard_end == len(text):
        return hard_end
    window_start = max(start + int(chunk_size * 0.6), start)
    candidates = [text.rfind(mark, window_start, hard_end) for mark in ["\n\n", ". ", "\n", " "]]
    boundary = max(candidates)
    if boundary <= start:
        return hard_end
    return boundary + 1

