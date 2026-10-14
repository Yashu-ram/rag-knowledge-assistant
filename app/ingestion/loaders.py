import re
import zipfile
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass
class PageText:
    text: str
    page_number: int | None
    method: str = "text"
    warnings: list[str] = field(default_factory=list)
    table_count: int = 0
    label: str | None = None


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(path: Path, file_type: str) -> list[PageText]:
    suffix = file_type.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".text", ".md", ".markdown"}:
        text = read_text(path)
        cleaned = normalize_text(text)
        if not cleaned:
            raise ValueError("Document contains no extractable text")
        return [PageText(text=cleaned, page_number=None)]
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(read_text(path))))
        return [PageText(render_rows(rows), None, method="csv", table_count=1)]
    if suffix == ".json":
        data = json.loads(read_text(path))
        records = []
        def visit(value, key="root"):
            if isinstance(value, dict):
                for name, child in value.items():
                    visit(child, f"{key}.{name}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{key}[{index}]")
            else:
                records.append(f"ROW {key}: {json.dumps(value, ensure_ascii=False)}")
        visit(data)
        if not records:
            raise ValueError("JSON contains no data values")
        return [PageText("\n\n".join(records), None, method="json")]
    raise ValueError("Unsupported file type")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    encoding = "utf-32" if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")) else "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("Text must be UTF-8, or UTF-16/32 with a byte-order mark") from exc
    if "\x00" in text:
        raise ValueError("File contains binary data rather than document text")
    return text


def render_rows(rows: list[list[str | None]]) -> str:
    if len(rows) < 2:
        raise ValueError("Table has no data rows")
    headers = [" ".join((cell or "").split()) for cell in rows[0]]
    if not all(headers) or len(set(headers)) != len(headers):
        raise ValueError("Table column labels are missing or ambiguous")
    output = []
    for index, row in enumerate(rows[1:], 1):
        if len(row) != len(headers):
            raise ValueError("Table row does not match its column labels")
        if any(cell and str(cell).strip() for cell in row):
            values = {key: " ".join(str(value or "").split()) for key, value in zip(headers, row)}
            output.append(f"ROW {index}: {json.dumps(values, ensure_ascii=False)}")
    if not output:
        raise ValueError("Table contains no data values")
    return "\n\n".join(output)


def load_docx(path: Path) -> list[PageText]:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.getinfo("word/document.xml").file_size > 50_000_000:
                raise ValueError("DOCX XML exceeds the extraction size limit")
            xml = archive.read("word/document.xml")
    except Exception as exc:
        raise ValueError("DOCX contains no extractable text") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValueError("DOCX contains no extractable text") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    warnings = []
    table_count = 0
    body = root.find("w:body", namespace)
    if body is None:
        raise ValueError("DOCX has no document body")
    for element in body:
        if element.tag.endswith("}tbl"):
            rows = [["".join(n.text or "" for n in cell.findall(".//w:t", namespace))
                     for cell in row.findall("w:tc", namespace)] for row in element.findall("w:tr", namespace)]
            try:
                paragraphs.append(render_rows(rows))
                table_count += 1
            except ValueError as exc:
                warnings.append(f"Skipped ambiguous Word table: {exc}")
        elif element.tag.endswith("}p"):
            text = "".join(node.text or "" for node in element.findall(".//w:t", namespace))
            if text.strip():
                paragraphs.append(text)
    cleaned = normalize_text("\n\n".join(paragraphs))
    if not cleaned:
        raise ValueError("DOCX contains no extractable text")
    return [PageText(text=cleaned, page_number=None, method="docx", warnings=warnings, table_count=table_count)]


def load_pdf(path: Path) -> list[PageText]:
    pages = load_pdf_with_pymupdf(path)
    if pages:
        if any(page.text.strip() for page in pages):
            return pages
        if any("not configured" in warning for page in pages for warning in page.warnings):
            raise ValueError("No extractable text found. This PDF may be scanned and OCR is not enabled.")
        if any("OCR" in warning for page in pages for warning in page.warnings):
            raise ValueError("No extractable text found. OCR failed to extract readable text from this PDF.")
        raise ValueError("PDF contains no extractable text")
    pages = load_pdf_with_pypdf(path)
    if pages:
        return pages
    raise ValueError("PDF contains no extractable text")


def load_pdf_with_pymupdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    try:
        import pymupdf
    except ImportError:
        return []
    try:
        with pymupdf.open(path) as doc:
            if not doc.is_pdf:
                raise ValueError("File content is not a PDF")
            if doc.needs_pass:
                raise ValueError("Password-protected PDF must be unlocked before indexing")
            for index, page in enumerate(doc, start=1):
                warnings = []
                pieces = []
                table_regions = []
                tables = []
                previous_table = None
                try:
                    tables = list(page.find_tables().tables)
                except Exception as exc:
                    warnings.append(f"Table detection failed: {type(exc).__name__}")
                for table in tables:
                    region = pymupdf.Rect(table.bbox)
                    table_regions.append(region)
                    try:
                        from app.ingestion.tables import pdf_table
                        text, previous_table, table_warnings = pdf_table(table, previous_table)
                        warnings.extend(table_warnings)
                        if text:
                            pieces.append((*region, text, "table"))
                    except ValueError as exc:
                        warnings.append(f"Skipped ambiguous table: {exc}")
                for block in page.get_text("blocks", sort=False):
                    if block[6] != 0:
                        continue
                    region = pymupdf.Rect(block[:4])
                    if any((region & table_region).get_area() > region.get_area() * 0.5 for table_region in table_regions):
                        continue
                    text = normalize_text(block[4])
                    if text:
                        pieces.append((*region, text, "text"))
                from app.ingestion.layout import reading_order
                text = normalize_text(reading_order(pieces, page.rect.width))
                method = "pymupdf_blocks_v4"
                images = page.get_images()
                if (not text and (images or page.get_drawings())) or (images and len(text) < 100):
                    warnings.append("OCR required for image-only page; OCR is not configured or enabled")
                pages.append(PageText(text, index, method=method, warnings=warnings,
                                      table_count=len(tables), label=page.get_label() or str(index)))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF cannot be parsed: {type(exc).__name__}") from exc
    return pages


def load_pdf_with_pypdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(PageText(text=text, page_number=index))
    except Exception:
        return []
    return pages

