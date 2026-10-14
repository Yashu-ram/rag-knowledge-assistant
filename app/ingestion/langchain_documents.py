from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.ingestion.loaders import PageText, load_document
from app.ingestion.pdf_timeout import load_pdf_with_timeout
from app.core.config import get_settings


class StructuredDocumentLoader(BaseLoader):
    """LangChain loader retaining the application's layout, OCR and table parsers."""

    def __init__(self, path, document_id, filename, file_type):
        self.path = Path(path)
        self.document_id = document_id
        self.filename = filename
        self.file_type = file_type

    def lazy_load(self):
        pages = (load_pdf_with_timeout(self.path, get_settings().pdf_extraction_timeout_seconds)
                 if self.file_type.lower() == ".pdf" else load_document(self.path, self.file_type))
        for page in pages:
            yield Document(page_content=page.text, metadata={
                "document_id": self.document_id, "filename": self.filename,
                "source_filename": self.filename, "source": self.filename,
                "page_number": page.page_number, "method": page.method,
                "warnings": page.warnings, "table_count": page.table_count, "label": page.label,
            })

    @staticmethod
    def pages(documents):
        return [PageText(d.page_content, d.metadata["page_number"], d.metadata["method"],
                         d.metadata["warnings"], d.metadata["table_count"], d.metadata["label"])
                for d in documents]
