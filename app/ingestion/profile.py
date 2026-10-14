from statistics import median

from app.ingestion.structure import source_units

INDEX_VERSION = "langchain-structured-v5"


def analyze(pages, token_counter, target_tokens: int, overlap_tokens: int, max_input_tokens: int) -> dict:
    counts = [token_counter(page.text[u.start:u.end]) for page in pages
              for u in source_units(page.text, pdf=page.page_number is not None)]
    tables = sum(page.table_count for page in pages)
    typical = median(counts) if counts else 0
    target = min(target_tokens, max_input_tokens - 8)
    overlap = min(overlap_tokens, target // 3)
    return {
        "index_version": INDEX_VERSION,
        "parser_methods": sorted({page.method for page in pages}),
        "page_count": len(pages), "readable_pages": sum(bool(page.text.strip()) for page in pages),
        "empty_pages": [page.page_number for page in pages if not page.text.strip()],
        "tables_detected": tables, "typical_unit_tokens": typical,
        "target_tokens": target, "overlap_tokens": overlap,
        "oversize_units": sum(count > target for count in counts),
        "warnings": [{"page": page.page_number, "message": warning} for page in pages for warning in page.warnings],
    }
