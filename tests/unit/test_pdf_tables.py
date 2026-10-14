import pymupdf

from app.ingestion.loaders import load_document
from app.ingestion.structure import source_units


def test_ruled_pdf_table_preserves_row_column_association(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    xs = [40, 180, 330, 480]
    ys = [70, 105, 140, 175]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    rows = [["Plan", "Monthly USD", "Annual USD"], ["Starter", "12", "120"], ["Pro", "25", "250"]]
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            page.insert_text((xs[col_index] + 7, ys[row_index] + 22), value)
    path = tmp_path / "fees.pdf"
    doc.save(path)
    doc.close()
    page = load_document(path, ".pdf")[0]
    assert page.table_count == 1
    rows = [page.text[u.start:u.end] for u in source_units(page.text, pdf=True) if u.kind == "table_row"]
    assert len(rows) == 2
    assert any('"Plan": "Pro"' in row and '"Monthly USD": "25"' in row for row in rows)
