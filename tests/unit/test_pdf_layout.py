from app.ingestion.layout import reading_order


def test_line_level_columns_are_not_interleaved():
    blocks = [
        (40, 100, 250, 112, "The pump pressure", "text"),
        (320, 100, 540, 112, "The warranty period", "text"),
        (40, 115, 250, 127, "is 12 bar.", "text"),
        (320, 115, 540, 127, "is 24 months.", "text"),
    ]
    text = reading_order(blocks, 600)
    assert "The pump pressure\nis 12 bar." in text
    assert "The warranty period\nis 24 months." in text
    assert text.index("12 bar") < text.index("warranty")


def test_full_width_table_remains_after_both_columns():
    blocks = [(40, 100, 250, 112, "Left paragraph.", "text"),
              (320, 100, 540, 112, "Right paragraph.", "text"),
              (40, 200, 540, 280, "ROW 1: table data", "table")]
    assert reading_order(blocks, 600).endswith("ROW 1: table data")
