"""Resolve PDF table labels from cell geometry, without domain-specific names."""

import json
import re


def numeric_cell(value):
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    return bool(lines) and all(re.fullmatch(r"[+\-\u2212]?\d[\d,.]*(?:%|\*)?", line) for line in lines)


def pdf_table(table, previous=None):
    rows = table.extract()
    if not rows:
        raise ValueError("Table has no cells")
    width = len(rows[0])
    data_start = next((index for index, row in enumerate(rows)
                       if sum(numeric_cell(cell) for cell in row) >= max(1, width // 2)
                       and not all(re.fullmatch(r"\(\d+\)", (cell or "").strip()) for cell in row)), 1)
    cells = next((row.cells for row in table.rows[data_start:] if all(row.cells)), None)
    if cells is None:
        raise ValueError("Table column geometry is incomplete")
    ranges = [(cell[0], cell[2]) for cell in cells]
    labels = []
    if data_start:
        for left, right in ranges:
            center = (left + right) / 2
            path = []
            for row_index in range(data_start):
                for value, cell in zip(rows[row_index], table.rows[row_index].cells):
                    value = " ".join((value or "").split())
                    if cell and cell[0] <= center <= cell[2] and value and not re.fullmatch(r"\(\d+\)", value):
                        if value not in path:
                            path.append(value)
            labels.append(" / ".join(path))
    elif table.header.external and all(table.header.names):
        labels = [" ".join(name.split()) for name in table.header.names]
    elif previous and len(previous["ranges"]) == width and all(
        abs(a - c) < 2 and abs(b - d) < 2 for (a, b), (c, d) in zip(ranges, previous["ranges"])
    ) and 0 <= table.bbox[1] - previous["bottom"] <= 80:
        labels = previous["labels"]
    if not labels or not all(labels) or len(set(labels)) != len(labels):
        raise ValueError("Table column labels are missing or ambiguous")
    units = " ".join(dict.fromkeys(name for name in table.header.names if name and "(" in name)) if table.header.external else ""
    if not units and previous and labels == previous["labels"]:
        units = previous.get("units", "")
    output = []
    warnings = []
    for row in rows[data_start:]:
        if len(row) != len(labels):
            warnings.append("Skipped table row with inconsistent column count")
            continue
        numeric_lines = [len(cell.splitlines()) for cell in row if numeric_cell(cell)]
        count = max(numeric_lines, default=1)
        if count > 1 and not all(len((cell or "").splitlines()) == count for cell in row):
            warnings.append("Skipped multi-value cells whose row alignment is ambiguous")
            continue
        expanded = list(zip(*(cell.splitlines() for cell in row))) if count > 1 else [row]
        for values in expanded:
            record = {key: " ".join((value or "").split()) for key, value in zip(labels, values)}
            if any(record.values()):
                output.append(f"ROW {len(output) + 1}: {json.dumps(record, ensure_ascii=False)}" + (f"; table units: {units}" if units else ""))
    state = {"ranges": ranges, "labels": labels, "bottom": table.bbox[3], "units": units}
    return "\n\n".join(output), state, warnings
