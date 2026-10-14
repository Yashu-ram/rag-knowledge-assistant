from dataclasses import dataclass
from functools import lru_cache
import re

import pysbd


@dataclass(frozen=True)
class SourceUnit:
    start: int
    end: int
    kind: str = "prose"
    section: str = ""
    context_start: int = 0
    context_end: int = 0


@lru_cache(maxsize=1)
def segmenter():
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def sentence_offsets(block: str, pdf: bool):
    if not pdf:
        offsets = [(span.start, span.end) for span in segmenter().segment(block)]
        return repair_numeric_endings(block, offsets)
    flat = []
    positions = []
    for token in re.finditer(r"\S+|\s+", block):
        if token.group().isspace():
            flat.append(" ")
            positions.append(token.start())
        else:
            flat.extend(token.group())
            positions.extend(range(token.start(), token.end()))
    result = []
    for span in segmenter().segment("".join(flat)):
        if span.end > span.start:
            result.append((positions[span.start], positions[span.end - 1] + 1))
    return repair_numeric_endings(block, result)


def repair_numeric_endings(block, offsets):
    result = list(offsets)
    for index in range(1, len(result)):
        prior_start, prior_end = result[index - 1]
        start, end = result[index]
        number = re.match(r"\d+\.\s+", block[start:end])
        if number and block[prior_start:prior_end].rstrip()[-1:] not in ".!?":
            boundary = start + number.end()
            result[index - 1] = (prior_start, boundary)
            result[index] = (boundary, end)
    return [(start, end) for start, end in result if end > start]


def source_units(text: str, *, pdf: bool = False) -> list[SourceUnit]:
    units = []
    section = ""
    start = 0
    pattern = r"\n\s*\n" if pdf else r"\n+"
    for boundary in list(re.finditer(pattern, text)) + [None]:
        end = boundary.start() if boundary else len(text)
        block = text[start:end]
        leading = len(block) - len(block.lstrip())
        block = block.strip()
        base = start + leading
        if block.startswith("#"):
            section = block.lstrip("# ")
        elif block:
            kind = "table_row" if block.startswith("ROW ") else "prose"
            if kind == "table_row" or block.startswith(("Q:", "Question:")):
                units.append(SourceUnit(base, base + len(block), kind, section, base, base + len(block)))
            else:
                spans = sentence_offsets(block, pdf)
                context_start, context_end = next(
                    ((a, b) for a, b in spans if len(block[a:b].strip()) >= 20
                     and re.search(r"[.!?][\)\]\"']*$", block[a:b].strip())),
                    spans[0] if spans else (0, 0),
                )
                for span_start, span_end in spans:
                    raw = block[span_start:span_end]
                    left = len(raw) - len(raw.lstrip())
                    right = len(raw.rstrip())
                    if right > left:
                        units.append(SourceUnit(base + span_start + left, base + span_start + right, kind, section,
                                                base + context_start, base + context_end))
        start = boundary.end() if boundary else len(text)
    return units
