import json
import re
from dataclasses import dataclass
from uuid import uuid4
from hashlib import sha256
from collections import Counter
from math import ceil

from app.api.schemas import Citation
from app.retrieval.fusion import RetrievedChunk
from app.retrieval.keyword_search import tokenize
from app.ingestion.structure import source_units
from app.core.config import get_settings
from app.generation.overview import is_overview_question

ABSTENTION = "I could not find enough information in the uploaded document to answer this."
STOPWORDS = set("what when where which who why how does did can could should would the and for from with about into that this are was were has have had check please tell explain give describe document documents uploaded answer question supported currently fix confirm returns returned required setting feature".split())
STOPWORDS.update("much during change changed changing decline declined increase increased decrease decreased reduce reduced rise rose fall fell support supports limit".split())
MONTHS = set("january february march april may june july august september october november december".split())


def terms(text: str) -> set[str]:
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    words = {word.strip(".-_") for word in tokenize(text, components=True)}
    aliases = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june",
               "jul": "july", "aug": "august", "sep": "september", "sept": "september", "oct": "october",
               "nov": "november", "dec": "december"}
    words = {aliases.get(word, word) for word in words}
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in words if (len(word) > 2 or any(char.isdigit() for char in word)) and word not in STOPWORDS}


@dataclass(frozen=True)
class Passage:
    passage_id: int
    evidence: RetrievedChunk
    quote: str
    context: str = ""


def source_context(item, quote):
    source = item.metadata.get("page_source_text", item.text)
    index = item.text.find(quote)
    if source is not None and index >= 0:
        units = item.metadata.get("source_units") or [u.__dict__ for u in source_units(item.text, pdf=item.source_filename.lower().endswith(".pdf"))]
        for unit in units:
            if unit["start"] <= index < unit["end"]:
                return source[unit.get("context_start", 0):unit.get("context_end", 0)].strip()
    return ""


def passage_key(quote: str) -> str:
    return sha256(quote.encode("utf-8")).hexdigest()


def passage_candidates(evidence):
    candidates = []
    seen = set()
    for item in evidence:
        is_pdf = item.source_filename.lower().endswith(".pdf")
        units = item.metadata.get("source_units") or [u.__dict__ for u in source_units(item.text, pdf=is_pdf)]
        previous = ""
        for unit in units:
            quote = item.text[unit["start"]:unit["end"]].strip()
            if is_pdf:
                quote = re.sub(r"^\d{1,4}\s*\n", "", quote)
            original_quote = quote
            if previous and quote.startswith(("The default ", "It ", "This ", "Its ", "For ")):
                start = item.text.find(previous)
                stop = item.text.find(quote, start + len(previous))
                if stop >= 0 and "\n\n" not in item.text[start:stop]:
                    quote = item.text[start:stop + len(quote)]
            previous = original_quote
            kind = unit.get("kind", "prose")
            if not quote or quote.endswith("?") or quote.startswith(("#", "```")):
                continue
            if kind != "table_row" and len(quote) < 20:
                continue
            if is_pdf and kind != "table_row" and (quote[:1].islower() or not re.search(r"[.!?][\)\]\"']*\d{0,3}$", quote)):
                continue
            if quote.startswith("|") and not quote.endswith("|"):
                continue
            key = (item.document_id, item.page_number, quote)
            if quote in item.text and key not in seen:
                seen.add(key)
                candidates.append((item, quote, kind))
    return candidates


def summary_passages(evidence):
    groups = []
    seen = set()
    for item in evidence:
        quotes = [quote for _, quote, _ in passage_candidates([item]) if len(quote) <= 1600]
        if not quotes and item.text.strip() and len(item.text.strip()) <= 1600:
            quotes = [item.text.strip()]
        groups.append((item, quotes))
    passages = []
    length = 0
    budget = get_settings().max_generation_context_chars
    # Round-robin avoids filling the prompt with sentences from only the first chunk.
    for position in range(max((len(quotes) for _, quotes in groups), default=0)):
        for item, quotes in groups:
            if position >= len(quotes):
                continue
            quote = quotes[position]
            if quote in seen or any(quote in p.quote or p.quote in quote for p in passages):
                continue
            context = source_context(item, quote)
            cost = len(quote) + (len(context) if context not in quote else 0)
            if length + cost > budget:
                continue
            seen.add(quote)
            passages.append(Passage(len(passages) + 1, item, quote, context))
            length += cost
            if len(passages) >= get_settings().max_generation_passages:
                return passages
    return passages


def answer_passages(question, evidence):
    return summary_passages(evidence) if is_overview_question(question) else supporting_passages(question, evidence)


def validate_summary_selection(raw, passages):
    answer, citations, metrics = validate_selection(raw, passages, max_claims=5)
    if metrics.get("validation_error") or metrics.get("rejected_claims"):
        return answer, citations, metrics
    selected = json.loads(raw)["passage_ids"]
    by_id = {p.passage_id: p for p in passages}
    selected = list(dict.fromkeys(selected))
    minimum = min(3, len(passages))
    # Complete a too-short overview with verified source quotes, never invented prose.
    changed = False
    if len(citations) < minimum:
        ordered = sorted(passages, key=lambda p: any(by_id[i].evidence.chunk_id == p.evidence.chunk_id for i in selected))
        for passage in ordered:
            if passage.passage_id not in selected and len(selected) < 5:
                selected.append(passage.passage_id)
                changed = True
            if len(selected) >= minimum:
                break
    represented = {by_id[i].evidence.chunk_id for i in selected}
    if len(represented) == 1:
        other = next((p for p in passages if p.evidence.chunk_id not in represented), None)
        if other is not None:
            if len(selected) == 5:
                selected[-1] = other.passage_id
            else:
                selected.append(other.passage_id)
            changed = True
    if changed:
        answer, citations, metrics = validate_selection(json.dumps({"passage_ids": selected}), passages, max_claims=5)
        metrics["summary_selection_completed_from_source"] = True
    if not citations:
        return answer, citations, metrics
    return "\n".join("- " + point for point in answer.split("\n\n")), citations, metrics


def supporting_passages(question: str, evidence: list[RetrievedChunk]) -> list[Passage]:
    query_terms = terms(question)
    if not query_terms:
        return []
    identifiers = {word for word in query_terms if any(char.isdigit() for char in word)}
    acronyms = {word.lower() for word in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*\b", question)}
    requested_months = query_terms & MONTHS
    start_months = {month for month in requested_months if re.search(r"\bfrom\s+" + month + r"\b", question, re.I)}
    personal_name = bool(re.search(r"\bmy\s+name\b", question, re.I))
    candidates = passage_candidates(evidence)
    mentioned_values = []
    for _, quote, kind in candidates:
        if kind != "table_row":
            continue
        body = re.search(r"\{.*\}", quote)
        if not body:
            continue
        try:
            record = json.loads(body.group())
        except ValueError:
            continue
        for value in record.values():
            value_terms = terms(str(value))
            if value_terms and any(char.isalpha() for char in str(value)) and value_terms <= query_terms:
                mentioned_values.append(value_terms)
    scope_match = re.search(r"\b(?:during|from|since|over|in|up to)\s+(.+)$", question, re.I)
    scope_terms = terms(scope_match.group(1)) if scope_match else set()
    frequencies = Counter(word for _, quote, _ in candidates for word in terms(quote))
    distinctive = {word for word in query_terms if 0 < frequencies[word] <= max(1, ceil(len(candidates) * 0.15))
                   and not any(char.isdigit() for char in word) and word not in scope_terms}
    qualifiers = query_terms & {"default", "maximum", "minimum", "median", "cumulative", "net", "gross"}
    passages = []
    seen = set()
    for item, quote, kind in candidates:
        if personal_name and not re.search(r"\b(?:my|your) name is\b", quote, re.I):
            continue
        quote_terms = terms(quote)
        context = source_context(item, quote)
        context_terms = quote_terms | terms(context) | terms(item.metadata.get("section", ""))
        if not (requested_months - start_months) <= quote_terms or not (acronyms - scope_terms) <= quote_terms:
            continue
        context_dates = {word for word in identifiers if re.fullmatch(r"\d{4}(?:-\d{2,4})?", word)}
        if not (identifiers - context_dates - scope_terms) <= quote_terms or not context_dates <= context_terms:
            continue
        if mentioned_values and not all(value <= quote_terms | terms(context) for value in mentioned_values):
            continue
        if scope_terms and not scope_terms <= context_terms:
            continue
        if kind != "table_row" and not distinctive <= quote_terms:
            continue
        if not qualifiers <= quote_terms:
            continue
        coverage = len(query_terms & (quote_terms | (scope_terms & context_terms))) / len(query_terms)
        similarity = item.metadata.get("passage_similarities", {}).get(passage_key(quote), 0)
        threshold = 1.0 if len(query_terms) <= 2 else 0.5
        if coverage < threshold and similarity < 0.5:
            continue
        if quote not in seen and len(quote) <= 4000:
            seen.add(quote)
            passages.append(Passage(len(passages) + 1, item, quote, context))
    passages.sort(key=lambda p: (
        len(query_terms & terms(p.quote)) / len(query_terms)
        + p.evidence.metadata.get("passage_similarities", {}).get(passage_key(p.quote), 0)
    ), reverse=True)
    bounded = []
    length = 0
    settings = get_settings()
    for passage in passages:
        if len(bounded) >= settings.max_generation_passages:
            break
        cost = len(passage.quote) + len(passage.context)
        if length + cost > settings.max_generation_context_chars:
            continue
        bounded.append(passage)
        length += cost
    return bounded


def validate_selection(raw: str, passages: list[Passage], max_claims: int = 4) -> tuple[str, list[Citation], dict]:
    try:
        payload = json.loads(raw)
        selected = payload["passage_ids"]
        if not isinstance(selected, list):
            raise ValueError("passage_ids must be a list")
        if len(selected) > max_claims:
            raise ValueError("Too many selected passages")
    except (ValueError, TypeError, KeyError):
        return ABSTENTION, [], {"faithfulness": None, "method": "exact_source_passages", "validation_error": "Invalid structured response"}

    by_id = {p.passage_id: p for p in passages}
    valid = []
    rejected = 0
    for passage_id in selected:
        passage = by_id.get(passage_id) if type(passage_id) is int else None
        if passage is None or passage.quote not in passage.evidence.text or (
            passage.context and passage.context not in passage.evidence.metadata.get("page_source_text", passage.evidence.text)
        ):
            rejected += 1
        elif passage not in valid:
            valid.append(passage)
    metrics = {
        "method": "exact_source_passages",
        "proposed_claims": len(selected), "rejected_claims": rejected,
        "selection_precision": (len(selected) - rejected) / len(selected) if selected else None,
        "faithfulness": 1.0 if valid and not rejected else None,
        "supported_claims": len(valid) if not rejected else 0,
    }
    # Fail closed: never silently turn a partly invalid selection into a verified answer.
    if rejected or not valid:
        return ABSTENTION, [], metrics
    # Overlap can yield both a short quote and a containing quote from the same page.
    valid = [p for p in valid if not any(
        p is not other and p.quote != other.quote and p.quote in other.quote
        and p.evidence.document_id == other.evidence.document_id
        and p.evidence.page_number == other.evidence.page_number for other in valid
    )]
    metrics["supported_claims"] = len(valid)
    citations = []
    lines = []
    for number, passage in enumerate(valid, 1):
        anchor = "source-" + uuid4().hex
        citations.append(Citation(
            section=passage.evidence.metadata.get("section") or None,
            citation_id=number, anchor=anchor, chunk_id=passage.evidence.chunk_id,
            document_id=passage.evidence.document_id,
            source_filename=passage.evidence.source_filename,
            page_number=passage.evidence.page_number, excerpt=passage.quote,
            context_excerpt=passage.context if passage.context != passage.quote else None,
        ))
        quote = re.sub(r"\s+", " ", passage.quote).strip().replace("\\", "\\\\")
        quote = re.sub(r"([`*_{}\[\]<>])", r"\\\1", quote)
        lines.append(f"{quote} [{number}](#{anchor})")
    return "\n\n".join(lines), citations, metrics
