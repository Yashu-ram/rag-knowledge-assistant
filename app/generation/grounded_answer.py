"""Grounded synthesis with source-bound citation validation."""
import json
import re
from uuid import uuid4

from app.api.schemas import Citation
from app.generation.grounding import ABSTENTION, terms


def has_negation(text):
    text = text.replace("\u2019", "'")
    return bool(re.search(r"\b(?:not|never|cannot|can't|isn't|aren't|doesn't|don't|mustn't|no)\b", text, re.I))


def context_records(evidence):
    from app.core.config import get_settings

    remaining = get_settings().max_generation_context_chars
    records = []
    for number, item in enumerate(evidence, 1):
        if remaining < 100:
            break
        text = item.text[:remaining]
        records.append({"source_id": number, "filename": item.source_filename,
                        "page": item.page_number, "section": item.metadata.get("section"),
                        "chunk_id": item.chunk_id, "text": text})
        remaining -= len(text)
    return records


def support_terms(text):
    # Common unit abbreviation, not a document-specific answer rule.
    return terms(re.sub(r"\bbps?\b", "basis points", text, flags=re.I))


def change_directions(text):
    directions = set()
    if re.search(r"\b(?:increase\w*|rose|rise\w*|grew|growth|higher)\b", text, re.I):
        directions.add("up")
    if re.search(r"\b(?:declin\w*|decreas\w*|reduc\w*|fell|fall\w*|lower)\b", text, re.I):
        directions.add("down")
    return directions


def parse_claims(raw):
    """Accept structured claims or sentence-level answer text, without inventing claims."""
    value = raw
    if isinstance(raw, str):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            value = json.loads(stripped)
        except ValueError:
            # Broken JSON is not a prose answer that can be safely recovered.
            if stripped.startswith(("{", "[")):
                raise ValueError("Malformed structured answer")
            value = stripped
    if isinstance(value, dict):
        if "claims" in value:
            claims = value["claims"]
            if not isinstance(claims, list) or len(claims) > 5:
                raise ValueError("Invalid claims")
            return claims
        value = value.get("answer", value.get("text"))
    if not isinstance(value, str) or not value.strip() or len(value) > 10000:
        raise ValueError("Invalid answer text")
    if value.strip() == ABSTENTION:
        return []
    sentences = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", value.strip())
    return [{"text": re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", text)}
            for text in sentences[:5] if text.strip()]


def supporting_snippet(text, record):
    from app.ingestion.structure import source_units

    candidates = [record["text"][u.start:u.end].strip() for u in
                  source_units(record["text"], pdf=record["page"] is not None)]
    candidates = [q for q in candidates if q and not q.startswith(("#", "```")) and not q.endswith("?")]
    claim_terms = support_terms(text)
    values = set(re.findall(r"\d+(?:\.\d+)?", text))
    def score(quote):
        coverage = len(claim_terms & support_terms(quote)) / max(1, len(claim_terms))
        return values <= set(re.findall(r"\d+(?:\.\d+)?", quote)), coverage
    quote = max(candidates, key=score, default=record["text"][:4000])
    if values and not score(quote)[0]:
        matching = [q for q in candidates if values & set(re.findall(r"\d+(?:\.\d+)?", q))]
        if matching:
            start = min(record["text"].find(q) for q in matching)
            end = max(record["text"].find(q) + len(q) for q in matching)
            quote = record["text"][start:end]
    return quote, score(quote)


def render_answer(raw, evidence, overview=False):
    metrics = {"method": "citation_and_lexical_support_proxy", "faithfulness": None,
               "supported_claims": 0, "rejected_claims": 0}
    try:
        claims = parse_claims(raw)
    except (ValueError, TypeError, KeyError):
        metrics["validation_error"] = "Invalid structured answer"
        return ABSTENTION, [], metrics
    if not claims and not overview:
        return ABSTENTION, [], metrics
    records = {r["source_id"]: r for r in context_records(evidence)}
    citations, points, proxies = [], [], []
    seen = set()
    for claim in claims:
        if not isinstance(claim, dict):
            metrics["rejected_claims"] += 1
            continue
        source_id, text, quote = claim.get("source_id"), claim.get("text"), claim.get("quote")
        if isinstance(text, str):
            markers = re.findall(r"\[(\d+)\]", text)
            text = re.sub(r"\[\d+\](?:\([^)]*\))?", "", text).strip()
            if source_id is None and markers:
                source_id = int(markers[0])
        if isinstance(source_id, str) and re.fullmatch(r"\[?\d+\]?", source_id.strip()):
            source_id = int(source_id.strip().strip("[]"))
        record = records.get(source_id) if type(source_id) is int else None
        if record is None and isinstance(text, str) and text.strip() and "quote" not in claim:
            candidates = [(candidate, *supporting_snippet(text, candidate)) for candidate in records.values()]
            best = max(candidates, key=lambda item: item[2], default=None)
            # Citation repair needs strong claim support, not just retrieval confidence.
            if best and best[2][0] and best[2][1] >= 0.6:
                record, quote, _ = best
                source_id = record["source_id"]
                metrics["citations_recovered"] = metrics.get("citations_recovered", 0) + 1
        if record and isinstance(text, str) and "quote" not in claim:
            quote, support = supporting_snippet(text, record)
            # Correct a misplaced source number only using stronger retrieved evidence.
            for candidate in records.values():
                candidate_quote, candidate_support = supporting_snippet(text, candidate)
                if candidate_support > support and candidate_support[1] >= max(0.35, support[1] + 0.15):
                    record, quote, support = candidate, candidate_quote, candidate_support
            if record["source_id"] != source_id:
                metrics["source_id_corrections"] = metrics.get("source_id_corrections", 0) + 1
                source_id = record["source_id"]
            if support[1] < (0.4 if overview else 0.01):
                metrics["rejected_claims"] += 1
                continue
        if record and isinstance(quote, str) and quote.strip() and quote not in record["text"]:
            # Accommodate PDF line wrapping only; return the exact original source span.
            match = re.search(r"\s+".join(re.escape(w) for w in quote.split()), record["text"])
            if match:
                quote = match.group()
        if (record is None or not isinstance(text, str) or not text.strip()
                or not isinstance(quote, str) or not quote.strip() or quote not in record["text"]
                or len(text) > 2000 or len(quote) > 4000):
            metrics["rejected_claims"] += 1
            continue
        if not set(re.findall(r"\d+(?:\.\d+)?", text)) <= set(re.findall(r"\d+(?:\.\d+)?", quote)):
            metrics["rejected_claims"] += 1
            continue
        claim_direction, source_direction = change_directions(text), change_directions(quote)
        if claim_direction and source_direction and claim_direction.isdisjoint(source_direction):
            text = quote
            metrics["direction_source_quote_fallbacks"] = metrics.get("direction_source_quote_fallbacks", 0) + 1
        if has_negation(text) != has_negation(quote):
            # A small local model can invert a negative fact: use verified source wording.
            text = quote
            metrics["polarity_source_quote_fallbacks"] = metrics.get("polarity_source_quote_fallbacks", 0) + 1
        key = re.sub(r"\s+", " ", text).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        item = evidence[source_id - 1]
        number = len(citations) + 1
        anchor = "source-" + uuid4().hex
        offset = item.text.find(quote)
        section = next((u.get("section") for u in item.metadata.get("source_units", [])
                        if u["start"] <= offset < u["end"]), item.metadata.get("section"))
        citations.append(Citation(citation_id=number, anchor=anchor, chunk_id=item.chunk_id,
            document_id=item.document_id, source_filename=item.source_filename,
            page_number=item.page_number, section=section or None, excerpt=quote))
        # This lexical proxy does not prove that a source entails the generated claim.
        claim_terms = support_terms(text)
        proxies.append(len(claim_terms & support_terms(quote)) / len(claim_terms) if claim_terms else 1.0)
        safe_text = re.sub(r"\s+", " ", text).strip()
        safe_text = re.sub(r"\[\d+\]", "", safe_text).strip()
        safe_text = safe_text.replace("<", "&lt;").replace(">", "&gt;")
        points.append(f"{safe_text} [{number}](#{anchor})")
    if overview and len(citations) < 3:
        from app.ingestion.structure import source_units

        used_chunks = {c.chunk_id for c in citations}
        ordered = sorted(records.values(), key=lambda r: evidence[r["source_id"] - 1].chunk_id in used_chunks)
        added = 0
        # Round-robin gives different chunks priority, then fills from longer chunks.
        for record in ordered * 3:
            for unit in source_units(record["text"], pdf=record["page"] is not None):
                quote = record["text"][unit.start:unit.end].strip()
                if (len(quote) < 20 or len(quote) > 1200 or quote.startswith(("#", "```"))
                        or quote.endswith("?") or any(quote == c.excerpt for c in citations)):
                    continue
                raw_point = json.dumps({"claims": [{"text": quote, "source_id": record["source_id"], "quote": quote}]})
                _, source_citations, _ = render_answer(raw_point, evidence)
                if not source_citations:
                    continue
                citation = source_citations[0]
                citation.citation_id = len(citations) + 1
                citations.append(citation)
                text = re.sub(r"\s+", " ", quote).replace("<", "&lt;").replace(">", "&gt;")
                points.append(f"{text} [{citation.citation_id}](#{citation.anchor})")
                proxies.append(1.0)
                added += 1
                break
            if len(citations) >= 3:
                break
        if added:
            metrics["summary_source_quote_fallback_points"] = added
    if not citations:
        return ABSTENTION, [], metrics
    metrics.update(supported_claims=len(points), citation_validity=1.0,
                   polarity_consistent=True, faithfulness=sum(proxies) / len(proxies))
    answer = "\n".join("- " + p for p in points) if overview else "\n\n".join(points)
    return answer, citations, metrics
