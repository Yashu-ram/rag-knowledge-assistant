"""Question-shaped answer planning and conservative completeness checks."""
import re
from dataclasses import dataclass

from app.generation.grounded_answer import context_records, support_terms
from app.generation.overview import is_overview_question
from app.ingestion.structure import source_units


@dataclass(frozen=True)
class AnswerPlan:
    style: str
    parts: tuple[str, ...] = ()


def answer_plan(question):
    if is_overview_question(question):
        return AnswerPlan("summary")
    comparison = bool(re.search(r"\b(?:compare|comparison|difference|versus|vs\.?|respectively|whereas)\b", question, re.I))
    multipart = comparison or bool(re.search(r"\b(?:and|both|while|list)\b|;", question, re.I))
    if not multipart:
        return AnswerPlan("explanation" if re.search(r"\b(?:why|explain|how does|how do)\b", question, re.I) else "factual")
    text = question.strip().rstrip("?.!")
    text = re.sub(r"^.*?\b(?:compare|difference between|differences between|both|list|explain)\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:what (?:is|are|did|does)|how (?:did|do|does)|tell me about|give me)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:in|during|up to|as of)\s+(?:H[12]:|Q[1-4]:|\d{4}|January|February|March|April|May|June|July|August|September|October|November|December).*", "", text, flags=re.I)
    parts = [re.sub(r"^(?:the|both)\s+|\s+(?:change|changed|introduce|add|provide|support|cost|last|offer|respectively)$", "", p.strip(), flags=re.I)
             for p in re.split(r"\s+(?:and|vs\.?|versus|while|whereas)\s+|;", text, flags=re.I)]
    parts = [p[:100] for p in parts if p and support_terms(p)]
    # Resolve a shared noun phrase: "fresh and outstanding loan rates".
    if len(parts) == 2 and len(parts[0].split()) == 1 and len(parts[1].split()) > 1:
        parts[0] += " " + " ".join(parts[1].split()[1:])
    return AnswerPlan("comparison" if comparison else "multipart",
                      tuple(dict.fromkeys(parts[:4])) if len(parts) >= 2 else ())


def anchors(part):
    return support_terms(part) - {"period", "duration", "window", "time", "amount", "detail", "change"}


def part_passages(plan, evidence):
    found = {}
    for part in plan.parts:
        keys = anchors(part)
        matches = []
        if not keys:
            continue
        for record in context_records(evidence):
            for unit in source_units(record["text"], pdf=record["page"] is not None):
                quote = record["text"][unit.start:unit.end].strip()
                if re.search(r"\b(?:does not (?:discuss|contain|provide)|not (?:mentioned|specified|provided)|no information)\b", quote, re.I):
                    continue
                if quote and not quote.startswith("#") and not quote.endswith("?") and keys <= support_terms(quote):
                    matches.append(quote)
        if matches:
            found[part] = matches
    return found


def completeness(plan, answer, citations, evidence):
    found = part_passages(plan, evidence)
    covered = []
    for part, quotes in found.items():
        keys = anchors(part)
        for line in answer.splitlines():
            if not keys <= support_terms(line):
                continue
            linked = [c for c in citations if f"[{c.citation_id}](#{c.anchor})" in line]
            if not any(keys <= support_terms(c.excerpt) for c in linked):
                continue
            values = {v for q in quotes for v in re.findall(r"\d+(?:\.\d+)?", q) if not (len(v) == 4 and v.startswith(("19", "20")))}
            if values and not values & set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]\([^)]*\)", "", line))):
                continue
            covered.append(part)
            break
    return {"requested_parts": list(plan.parts), "supported_parts": list(found), "covered_parts": covered,
            "missing_supported_parts": [p for p in found if p not in covered],
            "unavailable_parts": [p for p in plan.parts if p not in found]}


def retry_reason(plan, answer, citations, evidence, coverage, grounding):
    if coverage["missing_supported_parts"]:
        return "Answer omitted requested parts present in context."
    if citations and plan.style == "factual" and len(re.findall(r"\b\w+\b", re.sub(r"\[\d+\]\([^)]*\)", "", answer))) < 4:
        return "Answer needs a self-contained factual sentence."
    if citations and plan.style == "explanation" and len(answer.split()) < 10:
        return "Explanation is too short to address the question."
    if plan.style == "summary" and grounding.get("summary_source_quote_fallback_points"):
        return "Synthesize summary points rather than relying only on extractive fallback."
    return None
