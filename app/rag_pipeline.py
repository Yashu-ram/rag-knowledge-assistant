"""One retrieval -> relevance -> generation -> citations path for the API."""
import logging
import re
import time
from math import log
from collections import Counter
from dataclasses import dataclass

from app.generation.grounding import ABSTENTION, terms
from app.generation.grounded_answer import render_answer
from app.generation.overview import is_overview_question
from app.generation.synthesis import answer_plan, completeness, retry_reason
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    citations: list
    model_used: str | None
    answerability: str
    metadata: dict


def relevant_evidence(question, evidence):
    query_terms = terms(question)
    frequencies = Counter(word for item in evidence for word in terms(item.text))
    weights = {word: 1 + log((len(evidence) + 1) / (frequencies[word] + 1)) for word in query_terms}
    for item in evidence:
        # Question text copied into a document is not evidence of an answer.
        prose = " ".join(part for part in re.split(r"(?<=[.!?])\s+", item.text)
                         if not part.rstrip().endswith("?"))
        overlap = query_terms & terms(prose)
        coverage = sum(weights[word] for word in overlap) / sum(weights.values()) if weights else 0
        cosine = item.metadata.get("semantic_score", 0)
        if coverage >= 0.35 or cosine >= 0.55:
            return True
    return False


class RAGPipeline:
    def __init__(self, generator):
        self.generator = generator
        self.retrieval = RetrievalService()

    def run(self, db, user_id, question, document_ids, model_name, response_mode):
        started = time.perf_counter()
        logger.info("[RAG-QUERY] received user_id=%s selected_documents=%d", user_id, len(document_ids or []))
        overview = is_overview_question(question)
        plan = answer_plan(question)
        logger.info("[RAG-SYNTHESIS] answer_style=%s", plan.style)
        logger.debug("[RAG-MULTIPART] detected_terms=%r", str(plan.parts)[:150])
        context_k = 6 if overview or response_mode == "deep" else 4
        evidence = self.retrieval.retrieve(db, user_id, question, document_ids, overview=overview, final_k=context_k)
        diagnostics = dict(self.retrieval.search_diagnostics)
        retrieval_ms = round((time.perf_counter() - started) * 1000)
        coverage = completeness(plan, ABSTENTION, [], evidence)
        reason = None
        if not evidence:
            reason = "No completed document chunks exist."
        elif plan.parts and not coverage["supported_parts"] and all(
                re.search(r"\b(?:does not (?:discuss|contain|provide)|not (?:mentioned|specified|provided)|no information)\b", item.text, re.I)
                for item in evidence):
            reason = "Context states that the requested information is absent."
        elif not overview and not relevant_evidence(question, evidence) and not coverage["supported_parts"]:
            reason = "Retrieved context is unrelated to the question."
        grounding = {"method": "citation_and_lexical_support_proxy", "faithfulness": None, "supported_claims": 0}
        answer, citations, model, model_ms = ABSTENTION, [], None, 0
        retries = 0
        if reason is None:
            raw, model, model_ms = self.generator(question, evidence, model_name, response_mode)
            from app.core.config import get_settings
            level = logging.INFO if get_settings().debug_llm_output else logging.DEBUG
            logger.log(level, "[RAG-GROUNDING] raw_model_answer=%r", raw[:150])
            answer, citations, grounding = render_answer(raw, evidence, overview)
            coverage = completeness(plan, answer, citations, evidence)
            correction = retry_reason(plan, answer, citations, evidence, coverage, grounding)
            if correction:
                retries = 1
                logger.info("[RAG-RETRY] reason=%s", correction)
                feedback = correction + " Missing supported parts: " + ", ".join(coverage["missing_supported_parts"])
                retry_raw, retry_model, retry_ms = self.generator(question, evidence, model_name, response_mode, feedback)
                model_ms += retry_ms
                logger.log(level, "[RAG-GROUNDING] retry_raw_model_answer=%r", retry_raw[:150])
                retry_answer, retry_citations, retry_grounding = render_answer(retry_raw, evidence, overview)
                retry_coverage = completeness(plan, retry_answer, retry_citations, evidence)
                def quality(cov, cites, metrics):
                    return (len(cov["covered_parts"]), len(cites), -metrics.get("summary_source_quote_fallback_points", 0))
                if retry_citations and quality(retry_coverage, retry_citations, retry_grounding) >= quality(coverage, citations, grounding):
                    answer, citations, grounding, coverage, model = retry_answer, retry_citations, retry_grounding, retry_coverage, retry_model
            logger.info("[RAG-GROUNDING] supported=%d rejected=%d recovered_citations=%d validation_error=%s",
                        grounding.get("supported_claims", 0), grounding.get("rejected_claims", 0),
                        grounding.get("citations_recovered", 0), grounding.get("validation_error", "none"))
            if not citations:
                reason = "The model found no supported answer or returned invalid citations."
            else:
                for part in coverage["unavailable_parts"]:
                    answer += f"\n\n{part}: not found in the uploaded document context."
                for part in coverage["missing_supported_parts"]:
                    answer += f"\n\n{part}: supporting passages were retrieved, but I could not produce a validated answer for this part."
                logger.info("[RAG-GROUNDING] accepted reason=source_supported parts_covered=%d partial=%s",
                            len(coverage["covered_parts"]), bool(coverage["unavailable_parts"] or coverage["missing_supported_parts"]))
        if reason:
            logger.info("abstention reason=%s", reason)
        logger.info("citations returned count=%d", len(citations))
        metadata = {
            "evidence_count": len(evidence), "overview": overview, "top_k": context_k,
            "vector_top_k": 8, "bm25_top_k": 8,
            "model_latency_ms": model_ms, "retrieval_latency_ms": retrieval_ms,
            "retrieved_chunks": [{"chunk_id": e.chunk_id, "source_filename": e.source_filename,
                "score": e.score, "search_source": e.source} for e in evidence],
            "tool_calls": [{"tool": "hybrid_search", "result_count": len(evidence), "diagnostics": diagnostics}],
            "retry_count": retries, "answer_style": plan.style, "completeness": coverage,
            "evidence_reason": reason or "Hybrid retrieved context sent to the LLM.",
            "grounding": grounding,
        }
        partial = citations and (coverage["unavailable_parts"] or coverage["missing_supported_parts"])
        return RAGResult(answer, citations, model, "partial" if partial else "answerable" if citations else "unanswerable", metadata)
