from dataclasses import dataclass

from app.retrieval.fusion import RetrievedChunk
from app.generation.grounding import supporting_passages



@dataclass(frozen=True)
class EvidenceDecision:
    answerability: str
    reason: str
    supporting_chunk_ids: list[str]


def assess_evidence(question: str, evidence: list[RetrievedChunk]) -> EvidenceDecision:
    if not evidence:
        return EvidenceDecision("unanswerable", "No indexed chunks matched the question.", [])

    passages = supporting_passages(question, evidence)
    if not passages:
        return EvidenceDecision("unanswerable", "No source passage directly supports the question.", [])
    support = list(dict.fromkeys(p.evidence.chunk_id for p in passages))
    return EvidenceDecision("answerable", "Relevant declarative source passages found.", support)



def decide_answerability(evidence: list[RetrievedChunk]) -> str:
    return assess_evidence("", evidence).answerability

