import re


def is_overview_question(question: str) -> bool:
    """Match document-overview requests, not arbitrary queries containing 'summary'."""
    text = re.sub(r"\s+", " ", question.lower()).strip(" .?!")
    text = re.sub(r"^(?:please |could you |can you )", "", text)
    document = r"(?:(?:the uploaded|uploaded|this|the|my) )?(?:document|pdf|file)"
    citations = r"(?: (?:using|with) citations)?"
    patterns = [
      r"(?:give )?(?:the )?key points",
      r"explain briefly",
        rf"(?:what is|what's) {document} about{citations}",
        rf"summari[sz]e {document}{citations}",
        rf"(?:give me |show me |list )?(?:the )?key points(?: (?:of|from|in) {document})?{citations}",
        rf"what are (?:the )?main topics(?: (?:of|in) {document})?{citations}",
        rf"explain {document} briefly{citations}",
        rf"(?:give me )?(?:a |an )?(?:brief )?(?:summary|overview) of {document}{citations}",
    ]
    return any(re.fullmatch(pattern, text) for pattern in patterns)
