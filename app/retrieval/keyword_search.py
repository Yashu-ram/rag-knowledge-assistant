import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[^\W_]+(?:[_.-][^\W_]+)*", re.UNICODE)


def tokenize(text: str, *, components: bool = False) -> list[str]:
    words = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    if components:
        for word in list(words):
            if not any(char.isdigit() for char in word):
                parts = re.split(r"[._-]", word)
                if len(parts) > 1:
                    words.extend(part for part in parts if part)
    return words


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [tokenize(doc, components=True) for doc in documents]
    query_terms = tokenize(query, components=True)
    if not query_terms or not tokenized_docs:
        return [0.0 for _ in documents]

    avg_len = sum(len(doc) for doc in tokenized_docs) / max(len(tokenized_docs), 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    k1 = 1.5
    b = 0.75
    scores = []
    total_docs = len(tokenized_docs)
    for doc in tokenized_docs:
        freqs = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in freqs:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freqs[term] + k1 * (1 - b + b * len(doc) / (avg_len or 1))
            score += idf * (freqs[term] * (k1 + 1)) / denom
        scores.append(score)
    return scores

