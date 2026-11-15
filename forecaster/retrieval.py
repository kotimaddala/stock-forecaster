"""Semantic retrieval over long text (Checkpoint 3).

Design choices, in the order they apply:
1. Metadata filters run BEFORE similarity (ticker, form, date, superseded),
   so another company's look-alike paragraph can never be returned.
2. Hybrid scoring: TF-IDF cosine similarity for meaning plus an exact-term
   bonus for names that similarity can blur (program names, part names).
3. A relevance threshold: below it the tool answers "nothing relevant found"
   instead of returning weak matches.

TF-IDF (from scikit-learn) is the offline stand-in for neural embeddings: it
needs no download or GPU. The VectorIndex interface is the same either way,
so an embedding model or a vector database can replace it later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .guardrails import scan_for_injection
from .models import Chunk, Evidence

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_STOP = set("the a an of and or to in on for by with from as at is are was were be been this that "
            "what which why how does did do its it their our we has have had will would about".split())


def _terms(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class RetrievalLog:
    query: str
    filters: Dict[str, object]
    results: List[Dict[str, object]] = field(default_factory=list)
    nothing_relevant: bool = False


class VectorIndex:
    def __init__(self, min_score: float = 0.08, candidates: int = 20, top_k: int = 5,
                 semantic_weight: float = 0.75) -> None:
        self.chunks: List[Chunk] = []
        self.min_score = min_score
        self.candidates = candidates
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self.logs: List[RetrievalLog] = []

    # ---- indexing -------------------------------------------------------
    def add(self, chunks: Iterable[Chunk]) -> int:
        added = 0
        known = {c.chunk_id for c in self.chunks}
        for c in chunks:
            c.validate()
            if c.chunk_id not in known:
                self.chunks.append(c)
                known.add(c.chunk_id)
                added += 1
        self._mark_superseded()
        self._vectorizer = None
        return added

    def _mark_superseded(self) -> None:
        """Older versions of the same section are kept but excluded by default."""
        latest: Dict[tuple, str] = {}
        for c in self.chunks:
            key = (c.ticker, c.form, c.section)
            latest[key] = max(latest.get(key, ""), c.filed)
        for c in self.chunks:
            c.superseded = c.filed < latest[(c.ticker, c.form, c.section)]

    def _fit(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                                           min_df=1)
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    # ---- search -----------------------------------------------------------
    def search(self, query: str, ticker: str, forms: Optional[Sequence[str]] = None,
               since: Optional[str] = None, include_superseded: bool = False,
               top_k: Optional[int] = None) -> List[Evidence]:
        filters = {"ticker": ticker, "forms": list(forms) if forms else None, "since": since,
                   "include_superseded": include_superseded}
        log = RetrievalLog(query=query, filters=filters)
        self.logs.append(log)
        if not self.chunks:
            log.nothing_relevant = True
            return []
        if self._vectorizer is None:
            self._fit()

        idx = [i for i, c in enumerate(self.chunks)
               if (ticker == "*" or c.ticker.upper() == ticker.upper())
               and (forms is None or c.form in forms)
               and (since is None or c.filed >= since)
               and (include_superseded or not c.superseded)]
        if not idx:
            log.nothing_relevant = True
            return []

        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix[idx]).ravel()
        qterms = _terms(query)
        scored = []
        for pos, i in enumerate(idx):
            chunk = self.chunks[i]
            overlap = len(qterms & _terms(chunk.text)) / len(qterms) if qterms else 0.0
            score = self.semantic_weight * float(sims[pos]) + (1 - self.semantic_weight) * overlap
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: self.candidates]

        k = top_k or self.top_k
        results = []
        for score, chunk in scored[:k]:
            if score < self.min_score:
                continue
            ev = Evidence(chunk_id=chunk.chunk_id, text=chunk.text, source_url=chunk.source_url,
                          tier=chunk.tier, filed=chunk.filed, section=chunk.section,
                          score=round(score, 4), flagged_injection=bool(scan_for_injection(chunk.text)))
            results.append(ev.validate())
            log.results.append({"chunk_id": chunk.chunk_id, "score": round(score, 4),
                                "section": chunk.section, "filed": chunk.filed})
        log.nothing_relevant = not results
        return results
