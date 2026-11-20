"""Federal Register client: the status of government policy (Tier 1).

The stage of a policy matters as evidence: a proposed rule can only weaken a
pillar, while a final rule can break or restore it (Checkpoint 2.1).
"""

from __future__ import annotations

from typing import List

from ..models import Chunk, Fact
from .http import CachedHttp

FEDREG_URL = "https://www.federalregister.gov/api/v1/documents.json"

STATUS_BY_TYPE = {
    "Rule": "final",
    "Proposed Rule": "proposed",
    "Notice": "notice",
    "Presidential Document": "executive_action",
}


class PolicyClient:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    def search(self, term: str, per_page: int = 5) -> List[dict]:
        params = {
            "conditions[term]": term,
            "order": "newest",
            "per_page": per_page,
            "fields[]": ["title", "type", "abstract", "publication_date", "html_url", "agencies"],
        }
        data = self.http.get_json(FEDREG_URL, params=params)
        return data.get("results", [])

    def status_fact(self, term: str) -> Fact | None:
        """The most recent document's stage, as a Tier 1 fact."""
        results = self.search(term, per_page=1)
        if not results:
            return None
        doc = results[0]
        status = STATUS_BY_TYPE.get(doc.get("type", ""), "notice")
        return Fact("policy_status", status, doc.get("html_url", FEDREG_URL), 1, doc["publication_date"]).validate()

    def chunks(self, term: str, ticker: str) -> List[Chunk]:
        out = []
        for i, doc in enumerate(self.search(term)):
            text = f"{doc.get('title', '')}. {doc.get('abstract') or ''}".strip()
            if len(text) < 40:
                continue
            out.append(Chunk(
                chunk_id=f"FR-{doc['publication_date']}-{i}", ticker=ticker, form="FedReg",
                section=STATUS_BY_TYPE.get(doc.get("type", ""), "notice"), filed=doc["publication_date"],
                tier=1, source_url=doc.get("html_url", FEDREG_URL), text=text,
            ).validate())
        return out
