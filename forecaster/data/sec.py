"""SEC EDGAR client: exact figures (Tier 1 facts) and filing text (Tier 1 chunks).

Uses only free public endpoints. The SEC requires a descriptive User-Agent
with contact details; set SEC_USER_AGENT="Your Name your@email".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from ..models import Chunk, Fact
from .http import CachedHttp, DataUnavailable

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

SECTION_NAMES = {
    "10-K": {"1": "Business", "1A": "Risk Factors", "7": "Management's Discussion and Analysis"},
    "10-Q": {"2": "Management's Discussion and Analysis", "1A": "Risk Factors"},
}


@dataclass
class Filing:
    ticker: str
    cik: int
    form: str
    filed: str
    accession: str
    primary_doc: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_doc)


class SecClient:
    def __init__(self, http: CachedHttp) -> None:
        if not http.offline and not http.session.headers.get("User-Agent", "").count("@"):
            raise DataUnavailable("SEC_USER_AGENT must include a contact email, e.g. 'Jane Doe jane@example.com'")
        self.http = http
        self._ciks: Optional[Dict[str, int]] = None

    # ---- lookup -------------------------------------------------------
    def cik(self, ticker: str) -> int:
        if self._ciks is None:
            data = self.http.get_json(TICKERS_URL, max_age_s=7 * 86400)
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        try:
            return self._ciks[ticker.upper()]
        except KeyError as exc:
            raise DataUnavailable(f"unknown ticker {ticker}") from exc

    def recent_filings(self, ticker: str, forms=("10-K", "10-Q", "8-K"), limit: int = 6) -> List[Filing]:
        cik = self.cik(ticker)
        data = self.http.get_json(SUBMISSIONS_URL.format(cik=cik))
        recent = data.get("filings", {}).get("recent", {})
        out: List[Filing] = []
        for form, filed, acc, doc in zip(recent.get("form", []), recent.get("filingDate", []),
                                         recent.get("accessionNumber", []), recent.get("primaryDocument", [])):
            if form in forms and doc:
                out.append(Filing(ticker.upper(), cik, form, filed, acc, doc))
            if len(out) >= limit:
                break
        return out

    # ---- exact figures --------------------------------------------------
    def quarterly_facts(self, ticker: str) -> Dict[str, Fact]:
        """Revenue growth (YoY) and gross margin for the latest reported quarter."""
        cik = self.cik(ticker)
        data = self.http.get_json(FACTS_URL.format(cik=cik))
        gaap = data.get("facts", {}).get("us-gaap", {})
        revenue = _quarterly_series(gaap, REVENUE_CONCEPTS)
        if len(revenue) < 5:
            raise DataUnavailable(f"{ticker}: not enough quarterly revenue history")
        source = f"SEC XBRL companyfacts CIK{cik:010d}"
        latest_end, latest_val, latest_filed = revenue[-1]
        prior = _match_prior_year(revenue, latest_end)
        facts: Dict[str, Fact] = {}
        if prior:
            growth = (latest_val - prior[1]) / abs(prior[1])
            facts["revenue_growth_yoy"] = Fact("revenue_growth_yoy", round(growth, 4), source, 1, latest_filed)
        gross = _quarterly_series(gaap, ("GrossProfit",))
        gp = {end: val for end, val, _ in gross}
        rev = {end: val for end, val, _ in revenue}
        margins = [(end, gp[end] / rev[end]) for end in sorted(gp) if end in rev and rev[end]]
        if margins:
            facts["gross_margin"] = Fact("gross_margin", round(margins[-1][1], 4), source, 1, latest_filed)
            if len(margins) >= 2:
                change = margins[-1][1] - margins[-2][1]
                facts["gross_margin_change"] = Fact("gross_margin_change", round(change, 4), source, 1, latest_filed)
        return facts

    # ---- long text ------------------------------------------------------
    def filing_chunks(self, filing: Filing, chunk_words: int = 375, overlap_words: int = 56) -> List[Chunk]:
        html = self.http.get_text(filing.url, max_age_s=30 * 86400)
        text = html_to_text(html)
        sections = split_sections(text, filing.form)
        chunks: List[Chunk] = []
        for section, body in sections.items():
            for i, piece in enumerate(chunk_words_with_overlap(body, chunk_words, overlap_words)):
                chunks.append(Chunk(
                    chunk_id=f"{filing.ticker}-{filing.form}-{filing.filed}-{_slug(section)}-{i}",
                    ticker=filing.ticker, form=filing.form, section=section, filed=filing.filed,
                    tier=1, source_url=filing.url, text=piece,
                ).validate())
        return chunks


# ---- helpers (pure functions, unit-tested) --------------------------------

def _quarterly_series(gaap: dict, concepts) -> List[Tuple[str, float, str]]:
    """Return [(end_date, value, filed)] for ~3-month periods, de-duplicated, oldest first."""
    for concept in concepts:
        units = gaap.get(concept, {}).get("units", {}).get("USD")
        if not units:
            continue
        by_period: Dict[Tuple[str, str], Tuple[float, str]] = {}
        for row in units:
            start, end = row.get("start"), row.get("end")
            if not start or not end or row.get("form") not in ("10-Q", "10-K"):
                continue
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            if not 80 <= days <= 100:
                continue
            key = (start, end)
            if key not in by_period or row["filed"] > by_period[key][1]:
                by_period[key] = (float(row["val"]), row["filed"])
        series = sorted(((end, val, filed) for (_, end), (val, filed) in by_period.items()))
        if series:
            return series
    return []


def _match_prior_year(series, end: str):
    target = date.fromisoformat(end)
    best = None
    for e, val, filed in series:
        gap = (target - date.fromisoformat(e)).days
        if 350 <= gap <= 380:
            best = (e, val, filed)
    return best


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()  # tables are excluded: figures come from structured data
    text = soup.get_text("\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-—]?\s*(.{0,80})$", re.IGNORECASE | re.MULTILINE)


def split_sections(text: str, form: str) -> Dict[str, str]:
    """Split filing text into named sections, keeping the longest copy of each.

    The table of contents repeats every heading, so the real section is the
    occurrence followed by the most text.
    """
    wanted = SECTION_NAMES.get(form)
    if not wanted:
        return {"Current report" if form == "8-K" else form: text}
    marks = [(m.start(), m.group(1).upper()) for m in ITEM_RE.finditer(text)]
    best: Dict[str, str] = {}
    for idx, (pos, item) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        name = wanted.get(item)
        if name and len(body) > len(best.get(name, "")):
            best[name] = body
    return {k: v for k, v in best.items() if len(v.split()) > 40}


def chunk_words_with_overlap(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
