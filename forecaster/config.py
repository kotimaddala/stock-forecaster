"""Settings and hard risk limits.

Risk limits live here as frozen configuration. No agent can change them at
runtime; a proposed change is only ever written to the journal for the
human owner to review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class RiskLimits:
    """Static guardrails. Frozen so code cannot mutate them mid-run."""

    max_risk_per_trade_pct: float = 1.0      # % of account lost if the stop is hit
    max_position_pct: float = 10.0           # largest single position, % of account
    max_sector_pct: float = 30.0             # largest sector exposure, % of account
    daily_loss_limit_pct: float = 2.0        # pause new ideas after this daily loss
    min_price: float = 5.0                   # no penny stocks
    min_avg_dollar_volume: float = 20_000_000  # liquidity floor (20-day average)
    allowed_actions: tuple = ("hold", "add", "trim", "exit", "open")
    # Explicitly out of scope: short selling, options, margin.


@dataclass(frozen=True)
class SearchLimits:
    """Tree-of-Thoughts budgets (Checkpoint 4)."""

    depth1_branches: int = 3
    depth2_refinements: int = 2
    beam_width: int = 2
    max_depth: int = 3
    prune_below: float = 4.0
    tie_margin: float = 1.0
    early_stop_lead: float = 3.0
    max_retrieval_calls: int = 12
    max_llm_calls: int = 14
    settle_window_days: int = 14
    expand_depth2: bool = True


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_tokens: int = 500
    overlap_tokens: int = 75
    candidates: int = 20
    top_k: int = 5
    min_score: float = 0.08


@dataclass(frozen=True)
class RoutingSettings:
    """Risk-based routing thresholds (Checkpoint 6)."""

    large_trim_pct_of_account: float = 3.0
    min_reviewer_confidence: float = 0.6
    human_score_threshold: int = 3


@dataclass(frozen=True)
class Settings:
    risk: RiskLimits = field(default_factory=RiskLimits)
    search: SearchLimits = field(default_factory=SearchLimits)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)

    # External services (all optional; offline mode works without them).
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    analyst_model: str = field(default_factory=lambda: _env("ANALYST_MODEL", "openai/gpt-4o-mini"))
    reviewer_model: str = field(default_factory=lambda: _env("REVIEWER_MODEL", "anthropic/claude-sonnet-4.5"))
    researcher_model: str = field(default_factory=lambda: _env("RESEARCHER_MODEL", "openai/gpt-4o-mini"))
    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT"))
    alpaca_key: str = field(default_factory=lambda: _env("ALPACA_API_KEY"))
    alpaca_secret: str = field(default_factory=lambda: _env("ALPACA_SECRET_KEY"))

    data_dir: Path = field(default_factory=lambda: Path(_env("FORECASTER_DATA_DIR", "data")))
    max_data_age_days: int = 5

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "journal.sqlite"

    @property
    def live_llm(self) -> bool:
        return bool(self.openrouter_api_key)


def load_settings() -> Settings:
    _load_dotenv()
    return Settings()


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so python-dotenv is not required."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
