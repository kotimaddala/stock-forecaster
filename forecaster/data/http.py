"""Small HTTP helper with an on-disk cache and polite rate limiting.

Every live call goes through here, so a run can be replayed offline from the
cache and the evaluation never depends on a website being up.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class DataUnavailable(RuntimeError):
    """A source could not be reached and no cached copy exists."""


class CachedHttp:
    def __init__(self, cache_dir: Path, user_agent: str = "", min_interval: float = 0.15,
                 max_age_hours: float = 12.0, offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        self.min_interval = min_interval
        self.max_age_s = max_age_hours * 3600
        self.offline = offline
        self._last = 0.0
        self.calls = 0

    def _path(self, url: str, params: Optional[Dict[str, Any]]) -> Path:
        key = url + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".cache")

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None, max_age_s: Optional[float] = None) -> str:
        path = self._path(url, params)
        limit = self.max_age_s if max_age_s is None else max_age_s
        if path.exists() and (self.offline or time.time() - path.stat().st_mtime < limit):
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise DataUnavailable(f"offline and no cache for {url}")
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=30)
            self._last = time.time()
            self.calls += 1
            resp.raise_for_status()
        except requests.RequestException as exc:
            if path.exists():  # stale cache beats no data; the caller still checks dates
                return path.read_text(encoding="utf-8")
            raise DataUnavailable(f"{url}: {exc}") from exc
        path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None, max_age_s: Optional[float] = None) -> Any:
        text = self.get_text(url, params, headers, max_age_s)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataUnavailable(f"{url}: response was not JSON") from exc
