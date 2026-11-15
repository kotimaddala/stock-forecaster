"""Order placement, gated by approval. Paper trading only.

Orders can be sent only for recommendations the human approved, and every
buy carries a stop-loss (one-triggers-other order). There is no code path for
short selling, options or margin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

ALPACA_PAPER = "https://paper-api.alpaca.markets/v2/orders"


class OrderRejected(RuntimeError):
    pass


@dataclass
class Order:
    symbol: str
    side: str  # buy | sell
    qty: int
    stop_price: Optional[float] = None
    approved: bool = False

    def check(self, held_qty: int = 0) -> None:
        if not self.approved:
            raise OrderRejected("orders require human approval")
        if self.qty <= 0:
            raise OrderRejected("quantity must be positive")
        if self.side == "buy" and not self.stop_price:
            raise OrderRejected("every buy must carry a stop-loss")
        if self.side == "sell" and self.qty > held_qty:
            raise OrderRejected("selling more than held would be a short sale, which is out of scope")
        if self.side not in ("buy", "sell"):
            raise OrderRejected("unknown side")


class DryRunBroker:
    """Default: writes orders to a log file instead of sending them."""

    def __init__(self, log_path: Path | str = "data/orders.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, order: Order, held_qty: int = 0) -> dict:
        order.check(held_qty)
        entry = {"mode": "dry-run", **order.__dict__}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry


class AlpacaPaperBroker:
    def __init__(self, key: str, secret: str) -> None:
        if not key or not secret:
            raise OrderRejected("Alpaca paper keys are not set")
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def submit(self, order: Order, held_qty: int = 0) -> dict:
        order.check(held_qty)
        body = {"symbol": order.symbol, "qty": str(order.qty), "side": order.side, "type": "market",
                "time_in_force": "day"}
        if order.side == "buy":
            body.update({"order_class": "oto", "stop_loss": {"stop_price": str(order.stop_price)}})
        resp = requests.post(ALPACA_PAPER, json=body, headers=self.headers, timeout=30)
        if resp.status_code >= 400:
            raise OrderRejected(f"broker rejected order: {resp.text[:200]}")
        return resp.json()
