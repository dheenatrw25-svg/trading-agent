from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests


GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
REQUEST_TIMEOUT_SECONDS = 10
MARKET_MATCH_TOLERANCE = timedelta(seconds=60)


class PolymarketAPIError(RuntimeError):
    pass


class PolymarketClient:
    """Read-only access to public Gamma market data and CLOB quotes."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout

    def _get_json(
        self,
        base_url: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = self._session.get(
            f"{base_url}{path}",
            params=params,
            timeout=self._timeout,
        )
        try:
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PolymarketAPIError(
                f"Public Polymarket GET {path} failed: {exc}"
            ) from exc

    def find_markets(self, window_start: datetime, window_end: datetime) -> list[dict[str, Any]]:
        tolerance = MARKET_MATCH_TOLERANCE
        data = self._get_json(
            GAMMA_API_URL,
            "/markets",
            params={
                "closed": "false",
                "limit": 500,
                "order": "endDate",
                "ascending": "true",
                "start_date_min": (
                    window_start - tolerance
                ).astimezone(timezone.utc).isoformat(),
                "start_date_max": (
                    window_start + tolerance
                ).astimezone(timezone.utc).isoformat(),
                "end_date_min": (
                    window_end - tolerance
                ).astimezone(timezone.utc).isoformat(),
                "end_date_max": (
                    window_end + tolerance
                ).astimezone(timezone.utc).isoformat(),
            },
        )
        if not isinstance(data, list):
            raise PolymarketAPIError("Gamma markets response was not a list.")
        return [market for market in data if isinstance(market, dict)]

    def get_order_book(self, token_id: str) -> dict[str, Any]:
        data = self._get_json(
            CLOB_API_URL,
            "/book",
            params={"token_id": token_id},
        )
        if not isinstance(data, dict):
            raise PolymarketAPIError("CLOB order-book response was not an object.")
        return data

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        if side not in {"BUY", "SELL"}:
            raise ValueError("CLOB price side must be BUY or SELL.")
        data = self._get_json(
            CLOB_API_URL,
            "/price",
            params={"token_id": token_id, "side": side},
        )
        try:
            price = float(data["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PolymarketAPIError("CLOB price response did not contain a price.") from exc
        if not 0 < price < 1:
            raise PolymarketAPIError("CLOB returned a price outside the binary range.")
        return price

    def get_market(self, market_id: str) -> dict[str, Any]:
        data = self._get_json(GAMMA_API_URL, f"/markets/{market_id}")
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict):
            raise PolymarketAPIError("Gamma market response was not an object.")
        return data
