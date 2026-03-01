"""CoinMarketCap price lookup."""

import logging
import os

import requests

logger = logging.getLogger(__name__)

CMC_API_KEY = os.environ.get("COINMARKETCAP_API_KEY", "")
CMC_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"

DEFAULT_COINS = ["bitcoin", "ethereum", "solana", "tether", "bnb"]


def get_price(slug: str) -> dict | None:
    """Return {name, symbol, price_usd, pct_24h} for a coin slug, or None on failure."""
    if not CMC_API_KEY:
        logger.warning("[Crypto] COINMARKETCAP_API_KEY not set")
        return None
    try:
        resp = requests.get(
            CMC_URL,
            params={"slug": slug.lower().strip()},
            headers={
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": CMC_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        coin = next(iter(data["data"].values()))
        quote = coin["quote"]["USD"]
        return {
            "name": coin["name"],
            "symbol": coin["symbol"],
            "price_usd": quote["price"],
            "pct_24h": quote.get("percent_change_24h"),
        }
    except Exception:
        logger.exception("[Crypto] Failed to fetch price for %s", slug)
        return None


def format_price(info: dict) -> str:
    """Format a price dict into a readable message."""
    price = f"${info['price_usd']:,.2f}"
    pct = info.get("pct_24h")
    arrow = "🔺" if pct and pct >= 0 else "🔻"
    pct_str = f"{arrow} {pct:+.2f}%" if pct is not None else ""
    return f"{info['name']} ({info['symbol']})\nPrice: {price}  {pct_str}"
