"""
Crypto Backtest Data Fetcher
============================
Stage 1: Filter — low liquid circulating supply coins with high volume relative to that supply
          (turnover ratio = 24h_volume / market_cap)
Stage 2: Fetch 1-minute OHLCV bars for each filtered coin and save as parquet.
          Each coin is fetched from the first exchange in EXCHANGES that lists it.

Dependencies:
    pip install ccxt pandas pyarrow tqdm requests
"""

import time
import logging
import requests
from pathlib import Path

import ccxt
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Exchanges tried in order — first one that lists a coin wins.
# Covers CEX majors; add "gate", "mexc", "bitget" etc. as needed.
EXCHANGES = ["binance", "okx", "bybit", "kucoin", "huobi"]

QUOTE_CURRENCY  = "USDT"
CACHE_DIR       = Path("crypto_cache")

# ── Stage-1 thresholds ────────────────────────────────────────────────────────
# turnover_ratio = 24h_volume_usd / market_cap
# 0.20 = 20% of entire circulating supply traded in one day
MIN_TURNOVER_RATIO  = 0.20
MIN_VOLUME_USDT     = 1_000_000     # dead-coin floor
TOP_N               = 30

# ── Stage-2 OHLCV fetch ───────────────────────────────────────────────────────
TIMEFRAME       = "1m"
LIMIT           = 1000          # exchange will clip to its own cap — that's fine
REQUEST_DELAY   = 0.3

# ── CoinGecko ─────────────────────────────────────────────────────────────────
CG_BASE  = "https://api.coingecko.com/api/v3"
CG_DELAY = 1.5      # free tier ~30 req/min

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# COINGECKO
# ──────────────────────────────────────────────────────────────────────────────

def cg_get(path: str, params: dict = None):
    r = requests.get(CG_BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_cg_universe(max_pages: int = 4) -> pd.DataFrame:
    all_rows = []
    log.info("Fetching CoinGecko universe (%d pages × 250 coins) …", max_pages)
    for page in range(1, max_pages + 1):
        rows = cg_get("/coins/markets", params={
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
        })
        if not rows:
            break
        all_rows.extend(rows)
        log.info("  page %d — %d coins so far", page, len(all_rows))
        time.sleep(CG_DELAY)

    df = pd.DataFrame(all_rows)[[
        "id", "symbol", "name", "market_cap", "circulating_supply",
        "current_price", "total_volume",
    ]].copy()
    df.columns = ["cg_id", "symbol", "name", "market_cap", "circ_supply", "price", "volume_24h"]
    df = df.dropna(subset=["market_cap", "circ_supply", "price", "volume_24h"])
    df = df[df["market_cap"] > 0]
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1
# ──────────────────────────────────────────────────────────────────────────────

def stage1_filter() -> list[str]:
    df = fetch_cg_universe(max_pages=4)
    log.info("Universe: %d coins with complete data.", len(df))

    df["turnover_ratio"] = df["volume_24h"] / df["market_cap"]

    mask = (
        (df["turnover_ratio"] >= MIN_TURNOVER_RATIO) &
        (df["volume_24h"]     >= MIN_VOLUME_USDT)
    )
    filtered = df[mask].sort_values("turnover_ratio", ascending=False).head(TOP_N)

    log.info(
        "Stage 1: %d / %d coins passed (turnover ≥ %.0f%%).",
        len(filtered), len(df), MIN_TURNOVER_RATIO * 100,
    )
    if filtered.empty:
        log.warning("No coins passed. Lower MIN_TURNOVER_RATIO (currently %.2f).", MIN_TURNOVER_RATIO)
        return []

    log.info("\n%s", filtered[
        ["symbol", "name", "market_cap", "circ_supply", "volume_24h", "turnover_ratio"]
    ].to_string(index=False))

    return filtered["symbol"].str.upper().tolist()


# ──────────────────────────────────────────────────────────────────────────────
# EXCHANGE POOL
# ──────────────────────────────────────────────────────────────────────────────

def build_exchange_pool(exchange_ids: list[str]) -> list[ccxt.Exchange]:
    """Instantiate and load markets for every exchange in the list."""
    pool = []
    for eid in exchange_ids:
        try:
            ex = getattr(ccxt, eid)({"enableRateLimit": True, "timeout": 20_000})
            ex.load_markets()
            log.info("  ✓ loaded %-12s  (%d markets)", eid, len(ex.markets))
            pool.append(ex)
        except Exception as e:
            log.warning("  ✗ could not load %s: %s", eid, e)
    return pool


def find_exchange_for(symbol: str, quote: str, pool: list[ccxt.Exchange]) -> tuple[ccxt.Exchange, str] | tuple[None, None]:
    """Return (exchange, pair) for the first exchange in pool that lists the coin."""
    pair = f"{symbol}/{quote}"
    for ex in pool:
        if pair in ex.markets:
            return ex, pair
    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(ex: ccxt.Exchange, symbol: str) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"]).set_index("datetime").sort_index()


def stage2_fetch(cg_symbols: list[str], pool: list[ccxt.Exchange], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("\nStage 2 — fetching 1m bars for %d coins across %d exchanges …",
             len(cg_symbols), len(pool))

    results, skipped = {}, []

    for sym in tqdm(cg_symbols, desc="Fetching OHLCV"):
        ex, pair = find_exchange_for(sym, QUOTE_CURRENCY, pool)

        if ex is None:
            log.warning("  ✗ %s — not listed on any configured exchange.", sym)
            skipped.append(sym)
            continue

        try:
            df = fetch_ohlcv(ex, pair)
            if df.empty:
                log.warning("  ✗ %s — empty OHLCV from %s.", pair, ex.id)
                skipped.append(sym)
                continue

            safe_name = pair.replace("/", "_").replace(":", "_")
            path = cache_dir / f"{safe_name}.parquet"
            df.to_parquet(path, index=True)

            results[pair] = {
                "exchange": ex.id,
                "bars":     len(df),
                "from":     str(df.index[0]),
                "to":       str(df.index[-1]),
                "path":     str(path),
            }
            log.info("  ✓ %-18s  via %-10s  %d bars  [%s → %s]",
                     pair, ex.id, len(df),
                     df.index[0].strftime("%m-%d %H:%M"),
                     df.index[-1].strftime("%m-%d %H:%M"))

        except ccxt.BaseError as e:
            log.error("  ✗ %s — ccxt error: %s", pair, e)
            skipped.append(sym)
        except Exception as e:
            log.error("  ✗ %s — unexpected error: %s", pair, e)
            skipped.append(sym)

        time.sleep(REQUEST_DELAY)

    log.info(
        "\nDone. %d cached, %d skipped (not on any exchange).",
        len(results), len(skipped),
    )
    if skipped:
        log.info("Skipped (DEX-only or unlisted): %s", ", ".join(skipped))
    if results:
        print("\n" + pd.DataFrame(results).T.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cg_symbols = stage1_filter()
    if not cg_symbols:
        log.warning("Nothing passed Stage 1. Exiting.")
        return

    log.info("\nLoading exchange pool: %s …", EXCHANGES)
    pool = build_exchange_pool(EXCHANGES)
    if not pool:
        log.error("No exchanges loaded. Check EXCHANGES list.")
        return

    stage2_fetch(cg_symbols, pool, CACHE_DIR)


if __name__ == "__main__":
    main()