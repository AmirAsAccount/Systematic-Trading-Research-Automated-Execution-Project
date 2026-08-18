"""
fetch.py  —  Async OHLCV fetcher with parquet cache
Fetches 1m bars for all symbols concurrently (within rate limits)
Run this once, then iterate backtest.py freely against cached data

Usage:
    python fetch.py                  # fetch all symbols, default 14 days
    python fetch.py --days 7         # override lookback
    python fetch.py --symbols BTC/USDT ETH/USDT SOL/USDT   # specific symbols
"""

import asyncio
import argparse
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR          = Path("cache")
TIMEFRAME          = "1m"
BARS_PER_FETCH     = 1000
MIN_BARS           = 500
MAX_CONCURRENT     = 5          # concurrent symbol fetches — stay inside rate limits
EXCHANGE_PRIORITY  = ["binanceus"]

CANDIDATE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
    "LINK/USDT", "UNI/USDT", "TRX/USDT", "ATOM/USDT", "BCH/USDT",
    "XLM/USDT", "ICP/USDT", "FIL/USDT", "HBAR/USDT", "ETC/USDT",
    "NEAR/USDT", "VET/USDT", "APT/USDT", "OP/USDT", "ARB/USDT",
    "GRT/USDT", "RNDR/USDT", "THETA/USDT", "FTM/USDT", "MKR/USDT",
    "LDO/USDT", "TIA/USDT", "SUI/USDT", "INJ/USDT", "STX/USDT",
    "SEI/USDT", "IMX/USDT", "EGLD/USDT", "AAVE/USDT", "FLOW/USDT",
    "QNT/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT", "CHZ/USDT",
    "ALGO/USDT", "MINA/USDT", "AXS/USDT", "APE/USDT", "PEPE/USDT",
    "SHIB/USDT", "DOGE/USDT", "BONK/USDT", "FLOKI/USDT", "WIF/USDT",
    "MEME/USDT", "BOME/USDT", "JUP/USDT", "PYTH/USDT", "ONDO/USDT",
    "PENDLE/USDT", "FET/USDT", "AGIX/USDT", "OCEAN/USDT", "AKT/USDT",
    "W/USDT", "ENA/USDT", "STRK/USDT", "DYM/USDT", "MANTA/USDT",
    "ALT/USDT", "XAI/USDT", "NFP/USDT", "ACE/USDT", "JTO/USDT",
    "ORDI/USDT", "SATS/USDT", "NOT/USDT", "BB/USDT", "REZ/USDT",
    "ETHFI/USDT", "AEVO/USDT", "METIS/USDT", "RON/USDT", "PIXEL/USDT",
    "PORTAL/USDT", "ZETA/USDT", "OMNI/USDT", "TAO/USDT", "TNSR/USDT",
    "CRV/USDT", "SUSHI/USDT", "1INCH/USDT", "DYDX/USDT", "ENS/USDT",
    "WOO/USDT", "GMT/USDT", "JASMY/USDT", "CKB/USDT", "ZIL/USDT",
    "ENJ/USDT", "HOT/USDT", "ONE/USDT", "ANKR/USDT", "RVN/USDT",
    "BAT/USDT", "LRC/USDT", "KAVA/USDT", "COMP/USDT", "YFI/USDT",
    "SNX/USDT", "ZEC/USDT", "XMR/USDT", "DASH/USDT", "WAVES/USDT",
    "QTUM/USDT", "OMG/USDT", "ONT/USDT", "NEO/USDT", "IOTA/USDT",
    "ICX/USDT", "LSK/USDT", "NANO/USDT", "DGB/USDT", "SC/USDT",
    "BTT/USDT", "WIN/USDT", "DENT/USDT", "STMX/USDT", "COTI/USDT",
    "CTSI/USDT", "HIVE/USDT", "XDC/USDT", "FLR/USDT", "CORE/USDT",
    "KAS/USDT", "ALICE/USDT", "ILV/USDT", "TLM/USDT", "YGG/USDT",
    "SUPER/USDT", "AUDIO/USDT", "PERP/USDT", "RUNE/USDT", "BAL/USDT",
    "NMR/USDT", "UMA/USDT", "BAND/USDT", "KNC/USDT", "STORJ/USDT",
    "POWR/USDT", "REQ/USDT", "AGLD/USDT", "FXS/USDT", "CELO/USDT",
    "AR/USDT", "ZEN/USDT", "GLMR/USDT", "ASTR/USDT", "SKL/USDT",
    "NKN/USDT", "BEAM/USDT", "TRB/USDT", "CAKE/USDT", "XVS/USDT",
    "UNFI/USDT", "CELR/USDT", "HOOK/USDT", "MAGIC/USDT", "HFT/USDT",
    "STG/USDT", "SSV/USDT", "LQTY/USDT", "JOE/USDT", "RDNT/USDT",
    "ID/USDT", "EDU/USDT", "ARKM/USDT", "WLD/USDT", "CYBER/USDT",
    "NTRN/USDT", "BLUR/USDT", "AXL/USDT", "SAGA/USDT", "IO/USDT",
    "ZRO/USDT", "ZK/USDT", "RENDER/USDT", "TON/USDT", "DOGS/USDT",
    "GTC/USDT", "RLC/USDT", "VTHO/USDT", "WAN/USDT", "ZRX/USDT",
    "CLV/USDT", "KSM/USDT", "PHA/USDT", "CRO/USDT", "MNT/USDT",
    "HYPE/USDT", "ME/USDT", "BERA/USDT", "EIGEN/USDT", "POPCAT/USDT",
    "MOG/USDT", "BRETT/USDT", "NEIRO/USDT", "COW/USDT", "SCR/USDT",
    "DRIFT/USDT", "GOAT/USDT", "PENGU/USDT", "VIRTUAL/USDT", "AIXBT/USDT",
    "TRUMP/USDT", "MELANIA/USDT", "GIGA/USDT", "PNUT/USDT", "ACT/USDT",
    "BIO/USDT", "FARTCOIN/USDT", "GRASS/USDT",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

def cache_path(symbol):
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"

def is_fresh(symbol, since_ms):
    """Cache is fresh if it exists and covers the requested window."""
    p = cache_path(symbol)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p, columns=["Open"])
        first_ts = int(df.index[0].timestamp() * 1000)
        return first_ts <= since_ms + 60_000 * 10   # within 10 bars
    except Exception:
        return False

# ── async fetch ───────────────────────────────────────────────────────────────
async def fetch_symbol_async(exchange, symbol, since_ms, semaphore):
    async with semaphore:
        all_rows = {}
        cursor   = since_ms
        max_ts   = int(datetime.utcnow().timestamp() * 1000)
        stall    = 0

        while cursor < max_ts:
            try:
                batch = await exchange.fetch_ohlcv(
                    symbol, timeframe=TIMEFRAME,
                    since=cursor, limit=BARS_PER_FETCH
                )
            except Exception as e:
                log(f"  [{symbol}] fetch error: {e}")
                break

            if not batch:
                break

            new = 0
            for row in batch:
                if row[0] not in all_rows:
                    all_rows[row[0]] = row
                    new += 1

            last_ts = batch[-1][0]
            if new == 0 or last_ts <= cursor:
                stall += 1
                if stall >= 2:
                    break
            else:
                stall = 0

            cursor = last_ts + 60_000
            await asyncio.sleep(0.1)   # light async delay — rate limit shared across coroutines

        if len(all_rows) < MIN_BARS:
            log(f"  [{symbol}] SKIP — only {len(all_rows)} bars")
            return symbol, None

        rows = sorted(all_rows.values(), key=lambda r: r[0])
        df   = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df   = df[["Open","High","Low","Close","Volume"]].astype(float)
        df.dropna(inplace=True)

        log(f"  [{symbol}] {len(df)} bars fetched")
        return symbol, df

async def fetch_all(exchange_id, symbols, since_ms):
    exchange  = getattr(ccxtpro, exchange_id)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    await exchange.load_markets()
    available = set(exchange.markets.keys())
    symbols   = [s for s in symbols if s in available]
    log(f"  {len(symbols)} symbols available on {exchange_id}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks     = [
        fetch_symbol_async(exchange, sym, since_ms, semaphore)
        for sym in symbols
        if not is_fresh(sym, since_ms)
    ]

    skipped = len(symbols) - len(tasks)
    if skipped:
        log(f"  {skipped} symbols loaded from cache (skipping fetch)")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    await exchange.close()
    return results

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=14)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=args.days)
    since_ms = int(start_dt.timestamp() * 1000)

    symbols = args.symbols if args.symbols else CANDIDATE_SYMBOLS

    log("=" * 65)
    log("  Async OHLCV Fetcher")
    log("=" * 65)
    log(f"  Exchange priority : {EXCHANGE_PRIORITY}")
    log(f"  Symbols           : {len(symbols)}")
    log(f"  Period            : {start_dt.date()} → {end_dt.date()} ({args.days}d)")
    log(f"  Concurrent        : {MAX_CONCURRENT}")
    log(f"  Cache dir         : {CACHE_DIR.resolve()}")
    log("=" * 65)

    # try exchanges in priority order
    for eid in EXCHANGE_PRIORITY:
        try:
            log(f"\n  Trying {eid} ...")
            results = asyncio.run(fetch_all(eid, symbols, since_ms))

            saved = failed = 0
            for item in results:
                if isinstance(item, Exception):
                    failed += 1
                    continue
                symbol, df = item
                if df is None:
                    failed += 1
                    continue
                df.to_parquet(cache_path(symbol))
                saved += 1

            log(f"\n  Saved  : {saved} symbols → {CACHE_DIR}/")
            log(f"  Failed : {failed}")
            log("=" * 65)
            break

        except Exception as e:
            log(f"  ✗ {eid}: {e}")
            continue

if __name__ == "__main__":
    main()