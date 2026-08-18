#!/usr/bin/env python3
"""
jupiter_screener.py — dead-simple Jupiter momentum screener.
==============================================================
No discovery pagination, no RugCheck, no Birdeye, no OBV divergence,
no multi-source rate-limit machinery. Just:

  1. Ask Jupiter's Tokens API v2 for the tokens with the biggest 5-minute
     price increase (`/tokens/v2/toptrending/5m`). Jupiter already
     excludes majors/stables (SOL, USDC, etc.) from this list for you.
  2. Take the #1 mover (that also clears a basic liquidity floor — see
     MIN_LIQUIDITY_USD below), and open a PAPER position on it.
  3. Set stop-loss / profit-target off that token's own recent
     volatility (its 1h price-change magnitude), with two guardrails:
       - stop-loss is never tighter than FEE_PCT / 2, so...
       - profit-target (always exactly 2x the stop-loss) is never
         smaller than FEE_PCT — it always covers round-trip fees.
  4. Poll the live price until stop or target is hit, log the result,
     go back to step 1.

This is PAPER TRADING ONLY — it never sends a transaction. Wire up
Jupiter's /swap endpoint yourself in `open_position`/`close_position`
if you want it to trade for real (see the comments there).

Dependencies: pip install requests --break-system-packages
Optional: export JUP_API_KEY=... to use api.jup.ag (higher rate limits)
          instead of the free lite-api.jup.ag endpoint.
"""

from __future__ import annotations

import os
import csv
import time
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

JUP_API_KEY = os.environ.get("JUP_API_KEY", "")
# Pro base needs the key; lite base is free/keyless but more rate-limited.
JUP_BASE = "https://api.jup.ag" if JUP_API_KEY else "https://lite-api.jup.ag"
JUP_HEADERS = {"x-api-key": JUP_API_KEY} if JUP_API_KEY else {}

TREND_INTERVAL      = "5m"   # the "most increasing" window — matches the ask
CANDIDATE_LIMIT      = 20    # how many top-5m movers to pull before filtering
MIN_LIQUIDITY_USD    = 10_000  # skip anything this thin — you likely can't
                                 # get a real quote/exit on it anyway
EXCLUDED_SYMBOLS = {           # small backstop; Jupiter's toptrending
    "SOL", "WSOL", "USDC", "USDT", "USDH", "DAI",  # already filters majors
}

POLL_NEW_ENTRY_SECONDS = 300   # rescan for a new candidate every 5 minutes
                                 # while flat — matches the 5m signal window
POLL_POSITION_SECONDS  = 20    # how often to check price against
                                 # stop/target while a position is open
MAX_HOLD_SECONDS        = 4 * 3600  # force-exit safety net if neither stop
                                 # nor target ever fires. Set to None to
                                 # disable and hold forever.

# ── Volatility-based stop / target ──────────────────────────────────────
FEE_PCT        = 5.0    # assumed round-trip cost (swap fees + slippage +
                          # price impact on a thin meme pool)
MIN_STOP_PCT   = FEE_PCT / 2.0   # 2.5% floor — guarantees target (2x this)
                          # always covers FEE_PCT, no matter how low
                          # measured volatility is
MAX_STOP_PCT   = 20.0    # ceiling — caps risk even if 1h move was huge
VOL_MULTIPLIER = 0.5     # stop-loss = this fraction of the token's own
                          # trailing 1h |price change|. 0.5 means: give the
                          # trade room to breathe within half its recent
                          # hourly swing before calling it wrong.
TARGET_MULTIPLE = 2.0    # profit target = this many multiples of the stop

TRADE_LOG_PATH = "jupiter_screener_trades.csv"


@dataclass
class Position:
    symbol: str
    mint: str
    entry_price: float
    stop_price: float
    target_price: float
    stop_pct: float
    target_pct: float
    opened_at: float


# ──────────────────────────────────────────────────────────────────────────
# JUPITER CLIENT
# ──────────────────────────────────────────────────────────────────────────

def jup_get(path: str, params: dict | None = None) -> dict | list | None:
    url = f"{JUP_BASE}{path}"
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, headers=JUP_HEADERS, timeout=15)
        except requests.exceptions.RequestException as e:
            log.warning("  network error on %s: %s — retrying", path, e)
            time.sleep(3 * (attempt + 1))
            continue

        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 5 * (attempt + 1)))
            log.warning("  rate-limited on %s, sleeping %.0fs", path, wait)
            time.sleep(wait)
            continue

        if r.status_code >= 400:
            log.error("  Jupiter %d on %s: %s", r.status_code, path, r.text[:200])
            return None

        return r.json()

    log.error("  giving up on %s after retries", path)
    return None


def get_top_movers(limit: int = CANDIDATE_LIMIT) -> list[dict]:
    """Tokens ranked by biggest price increase over TREND_INTERVAL."""
    data = jup_get(f"/tokens/v2/toptrending/{TREND_INTERVAL}", {"limit": limit})
    if not data:
        return []
    # Defensive: sort ourselves too, don't just trust response order.
    return sorted(
        data,
        key=lambda t: (t.get("stats5m") or {}).get("priceChange", 0.0),
        reverse=True,
    )


def get_price(mint: str) -> float | None:
    data = jup_get("/price/v3", {"ids": mint})
    if not data or mint not in data:
        return None
    entry = data[mint]
    price = entry.get("usdPrice") if isinstance(entry, dict) else None
    return float(price) if price is not None else None


# ──────────────────────────────────────────────────────────────────────────
# CANDIDATE SELECTION + STOP/TARGET SIZING
# ──────────────────────────────────────────────────────────────────────────

def pick_candidate(movers: list[dict]) -> dict | None:
    for t in movers:
        symbol = (t.get("symbol") or "").upper()
        if symbol in EXCLUDED_SYMBOLS:
            continue
        if (t.get("liquidity") or 0) < MIN_LIQUIDITY_USD:
            continue
        change_5m = (t.get("stats5m") or {}).get("priceChange")
        if change_5m is None or change_5m <= 0:
            continue  # nothing actually increasing left in the list
        return t
    return None


def compute_stop_and_target(token: dict) -> tuple[float, float]:
    """
    Returns (stop_loss_pct, profit_target_pct), both positive percentages.
    Stop is sized off the token's trailing 1h |price change| — its own
    recent volatility — then floored/ceilinged, and target is always
    exactly TARGET_MULTIPLE x the stop (so raising the fee assumption
    or the multiplier only requires touching the constants above).
    """
    stats1h = token.get("stats1h") or {}
    change_1h = stats1h.get("priceChange")
    if change_1h is None:
        # Fallback if 1h stats are missing for a very new pool: use the
        # 5m move itself, scaled up, as a rough volatility proxy.
        change_5m = (token.get("stats5m") or {}).get("priceChange", 0.0)
        vol_pct = abs(change_5m) * 100 * 3
    else:
        vol_pct = abs(change_1h) * 100

    stop_pct = max(MIN_STOP_PCT, min(vol_pct * VOL_MULTIPLIER, MAX_STOP_PCT))
    target_pct = stop_pct * TARGET_MULTIPLE
    return stop_pct, target_pct


# ──────────────────────────────────────────────────────────────────────────
# TRADE LOGGING
# ──────────────────────────────────────────────────────────────────────────

def log_trade(row: dict) -> None:
    is_new = not os.path.exists(TRADE_LOG_PATH)
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────
# POSITION LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────

def open_position(token: dict) -> Position | None:
    mint = token["id"]
    symbol = token.get("symbol", mint[:6])
    entry_price = get_price(mint) or token.get("usdPrice")
    if not entry_price:
        log.warning("  couldn't get a live price for %s — skipping", symbol)
        return None

    stop_pct, target_pct = compute_stop_and_target(token)
    pos = Position(
        symbol=symbol,
        mint=mint,
        entry_price=entry_price,
        stop_price=entry_price * (1 - stop_pct / 100),
        target_price=entry_price * (1 + target_pct / 100),
        stop_pct=stop_pct,
        target_pct=target_pct,
        opened_at=time.time(),
    )
    change_5m = (token.get("stats5m") or {}).get("priceChange", 0.0) * 100
    log.info(
        "ENTER  %-8s  price=%.8f  5m_move=+%.1f%%  stop=-%.1f%% (%.8f)  "
        "target=+%.1f%% (%.8f)",
        symbol, entry_price, change_5m, stop_pct, pos.stop_price,
        target_pct, pos.target_price,
    )
    # ── LIVE TRADING WOULD GO HERE ──
    # This is where you'd call Jupiter's /swap (or /ultra) endpoint to
    # actually buy `mint` with SOL/USDC and sign+send the transaction.
    # Left out deliberately — this script is paper-only.
    return pos


def close_position(pos: Position, exit_price: float, reason: str) -> None:
    pnl_pct = (exit_price / pos.entry_price - 1) * 100
    held_for = time.time() - pos.opened_at
    log.info(
        "EXIT   %-8s  price=%.8f  reason=%s  pnl=%+.2f%%  held=%.0fs",
        pos.symbol, exit_price, reason, pnl_pct, held_for,
    )
    log_trade({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": pos.symbol,
        "mint": pos.mint,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "stop_pct": round(pos.stop_pct, 3),
        "target_pct": round(pos.target_pct, 3),
        "pnl_pct": round(pnl_pct, 3),
        "reason": reason,
        "held_seconds": round(held_for, 1),
    })
    # ── LIVE TRADING WOULD GO HERE ──
    # This is where you'd call Jupiter's /swap endpoint to sell the full
    # token balance back to SOL/USDC.


def monitor_position(pos: Position) -> None:
    while True:
        time.sleep(POLL_POSITION_SECONDS)
        price = get_price(pos.mint)
        if price is None:
            log.warning("  %-8s no price this tick, will retry", pos.symbol)
            continue

        pnl_pct = (price / pos.entry_price - 1) * 100
        log.info("  %-8s price=%.8f  pnl=%+.2f%%", pos.symbol, price, pnl_pct)

        if price <= pos.stop_price:
            close_position(pos, price, "stop_loss")
            return
        if price >= pos.target_price:
            close_position(pos, price, "profit_target")
            return
        if MAX_HOLD_SECONDS and (time.time() - pos.opened_at) > MAX_HOLD_SECONDS:
            close_position(pos, price, "max_hold_timeout")
            return


# ──────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                         help="scan once, open at most one paper trade, then exit")
    args = parser.parse_args()

    log.info("Starting Jupiter momentum screener (paper trading only).")
    log.info("Using %s for Jupiter API calls.", JUP_BASE)

    while True:
        log.info("── scanning top %s movers on Jupiter ──", TREND_INTERVAL)
        movers = get_top_movers()
        candidate = pick_candidate(movers)

        if candidate is None:
            log.info("No qualifying candidate this cycle. Sleeping %ds.",
                       POLL_NEW_ENTRY_SECONDS)
            time.sleep(POLL_NEW_ENTRY_SECONDS)
            if args.once:
                return
            continue

        pos = open_position(candidate)
        if pos is None:
            time.sleep(POLL_NEW_ENTRY_SECONDS)
            if args.once:
                return
            continue

        monitor_position(pos)  # blocks until stop/target/timeout

        if args.once:
            return


if __name__ == "__main__":
    main()