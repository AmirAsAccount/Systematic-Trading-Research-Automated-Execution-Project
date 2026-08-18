#!/usr/bin/env python3
"""
live_screener.py — Birdeye-backed live OBV-divergence screener.
==================================================================
Pure screener + paper-tracking. No fetch.py / GeckoTerminal dependency —
both discovery and OHLCV now come from Birdeye, which is purpose-built
for Solana and has a real rate-limit budget (paid tiers start at 15 RPS,
vs GeckoTerminal free tier's much lower effective ceiling).

REQUIRES: a Birdeye API key, set as the BIRDEYE_API_KEY environment
variable. Get one at https://bds.birdeye.so. The discovery endpoint used
below (/defi/v3/token/list with min_liquidity / min_volume_24h_usd
filters) is documented as Starter-plan-and-above — if your key is on
the free tier it may 403 on discovery specifically; everything else
(OHLCV, RugCheck, Jupiter) should work regardless of tier.

LOOP
  1. Periodically (UNIVERSE_REFRESH_SECONDS) rebuild the candidate list via
     discover_top_volatile_safe(): page Birdeye's floor-filtered token list
     (min liquidity, min 24h volume), rank everything seen so far by |1h
     price change|, and safety-gate (RugCheck + Jupiter) the most-volatile
     unchecked candidates first. Keeps paging until exactly
     TARGET_CANDIDATES (30) tokens have passed the gate, or MAX_DISCOVERY_
     PAGES is hit — whichever comes first — so the universe is always
     "the 30 most volatile tokens that are actually safe to trade", not
     just whatever cleared the floors first.
  2. Single active position at a time (mirrors backtest.py's mutual
     exclusivity): if something is open, ONLY that token gets polled —
     no scanning anyone else. If flat, all TARGET_CANDIDATES tokens get
     scanned every cycle (PASSIVE_BATCH_SIZE == TARGET_CANDIDATES, so no
     trimming). Scanning stops the instant one of them fires an entry.
  3. POLL_INTERVAL_SECONDS = 300 — the whole loop (entry check when flat,
     exit + unrealized/realized PnL check when in a position) runs every
     5th minute.

Dependencies:
    pip install pandas requests --break-system-packages
"""

from __future__ import annotations

import os
import time
import logging
import argparse
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd

import backtest as B   # check_entry / compute_obv / SWING_N / etc — reused as-is

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

BIRDEYE_BASE   = "https://public-api.birdeye.so"
BIRDEYE_API_KEY = "40a4ba5a1cca4a768c5a3a5f74b21fc7"
BIRDEYE_CHAIN   = "solana"

POLL_INTERVAL_SECONDS    = 300  # every 5th minute — direct entry/exit check
                                  # + PnL tracking cycle. With TARGET_CANDIDATES
                                  # capped at 30 and PASSIVE_BATCH_SIZE matching
                                  # it, a full passive scan is at most 30
                                  # sequential fetch_ohlcv calls, which fits
                                  # comfortably inside a 5-minute window even
                                  # at the adaptive delay's slow end.
SAFETY_REFRESH_SECONDS   = 900
UNIVERSE_REFRESH_SECONDS = 1800

# ── Discovery floors/ceilings ────────────────────────────────────────────
MIN_LIQUIDITY_USD  = 20_000
MIN_VOLUME_24H_USD = 50_000
MAX_MARKET_CAP_USD = 50_000_000
TOP_N              = 200   # raw-candidate cap for the legacy discover_candidates()
                            # helper (still used by refresh_price_change's cheap
                            # re-poll) — unrelated to TARGET_CANDIDATES below
DISCOVERY_PAGES    = 3      # Birdeye caps each call at 100 results — paginate
                             # via offset to actually reach TOP_N instead of
                             # silently truncating at 100

# ── Volatile-universe target (new) ───────────────────────────────────────
TARGET_CANDIDATES      = 20  # exact number of SAFETY-PASSED, most-volatile
                               # tokens kept under active scan at any time.
                               # Once this many are locked in, discovery
                               # stops and entry-scanning begins.
MAX_DISCOVERY_PAGES     = 20  # hard cap on how many 100-token pages we'll
                               # page through while hunting for
                               # TARGET_CANDIDATES — bounds worst-case
                               # Birdeye/RugCheck/Jupiter call volume if the
                               # floors return a huge universe but most of
                               # it fails the safety gate
CHECK_BATCH_MULTIPLIER  = 3   # each page, only run the safety gate on the
                               # top (needed * this) unchecked, volatility-
                               # ranked candidates — not the whole page —
                               # to keep RugCheck/Jupiter call volume down
MAX_CHECK_BATCH_SIZE    = 15  # hard sub-20 ceiling on top of the multiplier
                               # above — caps how many candidates get gated
                               # in a single round no matter how many are
                               # still "needed". This is what actually
                               # protects Jupiter's rate limit; the
                               # multiplier alone isn't enough on an early
                               # page where needed == target_n.
VOLATILITY_WINDOW_MIN   = 5   # final ranking of the safety-passed set uses
                               # the REAL % price change over this many
                               # minutes, computed locally from 1m OHLCV
                               # bars — not Birdeye's bulk list endpoint,
                               # which only exposes 1h/2h/4h/8h/24h
                               # timeframes. Set to 10 for a 10-minute
                               # window instead. Costs exactly one extra
                               # OHLCV call per already safety-gated
                               # candidate (≤ TARGET_CANDIDATES), never per
                               # raw discovery-page candidate.

# Small backstop on top of the market-cap ceiling — these are well-known
# enough that excluding by name too is just extra safety margin
EXCLUDED_SYMBOLS = {
    "USDT", "USDC", "USDH", "DAI", "UST", "USDS", "BUSD", "TUSD",
    "PYUSD", "FDUSD", "EURC", "USDE", "USD1",
    "SOL", "WSOL", "WBTC", "CBBTC", "WETH", "BTC", "ETH",
}

# ── OHLCV ────────────────────────────────────────────────────────────────
OHLCV_INTERVAL      = "1m"
OHLCV_LOOKBACK_MIN  = 1000   # minutes of 1m history per call (Birdeye caps at 1000 records)

# ── Birdeye adaptive rate limiting ──────────────────────────────────────────
BD_DELAY_FLOOR = 1.0   # conservative starting point; adapts down if it's fine
_bd_adaptive_delay = BD_DELAY_FLOOR
_bd_consecutive_ok = 0
_bd_last_request_ts = 0.0
_bd_rate_lock = threading.Lock()  # bd_get isn't called concurrently today,
                                    # but locking costs nothing and protects
                                    # against that changing later.

PASSIVE_BATCH_SIZE = TARGET_CANDIDATES  # universe is now fixed at exactly
                             # TARGET_CANDIDATES safety-passed tokens, so
                             # every cycle scans all of them — no batching/
                             # trimming needed.

# ── RugCheck adaptive rate limiting (separate budget from Birdeye/Jupiter) ──
RC_DELAY_FLOOR = 0.5
_rc_adaptive_delay = RC_DELAY_FLOOR
_rc_consecutive_ok = 0
_rc_last_request_ts = 0.0
_rc_rate_lock = threading.Lock()  # rugcheck_report() runs inside
                                    # ThreadPoolExecutor(SAFETY_WORKERS) —
                                    # this lock is load-bearing, not optional.

# ── Jupiter adaptive rate limiting (separate budget from Birdeye/RugCheck) ──
# lite-api.jup.ag's free tier rate-limits much sooner than the old paid
# quote-api key did. Without backoff here, a 429 looked identical to a
# genuine "no route" rejection — which is why late in a discovery cycle
# every remaining candidate (including obviously-liquid ones like XRP,
# POPCAT, ACT) started getting rubber-stamped "no_buy_route".
JUP_DELAY_FLOOR = 1.0  # raised from 0.5 — observed 429s within the first
                         # couple of calls even with the race fixed below,
                         # so the free tier's real budget is tighter than
                         # 0.5s/request. Adaptive logic still ramps this
                         # down if a run goes clean for a while.
_jup_adaptive_delay = JUP_DELAY_FLOOR
_jup_consecutive_ok = 0
_jup_last_request_ts = 0.0
_jup_rate_lock = threading.Lock()  # jup_quote() runs inside
                                     # ThreadPoolExecutor(SAFETY_WORKERS), 2
                                     # calls per candidate (buy+sell). WITHOUT
                                     # this lock, two worker threads can both
                                     # read _jup_last_request_ts before either
                                     # updates it, both conclude "clear to
                                     # send", and fire simultaneously —
                                     # silently bypassing the adaptive delay
                                     # entirely. This was the actual cause of
                                     # rate-limiting "off the bat": pacing was
                                     # being computed correctly but never
                                     # enforced across threads.

# ── Honeypot / RugCheck thresholds ──────────────────────────────────────────
# Paper-trading only right now, no real execution — so the gate is trimmed
# down to the two checks that are (a) free/fast and (b) catch a failure mode
# that's independent of volume entirely: a live mint authority means the dev
# can print unlimited new supply regardless of how much real trading is
# happening; a live freeze authority means your (paper) wallet could be
# frozen. LP-lock % and top-10 holder concentration are dropped — LP-lock
# in particular doesn't map cleanly onto pump.fun-style bonding-curve
# tokens anyway, which is most of this candidate set, and holder
# concentration was producing impossible (>100%) readings that suggested
# the underlying RugCheck field wasn't measuring what was assumed. If/when
# this moves toward real execution, those should come back properly fixed
# rather than just re-enabled with the same broken assumptions.
RUGCHECK_BASE  = "https://api.rugcheck.xyz/v1"
SAFETY_WORKERS = 2   # lower concurrency — each worker fires 2 Jupiter calls
                       # (buy+sell) per candidate, and lite-api.jup.ag's free
                       # tier throttles well before 4 workers' worth of traffic

# ── Jupiter trap thresholds ─────────────────────────────────────────────────
# quote-api.jup.ag/v6 was deprecated — migrated to lite-api.jup.ag/swap/v1/quote
# (same response schema, no API key needed for free tier usage)
JUP_QUOTE_URL              = "https://lite-api.jup.ag/swap/v1/quote"
USDC_MINT                  = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
PROBE_USDC_AMOUNT          = 25_000_000
MAX_JUP_PRICE_IMPACT_PCT   = 15.0
MAX_ROUND_TRIP_LOSS_PCT    = 40.0

# ── Position safety exits (independent of the OBV-divergence signal) ───────
# check_live_exit() can ONLY fire by finding the divergence pattern in NEW
# bars. If a pool stops trading right after entry — classic rug/liquidity-
# pull behavior — no new bars ever arrive, so that function has nothing to
# scan and will return None forever. Without these, a position can get
# stuck reporting stale PnL indefinitely, which is what happened to OPAN.
STOP_LOSS_PCT      = -20.0   # force-exit if unrealized PnL drops to/below this
MAX_HOLD_BARS       = 60     # force-exit if held this many 1m bars (~1h) with
                               # no signal either way — most rug-driven
                               # collapses happen fast; a healthy setup
                               # shouldn't need an hour to resolve
STALE_POLL_LIMIT    = 2      # force-exit if this many consecutive polls in a
                               # row show ZERO new bars despite real time
                               # passing — the strongest direct signal that
                               # the pool has actually stopped trading
                               # (vs. just "still waiting for the pattern")


# ──────────────────────────────────────────────────────────────────────────
# BIRDEYE CLIENT (adaptive rate limit, same pattern used for GeckoTerminal)
# ──────────────────────────────────────────────────────────────────────────

def bd_get(path: str, params: dict = None) -> dict | None:
    global _bd_adaptive_delay, _bd_consecutive_ok, _bd_last_request_ts

    if not BIRDEYE_API_KEY:
        log.error("BIRDEYE_API_KEY is not set — export it and restart. "
                    "Get a key at https://bds.birdeye.so")
        return None

    url = f"{BIRDEYE_BASE}{path}"
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": BIRDEYE_CHAIN}

    for attempt in range(5):
        with _bd_rate_lock:
            elapsed = time.time() - _bd_last_request_ts
            if elapsed < _bd_adaptive_delay:
                time.sleep(_bd_adaptive_delay - elapsed)
            _bd_last_request_ts = time.time()

        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=20)
        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            log.warning("  network error (%s), sleeping %ds …", e, wait)
            time.sleep(wait)
            continue

        if r.status_code == 403:
            log.error("  Birdeye 403 on %s — likely a plan-tier restriction "
                        "(discovery endpoint needs Starter+). Check your key's "
                        "plan at https://bds.birdeye.so", path)
            return None
        if r.status_code == 401:
            log.error("  Birdeye 401 — API key missing or invalid.")
            return None
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 5 * (attempt + 1)
            with _bd_rate_lock:
                _bd_consecutive_ok = 0
                _bd_adaptive_delay = min(_bd_adaptive_delay * 1.6 + 1.0, 30.0)
                delay_now = _bd_adaptive_delay
            log.warning("  Birdeye rate-limited, sleeping %.0fs … (delay now %.1fs)",
                         wait, delay_now)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            wait = 5 * (attempt + 1)
            log.warning("  Birdeye server error %d, sleeping %ds …", r.status_code, wait)
            time.sleep(wait)
            continue

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.error("  Birdeye HTTP error: %s", e)
            return None

        with _bd_rate_lock:
            _bd_consecutive_ok += 1
            if _bd_consecutive_ok >= 5 and _bd_adaptive_delay > BD_DELAY_FLOOR:
                _bd_adaptive_delay = max(BD_DELAY_FLOOR, _bd_adaptive_delay * 0.85)
                _bd_consecutive_ok = 0

        return r.json()

    log.error("  giving up after retries: %s", url)
    return None


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# DISCOVERY — Birdeye Token List V3 (floors only)
# ──────────────────────────────────────────────────────────────────────────

def discover_candidates(top_n: int = TOP_N, pages: int = DISCOVERY_PAGES) -> pd.DataFrame:
    rows = []
    n_excluded = 0

    for page in range(pages):
        params = {
            "sort_by": "volume_24h_usd",
            "sort_type": "desc",
            "min_liquidity": MIN_LIQUIDITY_USD,
            "min_volume_24h_usd": MIN_VOLUME_24H_USD,
            "max_market_cap": MAX_MARKET_CAP_USD,
            "offset": page * 100,
            "limit": 100,   # Birdeye's hard per-call max
        }
        data = bd_get("/defi/v3/token/list", params=params)
        if data is None or not data.get("success"):
            log.warning("Discovery page %d failed or returned no data.", page)
            break

        items = (data.get("data") or {}).get("items") or (data.get("data") or {}).get("tokens") or []
        if not items:
            log.info("Discovery: page %d empty — reached the end of candidates "
                       "that clear the $%s liquidity / $%s volume floor.",
                       page, f"{MIN_LIQUIDITY_USD:,}", f"{MIN_VOLUME_24H_USD:,}")
            break

        for item in items:
            addr = item.get("address")
            sym = (item.get("symbol") or "?").upper()
            if not addr:
                continue
            if sym in EXCLUDED_SYMBOLS:
                n_excluded += 1
                continue
            rows.append({
                "symbol":         item.get("symbol") or "?",
                "address":        addr,
                "liquidity_usd":  _safe_float(item.get("liquidity")) or 0.0,
                "volume_24h":     _safe_float(item.get("volume_24h_usd") or item.get("v24hUSD")) or 0.0,
                "price_change_1h": _safe_float(item.get("price_change_1h_percent")) or 0.0,
            })

        if len(items) < 100:
            break  # short page = no more results, no point requesting another

    if n_excluded:
        log.info("Discovery: excluded %d stablecoin/wrapped-major symbol(s).", n_excluded)
    log.info("Discovery: %d total candidates found (capped at %d).", len(rows), top_n)
    return pd.DataFrame(rows).head(top_n)


def _fetch_discovery_page(page: int) -> tuple[list[dict], int]:
    """
    Single page (up to 100 tokens) of Birdeye's token list, floors only.
    Returns (rows, raw_item_count) — raw_item_count (pre-exclusion) is used
    by the caller to detect "ran out of universe" vs. "everything on this
    page got excluded".
    """
    params = {
        "sort_by": "volume_24h_usd",
        "sort_type": "desc",
        "min_liquidity": MIN_LIQUIDITY_USD,
        "min_volume_24h_usd": MIN_VOLUME_24H_USD,
        "max_market_cap": MAX_MARKET_CAP_USD,
        "offset": page * 100,
        "limit": 100,
    }
    data = bd_get("/defi/v3/token/list", params=params)
    if data is None or not data.get("success"):
        return [], 0

    items = (data.get("data") or {}).get("items") or (data.get("data") or {}).get("tokens") or []
    rows = []
    for item in items:
        addr = item.get("address")
        sym = (item.get("symbol") or "?").upper()
        if not addr or sym in EXCLUDED_SYMBOLS:
            continue
        rows.append({
            "symbol":          item.get("symbol") or "?",
            "address":         addr,
            "liquidity_usd":   _safe_float(item.get("liquidity")) or 0.0,
            "volume_24h":      _safe_float(item.get("volume_24h_usd") or item.get("v24hUSD")) or 0.0,
            "price_change_1h": _safe_float(item.get("price_change_1h_percent")) or 0.0,
        })
    return rows, len(items)


def _recent_pct_change(bars: pd.DataFrame, window_min: int) -> float | None:
    """
    Real % price change over the last `window_min` 1-minute bars, computed
    directly from OHLCV closes. Returns None if there isn't enough history
    yet (e.g. a token that just started trading).
    """
    if bars.empty or len(bars) < window_min + 1:
        return None
    closes = bars["close"].values
    start, end = closes[-(window_min + 1)], closes[-1]
    if start == 0:
        return None
    return (end - start) / start * 100.0


def rank_by_recent_volatility(safe: dict[str, dict],
                               window_min: int = VOLATILITY_WINDOW_MIN) -> list[dict]:
    """
    Final refinement pass over an already safety-gated candidate set: pull
    OHLCV once per candidate (bounded to len(safe), i.e. <= TARGET_CANDIDATES
    — never per raw discovery-page candidate) and re-rank by the REAL
    `window_min`-minute % change instead of Birdeye's coarse bulk-endpoint
    1h figure. Falls back to the 1h figure only if a candidate doesn't have
    enough bar history yet for the shorter window.
    """
    enriched = []
    for addr, row in safe.items():
        bars = fetch_ohlcv(addr)
        recent = _recent_pct_change(bars, window_min)
        row = dict(row)
        row["price_change_recent"] = recent if recent is not None else row.get("price_change_1h", 0.0)
        row["price_change_recent_is_fallback"] = recent is None
        enriched.append(row)
    enriched.sort(key=lambda r: abs(r["price_change_recent"]), reverse=True)
    return enriched


def discover_top_volatile_safe(target_n: int = TARGET_CANDIDATES,
                                max_pages: int = MAX_DISCOVERY_PAGES) -> pd.DataFrame:
    """
    Pages through Birdeye's floor-filtered token list, ranking everything
    seen so far by |1h price change| (most volatile first), and only runs
    the RugCheck/Jupiter safety gate on the top volatility-ranked, not-yet-
    checked candidates each page — cheapest-first, instead of gating the
    whole raw universe. Keeps paging until exactly `target_n` candidates
    have passed the safety gate, or `max_pages` pages have been scanned
    without reaching it (hard stop so a thin/rug-heavy universe can't turn
    into an unbounded API-call loop).
    """
    seen:    dict[str, dict] = {}   # address -> raw candidate row
    safe:    dict[str, dict] = {}   # address -> safety-passed row
    checked: set[str]        = set()
    page = 0

    for page in range(max_pages):
        rows, n_items = _fetch_discovery_page(page)
        for r in rows:
            seen[r["address"]] = r

        if n_items == 0:
            log.info("Discovery: page %d empty — reached end of "
                       "floor-qualifying universe.", page)
            break

        # Re-rank everything collected so far by volatility, then only
        # safety-gate as many of the top unchecked ones as we still need
        # (with a buffer, since not every candidate passes the gate) — but
        # hard-capped at MAX_CHECK_BATCH_SIZE regardless of how many are
        # "needed". Without this cap, an early page where `needed` is still
        # the full target_n (e.g. 30) multiplies out to 90 candidates in
        # one round with only SAFETY_WORKERS=2 workers hammering Jupiter,
        # which spirals its adaptive rate-limit delay to the 15s ceiling
        # and starves out candidates that would otherwise have passed.
        ranked = sorted(seen.values(), key=lambda r: abs(r["price_change_1h"]), reverse=True)
        needed = max(target_n - len(safe), 0)
        batch_size = min(needed * CHECK_BATCH_MULTIPLIER, MAX_CHECK_BATCH_SIZE)
        to_check = [r for r in ranked if r["address"] not in checked][:batch_size]

        if to_check:
            log.info("Discovery page %d: %d raw seen, %d safety-passed so far — "
                       "gating top %d unchecked (by |1h%%|) …",
                       page, len(seen), len(safe), len(to_check))
            gated = run_safety_gate(pd.DataFrame(to_check))
            for _, r in gated.iterrows():
                safe[r["address"]] = r.to_dict()
            checked.update(r["address"] for r in to_check)

        if len(safe) >= target_n:
            break
        if n_items < 100:
            log.info("Discovery: short page %d (%d items) — reached end of "
                       "floor-qualifying universe.", page, n_items)
            break

    if len(safe) < target_n:
        log.warning("Only found %d/%d safety-passed volatile candidate(s) after "
                     "scanning %d page(s) (%d raw candidates seen). Proceeding "
                     "with what's available — loosen MIN_LIQUIDITY_USD / "
                     "MIN_VOLUME_24H_USD, or raise MAX_DISCOVERY_PAGES, if you "
                     "need the full %d.",
                     len(safe), target_n, page + 1, len(seen), target_n)
    else:
        log.info("Discovery: locked in %d/%d safety-passed candidates after "
                   "%d page(s) — refining ranking with real %dm price change …",
                   len(safe), target_n, page + 1, VOLATILITY_WINDOW_MIN)

    ranked_safe = rank_by_recent_volatility(safe, VOLATILITY_WINDOW_MIN)
    final = ranked_safe[:target_n]
    for r in final:
        tag = " (1h fallback — not enough bar history yet)" if r["price_change_recent_is_fallback"] else ""
        log.info("  %-12s  %+.2f%% / %dm%s", r["symbol"], r["price_change_recent"],
                   VOLATILITY_WINDOW_MIN, tag)
    return pd.DataFrame(final)


def refresh_price_change(addresses: list[str]) -> dict[str, float]:
    """
    Cheap re-poll of the same discovery endpoint, used only to re-rank
    passive candidates by movement since the last full universe refresh.
    No separate snapshot endpoint needed — Birdeye's token list already
    returns price-change % per timeframe in one call.
    """
    df = discover_candidates(top_n=100, pages=1)
    if df.empty:
        return {}
    df = df[df["address"].isin(addresses)]
    return dict(zip(df["address"], df["price_change_1h"]))


# ──────────────────────────────────────────────────────────────────────────
# OHLCV
# ──────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(address: str) -> pd.DataFrame:
    now = int(time.time())
    time_from = now - OHLCV_LOOKBACK_MIN * 60
    params = {
        "address": address,
        "type": OHLCV_INTERVAL,
        "currency": "usd",
        "time_from": time_from,
        "time_to": now,
    }
    data = bd_get("/defi/ohlcv", params=params)
    if data is None or not data.get("success"):
        return pd.DataFrame()

    items = (data.get("data") or {}).get("items") or []
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        ts = it.get("unixTime") or it.get("unix_time")
        o  = it.get("o", it.get("open"))
        h  = it.get("h", it.get("high"))
        l  = it.get("l", it.get("low"))
        c  = it.get("c", it.get("close"))
        v  = it.get("v", it.get("volume"))
        if None in (ts, o, h, l, c, v):
            continue
        rows.append((ts, o, h, l, c, v))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.drop(columns=["ts"]).set_index("datetime").sort_index()


# ──────────────────────────────────────────────────────────────────────────
# SAFETY GATE — RugCheck (unchanged logic, just no pool→mint lookup needed
# anymore since Birdeye already gives us the token mint address directly)
# ──────────────────────────────────────────────────────────────────────────

def rugcheck_report(mint: str) -> dict | None:
    global _rc_adaptive_delay, _rc_consecutive_ok, _rc_last_request_ts

    url = f"{RUGCHECK_BASE}/tokens/{mint}/report"
    for attempt in range(4):
        with _rc_rate_lock:
            elapsed = time.time() - _rc_last_request_ts
            if elapsed < _rc_adaptive_delay:
                time.sleep(_rc_adaptive_delay - elapsed)
            _rc_last_request_ts = time.time()

        try:
            r = SESSION.get(url, timeout=15)
        except requests.exceptions.RequestException as e:
            log.warning("  rugcheck %s -> network error: %s", mint[:8], e)
            time.sleep(2 * (attempt + 1))
            continue

        if r.status_code == 429:
            wait = 3 * (attempt + 1)
            with _rc_rate_lock:
                _rc_consecutive_ok = 0
                _rc_adaptive_delay = min(_rc_adaptive_delay * 1.6 + 0.5, 15.0)
                delay_now = _rc_adaptive_delay
            log.warning("  rugcheck %s -> 429, sleeping %ds … (delay now %.1fs)",
                         mint[:8], wait, delay_now)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            log.warning("  rugcheck %s -> HTTP %d", mint[:8], r.status_code)
            return None

        with _rc_rate_lock:
            _rc_consecutive_ok += 1
            if _rc_consecutive_ok >= 5 and _rc_adaptive_delay > RC_DELAY_FLOOR:
                _rc_adaptive_delay = max(RC_DELAY_FLOOR, _rc_adaptive_delay * 0.85)
                _rc_consecutive_ok = 0

        return r.json()

    log.warning("  rugcheck %s -> giving up after retries (rate limit)", mint[:8])
    return None


def _lp_locked_pct(report: dict) -> float:
    try:
        markets = report.get("markets") or []
        if not markets:
            return 0.0
        total, locked = 0.0, 0.0
        for m in markets:
            lp = m.get("lp") or {}
            pct_locked = lp.get("lpLockedPct")
            if pct_locked is None:
                continue
            total += 1.0
            locked += float(pct_locked)
        return (locked / total) if total > 0 else 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


def check_honeypot_rugcheck(mint: str) -> tuple[bool, str]:
    report = rugcheck_report(mint)
    if report is None:
        return False, "rugcheck_unavailable"
    if report.get("rugged") is True:
        return False, "rugcheck_flagged_rugged"

    mint_auth   = report.get("mintAuthority")
    freeze_auth = report.get("freezeAuthority")
    if mint_auth not in (None, ""):
        return False, "mint_authority_not_renounced"
    if freeze_auth not in (None, ""):
        return False, "freeze_authority_active"

    return True, "ok"


def jup_quote(input_mint: str, output_mint: str, amount: int) -> dict | None:
    global _jup_adaptive_delay, _jup_consecutive_ok, _jup_last_request_ts

    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount,
        "slippageBps": 100,
    }

    for attempt in range(4):
        with _jup_rate_lock:
            elapsed = time.time() - _jup_last_request_ts
            if elapsed < _jup_adaptive_delay:
                time.sleep(_jup_adaptive_delay - elapsed)
            _jup_last_request_ts = time.time()

        try:
            r = SESSION.get(JUP_QUOTE_URL, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            log.warning("  jupiter %s -> network error: %s", input_mint[:8], e)
            time.sleep(2 * (attempt + 1))
            continue

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 3 * (attempt + 1)
            with _jup_rate_lock:
                _jup_consecutive_ok = 0
                _jup_adaptive_delay = min(_jup_adaptive_delay * 1.6 + 0.5, 15.0)
                delay_now = _jup_adaptive_delay
            log.warning("  jupiter rate-limited, sleeping %.0fs … (delay now %.1fs)",
                         wait, delay_now)
            time.sleep(wait)
            continue

        if r.status_code != 200:
            # genuine non-200, non-rate-limit failure (bad mint, 5xx, etc.) —
            # this is the only case that should actually count as "no route"
            return None

        with _jup_rate_lock:
            _jup_consecutive_ok += 1
            if _jup_consecutive_ok >= 5 and _jup_adaptive_delay > JUP_DELAY_FLOOR:
                _jup_adaptive_delay = max(JUP_DELAY_FLOOR, _jup_adaptive_delay * 0.85)
                _jup_consecutive_ok = 0

        return r.json()

    log.warning("  jupiter %s -> giving up after retries (rate limit)", input_mint[:8])
    return None


def check_jupiter_traps(mint: str) -> tuple[bool, str]:
    buy = jup_quote(USDC_MINT, mint, PROBE_USDC_AMOUNT)
    if not buy or "outAmount" not in buy:
        return False, "no_buy_route"
    out_amount = int(buy["outAmount"])
    if out_amount <= 0:
        return False, "zero_buy_output"

    sell = jup_quote(mint, USDC_MINT, out_amount)
    if not sell or "outAmount" not in sell:
        return False, "no_sell_route_honeypot"

    buy_impact  = float(buy.get("priceImpactPct", 0) or 0) * 100
    sell_impact = float(sell.get("priceImpactPct", 0) or 0) * 100
    if buy_impact > MAX_JUP_PRICE_IMPACT_PCT:
        return False, f"buy_price_impact({buy_impact:.1f}%)"
    if sell_impact > MAX_JUP_PRICE_IMPACT_PCT:
        return False, f"sell_price_impact({sell_impact:.1f}%)"

    sell_usdc_back = int(sell["outAmount"])
    round_trip_loss_pct = (1 - sell_usdc_back / PROBE_USDC_AMOUNT) * 100
    if round_trip_loss_pct > MAX_ROUND_TRIP_LOSS_PCT:
        return False, f"round_trip_loss({round_trip_loss_pct:.1f}%)"

    return True, "ok"


def _safety_check_one(row: dict) -> dict | None:
    sym, mint = row["symbol"], row["address"]

    rc_ok, rc_reason = check_honeypot_rugcheck(mint)
    if not rc_ok:
        log.info("  ✗ %-12s — RugCheck reject: %s", sym, rc_reason)
        return None

    jup_ok, jup_reason = check_jupiter_traps(mint)
    if not jup_ok:
        log.info("  ✗ %-12s — Jupiter reject: %s", sym, jup_reason)
        return None

    log.info("  ✓ %-12s — passed honeypot + Jupiter checks", sym)
    return row


def run_safety_gate(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    log.info("Running safety gate (RugCheck + Jupiter) on %d candidates (%d workers) …",
               len(candidates), SAFETY_WORKERS)
    rows = [r.to_dict() for _, r in candidates.iterrows()]
    safe_rows = []
    with ThreadPoolExecutor(max_workers=SAFETY_WORKERS) as pool:
        for result in pool.map(_safety_check_one, rows):
            if result is not None:
                safe_rows.append(result)
    safe_df = pd.DataFrame(safe_rows)
    log.info("Safety gate: %d / %d candidates passed.", len(safe_df), len(candidates))
    return safe_df


# ──────────────────────────────────────────────────────────────────────────
# LIVE ENTRY/EXIT — reuses backtest.py logic
# ──────────────────────────────────────────────────────────────────────────

def check_live_exit(bars: pd.DataFrame, entry_idx: int, structure_high: float):
    obv_full = B.compute_obv(bars)
    after    = bars.iloc[entry_idx:].copy()
    closes   = after['close'].values
    opens    = after['open'].values
    highs    = after['high'].values
    vols     = after['volume'].values
    obv_vals = obv_full.iloc[entry_idx:].values

    confirmed_highs = []
    for i in range(1, len(after)):
        if closes[i] > structure_high:
            return i, "case1_divergence_fulfilled"

        lo, hi = i - B.SWING_N, i + B.SWING_N + 1
        if lo >= 0 and hi <= len(after):
            window_h = highs[lo:hi]
            if highs[i] == window_h.max() and list(window_h).count(highs[i]) == 1:
                confirmed_highs.append((i, highs[i], obv_vals[i]))
            if len(confirmed_highs) >= 2:
                sh1, sh2 = confirmed_highs[-2], confirmed_highs[-1]
                if sh2[1] > sh1[1] and sh2[2] < sh1[2]:
                    return sh2[0], "case2_bearish_obv_divergence"

        vlo = max(0, i - B.BEARISH_VOL_LOOKBACK)
        avg_vol = vols[vlo:i].mean() if i > vlo else float("nan")
        is_red = closes[i] < opens[i]
        is_spike = (avg_vol == avg_vol) and avg_vol > 0 and vols[i] >= B.BEARISH_VOL_MULT * avg_vol
        if is_red and is_spike:
            return i, "case3_bearish_volume"

    return None


@dataclass
class TokenState:
    symbol: str
    address: str
    in_position: bool = False
    entry_idx_abs: int | None = None
    entry_price: float | None = None
    entry_time: object = None
    structure_high: float | None = None
    last_bar_count: int | None = None   # for stale-pool detection while held
    stale_poll_count: int = 0            # consecutive polls with 0 new bars


def poll_token(state: TokenState, lookback: int) -> None:
    bars = fetch_ohlcv(state.address)
    if bars.empty or len(bars) < lookback + 5:
        log.warning("  %-12s — insufficient bars (%d), skipping this poll",
                     state.symbol, len(bars))
        return
    n = len(bars)

    if not state.in_position:
        window = bars.iloc[n - lookback - 1:]
        fired, structure_high = B.check_entry(window)
        if fired:
            entry_idx_abs = n - 1
            state.in_position    = True
            state.entry_idx_abs  = entry_idx_abs
            state.entry_price    = float(bars.iloc[entry_idx_abs]['close'])
            state.entry_time     = bars.index[entry_idx_abs]
            state.structure_high = structure_high
            state.last_bar_count = n
            state.stale_poll_count = 0
            log.info("  🟢 ENTRY  %-12s @ %.8f  (reclaim target %.8f)  t=%s",
                      state.symbol, state.entry_price, structure_high, state.entry_time)
        return

    result = check_live_exit(bars, state.entry_idx_abs, state.structure_high)

    held_bars = n - 1 - state.entry_idx_abs
    unrealized = (bars.iloc[-1]['close'] - state.entry_price) / state.entry_price * 100

    # Stale-pool detection: did this poll actually see any new bars at all?
    if state.last_bar_count is not None and n <= state.last_bar_count:
        state.stale_poll_count += 1
    else:
        state.stale_poll_count = 0
    state.last_bar_count = n

    force_reason = None
    if result is None:
        if unrealized <= STOP_LOSS_PCT:
            force_reason = "stop_loss"
        elif held_bars >= MAX_HOLD_BARS:
            force_reason = "max_hold_timeout"
        elif state.stale_poll_count >= STALE_POLL_LIMIT:
            force_reason = "pool_appears_dead_no_new_bars"

    if result is None and force_reason is None:
        log.info("  …   HOLD   %-12s  unrealized %+0.2f%%  (held %d bars)",
                  state.symbol, unrealized, held_bars)
        return

    if force_reason is not None:
        exit_price = float(bars.iloc[-1]['close'])
        pnl_pct = unrealized
        log.warning("  🛑 FORCE EXIT  %-12s @ %.8f  pnl=%+0.2f%%  reason=%s  t=%s",
                     state.symbol, exit_price, pnl_pct, force_reason, bars.index[-1])
    else:
        rel_idx, reason = result
        exit_idx_abs = state.entry_idx_abs + rel_idx
        exit_price = float(bars.iloc[exit_idx_abs]['close'])
        pnl_pct = (exit_price - state.entry_price) / state.entry_price * 100
        log.info("  🔴 EXIT   %-12s @ %.8f  pnl=%+0.2f%%  reason=%s  t=%s",
                  state.symbol, exit_price, pnl_pct, reason, bars.index[exit_idx_abs])

    state.in_position    = False
    state.entry_idx_abs  = None
    state.entry_price    = None
    state.entry_time     = None
    state.structure_high = None
    state.last_bar_count = None
    state.stale_poll_count = 0


# ──────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookback', type=int, default=B.LOOKBACK)
    args = parser.parse_args()

    if not BIRDEYE_API_KEY:
        log.error("BIRDEYE_API_KEY not set. export BIRDEYE_API_KEY=... and re-run.")
        return

    states: dict[str, TokenState] = {}
    last_universe_refresh = 0.0
    last_safety_refresh = 0.0

    log.info("Starting Birdeye live screener. Single active position at a time.")

    while True:
        now = time.time()

        if now - last_universe_refresh > UNIVERSE_REFRESH_SECONDS or not states:
            safe = discover_top_volatile_safe(target_n=TARGET_CANDIDATES,
                                                max_pages=MAX_DISCOVERY_PAGES)
            if safe.empty:
                log.warning("No candidates passed discovery+safety gate this cycle. "
                             "Sleeping and retrying.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            new_states = {}
            for _, row in safe.iterrows():
                addr = row["address"]
                new_states[addr] = states.get(addr) or TokenState(symbol=row["symbol"], address=addr)
            states = new_states
            last_universe_refresh = now
            last_safety_refresh = now

        elif now - last_safety_refresh > SAFETY_REFRESH_SECONDS:
            df = pd.DataFrame([{"symbol": s.symbol, "address": s.address} for s in states.values()])
            recheck = run_safety_gate(df)
            still_safe = set(recheck["address"]) if not recheck.empty else set()
            for addr in list(states.keys()):
                if addr not in still_safe and not states[addr].in_position:
                    log.warning("  %s failed safety re-check — dropping", states[addr].symbol)
                    del states[addr]
                elif addr not in still_safe:
                    log.warning("  %s failed safety re-check but is OPEN — tracking to exit only",
                                 states[addr].symbol)
            last_safety_refresh = now

        if not states:
            log.warning("No safe candidates to scan. Sleeping.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        held = next((s for s in states.values() if s.in_position), None)

        if held is not None:
            log.info("── poll: 1 token held (%s), not scanning for new entries ──", held.symbol)
            try:
                poll_token(held, args.lookback)
            except Exception as e:
                log.error("  %-12s — poll error: %s", held.symbol, e)
            time.sleep(max(0.0, POLL_INTERVAL_SECONDS - _bd_adaptive_delay))
            continue

        passive = list(states.values())
        passive_to_poll = passive
        if len(passive) > PASSIVE_BATCH_SIZE:
            change_map = refresh_price_change([p.address for p in passive])
            ranked = sorted(passive, key=lambda s: abs(change_map.get(s.address, 0.0)), reverse=True)
            passive_to_poll = ranked[:PASSIVE_BATCH_SIZE]
            skipped = [s.symbol for s in ranked[PASSIVE_BATCH_SIZE:]]
            if skipped:
                log.info("  (skipped this cycle — no notable h1 move: %s)", ", ".join(skipped))

        log.info("── poll: 0 in position, scanning %d passive token(s) ──", len(passive_to_poll))

        for state in passive_to_poll:
            try:
                poll_token(state, args.lookback)
            except Exception as e:
                log.error("  %-12s — poll error: %s", state.symbol, e)
            if state.in_position:
                log.info("  entry fired on %s — stopping passive scan for this cycle", state.symbol)
                break

        time.sleep(max(0.0, POLL_INTERVAL_SECONDS - len(passive_to_poll) * _bd_adaptive_delay))


if __name__ == "__main__":
    main()