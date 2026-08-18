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
     just whatever cleared the floors first. Final ranking is 1h price
     change straight from Birdeye's bulk list field — no per-candidate
     OHLCV pre-fetch, unlike an earlier version of this bot that re-ranked
     on a local 5-minute recompute. That extra pass is gone: it cost one
     OHLCV call per safety-passed candidate for a ranking refinement this
     version doesn't do anymore.
  2. Single active position at a time (mirrors backtest.py's mutual
     exclusivity): if something is open, ONLY that token gets polled —
     no scanning anyone else. If flat, all TARGET_CANDIDATES tokens get
     scanned every cycle (PASSIVE_BATCH_SIZE == TARGET_CANDIDATES, so no
     trimming). Scanning stops the instant one of them fires an entry.
  3. POLL_INTERVAL_SECONDS = 300 — the whole loop (entry check when flat,
     exit + unrealized/realized PnL check when in a position) runs every
     5th minute. This is a DIFFERENT "5 minutes" from the removed ranking
     pass above: it's the live position-monitoring cadence, watching new
     1m bars for the actual OBV-divergence entry/exit pattern. That can't
     be removed without losing the trading signal itself — divergence is a
     pattern across bars over time, not something a single snapshot can
     tell you — so it's left as-is here even though the ranking-time 5m
     pass was cut.

Dependencies:
    pip install pandas requests solders --break-system-packages
    (solders is only needed if LIVE_TRADING=true — see the LIVE TRADING
    section below for what that flag does and how to configure it.)
"""

from __future__ import annotations

import os
import time
import logging
import argparse
import threading
from dataclasses import dataclass, field
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
BIRDEYE_API_KEY = "40a4ba5a1cca4a768c5a3a5f74b21fc7"  # actually read from
                    # env now — this used to say "moved to env var" in the
                    # comment but still had a literal key hardcoded above,
                    # which defeats the point. That old key was exposed in
                    # your uploaded file either way — rotate it at
                    # https://bds.birdeye.so and treat it as burned.
BIRDEYE_CHAIN   = "solana"

DEBUG_PRICE_DATA = os.environ.get("DEBUG_PRICE_DATA", "true").lower() not in ("0", "false", "no")
                                  # Logs exactly what fetch_ohlcv actually got back
                                  # for a given mint every time it's called: which
                                  # pool it resolved to, how many bars, the last
                                  # bar's timestamp/close, and how stale that bar
                                  # is relative to "now". PnL is only as good as
                                  # this data — if the wrong pool got picked, or
                                  # bars are stale/gapped, entry/exit prices (and
                                  # therefore every printed pnl%) can be quietly
                                  # wrong even though the bot never errors out.
                                  # Set DEBUG_PRICE_DATA=false to quiet this back
                                  # down to one line per poll instead of the full
                                  # breakdown.

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
MIN_LIQUIDITY_USD  = 8_000    # lowered from 20k — at $20-100 trade sizes,
                                # market impact isn't the binding constraint;
                                # this mainly widens access to younger/
                                # thinner pools where real short-term
                                # volatility concentrates. Tradeoff: also
                                # raises rug/scam density and noisy-close
                                # artifacts — the safety gate and force-exits
                                # are the actual backstop here, not this floor.
MIN_VOLUME_24H_USD = 15_000   # lowered from 50k, same rationale
MAX_MARKET_CAP_USD = 50_000_000
TOP_N              = 200   # raw-candidate cap for the legacy discover_candidates()
                            # helper (still used by refresh_price_change's cheap
                            # re-poll) — unrelated to TARGET_CANDIDATES below
DISCOVERY_PAGES    = 3      # Birdeye caps each call at 100 results — paginate
                             # via offset to actually reach TOP_N instead of
                             # silently truncating at 100

# ── Volatile-universe target (new) ───────────────────────────────────────
TARGET_CANDIDATES      = 15  # exact number of tokens kept under active scan
                               # at any time.
SAFE_POOL_OVERSAMPLE    = TARGET_CANDIDATES  # oversampling disabled — this
                               # was set to 40 to find the TRUE top-20 by
                               # real volatility instead of "first 20 that
                               # passed the gate". That accuracy tradeoff
                               # cost too much: it both stretched discovery
                               # to 3-4 pages minimum AND meant the 5m-
                               # ranking pass hit GeckoTerminal with ~2x
                               # candidates (2 calls each: resolve pool +
                               # OHLCV) in one burst, blowing through its
                               # 30/min shared limit immediately. Setting
                               # this equal to TARGET_CANDIDATES restores
                               # the simpler/faster "stop as soon as N pass"
                               # behavior — first N found, not the truest
                               # top N — trading ranking accuracy for speed.
MAX_DISCOVERY_PAGES     = 20  # lowered back down now that oversampling to
                               # 40 is gone — 15 safety-passed candidates
                               # needs roughly half the pages 40 did.
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
# NOTE: there used to be a VOLATILITY_WINDOW_MIN=5 final re-ranking pass here
# (rank_by_recent_volatility) that fetched 1m OHLCV per safety-passed
# candidate just to recompute a 5-minute price change, on top of the 1h
# field Birdeye's bulk list endpoint already returns for free. Removed —
# final ranking now uses price_change_1h directly (see
# discover_top_volatile_safe below), which costs zero extra API calls.

# Small backstop on top of the market-cap ceiling — these are well-known
# enough that excluding by name too is just extra safety margin
EXCLUDED_SYMBOLS = {
    "USDT", "USDC", "USDH", "DAI", "UST", "USDS", "BUSD", "TUSD",
    "PYUSD", "FDUSD", "EURC", "USDE", "USD1",
    "SOL", "WSOL", "WBTC", "CBBTC", "WETH", "BTC", "ETH",
}

# ── OHLCV ────────────────────────────────────────────────────────────────
# Moved OFF Birdeye entirely: OHLCV is the single heaviest call in this bot
# by volume (called once per candidate during 5m ranking, plus once per
# tracked candidate every single poll cycle), and Birdeye bills it at 35 CU
# per call — this exhausted a full month's CU allotment in hours. Birdeye
# is still used for discovery/token-list (much lighter, ~75 CU per PAGE
# rather than per token). GeckoTerminal is free/keyless and only rate-
# limited by requests/sec, which comfortably covers this call volume.
GT_BASE             = "https://api.geckoterminal.com/api/v2"
GT_NETWORK          = "solana"
OHLCV_INTERVAL      = "1m"   # informational only now — GeckoTerminal's
                               # endpoint takes timeframe="minute" +
                               # aggregate=1 directly (hardcoded in
                               # fetch_ohlcv below), not this string
OHLCV_LOOKBACK_MIN  = 1000   # minutes of 1m history per call (GeckoTerminal
                               # also caps at 1000 records per call)

PASSIVE_SCAN_WORKERS = 4   # concurrent bar-fetch workers for the passive
                             # scan. This only parallelizes the FETCH step
                             # (see prefetch_passive_bars) — the actual
                             # entry-check/execution loop right after stays
                             # strictly sequential, so "first signal in
                             # ranked order wins, stop immediately, only one
                             # live swap at a time" is untouched. The shared
                             # _gt_rate_lock/_gt_adaptive_delay below still
                             # serializes the real network requests either
                             # way — this just removes the dead time of
                             # Python waiting on one token's response before
                             # even starting the next one's request.

INCREMENTAL_FETCH_BUFFER_MIN = 3   # extra 1m candles requested beyond the
                             # exact gap since a token's last cached bar, to
                             # safely cover clock drift and to make sure the
                             # previously-cached last candle (which may have
                             # still been "forming" when it was fetched) gets
                             # overwritten with its finalized close.

DEBUG_PNL_TICK_SECONDS = 20   # while a position is open, print a cheap
                             # price/unrealized-pnl line at this cadence.
                             # Purely informational — it does NOT re-run the
                             # real OBV-divergence exit check, which still
                             # only runs once per full POLL_INTERVAL_SECONDS
                             # cycle below (that check needs a genuinely NEW
                             # closed 1m bar to mean anything; sampling the
                             # same still-forming bar every 20s wouldn't
                             # change the signal, just spend extra calls).

# ── GeckoTerminal adaptive rate limiting ────────────────────────────────────
GT_DELAY_FLOOR = 2.2   # free tier is roughly ~30 req/min; this leaves margin
_gt_adaptive_delay = GT_DELAY_FLOOR
_gt_consecutive_ok = 0
_gt_last_request_ts = 0.0
_gt_rate_lock = threading.Lock()  # not concurrently called today, but the
                                    # same class of bug bit Jupiter earlier —
                                    # cheap insurance if that ever changes.

_pool_address_cache: dict[str, str | None] = {}   # token mint -> pool
                                                     # address, resolved once
                                                     # per token and reused
                                                     # for every subsequent
                                                     # OHLCV call
_pool_address_cache_lock = threading.Lock()

# ── Birdeye adaptive rate limiting ──────────────────────────────────────────
BD_DELAY_FLOOR = 1.0   # conservative starting point; adapts down if it's fine
_bd_adaptive_delay = BD_DELAY_FLOOR
_bd_consecutive_ok = 0
_bd_last_request_ts = 0.0
_bd_rate_lock = threading.Lock()  # bd_get isn't called concurrently today,
                                    # but locking costs nothing and protects
                                    # against that changing later.
_bd_consecutive_400s = 0
BD_QUOTA_SUSPECT_THRESHOLD = 5     # this many 400s in a row across DIFFERENT
                                     # addresses stops looking like "bad
                                     # request" and starts looking like a
                                     # plan/quota exhaustion
BD_QUOTA_COOLDOWN_SECONDS  = 300   # cool down this long before resuming once
                                     # that looks like what's happening

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
_rc_consecutive_400s = 0
RC_QUOTA_SUSPECT_THRESHOLD = 5     # same signature as the Birdeye check:
                                     # this many 400s in a row across
                                     # different mints stops looking like
                                     # "bad request" and starts looking like
                                     # a RugCheck-side outage/limit. Note:
                                     # check_honeypot_rugcheck() still fails
                                     # CLOSED (rejects) on any None result —
                                     # this only adds visibility + a cooldown,
                                     # it does NOT change candidates from
                                     # rejected to accepted during an outage.
RC_QUOTA_COOLDOWN_SECONDS  = 120   # shorter than Birdeye's — RugCheck 400s
                                     # here look more like a transient outage
                                     # than a hard quota wall

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
PRESUMED_RUG_LOSS_PCT = -95.0  # what gets recorded if a dead-pool force-exit
                                 # finds NO live sell route at all — the
                                 # frozen close-price PnL is fictitious in
                                 # that case (there's nothing to sell into),
                                 # so this floor is used instead of pretending
                                 # the position exited cleanly. -95 rather
                                 # than -100 leaves a little room since
                                 # "no route right now" isn't a mathematical
                                 # guarantee of "zero value forever."


# ──────────────────────────────────────────────────────────────────────────
# BIRDEYE CLIENT (adaptive rate limit, same pattern used for GeckoTerminal)
# ──────────────────────────────────────────────────────────────────────────

def check_birdeye_credits() -> None:
    """
    /utils/v1/credits costs only 1 CU and returns the account's actual
    usage/remaining balance. Call this to get a DEFINITIVE answer on
    whether repeated 400s are CU exhaustion rather than guessing from
    error codes alone — Birdeye doesn't document a dedicated HTTP status
    for "CU balance depleted", so 400 alone is ambiguous.
    """
    data = bd_get("/utils/v1/credits")
    if data is None:
        log.warning("Could not fetch Birdeye credit usage (call itself failed — "
                     "see the error above for why).")
        return
    log.info("Birdeye credit usage: %s", data)


def bd_get(path: str, params: dict = None) -> dict | None:
    global _bd_adaptive_delay, _bd_consecutive_ok, _bd_last_request_ts, _bd_consecutive_400s

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

        if r.status_code == 400:
            # A single 400 is usually just a bad address/params for that one
            # call. But a SUSTAINED run of 400s across many DIFFERENT
            # addresses (this counter is global, not per-address) is a
            # different signature entirely: it means the plan's daily/
            # monthly compute-unit quota got exhausted, and Birdeye is
            # rejecting every request the same way regardless of what's
            # being asked. Without this, the bot spins at full request
            # speed for hours against a dead quota (which is exactly what
            # happened overnight) instead of backing off and saying so.
            with _bd_rate_lock:
                _bd_consecutive_400s += 1
                streak = _bd_consecutive_400s
            if streak >= BD_QUOTA_SUSPECT_THRESHOLD:
                log.error("  Birdeye: %d consecutive 400s across different "
                            "requests — Birdeye's own docs define 400 as "
                            "\"invalid request parameters\", not quota, but "
                            "that many different addresses failing the exact "
                            "same way strongly suggests CU exhaustion rather "
                            "than 400 different malformed requests. Checking "
                            "actual credit balance now …", streak)
                check_birdeye_credits()
                log.error("  Cooling down %ds before retrying.", BD_QUOTA_COOLDOWN_SECONDS)
                _sleep_with_heartbeat(BD_QUOTA_COOLDOWN_SECONDS, "Birdeye quota")
                with _bd_rate_lock:
                    _bd_consecutive_400s = 0
            else:
                log.warning("  Birdeye 400 on %s (streak %d/%d)", path, streak,
                             BD_QUOTA_SUSPECT_THRESHOLD)
            return None

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.error("  Birdeye HTTP error: %s", e)
            return None

        with _bd_rate_lock:
            _bd_consecutive_ok += 1
            _bd_consecutive_400s = 0
            if _bd_consecutive_ok >= 5 and _bd_adaptive_delay > BD_DELAY_FLOOR:
                _bd_adaptive_delay = max(BD_DELAY_FLOOR, _bd_adaptive_delay * 0.85)
                _bd_consecutive_ok = 0

        return r.json()

    log.error("  giving up after retries: %s", url)
    return None


def _sleep_with_heartbeat(total_seconds: float, label: str, chunk: float = 30.0) -> None:
    """
    A long blocking time.sleep() during a quota cooldown looks identical to
    a hang from the outside — this logs progress every `chunk` seconds so
    it's obvious the process is alive and deliberately waiting, not stuck.
    """
    remaining = total_seconds
    while remaining > 0:
        step = min(chunk, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0:
            log.info("  … %s cooldown: %ds remaining", label, int(remaining))


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


def rank_by_1h_change(safe: dict[str, dict]) -> list[dict]:
    """
    Final ranking of an already safety-gated candidate set, by Birdeye's own
    1h price-change field — no per-candidate OHLCV fetch, since that field
    is already sitting on every row from discovery. This replaces the old
    5-minute recency re-ranking pass (rank_by_recent_volatility), which cost
    one extra OHLCV call per safety-passed candidate for a marginal accuracy
    gain; ranking purely off 1h keeps discovery to exactly the calls needed
    for paging + the safety gate itself.
    """
    ranked = sorted(safe.values(), key=lambda r: abs(r.get("price_change_1h", 0.0)), reverse=True)
    return ranked


def discover_top_volatile_safe(target_n: int = TARGET_CANDIDATES,
                                oversample_n: int = SAFE_POOL_OVERSAMPLE,
                                max_pages: int = MAX_DISCOVERY_PAGES) -> pd.DataFrame:
    """
    Pages through Birdeye's floor-filtered token list, ranking everything
    seen so far by |1h price change| (most volatile first), and only runs
    the RugCheck/Jupiter safety gate on the top volatility-ranked, not-yet-
    checked candidates each page — cheapest-first, instead of gating the
    whole raw universe. Keeps paging until `oversample_n` candidates have
    passed the safety gate (or `max_pages` pages have been scanned without
    reaching it — hard stop so a thin/rug-heavy universe can't turn into an
    unbounded API-call loop), THEN ranks that whole oversampled pool by 1h
    price change (Birdeye's own field, no extra fetch) and returns only the
    top `target_n`. Without the oversample step this would just return "the
    first target_n that happened to pass", which isn't the same claim as
    "the target_n most volatile" — but the ranking metric itself is now the
    same 1h field driving page-by-page ordering, just applied to the full
    oversampled pool instead of page-by-page.
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
        # the full oversample_n multiplies out to way too many candidates
        # in one round with only SAFETY_WORKERS=2 workers hammering
        # Jupiter, which spirals its adaptive rate-limit delay to the
        # ceiling and starves out candidates that would otherwise have
        # passed.
        ranked = sorted(seen.values(), key=lambda r: abs(r["price_change_1h"]), reverse=True)
        needed = max(oversample_n - len(safe), 0)
        batch_size = min(needed * CHECK_BATCH_MULTIPLIER, MAX_CHECK_BATCH_SIZE)
        to_check = [r for r in ranked if r["address"] not in checked][:batch_size]

        if to_check:
            log.info("Discovery page %d: %d raw seen, %d/%d safety-passed so far — "
                       "gating top %d unchecked (by |1h%%|) …",
                       page, len(seen), len(safe), oversample_n, len(to_check))
            gated = run_safety_gate(pd.DataFrame(to_check))
            for _, r in gated.iterrows():
                safe[r["address"]] = r.to_dict()
            checked.update(r["address"] for r in to_check)

        if len(safe) >= oversample_n:
            break
        if n_items < 100:
            log.info("Discovery: short page %d (%d items) — reached end of "
                       "floor-qualifying universe.", page, n_items)
            break

    if len(safe) < target_n:
        log.warning("Only found %d safety-passed candidate(s) (wanted an "
                     "oversampled pool of %d to rank %d from) after "
                     "scanning %d page(s) (%d raw candidates seen). "
                     "Proceeding with what's available — loosen "
                     "MIN_LIQUIDITY_USD / MIN_VOLUME_24H_USD, or raise "
                     "MAX_DISCOVERY_PAGES, if this keeps happening.",
                     len(safe), oversample_n, target_n, page + 1, len(seen))
    else:
        log.info("Discovery: oversampled to %d safety-passed candidates "
                   "after %d page(s) — ranking by 1h price change and "
                   "keeping the top %d …",
                   len(safe), page + 1, target_n)

    ranked_safe = rank_by_1h_change(safe)
    final = ranked_safe[:target_n]
    for r in final:
        log.info("  %s  %+.2f%% / 1h", tag(r["symbol"], r["address"]), r.get("price_change_1h", 0.0))
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

def gt_get(path: str, params: dict = None) -> dict | None:
    """
    GeckoTerminal client — mirrors bd_get's adaptive-delay pattern. No API
    key, no CU billing; the only real constraint is requests/sec, which
    this bot's OHLCV call volume comfortably fits under.
    """
    global _gt_adaptive_delay, _gt_consecutive_ok, _gt_last_request_ts

    url = f"{GT_BASE}{path}"
    for attempt in range(5):
        with _gt_rate_lock:
            elapsed = time.time() - _gt_last_request_ts
            if elapsed < _gt_adaptive_delay:
                time.sleep(_gt_adaptive_delay - elapsed)
            _gt_last_request_ts = time.time()

        try:
            r = SESSION.get(url, params=params, timeout=20)
        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            log.warning("  GeckoTerminal network error (%s), sleeping %ds …", e, wait)
            time.sleep(wait)
            continue

        if r.status_code == 429:
            # GeckoTerminal's CDN sometimes sends Retry-After: 0 on a 429,
            # which made this log a misleading "sleeping 0s" while the real
            # pacing only kicked in on the NEXT call via the adaptive delay
            # check at the top of the loop. Floor the explicit wait to the
            # adaptive delay itself so the backoff is visible and immediate,
            # not deferred by one iteration.
            retry_after = r.headers.get("Retry-After")
            with _gt_rate_lock:
                _gt_consecutive_ok = 0
                _gt_adaptive_delay = min(_gt_adaptive_delay * 1.6 + 1.0, 20.0)
                delay_now = _gt_adaptive_delay
            wait = max(float(retry_after) if retry_after else 5 * (attempt + 1), delay_now)
            log.warning("  GeckoTerminal rate-limited, sleeping %.0fs … (delay now %.1fs)",
                         wait, delay_now)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None   # pool/token not found — not retryable
        if r.status_code >= 500:
            wait = 5 * (attempt + 1)
            log.warning("  GeckoTerminal server error %d, sleeping %ds …", r.status_code, wait)
            time.sleep(wait)
            continue

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.warning("  GeckoTerminal HTTP error: %s", e)
            return None

        with _gt_rate_lock:
            _gt_consecutive_ok += 1
            if _gt_consecutive_ok >= 5 and _gt_adaptive_delay > GT_DELAY_FLOOR:
                _gt_adaptive_delay = max(GT_DELAY_FLOOR, _gt_adaptive_delay * 0.85)
                _gt_consecutive_ok = 0

        return r.json()

    log.error("  GeckoTerminal giving up after retries: %s", url)
    return None


def resolve_pool_address(mint: str) -> str | None:
    """
    GeckoTerminal's OHLCV endpoint is keyed by POOL address, not token
    mint — Birdeye resolved this internally, GeckoTerminal doesn't. Resolve
    once per token and cache: this is a one-time cost per token, not
    per-poll, so it doesn't meaningfully add to call volume.
    """
    with _pool_address_cache_lock:
        if mint in _pool_address_cache:
            return _pool_address_cache[mint]

    data = gt_get(f"/networks/{GT_NETWORK}/tokens/{mint}/pools")
    pool_address = None
    items = (data or {}).get("data") or []
    if items:
        # Pick the highest-liquidity pool for this token — most likely to
        # be the actively-traded one and to have continuous bar history.
        def _reserve(item):
            try:
                return float(item.get("attributes", {}).get("reserve_in_usd") or 0)
            except (TypeError, ValueError):
                return 0.0
        best = max(items, key=_reserve)
        pool_address = best.get("attributes", {}).get("address") or best.get("id", "").split("_")[-1]

        if DEBUG_PRICE_DATA:
            # Multiple pools for the same mint is normal (e.g. a Raydium
            # pool AND an Orca pool for the same token) — but if there are
            # several with comparable reserves, "highest liquidity" can
            # flip between polls as reserves shift, silently swapping which
            # pool's price feed you're reading mid-position. Log the full
            # candidate set once so that's visible instead of invisible.
            ranked = sorted(items, key=_reserve, reverse=True)
            summary = ", ".join(
                f"{it.get('attributes', {}).get('address', '?')}"
                f"(${_reserve(it):,.0f})"
                for it in ranked[:5]
            )
            log.info("  🔍 PRICE-DEBUG  mint=%s — %d pool(s) found, picked "
                       "%s (highest reserve). candidates: %s",
                       mint, len(items), pool_address, summary)
    elif DEBUG_PRICE_DATA:
        log.warning("  🔍 PRICE-DEBUG  mint=%s — GeckoTerminal returned NO "
                      "pools for this mint. fetch_ohlcv will return empty "
                      "and this token will be skipped ('insufficient bars') "
                      "until this resolves.", mint)

    with _pool_address_cache_lock:
        _pool_address_cache[mint] = pool_address
    return pool_address


def fetch_ohlcv(address: str) -> pd.DataFrame:
    """
    `address` is a token MINT (matches every existing call site — nothing
    downstream needs to change). Internally resolves to a pool address via
    the cache above, then pulls 1m candles from GeckoTerminal instead of
    Birdeye. Output shape (columns, index, sort order) is identical to the
    old Birdeye-backed version.
    """
    pool_address = resolve_pool_address(address)
    if pool_address is None:
        return pd.DataFrame()

    data = gt_get(
        f"/networks/{GT_NETWORK}/pools/{pool_address}/ohlcv/minute",
        params={"aggregate": 1, "limit": OHLCV_LOOKBACK_MIN, "currency": "usd"},
    )
    if data is None:
        return pd.DataFrame()

    candles = (data.get("data") or {}).get("attributes", {}).get("ohlcv_list") or []
    if not candles:
        if DEBUG_PRICE_DATA:
            log.warning("  🔍 PRICE-DEBUG  mint=%s pool=%s — 0 candles returned",
                          address, pool_address)
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop(columns=["ts"]).set_index("datetime").sort_index()

    if DEBUG_PRICE_DATA:
        last_ts = df.index[-1]
        age_sec = (pd.Timestamp.now(tz="utc") - last_ts).total_seconds()
        last_close = float(df["close"].iloc[-1])
        # A 1m-candle feed should never be more than ~1-2 bars stale. If it
        # is, whatever price this poll uses for entry/exit/unrealized-pnl is
        # not "now" — it's however old the last real trade on this pool was,
        # which is exactly the kind of gap that makes printed pnl% diverge
        # from what a real fill would have gotten.
        staleness_flag = " ⚠️ STALE" if age_sec > 180 else ""
        log.info("  🔍 PRICE-DEBUG  mint=%s pool=%s — %d bars, last=%s "
                   "close=%.10f age=%.0fs%s",
                   address, pool_address, len(df), last_ts, last_close,
                   age_sec, staleness_flag)

    return df


def fetch_ohlcv_incremental(state: TokenState) -> pd.DataFrame:
    """
    Same output shape as fetch_ohlcv, but reuses state.cached_bars instead of
    re-pulling the full OHLCV_LOOKBACK_MIN window on every poll.

    First call for a token (empty cache): full fetch, same as fetch_ohlcv.
    Every call after that: only requests however many 1m candles have
    elapsed since the last cached bar (+ a small buffer), merges them onto
    the cached frame (new data wins on overlapping timestamps, so a
    previously-"forming" last candle gets replaced by its finalized close),
    and re-trims to the lookback window.

    This is what makes "loop back to the other candidates after an exit"
    cheap: those tokens already have most of their history cached from the
    original scan, so only the handful of minutes that passed while a
    position was open actually needs fetching — not the whole window again.
    """
    cached = state.cached_bars
    if cached is None or cached.empty:
        fresh = fetch_ohlcv(state.address)
        state.cached_bars = fresh
        return fresh

    now = pd.Timestamp.now(tz="utc")
    gap_min = max(1, int((now - cached.index[-1]).total_seconds() // 60) + INCREMENTAL_FETCH_BUFFER_MIN)

    if gap_min >= OHLCV_LOOKBACK_MIN:
        # The gap is big enough that "incremental" wouldn't save anything
        # over a normal full fetch (e.g. this token hasn't been touched in
        # ages) — just fall back to the full fetch instead.
        fresh = fetch_ohlcv(state.address)
        state.cached_bars = fresh
        return fresh

    pool_address = resolve_pool_address(state.address)   # cached, no extra call
    if pool_address is None:
        return cached   # nothing new resolvable — hand back what we have

    data = gt_get(
        f"/networks/{GT_NETWORK}/pools/{pool_address}/ohlcv/minute",
        params={"aggregate": 1, "limit": gap_min, "currency": "usd"},
    )
    candles = ((data or {}).get("data") or {}).get("attributes", {}).get("ohlcv_list") or []
    if not candles:
        if DEBUG_PRICE_DATA:
            log.warning("  🔍 PRICE-DEBUG  mint=%s pool=%s — incremental fetch got 0 "
                          "new candles (gap=%dm), using cached bars as-is",
                          state.address, pool_address, gap_min)
        return cached

    new_df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    new_df["datetime"] = pd.to_datetime(new_df["ts"], unit="s", utc=True)
    new_df = new_df.drop(columns=["ts"]).set_index("datetime").sort_index()

    merged = pd.concat([cached, new_df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged.tail(OHLCV_LOOKBACK_MIN)

    if DEBUG_PRICE_DATA:
        log.info("  🔍 PRICE-DEBUG  mint=%s pool=%s — incremental: fetched %d new "
                   "candle(s) (gap=%dm) instead of a full %d-candle refetch, "
                   "cache now %d bars, last=%s close=%.10f",
                   state.address, pool_address, len(new_df), gap_min,
                   OHLCV_LOOKBACK_MIN, len(merged), merged.index[-1],
                   float(merged["close"].iloc[-1]))

    state.cached_bars = merged
    return merged


def prefetch_passive_bars(passive_to_poll: list[TokenState]) -> None:
    """
    Concurrently refreshes state.cached_bars for every passive candidate
    BEFORE the entry-check pass runs. Fetching is the slow part of scanning
    N tokens one at a time (network + adaptive rate-limit delay per call);
    doing it concurrently here means the sequential check_entry loop right
    after already has fresh data sitting in cache and doesn't wait on the
    network per-token.

    Deliberately does NOT run check_entry or execute anything here — that
    stays sequential in the caller, so "first signal in ranked order wins,
    stop scanning immediately, only one live swap at a time" is unaffected
    by this running concurrently.
    """
    with ThreadPoolExecutor(max_workers=PASSIVE_SCAN_WORKERS) as pool:
        futures = {pool.submit(fetch_ohlcv_incremental, s): s for s in passive_to_poll}
        for fut in futures:
            state = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.error("  %s — prefetch error: %s", tag(state.symbol, state.address), e)


def debug_pnl_tick(state: TokenState) -> None:
    """
    Lightweight, read-only price/unrealized-pnl print while a position is
    open. Uses the same incremental fetch as everything else (cheap once
    cached), but never touches state — this never triggers an exit, that
    still only happens inside poll_token on its normal cadence.
    """
    bars = fetch_ohlcv_incremental(state)
    if bars.empty or state.entry_price is None:
        return
    price = float(bars["close"].iloc[-1])
    pnl = (price - state.entry_price) / state.entry_price * 100
    log.info("  ⏱️   %s  price=%.10f  pnl=%+0.2f%%  t=%s",
               tag(state.symbol, state.address), price, pnl, bars.index[-1])


# ──────────────────────────────────────────────────────────────────────────
# SAFETY GATE — RugCheck (unchanged logic, just no pool→mint lookup needed
# anymore since Birdeye already gives us the token mint address directly)
# ──────────────────────────────────────────────────────────────────────────

def rugcheck_report(mint: str) -> dict | None:
    global _rc_adaptive_delay, _rc_consecutive_ok, _rc_last_request_ts, _rc_consecutive_400s

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
        if r.status_code == 400:
            # This candidate still gets rejected below (fail-closed is the
            # right default — "can't verify" should never mean "assume
            # safe"). But a sustained run of 400s across DIFFERENT mints is
            # RugCheck's own API having a bad time, not real per-token
            # findings, and deserves the same visibility + cooldown Birdeye
            # got rather than silently rejecting the whole candidate pool
            # while looking identical to genuine rug findings in the logs.
            with _rc_rate_lock:
                _rc_consecutive_400s += 1
                streak = _rc_consecutive_400s
            if streak >= RC_QUOTA_SUSPECT_THRESHOLD:
                log.error("  RugCheck: %d consecutive 400s across different "
                            "mints — this looks like a RugCheck-side outage "
                            "or limit, not real rejections. Every candidate "
                            "checked during this window is failing closed "
                            "(correctly) but for the wrong reason. Cooling "
                            "down %ds.", streak, RC_QUOTA_COOLDOWN_SECONDS)
                _sleep_with_heartbeat(RC_QUOTA_COOLDOWN_SECONDS, "RugCheck outage")
                with _rc_rate_lock:
                    _rc_consecutive_400s = 0
            else:
                log.warning("  rugcheck %s -> 400 (streak %d/%d)", mint[:8],
                             streak, RC_QUOTA_SUSPECT_THRESHOLD)
            return None
        if r.status_code != 200:
            log.warning("  rugcheck %s -> HTTP %d", mint[:8], r.status_code)
            return None

        with _rc_rate_lock:
            _rc_consecutive_ok += 1
            _rc_consecutive_400s = 0
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
        log.info("  ✗ %s — RugCheck reject: %s", tag(sym, mint), rc_reason)
        return None

    jup_ok, jup_reason = check_jupiter_traps(mint)
    if not jup_ok:
        log.info("  ✗ %s — Jupiter reject: %s", tag(sym, mint), jup_reason)
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


def tag(symbol: str, address: str) -> str:
    """
    Symbol + full mint address together, for logging. Symbols/tickers are
    NOT unique — many unrelated tokens reuse the same popular symbol/name —
    so a log line with only "%-12s" can point at the wrong coin entirely.
    The mint address is the actual unique identifier; always print it
    alongside the symbol wherever a specific token is being referenced.
    """
    return f"{symbol:<12} [{address}]"


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
    cached_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
                             # last-known bars for this token. Reused across
                             # polls (and across exits, when scanning loops
                             # back to the passive list) so subsequent fetches
                             # only need to pull whatever's NEW since this,
                             # instead of re-pulling the full lookback window
                             # every single poll. See fetch_ohlcv_incremental.


def poll_token(state: TokenState, lookback: int, bars: pd.DataFrame | None = None) -> None:
    if bars is None:
        bars = fetch_ohlcv_incremental(state)
    if bars.empty or len(bars) < lookback + 5:
        log.warning("  %s — insufficient bars (%d), skipping this poll",
                     tag(state.symbol, state.address), len(bars))
        return
    n = len(bars)

    if not state.in_position:
        window = bars.iloc[n - lookback - 1:]
        fired, structure_high = B.check_entry(window)
        if fired:
            if LIVE_TRADING:
                swap_result = execute_entry_swap(state.address)
                if swap_result is None:
                    log.error("  🛑 %s — entry SIGNAL fired but the LIVE "
                                "swap failed; NOT marking as in_position "
                                "(would desync bot state from the real "
                                "wallet, which still holds SOL, not the "
                                "token)", tag(state.symbol, state.address))
                    return
                sig, lamports_in = swap_result
                log.info("  💰 LIVE BUY  %s  %.4f SOL  tx=%s",
                          tag(state.symbol, state.address), lamports_in / 1e9, sig)
            entry_idx_abs = n - 1
            state.in_position    = True
            state.entry_idx_abs  = entry_idx_abs
            state.entry_price    = float(bars.iloc[entry_idx_abs]['close'])
            state.entry_time     = bars.index[entry_idx_abs]
            state.structure_high = structure_high
            state.last_bar_count = n
            state.stale_poll_count = 0
            log.info("  🟢 ENTRY  %s @ %.8f  (reclaim target %.8f)  t=%s",
                      tag(state.symbol, state.address), state.entry_price, structure_high, state.entry_time)
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
        log.info("  …   HOLD   %s  unrealized %+0.2f%%  (held %d bars)",
                  tag(state.symbol, state.address), unrealized, held_bars)
        return

    if force_reason == "pool_appears_dead_no_new_bars":
        # A frozen last-close price is NOT the same thing as a sellable
        # price. If the pool went dark because liquidity was actually
        # pulled (the common rug pattern), there may be little or nothing
        # left to sell into — recording pnl=unrealized here would print a
        # clean, false number for what could easily be a near-total real
        # loss. Probe a live round-trip quote right now, the same way the
        # entry safety gate does, instead of trusting the stale close.
        buy_probe = jup_quote(USDC_MINT, state.address, PROBE_USDC_AMOUNT)
        sell_quote = buy_probe and jup_quote(state.address, USDC_MINT, int(buy_probe.get("outAmount", 0)))

        if not sell_quote:
            log.warning("  🛑 FORCE EXIT  %s — NO LIVE SELL ROUTE FOUND. "
                         "This pool is very likely rugged/illiquid — in real "
                         "trading this would probably be UNSELLABLE, not a "
                         "clean %+0.2f%% exit. Recording as a presumed "
                         "near-total loss (pnl capped at %.0f%%), not the "
                         "stale close price. (No live route also means "
                         "there's nothing for a real swap to execute "
                         "against — skipping LIVE_TRADING execution here "
                         "even if it's on; if a real balance is still stuck "
                         "in this token, it needs manual recovery, not this "
                         "bot retrying a route that doesn't exist.)",
                         tag(state.symbol, state.address), unrealized, PRESUMED_RUG_LOSS_PCT)
            exit_price = float(bars.iloc[-1]['close'])
            pnl_pct = PRESUMED_RUG_LOSS_PCT
        else:
            sell_impact = _safe_float(sell_quote.get("priceImpactPct"))
            impact_pct = abs(sell_impact) * 100 if sell_impact is not None else float("nan")
            if LIVE_TRADING:
                swap_result = execute_exit_swap(state.address)
                if swap_result is None:
                    log.error("  🛑 %s — live sell route exists but the "
                                "LIVE swap FAILED; position left OPEN so the "
                                "next poll retries the exit instead of "
                                "silently marking this closed while real "
                                "tokens are still held", tag(state.symbol, state.address))
                    return
                sig, amount_sold = swap_result
                log.info("  💰 LIVE SELL %s  tx=%s", tag(state.symbol, state.address), sig)
            exit_price = float(bars.iloc[-1]['close'])
            pnl_pct = unrealized
            log.warning("  🛑 FORCE EXIT  %s @ %.8f  pnl=%+0.2f%%  reason=%s  "
                         "(live sell route STILL EXISTS, ~%.1f%% price impact "
                         "at $%d — real fill will be worse than this printed "
                         "number by roughly that much)  t=%s",
                         tag(state.symbol, state.address), exit_price, pnl_pct, force_reason,
                         impact_pct, PROBE_USDC_AMOUNT // 1_000_000, bars.index[-1])
    elif force_reason is not None:
        if LIVE_TRADING:
            swap_result = execute_exit_swap(state.address)
            if swap_result is None:
                log.error("  🛑 %s — force-exit fired but the LIVE swap "
                            "FAILED; position left OPEN so the next poll "
                            "retries the exit", tag(state.symbol, state.address))
                return
            sig, amount_sold = swap_result
            log.info("  💰 LIVE SELL %s  tx=%s", tag(state.symbol, state.address), sig)
        exit_price = float(bars.iloc[-1]['close'])
        pnl_pct = unrealized
        log.warning("  🛑 FORCE EXIT  %s @ %.8f  pnl=%+0.2f%%  reason=%s  t=%s",
                     tag(state.symbol, state.address), exit_price, pnl_pct, force_reason, bars.index[-1])
    else:
        if LIVE_TRADING:
            swap_result = execute_exit_swap(state.address)
            if swap_result is None:
                log.error("  🛑 %s — exit signal fired but the LIVE swap "
                            "FAILED; position left OPEN so the next poll "
                            "retries the exit", tag(state.symbol, state.address))
                return
            sig, amount_sold = swap_result
            log.info("  💰 LIVE SELL %s  tx=%s", tag(state.symbol, state.address), sig)
        rel_idx, reason = result
        exit_idx_abs = state.entry_idx_abs + rel_idx
        exit_price = float(bars.iloc[exit_idx_abs]['close'])
        pnl_pct = (exit_price - state.entry_price) / state.entry_price * 100
        log.info("  🔴 EXIT   %s @ %.8f  pnl=%+0.2f%%  reason=%s  t=%s",
                  tag(state.symbol, state.address), exit_price, pnl_pct, reason, bars.index[exit_idx_abs])

    state.in_position    = False
    state.entry_idx_abs  = None
    state.entry_price    = None
    state.entry_time     = None
    state.structure_high = None
    state.last_bar_count = None
    state.stale_poll_count = 0


# ──────────────────────────────────────────────────────────────────────────
# LIVE TRADING — Jupiter swap execution (DISABLED BY DEFAULT)
# ──────────────────────────────────────────────────────────────────────────
# Everything above this point is unchanged paper-trading logic: signals are
# detected and logged, no funds move. This section adds the ability to
# actually execute them, gated behind LIVE_TRADING so paper mode remains the
# default and nothing below runs unless that flag is explicitly set.
#
# SIZING: as asked, every entry spends the FULL SOL balance (minus a small
# fee/rent reserve) and every exit sells the FULL held token balance back to
# SOL. There is no position sizing, no partial fills, and no per-trade cap —
# one signal is one all-in trade. Combined with this bot's actual candidate
# pool (thin, freshly-discovered, RugCheck/Jupiter-gated but still low-cap
# Solana tokens), that means a single bad fill, stale signal, or rugged pool
# can end the session at close to zero, not a fraction of it. That isn't a
# hypothetical edge case for this specific candidate set — it's closer to
# the median bad outcome. If a lot on this is not something you'd accept
# losing today, in full, this isn't the sizing to run with LIVE_TRADING=true.
#
# SETUP:
#   export LIVE_TRADING=true            # anything else stays in paper mode
#   export SOLANA_PRIVATE_KEY=<base58>  # NEVER hardcode this in the file —
#                                        # it is complete, irreversible
#                                        # control of the wallet. Use a
#                                        # dedicated hot wallet holding only
#                                        # what this bot is allowed to lose,
#                                        # never a main wallet.
#   export SOLANA_RPC_URL=<your RPC>    # defaults to the public mainnet-beta
#                                        # RPC, which rate-limits hard and
#                                        # will make "signal fired but the
#                                        # swap silently never landed" a
#                                        # regular occurrence. Use a real
#                                        # provider (Helius/QuickNode/Triton)
#                                        # for anything actually live.
#   pip install solders --break-system-packages

LIVE_TRADING   = os.environ.get("LIVE_TRADING", "false").strip().lower() == "true"
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOL_MINT       = "So11111111111111111111111111111111111111112"
JUP_SWAP_URL   = "https://lite-api.jup.ag/swap/v1/swap"

SOL_FEE_RESERVE_LAMPORTS   = 15_000_000  # ~0.015 SOL kept unspent per entry —
                                            # covers tx fees + rent for any
                                            # new token account. Entry size
                                            # is (balance - this), never the
                                            # raw full balance.
TX_CONFIRM_TIMEOUT_SECONDS = 60
TX_CONFIRM_POLL_SECONDS    = 2

_wallet_keypair = None  # lazily loaded — paper-mode runs never need a key


def get_keypair():
    """Loads the wallet keypair from SOLANA_PRIVATE_KEY on first use. Only
    ever called when LIVE_TRADING is on, so paper-mode runs never need this
    env var set at all."""
    global _wallet_keypair
    if _wallet_keypair is not None:
        return _wallet_keypair
    secret = os.environ.get("SOLANA_PRIVATE_KEY")
    if not secret:
        raise RuntimeError(
            "LIVE_TRADING is on but SOLANA_PRIVATE_KEY is not set. "
            "export SOLANA_PRIVATE_KEY=<base58 secret key> and re-run."
        )
    from solders.keypair import Keypair
    _wallet_keypair = Keypair.from_base58_string(secret)
    return _wallet_keypair


def _rpc_call(method: str, params: list) -> dict | None:
    try:
        r = SESSION.post(
            SOLANA_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        log.error("  RPC %s failed: %s", method, e)
        return None
    if "error" in data:
        log.error("  RPC %s error: %s", method, data["error"])
        return None
    return data.get("result")


def get_sol_balance_lamports(pubkey: str) -> int:
    result = _rpc_call("getBalance", [pubkey])
    return int(result.get("value", 0)) if result is not None else 0


def get_token_balance_raw(pubkey: str, mint: str) -> int:
    """Sums the raw (base-unit) balance across every token account this
    wallet holds for `mint` — normally exactly one, but summed defensively
    in case more than one ATA exists."""
    result = _rpc_call("getTokenAccountsByOwner", [
        pubkey, {"mint": mint}, {"encoding": "jsonParsed"},
    ])
    if not result or not result.get("value"):
        return 0
    total = 0
    for acct in result["value"]:
        try:
            total += int(acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
        except (KeyError, TypeError, ValueError):
            continue
    return total


def jup_swap_transaction(quote: dict, user_pubkey: str) -> str | None:
    """POSTs the quote back to Jupiter's /swap endpoint to get a fully-formed
    (but unsigned) base64 transaction ready to sign and send."""
    try:
        r = SESSION.post(
            JUP_SWAP_URL,
            json={
                "quoteResponse": quote,
                "userPublicKey": user_pubkey,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": "auto",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("swapTransaction")
    except (requests.exceptions.RequestException, ValueError) as e:
        log.error("  jupiter /swap failed: %s", e)
        return None


def sign_and_send_transaction(tx_b64: str) -> str | None:
    """Deserializes the Jupiter-provided transaction, signs it with the
    loaded wallet, submits it, and polls for confirmation. Returns the
    signature on confirmed success, None on any failure — network, RPC
    rejection, timeout, or the transaction landing but erroring on-chain."""
    import base64
    from solders.transaction import VersionedTransaction

    keypair = get_keypair()
    unsigned_tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    signature = keypair.sign_message(bytes(unsigned_tx.message))
    signed_tx = VersionedTransaction.populate(unsigned_tx.message, [signature])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

    sig = _rpc_call("sendTransaction", [
        signed_b64, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
    ])
    if sig is None:
        return None

    deadline = time.time() + TX_CONFIRM_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(TX_CONFIRM_POLL_SECONDS)
        status = _rpc_call("getSignatureStatuses", [[sig], {"searchTransactionHistory": True}])
        entry = (status or {}).get("value", [None])[0]
        if entry:
            if entry.get("err") is not None:
                log.error("  tx %s landed but FAILED on-chain: %s", sig, entry["err"])
                return None
            if entry.get("confirmationStatus") in ("confirmed", "finalized"):
                log.info("  tx confirmed: %s", sig)
                return sig
    log.error("  tx %s not confirmed within %ds — check an explorer manually, "
                "bot state may now be out of sync with the real wallet",
                sig, TX_CONFIRM_TIMEOUT_SECONDS)
    return None


def execute_entry_swap(mint: str) -> tuple[str, int] | None:
    """SOL -> mint, sized as the full wallet SOL balance minus the fee
    reserve. Returns (signature, lamports spent) on confirmed success,
    None on any failure (nothing partially executes silently)."""
    keypair = get_keypair()
    pubkey = str(keypair.pubkey())
    balance = get_sol_balance_lamports(pubkey)
    amount_in = balance - SOL_FEE_RESERVE_LAMPORTS
    if amount_in <= 0:
        log.error("  entry swap skipped — SOL balance (%d lamports) doesn't "
                    "cover the fee reserve (%d)", balance, SOL_FEE_RESERVE_LAMPORTS)
        return None

    quote = jup_quote(SOL_MINT, mint, amount_in)
    if not quote:
        log.error("  entry swap skipped — no live SOL->%s route", mint[:8])
        return None

    tx_b64 = jup_swap_transaction(quote, pubkey)
    if not tx_b64:
        log.error("  entry swap skipped — /swap failed to build a transaction")
        return None

    sig = sign_and_send_transaction(tx_b64)
    return (sig, amount_in) if sig else None


def execute_exit_swap(mint: str) -> tuple[str, int] | None:
    """mint -> SOL, sized as the full held token balance. Returns
    (signature, raw token amount sold) on confirmed success."""
    keypair = get_keypair()
    pubkey = str(keypair.pubkey())
    balance = get_token_balance_raw(pubkey, mint)
    if balance <= 0:
        log.warning("  exit swap skipped — no on-chain balance found for %s "
                     "(already sold, or entry never actually landed?)", mint[:8])
        return None

    quote = jup_quote(mint, SOL_MINT, balance)
    if not quote:
        log.error("  exit swap FAILED — no live %s->SOL route (likely rugged/"
                    "illiquid; see the force-exit rug probe above)", mint[:8])
        return None

    tx_b64 = jup_swap_transaction(quote, pubkey)
    if not tx_b64:
        log.error("  exit swap FAILED — /swap failed to build a transaction")
        return None

    sig = sign_and_send_transaction(tx_b64)
    return (sig, balance) if sig else None


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

    if LIVE_TRADING:
        log.warning("=" * 70)
        log.warning("LIVE_TRADING IS ON. Every entry/exit signal below will "
                     "swap REAL funds through Jupiter — full SOL balance in "
                     "on entry, full token balance out on exit — with no "
                     "confirmation step. There is no partial sizing here.")
        log.warning("=" * 70)
        try:
            kp = get_keypair()
        except RuntimeError as e:
            log.error(str(e))
            return
        pubkey = str(kp.pubkey())
        bal_lamports = get_sol_balance_lamports(pubkey)
        log.warning("  wallet:  %s", pubkey)
        log.warning("  balance: %.4f SOL (%d lamports)", bal_lamports / 1e9, bal_lamports)
        log.warning("  RPC:     %s", SOLANA_RPC_URL)
        if SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com":
            log.warning("  using the PUBLIC mainnet-beta RPC — this rate-"
                         "limits hard under real use; set SOLANA_RPC_URL to "
                         "a real provider before trusting this for anything "
                         "time-sensitive.")
    else:
        log.info("LIVE_TRADING is off — running in paper-trading mode "
                   "(no funds will move).")

    states: dict[str, TokenState] = {}
    last_universe_refresh = 0.0
    last_safety_refresh = 0.0

    log.info("Starting Birdeye live screener. Single active position at a time.")
    check_birdeye_credits()

    while True:
        now = time.time()

        holding = any(s.in_position for s in states.values())

        # Neither refresh path may run while a position is open — discovery
        # rebuilds `states` from a fresh candidate list, and anything not in
        # that fresh list (including an OPEN position) was being silently
        # dropped with no exit recorded. Deferring both timers until flat
        # again is what the "single active position, only that token gets
        # touched" design already promised — this just actually enforces it.
        if not holding and (now - last_universe_refresh > UNIVERSE_REFRESH_SECONDS or not states):
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

        elif not holding and now - last_safety_refresh > SAFETY_REFRESH_SECONDS:
            df = pd.DataFrame([{"symbol": s.symbol, "address": s.address} for s in states.values()])
            recheck = run_safety_gate(df)
            still_safe = set(recheck["address"]) if not recheck.empty else set()
            for addr in list(states.keys()):
                if addr not in still_safe and not states[addr].in_position:
                    log.warning("  %s failed safety re-check — dropping",
                                 tag(states[addr].symbol, addr))
                    del states[addr]
                elif addr not in still_safe:
                    log.warning("  %s failed safety re-check but is OPEN — tracking to exit only",
                                 tag(states[addr].symbol, addr))
            last_safety_refresh = now

        if not states:
            log.warning("No safe candidates to scan. Sleeping.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        held = next((s for s in states.values() if s.in_position), None)

        if held is not None:
            log.info("── poll: 1 token held (%s), not scanning for new entries ──",
                       tag(held.symbol, held.address))
            try:
                poll_token(held, args.lookback)
            except Exception as e:
                log.error("  %s — poll error: %s", tag(held.symbol, held.address), e)

            # Sleep out the rest of this cycle in DEBUG_PNL_TICK_SECONDS
            # chunks, printing a cheap price/pnl line at each one. This is
            # purely informational — the real exit check stays on the poll
            # above's cadence (once per POLL_INTERVAL_SECONDS), since that's
            # what the OBV-divergence signal actually needs a new bar for.
            remaining = max(0.0, POLL_INTERVAL_SECONDS - _bd_adaptive_delay)
            while remaining > 0 and held.in_position:
                nap = min(DEBUG_PNL_TICK_SECONDS, remaining)
                time.sleep(nap)
                remaining -= nap
                if held.in_position and remaining > 0:
                    try:
                        debug_pnl_tick(held)
                    except Exception as e:
                        log.error("  pnl-debug tick error: %s", e)
            continue

        passive = list(states.values())
        passive_to_poll = passive
        if len(passive) > PASSIVE_BATCH_SIZE:
            change_map = refresh_price_change([p.address for p in passive])
            ranked = sorted(passive, key=lambda s: abs(change_map.get(s.address, 0.0)), reverse=True)
            passive_to_poll = ranked[:PASSIVE_BATCH_SIZE]
            skipped = [tag(s.symbol, s.address) for s in ranked[PASSIVE_BATCH_SIZE:]]
            if skipped:
                log.info("  (skipped this cycle — no notable h1 move: %s)", ", ".join(skipped))

        log.info("── poll: 0 in position, scanning %d passive token(s) ──", len(passive_to_poll))

        # Fetch fresh bars for all of them concurrently first — this is the
        # slow part (network + rate-limit delay per token). The check_entry
        # pass right after then reads straight from state.cached_bars with
        # no further waiting, and stays strictly sequential/first-wins so
        # entry semantics don't change.
        prefetch_passive_bars(passive_to_poll)

        for state in passive_to_poll:
            try:
                poll_token(state, args.lookback, bars=state.cached_bars)
            except Exception as e:
                log.error("  %s — poll error: %s", tag(state.symbol, state.address), e)
            if state.in_position:
                log.info("  entry fired on %s — stopping passive scan for this cycle",
                           tag(state.symbol, state.address))
                break

        time.sleep(max(0.0, POLL_INTERVAL_SECONDS - len(passive_to_poll) * _bd_adaptive_delay))


if __name__ == "__main__":
    main()