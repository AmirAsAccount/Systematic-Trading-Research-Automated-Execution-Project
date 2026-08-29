# config.py
# Development notes:
# This file has been through several strategy revisions.
# The long design/debug commentary from development was removed once the behavior stabilized.
# Comments below are intentionally short; the code and config names should carry most of the detail.
import argparse
import asyncio
import base64
import csv
import difflib
import json
import logging
import math
import os
import random
import re
import sys
import queue
import threading
import time
import traceback
import warnings
from collections import OrderedDict

from lxml import html
from html import unescape as _html_unescape
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    print("[ERROR] Python 3.9+ (zoneinfo) is required.")
    sys.exit(1)

try:
    from alpaca.data.live.news import NewsDataStream
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
except ImportError:
    print("[ERROR] alpaca-py not installed: pip install alpaca-py")
    sys.exit(1)

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Time zones used by the trading session.
NY_TZ = ZoneInfo("America/New_York")


# API credentials come from the environment.
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET")

SCHWAB_MARKETDATA_CLIENT_ID     = os.environ.get("SCHWAB_MARKETDATA_CLIENT_ID")
SCHWAB_MARKETDATA_CLIENT_SECRET = os.environ.get("SCHWAB_MARKETDATA_CLIENT_SECRET")
SCHWAB_MARKETDATA_REFRESH_TOKEN = os.environ.get("SCHWAB_MARKETDATA_REFRESH_TOKEN")
SCHWAB_MARKETDATA_REDIRECT_URI  = os.environ.get("SCHWAB_MARKETDATA_REDIRECT_URI", "https://127.0.0.1")

SCHWAB_TRADER_CLIENT_ID     = os.environ.get("SCHWAB_TRADER_CLIENT_ID")
SCHWAB_TRADER_CLIENT_SECRET = os.environ.get("SCHWAB_TRADER_CLIENT_SECRET")
SCHWAB_TRADER_REFRESH_TOKEN = os.environ.get("SCHWAB_TRADER_REFRESH_TOKEN")
SCHWAB_TRADER_REDIRECT_URI  = os.environ.get("SCHWAB_TRADER_REDIRECT_URI", "https://127.0.0.1")
SCHWAB_ACCOUNT_HASH  = os.environ.get("SCHWAB_ACCOUNT_HASH")

# Capital is divided into a fixed number of slots.
TOTAL_BUY_OPPORTUNITIES = 10
MIN_SLOTS_PER_TRADE     = 1
MAX_SLOTS_PER_TRADE     = 10
CAPITAL_POOL_FALLBACK_USD = None
EXPECTED_BUYING_POWER_FIELD = "cashAvailableForWithdrawal"
ALLOW_BUYING_POWER_FIELD_FALLBACK = False
LIMIT_ORDER_SLIPPAGE_PCT = 0.01

# Position sizing weights; these do not decide whether to enter.
MAGNITUDE_WEIGHT_FLOAT         = 0.25
MAGNITUDE_WEIGHT_PRICE_ACTION  = 0.25
MAGNITUDE_WEIGHT_LLM           = 0.30
MAGNITUDE_WEIGHT_NEWS_CATEGORY = 0.20

PRICE_ACTION_LOOKBACK_MIN = 20
PRICE_ACTION_CLIP_PCT     = 10.0

LLM_CONFIDENCE_SCORE            = {"high": 1.0, "medium": 0.6, "low": 0.3}
LLM_REASONING_RICHNESS_WORD_CAP = 40
LLM_CONFIDENCE_VS_RICHNESS_MIX  = 0.7

MAGNITUDE_NEWS_CATEGORY_TIERS = [
    ("regulatory_approval", 1.0, [
        "fda approval", "fda approves", "ema approval", "ce mark",
        "regulatory approval", "granted approval", "marketing authorization",
    ]),
    ("ma_acquisition", 0.9, [
        "to be acquired", "acquisition of", "merger agreement", "buyout",
        "tender offer", "definitive merger", "agrees to acquire", "to acquire",
    ]),
    ("major_financing_contract", 0.7, [
        "multi-year contract", "definitive agreement", "strategic partnership",
        "million contract", "billion contract", "supply agreement",
        "licensing agreement", "government contract",
    ]),
    ("uplisting_squeeze_setup", 0.6, [
        "uplist", "nasdaq listing", "short squeeze", "reverse stock split",
    ]),
    ("earnings_guidance", 0.4, [
        "beats estimates", "raises guidance", "record revenue",
        "raises full-year", "tops estimates", "beats revenue estimates",
    ]),
]
NEWS_CATEGORY_BASELINE_SCORE = 0.3

# Cheap filters for headlines that describe an old move.
STALE_RECAP_HEADLINE_PATTERNS = [
    "why shares of", "why is", "why are shares", "here's why",
    "here's what's driving", "shares surge:", "shares soar:", "shares jump:",
    "shares are up today", "shares are down today", "stocks moving today",
    "stocks that moved", "explained:", "stocks that have gone up",
    "stocks that have gone down", "stocks to watch today", "stocks in focus",
    "roundup:", "here's what's behind",
]

# Basic watchlist filters.
PRICE_CEILING_USD      = 1000000.0
FLOAT_CEILING_SHARES   = 10_000_000
MIN_AVG_VOLUME_SHARES  = 100_000
# Rolling-high breakout settings.
GAP_PCT_MIN = 10.0
RVOL_MIN = 3.0
PREMARKET_MIN_VOLUME_SHARES = 20_000
MARKET_SCAN_INTERVAL_SEC = 60
QUOTE_BATCH_CHUNK_SIZE = 100
OLD_BULLISH_NEWS_LOOKBACK_DAYS = 18
NEWS_TIER_REFRESH_SEC = 5 * 60
ROLLING_HIGH_POLL_INTERVAL_SEC = 5

# Breakout sizing.
BREAKOUT_WEIGHT_CONFIRMATION = 0.5
BREAKOUT_WEIGHT_GAP_SCANNER  = 0.5
GAP_PCT_SCORE_CLIP = 50.0
RVOL_SCORE_CLIP    = 10.0

# Sentiment thresholds.
NEGATIVE_PROB_MAX = 0.30
POSITIVE_PROB_MIN = 0.65

# Local LLM settings.
OLLAMA_ENABLED     = True
OLLAMA_URL         = "http://localhost:11434/api/generate"
OLLAMA_MODEL       = "qwen3:8b"
OLLAMA_TIMEOUT_SEC = 8

# FDA fast-path candidates.
FDA_APPROVAL_KEYWORDS = [
    "fda approval", "fda approves", "receives fda approval", "granted fda approval",
    "approved by the fda", "fda grants approval", "nda approved",
    "new drug application approved", "bla approved",
    "biologics license application approved", "fda clearance", "accelerated approval",
]
FDA_EXCLUDE_KEYWORDS = [
    "breakthrough therapy designation", "fast track designation",
    "orphan drug designation", "priority review", "pdufa date",
    "seeking approval", "expects approval", "up for approval", "fda to review",
]

# Hard-negative phrases.
RED_FLAG_KEYWORDS = [
    "bankruptcy", "chapter 11", "chapter 7 filing", "going concern",
    "delisting", "delisted", "deficiency notice", "restatement", "restates",
    "material weakness", "materially misstated", "fraud", "sec subpoena",
    "sec investigation", "sec charges", "class action lawsuit", "insolvency",
    "insolvent", "liquidation", "receivership", "covenant breach",
    "debt default", "defaults on", "auditor resignation", "resignation of auditor",
    "non-reliance", "clawback", "trading halt pending news", "cease trading",
    "dilutive offering", "going private",
]

# Fast volume confirmation.
TICK_POLL_INTERVAL_SEC      = 1.0
TICK_WINDOW_SEC             = 60
TICK_VOLUME_MULTIPLE        = 4.0
TICK_VOLUME_SESSION_CONFIRM_MULTIPLE = 2.0
TICK_UP_VOLUME_RATIO_MIN    = 0.60
SESSION_SECONDS_FOR_SCALING = 23_400

# Session-level volume baseline.
RVOL_SESSION_CONFIRM_MIN = 1.5
PREMARKET_OPEN_ET        = dt_time(4, 0)

SCHWAB_SEAMLESS_SESSION_START_ET = dt_time(7, 0)

# Exit settings.
SWING_WINDOW_EXIT        = 2

# Fixed stop is calculated once at entry.
ENTRY_STOP_ATR_PERIOD    = 14
ENTRY_STOP_ATR_MULTIPLE  = 2.5
MAX_LOSS_PCT             = 0.10
MIN_STOP_PCT             = 0.02
SWING_WINDOW_EXIT        = 2
EXIT_LOOKBACK_BARS       = 30
PT_TZ                    = ZoneInfo("America/Los_Angeles")
SECONDARY_EXIT_PST_HOUR  = 6
SECONDARY_EXIT_MOVE_TRIGGER_PCT = 5.0
CVD_LOOKBACK_BARS        = 5
CVD_BEARISH_THRESHOLD    = -0.50

# Retry limits until the broker confirms a fill.
EXIT_FILL_CHECK_SEC         = 5
EXIT_FILL_POLL_INTERVAL_SEC = 1.0
EXIT_LIMIT_WIDEN_STEP_PCT   = 0.01
EXIT_FILL_MAX_ATTEMPTS      = 5

# Entry orders use the same fill-confirmation pattern.
ENTRY_FILL_CHECK_SEC         = 5
ENTRY_FILL_POLL_INTERVAL_SEC = 1.0
ENTRY_LIMIT_WIDEN_STEP_PCT   = 0.01
ENTRY_FILL_MAX_ATTEMPTS      = 5

# Additional entry filters.
ENTRY_VWAP_GATE_ENABLED = True

MAX_SPREAD_PCT = 0.03

# General runtime settings.
POLL_INTERVAL_SEC  = 20
HEARTBEAT_SEC      = 30

FINVIZ_PAGE_DELAY_SEC   = 1.5
FINVIZ_BATCH_SIZE       = 3
FINVIZ_COOLDOWN_SEC     = 5.0
FUNDAMENTALS_CACHE_TTL_SEC = 60 * 60

CACHE_DIR  = Path("cache")
TRADES_LOG = Path("trades_log.csv")
CACHE_DIR.mkdir(exist_ok=True)

# Shared runtime state.
_positions_lock = threading.Lock()
open_positions: dict = {}


# New entries pause while a position is open.
def trading_paused() -> bool:
    """GLOBAL PAUSE — true the instant a confirmed entry lands in
    open_positions, cleared automatically the instant open_positions is
    empty again (mirrors run_is_complete()'s own "no open positions" check,
    so there's no separate pause flag/bookkeeping to drift out of sync with
    reality). While True, new-entry consideration is globally suspended:
    Tier 1 (market_scan_and_update), Tier 2 (refresh_bullish_news_tier),
    Tier 3 (_check_rolling_high_breakout_and_enter), the halt sub-path
    (_check_halt_reopen_and_enter), and the news pipeline (handle_news_event)
    all check this and no-op while paused. monitor_positions_loop /
    _check_position (halt check, stop check, secondary-exit check, PnL
    logging, and the exit fill-confirmation retry loop for the open
    position) are NEVER gated by this — they keep running at full speed
    regardless."""
    with _positions_lock:
        return len(open_positions) > 0


_exiting_lock = threading.Lock()
_exiting_tickers: set = set()

_watchlist_lock = threading.Lock()
watchlist: set = set()

# Keep websocket subscriptions in manageable batches.
NEWS_SUBSCRIBE_CHUNK_SIZE = 100

# Reconnect delay after a dropped news stream.
WS_RECONNECT_BACKOFF_SEC = 5

# Cached market data.
_fundamentals_lock = threading.Lock()
_fundamentals_cache: dict = {}

_avg_daily_volume_lock = threading.Lock()
AVG_DAILY_VOLUME: dict = {}

_latest_quotes_lock = threading.Lock()
_latest_quotes: dict = {}

SCHWAB: Optional["SchwabClient"] = None
SCHWAB_TRADER: Optional["SchwabClient"] = None
_vader = None
_lm = None
_last_news_at: Optional[datetime] = None
_news_lock = threading.Lock()

# News stream bookkeeping.
_acked_lock = threading.Lock()
acked_news_symbols: set = set()

# Prevent duplicate article processing.
_seen_news_lock = threading.Lock()
_seen_news_ids: "OrderedDict[int, None]" = OrderedDict()
_SEEN_NEWS_CACHE_SIZE = 5000


# Build the capital pool once at startup.
def detect_capital_pool(schwab_trader: "SchwabClient", account_hash: str) -> float:
    """Queries the account's real buying power ONCE and divides it into
    TOTAL_BUY_OPPORTUNITIES equal capital slots. FAILS CLOSED: if the
    account balance can't be read at all (and CAPITAL_POOL_FALLBACK_USD is
    also unset), this raises rather than silently trading with a guessed or
    zero capital figure — better to not start the run than to size every
    position off a wrong number."""
    global ACCOUNT_BUYING_POWER_USD, CAPITAL_PER_SLOT_USD
    buyingPower = schwab_trader.get_account_buying_power(account_hash)
    if buyingPower is None:
        if CAPITAL_POOL_FALLBACK_USD is not None:
            print(f"  [WARN] Could not detect account buying power — using "
                  f"CAPITAL_POOL_FALLBACK_USD=${CAPITAL_POOL_FALLBACK_USD:.2f} instead.")
            buyingPower = CAPITAL_POOL_FALLBACK_USD
        else:
            raise RuntimeError(
                "Could not detect Schwab account buying power at startup, and "
                "CAPITAL_POOL_FALLBACK_USD is not set — refusing to guess a capital "
                "figure to size live trades against. Check SCHWAB_ACCOUNT_HASH / "
                "SchwabClient.get_account_buying_power()'s field-name assumptions "
                "against your account's actual /accounts response, or set "
                "CAPITAL_POOL_FALLBACK_USD to a fixed override if you want the run "
                "to proceed anyway."
            )
    if buyingPower <= 0:
        raise RuntimeError(f"Detected buying power is ${buying_power:.2f} — nothing to trade with.")

    ACCOUNT_BUYING_POWER_USD = buyingPower
    CAPITAL_PER_SLOT_USD = buyingPower / TOTAL_BUY_OPPORTUNITIES
    return buyingPower


# Reserve capital slots atomically.
def try_claim_shares(desired_qty: int) -> int:
    """Claims up to `desired_qty` CAPITAL SLOTS from the pool atomically. May
    claim FEWER than requested if the pool has less remaining than asked for
    (e.g. a high-magnitude signal requesting MAX_SLOTS_PER_TRADE=10 when only
    4 remain claims those 4 rather than blocking). Returns the actual number
    of slots claimed (0 if the pool is already fully exhausted, in which case
    the caller should not enter). The dollar amount a slot is worth
    (CAPITAL_PER_SLOT_USD) is fixed once at startup by detect_capital_pool().
    Thread-safe against near-simultaneous news events on different tickers."""
    global opportunities_remaining
    with _opportunities_lock:
        if opportunities_remaining <= 0:
            return 0
        claimed = min(int(desired_qty), opportunities_remaining)
        opportunities_remaining -= claimed
        return claimed


# FDA entries can consume the remaining pool.
def claim_fda_all_remaining() -> int:
    """FDA-approval fast path is the SOLE exception: it claims ALL capital
    slots remaining at this instant (i.e. the entire remaining capital pool)
    in one order. Returns the number of slots claimed (0 if the pool was
    already exhausted, in which case the caller should not enter)."""
    global opportunities_remaining
    with _opportunities_lock:
        qty = opportunities_remaining
        opportunities_remaining = 0
        return qty


# Return unused slots when an entry cannot be placed.
def release_opportunity(qty: int):
    """Gives back `qty` capital slots if an entry attempt was aborted before
    an order was actually placed (halted / illiquid / order rejected / no
    usable price / insufficient capital for even 1 share), so the run can
    still catch other valid catalysts."""
    global opportunities_remaining
    if qty <= 0:
        return
    with _opportunities_lock:
        opportunities_remaining += qty


def opportunities_left() -> int:
    with _opportunities_lock:
        return opportunities_remaining


# The run ends when the pool is spent and positions are flat.
def run_is_complete() -> bool:
    """The run is done once every capital slot has been used AND every
    resulting position has closed."""
    with _positions_lock:
        noOpenPositions = len(open_positions) == 0
    return opportunities_left() <= 0 and noOpenPositions


# Stop the stream and worker threads cleanly.
def request_shutdown(reason: str):
    print(f"\n[SHUTDOWN] {reason}")
    if _stream_ref is not None:
        for method_name in ("stop", "close"):
            stopper = getattr(_stream_ref, method_name, None)
            if callable(stopper):
                try:
                    stopper()
                    break
                except Exception as e:
                    print(f"  [WARN] stream.{method_name}() failed: {e}")
    if _stop_event_ref is not None:
        _stop_event_ref.set()
    threading.Timer(5.0, lambda: os._exit(0)).start()

_opportunities_lock = threading.Lock()
opportunities_remaining = TOTAL_BUY_OPPORTUNITIES
_stop_event_ref: Optional[threading.Event] = None
_stream_ref = None

ACCOUNT_BUYING_POWER_USD: Optional[float] = None
CAPITAL_PER_SLOT_USD: float = 0.0
