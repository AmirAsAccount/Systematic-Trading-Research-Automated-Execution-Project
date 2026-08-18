"""
live_catalyst_pipeline.py
--------------------------
LIVE entry-signal pipeline (no order execution — signal + PnL tracking only).

Flow (revised):
  1. Finviz screens the watchlist ONCE at startup on FLOAT / SHORT-FLOAT /
     MIN AVG VOLUME only — price and market cap were dropped as filters since
     float + short float are what actually drive reactivity, not price or
     cap. This list is treated as static for the run: there is no repolling/
     refresh loop — that watchlist rarely changes intraday and repolling was
     pure wasted Finviz load for no practical benefit.
  2. Alpaca's news WEBSOCKET streams live headlines for every ticker on the
     watchlist as they drop. The FDA-approval fast path (#3 below) is
     checked FIRST — a true approval headline is rare and high-value enough
     that it gets first look, ahead of every filter below. Only if that
     doesn't match do these run, before the LLM call:
       - exact dedup by Alpaca's News.id (the same article redelivered).
       - fuzzy per-ticker dedup (difflib.SequenceMatcher, threshold
         NEWS_DEDUP_SIMILARITY_THRESHOLD) against that ticker's recent
         headline history, catching a story REPUBLISHED under a new id.
       - a stale/recap-headline filter (STALE_RECAP_HEADLINE_PATTERNS): a
         cheap dictionary first pass for headlines that report on an
         already-happened price move ("Why Shares Of X Are Up Today") or
         are secondhand roundup/listicle pieces ("20 Stocks That Have Gone
         Up Today And Why") rather than fresh, quantitative information —
         dropped before spending an Ollama call on them. The same Ollama
         call used for the catalyst check (below) is ALSO prompted to flag
         this pattern itself (is_stale_or_secondhand), so paraphrased/
         edge-case recap pieces the dictionary misses are still caught.
  3. FDA-approval fast path (approval decisions ONLY, not designations):
     FDA_APPROVAL_KEYWORDS is now only a CANDIDATE pre-filter — a match must
     ALSO be confirmed by a dedicated qwen prompt (call_ollama_fda_check) as
     a genuine, already-decided, explicit approval before the fast path
     fires, since a pure keyword match was found to be too blunt for
     something that bypasses every other gate. Once confirmed, this is the
     SOLE exception that bypasses every other gate (sentiment, RVOL, halt,
     liquidity, price ceiling) and enters IMMEDIATELY. It also claims ALL
     remaining buying opportunities in one shot (see #6). No M&A fast path —
     M&A run-ups are frequently pre-priced by rumor/leakage before the
     public announcement, so M&A news goes through the normal gates.
  4. Otherwise, entry requires BOTH of the following gates to pass, with no
     waiting period after the news timestamp:
       a. Composite VADER + Loughran-McDonald sentiment gate — headline AND
          summary must EACH
          show P(negative) < NEGATIVE_PROB_MAX and P(positive) > POSITIVE_PROB_MIN.
          This replaces the old "block on negative only" gate: neutral no
          longer passes, only a clearly positive reaction does.
       b. Instant tick-volume gate — fired the moment the sentiment gate
          passes, no 5-10 minute wait, and no waiting for a bar to close
          either. Schwab's quote endpoint is polled once a second starting
          at the news timestamp; cumulative volume-since-news must reach
          TICK_VOLUME_MULTIPLE (2x) of a baseline scaled from the ticker's
          average daily volume, AND at least TICK_UP_VOLUME_RATIO_MIN (60%)
          of that volume must be buy-side (classified via bid/ask on each
          print), within TICK_WINDOW_SEC (60s) of the drop. Each poll's
          success/failure is now tracked separately (data_quality: OK /
          PARTIAL / NO_DATA) so a genuine zero-volume read can be told apart
          from a swallowed API timeout.
     Both must pass for entry to fire. There is no separate delayed
     confirmation step (float rotation / anchored VWAP / swing-structure
     entry checks were removed along with the 5-minute wait).
  5. Before any entry (fast-path or gated), a halt-status check and a bid/ask
     spread (liquidity) check must both pass — EXCEPT the FDA fast path,
     which bypasses these too (see #3).
  6. Capital pool: at startup the script queries the Schwab account's ACTUAL
     available buying power ONCE (see detect_capital_pool()) and splits that
     entire dollar amount into TOTAL_BUY_OPPORTUNITIES (10) equal CAPITAL
     SLOTS (CAPITAL_PER_SLOT_USD each) — not 10 flat 1-share trades. A
     normal (non-FDA) catalyst that clears both gates claims
     MIN_SLOTS_PER_TRADE..MAX_SLOTS_PER_TRADE of those slots, converted to an
     actual share quantity at entry time (claimed_dollars / entry_price),
     sized by a composite MAGNITUDE SCORE (float + pre-news price action +
     LLM confidence/reasoning richness + news category — see
     compute_magnitude_score()) — a concentration bet that a smaller number
     of higher-conviction positions can outperform spreading the same
     capital thinly across many marginal entries, at the cost of more
     variance if the score is a poor predictor. Magnitude scoring ONLY
     affects position SIZE, never whether an entry happens — it only even
     runs for a catalyst that already cleared both gates. An FDA-approval
     fast-path hit is still the sole exception: it claims and buys with ALL
     slots (i.e. the entire remaining capital pool) remaining at that
     instant, unaffected by magnitude scoring. Multiple tickers can have
     open positions concurrently — this is no longer a single-trade-per-run
     script.
  7. Entry places a real LIMIT order via the Schwab Trader API (see
     SchwabClient.place_equity_order) and is logged (green circle in the
     console + a row in trades_log.csv).
  8. Every open position is polled once a minute against Schwab price history
     and exited on whichever comes first:
       - halted (skip check, don't force an exit into a halt — logged only)
       - a SET (non-trailing) entry-volatility stop: distance =
         ENTRY_STOP_ATR_MULTIPLE x ATR(ENTRY_STOP_ATR_PERIOD) measured ONCE
         at entry, capped at MAX_LOSS_PCT (10%) if volatility implies wider,
         floored at MIN_STOP_PCT if it implies tighter — fixed for the
         ENTIRE trade, never recalculated or trailed with price afterward.
         The prior volume-confirmed Chandelier ATR TRAIL is REMOVED: this
         strategy trades float-rotation setups whose intraday volatility
         regime doesn't resemble whatever regime a trailing ATR's own
         lookback sample assumes, so the trail was either badly too tight or
         badly too loose depending on the mismatch — not a calibration
         problem, a mechanism problem.
       - a SECONDARY exit — significant bearish CVD OR bearish OBV/price
         swing divergence — that only ARMS once EITHER local Pacific time
         reaches SECONDARY_EXIT_PST_HOUR (6am) OR price has moved
         SECONDARY_EXIT_MOVE_TRIGGER_PCT away from entry, and is checked
         only against bars AT OR AFTER entry_time (not the full day's
         history) — both fix a previously-reported false first-exit that
         fired immediately after entry off pre-entry bearish structure. The
         SET stop above stays active the entire time regardless of whether
         the secondary exit has armed yet.
  9. The run considers itself complete once TOTAL_BUY_OPPORTUNITIES have all
     been used AND every resulting position has been closed, at which point
     it shuts down.

Tuple logging of (features -> realized outcome) for future weight-fitting /
sentiment-model re-tuning is intentionally NOT implemented yet — deferred
per a later decision, not forgotten.

REQUIREMENTS
  pip install alpaca-py vaderSentiment pysentiment2 pandas numpy requests finviz

  Tier-2 sentiment escalation additionally requires Ollama running locally:
    1. Install Ollama: https://ollama.com/download
    2. Pull the model this script uses:  ollama pull qwen3:8b
    3. Make sure the service is running (`ollama serve`, or the Ollama app)
       before starting this script — checked automatically at [INIT], but
       will otherwise fail-closed (no entry) on every escalation attempt.

SECURITY NOTE
  Credentials are read from environment variables ONLY. No hardcoded fallbacks.

ENV VARS REQUIRED
  ALPACA_API_KEY, ALPACA_API_SECRET
  SCHWAB_MARKETDATA_CLIENT_ID, SCHWAB_MARKETDATA_CLIENT_SECRET, SCHWAB_MARKETDATA_REFRESH_TOKEN
  SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET, SCHWAB_TRADER_REFRESH_TOKEN
  SCHWAB_ACCOUNT_HASH

Two SEPARATE Schwab developer apps are used: one holding "Market Data
Production" (quotes/price history), one holding "Accounts and Trading
Production" (account hash + order placement) — Schwab caps how many apps
per account can hold the Trader product, so this is split across two apps
rather than fighting that quota, and it has the side benefit of giving
quote polling and order placement independent rate-limit budgets.

Getting a Schwab refresh token the first time requires one interactive login
per app (Schwab's OAuth doesn't support a fully headless flow). Run:
    python live_catalyst_pipeline.py --schwab-login --app marketdata
    python live_catalyst_pipeline.py --schwab-login --app trader
and follow the prompts once for each; each refresh token is valid ~7 days
and this script only ever uses the refresh-token grant after that (no
browser needed again until it expires).

VERIFY BEFORE RELYING ON THESE (flagged inline where they appear):
  - Finviz screener filter codes (sh_float_u10 / sh_short_o20 / sh_avgvol_o100)
    — build the screen once in Finviz's own UI and confirm the resulting URL
    matches, since these codes occasionally change.
  - The Schwab quote field used for halt status (`securityStatus`) — confirm
    against Schwab's current Trader API schema; this script fails open
    (treats unknown/missing status as NOT halted) if the field is absent.
  - The Schwab account field(s) used for buying power (`buyingPower` /
    `cashAvailableForTrading` / `cashBalance` on GET /accounts/{hash}) — see
    get_account_buying_power(). Unlike the halt-status field, this one FAILS
    CLOSED (raises and refuses to start the run) rather than failing open,
    since guessing a capital figure wrong would size every trade off it.
"""

import argparse
import asyncio
import base64
import csv
import difflib
import json
import logging
import math
import os
import re
import sys
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
except ImportError:  # pragma: no cover
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

# Load a local .env file if python-dotenv is installed and one exists next to
# this script. Falls back silently to shell-exported env vars if dotenv isn't
# installed or no .env file is present — this does NOT replace the
# "env vars only, no hardcoded secrets" policy, it's just a convenient way to
# set those env vars from a file instead of `export`.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NY_TZ = ZoneInfo("America/New_York")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — secrets from environment ONLY, no hardcoded fallbacks
# ══════════════════════════════════════════════════════════════════════════

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET")

# Two SEPARATE Schwab apps: one holding "Market Data Production" (quotes,
# price history), one holding "Accounts and Trading Production" a.k.a.
# "Retail Trader API Production" (account hash, order placement). Schwab
# caps how many apps can hold the Trader product per developer account, so
# this is split across two apps/credential sets rather than one — this also
# means quote polling and order placement draw from independent rate-limit
# budgets instead of sharing one.
SCHWAB_MARKETDATA_CLIENT_ID     = os.environ.get("SCHWAB_MARKETDATA_CLIENT_ID")
SCHWAB_MARKETDATA_CLIENT_SECRET = os.environ.get("SCHWAB_MARKETDATA_CLIENT_SECRET")
SCHWAB_MARKETDATA_REFRESH_TOKEN = os.environ.get("SCHWAB_MARKETDATA_REFRESH_TOKEN")
SCHWAB_MARKETDATA_REDIRECT_URI  = os.environ.get("SCHWAB_MARKETDATA_REDIRECT_URI", "https://127.0.0.1")

SCHWAB_TRADER_CLIENT_ID     = os.environ.get("SCHWAB_TRADER_CLIENT_ID")
SCHWAB_TRADER_CLIENT_SECRET = os.environ.get("SCHWAB_TRADER_CLIENT_SECRET")
SCHWAB_TRADER_REFRESH_TOKEN = os.environ.get("SCHWAB_TRADER_REFRESH_TOKEN")
SCHWAB_TRADER_REDIRECT_URI  = os.environ.get("SCHWAB_TRADER_REDIRECT_URI", "https://127.0.0.1")
SCHWAB_ACCOUNT_HASH  = os.environ.get("SCHWAB_ACCOUNT_HASH")   # required for live order placement;
                                                                 # get it via --schwab-account-hash --app trader

# ---- LIVE MULTI-ENTRY MODE ---------------------------------------------------
# The pipeline places real orders (not signal-only). At startup it queries
# the Schwab account's actual available buying power ONCE (see
# get_account_buying_power() / detect_capital_pool() below — same
# "detected/built once at startup, static for the run" pattern the Finviz
# watchlist already uses) and divides that entire amount into
# TOTAL_BUY_OPPORTUNITIES (10) equal CAPITAL SLOTS. This is a pool of
# DOLLARS split into slots, not a flat share count — a single
# high-conviction catalyst can now claim up to MAX_SLOTS_PER_TRADE (10, i.e.
# the WHOLE detected capital pool) in one entry; see the magnitude-scoring
# section below. This replaces the old flat "every catalyst = exactly 1
# share" allocation: with small starting capital and a mostly-bullish news
# tape, spreading the whole pool across many marginal 1-share positions
# dilutes the few names that would actually run 20-30%+ down to the same
# size as the ones that barely move. Position size per normal (non-FDA)
# entry is now MIN_SLOTS_PER_TRADE..MAX_SLOTS_PER_TRADE capital slots
# (converted to an actual share quantity at entry time via
# claimed_dollars / entry_price — see enter_position), scaled by the same
# composite magnitude score as before (float, pre-news price action, LLM
# confidence + reasoning richness, news category). An FDA-approval fast-path
# hit is still the sole exception: it bypasses every other gate AND claims
# ALL slots (i.e. the entire remaining capital pool) remaining at that
# instant in one order (see claim_fda_all_remaining()), unaffected by
# magnitude scoring. The run is "complete" once all slots are used AND every
# resulting position has closed.
TOTAL_BUY_OPPORTUNITIES = 10    # total CAPITAL SLOTS the detected buying power is split into
MIN_SLOTS_PER_TRADE     = 1     # floor: even the lowest-magnitude passing signal gets this many
MAX_SLOTS_PER_TRADE     = 10    # cap per single NORMAL (non-FDA) entry — a max-conviction signal
                                 # can claim the entire capital pool in one trade, by design
CAPITAL_POOL_FALLBACK_USD = None   # if buying-power detection fails at startup, the run ABORTS
                                    # rather than guessing a number — see detect_capital_pool().
                                    # Set this to a fixed USD amount ONLY if you want a manual
                                    # override to still allow the run to start despite that.
EXPECTED_BUYING_POWER_FIELD = "cashAvailableForTrading"  # the currentBalances field detect_capital_pool()
                                    # is expected to size the run's ENTIRE capital pool off of —
                                    # "buyingPower" assumes a MARGIN account. If this account is
                                    # actually a CASH account, set this to
                                    # "cashAvailableForTrading" instead (a config choice you make
                                    # deliberately, not something get_account_buying_power() should
                                    # guess for you).
ALLOW_BUYING_POWER_FIELD_FALLBACK = False   # get_account_buying_power() tries buyingPower ->
                                    # cashAvailableForTrading -> cashBalance in order (see its
                                    # docstring). If EXPECTED_BUYING_POWER_FIELD above is missing
                                    # and it has to fall through to a LOWER-priority field, it now
                                    # FAILS LOUD (raises) by default rather than silently sizing
                                    # every trade off a field nobody chose — this was previously a
                                    # silent fallback with no logging of which field got used at
                                    # all. Set True only once you've confirmed the fallback field
                                    # really is what you want this run to size off of.
LIMIT_ORDER_SLIPPAGE_PCT = 0.01  # 1% marketable-limit buffer over ask (buys) / under bid
                                  # (sells). Schwab does not accept MARKET orders outside
                                  # the 9:30am-4:00pm ET regular session, and rejects them
                                  # outright in pre/post-market — every order this script
                                  # places must be a LIMIT order, so this buffer keeps it
                                  # realistically fillable without being unbounded.

# ---- Magnitude scoring (position SIZE only — never gates entry) -------------
# Deliberately scoped to SIZING, not gating: this only runs for a catalyst
# that has ALREADY cleared the sentiment gate AND the instant tick-volume
# gate — it never adds a new way to block or approve an entry, which keeps
# the extra scoring work off the hot path for the (large majority of)
# headlines that never reach this point anyway.
#
# Four inputs, per a deliberate choice to leave OUT the instant RVOL multiple
# achieved and the up-volume ratio: both are real-time/short-window reads
# that are noisy signals of the FOLLOWING minutes at best, not of the
# eventual size of the whole move — they stay as pass/fail GATES only
# (unchanged) and are not part of this composite.
#   1. FLOAT — lower float => higher score (already screened < FLOAT_CEILING_SHARES
#      at watchlist-build time; this scores the DEGREE within that ceiling).
#   2. PRE-NEWS PRICE ACTION — % price change in the PRICE_ACTION_LOOKBACK_MIN
#      minutes immediately before the news hit. ASSUMPTION, NOT VALIDATED:
#      this scores existing pre-news UPWARD momentum higher, on a
#      "catalyst-on-top-of-momentum extends further" thesis. This is exactly
#      the kind of open question flagged when this was discussed — flip the
#      sign in price_action_score() below if evidence points the other way.
#   3. LLM CONFIDENCE + REASONING RICHNESS — qwen3's own confidence
#      (high/medium/low) plus a crude length-based proxy for how much
#      substantive justification it gave, on the theory that a thin
#      one-line justification is weaker signal than a detailed one even at
#      the same stated confidence level.
#   4. NEWS CATEGORY — a tiered keyword scorer (regulatory/M&A/financing/
#      earnings/etc.), same technique as the existing FDA_APPROVAL_KEYWORDS
#      / RED_FLAG_KEYWORDS hard-match lists above.
# Weights below are a reasoned STARTING POINT, not a fitted model (per the
# module docstring, outcome-tuple logging for future weight-fitting is
# intentionally deferred, not forgotten) — expect to revisit these once
# enough realized trades exist to check them against actual outcomes.
MAGNITUDE_WEIGHT_FLOAT         = 0.25
MAGNITUDE_WEIGHT_PRICE_ACTION  = 0.25
MAGNITUDE_WEIGHT_LLM           = 0.30
MAGNITUDE_WEIGHT_NEWS_CATEGORY = 0.20   # (sums to 1.0)

PRICE_ACTION_LOOKBACK_MIN = 20    # minutes of pre-news 1m bars examined for momentum
PRICE_ACTION_CLIP_PCT     = 10.0  # pre-news % move beyond +/- this is clipped before scoring

LLM_CONFIDENCE_SCORE            = {"high": 1.0, "medium": 0.6, "low": 0.3}
LLM_REASONING_RICHNESS_WORD_CAP = 40   # reasoning word count at/above this maps to full richness score
LLM_CONFIDENCE_VS_RICHNESS_MIX  = 0.7  # weight on confidence within the LLM sub-score (rest -> richness)

# Tiered keyword categories for the news-category sub-score: (category_name,
# score, keyword-list). Matched case-insensitively as plain substrings
# against "headline + summary"; the HIGHEST-scoring matched tier wins if
# more than one matches. Anything positive-but-uncategorized (i.e. it
# already cleared the sentiment gate but matches none of these) gets
# NEWS_CATEGORY_BASELINE_SCORE. Not exhaustive — add tiers/keywords as
# patterns are observed; VERIFY these phrasings against real headlines
# before relying on them, same caveat as FDA_APPROVAL_KEYWORDS below.
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
NEWS_CATEGORY_BASELINE_SCORE = 0.3   # positive-but-uncategorized catalyst

# ---- Stale/recap news filter (excluded from scoring entirely — not sized,
# not entered) --------------------------------------------------------------
# Headlines that REPORT ON a price move that already happened ("Why X Shares
# Are Up Today", "X Shares Surge: Here's What's Driving It") or that are
# secondary/roundup reporting rather than a primary, quantitative disclosure
# ("20 Stocks That Have Gone Up Today And Why") read as sentiment-positive to
# VADER/LM (and often to an LLM unless told to check) but carry no fresh
# information — the move they describe has already happened by the time the
# tick-volume gate would poll for it, or the "story" is just secondhand
# commentary rather than a company disclosure.
#
# This dictionary is kept as a cheap, zero-latency FIRST pass (still catches
# the obvious cases instantly, before spending an Ollama round trip) — it is
# no longer the ONLY check. Every headline that survives it still goes
# through call_ollama_catalyst_check(), which is now ALSO prompted (using
# these same example patterns) to flag secondhand/recap/listicle framing
# qwen recognizes but that doesn't literally match a phrase below — see
# _OLLAMA_CATALYST_PROMPT's "is_stale_or_secondhand" field and its use in
# handle_news_event(). That combination is meant to cover the edge cases a
# pure substring dictionary can't (paraphrased listicles, recap pieces that
# don't use any of these exact phrases, etc.) without giving up the instant
# reject on the obvious cases.
STALE_RECAP_HEADLINE_PATTERNS = [
    "why shares of", "why is", "why are shares", "here's why",
    "here's what's driving", "shares surge:", "shares soar:", "shares jump:",
    "shares are up today", "shares are down today", "stocks moving today",
    "stocks that moved", "explained:", "stocks that have gone up",
    "stocks that have gone down", "stocks to watch today", "stocks in focus",
    "roundup:", "here's what's behind",
]

# ---- Watchlist screen -------------------------------------------------------
PRICE_CEILING_USD      = 10.0         # caps per-share cost of the 1-share test trade
FLOAT_CEILING_SHARES   = 10_000_000   # float under 10M — kept tight per request
MIN_AVG_VOLUME_SHARES  = 100_000      # liquidity floor so positions are exitable — was
                                       # 50K; bumped to 100K to match the _finviz_url()
                                       # screen change (bugfix: this constant had drifted
                                       # out of sync with the actual scrape filter, so the
                                       # startup banner was printing the wrong number)
# ---- Rolling-high breakout path (parallel to the news pipeline) ------------
# A second, independent set of entry triggers — NOT part of handle_news_event()
# and does not touch any FDA/stale-recap/tick-volume news logic. Replaced the
# short-float/days-to-cover screen (that only ever caught the short-covering
# subset of explosive movers) with a live RVOL/gap scanner + halt-reopen
# detector, since most 100-200% low-float moves have nothing to do with short
# interest — see the big comment above the MARKET SCAN section further down.
#   TIER 1: RVOL/gap live scanner (market_scan_and_update) — batched quotes
#           across the WHOLE watchlist, refreshed every MARKET_SCAN_INTERVAL_SEC.
#           ALSO detects halt->reopen transitions in the same scan (see the
#           separate HALT sub-path further down — it does NOT require Tier 2).
#   TIER 2: a confirmed-bullish catalyst within OLD_BULLISH_NEWS_LOOKBACK_DAYS,
#           via Alpaca's HISTORICAL news endpoint + the same qwen check the
#           news pipeline uses — NOT just whatever streamed live this run.
#           Required for the RVOL/gap path; NOT required for the halt path
#           (a halt is its own catalyst, confirmed bullish by the reopen
#           volume split — see the HALT SUB-PATH comment for why).
#   TIER 3: a break of this session's high, confirmed by the same tick-volume
#           gate as news — the only stage polled every few seconds.
GAP_PCT_MIN = 10.0                     # % gap vs previous close to flag as a Tier 1
                                        # candidate — a starting heuristic (see
                                        # citations from the design discussion),
                                        # not a backtested threshold; tune once
                                        # you have logged hit-rate data
RVOL_MIN = 3.0                         # cumulative volume vs a time-of-day-scaled
                                        # baseline — 3x is the low end of what
                                        # scanner vendors flag as "unusual"
PREMARKET_MIN_VOLUME_SHARES = 20_000   # floor so 2-3 prints alone can't qualify
MARKET_SCAN_INTERVAL_SEC = 30          # Tier 1 (RVOL/gap) + halt-transition
                                        # detection — how often the WHOLE
                                        # watchlist gets batch-quoted
QUOTE_BATCH_CHUNK_SIZE = 100           # symbols per batched /quotes request —
                                        # VERIFY against current Schwab docs
                                        # for the actual max
OLD_BULLISH_NEWS_LOOKBACK_DAYS = 18    # ~2-3 weeks per request — a confirmed-positive
                                        # catalyst (already scored by call_ollama_catalyst_check)
                                        # must have landed within this many days for a
                                        # RVOL/gap candidate to be a breakout candidate at all
NEWS_TIER_REFRESH_SEC = 5 * 60         # Tier 2 — news can land anytime, so this
                                        # runs more often than Tier 1
ROLLING_HIGH_POLL_INTERVAL_SEC = 5     # Tier 3 only — how often each ticker still
                                        # standing after Tiers 1-2 gets its quote re-polled

# ---- Breakout magnitude scoring (SIZING ONLY for Path B, mirrors
# compute_magnitude_score's role for Path A) ----------------------------------
# Path B has no fresh LLM catalyst read to score (the catalyst, if any, is old
# news by design), so sizing instead scores (a) how strongly the tick-volume
# confirmation passed and (b) how large the gap/RVOL reading was that got the
# ticker flagged in the first place. Shared by BOTH the RVOL/gap breakout path
# and the halt-reopen path, per request.
BREAKOUT_WEIGHT_CONFIRMATION = 0.5     # (sums to 1.0 with the weight below)
BREAKOUT_WEIGHT_GAP_SCANNER  = 0.5
GAP_PCT_SCORE_CLIP = 50.0              # gap% saturates the scoring component above this
RVOL_SCORE_CLIP    = 10.0              # RVOL saturates the scoring component above this

# ---- Sentiment gate (VADER + Loughran-McDonald) — a COMPOSITE gate ----------
# Entry requires BOTH conditions, checked on the worse of headline/summary in
# each direction: negative must clear the ceiling AND positive must clear the
# floor. See the CALIBRATION NOTE further down (by score_text_full) for how
# these values were derived — they are NOT the same 0.10/0.80 that worked for
# FinBERT's calibrated softmax; those values would functionally block almost
# every headline under this dictionary-based ensemble (verified empirically).
# TREAT THESE AS A STARTING POINT — backtest against trades_log.csv before
# trusting this live, and expect this stack to still miss flat, factual
# positive news (numeric beats, guidance raises) that lacks emotionally-
# loaded language, until the LLM-escalation step is added on top of this.
NEGATIVE_PROB_MAX = 0.30   # block entry if P(negative) >= this, from EITHER
                            # headline or summary (worst case of the two)
POSITIVE_PROB_MIN = 0.65   # block entry if P(positive) <= this, from EITHER
                            # headline or summary (worst case of the two, i.e.
                            # the LOWER of the two positive scores must still
                            # clear 0.65)

# ---- Tier 2: local LLM escalation (Ollama, runs entirely on-machine) --------
# REVISED FLOW: tier 2 (the LLM) is now the PRIMARY gate. Every headline that
# reaches handle_news_event() is sent to Ollama first — tier 1 (VADER +
# Loughran-McDonald) alone can no longer approve an entry on its own. Tier 1
# is now a SECONDARY confirmation step that only runs after Ollama has
# already said catalyst=positive:
#   - Ollama says positive + confidence=high  -> entry proceeds regardless of
#     what tier 1 says (a confident LLM read overrides the dictionary stack,
#     which is known to be blind to flat/factual good news).
#   - Ollama says positive + confidence=medium/low -> tier 1 must ALSO pass
#     (P(negative) < NEGATIVE_PROB_MAX AND P(positive) > POSITIVE_PROB_MIN)
#     for entry to proceed; a lower-confidence LLM read needs the dictionary
#     stack's agreement before firing.
#   - Ollama says neutral/negative, or is unreachable/unparseable -> no entry,
#     fail-closed, tier 1 is never even consulted.
# Model asks for DIRECTION/legitimacy only ("is this a real positive
# catalyst?"), never magnitude — magnitude is what the instant-RVOL gate is
# for; text alone (local or frontier) cannot reliably size a move.
OLLAMA_ENABLED     = True
OLLAMA_URL         = "http://localhost:11434/api/generate"
OLLAMA_MODEL       = "qwen3:8b"
OLLAMA_TIMEOUT_SEC = 8      # local inference should be fast; a stall here must
                             # not hang the no-wait entry path — fail closed
                             # (no entry) rather than block indefinitely.

# ---- FDA fast path (approval decisions ONLY — no M&A fast path) -----------
# NOTE (bugfix/behavior change): these two lists are now a CANDIDATE
# pre-filter only, not the final decision. contains_fda_fastpath() (below)
# is cheap/instant and decides whether it's even worth spending an Ollama
# round trip — but a keyword hit no longer bypasses every gate by itself.
# Real wire headlines have enough edge cases (designations phrased to sound
# like approvals, approvals phrased in ways that don't hit these exact
# strings, etc.) that a pure dictionary match was judged too blunt for
# something that skips every other safety gate. Every candidate is now also
# confirmed by call_ollama_fda_check() (see TIER 2 section below), which is
# given these same two lists as explicit guidance in its prompt, and the
# bypass only fires if qwen agrees this is a genuine, already-decided,
# explicit FDA approval. See handle_news_event() for the actual gate order.
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
# NOTE: deliberately no M&A fast path. Target-stock run-ups are well documented
# as substantially rumor/leakage-driven ahead of the public announcement
# (Jarrell & Paulsen; Pound & Zeckhauser; Tang & Xu), so by the time a public
# M&A headline prints, a meaningful chunk of the move may already be priced
# in — M&A news is routed through the normal reaction-confirmation gate below,
# not bypassed.

# ---- Sentiment-gate hard-negative override (curated phrases, NOT raw LM word-counting) ----
# Testing found that Loughran-McDonald's raw negative-word COUNT cannot
# discriminate real red-flag news from generic sector vocabulary at headline
# length: "Defense Contractor Secures Multi-Year Missile Systems Contract"
# and "Company Defaults On Debt Covenant" both scored 2 LM-negative-word hits,
# while a genuine "Filed For Chapter 11 Bankruptcy" headline only scored 1 —
# there is no hit-count threshold that separates these (LM's dictionary is
# validated for full 10-K/10-Q sections spanning thousands of words, where
# noise averages out, not single sentences). So instead of gating on LM's
# word COUNT, the hard override below fires only on specific, unambiguous
# red-flag phrases — the same technique the FDA fast path above already uses
# successfully. LM's Positive/Negative/Polarity are still computed and
# returned by score_text_full() for logging/visibility, just not used to
# gate on their own.
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

# ---- Instant tick-volume gate (the second, no-wait trigger) -----------------
# Replaces the old bar-based instant RVOL gate. That version still had to wait
# for a Schwab 1-minute bar to exist before it could measure anything, which
# is where the lag was coming from — bars are the wrong data source for an
# "instant" check. This version polls Schwab's QUOTE endpoint directly (same
# get_quote_full() used for halt/liquidity checks) once a second, accumulates
# raw volume-since-news from the quote's cumulative totalVolume field, and
# splits that volume into up/down using the bid/ask of each print. No bar
# ever needs to close.
TICK_POLL_INTERVAL_SEC      = 1.0     # how often we hit the quote endpoint after news_time
TICK_WINDOW_SEC             = 60      # stop polling (no-entry) if neither condition below
                                       # has fired within this many seconds of news_time.
                                       # Widened from 30s -> 60s: gives the volume/up-ratio
                                       # read more time to actually confirm before giving up,
                                       # at the cost of being slightly later on the fastest-
                                       # firing genuine moves.
TICK_VOLUME_MULTIPLE        = 4.0     # volume-since-news must be >= this x the PRIMARY
                                       # (time-of-day-matched) baseline — see RVOL entry-signal
                                       # block below
TICK_VOLUME_SESSION_CONFIRM_MULTIPLE = 2.0   # AND >= this x the flat full-session-average
                                       # baseline (CONFIRMATION layer — see RVOL entry-signal
                                       # block below). Looser than TICK_VOLUME_MULTIPLE on
                                       # purpose: its job is only to rule out "this is a big
                                       # multiple of basically nothing," not to re-litigate the
                                       # primary threshold.
TICK_UP_VOLUME_RATIO_MIN    = 0.60    # AND at least this fraction of that volume must be
                                       # buy-side (up) rather than sell-side (down)
SESSION_SECONDS_FOR_SCALING = 23_400  # 6.5h regular session (9:30-16:00 ET). Used by
                                       # scaled_volume_baseline() to scale a ticker's average
                                       # DAILY volume down to a window-sized baseline — this is
                                       # a flat scale-down, NOT time-of-day-matched (real
                                       # intraday volume is U-shaped: heavier at the open/close,
                                       # thin overnight/premarket). It is no longer the primary
                                       # RVOL baseline (see RVOL_TOD_LOOKBACK_DAYS below); it now
                                       # serves only as the CONFIRMATION layer — "is the book
                                       # actually deep right now, or is this just what's normal
                                       # for 4am" — since a name that never trades much this
                                       # early could otherwise show a huge multiple over its own
                                       # dead-quiet time-of-day norm on a trivial handful of
                                       # shares.

# ---- RVOL entry signal: time-of-day-matched baseline (PRIMARY) + flat
# full-session-average baseline (CONFIRMATION) --------------------------------
# PRIMARY: compares the current window's volume to the average volume this
# SAME ticker traded in the same clock-time window on each of the last
# RVOL_TOD_LOOKBACK_DAYS prior sessions (see
# time_of_day_matched_volume_baseline / rvol_dual_check). This replaces the
# flat-session-average approximation as the actual "is this unusual RIGHT
# NOW" comparison — comparing 4:00-4:01am premarket volume to a flat
# fraction of the full day's average wildly overstates RVOL at that hour
# and understates it near the open/close.
# CONFIRMATION: the OLD flat scaled_volume_baseline() approximation, checked
# alongside the primary reading so a candidate can't clear the gate purely
# by beating a near-zero time-of-day norm — both must pass.
RVOL_TOD_LOOKBACK_DAYS   = 10     # prior trading sessions averaged into the TOD-matched baseline
MIN_TOD_SAMPLE_DAYS      = 3      # minimum usable prior-day samples before trusting the TOD
                                   # baseline at all; below this (e.g. a recent IPO, or a data
                                   # gap), fall back to the flat session baseline for BOTH
                                   # readings rather than blocking entry outright
RVOL_SESSION_CONFIRM_MIN = 1.5    # Tier-1/halt-reopen confirmation floor against the flat
                                   # full-session-average baseline (looser than RVOL_MIN on
                                   # purpose — same reasoning as TICK_VOLUME_SESSION_CONFIRM_MULTIPLE)
PREMARKET_OPEN_ET        = dt_time(4, 0)   # 4:00 AM ET — shared reference point for both the
                                   # elapsed-time approximation (_elapsed_seconds_since_premarket_open)
                                   # and the TOD-matched baseline's window start for the
                                   # cumulative-since-open RVOL checks (Tier 1 / halt-reopen)

# IMPORTANT — confirmed against Schwab's own order-type documentation
# (schwab.com "Mastering order types: limit orders" / international.schwab.com
# "Stock order types and conditions"): "Day + extended hours" (session=
# SEAMLESS) orders — what place_equity_order() submits for EVERY entry and
# exit — are only active 7:00 AM-8:00 PM ET. This bot watches/trades from
# PREMARKET_OPEN_ET (4:00 AM ET), a full 3 HOURS before Schwab's seamless
# session actually opens. An entry or exit signaled in that 4:00-7:00 AM ET
# window can be ACCEPTED by the API but is NOT necessarily eligible to
# route/fill until 7:00 AM ET — a real gap between "signal fired at price X"
# and "the order could actually transact," during which price can move well
# away from X. This is very likely why a reported exit's logged price/P&L
# didn't match what price action actually did: the "exit" printed in earlier
# versions was the SIGNAL's reference price, not a confirmed fill (see the
# exit_position rewrite below, which no longer reports/closes a position
# until a fill is actually confirmed).
SCHWAB_SEAMLESS_SESSION_START_ET = dt_time(7, 0)   # earliest ET time Schwab's seamless
                                   # (Day+extended) orders are actually eligible to fill

SWING_WINDOW_EXIT        = 2          # window used for exit-side swing-low / divergence checks

# ---- Exit logic (SET entry-volatility stop + gated secondary exit) ----------
# The Chandelier ATR TRAIL is REMOVED entirely — in practice it was a
# large-scale failure: this strategy trades float-rotation setups, and a
# volatility TRAIL keeps re-measuring ATR throughout the hold, which only
# works if the position's realized intraday volatility regime resembles
# whatever regime the ATR's own lookback sample was built from. Float
# rotation doesn't behave like the calmer/steadier regime a generic ATR
# trail assumes, so the trail was either badly too tight (stopped out on
# ordinary float-rotation noise) or badly too loose (gave back far more
# than intended), depending on which way the mismatch happened to cut on a
# given name/day — not fixable by retuning the multiplier, since the
# mismatch is in the trailing MECHANISM itself, not its calibration.
#
# Replacement — two conditions now:
#   1. A SET (non-trailing) stop-loss, sized ONCE at entry from that
#      moment's volatility (compute_entry_volatility_stop): distance =
#      ENTRY_STOP_ATR_MULTIPLE x ATR(ENTRY_STOP_ATR_PERIOD), capped at
#      MAX_LOSS_PCT (10%) if volatility implies something wider, floored at
#      MIN_STOP_PCT so a near-zero ATR reading can't produce a degenerate
#      near-0% stop. This price is fixed at entry and does NOT move with
#      price afterward — active for the ENTIRE duration of the trade.
#   2. A SECONDARY exit — significant bearish CVD (significant_cvd_bearish)
#      OR bearish OBV/price swing divergence (bearish_swing_divergence,
#      unchanged) — that only ARMS once EITHER SECONDARY_EXIT_PST_HOUR
#      (local Pacific time) has passed OR price has moved
#      SECONDARY_EXIT_MOVE_TRIGGER_PCT away from entry (see
#      _secondary_exit_armed). Before that, condition 1 is the ONLY active
#      exit. This also fixes the reported false first-exit-of-the-day: the
#      old check ran divergence against the full day's bars (including
#      bars from BEFORE the position existed, where bearish-looking
#      structure right before a breakout catalyst is common and isn't
#      informative about the trade itself) from the very first poll after
#      entry; both the entry-time bar filter below AND the arming gate
#      remove that false trigger.
ENTRY_STOP_ATR_PERIOD    = 14     # bars used to measure volatility ONCE, at entry, for the SET stop
ENTRY_STOP_ATR_MULTIPLE  = 2.5    # SET stop distance = this x ATR(ENTRY_STOP_ATR_PERIOD) at entry
MAX_LOSS_PCT             = 0.10   # hard CAP: SET stop is never wider than this off entry, even if
                                   # volatility implies more
MIN_STOP_PCT             = 0.02   # floor: SET stop is never tighter than this off entry, even if
                                   # volatility implies less (guards against a near-zero ATR read)
SWING_WINDOW_EXIT        = 2      # window used for the secondary exit's swing-divergence check
EXIT_LOOKBACK_BARS       = 30     # cap on how many since-entry bars are examined for the secondary
                                   # exit signals (CVD / divergence) — NOT used for the SET stop,
                                   # which is measured once at entry and never recalculated
PT_TZ                    = ZoneInfo("America/Los_Angeles")   # used for SECONDARY_EXIT_PST_HOUR —
                                   # "6am PST" is treated as 6am US-Pacific LOCAL time year-round
                                   # (i.e. PDT in summer), the common colloquial usage; adjust here
                                   # if you specifically mean fixed UTC-8 regardless of DST
SECONDARY_EXIT_PST_HOUR  = 6      # secondary exit (CVD/divergence) arms once local Pacific clock
                                   # time reaches this hour, OR the move-trigger below fires,
                                   # whichever comes first
SECONDARY_EXIT_MOVE_TRIGGER_PCT = 5.0   # OR arms early if price has moved this % (either
                                   # direction) away from entry, regardless of clock time
CVD_LOOKBACK_BARS        = 5      # bars examined for the "significant CVD bearish" magnitude check
CVD_BEARISH_THRESHOLD    = -0.50  # net CVD ratio (up_volume - down_volume) / total_volume over
                                   # CVD_LOOKBACK_BARS must fall to/below this to count as
                                   # "significant" bearish CVD (-0.50 == down-volume outweighing
                                   # up-volume by more than 3-to-1 over the window)

# ---- Exit ORDER FILL confirmation (bugfix) -----------------------------------
# Previously exit_position() placed one SELL limit order and never confirmed
# it actually filled — the position was dropped from open_positions
# immediately regardless, so a limit that never got hit (price moved away
# before the fill) could leave the position still open at the broker while
# the script believed it was flat. This now polls the order status for
# EXIT_FILL_CHECK_SEC after each attempt; if it hasn't filled by then, the
# order is canceled and resubmitted at a wider (more aggressive) discount off
# the bid, up to EXIT_FILL_MAX_ATTEMPTS times, so an exit signal reliably
# results in a closed position rather than a stuck unfilled order.
EXIT_FILL_CHECK_SEC         = 5      # seconds to wait for a fill before widening the limit
EXIT_FILL_POLL_INTERVAL_SEC = 1.0    # how often to poll order status within that window
EXIT_LIMIT_WIDEN_STEP_PCT   = 0.01   # each retry widens the discount off the bid by +1% more
EXIT_FILL_MAX_ATTEMPTS      = 5      # after this many widened attempts, stop auto-widening and warn

# ---- Liquidity gate ----------------------------------------------------------
MAX_SPREAD_PCT = 0.03   # block entry if (ask-bid)/mid exceeds this — tune to taste;
                         # low-float names run wide spreads by nature

# ---- Misc / infra -------------------------------------------------------------
POLL_INTERVAL_SEC  = 20     # PnL / exit-check cadence per open position
HEARTBEAT_SEC      = 30

FINVIZ_PAGE_DELAY_SEC   = 1.5
FINVIZ_BATCH_SIZE       = 3
FINVIZ_COOLDOWN_SEC     = 5.0
FUNDAMENTALS_CACHE_TTL_SEC = 60 * 60   # re-scrape a ticker's float/short-float at most hourly

CACHE_DIR  = Path("cache")
TRADES_LOG = Path("trades_log.csv")
CACHE_DIR.mkdir(exist_ok=True)

_positions_lock = threading.Lock()
open_positions: dict = {}   # ticker -> Position

# Guards against a race introduced by exit_position() no longer removing a
# position from open_positions until a fill is confirmed: the fill-retry
# loop (up to EXIT_FILL_MAX_ATTEMPTS x EXIT_FILL_CHECK_SEC ~= 25s) can run
# LONGER than POLL_INTERVAL_SEC (20s), so monitor_positions_loop's next
# cycle could otherwise call exit_position() again for the SAME ticker
# while the first attempt is still in flight, risking a duplicate SELL.
_exiting_lock = threading.Lock()
_exiting_tickers: set = set()

_watchlist_lock = threading.Lock()
watchlist: set = set()

# Alpaca's news websocket has no documented hard symbol-count limit, but
# subscribing hundreds of symbols in a single message is a known failure
# mode in practice (msgpack payload exceeds a practical byte/frame ceiling,
# or the connection resets mid-send) — multiple independent reports of
# partial or zero delivery when subscribing 500+ symbols in one shot on
# Alpaca's streaming infra. Subscribing in smaller batches, sent AFTER the
# stream is actually running (not queued before .run()), avoids this.
NEWS_SUBSCRIBE_CHUNK_SIZE = 100

# ---- Websocket reconnect (bugfix) -------------------------------------------
# The news websocket can silently time out / drop mid-run (idle-connection
# server-side timeout, brief network blip, etc.). Previously any exception out
# of stream.run() other than KeyboardInterrupt fell straight through to
# main()'s finally block and ended the entire run with no attempt to
# reconnect. See the reconnect loop in main() for the fix.
WS_RECONNECT_BACKOFF_SEC = 5

_fundamentals_lock = threading.Lock()
_fundamentals_cache: dict = {}   # ticker -> {"data": {...}, "fetched_at": float}

_halt_warned_once = False   # only print the "verify halt field" warning one time

SCHWAB: Optional["SchwabClient"] = None            # Market Data app — quotes/price history
SCHWAB_TRADER: Optional["SchwabClient"] = None      # Trader app — account hash + order placement
_vader = None
_lm = None
_last_news_at: Optional[datetime] = None
_news_lock = threading.Lock()

# ---- Subscription-ack tracking ----------------------------------------------
# `watchlist` (above) is just our local Python set, built from Finviz — it says
# nothing about whether Alpaca's server actually accepted/echoed those symbols
# for the news channel. Alpaca's websocket protocol sends back a `"subscription"`
# message that lists exactly which symbols it has you subscribed to; alpaca-py
# normally only surfaces that via Python `logging` at INFO level (which is
# silent by default and easy to miss), not via print(). We capture it directly
# below so it shows up next to the heartbeat as a hard cross-check.
_acked_lock = threading.Lock()
acked_news_symbols: set = set()

# ---- Article de-dup ----------------------------------------------------------
# Keyed on Alpaca's News.id. Needed for two reasons:
#  1. alpaca-py's own websocket dispatch calls the news handler once per
#     *matched subscribed symbol* on an article, not once per article — so an
#     article tagging 3 of your watchlist tickers fires on_news() 3 times,
#     each time reprocessing the full relevant-ticker list. Without dedup
#     that's 3x duplicate handle_news_event() calls per ticker.
#  2. The REST reconciliation safety-net (news_reconciliation_loop) may
#     re-fetch articles the live stream already delivered; dedup keeps it
#     from double-processing those too.
_seen_news_lock = threading.Lock()
_seen_news_ids: "OrderedDict[int, None]" = OrderedDict()
_SEEN_NEWS_CACHE_SIZE = 5000


def _already_seen_news(news_id) -> bool:
    """Returns True (and does nothing further) if news_id was already
    processed; otherwise records it and returns False."""
    if news_id is None:
        return False  # can't dedup without an id; let it through
    with _seen_news_lock:
        if news_id in _seen_news_ids:
            return True
        _seen_news_ids[news_id] = None
        if len(_seen_news_ids) > _SEEN_NEWS_CACHE_SIZE:
            _seen_news_ids.popitem(last=False)
        return False


# ---- Recycled/republished headline dedup (fuzzy, per ticker) ----------------
# _already_seen_news() above only catches the SAME Alpaca article id being
# redelivered. It does NOT catch a wire service (or Benzinga/Alpaca itself)
# republishing substantively the same story under a NEW id later the same
# day — a genuinely common pattern for small-cap catalyst headlines (an
# initial short wire hit followed by a fuller repost of the same news, or
# the same press release picked up by multiple distributors). This is a
# SEPARATE, fuzzy, per-ticker check: it compares each incoming headline
# against that ticker's own recent headline history using difflib's
# SequenceMatcher ratio (0.0-1.0), independent of article id.
NEWS_DEDUP_SIMILARITY_THRESHOLD = 0.85   # ratio >= this counts as "recycled"
NEWS_DEDUP_HISTORY_PER_TICKER   = 20     # how many recent headlines to keep, per ticker

_recent_headlines_lock = threading.Lock()
_recent_headlines: "dict[str, list[str]]" = {}   # ticker -> [normalized headline, ...] (oldest first)


# ---- Confirmed-bullish news history (feeds the rolling-high breakout path) --
# Separate from _recent_headlines above (which is only a short-window dedup
# list of raw headline strings). This records a TIMESTAMPED entry every time
# handle_news_event() gets a genuine "positive" catalyst verdict from
# call_ollama_catalyst_check — i.e. news that already passed the sentiment
# gate, whether or not the immediate tick-volume reaction check also fired.
# rolling_high_breakout_loop() checks this to implement the "old news, still
# bullish, price only now breaking out" pattern discussed — a slow-diffusion/
# drift setup rather than an instant reaction. Kept in memory only (resets on
# each daily restart), which matches the bot's actual daily-restart pattern.
_bullish_news_history_lock = threading.Lock()
_bullish_news_history: "dict[str, list[datetime]]" = {}   # ticker -> [confirmed-bullish timestamp, ...]


def _record_bullish_news(ticker: str, when: datetime):
    with _bullish_news_history_lock:
        _bullish_news_history.setdefault(ticker, []).append(when)


def has_recent_bullish_news(ticker: str, lookback_days: int = OLD_BULLISH_NEWS_LOOKBACK_DAYS) -> Optional[datetime]:
    """Returns the most recent confirmed-bullish-catalyst timestamp for this
    ticker within the last `lookback_days`, or None if there isn't one. This
    is a NECESSARY condition for the rolling-high breakout path — it does not
    care how old within the window the news is, since the whole point is to
    catch delayed/slow-diffusion follow-through, not just same-day reactions
    (same-day reactions are the news pipeline's job, not this path's)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with _bullish_news_history_lock:
        timestamps = _bullish_news_history.get(ticker, [])
        recent = [t for t in timestamps if t >= cutoff]
    return max(recent) if recent else None


def _is_recycled_headline(ticker: str, headline: str,
                           threshold: float = NEWS_DEDUP_SIMILARITY_THRESHOLD) -> bool:
    """Returns True if `headline` is a near-duplicate of a headline already
    seen for this ticker (fuzzy match, not exact/id match — see module
    comment above). Always records the new headline into history regardless
    of the result, so later headlines are compared against the fullest
    available history; capped at NEWS_DEDUP_HISTORY_PER_TICKER per ticker."""
    normalized = re.sub(r"\s+", " ", headline).strip().lower()
    if not normalized:
        return False
    with _recent_headlines_lock:
        history = _recent_headlines.setdefault(ticker, [])
        is_dupe = any(
            difflib.SequenceMatcher(None, normalized, prior).ratio() >= threshold
            for prior in history
        )
        history.append(normalized)
        if len(history) > NEWS_DEDUP_HISTORY_PER_TICKER:
            del history[0]
    return is_dupe

# ---- 10-slot CAPITAL buying pool ---------------------------------------------
_opportunities_lock = threading.Lock()
opportunities_remaining = TOTAL_BUY_OPPORTUNITIES   # decremented in SLOTS as catalysts are traded
_stop_event_ref: Optional[threading.Event] = None
_stream_ref = None              # the NewsDataStream instance, for shutdown

# Set ONCE at startup by detect_capital_pool() (called from main()) — the
# actual detected Schwab buying power and that amount divided evenly across
# TOTAL_BUY_OPPORTUNITIES slots. Left as None/0.0 until then; every function
# below that needs a dollar amount runs AFTER main() has populated these, so
# there's no lazy-init race to worry about.
ACCOUNT_BUYING_POWER_USD: Optional[float] = None
CAPITAL_PER_SLOT_USD: float = 0.0


def detect_capital_pool(schwab_trader: "SchwabClient", account_hash: str) -> float:
    """Queries the account's real buying power ONCE and divides it into
    TOTAL_BUY_OPPORTUNITIES equal capital slots. FAILS CLOSED: if the
    account balance can't be read at all (and CAPITAL_POOL_FALLBACK_USD is
    also unset), this raises rather than silently trading with a guessed or
    zero capital figure — better to not start the run than to size every
    position off a wrong number."""
    global ACCOUNT_BUYING_POWER_USD, CAPITAL_PER_SLOT_USD
    buying_power = schwab_trader.get_account_buying_power(account_hash)
    if buying_power is None:
        if CAPITAL_POOL_FALLBACK_USD is not None:
            print(f"  [WARN] Could not detect account buying power — using "
                  f"CAPITAL_POOL_FALLBACK_USD=${CAPITAL_POOL_FALLBACK_USD:.2f} instead.")
            buying_power = CAPITAL_POOL_FALLBACK_USD
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
    if buying_power <= 0:
        raise RuntimeError(f"Detected buying power is ${buying_power:.2f} — nothing to trade with.")

    ACCOUNT_BUYING_POWER_USD = buying_power
    CAPITAL_PER_SLOT_USD = buying_power / TOTAL_BUY_OPPORTUNITIES
    return buying_power


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


def run_is_complete() -> bool:
    """The run is done once every capital slot has been used AND every
    resulting position has closed."""
    with _positions_lock:
        no_open_positions = len(open_positions) == 0
    return opportunities_left() <= 0 and no_open_positions


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
    # Fallback: if the websocket's run() loop doesn't return promptly after
    # stop()/close(), force the process to exit rather than hang.
    threading.Timer(5.0, lambda: os._exit(0)).start()


# ══════════════════════════════════════════════════════════════════════════
# SCHWAB CLIENT — OAuth refresh-token flow + price history + quotes
# ══════════════════════════════════════════════════════════════════════════

class SchwabClient:
    TOKEN_URL   = "https://api.schwabapi.com/v1/oauth/token"
    BASE_URL    = "https://api.schwabapi.com/marketdata/v1"
    TRADER_URL  = "https://api.schwabapi.com/trader/v1"   # accounts + order placement

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _refresh(self):
        creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        resp = requests.post(self.TOKEN_URL, headers=headers, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 1800) - 60

    def _auth_headers(self) -> dict:
        with self._lock:
            if self._access_token is None or time.time() >= self._expires_at:
                self._refresh()
            return {"Authorization": f"Bearer {self._access_token}"}

    def _price_history(self, symbol: str, period_days: int, extended_hours: bool) -> pd.DataFrame:
        # IMPORTANT: pass explicit startDate/endDate (epoch ms) rather than
        # relying on Schwab's periodType="day" + period=N day-counting alone.
        # That combination was observed returning candles only through the
        # last FULLY COMPLETED trading day — i.e. it silently excluded the
        # still-in-progress/just-closed current session (a ~24h gap between
        # "latest available bar" and news_time). Schwab's price history
        # endpoint ignores `period` when startDate/endDate are supplied, so
        # this forces the window to unambiguously end at "now" and include
        # today's bars.
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=period_days)
        params = {
            "symbol": symbol,
            "periodType": "day",
            "frequencyType": "minute",
            "frequency": 1,
            "needExtendedHoursData": str(extended_hours).lower(),
            "startDate": int(start_dt.timestamp() * 1000),
            "endDate": int(end_dt.timestamp() * 1000),
        }
        resp = requests.get(f"{self.BASE_URL}/pricehistory", headers=self._auth_headers(),
                             params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("timestamp").rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
        return df.dropna()

    def get_price_history_1m(self, symbol: str, extended_hours: bool = True) -> pd.DataFrame:
        """Today's 1-minute OHLCV bars (pre/post market included)."""
        return self._price_history(symbol, period_days=1, extended_hours=extended_hours)

    def get_price_history_multiday_1m(self, symbol: str, days: int, extended_hours: bool = True) -> pd.DataFrame:
        """Multi-day 1-minute bars, used only for the RVOL same-time-of-day baseline."""
        return self._price_history(symbol, period_days=days, extended_hours=extended_hours)

    def get_quote(self, symbol: str) -> Optional[float]:
        """Simple last-price lookup."""
        full = self.get_quote_full(symbol)
        return full.get("last") if full else None

    def get_quote_full(self, symbol: str) -> dict:
        """Returns last/bid/ask/security-status in one call, used for the
        liquidity gate and the halt check."""
        resp = requests.get(f"{self.BASE_URL}/quotes", headers=self._auth_headers(),
                             params={"symbols": symbol}, timeout=15)
        resp.raise_for_status()
        node = resp.json().get(symbol, {})
        quote = node.get("quote", {})
        return {
            "last": quote.get("lastPrice") or quote.get("mark") or quote.get("closePrice"),
            "bid": quote.get("bidPrice"),
            "ask": quote.get("askPrice"),
            # Cumulative shares traded so far today — used by the instant
            # tick-volume gate (see instant_tick_volume_check) to derive
            # volume-since-news without needing any bar to close.
            # NOTE: field name unverified against current Schwab schema — see
            # module docstring.
            "total_volume": quote.get("totalVolume"),
            # NOTE: field name unverified against current Schwab schema — see
            # module docstring. Fails open (None -> treated as not-halted).
            # Previous session's close — used by the RVOL/gap scanner (see
            # market_scan_and_update) to compute gap%. NOTE: field name
            # unverified against current Schwab schema — see module docstring.
            "previous_close": quote.get("closePrice"),
            "security_status": quote.get("securityStatus"),
        }

    def get_quotes_batch(self, symbols: list) -> dict:
        """Batched quote fetch for MANY symbols in ONE request — Schwab's
        /quotes endpoint accepts a comma-separated symbols param. This is
        what the RVOL/gap scanner and halt-transition detector use to scan
        the whole watchlist without doing one HTTP call per ticker (the same
        class of mistake the Finviz per-ticker scrape made before it was
        fixed to use a bulk screener query). Returns {symbol: {...}, ...} in
        the same shape as get_quote_full; symbols missing from Schwab's
        response are simply absent from the returned dict rather than
        raising, so one bad symbol in a chunk can't kill the whole batch.
        VERIFY the max symbols per request against current Schwab docs —
        chunked by QUOTE_BATCH_CHUNK_SIZE by callers as a conservative
        starting point."""
        if not symbols:
            return {}
        resp = requests.get(f"{self.BASE_URL}/quotes", headers=self._auth_headers(),
                             params={"symbols": ",".join(symbols)}, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        out = {}
        for sym in symbols:
            node = raw.get(sym, {})
            quote = node.get("quote", {})
            if not quote:
                continue
            out[sym] = {
                "last": quote.get("lastPrice") or quote.get("mark") or quote.get("closePrice"),
                "bid": quote.get("bidPrice"),
                "ask": quote.get("askPrice"),
                "total_volume": quote.get("totalVolume"),
                "previous_close": quote.get("closePrice"),
                "security_status": quote.get("securityStatus"),
            }
        return out

    def get_account_hashes(self) -> list:
        """One-time lookup: returns [{'accountNumber':, 'hashValue':}, ...].
        Order-placement endpoints require the hashValue, not the raw account
        number — this is Schwab's privacy-preserving account identifier."""
        resp = requests.get(f"{self.TRADER_URL}/accounts/accountNumbers",
                             headers=self._auth_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_account_buying_power(self, account_hash: str) -> Optional[float]:
        """Fetches the account's currently available trading capital in USD.
        Used ONCE at startup to size the whole run's capital pool (see
        CAPITAL_PER_SLOT_USD) — not repolled mid-run, same "detected once,
        static for the run" pattern as the Finviz watchlist.

        NOTE (unverified against current Schwab schema — see the module
        docstring's VERIFY list): the exact field depends on account type.
        Tries, in order:
          1. currentBalances.buyingPower       (margin accounts)
          2. currentBalances.cashAvailableForTrading  (cash accounts)
          3. currentBalances.cashBalance       (fallback)

        ALWAYS logs which of these three fields was actually used — this
        used to fall through silently, so a cash-vs-margin mismatch (or a
        Schwab schema change) could size the ENTIRE run's capital pool off
        a field nobody chose, with no trace of it in the logs. If the field
        used is NOT EXPECTED_BUYING_POWER_FIELD, this now FAILS LOUD
        (raises RuntimeError) rather than proceeding quietly, unless
        ALLOW_BUYING_POWER_FIELD_FALLBACK is explicitly set True.

        Returns None (fail closed — caller must not guess a number) only if
        NONE of the three fields are present, or the request itself fails."""
        try:
            resp = requests.get(f"{self.TRADER_URL}/accounts/{account_hash}",
                                 headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            balances = resp.json().get("securitiesAccount", {}).get("currentBalances", {})
        except Exception as e:
            print(f"  [WARN] Could not fetch account buying power: {e}")
            return None

        for field in ("buyingPower", "cashAvailableForTrading", "cashBalance"):
            raw_value = balances.get(field)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            if field == EXPECTED_BUYING_POWER_FIELD:
                print(f"  [CAPITAL] Buying power detected from currentBalances.{field} = "
                      f"${value:,.2f} (expected field — the run will size its capital pool off this).")
            elif ALLOW_BUYING_POWER_FIELD_FALLBACK:
                print(f"  [WARN] Buying power detected from currentBalances.{field} = ${value:,.2f} "
                      f"— NOT the expected field (currentBalances.{EXPECTED_BUYING_POWER_FIELD} was "
                      f"missing/None on this account). Proceeding anyway because "
                      f"ALLOW_BUYING_POWER_FIELD_FALLBACK=True.")
            else:
                raise RuntimeError(
                    f"Buying-power detection found currentBalances.{field}=${value:,.2f}, but "
                    f"currentBalances.{EXPECTED_BUYING_POWER_FIELD} (the expected field — see "
                    f"EXPECTED_BUYING_POWER_FIELD) was missing or null on this account. Refusing "
                    f"to silently size this run's ENTIRE capital pool off a lower-priority "
                    f"fallback field the account type wasn't expected to need. Either this isn't "
                    f"the account type you thought it was, or Schwab's schema doesn't match what "
                    f"this script assumes — check the account's actual /accounts response before "
                    f"proceeding. If currentBalances.{field} really is what you want this run to "
                    f"size off of, set ALLOW_BUYING_POWER_FIELD_FALLBACK=True and rerun."
                )
            return value
        return None

    def place_equity_order(self, account_hash: str, symbol: str, side: str, quantity: float,
                            limit_price: float) -> dict:
        """Places a live LIMIT equity order, eligible for execution in
        pre-market, regular hours, AND after-hours within the current day.

        Deliberately never a MARKET order: Schwab does not accept MARKET
        orders outside the 9:30am-4:00pm ET regular session at all — extended
        hours (7:00-9:25am / 4:05-8:00pm ET) requires LIMIT orders, full
        stop. session="SEAMLESS" + duration="DAY" is Schwab's "Day +
        extended" order type, the only combination eligible across all three
        sessions same-day; using "NORMAL" here would silently make any
        premarket-triggered entry ineligible to fill until regular-hours
        open, defeating the point of a premarket catalyst strategy.

        limit_price must be supplied by the caller (typically last quoted
        price +/- LIMIT_ORDER_SLIPPAGE_PCT, using ask for buys / bid for
        sells when available) so the order remains realistically fillable in
        thin extended-hours liquidity without being unbounded.

        quantity is always a whole-share count, derived from the claimed
        CAPITAL SLOTS' dollar value divided by entry price (magnitude-scaled
        between MIN_SLOTS_PER_TRADE and MAX_SLOTS_PER_TRADE slots for normal
        entries, or the full remaining capital pool for the FDA fast path —
        see enter_position), so this doesn't depend on the API accepting
        fractional-share orders.

        CONFIRMED for both sides of a trade: this is the ONLY order-placement
        path in the script — exit_position()'s SELL orders call this exact
        same method (side="SELL") rather than a separate exit-order code
        path, so the Day + extended-hours session/duration below is not
        something that only applies to entries; it's enforced by the
        assertion right below for whichever side calls this. This matters
        because the trade log shows fills happening in premarket, so an
        exit that silently reverted to "NORMAL" (regular-hours-only) would
        leave a position unable to close until 9:30am ET regardless of what
        the exit logic decided.
        """
        if not account_hash:
            raise RuntimeError("SCHWAB_ACCOUNT_HASH is not set — run --schwab-account-hash first.")
        body = {
            "orderType": "LIMIT",
            "price": f"{limit_price:.2f}",
            "session": "SEAMLESS",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": side.upper(),
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"},
            }],
        }
        # Day + extended-hours confirmation (task: verify exit orders fill
        # premarket like the trade log shows). session="SEAMLESS" +
        # duration="DAY" is the only Schwab combination eligible across
        # pre/regular/after-hours in one order — asserted here (not just
        # commented) so a future edit that swaps in "NORMAL" or a narrower
        # session by mistake fails immediately and loudly, for BOTH BUY and
        # SELL, instead of silently reintroducing the "can't fill outside
        # 9:30-4:00" bug this was built to avoid.
        assert body["session"] == "SEAMLESS" and body["duration"] == "DAY", (
            f"place_equity_order() must submit Day + extended-hours "
            f"(session=SEAMLESS, duration=DAY) for every order — entries AND "
            f"exits both rely on this to fill in premarket/after-hours. Got "
            f"session={body['session']!r} duration={body['duration']!r} for "
            f"{side.upper()} {symbol}."
        )
        resp = requests.post(f"{self.TRADER_URL}/accounts/{account_hash}/orders",
                              headers={**self._auth_headers(), "Content-Type": "application/json"},
                              json=body, timeout=15)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # DEBUG HELP: Schwab rejects some symbols outright via the API —
            # e.g. certain low-float/OTC/restricted securities that Schwab
            # requires to be bought through a live broker rep rather than the
            # Trader API, halted securities, or bad account/symbol state.
            # resp.raise_for_status()'s own exception text only carries the
            # HTTP status + reason phrase, not Schwab's actual error body, so
            # callers get an unhelpful "400 Client Error" with no explanation.
            # Surface the real body here so enter_position()/exit_position()'s
            # [ABORT]/[ERROR] prints show *why* Schwab wouldn't take it.
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise RuntimeError(
                f"Schwab order rejected (HTTP {resp.status_code}) for {side.upper()} "
                f"{quantity} {symbol} @ limit ${limit_price:.2f}: {detail}"
            ) from e
        location = resp.headers.get("Location")
        # Location is typically ".../accounts/{hash}/orders/{orderId}" — pull
        # the trailing id so callers (see exit_position's fill-confirmation
        # retry loop) can poll/cancel this specific order without doing their
        # own URL parsing.
        order_id = location.rstrip("/").split("/")[-1] if location else None
        return {"status_code": resp.status_code, "order_location": location, "order_id": order_id}

    def get_order_status(self, account_hash: str, order_id: str) -> Optional[str]:
        """Returns the order's current status string (e.g. 'FILLED', 'WORKING',
        'CANCELED', 'REJECTED', 'EXPIRED', 'PENDING_ACTIVATION') or None if the
        lookup fails / the order id is falsy."""
        details = self.get_order_fill_details(account_hash, order_id)
        return details.get("status") if details else None

    def get_order_fill_details(self, account_hash: str, order_id: str) -> Optional[dict]:
        """Returns {status, filled_quantity, avg_fill_price} for an order, or
        None if the lookup fails / the order id is falsy. Used by
        exit_position()'s fill-confirmation loop so the logged/reported P&L
        is the ACTUAL fill price, not just the reference price the limit was
        computed from — those can differ from the reference price, and if
        the order never fills at all there IS no real fill price to report.

        NOTE (unverified against current Schwab schema — see the module
        docstring's VERIFY list): tries orderActivityCollection's execution
        legs for a volume-weighted average fill price first
        (executionLegs[].price x executionLegs[].quantity, summed and
        divided by filledQuantity); falls back to a top-level `price` field
        if that structure isn't present. avg_fill_price is None if neither
        yields a usable number — callers must treat that as "fill price
        unconfirmed," not assume it equals the limit/reference price."""
        if not order_id:
            return None
        try:
            resp = requests.get(f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}",
                                 headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            order = resp.json()
        except Exception as e:
            print(f"  [WARN] Could not fetch order status for order {order_id}: {e}")
            return None

        status = order.get("status")
        filled_quantity = order.get("filledQuantity") or 0

        avg_fill_price = None
        total_shares = 0.0
        total_notional = 0.0
        for activity in order.get("orderActivityCollection", []) or []:
            for leg in activity.get("executionLegs", []) or []:
                qty = leg.get("quantity")
                px = leg.get("price")
                if qty and px:
                    total_shares += float(qty)
                    total_notional += float(qty) * float(px)
        if total_shares > 0:
            avg_fill_price = total_notional / total_shares
        elif order.get("price") and status == "FILLED":
            # Fallback only if we KNOW it filled but couldn't parse execution
            # legs — still logged as a fallback, never silently assumed.
            avg_fill_price = float(order["price"])

        return {"status": status, "filled_quantity": float(filled_quantity), "avg_fill_price": avg_fill_price}

    def cancel_order(self, account_hash: str, order_id: str) -> bool:
        """Cancels a still-working order. Returns True on success. Treated as
        best-effort by callers — if the order already filled/canceled/was
        rejected between the last status poll and this call, Schwab may
        return an error here, which is fine (nothing to cancel)."""
        if not order_id:
            return False
        try:
            resp = requests.delete(f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}",
                                    headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"  [WARN] Could not cancel order {order_id} (may have already filled/closed): {e}")
            return False


def schwab_oauth_bootstrap(app: str):
    """One-time interactive helper to mint the first refresh token for
    whichever app ('marketdata' or 'trader') is selected."""
    if app == "trader":
        client_id, client_secret, redirect_uri = (
            SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET, SCHWAB_TRADER_REDIRECT_URI)
        env_prefix = "SCHWAB_TRADER"
    else:
        client_id, client_secret, redirect_uri = (
            SCHWAB_MARKETDATA_CLIENT_ID, SCHWAB_MARKETDATA_CLIENT_SECRET, SCHWAB_MARKETDATA_REDIRECT_URI)
        env_prefix = "SCHWAB_MARKETDATA"

    if not client_id or not client_secret:
        print(f"[ERROR] Set {env_prefix}_CLIENT_ID / {env_prefix}_CLIENT_SECRET first.")
        return
    auth_url = (f"https://api.schwabapi.com/v1/oauth/authorize?"
                f"client_id={client_id}&redirect_uri={redirect_uri}")
    print(f"[{app.upper()} APP] 1) Open this URL, log in with your Schwab account, and approve access:\n   {auth_url}")
    print("2) You'll land on a blank/error page after approving — that's expected.")
    print("   Copy the FULL resulting URL from your browser's address bar.")
    returned_url = input("Paste the full redirected URL here: ").strip()
    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(returned_url).query)
    code = query.get("code", [None])[0]
    if not code:
        print("[ERROR] Couldn't find a 'code=' parameter in that URL.")
        return
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}
    resp = requests.post(SchwabClient.TOKEN_URL, headers=headers, data=payload, timeout=15)
    resp.raise_for_status()
    tokens = resp.json()
    print(f"\nSuccess. Export this (refresh token is valid ~7 days):\n")
    print(f"  export {env_prefix}_REFRESH_TOKEN='{tokens['refresh_token']}'")


def schwab_account_hash_bootstrap():
    """One-time helper: prints the account hash(es) needed for order placement.
    Always uses the TRADER app's credentials since accountNumbers is a
    trader/v1 endpoint."""
    missing = [n for n, v in [
        ("SCHWAB_TRADER_CLIENT_ID", SCHWAB_TRADER_CLIENT_ID),
        ("SCHWAB_TRADER_CLIENT_SECRET", SCHWAB_TRADER_CLIENT_SECRET),
        ("SCHWAB_TRADER_REFRESH_TOKEN", SCHWAB_TRADER_REFRESH_TOKEN),
    ] if not v]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}. Run --schwab-login --app trader first if needed.")
        return
    client = SchwabClient(SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET, SCHWAB_TRADER_REFRESH_TOKEN)
    try:
        accounts = client.get_account_hashes()
    except Exception as e:
        print(f"[ERROR] Could not fetch account hashes: {e}")
        return
    if not accounts:
        print("[ERROR] No accounts returned — check the account is enabled for API trading.")
        return
    print("\nFound account(s):")
    for a in accounts:
        print(f"  accountNumber=...{str(a.get('accountNumber'))[-4:]}  hashValue={a.get('hashValue')}")
    print("\nExport the hashValue for the account you want this script to trade in:")
    print(f"  export SCHWAB_ACCOUNT_HASH='{accounts[0].get('hashValue')}'")


# ══════════════════════════════════════════════════════════════════════════
# HALT + LIQUIDITY GATES
# ══════════════════════════════════════════════════════════════════════════

def _is_halted_status(status) -> bool:
    """Shared classification logic — used by BOTH is_halted() (single-symbol,
    per-check) and market_scan_and_update() (batched, whole-watchlist) so the
    two never silently disagree on what counts as halted."""
    return bool(status) and str(status).strip().lower() == "halted"


def is_halted(symbol: str) -> bool:
    global _halt_warned_once
    try:
        q = SCHWAB.get_quote_full(symbol)
    except Exception as e:
        print(f"  [WARN] Halt check failed for {symbol}: {e}")
        return False
    status = q.get("security_status")
    if status is None and not _halt_warned_once:
        _halt_warned_once = True
        print("  [WARN] Schwab quote has no 'securityStatus' field — halt checks are "
              "failing open (treated as not-halted). Verify the correct field name "
              "against current Schwab docs.")
    return _is_halted_status(status)


def passes_liquidity_gate(symbol: str) -> bool:
    try:
        q = SCHWAB.get_quote_full(symbol)
    except Exception as e:
        print(f"  [WARN] Liquidity check failed for {symbol}: {e}")
        return False
    bid, ask = q.get("bid"), q.get("ask")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return False
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid
    if spread_pct > MAX_SPREAD_PCT:
        print(f"  [GATE] {symbol}: spread {spread_pct*100:.2f}% exceeds {MAX_SPREAD_PCT*100:.1f}% cap — blocked.")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# FINVIZ WATCHLIST SCREEN — float / short-float / avg-volume only
# ══════════════════════════════════════════════════════════════════════════

def _finviz_url() -> str:
    # VERIFY these filter codes against Finviz's own screener UI before relying
    # on them — build the screen there once and confirm the URL matches.
    #
    # Narrowed per review: float_u20 (Under 20M) is loose enough that squeeze
    # dynamics are already diluted for names at the top of that band, and
    # avgvol_o50 (Over 50K) is a low enough liquidity bar to let in names too
    # thin to have real two-sided interest — a genuine catalyst can hit and
    # the stock just sits because there's nobody there to trade it. Tighter
    # float (< 10M) + a higher proven-liquidity floor (> 100K avg volume)
    # biases the watchlist toward names that both have real squeeze potential
    # and demonstrated trading interest, rather than illiquid names that get
    # "goosed" without real participation.
    float_code   = "sh_float_u10"     # Float: Under 10M
    market_cap = "cap_microunder"     # MICRO CAP
    avgvol_code  = "sh_avgvol_o100"   # Average Volume: Over 100K
    return (f"https://finviz.com/screener.ashx?v=111&"
            f"f={float_code},{market_cap},{avgvol_code}&ft=4&o=ticker")


def _fetch_finviz_page(session: requests.Session, page_url: str):
    """Fetch one screener page through a persistent session with real
    browser-like headers. Bare single-shot requests (fresh connection, no
    cookies, User-Agent only) are what finviz's anti-scraping layer is
    tuned to flag after the first request or two — it doesn't necessarily
    error out, it can just quietly hand back a page with no matching table
    rows, which looks like "success" to naive parsing."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/screener.ashx",
    }
    resp = session.get(page_url, headers=headers, timeout=15)
    resp.raise_for_status()
    if resp.text.strip() == "Too many requests.":
        raise RuntimeError("Too many requests.")
    return resp


_TICKER_LINK_RE = re.compile(r'[?&]t=([A-Za-z][A-Za-z\.\-]{0,5})(?=[&"\'])')


def _extract_tickers_from_html(html_text: str) -> list:
    """Pull ticker symbols out of '...t=TICKER...' query-string fragments
    anywhere in the raw page HTML, in document order, de-duplicated.

    NOTE: this is now matching on the 't=' query param generically rather
    than assuming the 'quote.ashx?t=' path prefix, because that assumption
    itself turned out to be stale (see debug dump below if this still comes
    back empty — Finviz's link structure has apparently moved since the
    finviz package's own tr[valign="top"]/header-zip scraper was written).
    """
    seen_local = set()
    out = []
    for t in _TICKER_LINK_RE.findall(html_text):
        t = t.upper()
        if t not in seen_local:
            seen_local.add(t)
            out.append(t)
    return out


def _dump_debug_html(page_url: str, response) -> str:
    """Saves the raw response so we can see exactly what came back instead
    of guessing at the link format again. Prints a few cheap signal checks
    inline so you don't even have to open the file to get a first read."""
    debug_path = CACHE_DIR / "finviz_debug_page.html"
    try:
        debug_path.write_text(response.text, encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [DEBUG] Couldn't write debug HTML: {e}")
    text = response.text
    print(f"  [DEBUG] {page_url}")
    print(f"  [DEBUG] status={response.status_code}  content-length={len(text)}")
    print(f"  [DEBUG] contains 'quote.ashx'? {'quote.ashx' in text}   "
          f"contains 't='? {'t=' in text}   contains 'screener'? {'screener' in text.lower()}")
    print(f"  [DEBUG] full HTML saved to {debug_path.resolve()} — open it and search for one of "
          f"your watchlist tickers' text to see what markup actually wraps it, then send me that snippet.")
    return debug_path.as_posix()


def _scrape_finviz_tickers(url: str, label: str) -> list:
    """The actual scrape/paginate/parse loop, factored out of get_all_tickers()
    so it can be reused against a DIFFERENT Finviz screener URL (see
    get_short_squeeze_candidates() below) without touching the Path A
    watchlist build at all. `label` is just for the log lines."""
    try:
        from finviz.screener import Screener
        from finviz.helper_functions import scraper_functions as scrape
    except ImportError:
        print("[ERROR] finviz not installed: pip install finviz")
        sys.exit(1)

    # Use Screener.init_from_url only to discover total_rows/page_content/url —
    # NOT its .data (see _extract_tickers_from_html docstring for why).
    probe = Screener.init_from_url(url, rows=20)
    total_rows = getattr(probe, "_total_rows", 0)
    if total_rows <= 0:
        return _extract_tickers_from_html(str(html.tostring(probe._page_content)))

    page_urls = scrape.get_page_urls(probe._page_content, total_rows, probe._url)
    print(f"  [FINVIZ:{label}] {total_rows} total matches across {len(page_urls)} pages — fetching all in paced batches.")

    tickers: list = []
    seen: set = set()
    session = requests.Session()

    for i, page_url in enumerate(page_urls, start=1):
        if i > 1:
            time.sleep(FINVIZ_PAGE_DELAY_SEC)

        response = None

        def _try_fetch():
            nonlocal response
            response = _fetch_finviz_page(session, page_url)
            page_tickers = _extract_tickers_from_html(response.text)
            if not page_tickers:
                raise RuntimeError("page returned 0 tickers (likely throttled/blocked, or link format changed)")
            return page_tickers

        try:
            page_tickers = _try_fetch()
        except Exception as e:
            print(f"  [WARN] Finviz:{label} page {i} failed ({e}) — "
                  f"cooling down {FINVIZ_COOLDOWN_SEC}s and retrying once.")
            time.sleep(FINVIZ_COOLDOWN_SEC)
            try:
                page_tickers = _try_fetch()
            except Exception as e2:
                print(f"  [WARN] Page {i} failed again ({e2}) — "
                      f"stopping pagination early with {len(tickers)}/{total_rows} rows collected.")
                if response is not None:
                    _dump_debug_html(page_url, response)
                break

        new_on_page = 0
        for t in page_tickers:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
                new_on_page += 1
        print(f"  [FINVIZ:{label}] page {i}/{len(page_urls)}: {len(page_tickers)} tickers on page, "
              f"{new_on_page} new (running total {len(tickers)}/{total_rows})")

        if i % FINVIZ_BATCH_SIZE == 0 and i < len(page_urls):
            time.sleep(FINVIZ_COOLDOWN_SEC)

    if len(tickers) < total_rows:
        print(f"  [FINVIZ:{label}] WARNING: collected {len(tickers)}/{total_rows} — "
              f"some pages came back short or failed. See [WARN] lines above.")

    return tickers


def get_all_tickers() -> list:
    return _scrape_finviz_tickers(_finviz_url(), label="watchlist")


# ---- Per-ticker fundamentals (float / short-float), fetched lazily ---------

def _parse_share_count(s) -> Optional[float]:
    """Parses Finviz-style strings like '5.20M', '800.00K', '12.5%' -> float."""
    if s is None:
        return None
    s = str(s).strip().replace("%", "")
    if not s or s == "-":
        return None
    mult = 1.0
    if s[-1] in ("K", "k"):
        mult, s = 1_000.0, s[:-1]
    elif s[-1] in ("M", "m"):
        mult, s = 1_000_000.0, s[:-1]
    elif s[-1] in ("B", "b"):
        mult, s = 1_000_000_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def get_ticker_fundamentals(ticker: str) -> dict:
    """Lazily scrapes float + short-float for a single ticker via the `finviz`
    package's per-stock fundamentals lookup, cached for FUNDAMENTALS_CACHE_TTL_SEC.
    Only called at confirmation time for tickers actually undergoing a news
    reaction check — not bulk-fetched for the whole watchlist."""
    now = time.time()
    with _fundamentals_lock:
        cached = _fundamentals_cache.get(ticker)
        if cached and (now - cached["fetched_at"]) < FUNDAMENTALS_CACHE_TTL_SEC:
            return cached["data"]

    try:
        import finviz
        raw = finviz.get_stock(ticker)   # dict of fundamental fields keyed by label
    except Exception as e:
        print(f"  [WARN] Could not fetch fundamentals for {ticker}: {e}")
        raw = {}

    data = {
        "float_shares": _parse_share_count(raw.get("Shs Float")),
        "short_float_pct": _parse_share_count(raw.get("Short Float")),
        # VERIFY BEFORE RELYING ON THIS — "Short Ratio" is Finviz's label for
        # days-to-cover (short interest / avg daily volume) on the per-stock
        # quote page as of when this was written; confirm the label still
        # matches by checking finviz.get_stock() output directly, same as the
        # other fields here.
        "days_to_cover": _parse_share_count(raw.get("Short Ratio")),
        # Used only by the instant tick-volume gate to build a scaled baseline
        # (see TICK_VOLUME_MULTIPLE) — no historical bars needed for this.
        "avg_volume": _parse_share_count(raw.get("Avg Volume")),
    }
    with _fundamentals_lock:
        _fundamentals_cache[ticker] = {"data": data, "fetched_at": now}
    return data


# ══════════════════════════════════════════════════════════════════════════
# VADER + LOUGHRAN-McDONALD SCORING — used as a GATE (block on negative,
# require positive), not a magnitude score. Replaces FinBERT.
#
# CALIBRATION NOTE (read before tuning NEGATIVE_PROB_MAX / POSITIVE_PROB_MIN):
# these are lexicon/dictionary methods, not a calibrated softmax over 3
# classes like FinBERT was — their output distributions behave differently,
# and the thresholds below have been RECALIBRATED accordingly (see the empirical
# test results this was built against):
#   - p_positive / p_negative are a complementary pair derived from VADER's
#     compound score (-1..1), rescaled to 0..1: p_positive=(compound+1)/2,
#     p_negative=1-p_positive. Raw VADER pos/neg proportions were NOT used —
#     they almost never reach 0.80 on short headlines (most tokens are
#     neutral function words/tickers).
#   - VADER has NO concept of financial magnitude — "Beats Estimates, Raises
#     Guidance" and "Up 59% YoY" both score as flatly NEUTRAL (compound=0.0)
#     even though these are strong beats. This is a real, confirmed blind
#     spot this dictionary-only stack cannot fix — closing it is exactly
#     what the planned LLM-escalation step is for.
#   - Loughran-McDonald's positive word list is small (~354 words) and built
#     for 10-K/10-Q legal boilerplate, not press-release headlines. Testing
#     found its raw negative-word COUNT cannot discriminate real red flags
#     from generic sector vocabulary at headline length — "Defense Contractor
#     Secures Multi-Year Missile Systems Contract" and "Company Defaults On
#     Debt Covenant" both scored 2 LM-negative hits, while a genuine "Filed
#     For Chapter 11 Bankruptcy" headline only scored 1. There is no
#     hit-count threshold that separates these. Because of this, LM's raw
#     word-counting is NOT used to gate anything — see RED_FLAG_KEYWORDS
#     (defined near the FDA fast path above) for the curated-phrase override
#     that replaces it. LM's Positive/Negative/Polarity are still computed
#     and returned for logging/visibility only.
# NEGATIVE_PROB_MAX=0.30 / POSITIVE_PROB_MIN=0.65 below were chosen from
# real headline testing (they correctly pass a "$50M contract win" headline
# at p_neg=0.16/p_pos=0.84 while still blocking weak-earnings, bankruptcy,
# and going-concern/delisting headlines) — but they are a starting point,
# not a validated calibration. Backtest against your own trades_log.csv
# before trusting this live, and expect this stack to still miss flat,
# factual positive news (numeric beats, guidance raises) until the LLM
# escalation step is added.
# ══════════════════════════════════════════════════════════════════════════

def get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("[INIT] Loading VADER...")
        _vader = SentimentIntensityAnalyzer()
    return _vader


def get_lm():
    global _lm
    if _lm is None:
        import pysentiment2 as ps
        print("[INIT] Loading Loughran-McDonald dictionary...")
        _lm = ps.LM()
    return _lm


def score_text_full(text: str) -> dict:
    """Returns {'label':, 'confidence':, 'p_negative':, 'p_positive':, 'p_neutral':,
    'vader_compound':, 'lm_polarity':, 'lm_pos_words':, 'lm_neg_words':,
    'red_flag_hit':}. p_negative/p_positive are consumed directly by the
    sentiment gate, same interface FinBERT used to fill, now built from
    VADER + a curated red-flag phrase list (see RED_FLAG_KEYWORDS above for
    why raw LM word-counting was dropped as the override mechanism — LM's
    Positive/Negative/Polarity are still computed and returned here for
    logging/visibility, they just no longer gate anything on their own)."""
    empty = {"label": "neutral", "confidence": 0.0, "p_negative": 0.5, "p_positive": 0.5, "p_neutral": 1.0,
             "vader_compound": 0.0, "lm_polarity": 0.0, "lm_pos_words": 0, "lm_neg_words": 0, "red_flag_hit": None}
    if not text or not text.strip():
        return empty
    try:
        compound = get_vader().polarity_scores(text)["compound"]   # -1..1

        lm = get_lm()
        lm_score = lm.get_score(lm.tokenize(text))
        lm_pos, lm_neg = int(lm_score["Positive"]), int(lm_score["Negative"])
        lm_polarity = float(lm_score["Polarity"])   # -1..1, 0.0 when no LM-dictionary words are found

        # VADER compound rescaled to a complementary 0..1 positive/negative pair.
        vader_positive_like = (compound + 1.0) / 2.0
        vader_negative_like = 1.0 - vader_positive_like

        # Hard override on a curated, unambiguous red-flag phrase — NOT raw
        # LM word-counting (see RED_FLAG_KEYWORDS comment for why: hit-count
        # thresholds cannot separate real red flags from generic sector
        # vocabulary at headline length, verified empirically).
        t_lower = text.lower()
        red_flag_hit = next((k for k in RED_FLAG_KEYWORDS if k in t_lower), None)

        p_negative = 1.0 if red_flag_hit else vader_negative_like
        p_positive = vader_positive_like
        # small optional confirmation bonus only — informational LM agreement,
        # never required, never applied if the red-flag override already fired.
        if not red_flag_hit and lm_pos >= 2 and lm_neg == 0:
            p_positive = min(1.0, p_positive + 0.05)

        p_negative = round(min(max(p_negative, 0.0), 1.0), 4)
        p_positive = round(min(max(p_positive, 0.0), 1.0), 4)
        p_neutral = round(max(0.0, 1.0 - max(p_negative, p_positive)), 4)
        label = ("negative" if p_negative >= 0.5 and p_negative >= p_positive
                  else "positive" if p_positive >= 0.5 else "neutral")

        return {
            "label": label, "confidence": round(max(p_negative, p_positive, p_neutral), 4),
            "p_negative": p_negative, "p_positive": p_positive, "p_neutral": p_neutral,
            "vader_compound": round(compound, 4), "lm_polarity": round(lm_polarity, 4),
            "lm_pos_words": lm_pos, "lm_neg_words": lm_neg, "red_flag_hit": red_flag_hit,
        }
    except Exception as e:
        print(f"  [WARN] VADER/Loughran-McDonald scoring failed: {e}")
        return empty


def passes_sentiment_gate(headline: str, summary: str) -> tuple:
    """COMPOSITE gate: entry requires BOTH negative and positive halves to
    pass, checked on the worse of headline/summary in each direction —
    P(negative) < NEGATIVE_PROB_MAX (worst/max of the two) AND
    P(positive) > POSITIVE_PROB_MIN (worst/min of the two). Neutral
    headlines that used to pass under the old negative-only gate now fail
    the positive-floor half. Returns (passed: bool, detail: dict) for logging."""
    hdl = score_text_full(headline)
    sm = score_text_full(summary if summary and summary.strip() else headline)
    worst_p_negative = max(hdl["p_negative"], sm["p_negative"])
    worst_p_positive = min(hdl["p_positive"], sm["p_positive"])
    negative_ok = worst_p_negative < NEGATIVE_PROB_MAX
    positive_ok = worst_p_positive > POSITIVE_PROB_MIN
    passed = negative_ok and positive_ok
    return passed, {
        "headline": hdl, "summary": sm,
        "p_negative_worst": worst_p_negative, "p_positive_worst": worst_p_positive,
        "negative_ok": negative_ok, "positive_ok": positive_ok,
        "red_flag_hit": hdl.get("red_flag_hit") or sm.get("red_flag_hit"),
    }


def contains_fda_fastpath(text: str) -> bool:
    """CANDIDATE pre-filter only (see the FDA_APPROVAL_KEYWORDS config note
    above) — a True here means "worth asking qwen to confirm", not "bypass
    every gate". call_ollama_fda_check() below makes the actual decision."""
    t = text.lower()
    if any(bad in t for bad in FDA_EXCLUDE_KEYWORDS):
        return False
    return any(k in t for k in FDA_APPROVAL_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — LOCAL LLM ESCALATION (Ollama, runs entirely on-machine)
# Only called when tier 1 (VADER + Loughran-McDonald) blocks a headline that
# did NOT hit a confident red-flag phrase — see the config comment above
# OLLAMA_ENABLED for why. Asks for DIRECTION/legitimacy only, never magnitude.
# ══════════════════════════════════════════════════════════════════════════

_OLLAMA_CATALYST_PROMPT = """You are screening a stock-news headline for a low-float catalyst trading strategy. Decide ONLY whether this is a genuine, real, price-positive catalyst for the stock — NOT how big the move will be, NOT general tone. Routine/procedural news (routine filings, generic conference appearances, analyst coverage initiations with no rating, etc.) is NOT a catalyst even if worded neutrally-to-positively. Numeric beats, guidance raises, contract wins, approvals, and similar concrete positive business developments ARE catalysts even if the wording is flat/factual.

Also decide whether this headline is SECONDHAND/RECAP reporting rather than a primary, quantitative disclosure — for example, an article that reports on a price move that ALREADY happened ("Why Shares Of X Are Up Today", "X Surges: Here's What's Driving It"), or a roundup/listicle piece covering many unrelated tickers at once ("20 Stocks That Have Gone Up Today And Why"). This is true even if the headline doesn't literally contain those words — judge the underlying pattern (reporting ABOUT a move / aggregating many stocks) not just the phrasing. A single-company article about a concrete new business development (earnings, a contract, an approval, guidance) is NOT secondhand/recap just because a reporter wrote it.

Headline: {headline}
Summary: {summary}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"catalyst": "positive" or "negative" or "neutral", "confidence": "high" or "medium" or "low", "reasoning": "one short sentence", "is_stale_or_secondhand": true or false}}"""


def call_ollama_catalyst_check(headline: str, summary: str) -> Optional[dict]:
    """Returns {'catalyst':, 'confidence':, 'reasoning':, 'is_stale_or_secondhand':}
    or None if Ollama is unreachable/times out/returns something unparseable —
    callers must treat None as fail-closed (no override, no entry), never as
    an implicit pass.

    'is_stale_or_secondhand' is the qwen-side counterpart to the
    STALE_RECAP_HEADLINE_PATTERNS dictionary check in handle_news_event() —
    it exists to catch recap/listicle/secondhand-reporting headlines that
    don't literally match one of those substrings (paraphrased wording,
    patterns not yet observed/added to the list, etc.). It is parsed
    permissively: if the model omits it or returns something unparseable,
    this defaults to False (i.e. it does not itself cause a drop) rather
    than invalidating the whole response, since catalyst/confidence are the
    fields the rest of the pipeline depends on for fail-closed behavior."""
    prompt = _OLLAMA_CATALYST_PROMPT.format(headline=headline, summary=summary or headline)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",          # ask Ollama to constrain output to valid JSON
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
        parsed = json.loads(raw_text)
        catalyst = str(parsed.get("catalyst", "")).strip().lower()
        confidence = str(parsed.get("confidence", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
        is_stale_or_secondhand = bool(parsed.get("is_stale_or_secondhand", False))
        if catalyst not in ("positive", "negative", "neutral") or confidence not in ("high", "medium", "low"):
            print(f"  [WARN] Ollama returned an unexpected shape, treating as no-override: {raw_text[:200]}")
            return None
        return {"catalyst": catalyst, "confidence": confidence, "reasoning": reasoning,
                 "is_stale_or_secondhand": is_stale_or_secondhand}
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Ollama unreachable at {OLLAMA_URL} (is `ollama serve` running?) — no LLM escalation this event.")
        return None
    except requests.exceptions.Timeout:
        print(f"  [WARN] Ollama call timed out after {OLLAMA_TIMEOUT_SEC}s — no LLM escalation this event.")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [WARN] Could not parse Ollama's response as the expected JSON shape: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Ollama escalation call failed: {e}")
        return None


# ---- FDA fast-path qwen confirmation (bugfix/behavior change) ---------------
# Only called on candidates that already hit contains_fda_fastpath() (see that
# function's docstring) — this keeps the extra Ollama round trip off the vast
# majority of headlines that have nothing to do with FDA news at all, while
# still covering the edge cases a pure keyword match misses (a genuine
# approval phrased outside FDA_APPROVAL_KEYWORDS, or a designation/PDUFA-date
# story that happens to trip one of those phrases anyway).
_OLLAMA_FDA_PROMPT = """You are confirming whether a stock-news headline is reporting a genuine, ALREADY-GRANTED, explicit FDA marketing approval for a drug/device — the single most reliable, first-look catalyst this strategy trades, so this must be conservative.

Answer YES only for an approval that has actually been decided and granted right now (e.g. FDA approval, NDA/BLA approved, FDA clearance, accelerated approval granted).
Answer NO for anything short of that, even if it sounds similar or uses the word "approval" — for example: breakthrough therapy / fast track / orphan drug DESIGNATIONS, priority review grants, a PDUFA date being set, the company merely "seeking", "expecting", or "up for" approval, or an FDA advisory committee recommendation (not yet an actual agency approval decision).

Headline: {headline}
Summary: {summary}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"is_explicit_approval": true or false, "confidence": "high" or "medium" or "low", "reasoning": "one short sentence"}}"""


def call_ollama_fda_check(headline: str, summary: str) -> Optional[dict]:
    """Returns {'is_explicit_approval':, 'confidence':, 'reasoning':} or None
    if Ollama is unreachable/times out/returns something unparseable.
    Callers must treat None as fail-closed for the FAST PATH specifically —
    i.e. do NOT bypass the gates — but the headline should still fall through
    to the normal sentiment+tick-volume gated flow rather than being dropped
    outright, since a real approval headline can still clear those gates on
    its own merits even without the instant bypass."""
    prompt = _OLLAMA_FDA_PROMPT.format(headline=headline, summary=summary or headline)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
        parsed = json.loads(raw_text)
        is_explicit_approval = bool(parsed.get("is_explicit_approval", False))
        confidence = str(parsed.get("confidence", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
        if confidence not in ("high", "medium", "low"):
            print(f"  [WARN] Ollama FDA-check returned an unexpected shape, treating as no-override: {raw_text[:200]}")
            return None
        return {"is_explicit_approval": is_explicit_approval, "confidence": confidence, "reasoning": reasoning}
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Ollama unreachable at {OLLAMA_URL} (is `ollama serve` running?) — no FDA fast-path confirmation this event.")
        return None
    except requests.exceptions.Timeout:
        print(f"  [WARN] Ollama FDA-check call timed out after {OLLAMA_TIMEOUT_SEC}s — no FDA fast-path confirmation this event.")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [WARN] Could not parse Ollama's FDA-check response as the expected JSON shape: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Ollama FDA-check call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# SWING-POINT UTILITIES — shared by entry confirmation and exit management
# ══════════════════════════════════════════════════════════════════════════

def find_swing_highs(values: np.ndarray, window: int) -> list:
    idx = []
    n = len(values)
    for i in range(window, n - window):
        left, right = values[i - window:i], values[i + 1:i + window + 1]
        if len(left) and len(right) and values[i] > left.max() and values[i] > right.max():
            idx.append(i)
    return idx


def find_swing_lows(values: np.ndarray, window: int) -> list:
    idx = []
    n = len(values)
    for i in range(window, n - window):
        left, right = values[i - window:i], values[i + 1:i + window + 1]
        if len(left) and len(right) and values[i] < left.min() and values[i] < right.min():
            idx.append(i)
    return idx


def compute_obv_series(df: pd.DataFrame) -> np.ndarray:
    price_change = df["Close"].diff()
    return (np.sign(price_change) * df["Volume"]).fillna(0).cumsum().values


def bearish_swing_divergence(df: pd.DataFrame, window: int) -> bool:
    """Classic divergence: price makes a higher swing high while OBV's value
    at that same bar is lower than OBV's value at the prior swing high."""
    close = df["Close"].values
    if len(close) < (2 * window + 3):
        return False
    obv = compute_obv_series(df)
    highs = find_swing_highs(close, window)
    if len(highs) < 2:
        return False
    i_new, i_old = highs[-1], highs[-2]
    return bool(close[i_new] > close[i_old] and obv[i_new] < obv[i_old])


def most_recent_swing_low_price(df: pd.DataFrame, window: int) -> Optional[float]:
    close = df["Close"].values
    if len(close) < (2 * window + 1):
        return None
    lows = find_swing_lows(close, window)
    if not lows:
        return None
    return float(close[lows[-1]])


# ---- SET entry-volatility stop (measured once at entry, never trailed) -----

def compute_atr_series(df: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder's ATR (True Range smoothed with Wilder's RMA)."""
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(close)
    atr = np.full(n, np.nan)
    if n == 0:
        return atr
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_entry_volatility_stop(df: pd.DataFrame, entry_price: float,
                                   atr_period: int = ENTRY_STOP_ATR_PERIOD,
                                   atr_multiple: float = ENTRY_STOP_ATR_MULTIPLE,
                                   max_loss_pct: float = MAX_LOSS_PCT,
                                   min_stop_pct: float = MIN_STOP_PCT) -> dict:
    """Measures volatility ONCE, at entry, to size a SET (non-trailing)
    stop-loss — replaces the volume-confirmed Chandelier ATR TRAIL (see the
    config comment above ENTRY_STOP_ATR_PERIOD for why the trail was
    removed). Called exactly once, from enter_position, right after the
    entry price is known; the result is stored on the Position and never
    recalculated afterward.

    stop_distance_pct = min(atr_multiple x ATR(atr_period) / entry_price,
    max_loss_pct) — sized off volatility, but CAPPED at max_loss_pct (10%)
    if volatility implies something wider. Floored at min_stop_pct so a
    near-zero ATR reading can't produce a degenerate near-0% stop.

    Returns {stop_price, stop_pct, atr, capped}. If there isn't yet enough
    bar history to seed an ATR at entry, falls back to the flat
    max_loss_pct outright (atr=None, capped=True)."""
    atr_series = compute_atr_series(df, atr_period)
    atr = float(atr_series[-1]) if len(atr_series) and not np.isnan(atr_series[-1]) else None

    if atr is not None and entry_price > 0:
        vol_implied_pct = (atr_multiple * atr) / entry_price
    else:
        vol_implied_pct = max_loss_pct   # not enough bars yet at entry — fall back to the flat cap

    stop_pct = max(min_stop_pct, min(vol_implied_pct, max_loss_pct))
    capped = vol_implied_pct >= max_loss_pct
    stop_price = entry_price * (1 - stop_pct)

    return {"stop_price": stop_price, "stop_pct": stop_pct, "atr": atr, "capped": capped}


# ---- Secondary exit: significant CVD-bearish + bearish swing divergence ----
# Both gated behind _secondary_exit_armed (6am Pacific OR a significant move
# from entry) — see the config comment above SECONDARY_EXIT_PST_HOUR.

def compute_cvd_series(df: pd.DataFrame) -> np.ndarray:
    """Cumulative Volume Delta: running sum of each bar's volume signed by
    that bar's own price direction (up bar -> +volume, down bar ->
    -volume). Same construction as compute_obv_series; kept as its own
    named function because significant_cvd_bearish() below is a
    MAGNITUDE/threshold read on this series over a short recent window, not
    a swing-structure comparison like bearish_swing_divergence."""
    price_change = df["Close"].diff()
    return (np.sign(price_change) * df["Volume"]).fillna(0).cumsum().values


def significant_cvd_bearish(df: pd.DataFrame, lookback_bars: int = CVD_LOOKBACK_BARS,
                             threshold: float = CVD_BEARISH_THRESHOLD) -> dict:
    """'Significant CVD bearish' — cumulative volume delta over the last
    `lookback_bars` bars actively dominated by down-volume, independent of
    the swing-structure divergence check. Returns {net_cvd_ratio,
    significant} where net_cvd_ratio = (up_volume - down_volume) /
    total_volume over that window (in [-1, 1]; positive = net buying,
    negative = net selling), and `significant` is True when
    net_cvd_ratio <= threshold (default -0.50, i.e. down-volume outweighs
    up-volume by more than 3-to-1 over the window)."""
    close = df["Close"].values
    volume = df["Volume"].values
    tail_close = close[-(lookback_bars + 1):]
    tail_volume = volume[-(lookback_bars + 1):]
    if len(tail_close) < 2:
        return {"net_cvd_ratio": 0.0, "significant": False}

    price_change = np.diff(tail_close)
    bar_volume = tail_volume[1:]   # volume attributed to the bar that produced each price_change
    down_volume = float(bar_volume[price_change < 0].sum())
    up_volume = float(bar_volume[price_change >= 0].sum())
    total_volume = down_volume + up_volume
    net_cvd_ratio = ((up_volume - down_volume) / total_volume) if total_volume > 0 else 0.0

    return {"net_cvd_ratio": net_cvd_ratio, "significant": net_cvd_ratio <= threshold}


def _secondary_exit_armed(pos: "Position", current_price: float) -> bool:
    """The secondary exit (significant CVD-bearish / bearish swing
    divergence) does NOT check from the moment of entry — it only arms
    once EITHER of these holds:
      - local Pacific clock time is >= SECONDARY_EXIT_PST_HOUR (6:00 AM), OR
      - price has moved (either direction) >= SECONDARY_EXIT_MOVE_TRIGGER_PCT
        away from entry.
    Until armed, the SET entry-volatility stop (compute_entry_volatility_stop)
    is the ONLY active exit. This (plus filtering to bars at/after entry_time
    in _check_position) is what fixes the reported false first-exit-of-the-day:
    previously the divergence check ran from the very first poll after entry
    using the full day's bars — including bars from BEFORE the position
    existed, where bearish-looking pre-catalyst structure is common and
    isn't informative about the trade itself."""
    now_pt = datetime.now(timezone.utc).astimezone(PT_TZ)
    time_armed = now_pt.hour >= SECONDARY_EXIT_PST_HOUR
    move_pct = abs(current_price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price else 0.0
    move_armed = move_pct >= SECONDARY_EXIT_MOVE_TRIGGER_PCT
    return bool(time_armed or move_armed)


# ══════════════════════════════════════════════════════════════════════════
# INSTANT TICK VOLUME  (Schwab quote polling) — the instant post-news
# volume + up/down-split gate. No bars, no waiting for a bar to close.
# ══════════════════════════════════════════════════════════════════════════

def scaled_volume_baseline(avg_daily_volume: Optional[float],
                            window_sec: int = TICK_WINDOW_SEC) -> Optional[float]:
    """Scales a ticker's average DAILY volume down to a window_sec-sized
    baseline, using a flat fraction of the regular session
    (SESSION_SECONDS_FOR_SCALING). This is a flat approximation, not a
    time-of-day-matched baseline — see the config comment above
    SESSION_SECONDS_FOR_SCALING for why. No longer the PRIMARY RVOL
    baseline (see time_of_day_matched_volume_baseline); kept as the
    CONFIRMATION layer in rvol_dual_check."""
    if not avg_daily_volume or avg_daily_volume <= 0:
        return None
    return float(avg_daily_volume) * (window_sec / SESSION_SECONDS_FOR_SCALING)


def time_of_day_matched_volume_baseline(symbol: str, window_start_et: dt_time, window_sec: float,
                                         lookback_days: int = RVOL_TOD_LOOKBACK_DAYS
                                         ) -> Optional[float]:
    """PRIMARY RVOL baseline. Averages the volume traded in a
    window_sec-wide window that starts at the SAME ET clock time
    (window_start_et) on each of the last `lookback_days` PRIOR trading
    sessions — e.g. "how much does this ticker normally trade between
    4:00:00 and 4:01:00am ET" rather than "1/23400th of an average FULL
    DAY's volume." Real intraday volume is U-shaped (heavy at the
    open/close, thin overnight/premarket), so a flat scale-down of the
    daily average understates what's normal near the open/close and
    overstates what's normal at 4am — exactly backwards for an "is this
    unusual RIGHT NOW" comparison.

    Uses SchwabClient.get_price_history_multiday_1m (previously fetched but
    unused — this is what it was for). Returns None — caller falls back to
    the flat session-average baseline for both readings, see
    rvol_dual_check — if fewer than MIN_TOD_SAMPLE_DAYS prior sessions have
    a usable reading in that window (e.g. a recent IPO, or a data-fetch
    failure)."""
    try:
        df = SCHWAB.get_price_history_multiday_1m(symbol, days=lookback_days + 5, extended_hours=True)
    except Exception as e:
        print(f"  [WARN] {symbol}: multi-day history fetch failed ({e}) — no TOD-matched RVOL "
              f"baseline this check, falling back to the flat session-average baseline.")
        return None
    if df.empty:
        return None

    idx_et = df.index.tz_convert(NY_TZ)
    today_et = datetime.now(timezone.utc).astimezone(NY_TZ).date()

    daily_sums = []
    for day in sorted({ts.date() for ts in idx_et}, reverse=True):
        if day >= today_et:
            continue   # never use today (or a stale future-dated bar) as a baseline sample
        window_start = datetime.combine(day, window_start_et, tzinfo=NY_TZ)
        window_end = window_start + timedelta(seconds=window_sec)
        mask = (idx_et >= window_start) & (idx_et < window_end)
        if not mask.any():
            continue   # no bars at all in this window that day (e.g. no premarket data) — skip,
                       # don't silently count it as a real zero
        daily_sums.append(float(df.loc[mask, "Volume"].sum()))
        if len(daily_sums) >= lookback_days:
            break

    if len(daily_sums) < MIN_TOD_SAMPLE_DAYS:
        return None
    return sum(daily_sums) / len(daily_sums)


def rvol_dual_check(symbol: str, volume_in_window: float, window_sec: float,
                     window_start_et: dt_time, avg_daily_volume: Optional[float],
                     primary_multiple: float, confirm_multiple: float) -> dict:
    """The actual RVOL entry-signal check. PRIMARY reading is against the
    TIME-OF-DAY-MATCHED baseline (time_of_day_matched_volume_baseline),
    CONFIRMED against the flat full-session-average baseline
    (scaled_volume_baseline) — "is the book actually deep right now, or is
    this just what's normal for 4am." A ticker that never trades much this
    early could otherwise clear the primary gate on a trivial handful of
    shares simply because its own time-of-day norm is near zero; requiring
    the session-average reading to ALSO clear its (looser) floor guards
    against that. Both must pass for `pass` to be True.

    If no TOD baseline could be built (not enough matching history — see
    MIN_TOD_SAMPLE_DAYS), falls back to the flat session baseline for BOTH
    readings (`used_fallback`=True) rather than blocking entry outright
    just because multi-day history isn't available yet."""
    tod_baseline = time_of_day_matched_volume_baseline(symbol, window_start_et, window_sec)
    session_baseline = scaled_volume_baseline(avg_daily_volume, window_sec)

    used_fallback = tod_baseline is None
    primary_baseline = tod_baseline if tod_baseline is not None else session_baseline

    rvol_tod = (volume_in_window / primary_baseline) if primary_baseline and primary_baseline > 0 else 0.0
    rvol_session = (volume_in_window / session_baseline) if session_baseline and session_baseline > 0 else 0.0

    tod_ok = bool(primary_baseline and primary_baseline > 0 and rvol_tod >= primary_multiple)
    session_ok = bool(session_baseline and session_baseline > 0 and rvol_session >= confirm_multiple)

    return {
        "rvol_tod": rvol_tod, "rvol_session": rvol_session,
        "tod_baseline": tod_baseline, "session_baseline": session_baseline,
        "used_fallback": used_fallback,
        "tod_ok": tod_ok, "session_ok": session_ok,
        "pass": bool(tod_ok and session_ok),
    }


def classify_tick_volume(prev_quote: Optional[dict], curr_quote: dict, delta_volume: float) -> tuple:
    """Splits delta_volume (new cumulative volume since the previous poll)
    into (up_volume, down_volume) using the bid/ask of the CURRENT quote:
      - last >= ask  -> buy-side / up
      - last <= bid  -> sell-side / down
      - last strictly between bid/ask, or bid/ask missing -> fall back to
        comparing curr last vs prev last (rose -> up, fell -> down, no
        change/no prior -> split 50/50 so we never silently drop volume)."""
    if delta_volume <= 0:
        return 0.0, 0.0

    last = curr_quote.get("last")
    bid = curr_quote.get("bid")
    ask = curr_quote.get("ask")

    if last is not None and ask is not None and last >= ask:
        return delta_volume, 0.0
    if last is not None and bid is not None and last <= bid:
        return 0.0, delta_volume

    prev_last = prev_quote.get("last") if prev_quote else None
    if last is not None and prev_last is not None:
        if last > prev_last:
            return delta_volume, 0.0
        if last < prev_last:
            return 0.0, delta_volume

    half = delta_volume / 2.0
    return half, half


def instant_tick_volume_check(symbol: str, news_time: datetime,
                               avg_daily_volume: Optional[float]) -> dict:
    """Polls Schwab's quote endpoint once a second starting at news_time
    (blocking call, meant to be run from the news-handling thread/task).
    Fires the moment BOTH hold:
      - cumulative volume-since-news RVOL gate passes (see rvol_dual_check):
        volume-since-news >= TICK_VOLUME_MULTIPLE x a TIME-OF-DAY-MATCHED
        baseline (average volume this ticker traded in the same clock-time
        window on prior sessions) AND >= TICK_VOLUME_SESSION_CONFIRM_MULTIPLE
        x the flat full-session-average baseline (confirmation layer)
      - up_volume / total_volume >= TICK_UP_VOLUME_RATIO_MIN
    Gives up (pass=False) after TICK_WINDOW_SEC seconds with no fire.

    Returns a dict: {pass, volume_since_news, up_volume, down_volume,
    up_ratio, rvol_tod, rvol_session, tod_baseline, session_baseline,
    used_fallback_baseline, elapsed_sec, quote_polls_ok, quote_polls_failed,
    data_quality} for logging. `data_quality` is one of:
      "OK"      — at least one poll returned a usable total_volume reading
      "PARTIAL" — at least one usable reading, but some polls also failed
      "NO_DATA" — every single poll failed or never returned a usable
                  total_volume field, so volume_since_news==0 here means
                  "we never measured it," NOT "measured volume was zero."
    This exists specifically so a genuine zero-volume read can be told apart
    from a silently swallowed API timeout/error — previously both looked
    identical (volume_since_news=0) in the [TICK-CHECK] log line."""
    # window_start_et is the clock time (ET) the volume-since-news window
    # STARTS at — news_time's own time-of-day — so the TOD-matched baseline
    # compares against "what's normal volume in a TICK_WINDOW_SEC window
    # starting at THIS same clock time" on prior sessions, not a flat
    # fraction of an average day.
    window_start_et = news_time.astimezone(NY_TZ).time()

    start_totalvol = None
    prev_quote = None
    up_total = 0.0
    down_total = 0.0
    quote_polls_ok = 0       # polls that returned a usable total_volume reading
    quote_polls_failed = 0   # polls that raised, or returned no usable total_volume
    deadline = time.monotonic() + TICK_WINDOW_SEC

    while True:
        try:
            quote = SCHWAB.get_quote_full(symbol)
        except Exception as e:
            print(f"  [WARN] Instant tick-volume quote fetch failed for {symbol}: {e}")
            quote = None

        total_vol = quote.get("total_volume") if quote is not None else None
        if total_vol is not None:
            quote_polls_ok += 1
            if start_totalvol is None:
                start_totalvol = total_vol
            else:
                delta = float(total_vol) - float(start_totalvol) - (up_total + down_total)
                # delta is the volume added since the LAST poll only (we
                # track cumulative up_total+down_total so this is never
                # double counted across iterations)
                up_delta, down_delta = classify_tick_volume(prev_quote, quote, max(delta, 0.0))
                up_total += up_delta
                down_total += down_delta
            prev_quote = quote
        else:
            # Either the fetch raised, or it returned a quote with no usable
            # total_volume field — either way this poll contributed nothing,
            # and must NOT be silently indistinguishable from "0 shares
            # actually traded."
            quote_polls_failed += 1

        if quote_polls_ok == 0:
            data_quality = "NO_DATA"
        elif quote_polls_failed > 0:
            data_quality = "PARTIAL"
        else:
            data_quality = "OK"

        volume_since_news = up_total + down_total
        up_ratio = (up_total / volume_since_news) if volume_since_news > 0 else 0.0
        rvol = rvol_dual_check(symbol, volume_since_news, TICK_WINDOW_SEC, window_start_et,
                                avg_daily_volume, primary_multiple=TICK_VOLUME_MULTIPLE,
                                confirm_multiple=TICK_VOLUME_SESSION_CONFIRM_MULTIPLE)
        volume_ok = rvol["pass"]
        ratio_ok = up_ratio >= TICK_UP_VOLUME_RATIO_MIN

        if volume_ok and ratio_ok:
            return {
                "pass": True, "volume_since_news": volume_since_news,
                "up_volume": up_total, "down_volume": down_total,
                "up_ratio": up_ratio,
                "rvol_tod": rvol["rvol_tod"], "rvol_session": rvol["rvol_session"],
                "tod_baseline": rvol["tod_baseline"], "session_baseline": rvol["session_baseline"],
                "used_fallback_baseline": rvol["used_fallback"],
                "baseline": rvol["tod_baseline"] if rvol["tod_baseline"] is not None else rvol["session_baseline"],
                "elapsed_sec": TICK_WINDOW_SEC - max(deadline - time.monotonic(), 0.0),
                "quote_polls_ok": quote_polls_ok, "quote_polls_failed": quote_polls_failed,
                "data_quality": data_quality,
            }

        if time.monotonic() >= deadline:
            return {
                "pass": False, "volume_since_news": volume_since_news,
                "up_volume": up_total, "down_volume": down_total,
                "up_ratio": up_ratio,
                "rvol_tod": rvol["rvol_tod"], "rvol_session": rvol["rvol_session"],
                "tod_baseline": rvol["tod_baseline"], "session_baseline": rvol["session_baseline"],
                "used_fallback_baseline": rvol["used_fallback"],
                "baseline": rvol["tod_baseline"] if rvol["tod_baseline"] is not None else rvol["session_baseline"],
                "elapsed_sec": TICK_WINDOW_SEC,
                "quote_polls_ok": quote_polls_ok, "quote_polls_failed": quote_polls_failed,
                "data_quality": data_quality,
            }

        time.sleep(TICK_POLL_INTERVAL_SEC)


# ══════════════════════════════════════════════════════════════════════════
# MAGNITUDE SCORING — position SIZE only (see config block above for the
# rationale / weight choices). Never gates entry: this is only ever called
# for a catalyst that has ALREADY passed the sentiment gate and the instant
# tick-volume gate, right before it's about to enter.
# ══════════════════════════════════════════════════════════════════════════

def _is_stale_recap_headline(headline: str) -> bool:
    """True if the headline matches a known 'reporting on an already-happened
    move' pattern (see STALE_RECAP_HEADLINE_PATTERNS above) rather than
    delivering fresh information. Checked BEFORE the sentiment/LLM gate in
    handle_news_event so these never reach — and waste — an Ollama call."""
    text = headline.lower()
    return any(pat in text for pat in STALE_RECAP_HEADLINE_PATTERNS)


def float_score(fundamentals: dict) -> tuple:
    """Lower float -> higher score, scaled against the same FLOAT_CEILING_SHARES
    the watchlist screen already enforces (so every candidate here is already
    under that ceiling — this scores the DEGREE within it). Unknown float
    (scrape failed) falls back to a neutral 0.5 rather than penalizing or
    rewarding a name just because Finviz didn't return data for it."""
    float_shares = fundamentals.get("float_shares")
    if not float_shares or float_shares <= 0:
        return 0.5, None
    score = 1.0 - (float_shares / FLOAT_CEILING_SHARES)
    return max(0.0, min(1.0, score)), float_shares


def price_action_score(ticker: str, news_time: datetime) -> tuple:
    """Scores the PRICE_ACTION_LOOKBACK_MIN minutes of price action strictly
    BEFORE news_time. See the ASSUMPTION note in the config block above —
    this currently rewards pre-existing upward momentum; direction is
    flippable if evidence says otherwise. Returns (score, pct_change) where
    pct_change is None whenever there wasn't enough pre-news bar history to
    measure (score falls back to a neutral 0.5 in that case, e.g. news
    dropping right at/before the open)."""
    try:
        df = SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] Magnitude price-action fetch failed for {ticker}: {e}")
        return 0.5, None
    if df.empty:
        return 0.5, None

    pre_news = df[df.index < news_time]
    if len(pre_news) < 2:
        return 0.5, None

    lookback = pre_news.tail(PRICE_ACTION_LOOKBACK_MIN)
    start_close = float(lookback["Close"].iloc[0])
    end_close = float(lookback["Close"].iloc[-1])
    if start_close <= 0:
        return 0.5, None

    pct_change = (end_close - start_close) / start_close * 100.0
    clipped = max(-PRICE_ACTION_CLIP_PCT, min(PRICE_ACTION_CLIP_PCT, pct_change))
    score = (clipped + PRICE_ACTION_CLIP_PCT) / (2 * PRICE_ACTION_CLIP_PCT)
    return score, pct_change


def llm_score(llm: dict) -> tuple:
    """Combines qwen3's own confidence label with a crude length-based proxy
    for how substantive its reasoning was. Returns
    (score, confidence_component, richness_component, word_count)."""
    confidence_component = LLM_CONFIDENCE_SCORE.get(llm.get("confidence"), 0.3)
    reasoning = llm.get("reasoning") or ""
    word_count = len(reasoning.split())
    richness_component = min(word_count / LLM_REASONING_RICHNESS_WORD_CAP, 1.0)
    score = (LLM_CONFIDENCE_VS_RICHNESS_MIX * confidence_component
             + (1 - LLM_CONFIDENCE_VS_RICHNESS_MIX) * richness_component)
    return score, confidence_component, richness_component, word_count


def news_category_score(headline: str, summary: str) -> tuple:
    """Highest-scoring matched tier from MAGNITUDE_NEWS_CATEGORY_TIERS, else
    NEWS_CATEGORY_BASELINE_SCORE. Returns (score, category_name)."""
    text = f"{headline} {summary}".lower()
    best_score, best_cat = NEWS_CATEGORY_BASELINE_SCORE, "uncategorized"
    for category, score, keywords in MAGNITUDE_NEWS_CATEGORY_TIERS:
        if score > best_score and any(kw in text for kw in keywords):
            best_score, best_cat = score, category
    return best_score, best_cat


def magnitude_score_to_slots(composite: float) -> int:
    """Linear map from a composite score in [0,1] to an integer CAPITAL SLOT
    count in [MIN_SLOTS_PER_TRADE, MAX_SLOTS_PER_TRADE]. Each slot is worth
    CAPITAL_PER_SLOT_USD dollars (fixed once at startup by
    detect_capital_pool()) — converted to an actual share quantity later, in
    enter_position, once the entry price is known."""
    raw = MIN_SLOTS_PER_TRADE + composite * (MAX_SLOTS_PER_TRADE - MIN_SLOTS_PER_TRADE)
    return int(round(max(MIN_SLOTS_PER_TRADE, min(MAX_SLOTS_PER_TRADE, raw))))


def compute_magnitude_score(ticker: str, headline: str, summary: str, llm: dict,
                             news_time: datetime, fundamentals: dict) -> dict:
    """Composite magnitude score -> desired CAPITAL SLOT count for a NORMAL
    (non-FDA) entry that has already cleared both gates. Returns a dict with
    every sub-score included, for full visibility in the [MAGNITUDE] log
    line rather than just the final number."""
    f_score, float_shares = float_score(fundamentals)
    p_score, pct_change = price_action_score(ticker, news_time)
    l_score, conf_c, rich_c, word_count = llm_score(llm)
    n_score, category = news_category_score(headline, summary)

    composite = (MAGNITUDE_WEIGHT_FLOAT * f_score
                 + MAGNITUDE_WEIGHT_PRICE_ACTION * p_score
                 + MAGNITUDE_WEIGHT_LLM * l_score
                 + MAGNITUDE_WEIGHT_NEWS_CATEGORY * n_score)
    composite = max(0.0, min(1.0, composite))

    return {
        "composite": composite, "slots": magnitude_score_to_slots(composite),
        "float_score": f_score, "float_shares": float_shares,
        "price_action_score": p_score, "price_action_pct": pct_change,
        "llm_score": l_score, "llm_confidence_component": conf_c,
        "llm_richness_component": rich_c, "llm_reasoning_words": word_count,
        "news_category_score": n_score, "news_category": category,
    }


def confirmation_score(tick_result: dict) -> float:
    """Path B sizing, component 1/2. Scores HOW STRONGLY a tick-volume
    confirmation passed, not just pass/fail — a volume read barely over
    TICK_VOLUME_MULTIPLE and an up_ratio barely over TICK_UP_VOLUME_RATIO_MIN
    is a much weaker signal than one that blows both thresholds away. Same
    clip-then-normalize pattern as price_action_score above. Returns 0-1;
    caller (compute_breakout_magnitude_score) must already have result['pass']
    True — this scores the MARGIN of the pass, not whether it passed."""
    baseline = tick_result.get("baseline")
    volume_since = tick_result.get("volume_since_news", 0.0)
    up_ratio = tick_result.get("up_ratio", 0.0)

    if not baseline or baseline <= 0:
        volume_component = 0.5   # unknown baseline -> neutral, same convention as float_score
    else:
        multiple_achieved = volume_since / baseline
        clipped = max(0.0, min(multiple_achieved, TICK_VOLUME_MULTIPLE * 3))
        volume_component = clipped / (TICK_VOLUME_MULTIPLE * 3)

    if up_ratio >= TICK_UP_VOLUME_RATIO_MIN:
        up_ratio_component = (up_ratio - TICK_UP_VOLUME_RATIO_MIN) / (1.0 - TICK_UP_VOLUME_RATIO_MIN)
    else:
        up_ratio_component = 0.0   # shouldn't happen if caller already checked result['pass']

    return max(0.0, min(1.0, 0.5 * volume_component + 0.5 * up_ratio_component))


def gap_scanner_score(gap_pct: float, rvol: float) -> float:
    """Path B sizing, component 2/2. Scores the raw size of the gap/RVOL
    reading that got a ticker flagged by Tier 1 (or, for the halt path, the
    gap/RVOL measured at the moment of reopen) — clipped/normalized the same
    way price_action_score clips pre-news momentum, so one extreme outlier
    doesn't dominate the composite. Returns 0-1."""
    gap_component = max(0.0, min(gap_pct, GAP_PCT_SCORE_CLIP)) / GAP_PCT_SCORE_CLIP
    rvol_component = max(0.0, min(rvol, RVOL_SCORE_CLIP)) / RVOL_SCORE_CLIP
    return max(0.0, min(1.0, 0.5 * gap_component + 0.5 * rvol_component))


def compute_breakout_magnitude_score(tick_result: dict, gap_pct: float, rvol: float) -> dict:
    """Path B's equivalent of compute_magnitude_score — used by BOTH the
    RVOL/gap breakout path and the halt-reopen path (per request), since both
    ultimately have the same two ingredients available: a tick-volume
    confirmation result, and a gap%/RVOL reading. No LLM component — there is
    no fresh catalyst read for either sub-path by design."""
    c_score = confirmation_score(tick_result)
    g_score = gap_scanner_score(gap_pct, rvol)
    composite = max(0.0, min(1.0, BREAKOUT_WEIGHT_CONFIRMATION * c_score
                                    + BREAKOUT_WEIGHT_GAP_SCANNER * g_score))
    return {
        "composite": composite, "slots": magnitude_score_to_slots(composite),
        "confirmation_score": c_score, "gap_scanner_score": g_score,
        "gap_pct": gap_pct, "rvol": rvol,
    }


# ══════════════════════════════════════════════════════════════════════════
# POSITIONS — entry signal + PnL tracking (no order execution / no quote API)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    ticker: str
    entry_time: datetime
    entry_price: float
    peak_price: float
    reason: str
    quantity: float = 0.0
    news_time: Optional[datetime] = None
    # SET (non-trailing) entry-volatility stop — computed ONCE in
    # enter_position via compute_entry_volatility_stop, never recalculated
    # or trailed with price afterward. stop_price defaults to entry_price
    # here only as a safe placeholder in case something upstream ever
    # constructs a Position without going through enter_position; the real
    # trades entrance always overwrites this immediately with the actual
    # volatility-derived stop.
    stop_price: float = 0.0
    stop_pct: float = MAX_LOSS_PCT
    stop_capped: bool = True
    entry_atr: Optional[float] = None


def log_trade_row(row: dict):
    write_header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "ticker", "time", "price", "value", "reason", "headline"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def _seamless_session_gap_warning() -> Optional[str]:
    """Returns a warning string if it's currently before
    SCHWAB_SEAMLESS_SESSION_START_ET (7:00 AM ET) — the real start of
    Schwab's Day+extended "seamless" order eligibility, three hours after
    this bot's own PREMARKET_OPEN_ET (4:00 AM ET) watch window opens. An
    order (BUY or SELL) placed in that gap can be ACCEPTED by the API
    without necessarily being eligible to fill yet — see the config comment
    above SCHWAB_SEAMLESS_SESSION_START_ET. Returns None once past 7:00 AM
    ET (no gap)."""
    now_et = datetime.now(timezone.utc).astimezone(NY_TZ)
    if now_et.time() < SCHWAB_SEAMLESS_SESSION_START_ET:
        return (f"placed at {now_et.strftime('%H:%M:%S')} ET, before Schwab's "
                f"{SCHWAB_SEAMLESS_SESSION_START_ET.strftime('%H:%M')} ET seamless-session start — "
                f"may sit accepted-but-not-yet-eligible-to-fill until then")
    return None


def enter_position(ticker: str, reason: str, headline: str, news_time: Optional[datetime] = None,
                    bypass_gates: bool = False, desired_slots: Optional[int] = None):
    """
    bypass_gates=True is the FDA-approval fast path ONLY — the sole exception
    that skips the halt check, the liquidity check, and the price-ceiling
    check, and claims ALL remaining CAPITAL SLOTS (i.e. the entire remaining
    capital pool) instead of a magnitude-scaled amount. Everything else
    (normal catalysts) draws `desired_slots` (computed by
    compute_magnitude_score — see caller), clamped to [MIN_SLOTS_PER_TRADE,
    MAX_SLOTS_PER_TRADE], from the shared pool — possibly fewer slots than
    requested if the pool is running low — and still must pass
    halt/liquidity/price-ceiling. Slots are converted to an actual share
    quantity (claimed_slots * CAPITAL_PER_SLOT_USD / entry_price) once the
    entry price is known, further down.
    """
    with _positions_lock:
        if ticker in open_positions:
            print(f"  [SKIP] {ticker} already has an open position — ignoring duplicate signal.")
            return

    if bypass_gates:
        slots = claim_fda_all_remaining()
        if slots <= 0:
            print(f"  [SKIP] {ticker}: FDA fast-path fired but 0 capital slots remain — not entering.")
            return
    else:
        requested = desired_slots if desired_slots else MIN_SLOTS_PER_TRADE
        requested = max(MIN_SLOTS_PER_TRADE, min(int(requested), MAX_SLOTS_PER_TRADE))
        slots = try_claim_shares(requested)
        if slots <= 0:
            print(f"  [SKIP] {ticker}: no capital slots remain this run ({opportunities_left()} left) — not entering.")
            return
        if slots < requested:
            print(f"  [SIZING] {ticker}: magnitude score requested {requested} slot(s) but only "
                  f"{slots} remained in the pool — claimed {slots}.")

        if is_halted(ticker):
            print(f"  [GATE] {ticker}: halted — entry blocked.")
            release_opportunity(slots)
            return
        if not passes_liquidity_gate(ticker):
            release_opportunity(slots)
            return

    quote = SCHWAB.get_quote_full(ticker)
    entry_price = quote.get("last")
    if not entry_price:
        try:
            df = SCHWAB.get_price_history_1m(ticker)
            if not df.empty:
                entry_price = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"  [WARN] Could not fetch entry price for {ticker} from price history: {e}")
    if not entry_price:
        print(f"  [ABORT] {ticker}: no usable price — cannot place the order.")
        release_opportunity(slots)
        return

    # PRICE_CEILING_USD caps per-share cost — skipped for the FDA fast path
    # per the "bypasses everything" instruction; still enforced for normal
    # (gated) entries.
    if not bypass_gates and entry_price > PRICE_CEILING_USD:
        print(f"  [ABORT] {ticker}: price ${entry_price:.2f} exceeds ${PRICE_CEILING_USD:.2f}/share ceiling — entry blocked.")
        release_opportunity(slots)
        return

    # ---- Convert claimed CAPITAL SLOTS into an actual share quantity ------
    # Each slot is worth CAPITAL_PER_SLOT_USD dollars (fixed once at startup
    # by detect_capital_pool()). If the initial slot allocation can't even
    # afford 1 share at this price (e.g. a low-magnitude signal got 1 slot
    # but the stock trades near PRICE_CEILING_USD), try claiming additional
    # slots — up to MAX_SLOTS_PER_TRADE and whatever the pool has left —
    # before giving up. The FDA path already claimed the ENTIRE pool, so
    # there's nothing further to claim there if it still can't afford 1 share.
    claimed_dollars = slots * CAPITAL_PER_SLOT_USD
    quantity = int(claimed_dollars // entry_price) if entry_price > 0 else 0

    if quantity < 1 and not bypass_gates:
        while quantity < 1 and slots < MAX_SLOTS_PER_TRADE:
            extra = try_claim_shares(1)
            if extra <= 0:
                break
            slots += extra
            claimed_dollars = slots * CAPITAL_PER_SLOT_USD
            quantity = int(claimed_dollars // entry_price)

    if quantity < 1:
        print(f"  [ABORT] {ticker}: allocated capital (${claimed_dollars:.2f} across {slots} "
              f"slot(s) of ${CAPITAL_PER_SLOT_USD:.2f} each) can't buy even 1 share at "
              f"${entry_price:.2f} — entry blocked.")
        release_opportunity(slots)
        return

    # Marketable limit price for the buy: prefer the live ask (tighter/more
    # realistic than last in a fast-moving premarket tape), fall back to last.
    # place_equity_order() no longer supports MARKET orders — Schwab flatly
    # rejects those outside regular hours, so every entry needs a real price.
    buy_ref = quote.get("ask") or entry_price
    limit_price = round(buy_ref * (1 + LIMIT_ORDER_SLIPPAGE_PCT), 2)

    gap_warning = _seamless_session_gap_warning()
    if gap_warning:
        print(f"  [WARN] {ticker}: BUY order {gap_warning}. If it doesn't fill by 7:00 AM ET, "
              f"it's still working at Schwab, not lost — check the account, don't assume it failed.")

    try:
        SCHWAB_TRADER.place_equity_order(SCHWAB_ACCOUNT_HASH, ticker, "BUY", quantity, limit_price)
    except Exception as e:
        # Order was rejected by Schwab. DEBUG HELP: some securities (certain
        # low-float/OTC/restricted names) can't be bought through the Trader
        # API at all and require a live broker rep — place_equity_order()
        # surfaces Schwab's actual error body in `e` so that reason shows up
        # here instead of a bare HTTP status. Give back the claimed
        # opportunity(ies) so the run can still catch a different catalyst.
        print(f"  [ABORT] {ticker}: order rejected — {e}")
        print(f"          {quantity} whole share(s), limit=${limit_price:.2f}/share "
              f"(notional ~${limit_price*quantity:.2f}) was rejected by the API. "
              f"If this is a broker-assisted-only security, Schwab's error body above "
              f"should say so explicitly — this order will NOT show up as filled/working "
              f"in the account.")
        release_opportunity(slots)
        return

    # ---- SET (non-trailing) entry-volatility stop, computed ONCE here -----
    # See compute_entry_volatility_stop's docstring / the config comment
    # above ENTRY_STOP_ATR_PERIOD for why this replaced the Chandelier ATR
    # TRAIL. Needs today's 1-minute bars to measure ATR; if that fetch fails
    # or there aren't enough bars yet, compute_entry_volatility_stop falls
    # back to the flat MAX_LOSS_PCT on its own (logged below either way).
    try:
        entry_df = SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] {ticker}: couldn't fetch bars for the entry-volatility stop ({e}) — "
              f"using the flat {MAX_LOSS_PCT*100:.0f}% stop instead.")
        entry_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    stop = compute_entry_volatility_stop(entry_df, entry_price)
    cap_note = " [10% cap — ATR implied wider]" if stop["capped"] else ""
    atr_note = "ATR unavailable" if stop["atr"] is None else f"ATR={stop['atr']:.4f}"
    print(f"  [STOP] {ticker}: SET entry-volatility stop = ${stop['stop_price']:.2f} "
          f"({stop['stop_pct']*100:.1f}% off entry{cap_note}, {atr_note}) — "
          f"fixed for the entire trade, does not trail.")

    pos = Position(ticker=ticker, entry_time=datetime.now(timezone.utc),
                    entry_price=entry_price, peak_price=entry_price, reason=reason,
                    quantity=quantity, news_time=news_time,
                    stop_price=stop["stop_price"], stop_pct=stop["stop_pct"],
                    stop_capped=stop["capped"], entry_atr=stop["atr"])
    with _positions_lock:
        open_positions[ticker] = pos

    print(f"\n  \U0001F7E2 ENTRY  {ticker:<6}  ref=${entry_price:.2f}  limit=${limit_price:.2f}  qty={quantity}  "
          f"notional=${limit_price*quantity:.2f}  slots={slots}/{TOTAL_BUY_OPPORTUNITIES} "
          f"(${claimed_dollars:.2f} allocated)  reason={reason}")
    print(f"     \u2514\u2500 {headline}\n")
    print(f"  [OPPORTUNITIES] {opportunities_left()}/{TOTAL_BUY_OPPORTUNITIES} capital slot(s) remaining this run.\n")

    log_trade_row({
        "event": "ENTRY", "ticker": ticker, "time": pos.entry_time.isoformat(),
        "price": f"{entry_price:.4f}", "value": f"qty={quantity}", "reason": reason, "headline": headline,
    })


def exit_position(ticker: str, price: float, pnl_pct: float, reason: str):
    """`price`/`pnl_pct` are the REFERENCE values from the signal that
    triggered this call (current market price at the moment the stop/exit
    condition was checked) — NOT a confirmed fill. This function only
    treats the position as closed, and only logs/reports it as an EXIT
    with a P&L number, once an actual fill is confirmed via
    get_order_fill_details(). Previously the position was popped from
    open_positions (removing it from further monitoring) and an "EXIT"
    row with `pnl_pct` was logged UNCONDITIONALLY, even if the SELL order
    never filled — meaning a limit that price never actually reached could
    print a P&L number that had nothing to do with what the account
    actually did, while simultaneously abandoning the position from any
    further stop/exit monitoring. That's the reported bug where an exit
    was logged/printed but "the order did not close" and price never
    touched the level it claimed to have exited at."""
    with _positions_lock:
        pos = open_positions.get(ticker)   # PEEK, don't remove yet — removed only on confirmed fill
    if pos is None:
        return

    with _exiting_lock:
        if ticker in _exiting_tickers:
            print(f"  [SKIP] {ticker}: an exit is already in flight for this ticker — not "
                  f"starting a second concurrent SELL attempt.")
            return
        _exiting_tickers.add(ticker)
    try:
        return _exit_position_inner(ticker, price, pnl_pct, reason, pos)
    finally:
        with _exiting_lock:
            _exiting_tickers.discard(ticker)


def _exit_position_inner(ticker: str, price: float, pnl_pct: float, reason: str, pos: "Position"):
    gap_warning = _seamless_session_gap_warning()
    if gap_warning:
        print(f"  [WARN] {ticker}: SELL (exit) order {gap_warning}. A signal firing in this "
              f"window can be accepted without being fillable yet — that gap, not a broken "
              f"exit signal, is the most likely explanation if a triggered exit doesn't "
              f"promptly show up as filled.")

    # Marketable limit price for the sell: prefer the live bid (tighter/more
    # realistic than the reference `price` in a fast-moving tape), fall back
    # to `price`. Same reasoning as enter_position — MARKET orders are no
    # longer supported by place_equity_order() since Schwab rejects them
    # outside regular hours.
    try:
        sell_ref = SCHWAB.get_quote_full(ticker).get("bid") or price
    except Exception:
        sell_ref = price
    limit_price = round(sell_ref * (1 - LIMIT_ORDER_SLIPPAGE_PCT), 2)

    # ── EXIT FILL CONFIRMATION ───────────────────────────────────────────────
    # Give each attempt EXIT_FILL_CHECK_SEC to fill; if it hasn't, cancel it
    # and resubmit at a wider discount off the current bid, up to
    # EXIT_FILL_MAX_ATTEMPTS times. The position stays in open_positions
    # (still monitored, still protected by the SET stop / secondary exit)
    # for the ENTIRE loop below — it's only removed after the `if filled:`
    # block further down confirms an actual fill.
    order_id = None
    filled = False
    fill_details = None
    for attempt in range(1, EXIT_FILL_MAX_ATTEMPTS + 1):
        try:
            result = SCHWAB_TRADER.place_equity_order(SCHWAB_ACCOUNT_HASH, ticker, "SELL",
                                                        pos.quantity, limit_price)
            order_id = result.get("order_id")
        except Exception as e:
            # DEBUG HELP: same as the BUY side — place_equity_order() surfaces
            # Schwab's actual rejection body (restricted security, halted, bad
            # account state, etc.) here instead of a bare HTTP error.
            print(f"  [ERROR] SELL order failed for {ticker} (qty={pos.quantity}, limit=${limit_price:.2f}): {e}")
            print(f"          Position remains in open_positions and under active monitoring — "
                  f"this exit attempt will be retried on the next poll if the exit condition "
                  f"still holds. If Schwab's error body above mentions a broker-assisted-only / "
                  f"restricted security, this symbol cannot be closed via the Trader API and "
                  f"needs a phone/rep order.")
            break

        deadline = time.monotonic() + EXIT_FILL_CHECK_SEC
        status = None
        while time.monotonic() < deadline:
            fill_details = SCHWAB_TRADER.get_order_fill_details(SCHWAB_ACCOUNT_HASH, order_id)
            status = fill_details.get("status") if fill_details else None
            if status == "FILLED":
                filled = True
                break
            if status in ("CANCELED", "REJECTED", "EXPIRED"):
                break
            time.sleep(EXIT_FILL_POLL_INTERVAL_SEC)

        if filled:
            print(f"  [FILLED] {ticker}: exit order {order_id} confirmed filled within "
                  f"{EXIT_FILL_CHECK_SEC}s (attempt {attempt}/{EXIT_FILL_MAX_ATTEMPTS}).")
            break

        # Not filled within the window — cancel this attempt (best-effort;
        # it may have already filled/rejected/expired between our last poll
        # and now) and, if attempts remain, widen the limit and resubmit.
        SCHWAB_TRADER.cancel_order(SCHWAB_ACCOUNT_HASH, order_id)

        if attempt < EXIT_FILL_MAX_ATTEMPTS:
            try:
                sell_ref = SCHWAB.get_quote_full(ticker).get("bid") or sell_ref
            except Exception:
                pass
            widen = EXIT_LIMIT_WIDEN_STEP_PCT * attempt
            limit_price = round(sell_ref * (1 - LIMIT_ORDER_SLIPPAGE_PCT - widen), 2)
            print(f"  [RETRY] {ticker}: exit order not filled within {EXIT_FILL_CHECK_SEC}s "
                  f"(last status={status}) — widening limit to ${limit_price:.2f} and resubmitting "
                  f"(attempt {attempt + 1}/{EXIT_FILL_MAX_ATTEMPTS}).")
        else:
            print(f"  [WARN] {ticker}: exit still unfilled after {EXIT_FILL_MAX_ATTEMPTS} widened "
                  f"attempts (last status={status}, last limit=${limit_price:.2f}).")

    if not filled:
        # UNFILLED: the position is NOT removed from open_positions and is
        # NOT logged as an "EXIT" with a P&L number — there is no fill, so
        # there is no realized P&L to report. It remains fully monitored;
        # the same exit condition (or a different one) will be re-evaluated
        # on the next poll (POLL_INTERVAL_SEC), which will call this
        # function again and try again with a fresh reference price.
        print(f"  [UNFILLED] {ticker}: exit NOT confirmed — position remains open and under "
              f"active monitoring (SET stop + secondary exit both still apply). This will be "
              f"retried automatically on the next poll if the exit condition still holds.")
        log_trade_row({
            "event": "EXIT_ATTEMPT_UNFILLED", "ticker": ticker, "time": datetime.now(timezone.utc).isoformat(),
            "price": f"{price:.4f}", "value": "unfilled", "reason": reason, "headline": "",
        })
        return

    # FILLED: NOW remove from open_positions and compute REAL P&L from the
    # actual average fill price — falling back to the reference `price`
    # (marked UNCONFIRMED) only if fill_details couldn't extract one.
    with _positions_lock:
        pos = open_positions.pop(ticker, None)
    if pos is None:
        return   # shouldn't happen (we still held the lock the whole time above), but don't crash

    avg_fill_price = fill_details.get("avg_fill_price") if fill_details else None
    if avg_fill_price is not None:
        actual_price = avg_fill_price
        price_note = ""
    else:
        actual_price = price
        price_note = " [UNCONFIRMED — fill price unavailable, using reference price]"
    actual_pnl_pct = (actual_price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price else 0.0

    icon = "\U0001F534" if actual_pnl_pct < 0 else "\U0001F7E2"
    dollar_pnl = (actual_price - pos.entry_price) * pos.quantity
    print(f"\n  {icon} EXIT   {ticker:<6}  fill=${actual_price:.2f}{price_note}  qty={pos.quantity}  "
          f"pnl={actual_pnl_pct:+.2f}% (${dollar_pnl:+.4f})  reason={reason}\n")
    log_trade_row({
        "event": "EXIT", "ticker": ticker, "time": datetime.now(timezone.utc).isoformat(),
        "price": f"{actual_price:.4f}", "value": f"{actual_pnl_pct:+.2f}", "reason": reason, "headline": "",
    })

    if run_is_complete():
        print("=" * 60)
        print(f"  RUN COMPLETE — all {TOTAL_BUY_OPPORTUNITIES} buying opportunities used "
              f"and all positions closed.")
        print("=" * 60)
        request_shutdown("All buying opportunities used and all positions closed.")


def _check_position(ticker: str):
    with _positions_lock:
        pos = open_positions.get(ticker)
    if pos is None:
        return

    if is_halted(ticker):
        print(f"  [HALT] {ticker} is halted — holding, no exit check this cycle.")
        return

    try:
        df = SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] PnL poll failed for {ticker}: {e}")
        return
    if df.empty:
        return

    current_price = float(df["Close"].iloc[-1])

    with _positions_lock:
        pos.peak_price = max(pos.peak_price, current_price)
        peak = pos.peak_price
        entry_price = pos.entry_price
        stop_price = pos.stop_price
        stop_pct = pos.stop_pct
        stop_capped = pos.stop_capped

    pnl_pct = (current_price - entry_price) / entry_price * 100.0 if entry_price else 0.0

    exit_reason = None

    # 1. SET (non-trailing) entry-volatility stop — active for the ENTIRE
    #    duration of the trade, computed ONCE at entry (enter_position /
    #    compute_entry_volatility_stop) and never recalculated or trailed
    #    with price. This check runs regardless of time-of-day or move size.
    if current_price <= stop_price:
        cap_note = " [10% cap]" if stop_capped else ""
        exit_reason = (f"SET entry-volatility stop break (stop=${stop_price:.2f}, "
                        f"{stop_pct*100:.1f}% off entry{cap_note})")

    # 2. Secondary exit — significant bearish CVD OR bearish OBV/price swing
    #    divergence — only checked once _secondary_exit_armed() says so
    #    (6am Pacific OR a significant move from entry). Restricted to bars
    #    AT OR AFTER entry_time so pre-entry structure (often already
    #    bearish-looking right before a breakout catalyst) can't be misread
    #    as a signal about THIS trade — the root cause behind the reported
    #    false first-exit-of-the-day.
    armed = _secondary_exit_armed(pos, current_price)
    if exit_reason is None and armed:
        since_entry = df[df.index >= pd.Timestamp(pos.entry_time)].tail(EXIT_LOOKBACK_BARS)
        if len(since_entry) >= 2:
            divergence = bearish_swing_divergence(since_entry, window=SWING_WINDOW_EXIT)
            cvd = significant_cvd_bearish(since_entry, lookback_bars=CVD_LOOKBACK_BARS,
                                           threshold=CVD_BEARISH_THRESHOLD)
            if divergence:
                exit_reason = "Bearish OBV/price swing divergence (secondary exit, armed)"
            elif cvd["significant"]:
                exit_reason = (f"Significant bearish CVD (secondary exit, armed; "
                                f"net_cvd_ratio={cvd['net_cvd_ratio']:.2f} <= {CVD_BEARISH_THRESHOLD})")

    if exit_reason:
        exit_position(ticker, current_price, pnl_pct, exit_reason)
    else:
        print(f"  [PNL] {ticker:<6} px=${current_price:.2f}  pnl={pnl_pct:+.2f}%  peak=${peak:.2f}  "
              f"stop=${stop_price:.2f}  secondary_armed={armed}")


def monitor_positions_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        with _positions_lock:
            tickers = list(open_positions.keys())
        for ticker in tickers:
            try:
                _check_position(ticker)
            except Exception as e:
                print(f"  [WARN] monitor error on {ticker}: {e}")
        stop_event.wait(POLL_INTERVAL_SEC)


def heartbeat_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        with _watchlist_lock:
            wl = set(watchlist)
        with _positions_lock:
            pos_count = len(open_positions)
        with _news_lock:
            last_news = _last_news_at
        with _acked_lock:
            acked = set(acked_news_symbols)

        now = datetime.now(timezone.utc)
        last_news_str = "none yet" if last_news is None else f"{int((now - last_news).total_seconds())}s ago"

        # watchlist = our local Finviz-scraped set. acked = symbols Alpaca's
        # server has actually confirmed subscribing us to for news, per its
        # own subscription-ack frames (see VerboseNewsDataStream). If these
        # two numbers diverge and stay diverged, some tickers were sent in
        # the subscribe request but never confirmed.
        missing = wl - acked
        acked_str = f"{len(acked)}/{len(wl)}"
        if missing and len(missing) <= 10:
            acked_str += f"  UNCONFIRMED={sorted(missing)}"
        elif missing:
            acked_str += f"  UNCONFIRMED={len(missing)} tickers (e.g. {sorted(missing)[:5]}...)"

        # Displayed in US/Eastern (the market's own timezone) rather than UTC
        # for at-a-glance readability during the trading session — all
        # internal timestamps/comparisons elsewhere remain UTC (news_time,
        # entry_time, log_trade_row, etc.); this is a display-only change.
        now_et = now.astimezone(NY_TZ)
        print(f"  [HEARTBEAT] {now_et.strftime('%H:%M:%S')} ET  alive  "
              f"watchlist={len(wl)}  server_acked={acked_str}  "
              f"open_positions={pos_count}  opportunities_left={opportunities_left()}/{TOTAL_BUY_OPPORTUNITIES}  "
              f"last_news={last_news_str}")
        stop_event.wait(HEARTBEAT_SEC)


# ══════════════════════════════════════════════════════════════════════════
# NEWS HANDLER — FDA bypass, else composite sentiment gate -> instant RVOL gate
# (no delayed confirmation window anymore — both gates fire immediately)
# ══════════════════════════════════════════════════════════════════════════

def handle_news_event(ticker: str, headline: str, summary: str, news_time: Optional[datetime] = None):
    if opportunities_left() <= 0:
        return   # pool exhausted — no more entries this run (FDA can't claim 0 either)

    with _positions_lock:
        already_open = ticker in open_positions
    if already_open:
        print(f"  [SKIP] {ticker} already has an open position — not re-screening this news.")
        return

    full_text = f"{headline} {summary}".strip()
    # Anchor on the news article's own timestamp, not "now" — using local
    # receive time here would mean the instant-RVOL window and its baseline
    # alignment are keyed off whenever this thread got around to processing
    # the event (websocket delivery + queueing behind other tickers in the
    # same headline + VADER/LM scoring time), not the actual news drop.
    if news_time is None:
        news_time = datetime.now(timezone.utc)
    elif news_time.tzinfo is None:
        news_time = news_time.replace(tzinfo=timezone.utc)
    else:
        news_time = news_time.astimezone(timezone.utc)

    if contains_fda_fastpath(full_text):
        # Keyword match is now only a CANDIDATE — qwen must confirm this is a
        # genuine, already-decided, explicit approval before the bypass
        # fires. Deliberately checked BEFORE the recycled/stale-recap filters
        # below: a true approval candidate should never be suppressible by
        # either — it's rare and high-value enough to get first look, full
        # stop, ahead of any other filter in this function. If qwen is
        # unreachable or doesn't confirm, this does NOT drop the headline —
        # it just falls through to the normal gated flow below instead of
        # the instant bypass, since OLLAMA_ENABLED not being disabled entirely.
        fda_check = None
        if OLLAMA_ENABLED:
            fda_check = call_ollama_fda_check(headline, summary)
        if fda_check is not None and fda_check["is_explicit_approval"] and fda_check["confidence"] != "low":
            print(f"  [FDA FAST-PATH] {ticker}: keyword match confirmed by {OLLAMA_MODEL} "
                  f"(confidence={fda_check['confidence']}) as a genuine explicit approval — "
                  f"bypassing every gate, claiming all remaining opportunities, entering directly. "
                  f"Reasoning: {fda_check['reasoning'][:120]}")
            enter_position(ticker, reason="FDA_APPROVAL_FASTPATH", headline=headline,
                            news_time=news_time, bypass_gates=True)
            return
        elif fda_check is not None:
            print(f"  [FDA FAST-PATH] {ticker}: keyword match but {OLLAMA_MODEL} did NOT confirm an "
                  f"explicit granted approval (is_explicit_approval={fda_check['is_explicit_approval']}, "
                  f"confidence={fda_check['confidence']}) — no bypass, falling through to the normal "
                  f"gated flow. Reasoning: {fda_check['reasoning'][:120]}")
        else:
            print(f"  [FDA FAST-PATH] {ticker}: keyword match but qwen confirmation unavailable — "
                  f"no bypass (fail-closed on the fast path only), falling through to the normal "
                  f"gated flow instead of dropping this headline entirely.")

    if _is_recycled_headline(ticker, headline):
        print(f"  [NEWS-DROP] {ticker}: headline is a near-duplicate of a recent headline for "
              f"this ticker (fuzzy similarity >= {NEWS_DEDUP_SIMILARITY_THRESHOLD}, independent of "
              f"article id) — treating as recycled/republished, skipping. {headline[:70]}")
        return

    if _is_stale_recap_headline(headline):
        print(f"  [NEWS-DROP] {ticker}: headline matches a known 'reporting on an already-happened "
              f"move' recap pattern (e.g. 'why shares are up') rather than delivering fresh "
              f"information — skipping before spending an LLM call on it.  {headline[:70]}")
        return

    # ── TIER 2 (LLM) IS NOW THE PRIMARY GATE ────────────────────────────────
    # Tier 1 (VADER + Loughran-McDonald) can no longer approve an entry by
    # itself. Every headline goes to Ollama first; tier 1 only ever acts as a
    # secondary confirmation AFTER Ollama has already called it positive —
    # see the config comment above OLLAMA_ENABLED for the exact rule matrix.
    if not OLLAMA_ENABLED:
        print(f"  [GATE] {ticker}: blocked — LLM escalation is disabled (OLLAMA_ENABLED=False), "
              f"and tier-1 alone is no longer sufficient for entry. No entry.  {headline[:70]}")
        return

    llm = call_ollama_catalyst_check(headline, summary)
    if llm is None:
        print(f"  [ESCALATE] {ticker}: local LLM unavailable/unparseable — fail-closed, no entry "
              f"(tier-1 is never consulted without a tier-2 read first).  {headline[:70]}")
        return

    if llm["is_stale_or_secondhand"]:
        # qwen's own recap/listicle/secondhand-reporting check — catches
        # cases the STALE_RECAP_HEADLINE_PATTERNS dictionary above missed
        # (paraphrased recap wording, roundup pieces that don't match a
        # literal phrase in the list, etc.). See the dictionary's config
        # comment for why both checks exist rather than just one.
        print(f"  [ESCALATE] {ticker}: {OLLAMA_MODEL} flagged this as secondhand/recap/listicle "
              f"reporting rather than a primary catalyst — no entry. Reasoning: {llm['reasoning'][:120]}")
        return

    if llm["catalyst"] != "positive":
        print(f"  [ESCALATE] {ticker}: {OLLAMA_MODEL} says catalyst={llm['catalyst']} "
              f"confidence={llm['confidence']} — no entry. Reasoning: {llm['reasoning'][:120]}")
        return

    # Confirmed-positive, non-stale catalyst — record it for the rolling-high
    # breakout path regardless of whether THIS instant reaction check (below)
    # ends up firing. That path is specifically for delayed follow-through on
    # news that already cleared this exact gate, so it needs to know this
    # happened even if today's immediate tick-volume read doesn't confirm.
    _record_bullish_news(ticker, news_time or datetime.now(timezone.utc))

    passed, detail = passes_sentiment_gate(headline, summary)

    if llm["confidence"] == "high":
        # Confident LLM positive is sufficient on its own — tier 1 is only
        # logged here for visibility, never used to veto a high-confidence
        # tier-2 read (the dictionary stack is the one known to be blind to
        # flat/factual good news, not the other way around).
        print(f"  [ESCALATE-PASS] {ticker}: {OLLAMA_MODEL} confirms a POSITIVE catalyst "
              f"(confidence=high) — tier-1 confirmation not required at this confidence "
              f"(tier-1 read: P(neg)={detail['p_negative_worst']:.3f} P(pos)={detail['p_positive_worst']:.3f}, "
              f"pass={passed}) — firing instant RVOL check now. Reasoning: {llm['reasoning'][:120]}")
        _instant_rvol_check_and_maybe_enter(ticker, headline, summary, news_time, llm)
        return

    # medium/low confidence: tier 1 (VADER+LM) must ALSO confirm positive.
    if not passed:
        print(f"  [GATE] {ticker}: {OLLAMA_MODEL} said POSITIVE (confidence={llm['confidence']}) but "
              f"tier-1 secondary confirmation failed — P(negative)={detail['p_negative_worst']:.3f} "
              f"(need <{NEGATIVE_PROB_MAX}) / P(positive)={detail['p_positive_worst']:.3f} "
              f"(need >{POSITIVE_PROB_MIN}). At sub-high LLM confidence both tiers must agree — no entry. "
              f"{headline[:70]}")
        return

    print(f"  [ESCALATE-PASS] {ticker}: {OLLAMA_MODEL} confirms a POSITIVE catalyst "
          f"(confidence={llm['confidence']}), tier-1 secondary confirmation also passed "
          f"(P(neg)={detail['p_negative_worst']:.3f} P(pos)={detail['p_positive_worst']:.3f}) — "
          f"firing instant RVOL check now. Reasoning: {llm['reasoning'][:120]}")
    _instant_rvol_check_and_maybe_enter(ticker, headline, summary, news_time, llm)


def _instant_rvol_check_and_maybe_enter(ticker: str, headline: str, summary: str,
                                         news_time: datetime, llm: dict):
    """The second (and final) entry GATE. Fires IMMEDIATELY after the
    sentiment gate passes — no delay, no waiting for any bar to close. Polls
    Schwab quotes once a second and fires the instant BOTH hold:
      - volume-since-news >= TICK_VOLUME_MULTIPLE x a baseline scaled from the
        ticker's average daily volume
      - up_volume / total_volume >= TICK_UP_VOLUME_RATIO_MIN (bid/ask split)
    Gives up after TICK_WINDOW_SEC seconds. This replaces the old bar-based
    instant RVOL gate, which had to wait for a 1-minute bar to exist before it
    could measure anything — that wait was the source of the lag.

    `summary` and `llm` are only used AFTER this gate passes, to size the
    position via compute_magnitude_score() — they play no role in the
    gate itself."""
    with _positions_lock:
        if ticker in open_positions:
            return

    if is_halted(ticker):
        print(f"  [TICK-CHECK] {ticker}: halted — skipping this signal.")
        return

    fundamentals = get_ticker_fundamentals(ticker)
    avg_volume = fundamentals.get("avg_volume")

    result = instant_tick_volume_check(ticker, news_time, avg_volume)

    print(f"  [TICK-CHECK] {ticker:<6} volume_since_news={result['volume_since_news']:.0f} "
          f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
          f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
          f"up_ratio={result['up_ratio']:.2f} (need >= {TICK_UP_VOLUME_RATIO_MIN}) "
          f"elapsed={result['elapsed_sec']:.1f}s pass={result['pass']} "
          f"data_quality={result['data_quality']} "
          f"(polls_ok={result['quote_polls_ok']} polls_failed={result['quote_polls_failed']})")

    if result["data_quality"] == "NO_DATA":
        print(f"  [WARN] {ticker}: volume_since_news={result['volume_since_news']:.0f} is NOT a "
              f"confirmed real reading — every quote poll during this window failed or returned "
              f"no total_volume field ({result['quote_polls_failed']} failed poll(s), 0 usable). "
              f"Treat this as a data outage, not a genuine zero-volume reaction. Check Schwab "
              f"connectivity / the totalVolume field name (see module docstring VERIFY note).")

    if result["pass"]:
        magnitude = compute_magnitude_score(ticker, headline, summary, llm, news_time, fundamentals)
        print(f"  [MAGNITUDE] {ticker:<6} composite={magnitude['composite']:.2f} -> "
              f"slots={magnitude['slots']}/{MAX_SLOTS_PER_TRADE} "
              f"(~${magnitude['slots']*CAPITAL_PER_SLOT_USD:.2f} requested)  "
              f"float_score={magnitude['float_score']:.2f}"
              f"(float={magnitude['float_shares']})  "
              f"price_action_score={magnitude['price_action_score']:.2f}"
              f"(pct={magnitude['price_action_pct']})  "
              f"llm_score={magnitude['llm_score']:.2f}"
              f"(confidence={llm.get('confidence')}, reasoning_words={magnitude['llm_reasoning_words']})  "
              f"news_category_score={magnitude['news_category_score']:.2f}"
              f"(category={magnitude['news_category']})")
        enter_position(ticker, reason="SENTIMENT_AND_TICK_VOLUME_CONFIRMED", headline=headline,
                        news_time=news_time, desired_slots=magnitude["slots"])
    else:
        print(f"  [TICK-CHECK] {ticker}: volume/up-down split not confirmed within "
              f"{TICK_WINDOW_SEC}s of news drop — no entry.")


# ══════════════════════════════════════════════════════════════════════════
# ROLLING-HIGH BREAKOUT PATH (parallel to the news pipeline — does not touch
# handle_news_event, the FDA fast path, or the stale-recap filter)
#
# Rationale (see conversation this was built from): the news pipeline only
# ever fires on an instant reaction to a fresh headline, and the ORIGINAL
# version of this path only ever caught the short-covering subset of
# explosive movers (short_float + days_to_cover). Most 100-200% low-float
# moves have nothing to do with short interest at all — float rotation,
# social/attention cascades, and halt-and-reopen dynamics are all mechanisms
# that fire with zero short-covering component. This version replaces the
# short-float/days-to-cover screen with a live RVOL/gap scanner (catches the
# "already showing unusual activity" mechanism directly, no short interest
# required) plus a dedicated halt-reopen sub-path (catches the "halt is
# itself the catalyst" mechanism, also independent of short interest). The
# "old bullish news, quiet for 1-3 weeks, THEN a delayed breakout" logic is
# KEPT — that part is well-supported by the post-earnings-announcement-drift
# literature (drift is strongest in small, illiquid, under-covered names) —
# but it now only gates the RVOL/gap sub-path, not the halt sub-path (a halt
# is its own catalyst; see the HALT SUB-PATH comment below for why).
#
# ENGINEERING NOTE (same lesson as the Finviz-scrape fix earlier): scanning
# the whole ~600-symbol watchlist for RVOL/gap/halt-status every few seconds
# with one HTTP call per ticker would hammer Schwab's API. market_scan_and_update()
# uses get_quotes_batch() to fetch MANY symbols per request instead — this is
# the only stage that touches the full watchlist, and even it runs on a
# 30-second cadence, not every 5 seconds:
#   TIER 1 (market_scan_and_update, every MARKET_SCAN_INTERVAL_SEC): ONE pass
#     over the whole watchlist in batched quote requests. Computes gap% and
#     RVOL for every ticker and flags GAP_PCT_MIN/RVOL_MIN/PREMARKET_MIN_VOLUME_SHARES
#     candidates into _gap_rvol_qualified. In the SAME pass, tracks each
#     ticker's halted/not-halted state and detects halt->reopen transitions
#     (see the HALT SUB-PATH below — those don't wait for Tier 2/3).
#   TIER 2 (refresh_bullish_news_tier, every NEWS_TIER_REFRESH_SEC): queries
#     Alpaca's HISTORICAL News REST endpoint (not just what the live
#     websocket happened to stream during this run) for Tier 1's current
#     survivors, and classifies any not-yet-scored article via the SAME qwen
#     catalyst check the news pipeline uses. Each historical article is only
#     ever scored by qwen ONCE (cached by article id).
#   TIER 3 (_check_rolling_high_breakout_and_enter, every
#     ROLLING_HIGH_POLL_INTERVAL_SEC): a break of this session's high,
#     confirmed by instant_tick_volume_check — the SAME function the news
#     path uses — only against whatever small set Tiers 1-2 leave standing.
#
# HALT SUB-PATH (independent of Tiers 1-3, no bullish-news requirement):
# a halt is treated as its OWN catalyst rather than requiring an unrelated
# old headline from weeks ago — forcing that condition on a halt wouldn't
# make sense (a stock can halt on a brand-new development that has nothing
# to do with 2-3-week-old drift). Instead, the reopen's own volume gets the
# SAME buy-side-dominant confirmation check the rest of this script uses
# before entering — a halt resume can reverse just as easily as continue
# (see the halt-reopen research this was built from), so the confirmation
# step is what separates "real continuation" from "reopened and immediately
# faded." A T1 news-pending halt in particular resumes with the market still
# not having read the news — instant_tick_volume_check's post-resume window
# is what stands in for that missing information here.
# ══════════════════════════════════════════════════════════════════════════

_gap_rvol_lock = threading.Lock()
_gap_rvol_qualified: "dict[str, dict]" = {}   # ticker -> {"gap_pct":, "rvol":} — Tier 1 survivors

_halt_state_lock = threading.Lock()
_halt_state: "dict[str, bool]" = {}   # ticker -> was-halted-as-of-last-scan (for edge detection)

_scored_historical_article_ids_lock = threading.Lock()
_scored_historical_article_ids: set = set()   # dedup so Tier 2 never re-scores the same historical article

_session_high_lock = threading.Lock()
_session_high: "dict[str, float]" = {}   # ticker -> highest `last` price seen this run


def _elapsed_seconds_since_premarket_open() -> float:
    """'How far into the trading day are we' — the window_sec used for the
    Tier-1/halt-reopen cumulative RVOL checks. Uses PREMARKET_OPEN_ET
    (4:00 AM ET) as the reference point since this bot is meant to be
    watched from premarket on — the SAME reference point
    time_of_day_matched_volume_baseline uses for those checks' window_start,
    so "elapsed since 4am" and "the TOD baseline's window" line up."""
    now_et = datetime.now(timezone.utc).astimezone(NY_TZ)
    session_start = now_et.replace(hour=PREMARKET_OPEN_ET.hour, minute=PREMARKET_OPEN_ET.minute,
                                    second=0, microsecond=0)
    if now_et < session_start:
        session_start -= timedelta(days=1)
    return max(60.0, (now_et - session_start).total_seconds())   # floor at 60s to avoid a huge early-morning RVOL spike from tiny denominators


def market_scan_and_update():
    """TIER 1 + halt-transition detection. Batched quotes across the WHOLE
    watchlist (see get_quotes_batch) — this is the only part of this path
    that touches every ticker, and it does so in a handful of requests, not
    hundreds. Updates _gap_rvol_qualified (Tier 1 survivors, feeds Tier 3 via
    rolling_high_breakout_loop) and fires the halt sub-path directly on any
    halt->reopen transition detected in this same pass."""
    with _watchlist_lock:
        wl = sorted(watchlist)

    quotes: dict = {}
    for i in range(0, len(wl), QUOTE_BATCH_CHUNK_SIZE):
        chunk = wl[i:i + QUOTE_BATCH_CHUNK_SIZE]
        try:
            quotes.update(SCHWAB.get_quotes_batch(chunk))
        except Exception as e:
            print(f"  [WARN] Batched quote fetch failed for chunk starting {chunk[0]}: {e}")

    elapsed = _elapsed_seconds_since_premarket_open()
    new_qualified: dict = {}

    for ticker, quote in quotes.items():
        last = quote.get("last")
        prev_close = quote.get("previous_close")
        total_volume = quote.get("total_volume")
        halted_now = _is_halted_status(quote.get("security_status"))

        # --- halt->reopen edge detection (independent of gap/RVOL below) ---
        with _halt_state_lock:
            was_halted = _halt_state.get(ticker, False)
            _halt_state[ticker] = halted_now
        if was_halted and not halted_now:
            try:
                _check_halt_reopen_and_enter(ticker, quote)
            except Exception as e:
                print(f"  [WARN] Halt-reopen check failed for {ticker}: {e}")

        # --- Tier 1: gap% + RVOL ---
        if not last or not prev_close or prev_close <= 0 or total_volume is None:
            continue
        gap_pct = (last - prev_close) / prev_close * 100.0
        avg_daily_volume = get_ticker_fundamentals(ticker).get("avg_volume")
        rvol_result = rvol_dual_check(ticker, total_volume, elapsed, PREMARKET_OPEN_ET,
                                       avg_daily_volume, primary_multiple=RVOL_MIN,
                                       confirm_multiple=RVOL_SESSION_CONFIRM_MIN)

        if (gap_pct >= GAP_PCT_MIN and rvol_result["pass"]
                and total_volume >= PREMARKET_MIN_VOLUME_SHARES):
            new_qualified[ticker] = {"gap_pct": gap_pct, "rvol": rvol_result["rvol_tod"],
                                      "rvol_session": rvol_result["rvol_session"],
                                      "used_fallback_baseline": rvol_result["used_fallback"]}

    with _gap_rvol_lock:
        newly_added = set(new_qualified) - set(_gap_rvol_qualified)
        _gap_rvol_qualified.clear()
        _gap_rvol_qualified.update(new_qualified)

    if newly_added:
        print(f"  [MARKET-SCAN] Tier 1 (gap>={GAP_PCT_MIN:.0f}%, RVOL>={RVOL_MIN:.1f}x, "
              f"vol>={PREMARKET_MIN_VOLUME_SHARES:,}): {len(new_qualified)} candidate(s) total, "
              f"newly added: {sorted(newly_added)}")


def market_scan_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            market_scan_and_update()
        except Exception as e:
            print(f"  [WARN] market_scan_loop error: {e}")
        stop_event.wait(MARKET_SCAN_INTERVAL_SEC)


def refresh_bullish_news_tier():
    """TIER 2. Only ever queries Alpaca's historical News REST endpoint for
    Tier 1's current (small) survivor set — this is what makes it affordable
    to look back OLD_BULLISH_NEWS_LOOKBACK_DAYS instead of just whatever the
    live websocket happened to stream since this process started. Every
    not-yet-scored article gets ONE qwen call, cached by article id in
    _scored_historical_article_ids forever after."""
    with _gap_rvol_lock:
        candidates = sorted(_gap_rvol_qualified)
    if not candidates:
        return

    client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    start = datetime.now(timezone.utc) - timedelta(days=OLD_BULLISH_NEWS_LOOKBACK_DAYS)

    for i in range(0, len(candidates), 100):
        chunk = candidates[i:i + 100]
        try:
            req = NewsRequest(symbols=",".join(chunk), start=start, limit=50,
                               include_content=True, exclude_contentless=False)
            newsset = client.get_news(req)
            articles = newsset.data.get("news", [])
        except Exception as e:
            print(f"  [WARN] Tier-2 historical news fetch failed for chunk starting {chunk[0]}: {e}")
            continue

        for article in articles:
            article_id = getattr(article, "id", None)
            if article_id is None:
                continue
            with _scored_historical_article_ids_lock:
                if article_id in _scored_historical_article_ids:
                    continue
                _scored_historical_article_ids.add(article_id)

            headline = getattr(article, "headline", "") or ""
            summary = getattr(article, "summary", "") or headline
            created_at = getattr(article, "created_at", None) or datetime.now(timezone.utc)
            article_symbols = getattr(article, "symbols", []) or []

            if _is_stale_recap_headline(headline):
                continue   # same dictionary pre-filter the live path uses

            llm = call_ollama_catalyst_check(headline, summary) if OLLAMA_ENABLED else None
            if llm is None or llm["catalyst"] != "positive" or llm["is_stale_or_secondhand"]:
                continue

            for ticker in article_symbols:
                if ticker in candidates:
                    _record_bullish_news(ticker, created_at)
                    print(f"  [MARKET-SCAN] Tier 2: {ticker} has a confirmed-bullish historical "
                          f"headline from {created_at.isoformat()}: {headline[:80]}")


def bullish_news_tier_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            refresh_bullish_news_tier()
        except Exception as e:
            print(f"  [WARN] bullish_news_tier_loop error: {e}")
        stop_event.wait(NEWS_TIER_REFRESH_SEC)


def _check_rolling_high_breakout_and_enter(ticker: str):
    """TIER 3. Called every ROLLING_HIGH_POLL_INTERVAL_SEC (see
    rolling_high_breakout_loop) for whatever small set Tiers 1-2 currently
    leave standing. Tracks this run's session high per ticker; fires the
    SAME tick-volume confirmation as the news path the moment a NEW high
    prints, so a fakeout single tick can't trigger it any more than it can
    on the news side. Sized via compute_breakout_magnitude_score using the
    Tier 1 gap/RVOL reading + this confirmation's own strength."""
    with _positions_lock:
        if ticker in open_positions:
            return
    if is_halted(ticker):
        return

    try:
        quote = SCHWAB.get_quote_full(ticker)
    except Exception as e:
        print(f"  [WARN] Rolling-high quote fetch failed for {ticker}: {e}")
        return
    last = quote.get("last") if quote else None
    if not last:
        return

    with _session_high_lock:
        prior_high = _session_high.get(ticker)
        if prior_high is None or last > prior_high:
            _session_high[ticker] = last

    if prior_high is None:
        return   # first time seeing this ticker this run — nothing to break yet
    if last <= prior_high:
        return   # no new high — nothing to check

    bullish_since = has_recent_bullish_news(ticker)
    if bullish_since is None:
        return   # Tier 2 hasn't (or no longer) confirms bullish news for this ticker

    with _gap_rvol_lock:
        gap_rvol = _gap_rvol_qualified.get(ticker)
    if gap_rvol is None:
        return   # dropped off Tier 1 since being flagged — don't act on stale data

    print(f"  [ROLLING-HIGH] {ticker}: new session high ${last:.2f} (prior ${prior_high:.2f}), "
          f"gap={gap_rvol['gap_pct']:.1f}% RVOL={gap_rvol['rvol']:.1f}x, "
          f"confirmed-bullish news from {bullish_since.isoformat()} — "
          f"running the same tick-volume confirmation as the news path.")

    fundamentals = get_ticker_fundamentals(ticker)
    result = instant_tick_volume_check(ticker, datetime.now(timezone.utc), fundamentals.get("avg_volume"))
    print(f"  [ROLLING-HIGH-CHECK] {ticker:<6} volume_since_break={result['volume_since_news']:.0f} "
          f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
          f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
          f"up_ratio={result['up_ratio']:.2f} (need >= {TICK_UP_VOLUME_RATIO_MIN}) "
          f"elapsed={result['elapsed_sec']:.1f}s pass={result['pass']} data_quality={result['data_quality']}")

    if not result["pass"]:
        print(f"  [ROLLING-HIGH-CHECK] {ticker}: level break not confirmed by sustained volume "
              f"within {TICK_WINDOW_SEC}s — no entry (treated as a likely fakeout, not a real breakout).")
        return

    score = compute_breakout_magnitude_score(result, gap_rvol["gap_pct"], gap_rvol["rvol"])
    print(f"  [MAGNITUDE-B] {ticker}: confirmation_score={score['confirmation_score']:.2f} "
          f"gap_scanner_score={score['gap_scanner_score']:.2f} -> composite={score['composite']:.2f} "
          f"-> {score['slots']} slot(s)")

    enter_position(ticker, reason="ROLLING_HIGH_BREAKOUT",
                    headline=f"Gap/RVOL candidate (gap={gap_rvol['gap_pct']:.1f}%, "
                             f"RVOL={gap_rvol['rvol']:.1f}x) + old bullish news "
                             f"({bullish_since.isoformat()}) + session-high break at ${last:.2f}",
                    news_time=None, desired_slots=score["slots"])


def rolling_high_breakout_loop(stop_event: threading.Event):
    """TIER 3's loop — the only part of the gap/RVOL sub-path that runs every
    ROLLING_HIGH_POLL_INTERVAL_SEC (5s), and only against whatever tiny set
    Tiers 1-2 (running on their own much slower loops above) currently leave
    standing."""
    while not stop_event.is_set():
        with _gap_rvol_lock:
            candidates = sorted(_gap_rvol_qualified)
        for ticker in candidates:
            if stop_event.is_set():
                break
            if opportunities_left() <= 0:
                break
            try:
                _check_rolling_high_breakout_and_enter(ticker)
            except Exception as e:
                print(f"  [WARN] rolling_high_breakout_loop error for {ticker}: {e}")
        stop_event.wait(ROLLING_HIGH_POLL_INTERVAL_SEC)


def _check_halt_reopen_and_enter(ticker: str, reopen_quote: dict):
    """HALT SUB-PATH — fired directly from market_scan_and_update() the
    instant a halt->reopen transition is detected (not gated by Tiers 1-2 at
    all — see the big comment above for why). Runs the SAME tick-volume
    confirmation as everything else in this script, starting from the
    reopen, before entering — a resume can reverse just as easily as
    continue, and this is what tells the two apart instead of just buying
    the reopen print itself."""
    with _positions_lock:
        if ticker in open_positions:
            return

    last = reopen_quote.get("last")
    prev_close = reopen_quote.get("previous_close")
    if not last:
        return
    gap_pct = ((last - prev_close) / prev_close * 100.0) if prev_close and prev_close > 0 else 0.0

    print(f"  [HALT-REOPEN] {ticker}: halt->reopen detected at ${last:.2f} "
          f"(gap vs previous close: {gap_pct:.1f}%) — running the same tick-volume "
          f"confirmation as the rest of this script before entering.")

    fundamentals = get_ticker_fundamentals(ticker)
    result = instant_tick_volume_check(ticker, datetime.now(timezone.utc), fundamentals.get("avg_volume"))
    avg_daily_volume = fundamentals.get("avg_volume")
    elapsed = _elapsed_seconds_since_premarket_open()
    rvol_result = rvol_dual_check(ticker, reopen_quote.get("total_volume") or 0.0, elapsed,
                                   PREMARKET_OPEN_ET, avg_daily_volume,
                                   primary_multiple=RVOL_MIN, confirm_multiple=RVOL_SESSION_CONFIRM_MIN)
    rvol = rvol_result["rvol_tod"]

    print(f"  [HALT-REOPEN-CHECK] {ticker:<6} volume_since_reopen={result['volume_since_news']:.0f} "
          f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
          f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
          f"up_ratio={result['up_ratio']:.2f} (need >= {TICK_UP_VOLUME_RATIO_MIN}) "
          f"elapsed={result['elapsed_sec']:.1f}s pass={result['pass']} data_quality={result['data_quality']}")

    if not result["pass"]:
        print(f"  [HALT-REOPEN-CHECK] {ticker}: reopen volume not confirmed bullish within "
              f"{TICK_WINDOW_SEC}s — treated as a likely fade/exhaustion reopen, no entry.")
        return

    score = compute_breakout_magnitude_score(result, gap_pct, rvol)
    print(f"  [MAGNITUDE-B] {ticker}: confirmation_score={score['confirmation_score']:.2f} "
          f"gap_scanner_score={score['gap_scanner_score']:.2f} -> composite={score['composite']:.2f} "
          f"-> {score['slots']} slot(s)")

    enter_position(ticker, reason="HALT_REOPEN_CONFIRMED",
                    headline=f"Halt->reopen at ${last:.2f} (gap={gap_pct:.1f}%), "
                             f"confirmed bullish by sustained buy-side volume post-resume",
                    news_time=None, desired_slots=score["slots"])


# ══════════════════════════════════════════════════════════════════════════
# MULTI-SYMBOL RELEVANCE FILTER
# Alpaca/Benzinga tags multi-symbol "roundup" articles (market-wrap pieces,
# sector digests, "Stocks That Moved Today" style stories) with EVERY ticker
# mentioned anywhere in the piece, not just the one the headline is actually
# about. Previously this pipeline fired the exact same headline/summary text
# at every tagged ticker with no relevance check — e.g. a Domino's-specific
# headline ("Nasdaq Surges 1%; Domino's Shares Gain After Q2 Results") was
# tagged with ADVB/BIYA/GVH/LBGJ/VCIG/ZYBT (all totally unrelated small-caps
# incidentally mentioned elsewhere in the same wire story) and each one got
# scored — and in that log, escalated to and PASSED by the LLM — using
# reasoning that was entirely about Domino's, not about that ticker at all.
# This filter runs BEFORE any sentiment/LLM scoring: if an article tags
# MULTIPLE symbols and the headline text is clearly anchored to one specific
# OTHER named company (a company-name-shaped phrase not matching this
# ticker), this ticker is dropped from processing for this article instead
# of being scored against unrelated text. Single-symbol articles are never
# affected — this only guards the fan-out case.
# ══════════════════════════════════════════════════════════════════════════

# Common corporate suffixes used to detect "this headline names a specific
# company" vs. generic market-wide phrasing ("Nasdaq Surges 1%", "Dow Falls
# Over 100 Points", "Stocks Moving Today").
_COMPANY_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z&'.\-]*(?:\s+[A-Z][A-Za-z&'.\-]*){0,4}\s+"
    r"(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|Holdings?|"
    r"Group|Technologies|Pharmaceuticals|Therapeutics|Biotechnology|Biotech|"
    r"Solutions|International|Global|Industries|Systems)\.?)\b"
)

# Generic market/index-wide openers that do NOT count as "naming a specific
# company" even though they're capitalized (Nasdaq, Dow, S&P, etc. moves are
# market-wide context, not a claim that every tagged ticker caused the move).
_MARKET_WIDE_OPENERS = (
    "nasdaq", "dow", "s&p", "s&p 500", "russell", "market", "markets",
    "stocks", "wall street", "futures",
)


def _extract_named_company_phrase(headline: str) -> Optional[str]:
    """Best-effort extraction of a specific company name the headline is
    ANCHORED to (e.g. 'Domino's Shares Gain' -> 'Domino's', 'Baiya
    International Group Shares Halted' -> 'Baiya International Group').
    Returns None if no clear single-company anchor is found (e.g. pure
    market-wide headlines, or headlines that are just a bare list of
    tickers with no distinguishing company name)."""
    if not headline:
        return None
    # "<Name> Shares <verb>" / "<Name> shares are trading <adj>" is the
    # dominant Benzinga headline pattern for single-company news. Searched
    # anywhere in the headline (not just at position 0) since market-wide
    # openers often precede the actual company name, e.g. "Nasdaq Surges 1%;
    # Domino's Shares Gain After Q2 Results" or "Dow Falls Over 100 Points;
    # IREN Shares Jump" — the anchor company is the one attached to "Shares".
    for m in re.finditer(r"([A-Z][A-Za-z0-9&'.\-]*(?:\s+[A-Z][A-Za-z0-9&'.\-]*){0,5})\s+[Ss]hares\b", headline):
        candidate = m.group(1).strip()
        # Strip a leading market-wide opener word/phrase if regex captured
        # extra leading tokens across a "; " or similar separator.
        candidate_clean = re.split(r"[;:,]\s*", candidate)[-1].strip()
        if candidate_clean.lower() not in _MARKET_WIDE_OPENERS and candidate_clean:
            return candidate_clean
    m = _COMPANY_SUFFIX_RE.search(headline)
    if m:
        return m.group(1).strip()
    return None


def _ticker_relevant_to_headline(ticker: str, headline: str, summary: str, num_symbols: int) -> bool:
    """Returns False only when we have a strong positive reason to believe
    this ticker is an incidental tag on a roundup article about a DIFFERENT
    named company — never used to filter single-symbol articles.

    IMPORTANT LIMITATION (documented, not hidden): a ticker symbol frequently
    does NOT literally contain the company name it stands for (Domino's
    Pizza -> DPZ, not "DOM"), and this pipeline has no market-data reference
    table mapping company names to tickers available locally. Reliably
    telling "this IS the anchor company's own ticker" apart from "this is an
    unrelated incidental tag" in the general case would require exactly that
    lookup. Rather than guess with a fragile name/ticker heuristic (which
    would risk the WORSE failure mode of wrongly dropping the one ticker the
    news is actually about), this filter is deliberately conservative and
    fails open on the single-ticker case: it only excludes a tagged ticker
    when the article fans out across MANY symbols (>= 3, comfortably beyond
    a normal same-story multi-company mention) AND that ticker's symbol is
    nowhere in the article text AND a confident single-company anchor was
    found. This reliably kills the observed bug pattern (a market-wide
    opener like "Nasdaq Surges 1%" followed by a named company, tagged with
    a long tail of unrelated small-caps) while minimizing the risk of
    dropping a genuinely relevant ticker whose name just doesn't textually
    resemble its own symbol. If you have (or can add) a ticker->company-name
    reference table, wire it in here for a strictly more accurate check."""
    if num_symbols < 3:
        return True   # low fan-out — not the roundup pattern this guards against

    anchor = _extract_named_company_phrase(headline)
    if not anchor:
        return True   # no clear single-company anchor found — don't guess, keep it

    ticker_lower = ticker.lower()
    combined_lower = f"{headline} {summary}".lower()

    # If the ticker symbol itself literally appears anywhere in the article
    # text, it's legitimately relevant regardless of which company the
    # headline opens with (e.g. a genuine multi-company roundup that names
    # each ticker individually in the summary).
    if re.search(rf"\b{re.escape(ticker_lower)}\b", combined_lower):
        return True

    anchor_lower = anchor.lower()
    if ticker_lower in anchor_lower or anchor_lower in ticker_lower:
        return True   # crude but safe: covers cases where they do overlap

    # Ticker's symbol is absent from the text, a confident different-company
    # anchor was found, and the fan-out is wide (>=3 symbols) — this is the
    # incidental-tag roundup pattern. Drop it.
    return False


def _strip_html(raw: str) -> str:
    """Alpaca/Benzinga's News.content field 'might contain HTML' per the
    model docstring — strip tags down to plain text for sentiment/LLM input.
    Falls back to a regex strip if lxml chokes on malformed markup."""
    if not raw:
        return ""
    try:
        text = html.fromstring(raw).text_content()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _process_news_article(news, tag: str = "NEWS-IN"):
    """
    Shared processing path for a single News object, regardless of whether
    it arrived via the live websocket (on_news) or the REST reconciliation
    safety net (news_reconciliation_loop). De-duplicated by article id so
    the same article is never handled twice no matter which path delivered
    it, or how many times.
    """
    global _last_news_at
    news_id = getattr(news, "id", None)
    symbols = list(getattr(news, "symbols", None) or [])
    headline = getattr(news, "headline", "") or ""
    summary = getattr(news, "summary", "") or ""
    # Alpaca/Benzinga headlines and summaries commonly arrive with HTML
    # entities still encoded (e.g. "Here&#39;s Why" instead of "Here's Why").
    # Decoded HERE, once, at the point of extraction — every downstream
    # consumer of these two variables (stale-recap pattern match, FDA
    # keyword match, fuzzy recycled-headline dedup, sentiment gate,
    # news-category magnitude scorer) depends on this running first, since
    # none of those do their own decoding. A previous FLYE headline
    # ("...After Hours: Here&#39;s Why") slipped past the stale-recap filter
    # for exactly this reason — the raw string never matched the plain
    # "here's why" pattern.
    headline = _html_unescape(headline)
    summary = _html_unescape(summary)
    # Alpaca/Benzinga's News object has a separate `content` field with the
    # full article body (the live websocket sends it on every article; the
    # historical REST endpoint needs include_content=True, set below on the
    # reconciliation loop's NewsRequest). Previously only `summary` — often
    # barely longer than the headline itself — ever reached tier-1/tier-2,
    # meaning both were screening on a teaser rather than the actual story.
    raw_content = getattr(news, "content", "") or ""
    content_text = _strip_html(raw_content)
    if content_text and len(content_text) > 4000:
        content_text = content_text[:4000]  # bound prompt size/latency, not the whole article is needed
    if content_text and content_text not in summary:
        summary = f"{summary}\n\n{content_text}".strip()
    # News.created_at is the article's own RFC-3339 publish timestamp from
    # Alpaca — use it as the anchor instead of local receive time. Falls back
    # to "now" only if the field is ever missing.
    article_time = getattr(news, "created_at", None) or datetime.now(timezone.utc)

    if _already_seen_news(news_id):
        return

    with _news_lock:
        _last_news_at = datetime.now(timezone.utc)

    # Every article that reaches this function gets logged, BEFORE any
    # filtering — previously the two early-returns below (no symbols /
    # no symbols on our watchlist) were completely silent, so there was no
    # way to tell "this article was correctly filtered out" apart from
    # "this article should have matched and something's wrong." Now every
    # article shows its raw symbol list and what (if anything) matched.
    print(f"  [{tag}] id={news_id} symbols={symbols}  {headline[:80]}")

    if not symbols:
        print(f"  [NEWS-DROP] no symbols tagged on this article — skipped.  {headline[:70]}")
        return

    with _watchlist_lock:
        wl = set(watchlist)
    relevant = [s for s in symbols if s in wl]
    if not relevant:
        print(f"  [NEWS-DROP] {symbols} not on current watchlist ({len(wl)} tickers) — skipped.")
        return

    for ticker in relevant:
        if not _ticker_relevant_to_headline(ticker, headline, summary, len(symbols)):
            print(f"  [NEWS-DROP] {ticker}: article tags {len(symbols)} symbols but headline text "
                  f"is anchored to a different named company — skipping {ticker} for this article "
                  f"instead of scoring it against unrelated text.  {headline[:70]}")
            continue
        handle_news_event(ticker, headline, summary, article_time)


async def on_news(news):
    try:
        await asyncio.to_thread(_process_news_article, news, "NEWS-IN")
    except Exception:
        # An unhandled exception here propagates into alpaca-py's internal
        # message-dispatch task, which can kill the websocket's receive loop
        # entirely — surfacing as a "no close frame received or sent"
        # disconnect, then a reconnect, then the SAME article-shaped bug
        # crashing it again on the next similar article. One bad article
        # must never be able to take down the whole connection silently.
        news_id = getattr(news, "id", "?")
        print(f"  [PROCESS-ERROR] on_news crashed processing article id={news_id} — "
              f"logged and skipped, connection kept alive. Traceback:")
        traceback.print_exc()


def news_reconciliation_loop(stop_event: threading.Event, lookback_minutes: int = 10,
                              interval_seconds: int = 180):
    """
    Safety net that runs independently of the live websocket's health. Every
    `interval_seconds`, it pulls recent news via Alpaca's historical News
    REST endpoint for the whole watchlist and feeds anything not already
    seen through the exact same processing path as the live stream
    (_process_news_article, deduped by article id).

    Why this is needed, concretely:
      - alpaca-py's live news websocket does NOT replay missed messages
        after a reconnect (dropped connection, server restart, brief network
        blip, etc.) — any article published during a disconnected window is
        otherwise gone for good. This loop re-covers the last
        `lookback_minutes` on every pass, so a brief outage self-heals on
        the next cycle instead of silently losing articles.
      - This does NOT widen news *coverage* — Alpaca's news (both the
        websocket and this REST endpoint) is 100% sourced from Benzinga
        (per Alpaca's own docs). An article that Benzinga never publishes or
        never tags with a given ticker will never show up here, live stream
        or reconciliation, regardless of this loop.
    """
    client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    while not stop_event.is_set():
        if stop_event.wait(interval_seconds):
            break
        with _watchlist_lock:
            wl = sorted(watchlist)
        if not wl:
            continue
        start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        # Chunk symbols: NewsRequest.symbols is a single comma-separated
        # string, and very long query strings are best avoided.
        for i in range(0, len(wl), 100):
            chunk = wl[i:i + 100]
            try:
                req = NewsRequest(symbols=",".join(chunk), start=start, limit=50,
                                   include_content=True, exclude_contentless=False)
                newsset = client.get_news(req)
                articles = newsset.data.get("news", [])
            except Exception as e:
                print(f"  [RECONCILE-ERROR] chunk starting {chunk[0]}: {e}")
                continue
            for article in articles:
                # _process_news_article no-ops via dedup if the live stream
                # already delivered this one; only genuinely missed articles
                # get logged/processed here. Wrapped defensively: an
                # unhandled exception on ANY single article here would
                # otherwise propagate past this for-loop and the enclosing
                # while-loop, silently killing this entire daemon thread for
                # the rest of the run (no more reconciliation, ever, with no
                # visible error unless someone happened to be watching
                # stderr) — confirmed as the likely cause of a live run where
                # this loop produced zero output for ~9 hours despite ~110
                # real articles existing for the watchlist that day.
                try:
                    _process_news_article(article, tag="RECONCILE")
                except Exception:
                    article_id = getattr(article, "id", "?")
                    print(f"  [PROCESS-ERROR] reconciliation loop crashed processing "
                          f"article id={article_id} — logged and skipped, loop kept alive. Traceback:")
                    traceback.print_exc()


def chunked_subscribe_news_loop(stream: "VerboseNewsDataStream", symbols: list, stop_event: threading.Event):
    """Subscribes to `symbols` in small batches (NEWS_SUBSCRIBE_CHUNK_SIZE at a
    time), waiting for the stream to actually be connected/running before each
    batch is sent.

    Why this exists: alpaca-py's DataStream._subscribe() only sends a
    subscribe message over the wire `if self._running`. Calling
    subscribe_news() with all 608 symbols BEFORE stream.run() just queues
    them locally — the first subscribe message IS still sent automatically
    once the connection opens (via _run_forever), but with every symbol
    crammed into one message. That one-shot mass subscribe is a known
    failure mode on Alpaca's streaming infra (msgpack payload too large /
    connection reset mid-send), which is almost certainly why server_acked
    stayed at 0/608 with no logged [SUB-ERROR] either — the message likely
    never completed a round trip at all.

    Run this from its own daemon thread, started right after stream.run()
    is kicked off on the main thread."""
    remaining = list(symbols)
    total = len(remaining)
    sent = 0

    # Wait for the stream to actually be connected before sending anything —
    # sending while _running is False would just silently re-queue locally.
    while not stop_event.is_set() and not getattr(stream, "_running", False):
        time.sleep(0.25)

    while remaining and not stop_event.is_set():
        batch = remaining[:NEWS_SUBSCRIBE_CHUNK_SIZE]
        remaining = remaining[NEWS_SUBSCRIBE_CHUNK_SIZE:]
        try:
            stream.subscribe_news(on_news, *batch)
        except Exception as e:
            print(f"  [SUB-ERROR] Batch subscribe crashed for {len(batch)} symbol(s) "
                  f"(e.g. {batch[:3]}...): {e} — will not retry this batch automatically.")
            continue
        sent += len(batch)
        print(f"  [SUB-SEND] Sent subscribe batch of {len(batch)} symbol(s) "
              f"({sent}/{total} sent so far) — waiting briefly for ack before next batch.")
        # Small pacing gap between batches so we're not hammering the
        # connection with back-to-back subscribe messages, and so the ack
        # log lines interleave readably with [SUB-SEND] lines.
        time.sleep(1.0)

    if not stop_event.is_set():
        print(f"  [SUB-SEND] All {total} symbol(s) sent in batches of "
              f"{NEWS_SUBSCRIBE_CHUNK_SIZE}. Check heartbeat server_acked for confirmation.")


class VerboseNewsDataStream(NewsDataStream):
    """
    Same as NewsDataStream, except it also records which symbols Alpaca's
    server actually acknowledged for the news channel, and prints that
    (plus any subscription errors) straight to the console.

    Why this matters: alpaca-py's base DataStream._dispatch() already
    receives a `"subscription"` ack frame from the server listing the
    symbols it has you subscribed to (and an `"error"` frame if a
    subscribe request is rejected) — but it only logs them via
    `logging.getLogger("alpaca.data.live.websocket")` at INFO/ERROR level.
    Since this script never calls logging.basicConfig(), that logger has
    no handler and those messages go nowhere. That means the acking (or
    rejection) of the 621-symbol subscribe request was silently invisible
    in the heartbeat output you were looking at — watchlist=621 only ever
    reflected the local Finviz-scraped set, never a server confirmation.
    """

    async def _dispatch(self, msg):
        msg_type = msg.get("T")
        if msg_type == "subscription":
            acked = list(msg.get("news", []))
            with _acked_lock:
                acked_news_symbols.update(acked)
                total_acked = len(acked_news_symbols)
            print(f"  [SUB-ACK] Alpaca confirmed {len(acked)} news symbol(s) in this ack "
                  f"(cumulative server-confirmed total: {total_acked})")
        elif msg_type == "error":
            print(f"  [SUB-ERROR] Alpaca rejected/errored on subscription: "
                  f"{msg.get('msg')} (code {msg.get('code')})")
        await super()._dispatch(msg)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    global SCHWAB, SCHWAB_TRADER, _stop_event_ref, _stream_ref

    parser = argparse.ArgumentParser(description="Live catalyst entry-signal pipeline")
    parser.add_argument("--schwab-login", action="store_true",
                         help="Run the one-time interactive Schwab OAuth bootstrap and exit.")
    parser.add_argument("--app", choices=["marketdata", "trader"], default="marketdata",
                         help="Which app's credentials --schwab-login should bootstrap "
                              "(default: marketdata).")
    parser.add_argument("--schwab-account-hash", action="store_true",
                         help="Fetch and print the account hash needed for order placement, then exit. "
                              "Always uses the trader app's credentials.")
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument("--pathA", action="store_true",
                             help="Run ONLY the news-driven pipeline (websocket + reconciliation). "
                                  "The rolling-high breakout path (Tiers 1-4) is not started. "
                                  "Default (neither flag given): both paths run in parallel.")
    path_group.add_argument("--pathB", action="store_true",
                             help="Run ONLY the rolling-high breakout path (Tiers 1-4). The news "
                                  "websocket is not connected and no news-driven entries can fire. "
                                  "Default (neither flag given): both paths run in parallel.")
    args = parser.parse_args()

    if args.schwab_login:
        schwab_oauth_bootstrap(args.app)
        return
    if args.schwab_account_hash:
        schwab_account_hash_bootstrap()
        return

    required = [
        ("ALPACA_API_KEY", ALPACA_API_KEY), ("ALPACA_API_SECRET", ALPACA_API_SECRET),
        ("SCHWAB_MARKETDATA_CLIENT_ID", SCHWAB_MARKETDATA_CLIENT_ID),
        ("SCHWAB_MARKETDATA_CLIENT_SECRET", SCHWAB_MARKETDATA_CLIENT_SECRET),
        ("SCHWAB_MARKETDATA_REFRESH_TOKEN", SCHWAB_MARKETDATA_REFRESH_TOKEN),
        ("SCHWAB_TRADER_CLIENT_ID", SCHWAB_TRADER_CLIENT_ID),
        ("SCHWAB_TRADER_CLIENT_SECRET", SCHWAB_TRADER_CLIENT_SECRET),
        ("SCHWAB_TRADER_REFRESH_TOKEN", SCHWAB_TRADER_REFRESH_TOKEN),
    ]
    required.append(("SCHWAB_ACCOUNT_HASH", SCHWAB_ACCOUNT_HASH))   # orders are always live now
    missing = [n for n, v in required if not v]
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("        Run with --schwab-login --app marketdata / --app trader if you don't have refresh tokens yet.")
        print("        Run with --schwab-account-hash first if you don't have SCHWAB_ACCOUNT_HASH yet.")
        sys.exit(1)

    SCHWAB = SchwabClient(SCHWAB_MARKETDATA_CLIENT_ID, SCHWAB_MARKETDATA_CLIENT_SECRET,
                          SCHWAB_MARKETDATA_REFRESH_TOKEN)
    SCHWAB_TRADER = SchwabClient(SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET,
                                SCHWAB_TRADER_REFRESH_TOKEN)

    print("[INIT] Detecting account buying power...")
    detected_capital = detect_capital_pool(SCHWAB_TRADER, SCHWAB_ACCOUNT_HASH)
    print(f"[INIT] Detected ${detected_capital:.2f} buying power -> "
          f"{TOTAL_BUY_OPPORTUNITIES} slots of ${CAPITAL_PER_SLOT_USD:.2f} each.")

    print("=" * 80)
    print("  LIVE CATALYST ENTRY PIPELINE")
    print(f"  *** LIVE ORDERS — ${ACCOUNT_BUYING_POWER_USD:.2f} detected buying power split into "
          f"{TOTAL_BUY_OPPORTUNITIES} capital slots (${CAPITAL_PER_SLOT_USD:.2f}/slot). A normal "
          f"entry claims {MIN_SLOTS_PER_TRADE}-{MAX_SLOTS_PER_TRADE} slots (magnitude-scaled), "
          f"converted to shares at entry price (${PRICE_CEILING_USD:.2f}/share ceiling). "
          f"FDA-approval fast path claims ALL slots (the entire remaining pool) at that instant. ***")
    print(f"  Watchlist: float<{FLOAT_CEILING_SHARES/1e6:.0f}M  MICROCAP(SUB 300M) "
          f"avg-vol>{MIN_AVG_VOLUME_SHARES/1e3:.0f}K")
    print(f"  Rolling-high breakout path (parallel to news, 3-tier funnel + halt sub-path): "
          f"Tier1 gap>={GAP_PCT_MIN:.0f}% AND RVOL>={RVOL_MIN:.1f}x AND vol>={PREMARKET_MIN_VOLUME_SHARES:,} "
          f"(batched Schwab quotes across the whole watchlist, refreshed every "
          f"{MARKET_SCAN_INTERVAL_SEC}s) -> Tier2 confirmed-bullish catalyst within "
          f"{OLD_BULLISH_NEWS_LOOKBACK_DAYS} days (Alpaca historical news + qwen, refreshed every "
          f"{NEWS_TIER_REFRESH_SEC//60}min) -> Tier3 premarket/session-high break confirmed by the "
          f"same tick-volume gate as news (>= {TICK_VOLUME_MULTIPLE}x baseline, "
          f">= {TICK_UP_VOLUME_RATIO_MIN*100:.0f}% up-ratio, within {TICK_WINDOW_SEC}s, polled every "
          f"{ROLLING_HIGH_POLL_INTERVAL_SEC}s).  HALT SUB-PATH (independent, no news requirement): "
          f"any halt->reopen detected in the Tier-1 scan is confirmed by the same tick-volume gate "
          f"before entering.")
    print(f"  FDA fast-path candidate keywords (must ALSO be confirmed by {OLLAMA_MODEL} as an "
          f"explicit granted approval before bypassing every other gate): {FDA_APPROVAL_KEYWORDS}")
    print(f"  Sentiment gate tier 2 (PRIMARY, local LLM): {OLLAMA_MODEL} via Ollama, "
          f"{'ENABLED' if OLLAMA_ENABLED else 'DISABLED'} — runs FIRST on every headline; "
          f"tier-1 alone can no longer approve entry")
    print(f"  Sentiment gate tier 1 (SECONDARY confirmation, VADER+LM composite): "
          f"P(negative) < {NEGATIVE_PROB_MAX} AND P(positive) > {POSITIVE_PROB_MIN} — "
          f"only consulted after tier-2 says positive; skipped entirely (not required) "
          f"when tier-2 confidence=high")
    print(f"  Instant tick-volume gate (no wait, no bars): volume-since-news >= "
          f"{TICK_VOLUME_MULTIPLE}x a TIME-OF-DAY-MATCHED baseline (avg of the same "
          f"clock-time window over the last {RVOL_TOD_LOOKBACK_DAYS} sessions) AND >= "
          f"{TICK_VOLUME_SESSION_CONFIRM_MULTIPLE}x the flat full-session-average baseline "
          f"(confirmation) AND up-volume ratio >= {TICK_UP_VOLUME_RATIO_MIN*100:.0f}% "
          f"within {TICK_WINDOW_SEC}s of the drop (polled every {TICK_POLL_INTERVAL_SEC}s)")
    print(f"  Exits: SET entry-volatility stop (measured once at entry, "
          f"{ENTRY_STOP_ATR_MULTIPLE}x ATR{ENTRY_STOP_ATR_PERIOD}, capped at "
          f"{MAX_LOSS_PCT*100:.0f}%, floored at {MIN_STOP_PCT*100:.0f}% — never trails)  |  "
          f"secondary (armed after {SECONDARY_EXIT_PST_HOUR}am PT or a "
          f"{SECONDARY_EXIT_MOVE_TRIGGER_PCT:.0f}% move from entry): significant bearish CVD "
          f"(net ratio <= {CVD_BEARISH_THRESHOLD}) or bearish OBV/price divergence")
    print(f"  Magnitude scoring (SIZING ONLY, never gates entry): float "
          f"(w={MAGNITUDE_WEIGHT_FLOAT}) + pre-news price action "
          f"(w={MAGNITUDE_WEIGHT_PRICE_ACTION}) + LLM confidence/reasoning "
          f"(w={MAGNITUDE_WEIGHT_LLM}) + news category (w={MAGNITUDE_WEIGHT_NEWS_CATEGORY}) "
          f"-> {MIN_SLOTS_PER_TRADE}-{MAX_SLOTS_PER_TRADE} slots/entry "
          f"(${CAPITAL_PER_SLOT_USD:.2f} each)")
    print("=" * 80)

    print("[INIT] Building initial watchlist from Finviz...")
    initial = get_all_tickers()
    with _watchlist_lock:
        watchlist.update(initial)
    print(f"[INIT] Watchlist size: {len(watchlist)}")

    get_vader()
    get_lm()

    if OLLAMA_ENABLED:
        print(f"[INIT] Checking Ollama tier-2 escalation ({OLLAMA_MODEL} at {OLLAMA_URL})...")
        test = call_ollama_catalyst_check(
            "Company Reports Record Revenue, Beats Estimates, Raises Full-Year Guidance",
            "The company beat analyst estimates and raised its full-year outlook.")
        if test is None:
            print(f"  [WARN] Ollama did not respond at startup. Tier-2 escalation will fail-closed "
                  f"(no entry) on every headline it would otherwise have rescued until this is fixed. "
                  f"Check: is `ollama serve` running? Is `{OLLAMA_MODEL}` pulled (`ollama pull {OLLAMA_MODEL}`)?")
        else:
            print(f"  [OK] Ollama responded: catalyst={test['catalyst']} confidence={test['confidence']}")

    stop_event = threading.Event()
    _stop_event_ref = stop_event

    run_path_a = not args.pathB   # news pipeline (websocket + reconciliation)
    run_path_b = not args.pathA   # rolling-high breakout funnel (Tiers 1-4)
    print(f"[RUN-MODE] Path A (news): {'ON' if run_path_a else 'OFF'}   "
          f"Path B (rolling-high breakout): {'ON' if run_path_b else 'OFF'}")

    sorted_watchlist = sorted(watchlist)

    threading.Thread(target=monitor_positions_loop, args=(stop_event,), daemon=True).start()
    threading.Thread(target=heartbeat_loop, args=(stop_event,), daemon=True).start()
    if run_path_a:
        threading.Thread(target=news_reconciliation_loop, args=(stop_event,), daemon=True).start()
    if run_path_b:
        threading.Thread(target=market_scan_loop, args=(stop_event,), daemon=True).start()
        threading.Thread(target=bullish_news_tier_loop, args=(stop_event,), daemon=True).start()
        threading.Thread(target=rolling_high_breakout_loop, args=(stop_event,), daemon=True).start()

    if not run_path_a:
        # --pathB: no websocket to connect, so just block here until Ctrl+C
        # while the Path B threads above (and monitor/heartbeat) do the work.
        print("[RUN] Path A disabled (--pathB) — no news websocket connected. "
              "Rolling-high breakout path running standalone. (Ctrl+C to stop)\n")
        try:
            while not stop_event.is_set():
                stop_event.wait(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            print("\n[SHUTDOWN] Stopped.")
        return

    print("[RUN] Listening for live news on the watchlist... (Ctrl+C to stop)\n")

    # alpaca-py's _run_forever() won't open the websocket connection at all
    # until at least one symbol is registered in _handlers (it spin-waits on
    # `any(v for k, v in self._handlers.items() ...)` before calling
    # _connect()). So we still need ONE subscribe_news() call before .run()
    # — but only a small seed batch, not the full watchlist. The rest is
    # sent afterward in small batches by chunked_subscribe_news_loop(), once
    # the connection is confirmed up. This avoids the original bug (one
    # 608-symbol subscribe crammed into the pre-run queue, which is almost
    # certainly why server_acked stayed at 0 with no logged error either).


    # ── WEBSOCKET RECONNECT LOOP (bugfix) ───────────────────────────────────
    # The news websocket can silently time out / drop mid-run (idle-connection
    # server-side timeout, brief network blip, etc.). Previously any
    # exception out of stream.run() other than KeyboardInterrupt fell
    # straight through to the `finally` below and ended the ENTIRE run with
    # no attempt to recover — a single disconnect could end the whole live
    # session. This now rebuilds a fresh stream (alpaca-py's stream objects
    # are not safely re-runnable once their receive loop has exited),
    # reseeds + resubscribes the same watchlist, and retries after a short
    # backoff. news_reconciliation_loop (already running above) independently
    # covers anything published during the gap between disconnect and
    # reconnect, so no articles are lost even while this loop is recovering.
    reconnect_attempt = 0
    try:
        while not stop_event.is_set():
            stream = VerboseNewsDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)
            _stream_ref = stream

            seed_batch = sorted_watchlist[:NEWS_SUBSCRIBE_CHUNK_SIZE]
            rest_batch = sorted_watchlist[NEWS_SUBSCRIBE_CHUNK_SIZE:]
            if seed_batch:
                stream.subscribe_news(on_news, *seed_batch)
                print(f"  [SUB-SEND] Seeded first subscribe batch of {len(seed_batch)} symbol(s) "
                      f"pre-run so the connection has something to open with.")
            if rest_batch:
                threading.Thread(
                    target=chunked_subscribe_news_loop,
                    args=(stream, rest_batch, stop_event),
                    daemon=True,
                ).start()

            reconnect_attempt += 1
            if reconnect_attempt > 1:
                print(f"  [RECONNECT] Attempt #{reconnect_attempt} — reopening the news websocket "
                      f"and resubscribing {len(sorted_watchlist)} symbol(s).")

            try:
                stream.run()
                # stream.run() returned on its own (not via exception) —
                # nothing more to do, exit the reconnect loop cleanly.
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if stop_event.is_set():
                    break
                print(f"  [WARN] News websocket dropped/timed out: {e!r} — "
                      f"reconnecting in {WS_RECONNECT_BACKOFF_SEC}s.")
                stop_event.wait(WS_RECONNECT_BACKOFF_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\n[SHUTDOWN] Stopped.")


if __name__ == "__main__":
    main()