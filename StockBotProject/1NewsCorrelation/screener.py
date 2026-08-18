"""
rvol_backtest_screener.py
--------------------------
BACKWARDS-FACING version of the RVOL screener: instead of polling live,
this scans the PAST 60 DAYS of 5-minute bars (via yfinance) for a
watchlist pulled fresh from Finviz (float < 10M, avg volume > 50K --
same reactivity-focused screen fullpipeline.py uses, just a tighter
float cutoff per request), finds every point where RVOL has been
SUSTAINED >= VOL_SPIKE_RVOL_THRESHOLD for RVOL_SUSTAIN_MINUTES straight
(default 15m = 3 consecutive 5m bars, not just a single spiking bar)
while price was already in a short-term uptrend (same pattern as the
live screener), and then measures what happened AFTER each confirmed
signal: the % gain/loss at +30m, +1h, +2h, +4h, and +1 day forward from
the confirmation bar.

Output: one row per signal (symbol + exact crossing timestamp), with
forward-return columns, printed to console and saved to CSV.

Requires: pip install yfinance pandas requests finviz

IMPORTANT yfinance limits (as of 2026):
  - 5m bars: available for roughly the past 60 days -- this script's
    default lookback (60d) sits right at that ceiling. If yfinance trims
    it silently for your account, drop LOOKBACK_PERIOD to "59d".
  - 1m bars: only available for the past ~7 days (not used here).
  - Yahoo may throttle/rate-limit large batches of tickers -- this script
    downloads in small batches with a short delay between them.
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Config ───────────────────────────────────────────────────────────────
LOOKBACK_PERIOD = "60d"         # yfinance period string (5m bars: ~60d is the max window)
BAR_INTERVAL = "5m"             # 5-minute bars

BASELINE_BARS = 20              # rolling window for RVOL baseline (same as live screener)
VOL_SPIKE_RVOL_THRESHOLD = 5.0  # signal fires when RVOL crosses >= this
RVOL_SUSTAIN_MINUTES = 15       # RVOL must stay >= threshold for this long before firing
RVOL_SUSTAIN_BARS = RVOL_SUSTAIN_MINUTES // 5   # 15m / 5m bars = 3 consecutive bars
UPTREND_LOOKBACK_BARS = 4       # "already trending up" = positive return over last N bars
UPTREND_MIN_RETURN_PCT = 2.0    # minimum cumulative return over that window to count as uptrend

# Forward outcome windows, expressed in number of 5-minute bars.
# 30m=6 bars, 1h=12 bars, 2h=24 bars, 4h=48 bars.
# The daily outcome is handled separately (see compute_outcomes) because it
# needs to look at the *next trading day's* close, not just +78 bars, since
# bars only exist during market hours and a raw bar-count offset would
# silently roll forward across the closed overnight/weekend gap.
FORWARD_WINDOWS_BARS = {
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
}

BATCH_SIZE = 20                 # tickers per yfinance batch download
BATCH_DELAY_SEC = 1.5           # pause between batches to avoid throttling

OUTPUT_CSV = "rvol_backtest_signals.csv"

# ── Finviz watchlist config (ported from fullpipeline.py's get_all_tickers) ──
FINVIZ_PAGE_DELAY_SEC = 1.5
FINVIZ_BATCH_SIZE = 3
FINVIZ_COOLDOWN_SEC = 5.0
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_TICKER_LINK_RE = re.compile(r'[?&]t=([A-Za-z][A-Za-z\.\-]{0,5})(?=[&"\'])')


def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _finviz_url() -> str:
    # VERIFY these filter codes against Finviz's own screener UI before relying
    # on them -- build the screen there once and confirm the URL matches.
    float_code = "sh_float_u10"     # Float: Under 10M
    avgvol_code = "sh_avgvol_o50"   # Average Volume: Over 50K
    return (f"https://finviz.com/screener.ashx?v=111&"
            f"f={float_code},{avgvol_code}&ft=4&o=ticker")


def _fetch_finviz_page(session: requests.Session, page_url: str):
    """Fetch one screener page through a persistent session with real
    browser-like headers. Bare single-shot requests (fresh connection, no
    cookies, User-Agent only) are what finviz's anti-scraping layer is
    tuned to flag after the first request or two -- it doesn't necessarily
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


def _extract_tickers_from_html(html_text: str) -> list:
    """Pull ticker symbols out of '...t=TICKER...' query-string fragments
    anywhere in the raw page HTML, in document order, de-duplicated."""
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
    of guessing at the link format again."""
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
    print(f"  [DEBUG] full HTML saved to {debug_path.resolve()} -- open it and search for one of "
          f"your watchlist tickers' text to see what markup actually wraps it, then send me that snippet.")
    return debug_path.as_posix()


def get_all_tickers() -> list:
    """Pulls the current Finviz screen (float < 10M, avg vol > 50K) and
    returns every matching ticker, paginating through all result pages.
    Ported directly from fullpipeline.py's get_all_tickers(), same
    pacing/retry/debug behavior -- only _finviz_url()'s filter codes changed."""
    try:
        from finviz.screener import Screener
        from finviz.helper_functions import scraper_functions as scrape
    except ImportError:
        print("[ERROR] finviz not installed: pip install finviz")
        sys.exit(1)

    from lxml import html  # needed only for the total_rows<=0 fallback path below

    url = _finviz_url()
    # Use Screener.init_from_url only to discover total_rows/page_content/url --
    # NOT its .data (see _extract_tickers_from_html docstring for why).
    probe = Screener.init_from_url(url, rows=20)
    total_rows = getattr(probe, "_total_rows", 0)
    if total_rows <= 0:
        return _extract_tickers_from_html(str(html.tostring(probe._page_content)))

    page_urls = scrape.get_page_urls(probe._page_content, total_rows, probe._url)
    print(f"  [FINVIZ] {total_rows} total matches across {len(page_urls)} pages -- fetching all in paced batches.")

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
            print(f"  [WARN] Finviz page {i} failed ({e}) -- "
                  f"cooling down {FINVIZ_COOLDOWN_SEC}s and retrying once.")
            time.sleep(FINVIZ_COOLDOWN_SEC)
            try:
                page_tickers = _try_fetch()
            except Exception as e2:
                print(f"  [WARN] Page {i} failed again ({e2}) -- "
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
        print(f"  [FINVIZ] page {i}/{len(page_urls)}: {len(page_tickers)} tickers on page, "
              f"{new_on_page} new (running total {len(tickers)}/{total_rows})")

        if i % FINVIZ_BATCH_SIZE == 0 and i < len(page_urls):
            time.sleep(FINVIZ_COOLDOWN_SEC)

    if len(tickers) < total_rows:
        print(f"  [FINVIZ] WARNING: collected {len(tickers)}/{total_rows} -- "
              f"some pages came back short or failed. See [WARN] lines above.")

    return tickers


def download_bars(symbols: list) -> dict:
    """Download 5m bars for the past LOOKBACK_PERIOD for each symbol, batched.
    Returns {symbol: DataFrame} with columns Open/High/Low/Close/Volume,
    a tz-aware DatetimeIndex, one entry per symbol that returned data.
    """
    data: dict = {}

    for batch in chunk_list(symbols, BATCH_SIZE):
        try:
            raw = yf.download(
                tickers=batch,
                period=LOOKBACK_PERIOD,
                interval=BAR_INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"  [WARN] batch download failed for {batch}: {e}")
            time.sleep(BATCH_DELAY_SEC)
            continue

        if raw is None or raw.empty:
            print(f"  [WARN] no data returned for batch {batch}")
            time.sleep(BATCH_DELAY_SEC)
            continue

        for sym in batch:
            try:
                # yf.download returns a flat frame when only 1 ticker is
                # requested, and a MultiIndex-columned frame (ticker on the
                # first level) when multiple tickers are requested. Handle
                # both shapes explicitly rather than assuming one.
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].copy()
                else:
                    df = raw.copy()

                df = df.dropna(subset=["Close", "Volume"])
                if df.empty:
                    continue
                data[sym] = df
            except Exception as e:
                print(f"  [WARN] couldn't extract {sym} from batch frame: {e}")

        print(f"  Downloaded batch of {len(batch)} tickers "
              f"({sum(1 for s in batch if s in data)} with usable data)")
        time.sleep(BATCH_DELAY_SEC)

    return data


def compute_rvol_series(volume: pd.Series) -> pd.Series:
    """RVOL at each bar = that bar's volume / average volume of the
    BASELINE_BARS bars immediately preceding it (excludes the bar itself,
    matching the live screener's rvol() definition)."""
    baseline_avg = volume.shift(1).rolling(window=BASELINE_BARS - 1, min_periods=BASELINE_BARS - 1).mean()
    return volume / baseline_avg


def compute_uptrend_flag(close: pd.Series) -> pd.Series:
    """True at bar i if cumulative return from bar (i - UPTREND_LOOKBACK_BARS)
    to bar i is >= UPTREND_MIN_RETURN_PCT."""
    past = close.shift(UPTREND_LOOKBACK_BARS)
    pct_return = 100 * (close - past) / past
    return pct_return >= UPTREND_MIN_RETURN_PCT


def find_signals(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of every bar in df where RVOL has just finished
    being sustained >= VOL_SPIKE_RVOL_THRESHOLD for RVOL_SUSTAIN_BARS
    CONSECUTIVE bars (RVOL_SUSTAIN_MINUTES straight, default 15m = 3 bars
    at 5m each) while the uptrend condition is true on the confirmation
    bar -- i.e. this filters out one-bar volume spikes that immediately
    fade, only firing once the sustained window is confirmed. The signal
    timestamp/price used for everything downstream (RVOL value, entry
    price, forward outcomes) is the LAST bar of that sustained window --
    the bar where the condition is first fully confirmed -- not the first
    bar where RVOL initially crossed above threshold.

    Fires once per sustained run: if RVOL stays elevated for 10 bars
    straight, that's one signal (at bar 3 of the run), not eight
    overlapping ones."""
    rvol = compute_rvol_series(df["Volume"])
    uptrend = compute_uptrend_flag(df["Close"])

    above = rvol >= VOL_SPIKE_RVOL_THRESHOLD
    # True at bar i only if the current bar AND the (RVOL_SUSTAIN_BARS - 1)
    # bars immediately before it are ALL above threshold.
    sustained = above.rolling(window=RVOL_SUSTAIN_BARS, min_periods=RVOL_SUSTAIN_BARS).apply(
        lambda w: bool(w.all()), raw=True
    ).fillna(0).astype(bool)

    # Fire only on the FIRST bar a sustained run is confirmed, not every
    # bar afterward that the run continues (same "crossing" idea as
    # before, just applied to the sustained flag instead of the raw
    # above-threshold flag).
    newly_confirmed = sustained & ~sustained.shift(1, fill_value=False)

    signal_mask = newly_confirmed & uptrend
    if not signal_mask.any():
        return pd.DataFrame()

    out = df.loc[signal_mask, ["Close", "Volume"]].copy()
    out["Symbol"] = symbol
    out["RVOL"] = rvol.loc[signal_mask].round(2)
    out["SignalTimestamp"] = out.index
    out["SignalBarIndex"] = [df.index.get_loc(ts) for ts in out.index]
    return out.reset_index(drop=True)


def pct_change(entry_price: float, exit_price: float):
    if entry_price is None or exit_price is None or entry_price == 0 or pd.isna(entry_price) or pd.isna(exit_price):
        return None
    return round(100 * (exit_price - entry_price) / entry_price, 2)


def compute_outcomes(df: pd.DataFrame, signal_row: pd.Series) -> dict:
    """Given the full bar DataFrame for a symbol and one signal row, compute
    forward % gain/loss for each window in FORWARD_WINDOWS_BARS plus the
    next full trading day's close-to-close return."""
    entry_idx = int(signal_row["SignalBarIndex"])
    entry_price = df["Close"].iloc[entry_idx]

    outcomes = {}
    n_bars = len(df)

    for label, bars_ahead in FORWARD_WINDOWS_BARS.items():
        target_idx = entry_idx + bars_ahead
        if target_idx < n_bars:
            exit_price = df["Close"].iloc[target_idx]
            outcomes[f"{label}_pct"] = pct_change(entry_price, exit_price)
        else:
            outcomes[f"{label}_pct"] = None  # not enough future bars yet (signal too recent)

    # Daily outcome: find the last available bar on the NEXT calendar day
    # that has bars after the signal's day, and compare its close to the
    # signal bar's close. This avoids the overnight/weekend gap problem of
    # a fixed bar-count offset (market is closed ~16.5h/day + weekends, so
    # "+78 bars" would not reliably land on "next day's close").
    signal_date = signal_row["SignalTimestamp"].date()
    future_dates = sorted({ts.date() for ts in df.index if ts.date() > signal_date})
    if future_dates:
        next_day = future_dates[0]
        next_day_bars = df.loc[[ts for ts in df.index if ts.date() == next_day]]
        if not next_day_bars.empty:
            daily_exit_price = next_day_bars["Close"].iloc[-1]
            outcomes["1d_pct"] = pct_change(entry_price, daily_exit_price)
        else:
            outcomes["1d_pct"] = None
    else:
        outcomes["1d_pct"] = None  # signal happened too recently, next day hasn't traded yet

    return outcomes


def run_screen(symbols: list) -> pd.DataFrame:
    print(f"Downloading {BAR_INTERVAL} bars for the past {LOOKBACK_PERIOD} "
          f"across {len(symbols)} symbols...")
    bar_data = download_bars(symbols)
    print(f"Got usable data for {len(bar_data)}/{len(symbols)} symbols.\n")

    all_signals = []
    for symbol, df in bar_data.items():
        df = df.sort_index()
        sig_df = find_signals(symbol, df)
        if sig_df.empty:
            continue

        for _, row in sig_df.iterrows():
            outcomes = compute_outcomes(df, row)
            all_signals.append({
                "Symbol": symbol,
                "SignalTimestamp": row["SignalTimestamp"],
                "Close": round(row["Close"], 4),
                "Volume": int(row["Volume"]),
                "RVOL": row["RVOL"],
                **outcomes,
            })

    if not all_signals:
        return pd.DataFrame()

    result = pd.DataFrame(all_signals).sort_values(["Symbol", "SignalTimestamp"]).reset_index(drop=True)
    return result


def print_summary(result: pd.DataFrame):
    print(f"\n{'=' * 100}")
    print(f"  Backwards-facing RVOL screen: RVOL >= {VOL_SPIKE_RVOL_THRESHOLD}x sustained "
          f"{RVOL_SUSTAIN_MINUTES}m+ ({RVOL_SUSTAIN_BARS} bars) + {UPTREND_MIN_RETURN_PCT}%+ move "
          f"over last {UPTREND_LOOKBACK_BARS} bars, past {LOOKBACK_PERIOD}")
    print(f"{'=' * 100}")

    if result.empty:
        print("  No qualifying signals found in the lookback window.")
        return

    print(f"\n  {result['Symbol'].nunique()} distinct stocks qualified, {len(result)} total crossing timestamps\n")

    for symbol, rows in result.groupby("Symbol"):
        timestamps = ", ".join(t.strftime("%Y-%m-%d %H:%M %Z") if t.tzinfo else t.strftime("%Y-%m-%d %H:%M")
                                for t in rows["SignalTimestamp"])
        print(f"  {symbol}: {timestamps}")

    outcome_cols = ["30m_pct", "1h_pct", "2h_pct", "4h_pct", "1d_pct"]
    print("\n  Average forward outcome by window (across all signals, where available):")
    for col in outcome_cols:
        valid = result[col].dropna()
        if len(valid):
            print(f"    {col:>8}: avg {valid.mean():+.2f}%   median {valid.median():+.2f}%   "
                  f"win-rate {100 * (valid > 0).mean():.0f}%   (n={len(valid)})")
        else:
            print(f"    {col:>8}: no data yet (signals too recent)")

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Full signal-by-signal log with all outcome columns -> {OUTPUT_CSV}")


def main(symbols: list):
    result = run_screen(symbols)
    print_summary(result)
    return result


if __name__ == "__main__":
    # Default: pull the watchlist fresh from Finviz (float<10M, avg-vol>50K),
    # same as fullpipeline.py's get_all_tickers(). Pass a comma-separated
    # ticker list as the first argument to override this and skip Finviz
    # entirely, e.g.: python rvol_backtest_screener.py AAPL,TSLA,NVDA
    if len(sys.argv) > 1:
        watch_symbols = [s.strip().upper() for s in sys.argv[1].split(",")]
    else:
        print("[INIT] No symbols passed -- building watchlist from Finviz "
              "(float<10M, avg-vol>50K)...")
        watch_symbols = get_all_tickers()
        print(f"[INIT] Watchlist size: {len(watch_symbols)}")
        if not watch_symbols:
            print("[ERROR] Finviz returned 0 tickers -- check cache/finviz_debug_page.html if it was written.")
            sys.exit(1)

    main(watch_symbols)