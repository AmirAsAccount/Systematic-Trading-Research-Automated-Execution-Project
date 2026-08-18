"""
test_pipeline_logic.py
-----------------------
Offline test harness for live_catalyst_pipeline.py (fullpipeline.py).

Goal: exercise the REAL confirmation/exit decision logic (RVOL, float
rotation, anchored VWAP, swing structure, tight/structural stops) with
ZERO network calls and ZERO real orders — no API keys required.

How it works:
  - FakeSchwabClient replaces the real SchwabClient. Its price/quote
    methods return synthetic (or CSV-loaded) pandas DataFrames instead of
    hitting Schwab's API.
  - place_equity_order() just logs what WOULD have been sent — it never
    calls requests.post, so nothing live is ever placed.
  - We monkeypatch the module-level SCHWAB object in fullpipeline, then
    call the real confirm_and_maybe_enter() / _check_position() functions
    directly, bypassing the websocket/news-stream/threading entirely.

Run:
    python test_pipeline_logic.py

Edit SCENARIOS at the bottom to try different synthetic patterns
(clean breakout, fakeout that should get stopped, insufficient data, etc).
"""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import fullpipeline as pipe


# ══════════════════════════════════════════════════════════════════════════
# FAKE SCHWAB CLIENT — no network, no real orders
# ══════════════════════════════════════════════════════════════════════════

class FakeSchwabClient:
    def __init__(self, today_bars: pd.DataFrame, historical_bars: pd.DataFrame,
                 last_price: float, bid: float, ask: float, security_status: str = None):
        self._today = today_bars
        self._hist = historical_bars
        self._last = last_price
        self._bid = bid
        self._ask = ask
        self._status = security_status
        self.orders_placed = []   # for assertions after the run

    def get_price_history_1m(self, symbol, extended_hours=True):
        return self._today

    def get_price_history_multiday_1m(self, symbol, days, extended_hours=True):
        return self._hist

    def get_quote(self, symbol):
        return self._last

    def get_quote_full(self, symbol):
        return {"last": self._last, "bid": self._bid, "ask": self._ask,
                "security_status": self._status}

    def place_equity_order(self, account_hash, symbol, side, quantity):
        # Logs only — never touches the network, never places a real order.
        print(f"    [FAKE ORDER] {side} {quantity} {symbol} "
              f"(would notional ~${self._last * quantity:.2f})")
        self.orders_placed.append({"side": side, "symbol": symbol, "quantity": quantity})
        return {"status_code": 200, "order_location": "fake://no-op"}


# ══════════════════════════════════════════════════════════════════════════
# SYNTHETIC BAR BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def make_bullish_breakout_bars(news_time: datetime, minutes: int = 12,
                                start_price: float = 5.00) -> pd.DataFrame:
    """Clean, textbook confirm: steady climb w/ higher lows, volume, above VWAP."""
    idx = pd.date_range(news_time, periods=minutes, freq="1min", tz=timezone.utc)
    # a wobble-up pattern so swing lows are actually 'higher lows'
    closes = start_price + np.array([0, 0.05, 0.03, 0.10, 0.08, 0.16, 0.14,
                                      0.22, 0.20, 0.30, 0.28, 0.38])[:minutes]
    highs = closes + 0.03
    lows = closes - 0.04
    opens = np.roll(closes, 1); opens[0] = start_price
    volume = np.full(minutes, 40_000.0)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                          "Close": closes, "Volume": volume}, index=idx)


def make_fakeout_bars(news_time: datetime, minutes: int = 12,
                       start_price: float = 5.00) -> pd.DataFrame:
    """Spikes then dumps hard — should NOT confirm, and if entered, should
    trip the tight invalidation stop."""
    idx = pd.date_range(news_time, periods=minutes, freq="1min", tz=timezone.utc)
    closes = start_price + np.array([0, 0.30, 0.45, 0.20, -0.10, -0.35,
                                      -0.55, -0.60, -0.62, -0.65, -0.66, -0.68])[:minutes]
    highs = closes + 0.03
    lows = closes - 0.04
    opens = np.roll(closes, 1); opens[0] = start_price
    volume = np.full(minutes, 40_000.0)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                          "Close": closes, "Volume": volume}, index=idx)


def make_baseline_history(news_time: datetime, days: int = 10,
                           avg_minute_volume: float = 3_000.0) -> pd.DataFrame:
    """Prior-days same-time-of-day baseline, deliberately low volume so a
    breakout's ~40k/min easily clears RVOL_MIN."""
    frames = []
    for d in range(1, days + 1):
        day_start = news_time - timedelta(days=d)
        idx = pd.date_range(day_start, periods=12, freq="1min", tz=timezone.utc)
        vol = np.full(12, avg_minute_volume)
        price = np.full(12, 5.00)
        frames.append(pd.DataFrame({"Open": price, "High": price + 0.02,
                                     "Low": price - 0.02, "Close": price,
                                     "Volume": vol}, index=idx))
    return pd.concat(frames)


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_scenario(name: str, today_bars: pd.DataFrame, hist_bars: pd.DataFrame,
                  news_time: datetime, headline: str, summary: str = ""):
    print(f"\n{'='*70}\nSCENARIO: {name}\n{'='*70}")

    last_price = float(today_bars["Close"].iloc[-1])
    bid, ask = last_price - 0.01, last_price + 0.01  # tight, passes liquidity gate

    fake = FakeSchwabClient(today_bars, hist_bars, last_price, bid, ask)
    pipe.SCHWAB = fake                       # monkeypatch the market-data client
    pipe.SCHWAB_TRADER = fake                # monkeypatch the trader/order client too
    pipe.SCHWAB_ACCOUNT_HASH = "FAKE_HASH"
    pipe.open_positions.clear()
    pipe._trade_claimed = False
    pipe.SINGLE_TRADE_MODE = False           # let the harness run multiple scenarios freely

    # Fundamentals: skip the real Finviz scrape, feed a float directly
    pipe.get_ticker_fundamentals = lambda ticker: {"float_shares": 3_000_000, "short_float_pct": 25.0}

    print(f"  FDA fast-path? {pipe.contains_fda_fastpath(f'{headline} {summary}')}")

    # Skip FinBERT (heavy model load) unless you want to test it directly —
    # here we call the confirmation step directly, same as production would
    # after the sentiment gate + delay.
    pipe.confirm_and_maybe_enter("TEST", headline, news_time)

    if "TEST" in pipe.open_positions:
        print("  -> Position OPENED. Now feeding post-entry bars through the exit check...")
        pipe._check_position("TEST")
        print(f"  Orders placed this scenario: {fake.orders_placed}")
    else:
        print("  -> No entry (reaction did not confirm).")


if __name__ == "__main__":
    news_time = datetime.now(timezone.utc) - timedelta(minutes=5)

    run_scenario(
        "Clean breakout — should CONFIRM and enter",
        today_bars=make_bullish_breakout_bars(news_time),
        hist_bars=make_baseline_history(news_time),
        news_time=news_time,
        headline="Company announces positive Phase 3 results",
    )

    run_scenario(
        "Spike-and-fade fakeout — should NOT confirm",
        today_bars=make_fakeout_bars(news_time),
        hist_bars=make_baseline_history(news_time),
        news_time=news_time,
        headline="Company announces partnership",
    )