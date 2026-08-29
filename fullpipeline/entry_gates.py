# entry_gates.py
# Development notes:
# - The long design and debugging notes from development were removed after the logic stabilized.
# - Comments are kept short and close to the code they explain.
# - Local naming is intentionally mixed where it reads naturally; public config names stay unchanged.
#
import re
import time
from datetime import datetime, time as dt_time
from typing import Optional
import logging

import config
from config import (
    AVG_DAILY_VOLUME,
    MAX_SPREAD_PCT,
    NY_TZ,
    SESSION_SECONDS_FOR_SCALING,
    TICK_POLL_INTERVAL_SEC,
    TICK_UP_VOLUME_RATIO_MIN,
    TICK_VOLUME_MULTIPLE,
    TICK_VOLUME_SESSION_CONFIRM_MULTIPLE,
    TICK_WINDOW_SEC,
)


# Treat only an explicit halted status as halted.
def _is_halted_status(status) -> bool:
    return bool(status) and str(status).strip().lower() == "halted"


# Reject names with invalid quotes or a wide spread.
def passes_liquidity_gate(symbol: str) -> bool:
    try:
        q = config.SCHWAB.get_quote_full(symbol)
    except Exception as e:
        print(f"  [WARN] Liquidity check failed for {symbol}: {e}")
        return False
    bid, ask = q.get("bid"), q.get("ask")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return False
    mid = (bid + ask) / 2.0
    spreadPct = (ask - bid) / mid
    if spreadPct > MAX_SPREAD_PCT:
        print(f"  [GATE] {symbol}: spread {spreadPct*100:.2f}% exceeds {MAX_SPREAD_PCT*100:.1f}% cap — blocked.")
        return False
    return True


# Scale average daily volume down to the polling window.
def scaled_volume_baseline(avg_daily_volume: Optional[float],
                            window_sec: int = TICK_WINDOW_SEC) -> Optional[float]:
    if not avg_daily_volume or avg_daily_volume <= 0:
        return None
    return float(avg_daily_volume) * (window_sec / SESSION_SECONDS_FOR_SCALING)


# Compare current volume with the session-scaled baseline.
def rvol_dual_check(symbol: str, volume_in_window: float, window_sec: float,
                     window_start_et: dt_time, avg_daily_volume: Optional[float],
                     primary_multiple: float, confirm_multiple: float) -> dict:
    baseline = scaled_volume_baseline(avg_daily_volume, window_sec)
    rvol = (volume_in_window / baseline) if baseline and baseline > 0 else 0.0
    passed = bool(baseline and baseline > 0 and rvol >= primary_multiple)

    return {
        "rvol_tod": rvol, "rvol_session": rvol,
        "tod_baseline": baseline, "session_baseline": baseline,
        "used_fallback": False,
        "tod_ok": passed, "session_ok": passed,
        "pass": passed,
    }


# Split new volume into buy and sell pressure.
def classify_tick_volume(prev_quote: Optional[dict], curr_quote: dict, delta_volume: float) -> tuple:
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


# Poll quotes until volume and buy pressure confirm.
def instant_tick_volume_check(symbol: str, news_time: datetime,
                               avg_daily_volume: Optional[float]) -> dict:
    window_start_et = news_time.astimezone(NY_TZ).time()

# Keep the starting cumulative volume for this window.
    start_total_vol = None
    prev_quote = None
    upTotal = 0.0
    downTotal = 0.0
# Track usable and failed quote polls.
    quote_polls_ok = 0
    quote_polls_failed = 0
    deadline = time.monotonic() + TICK_WINDOW_SEC

    while True:
        try:
            quote = config.SCHWAB.get_quote_full(symbol)
        except Exception as e:
            print(f"  [WARN] Instant tick-volume quote fetch failed for {symbol}: {e}")
            quote = None

        total_vol = quote.get("total_volume") if quote is not None else None
        if total_vol is not None:
            quote_polls_ok += 1
            if start_total_vol is None:
                start_total_vol = total_vol
            else:
                delta = float(total_vol) - float(start_total_vol) - (upTotal + downTotal)
                up_delta, down_delta = classify_tick_volume(prev_quote, quote, max(delta, 0.0))
                upTotal += up_delta
                downTotal += down_delta
            prev_quote = quote
        else:
            quote_polls_failed += 1

        if quote_polls_ok == 0:
            data_quality = "NO_DATA"
        elif quote_polls_failed > 0:
            data_quality = "PARTIAL"
        else:
            data_quality = "OK"

        volume_since_news = upTotal + downTotal
        up_ratio = (up_total / volume_since_news) if volume_since_news > 0 else 0.0
        rvol = rvol_dual_check(symbol, volume_since_news, TICK_WINDOW_SEC, window_start_et,
                                avg_daily_volume, primary_multiple=TICK_VOLUME_MULTIPLE,
                                confirm_multiple=TICK_VOLUME_SESSION_CONFIRM_MULTIPLE)
        volume_ok = rvol["pass"]
        ratio_ok = up_ratio >= TICK_UP_VOLUME_RATIO_MIN

# Fire as soon as both entry conditions pass.
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

# Stop once the confirmation window expires.
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
