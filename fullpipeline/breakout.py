"""Rolling-high breakout and market-scan strategy."""

from datetime import datetime, timedelta, timezone
import threading
from typing import Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import config
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    GAP_PCT_MIN,
    MARKET_SCAN_INTERVAL_SEC,
    NEWS_TIER_REFRESH_SEC,
    NY_TZ,
    OLD_BULLISH_NEWS_LOOKBACK_DAYS,
    OLLAMA_ENABLED,
    PREMARKET_MIN_VOLUME_SHARES,
    PREMARKET_OPEN_ET,
    QUOTE_BATCH_CHUNK_SIZE,
    ROLLING_HIGH_POLL_INTERVAL_SEC,
    RVOL_MIN,
    RVOL_SESSION_CONFIRM_MIN,
    TICK_UP_VOLUME_RATIO_MIN,
    TICK_VOLUME_MULTIPLE,
    TICK_WINDOW_SEC,
    _latest_quotes,
    _latest_quotes_lock,
    _positions_lock,
    _watchlist_lock,
    open_positions,
    opportunities_left,
    trading_paused,
    watchlist,
)

from execution import enter_position
from entry_gates import _is_halted_status, instant_tick_volume_check, rvol_dual_check
from watchlist import get_avg_daily_volume
from sentiment import call_ollama_catalyst_check
from sizing import compute_breakout_magnitude_score
from news_stream import (
    _is_stale_recap_headline,
    _bullish_news_history,
    _bullish_news_history_lock,
)


def _record_bullish_news(ticker: str, when: datetime):
    with _bullish_news_history_lock:
        _bullish_news_history.setdefault(ticker, []).append(when)


def has_recent_bullish_news(
    ticker: str,
    lookback_days: int = OLD_BULLISH_NEWS_LOOKBACK_DAYS,
) -> Optional[datetime]:
    """Return the most recent bullish-news timestamp within the lookback window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    with _bullish_news_history_lock:
        timestamps = _bullish_news_history.get(ticker, [])
        recent = [timestamp for timestamp in timestamps if timestamp >= cutoff]

    return max(recent) if recent else None


_gap_rvol_lock = threading.Lock()
_gap_rvol_qualified: dict[str, dict] = {}

_halt_state_lock = threading.Lock()
_halt_state: dict[str, bool] = {}

_scored_historical_article_ids_lock = threading.Lock()
_scored_historical_article_ids: set = set()

_session_high_lock = threading.Lock()
_session_high: dict[str, float] = {}


def _elapsed_seconds_since_premarket_open() -> float:
    """Return the number of seconds elapsed since the configured premarket open."""
    now_et = datetime.now(timezone.utc).astimezone(NY_TZ)
    session_start = now_et.replace(
        hour=PREMARKET_OPEN_ET.hour,
        minute=PREMARKET_OPEN_ET.minute,
        second=0,
        microsecond=0,
    )

    if now_et < session_start:
        session_start -= timedelta(days=1)

    # Avoid an unstable RVOL calculation during the first minute.
    return max(60.0, (now_et - session_start).total_seconds())


def market_scan_and_update():
    """Update gap/RVOL candidates and detect halt-to-reopen transitions."""
    if trading_paused():
        return

    with _watchlist_lock:
        symbols = sorted(watchlist)

    quotes = {}
    for i in range(0, len(symbols), QUOTE_BATCH_CHUNK_SIZE):
        chunk = symbols[i:i + QUOTE_BATCH_CHUNK_SIZE]
        try:
            quotes.update(config.SCHWAB.get_quotes_batch(chunk))
        except Exception as exc:
            print(
                f"  [WARN] Batched quote fetch failed for chunk "
                f"starting {chunk[0]}: {exc}"
            )

    with _latest_quotes_lock:
        _latest_quotes.update(quotes)

    elapsed = _elapsed_seconds_since_premarket_open()
    qualified = {}

    for ticker, quote in quotes.items():
        last = quote.get("last")
        previous_close = quote.get("previous_close")
        total_volume = quote.get("total_volume")
        halted_now = _is_halted_status(quote.get("security_status"))

        with _halt_state_lock:
            was_halted = _halt_state.get(ticker, False)
            _halt_state[ticker] = halted_now

        if was_halted and not halted_now:
            try:
                _check_halt_reopen_and_enter(ticker, quote)
            except Exception as exc:
                print(f"  [WARN] Halt-reopen check failed for {ticker}: {exc}")

        if not last or not previous_close or previous_close <= 0 or total_volume is None:
            continue

        gap_pct = (last - previous_close) / previous_close * 100.0
        avg_daily_volume = get_avg_daily_volume(ticker)

        rvol_result = rvol_dual_check(
            ticker,
            total_volume,
            elapsed,
            PREMARKET_OPEN_ET,
            avg_daily_volume,
            primary_multiple=RVOL_MIN,
            confirm_multiple=RVOL_SESSION_CONFIRM_MIN,
        )

        if (
            gap_pct >= GAP_PCT_MIN
            and rvol_result["pass"]
            and total_volume >= PREMARKET_MIN_VOLUME_SHARES
        ):
            qualified[ticker] = {
                "gap_pct": gap_pct,
                "rvol": rvol_result["rvol_tod"],
                "rvol_session": rvol_result["rvol_session"],
                "used_fallback_baseline": rvol_result["used_fallback"],
            }

    with _gap_rvol_lock:
        newly_added = set(qualified) - set(_gap_rvol_qualified)
        _gap_rvol_qualified.clear()
        _gap_rvol_qualified.update(qualified)

    if newly_added:
        print(
            f"  [MARKET-SCAN] Tier 1 "
            f"(gap>={GAP_PCT_MIN:.0f}%, RVOL>={RVOL_MIN:.1f}x, "
            f"vol>={PREMARKET_MIN_VOLUME_SHARES:,}): "
            f"{len(qualified)} candidate(s) total, newly added: "
            f"{sorted(newly_added)}"
        )


def market_scan_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            market_scan_and_update()
        except Exception as exc:
            print(f"  [WARN] market_scan_loop error: {exc}")
        stop_event.wait(MARKET_SCAN_INTERVAL_SEC)


def refresh_bullish_news_tier():
    """Check historical news for the current gap/RVOL candidates."""
    if trading_paused():
        return

    with _gap_rvol_lock:
        candidates = sorted(_gap_rvol_qualified)

    if not candidates:
        return

    client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    start = datetime.now(timezone.utc) - timedelta(
        days=OLD_BULLISH_NEWS_LOOKBACK_DAYS
    )

    for i in range(0, len(candidates), 100):
        chunk = candidates[i:i + 100]

        try:
            request = NewsRequest(
                symbols=",".join(chunk),
                start=start,
                limit=50,
                include_content=True,
                exclude_contentless=False,
            )
            news = client.get_news(request)
            articles = news.data.get("news", [])
        except Exception as exc:
            print(
                f"  [WARN] Tier-2 historical news fetch failed for "
                f"chunk starting {chunk[0]}: {exc}"
            )
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
            created_at = (
                getattr(article, "created_at", None)
                or datetime.now(timezone.utc)
            )
            article_symbols = getattr(article, "symbols", []) or []

            if _is_stale_recap_headline(headline):
                continue

            llm = (
                call_ollama_catalyst_check(headline, summary)
                if OLLAMA_ENABLED
                else None
            )

            if (
                llm is None
                or llm["catalyst"] != "positive"
                or llm["is_stale_or_secondhand"]
            ):
                continue

            for ticker in article_symbols:
                if ticker in candidates:
                    _record_bullish_news(ticker, created_at)
                    print(
                        f"  [MARKET-SCAN] Tier 2: {ticker} has a "
                        f"confirmed-bullish historical headline from "
                        f"{created_at.isoformat()}: {headline[:80]}"
                    )


def bullish_news_tier_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            refresh_bullish_news_tier()
        except Exception as exc:
            print(f"  [WARN] bullish_news_tier_loop error: {exc}")
        stop_event.wait(NEWS_TIER_REFRESH_SEC)


def _check_rolling_high_breakout_and_enter(ticker: str):
    """Check for a new session high and confirm it with volume."""
    if trading_paused():
        return

    with _positions_lock:
        if ticker in open_positions:
            return

    with _latest_quotes_lock:
        quote = _latest_quotes.get(ticker)

    last = quote.get("last") if quote else None
    if not last:
        return

    with _session_high_lock:
        prior_high = _session_high.get(ticker)
        if prior_high is None or last > prior_high:
            _session_high[ticker] = last

    if prior_high is None or last <= prior_high:
        return

    bullish_since = has_recent_bullish_news(ticker)
    if bullish_since is None:
        return

    with _gap_rvol_lock:
        gap_rvol = _gap_rvol_qualified.get(ticker)

    if gap_rvol is None:
        return

    print(
        f"  [ROLLING-HIGH] {ticker}: new session high ${last:.2f} "
        f"(prior ${prior_high:.2f}), gap={gap_rvol['gap_pct']:.1f}% "
        f"RVOL={gap_rvol['rvol']:.1f}x, bullish news from "
        f"{bullish_since.isoformat()}"
    )

    avg_volume = get_avg_daily_volume(ticker)
    result = instant_tick_volume_check(
        ticker,
        datetime.now(timezone.utc),
        avg_volume,
    )

    print(
        f"  [ROLLING-HIGH-CHECK] {ticker:<6} "
        f"volume_since_break={result['volume_since_news']:.0f} "
        f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
        f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
        f"up_ratio={result['up_ratio']:.2f} "
        f"(need >= {TICK_UP_VOLUME_RATIO_MIN}) "
        f"elapsed={result['elapsed_sec']:.1f}s "
        f"pass={result['pass']} data_quality={result['data_quality']}"
    )

    if not result["pass"]:
        print(
            f"  [ROLLING-HIGH-CHECK] {ticker}: breakout volume was not "
            f"confirmed within {TICK_WINDOW_SEC}s — no entry."
        )
        return

    score = compute_breakout_magnitude_score(
        result,
        gap_rvol["gap_pct"],
        gap_rvol["rvol"],
    )

    print(
        f"  [MAGNITUDE-B] {ticker}: "
        f"confirmation_score={score['confirmation_score']:.2f} "
        f"gap_scanner_score={score['gap_scanner_score']:.2f} "
        f"-> composite={score['composite']:.2f} -> {score['slots']} slot(s)"
    )

    enter_position(
        ticker,
        reason="ROLLING_HIGH_BREAKOUT",
        headline=(
            f"Gap/RVOL candidate (gap={gap_rvol['gap_pct']:.1f}%, "
            f"RVOL={gap_rvol['rvol']:.1f}x) + old bullish news "
            f"({bullish_since.isoformat()}) + session-high break "
            f"at ${last:.2f}"
        ),
        news_time=None,
        desired_slots=score["slots"],
    )


def rolling_high_breakout_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        with _gap_rvol_lock:
            candidates = sorted(_gap_rvol_qualified)

        for ticker in candidates:
            if stop_event.is_set() or opportunities_left() <= 0:
                break

            try:
                _check_rolling_high_breakout_and_enter(ticker)
            except Exception as exc:
                print(
                    f"  [WARN] rolling_high_breakout_loop error for "
                    f"{ticker}: {exc}"
                )

        stop_event.wait(ROLLING_HIGH_POLL_INTERVAL_SEC)


def _check_halt_reopen_and_enter(ticker: str, reopen_quote: dict):
    """Check a halt-to-reopen move and confirm it with volume."""
    if trading_paused():
        return

    with _positions_lock:
        if ticker in open_positions:
            return

    last = reopen_quote.get("last")
    previous_close = reopen_quote.get("previous_close")

    if not last:
        return

    gap_pct = (
        (last - previous_close) / previous_close * 100.0
        if previous_close and previous_close > 0
        else 0.0
    )

    print(
        f"  [HALT-REOPEN] {ticker}: halt->reopen at ${last:.2f} "
        f"(gap={gap_pct:.1f}%)"
    )

    avg_daily_volume = get_avg_daily_volume(ticker)
    result = instant_tick_volume_check(
        ticker,
        datetime.now(timezone.utc),
        avg_daily_volume,
    )

    elapsed = _elapsed_seconds_since_premarket_open()
    rvol_result = rvol_dual_check(
        ticker,
        reopen_quote.get("total_volume") or 0.0,
        elapsed,
        PREMARKET_OPEN_ET,
        avg_daily_volume,
        primary_multiple=RVOL_MIN,
        confirm_multiple=RVOL_SESSION_CONFIRM_MIN,
    )
    rvol = rvol_result["rvol_tod"]

    print(
        f"  [HALT-REOPEN-CHECK] {ticker:<6} "
        f"volume_since_reopen={result['volume_since_news']:.0f} "
        f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
        f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
        f"up_ratio={result['up_ratio']:.2f} "
        f"(need >= {TICK_UP_VOLUME_RATIO_MIN}) "
        f"elapsed={result['elapsed_sec']:.1f}s "
        f"pass={result['pass']} data_quality={result['data_quality']}"
    )

    if not result["pass"]:
        print(
            f"  [HALT-REOPEN-CHECK] {ticker}: reopen volume was not "
            f"confirmed within {TICK_WINDOW_SEC}s — no entry."
        )
        return

    score = compute_breakout_magnitude_score(result, gap_pct, rvol)

    print(
        f"  [MAGNITUDE-B] {ticker}: "
        f"confirmation_score={score['confirmation_score']:.2f} "
        f"gap_scanner_score={score['gap_scanner_score']:.2f} "
        f"-> composite={score['composite']:.2f} -> {score['slots']} slot(s)"
    )

    enter_position(
        ticker,
        reason="HALT_REOPEN_CONFIRMED",
        headline=(
            f"Halt->reopen at ${last:.2f} (gap={gap_pct:.1f}%), "
            "confirmed bullish by sustained buy-side volume post-resume"
        ),
        news_time=None,
        desired_slots=score["slots"],
    )
