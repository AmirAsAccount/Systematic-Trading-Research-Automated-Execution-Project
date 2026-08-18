"""
Hawkes Process Screener
-----------------------
1. Uses finviz Screener to find stocks up +100%+ today with 1M+ volume
2. Fetches OHLCV data for each via yfinance
3. Runs rolling Hawkes process MLE fitting
4. Reports per-ticker: positive/negative, all windows, first positive occurrence

Dependencies:
    pip install finviz yfinance scipy numpy
"""

import sys
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

try:
    from finviz.screener import Screener
except ImportError:
    print("finviz not installed. Run: pip install finviz")
    sys.exit(1)


# ── Finviz screener ───────────────────────────────────────────────────────────

def get_movers():
    """
    Screen for stocks with:
      - Change today >= +100%  (ta_change_u100)
      - Average volume >= 1M   (sh_avgvol_o1000)
    Returns list of ticker strings.
    """
    print("Scanning finviz for stocks up +100%+ today with 1M+ avg volume...")

    filters = [
        "ta_change_o20",    # price change today >= +100%
    ]

    try:
        screen = Screener(filters=filters, table="Performance", order="-perf_0")
    except Exception as e:
        print(f"  Screener error: {e}")
        return []

    tickers = []
    for row in screen:
        ticker = row.get("Ticker") or row.get("ticker")
        if ticker:
            tickers.append(ticker.strip().upper())

    return tickers


# ── Hawkes process helpers ────────────────────────────────────────────────────

def compute_events(log_returns, threshold_multiplier=1.0):
    vol = np.std(log_returns)
    if vol == 0:
        return np.array([])
    threshold = threshold_multiplier * vol
    return np.where(np.abs(log_returns) > threshold)[0].astype(float)


def hawkes_log_likelihood(params, events, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10

    n = len(events)
    if n == 0:
        return 1e10

    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))

    log_sum = 0.0
    R = 0.0
    for i in range(n):
        if i > 0:
            R = np.exp(-beta * (events[i] - events[i - 1])) * (1 + R)
        intensity = mu + alpha * R
        if intensity <= 0:
            return 1e10
        log_sum += np.log(intensity)

    return integral - log_sum


def fit_hawkes(events, T):
    """MLE fit. Returns (params_dict, success_bool)."""
    best_result = None
    best_val = np.inf

    for x0 in [[0.5, 0.3, 0.8], [0.1, 0.5, 1.0], [1.0, 0.2, 0.5], [0.3, 0.7, 1.5]]:
        res = minimize(
            hawkes_log_likelihood,
            x0,
            args=(events, T),
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None:
        return None, False

    mu, alpha, beta = best_result.x
    return {"mu": mu, "alpha": alpha, "beta": beta, "branching_ratio": alpha / beta}, True


def rolling_hawkes(log_returns, dates, window, step, threshold_mult=1.0):
    """Slide window over returns, fit Hawkes at each position."""
    results = []
    n = len(log_returns)

    for start in range(0, n - window, step):
        end = start + window
        events = compute_events(log_returns[start:end], threshold_mult)

        if len(events) < 5:
            continue

        params, ok = fit_hawkes(events, float(window))
        if not ok or params is None:
            continue

        br = params["branching_ratio"]
        results.append({
            "window_start": dates[start],
            "window_end":   dates[end - 1],
            "branching_ratio": br,
            "mu":    params["mu"],
            "alpha": params["alpha"],
            "beta":  params["beta"],
            "positive": br >= 0.5,
        })

    return results


# ── Per-ticker analysis ───────────────────────────────────────────────────────

def analyse_ticker(ticker, period, interval, window, step):
    print(f"\n  Fetching {ticker} ({period}, {interval})...")

    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  ERROR downloading {ticker}: {e}")
        return None

    if df.empty or len(df) < window + 10:
        print(f"  Skipping {ticker}: insufficient data ({len(df)} bars, need {window + 10})")
        return None

    close = df["Close"].squeeze().dropna().values
    dates = df.index[:len(close)]

    log_returns = np.diff(np.log(close))
    dates = dates[1:]

    windows = rolling_hawkes(log_returns, dates, window, step)

    if not windows:
        return {"ticker": ticker, "windows": [], "positive_count": 0,
                "total_count": 0, "overall_positive": False,
                "first_positive": None, "peak": None}

    positives = [w for w in windows if w["positive"]]
    pct = 100 * len(positives) / len(windows)
    overall = pct >= 30

    first_pos = positives[0] if positives else None
    peak = max(positives, key=lambda r: r["branching_ratio"]) if positives else None

    return {
        "ticker":           ticker,
        "windows":          windows,
        "positive_count":   len(positives),
        "total_count":      len(windows),
        "pct_positive":     pct,
        "overall_positive": overall,
        "first_positive":   first_pos,
        "peak":             peak,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def print_ticker_report(r, show_all_windows):
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {r['ticker']}  —  "
          f"{'✓ HAWKES POSITIVE' if r['overall_positive'] else '✗ NOT SIGNIFICANT'}")
    print(bar)
    print(f"  Windows tested   : {r['total_count']}")
    print(f"  Positive (BR≥0.5): {r['positive_count']}  ({r['pct_positive']:.1f}%)")

    if r["first_positive"]:
        fp = r["first_positive"]
        print(f"\n  First positive window:")
        print(f"    {fp['window_start'].date()}  →  {fp['window_end'].date()}")
        print(f"    Branching ratio : {fp['branching_ratio']:.4f}")
        print(f"    μ={fp['mu']:.4f}  α={fp['alpha']:.4f}  β={fp['beta']:.4f}")

    if r["peak"] and r["peak"] is not r["first_positive"]:
        pk = r["peak"]
        print(f"\n  Peak window:")
        print(f"    {pk['window_start'].date()}  →  {pk['window_end'].date()}")
        print(f"    Branching ratio : {pk['branching_ratio']:.4f}")

    if show_all_windows and r["windows"]:
        print(f"\n  {'Window Start':<14} {'Window End':<14} {'BR':>8}  Signal")
        print(f"  {'-'*12:<14} {'-'*12:<14} {'------':>8}  ------")
        for w in r["windows"]:
            flag = "✓ POSITIVE" if w["positive"] else "  negative"
            print(f"  {str(w['window_start'].date()):<14} "
                  f"{str(w['window_end'].date()):<14} "
                  f"{w['branching_ratio']:>8.4f}  {flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tickers = get_movers()
    print(tickers)
    print("=" * 62)
    print("  HAWKES PROCESS SCREENER  (finviz + yfinance)")
    print("=" * 62)

    # ── User parameters ───────────────────────────────────────────────────────
    period   = input("\nData period for yfinance (e.g. 3mo, 6mo, 1y) [default: 6mo]: ").strip() or "6mo"
    interval = input("Interval (1d / 1h) [default: 1d]: ").strip() or "1d"

    try:
        window = int(input("Rolling window size in bars [default: 60]: ").strip() or "60")
    except ValueError:
        window = 60

    try:
        step = int(input("Step size in bars [default: 10]: ").strip() or "10")
    except ValueError:
        step = 10

    show_all = input("Show all windows per ticker? (y/n) [default: n]: ").strip().lower() == "y"

    # ── Screener ──────────────────────────────────────────────────────────────
    tickers = get_movers()
    print(tickers)

    if not tickers:
        print("\nNo tickers matched the filter (up +100%+, 1M+ avg vol) today.")
        print("Markets may be closed, or no stocks qualify right now.")
        return

    print(f"\nFound {len(tickers)} ticker(s): {', '.join(tickers)}")

    # ── Hawkes analysis ───────────────────────────────────────────────────────
    print(f"\nRunning Hawkes analysis (window={window}, step={step})...")

    results = []
    for ticker in tickers:
        r = analyse_ticker(ticker, period, interval, window, step)
        if r:
            results.append(r)

    if not results:
        print("\nNo results — check your internet connection or try a longer period.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    positives = [r for r in results if r["overall_positive"]]
    print("\n\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  {'Ticker':<10} {'Windows':>8} {'% Pos':>8}  {'Overall'}")
    print(f"  {'-'*8:<10} {'-------':>8} {'-----':>8}  {'-------'}")
    for r in sorted(results, key=lambda x: -x["pct_positive"]):
        flag = "✓ HAWKES POSITIVE" if r["overall_positive"] else "  not significant"
        print(f"  {r['ticker']:<10} {r['total_count']:>8} {r['pct_positive']:>7.1f}%  {flag}")

    print(f"\n  {len(positives)}/{len(results)} tickers show Hawkes self-excitation")

    # ── Detailed reports ──────────────────────────────────────────────────────
    for r in results:
        print_ticker_report(r, show_all)

    print("\n" + "=" * 62)
    print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()