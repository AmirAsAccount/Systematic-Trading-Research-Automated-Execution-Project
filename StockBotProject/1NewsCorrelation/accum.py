"""
accum_dist_research.py
----------------------
Standalone RESEARCH script — no broker connection, no order placement.

Fetches 5-minute OHLCV bars via yfinance for a hardcoded list of tickers
and computes three different accumulation/distribution style coefficients
side by side, purely for offline statistical inspection:

  1. Chaikin A/D Line coefficient
       Money Flow Multiplier = ((Close-Low)-(High-Close)) / (High-Low)
       Money Flow Volume     = Money Flow Multiplier * Volume
       A/D Line              = cumulative sum of Money Flow Volume
       Coefficient           = A/D Line change normalized by total volume
                                (so it's comparable across tickers/timeframes)

  2. Simple up/down volume split
       Coefficient = (sum of volume on up-close bars - sum of volume on
                      down-close bars) / total volume
       Range: -1 (all volume on down bars) to +1 (all volume on up bars)

  3. OBV-style signed volume
       Running sum: +volume on up-close bars, -volume on down-close bars,
                     0 contribution on unchanged-close bars
       Coefficient = OBV change normalized by total volume over the window

For all three: NEGATIVE = distribution-leaning, POSITIVE = accumulation-leaning.

WINDOW SELECTION -- fully manual, set by you:
  Each entry in TICKER_WINDOWS below is a (start, end) timestamp pair that
  YOU choose by looking at a chart, your own notes, or a news timestamp you
  already know. The script does not search for, detect, or infer where any
  window should begin or end -- it only slices the data at the exact
  boundaries you provide. If a ticker has no entry in TICKER_WINDOWS, it
  falls back to the full available lookback (see fetch_bars).

Yahoo Finance limits 5-minute interval data to the trailing 60 calendar
days — this script auto-clamps the lookback to that limit rather than
silently failing on a longer request.

Usage:
    python accum_dist_research.py

Edit TICKERS and TICKER_WINDOWS below to change what's analyzed.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance is required. Install with: pip install yfinance --break-system-packages")
    sys.exit(1)


# ---- Configuration ----------------------------------------------------------
TICKERS = ["AMIX"]   # <-- edit this list

# Manually-set analysis windows, per ticker. YOU decide these boundaries --
# e.g. a news timestamp you already know, and wherever you want to stop
# looking (an hour later, the next day, a specific bar you picked out on a
# chart, etc). Format: "YYYY-MM-DD HH:MM" in the exchange's local time
# (yfinance intraday timestamps are US/Eastern for US equities).
#
# Any ticker NOT listed here uses the full available lookback instead.
TICKER_WINDOWS = {
    "AMIX": ("2026-07-15 09:30", "2026-07-28 04:00")
}

# A bar's volume is flagged as an "outlier bar" in the report if it's this
# many standard deviations above the window's own mean bar volume. This is
# purely descriptive/reporting -- it does not change which bars are
# included in the coefficient calculations, and it never truncates the
# window itself.
OUTLIER_VOLUME_Z_THRESHOLD = 2.5

INTERVAL = "5m"
MAX_5M_LOOKBACK_DAYS = 59   # Yahoo's real limit is 60 days; 59 avoids edge-case rejections


def fetch_bars(ticker: str) -> pd.DataFrame:
    """Fetch 5-minute OHLCV bars for a ticker.

    If TICKER_WINDOWS has an entry for this ticker, fetch exactly that
    manually-specified start/end range. Otherwise fall back to the maximum
    available lookback (bounded by Yahoo's 60-day limit for 5m data).
    """
    if ticker in TICKER_WINDOWS:
        start_str, end_str = TICKER_WINDOWS[ticker]
        # Parse to datetime objects rather than passing raw strings through --
        # some yfinance versions mishandle "YYYY-MM-DD HH:MM" strings directly.
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
        df = yf.download(
            ticker,
            start=start_dt,
            end=end_dt,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
        )
    else:
        end = datetime.now()
        start = end - timedelta(days=MAX_5M_LOOKBACK_DAYS)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
        )

    if df.empty:
        return df

    # yfinance sometimes returns MultiIndex columns (Price, Ticker) even for
    # a single symbol -- flatten to plain OHLCV column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def find_outlier_bars(df: pd.DataFrame, z_threshold: float = OUTLIER_VOLUME_Z_THRESHOLD) -> pd.DataFrame:
    """Descriptive-only: report which bars in the window had unusually high
    volume, for visibility into where large prints sit in time. This does
    NOT remove, trim, or truncate anything -- it's purely informational
    output alongside the coefficients, which are always computed on the
    full window you specified.
    """
    vol = df["Volume"]
    mean_vol = vol.mean()
    std_vol = vol.std()

    if std_vol == 0 or np.isnan(std_vol):
        return pd.DataFrame(columns=["timestamp", "volume", "close", "z_score"])

    z_scores = (vol - mean_vol) / std_vol
    mask = z_scores > z_threshold

    outliers = pd.DataFrame({
        "timestamp": df.index[mask],
        "volume": vol[mask].values,
        "close": df["Close"][mask].values,
        "z_score": z_scores[mask].values,
    })
    return outliers.sort_values("timestamp").reset_index(drop=True)


def chaikin_ad_coefficient(df: pd.DataFrame) -> float:
    """Chaikin Accumulation/Distribution line, normalized to total volume."""
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]

    rng = (high - low).replace(0, np.nan)  # avoid div-by-zero on flat bars
    mf_multiplier = ((close - low) - (high - close)) / rng
    mf_multiplier = mf_multiplier.fillna(0.0)  # flat bars contribute 0

    mf_volume = mf_multiplier * vol
    ad_line = mf_volume.cumsum()

    total_volume = vol.sum()
    if total_volume == 0:
        return float("nan")

    # Net change in the A/D line over the window, normalized by total volume
    # traded, so the coefficient is comparable across tickers/timeframes.
    net_change = ad_line.iloc[-1] - ad_line.iloc[0]
    return float(net_change / total_volume)


def up_down_volume_split_coefficient(df: pd.DataFrame) -> float:
    """(up-close volume - down-close volume) / total volume. Range: -1..+1."""
    close = df["Close"]
    vol = df["Volume"]

    price_change = close.diff()
    up_vol = vol[price_change > 0].sum()
    down_vol = vol[price_change < 0].sum()
    total_vol = vol.sum()

    if total_vol == 0:
        return float("nan")

    return float((up_vol - down_vol) / total_vol)


def obv_style_coefficient(df: pd.DataFrame) -> float:
    """OBV-style signed running volume, normalized by total volume."""
    close = df["Close"]
    vol = df["Volume"]

    direction = np.sign(close.diff().fillna(0))  # +1, -1, or 0 per bar
    signed_vol = direction * vol
    obv = signed_vol.cumsum()

    total_vol = vol.sum()
    if total_vol == 0:
        return float("nan")

    net_change = obv.iloc[-1] - obv.iloc[0]
    return float(net_change / total_vol)


def classify(coef: float) -> str:
    if np.isnan(coef):
        return "n/a"
    if coef > 0.05:
        return "accumulation-leaning"
    if coef < -0.05:
        return "distribution-leaning"
    return "neutral"


def analyze_ticker(ticker: str) -> dict:
    df = fetch_bars(ticker)
    if df.empty:
        return {"ticker": ticker, "error": "no data returned"}

    return {
        "ticker": ticker,
        "bars": len(df),
        "start": df.index[0],
        "end": df.index[-1],
        "window_source": "manual" if ticker in TICKER_WINDOWS else "full lookback",
        "chaikin_ad": chaikin_ad_coefficient(df),
        "up_down_split": up_down_volume_split_coefficient(df),
        "obv_style": obv_style_coefficient(df),
        "outliers": find_outlier_bars(df),
    }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("ACCUMULATION / DISTRIBUTION RESEARCH REPORT  (negative = distribution-leaning)")
    print("=" * 100)

    for r in results:
        if "error" in r:
            print(f"\n{r['ticker']}: ERROR - {r['error']}")
            continue

        print(f"\n{r['ticker']}  ({r['bars']} bars, {r['start']} -> {r['end']})  [window: {r['window_source']}]")
        print("-" * 100)
        print(f"  {'Method':<28}{'Coefficient':>14}{'Interpretation':>28}")
        print(f"  {'Chaikin A/D line':<28}{r['chaikin_ad']:>14.4f}{classify(r['chaikin_ad']):>28}")
        print(f"  {'Up/down volume split':<28}{r['up_down_split']:>14.4f}{classify(r['up_down_split']):>28}")
        print(f"  {'OBV-style signed volume':<28}{r['obv_style']:>14.4f}{classify(r['obv_style']):>28}")

        outliers = r["outliers"]
        if outliers.empty:
            print(f"\n  No bars exceeded the {OUTLIER_VOLUME_Z_THRESHOLD}-sigma volume threshold in this window.")
        else:
            print(f"\n  Bars exceeding {OUTLIER_VOLUME_Z_THRESHOLD}-sigma volume (for visibility only -- "
                  f"still included in coefficients above):")
            print(f"    {'Timestamp':<22}{'Volume':>12}{'Close':>10}{'Z-score':>10}")
            for _, row in outliers.iterrows():
                print(f"    {str(row['timestamp']):<22}{row['volume']:>12,.0f}{row['close']:>10.2f}{row['z_score']:>10.2f}")

    print("\n" + "=" * 100)
    print("Notes:")
    print("  - All coefficients are normalized to be roughly comparable across tickers.")
    print("  - Chaikin A/D uses intrabar high/low/close positioning; the other two")
    print("    use only bar-to-bar close direction. They can disagree on choppy bars.")
    print("  - This is descriptive/exploratory only -- not a validated trading signal.")
    print("=" * 100 + "\n")


def main():
    results = [analyze_ticker(t) for t in TICKERS]
    print_report(results)


if __name__ == "__main__":
    main()