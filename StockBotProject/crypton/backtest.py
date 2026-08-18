#!/usr/bin/env python3
"""
backtest.py
-----------
Scans all crypto_cache/*.parquet files produced by crypto_fetch.py.

Entry — Bullish Volume Divergence via swing point detection:
  1. Find the two most recent confirmed swing lows in the window
  2. Price made a lower low at the second swing (LL)
  3. Volume at the second swing low is >= VOL_SPIKE_MULT × volume at first swing low
  4. Average volume in the bars surrounding the second swing > first swing (context)

A swing low is confirmed when N bars on each side are all higher (SWING_N).

Exit — Bearish Volume Divergence via swing point detection:
  1. Find two most recent confirmed swing highs after entry
  2. Price made a higher high (HH)
  3. OBV at the second swing high is declining vs the first

Only long (bullish) trades.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("crypto_cache")

# ── Swing detection ───────────────────────────────────────────────────────────
SWING_N        = 3      # bars each side needed to confirm a swing point

# ── Entry ─────────────────────────────────────────────────────────────────────
VOL_SPIKE_MULT = 6.0    # volume at 2nd swing low must be N× the 1st
AVG_VOL_RATIO  = 1.5    # avg vol in context window around 2nd low vs 1st
CONTEXT_BARS   = 3      # bars each side of swing to avg for context volume

# ── Exit ──────────────────────────────────────────────────────────────────────
DIV_WIN        = 6      # bars to look back for swing highs after entry
OBV_DROP_MULT  = 0.97   # OBV at 2nd swing high must be < this × OBV at 1st


# ─── Swing point detection ────────────────────────────────────────────────────

def find_swing_lows(series: pd.Series, n: int) -> list[int]:
    """
    Return positional indices of confirmed swing lows.
    A swing low at position i is confirmed if all bars within [i-n, i+n] are higher.
    """
    vals    = series.values
    length  = len(vals)
    swings  = []
    for i in range(n, length - n):
        window = vals[i - n: i + n + 1]
        if vals[i] == window.min() and list(window).count(vals[i]) == 1:
            swings.append(i)
    return swings


def find_swing_highs(series: pd.Series, n: int) -> list[int]:
    """Return positional indices of confirmed swing highs."""
    vals   = series.values
    length = len(vals)
    swings = []
    for i in range(n, length - n):
        window = vals[i - n: i + n + 1]
        if vals[i] == window.max() and list(window).count(vals[i]) == 1:
            swings.append(i)
    return swings


# ─── Entry ────────────────────────────────────────────────────────────────────

def check_entry(window: pd.DataFrame) -> bool:
    """
    Bullish volume divergence using real swing lows:
      - Find the last two confirmed swing lows in the window
      - Price lower low at the second swing
      - Volume spike at the second swing vs the first
      - Average volume context higher around the second swing
    """
    close  = window['close']
    vol    = window['volume']
    swings = find_swing_lows(close, SWING_N)

    if len(swings) < 2:
        return False

    sw1, sw2 = swings[-2], swings[-1]   # second-to-last and last swing low

    price_ll = close.iloc[sw2] < close.iloc[sw1]
    vol_spike = vol.iloc[sw2] >= vol.iloc[sw1] * VOL_SPIKE_MULT

    # Context volume: average of CONTEXT_BARS bars around each swing
    def ctx_vol(idx):
        lo = max(0, idx - CONTEXT_BARS)
        hi = min(len(vol), idx + CONTEXT_BARS + 1)
        return vol.iloc[lo:hi].mean()

    avg_vol_up = ctx_vol(sw2) > ctx_vol(sw1) * AVG_VOL_RATIO

    return bool(price_ll and vol_spike and avg_vol_up)


# ─── Exit ─────────────────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.Series:
    pc  = df['close'].diff()
    obv = [0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + df['volume'].iloc[i] * np.sign(pc.iloc[i]))
    return pd.Series(obv, index=df.index)


def find_exit(bars: pd.DataFrame, entry_idx: int) -> dict:
    """
    Bearish volume divergence on swing highs:
      - Scan bars after entry for confirmed swing highs
      - Once we have two: if price HH but OBV declining → exit
      - Fallback: end of data
    """
    obv_full   = compute_obv(bars)
    bars_after = bars.iloc[entry_idx:].copy()
    closes     = bars_after['close'].values
    highs      = bars_after['high'].values
    obv_vals   = obv_full.iloc[entry_idx:].values

    exit_idx    = None
    exit_reason = None

    # Build up swing highs incrementally as we walk forward
    confirmed_highs = []   # list of (pos_in_bars_after, high_price, obv_value)

    for i in range(SWING_N, len(bars_after) - SWING_N):
        # Check if position i is a swing high in the slice we can confirm
        lo = i - SWING_N
        hi = i + SWING_N + 1
        window_h = highs[lo:hi]
        if highs[i] == window_h.max() and list(window_h).count(highs[i]) == 1:
            confirmed_highs.append((i, highs[i], obv_vals[i]))

        if len(confirmed_highs) >= 2:
            sh1 = confirmed_highs[-2]
            sh2 = confirmed_highs[-1]
            price_hh  = sh2[1] > sh1[1]
            obv_lower = sh2[2] < sh1[2] * OBV_DROP_MULT
            if price_hh and obv_lower:
                exit_idx    = sh2[0]
                exit_reason = 'bearish_vol_divergence'
                break

    if exit_idx is None:
        exit_idx, exit_reason = len(bars_after) - 1, 'end_of_data'

    entry_price  = closes[0]
    exit_price   = closes[exit_idx]
    peak_price   = highs[:exit_idx + 1].max()
    max_gain     = peak_price - entry_price
    peak_capture = max(-100.0, (exit_price - entry_price) / max_gain * 100) \
                   if max_gain > entry_price * 0.005 else 0.0

    return dict(exit_price=exit_price, peak_price=peak_price,
                bars_to_exit=exit_idx, peak_capture=peak_capture,
                exit_reason=exit_reason)


# ─── Scan one file ────────────────────────────────────────────────────────────

def scan_pair(pair: str, path: Path, lookback: int) -> list[dict]:
    bars = pd.read_parquet(path)
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    bars.columns = [c.lower() for c in bars.columns]

    trades = []
    for entry_idx in range(lookback, len(bars) - 2):
        window = bars.iloc[entry_idx - lookback: entry_idx + 1]
        if not check_entry(window):
            continue

        exit_info    = find_exit(bars, entry_idx)
        entry_price  = bars.iloc[entry_idx]['close']
        exit_idx_abs = entry_idx + exit_info['bars_to_exit']
        exit_time    = bars.index[min(exit_idx_abs, len(bars) - 1)]
        pnl          = exit_info['exit_price'] - entry_price
        pnl_pct      = pnl / entry_price * 100

        trades.append(dict(
            pair         = pair,
            entry_time   = bars.index[entry_idx],
            exit_time    = exit_time,
            entry_price  = round(entry_price, 6),
            exit_price   = round(exit_info['exit_price'], 6),
            peak_price   = round(exit_info['peak_price'], 6),
            pnl          = round(pnl, 6),
            pnl_pct      = round(pnl_pct, 4),
            bars_to_exit = exit_info['bars_to_exit'],
            peak_capture = round(exit_info['peak_capture'], 2),
            exit_reason  = exit_info['exit_reason'],
        ))

    return trades


# ─── Reporting ────────────────────────────────────────────────────────────────

def remove_outlier_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    avg_pnl  = df.groupby('pair')['pnl_pct'].mean()
    q1, q3   = avg_pnl.quantile(0.25), avg_pnl.quantile(0.75)
    iqr      = q3 - q1
    lo, hi   = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = avg_pnl[(avg_pnl < lo) | (avg_pnl > hi)].index.tolist()
    return df[~df['pair'].isin(outliers)].copy(), outliers


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No trades found.")
        return

    df, outliers = remove_outlier_pairs(df)
    if outliers:
        print(f"\n  Outlier pairs removed : {', '.join(outliers)}")
    if df.empty:
        print("No trades remaining after outlier removal.")
        return

    winners = df[df['pnl'] > 0]
    losers  = df[df['pnl'] <= 0]
    gp      = winners['pnl_pct'].sum()
    gl      = abs(losers['pnl_pct'].sum())
    pf      = gp / gl if gl > 0 else float('inf')

    print(f"\n{'═'*52}")
    print(f"  BACKTEST RESULTS  —  {len(df)} trades across {df['pair'].nunique()} pairs")
    print(f"{'═'*52}")
    print(f"  Win Rate       : {len(winners)/len(df)*100:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Avg PnL        : {df['pnl_pct'].mean():.3f}%")
    print(f"  Median PnL     : {df['pnl_pct'].median():.3f}%")
    print(f"  Total PnL      : {df['pnl_pct'].sum():.2f}%")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Avg Bars Held  : {df['bars_to_exit'].mean():.1f}")
    print(f"  Avg Peak Cap.  : {df['peak_capture'].mean():.1f}%")
    print(f"\n  Exit reasons:")
    for reason, cnt in df['exit_reason'].value_counts().items():
        avg = df[df['exit_reason'] == reason]['pnl_pct'].mean()
        print(f"    {reason:<35} {cnt:>4}   avg pnl={avg:.3f}%")
    print(f"\n  Top 5 pairs by avg PnL:")
    top = df.groupby('pair')['pnl_pct'].mean().sort_values(ascending=False).head(5)
    for t, v in top.items():
        print(f"    {t:<14} {v:+.3f}%")
    print(f"{'═'*56}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',   default='backtest_results.csv')
    parser.add_argument('--lookback', type=int, default=30,
                        help='Bars fed into the entry window (default: 30)')
    parser.add_argument('--pairs',    nargs='*', default=None,
                        help='Specific pairs e.g. AGLD_USDT (default: all in cache)')
    args = parser.parse_args()

    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {CACHE_DIR.resolve()}/")
        print("Run crypto_fetch.py first.")
        return

    pair_map = {p.stem: p for p in parquet_files}
    pairs    = args.pairs or list(pair_map.keys())
    print(f"Scanning {len(pairs)} pairs  (lookback={args.lookback} bars @ 1m, swing_n={SWING_N})")

    all_trades = []
    for i, pair in enumerate(pairs, 1):
        path = pair_map.get(pair)
        if path is None:
            print(f"  [{i}/{len(pairs)}] {pair:<14} — not in cache, skipping")
            continue
        trades = scan_pair(pair, path, args.lookback)
        print(f"  [{i}/{len(pairs)}] {pair:<14} — {len(trades)} signals")
        all_trades.extend(trades)

    if not all_trades:
        print("No signals found. Try lowering SWING_N or VOL_SPIKE_MULT.")
        return

    df = pd.DataFrame(all_trades)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} trades → {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()