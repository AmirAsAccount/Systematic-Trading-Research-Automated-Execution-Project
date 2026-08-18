#!/usr/bin/env python3
"""
pnl_divergence_bb_adx_exit.py
------------------------------
Scans all cache/{TICKER}_15m_recent.parquet files directly.
No news CSV required.

Entry — squeeze is PRIMARY, divergences are SECONDARY (only checked if squeeze fires):
  1. BB Squeeze (PRIMARY)       — bandwidth in lowest percentile AND persists
                                  for squeeze_persist consecutive bars
  2. Bullish Volume Divergence  — price lower/equal low, OBV higher low
  3. Bullish RSI Divergence     — price lower/equal low, RSI higher low

Exit (first hit):
  A. Bearish Volume Divergence  — price HH but OBV declining over 10-bar window
  B. End of data
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("cache")

# Squeeze tuning
SQUEEZE_PERIOD    = 20    # BB period for bandwidth calculation
SQUEEZE_LOOKBACK  = 100   # how many bars of history to rank bandwidth against
SQUEEZE_PCTILE    = 10    # bandwidth must be in bottom N% of recent history
SQUEEZE_PERSIST   = 3     # how many consecutive bars must be in squeeze


# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.Series:
    price_change = df['Close'].diff()
    obv = [0]
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif price_change.iloc[i] < 0:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_bb_bandwidth(close: pd.Series, period: int = SQUEEZE_PERIOD,
                         std_dev: float = 2.0) -> pd.Series:
    """
    Bandwidth = (upper - lower) / mid  — normalised so price level doesn't matter.
    A squeeze is when bandwidth is historically low.
    """
    upper, mid, lower = compute_bollinger(close, period, std_dev)
    return (upper - lower) / mid.replace(0, np.nan)


def compute_adx_dm(bars: pd.DataFrame, period: int = 14):
    high       = bars['High'].values
    low        = bars['Low'].values
    prev_high  = bars['High'].shift(1).values
    prev_low   = bars['Low'].shift(1).values
    prev_close = bars['Close'].shift(1).values

    tr = np.nanmax(np.vstack([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ]), axis=0)

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move)  & (up_move > 0),  up_move,  0.0)
    minus_dm  = np.where((down_move > up_move)  & (down_move > 0), down_move, 0.0)

    def wilder_sum(arr):
        return pd.Series(arr).rolling(period, min_periods=1).sum().values

    tr_s   = np.where(wilder_sum(tr) == 0, 1e-9, wilder_sum(tr))
    pdi    = 100.0 * wilder_sum(plus_dm)  / tr_s
    mdi    = 100.0 * wilder_sum(minus_dm) / tr_s
    di_sum = np.where(pdi + mdi == 0, 1e-9, pdi + mdi)
    adx    = pd.Series(100.0 * np.abs(pdi - mdi) / di_sum) \
               .rolling(period, min_periods=1).mean().values
    return pdi, mdi, adx


# ─── Entry signals ────────────────────────────────────────────────────────────

def check_entry(window: pd.DataFrame,
                squeeze_lookback: int = SQUEEZE_LOOKBACK,
                squeeze_pctile:   int = SQUEEZE_PCTILE,
                squeeze_persist:  int = SQUEEZE_PERSIST) -> dict:
    """
    window  : fixed-size slice of bars ending at the candidate entry bar
    Returns a dict of individual signal flags + all_valid.

    Order of evaluation (squeeze is PRIMARY — gates everything else):
      1. BB Squeeze   — bandwidth in bottom squeeze_pctile% for last
                        squeeze_persist consecutive bars (tight compression)
      2. Bullish Vol Divergence — price LL, OBV HL  (secondary)
      3. Bullish RSI Divergence — price LL, RSI HL  (secondary)
    """
    out = dict(vol_div=False, rsi_div=False, bb_squeeze=False, all_valid=False)
    if len(window) < 25:
        return out

    close = window['Close']
    bw    = compute_bb_bandwidth(close)

    # ── PRIMARY: BB Squeeze ───────────────────────────────────────────────────
    # Rank bandwidth against the bars BEFORE the most recent squeeze_persist
    # bars so the history window is independent of the bars being tested.
    bw_clean = bw.dropna()
    if len(bw_clean) < squeeze_persist + 10:
        return out

    # History = everything except the last squeeze_persist bars
    bw_history  = bw_clean.iloc[:-squeeze_persist]
    bw_tail     = bw_clean.iloc[-squeeze_persist:]   # bars that must all be tight

    if len(bw_history) >= 10:
        threshold = np.percentile(bw_history.values, squeeze_pctile)
        # All persist-bars must individually be below the threshold
        out['bb_squeeze'] = bool((bw_tail <= threshold).all())

    # Gate: if no squeeze, skip the more expensive divergence checks
    if not out['bb_squeeze']:
        return out

    # ── SECONDARY: Divergence signals ────────────────────────────────────────
    obv = compute_obv(window)
    rsi = compute_rsi(close)
    mid = len(window) // 2

    # Bullish volume divergence — price LL, OBV HL
    price_ll = close.iloc[mid:].min() <= close.iloc[:mid].min() * 1.005
    obv_hl   = obv.iloc[mid:].min()   >  obv.iloc[:mid].min()
    out['vol_div'] = bool(price_ll and obv_hl)

    # Bullish RSI divergence — price LL (same), RSI HL
    rsi_hl = rsi.iloc[mid:].min() > rsi.iloc[:mid].min()
    out['rsi_div'] = bool(price_ll and rsi_hl)

    out['all_valid'] = out['vol_div'] and out['rsi_div']
    return out


# ─── Exit logic ───────────────────────────────────────────────────────────────

def find_exit(bars: pd.DataFrame, entry_idx: int) -> dict:
    obv_full  = compute_obv(bars)

    bars_after = bars.iloc[entry_idx:]
    closes     = bars_after['Close'].values
    highs      = bars_after['High'].values
    obv_after  = obv_full.iloc[entry_idx:].values

    DIV_WIN     = 10
    exit_idx    = None
    exit_reason = None

    for i in range(1, len(bars_after)):
        # Bearish volume divergence — price making HH but OBV declining
        if i >= DIV_WIN:
            s = slice(i - DIV_WIN, i + 1)
            price_hh = highs[s][-1] >= np.max(highs[s][:-1]) * 0.99
            obv_drop = obv_after[s][-1] < np.mean(obv_after[s][:-3])
            if price_hh and obv_drop:
                exit_idx, exit_reason = i, 'bearish_vol_divergence'
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


# ─── Scan one ticker ──────────────────────────────────────────────────────────

def scan_ticker(ticker: str, lookback: int,
                squeeze_lookback: int = SQUEEZE_LOOKBACK,
                squeeze_pctile:   int = SQUEEZE_PCTILE,
                squeeze_persist:  int = SQUEEZE_PERSIST) -> list[dict]:
    path = CACHE_DIR / f"{ticker}_15m_recent.parquet"
    bars = pd.read_parquet(path)
    bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()

    trades = []
    for entry_idx in range(lookback, len(bars) - 2):
        window  = bars.iloc[entry_idx - lookback: entry_idx + 1]
        signals = check_entry(window,
                              squeeze_lookback=squeeze_lookback,
                              squeeze_pctile=squeeze_pctile,
                              squeeze_persist=squeeze_persist)

        if not signals['all_valid']:
            continue

        exit_info    = find_exit(bars, entry_idx)
        entry_price  = bars.iloc[entry_idx]['Close']
        exit_idx_abs = entry_idx + exit_info['bars_to_exit']
        exit_time    = bars.index[exit_idx_abs] if exit_idx_abs < len(bars) else bars.index[-1]
        pnl          = exit_info['exit_price'] - entry_price
        pnl_pct      = pnl / entry_price * 100

        trades.append(dict(
            ticker        = ticker,
            entry_time    = bars.index[entry_idx],
            exit_time     = exit_time,
            entry_price   = round(entry_price, 4),
            exit_price    = round(exit_info['exit_price'], 4),
            peak_price    = round(exit_info['peak_price'], 4),
            pnl           = round(pnl, 4),
            pnl_pct       = round(pnl_pct, 4),
            bars_to_exit  = exit_info['bars_to_exit'],
            peak_capture  = round(exit_info['peak_capture'], 2),
            vol_div       = signals['vol_div'],
            rsi_div       = signals['rsi_div'],
            bb_squeeze    = signals['bb_squeeze'],
            exit_reason   = exit_info['exit_reason'],
        ))

    return trades


# ─── Reporting ────────────────────────────────────────────────────────────────

def remove_outlier_tickers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop tickers whose avg pnl_pct is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    across all tickers. Returns (clean_df, removed_ticker_list).
    """
    avg_pnl  = df.groupby('ticker')['pnl_pct'].mean()
    q1, q3   = avg_pnl.quantile(0.25), avg_pnl.quantile(0.75)
    iqr      = q3 - q1
    lo, hi   = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = avg_pnl[(avg_pnl < lo) | (avg_pnl > hi)].index.tolist()
    return df[~df['ticker'].isin(outliers)].copy(), outliers


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No trades found.")
        return

    df, outliers = remove_outlier_tickers(df)
    if outliers:
        print(f"\n  Outlier tickers removed : {', '.join(outliers)}")
    if df.empty:
        print("No trades remaining after outlier removal.")
        return

    winners = df[df['pnl'] > 0]
    losers  = df[df['pnl'] <= 0]
    gp      = winners['pnl_pct'].sum()
    gl      = abs(losers['pnl_pct'].sum())
    pf      = gp / gl if gl > 0 else float('inf')

    print(f"\n{'═'*52}")
    print(f"  BACKTEST RESULTS  —  {len(df)} trades across {df['ticker'].nunique()} tickers")
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
    print(f"\n  Top 5 tickers by avg PnL:")
    top = df.groupby('ticker')['pnl_pct'].mean().sort_values(ascending=False).head(5)
    for t, v in top.items():
        print(f"    {t:<8} {v:+.3f}%")
    print(f"{'═'*52}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',           default='pnl_divergence_bb_results.csv')
    parser.add_argument('--lookback',         type=int, default=30,
                        help='Bars in the divergence window (default: 30)')
    parser.add_argument('--squeeze-lookback', type=int, default=SQUEEZE_LOOKBACK,
                        help=f'Bars of bandwidth history to rank against (default: {SQUEEZE_LOOKBACK})')
    parser.add_argument('--squeeze-persist',  type=int, default=SQUEEZE_PERSIST,
                        help=f'Consecutive bars that must all be in squeeze (default: {SQUEEZE_PERSIST})')
    parser.add_argument('--squeeze-pctile',   type=int, default=SQUEEZE_PCTILE,
                        help=f'Bandwidth percentile threshold for squeeze (default: {SQUEEZE_PCTILE})')
    parser.add_argument('--tickers',          nargs='*', default=None,
                        help='Limit to specific tickers (default: all in cache)')
    args = parser.parse_args()

    parquet_files = sorted(CACHE_DIR.glob("*_15m_recent.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {CACHE_DIR.resolve()}/")
        return

    tickers = args.tickers or [p.stem.replace('_15m_recent', '') for p in parquet_files]
    print(f"Scanning {len(tickers)} tickers  "
          f"(lookback={args.lookback}, squeeze_lookback={args.squeeze_lookback}, "
          f"squeeze_pctile={args.squeeze_pctile}, squeeze_persist={args.squeeze_persist})")

    all_trades = []
    for i, ticker in enumerate(tickers, 1):
        path = CACHE_DIR / f"{ticker}_15m_recent.parquet"
        if not path.exists():
            print(f"  [{i}/{len(tickers)}] {ticker:<8} — no cache, skipping")
            continue
        trades = scan_ticker(ticker, args.lookback,
                             squeeze_lookback=args.squeeze_lookback,
                             squeeze_pctile=args.squeeze_pctile,
                             squeeze_persist=args.squeeze_persist)
        print(f"  [{i}/{len(tickers)}] {ticker:<8} — {len(trades)} signals")
        all_trades.extend(trades)

    if not all_trades:
        print("No valid signals found across all tickers.")
        return

    df = pd.DataFrame(all_trades)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} trades → {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()