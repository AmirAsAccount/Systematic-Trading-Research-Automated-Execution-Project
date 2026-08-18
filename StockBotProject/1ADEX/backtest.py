#!/usr/bin/env python3
"""
backtest.py — Bullish OBV Divergence (swing-point based) for low-liquidity,
              high-turnover Solana pools.

ENTRY
  Bullish volume divergence at a confirmed swing low:
    price makes a Lower Low (LL) vs the prior confirmed swing low
    OBV     makes a Higher Low (HL) at the same point
  i.e. price is making new lows but underlying buy/sell pressure (OBV) is NOT
  confirming the weakness — classic absorption / exhaustion signature.

EXIT — checked in priority order, every bar, first match wins:
  Case 1 (PRIMARY)   — divergence fulfilled: price closes back above the
                        nearest confirmed swing HIGH that preceded entry
                        (structure reclaimed = the reversal thesis played out)
  Case 2 (SECONDARY) — bearish volume divergence: two confirmed swing highs
                        form post-entry where price makes a Higher High but
                        OBV makes a Lower High (momentum fading on the way up)
  Case 3 (TERTIARY)  — bearish volume: a single red candle (close < open)
                        with volume spiking over its local average — a sharp,
                        volume-confirmed rejection that doesn't need a full
                        swing pattern to justify cutting the trade
  FALLBACK           — end of data

No transaction costs are modeled by default — see TAKER_FEE_BPS / SLIPPAGE_BPS
below. For thin Solana pools these are NOT negligible; turn them on before
trusting any PnL numbers.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

STABLE_SYMBOLS = {
    "USDT", "USDC", "USDH", "DAI", "UST", "USDS", "BUSD", "TUSD",
    "PYUSD", "FDUSD", "EURC", "USDE",
}

CACHE_DIR = Path("dex_cache")

# ── Swing structure ─────────────────────────────────────────────────────────
SWING_N = 3          # bars each side required to confirm a swing point
                      # (so a swing point is only "known" N bars after it forms)

# ── Entry: bullish OBV divergence at swing lows ─────────────────────────────
LOOKBACK = 30         # bars scanned to find the last two confirmed swing lows

# ── Exit Case 3: bearish volume spike ───────────────────────────────────────
BEARISH_VOL_MULT  = 3.0   # current bar volume >= this × rolling avg volume
BEARISH_VOL_LOOKBACK = 20 # bars used for the rolling average

# ── Costs (OFF by default — flip on once you're trusting numbers) ──────────
APPLY_COSTS    = False
TAKER_FEE_BPS  = 30.0      # ~0.30% per side is typical for thin Raydium pools
SLIPPAGE_BPS   = 50.0      # placeholder — should be sized to liquidity per pool


# ──────────────────────────────────────────────────────────────────────────────
# SWING DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def find_swing_lows(series: pd.Series, n: int) -> list[int]:
    vals = series.values
    return [
        i for i in range(n, len(vals) - n)
        if vals[i] == min(vals[i - n: i + n + 1])
        and list(vals[i - n: i + n + 1]).count(vals[i]) == 1
    ]


def find_swing_highs(series: pd.Series, n: int) -> list[int]:
    vals = series.values
    return [
        i for i in range(n, len(vals) - n)
        if vals[i] == max(vals[i - n: i + n + 1])
        and list(vals[i - n: i + n + 1]).count(vals[i]) == 1
    ]


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Vectorized On-Balance Volume."""
    direction = np.sign(df['close'].diff()).fillna(0.0)
    return (df['volume'] * direction).cumsum()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY — bullish OBV divergence at the latest confirmed swing low
# ──────────────────────────────────────────────────────────────────────────────

def check_entry(window: pd.DataFrame) -> tuple[bool, float]:
    """
    Returns (fired, structure_high_price).
    structure_high_price = price of the most recent confirmed swing HIGH
    *before* the divergence low — this is the Case-1 exit target (the level
    that must be reclaimed for the divergence thesis to be considered fulfilled).
    """
    close = window['close']
    obv   = compute_obv(window)

    lows = find_swing_lows(close, SWING_N)
    if len(lows) < 2:
        return False, 0.0

    sw1, sw2 = lows[-2], lows[-1]          # prior low, latest confirmed low
    price_ll = close.iloc[sw2] < close.iloc[sw1]   # price: lower low
    obv_hl   = obv.iloc[sw2] > obv.iloc[sw1]        # OBV: higher low (divergence)

    if not (price_ll and obv_hl):
        return False, 0.0

    # find nearest confirmed swing high before sw2 to use as the reclaim target
    highs = [h for h in find_swing_highs(close, SWING_N) if h < sw2]
    if highs:
        structure_high = float(close.iloc[highs[-1]])
    else:
        # no confirmed swing high in window — fall back to the local max
        # of the window up to sw2 as a reasonable reclaim target
        structure_high = float(close.iloc[: sw2 + 1].max())

    return True, structure_high


# ──────────────────────────────────────────────────────────────────────────────
# EXIT — three cases, checked in priority order every bar
# ──────────────────────────────────────────────────────────────────────────────

def find_exit(bars: pd.DataFrame, entry_idx: int, structure_high: float) -> dict:
    obv_full   = compute_obv(bars)
    after      = bars.iloc[entry_idx:].copy()
    closes     = after['close'].values
    opens      = after['open'].values
    highs      = after['high'].values
    vols       = after['volume'].values
    obv_vals   = obv_full.iloc[entry_idx:].values

    confirmed_highs: list[tuple[int, float, float]] = []   # (idx, price, obv)
    exit_idx, exit_reason = None, None

    for i in range(1, len(after)):
        # ── Case 1 (PRIMARY): divergence fulfilled ──────────────────────────
        if closes[i] > structure_high:
            exit_idx, exit_reason = i, "case1_divergence_fulfilled"
            break

        # ── Case 2 (SECONDARY): bearish volume divergence on swing highs ───
        lo, hi = i - SWING_N, i + SWING_N + 1
        if lo >= 0 and hi <= len(after):
            window_h = highs[lo:hi]
            if highs[i] == window_h.max() and list(window_h).count(highs[i]) == 1:
                confirmed_highs.append((i, highs[i], obv_vals[i]))
            if len(confirmed_highs) >= 2:
                sh1, sh2 = confirmed_highs[-2], confirmed_highs[-1]
                price_hh = sh2[1] > sh1[1]      # price: higher high
                obv_lh   = sh2[2] < sh1[2]      # OBV: lower high (divergence)
                if price_hh and obv_lh:
                    exit_idx, exit_reason = sh2[0], "case2_bearish_obv_divergence"
                    break

        # ── Case 3 (TERTIARY): bearish volume spike ─────────────────────────
        vlo = max(0, i - BEARISH_VOL_LOOKBACK)
        avg_vol = vols[vlo:i].mean() if i > vlo else np.nan
        is_red       = closes[i] < opens[i]
        is_vol_spike = (not np.isnan(avg_vol)) and avg_vol > 0 and vols[i] >= BEARISH_VOL_MULT * avg_vol
        if is_red and is_vol_spike:
            exit_idx, exit_reason = i, "case3_bearish_volume"
            break

    if exit_idx is None:
        exit_idx, exit_reason = len(after) - 1, "end_of_data"

    entry_price = closes[0]
    exit_price  = closes[exit_idx]
    peak_price  = highs[: exit_idx + 1].max()
    max_gain    = peak_price - entry_price
    peak_capture = (
        max(-100.0, (exit_price - entry_price) / max_gain * 100)
        if max_gain > entry_price * 0.005 else 0.0
    )

    return dict(
        exit_price=exit_price, peak_price=peak_price,
        bars_to_exit=exit_idx, peak_capture=peak_capture,
        exit_reason=exit_reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
# COSTS
# ──────────────────────────────────────────────────────────────────────────────

def apply_costs(entry_price: float, exit_price: float) -> tuple[float, float]:
    if not APPLY_COSTS:
        return entry_price, exit_price
    total_bps = TAKER_FEE_BPS + SLIPPAGE_BPS
    entry_adj = entry_price * (1 + total_bps / 10_000)   # pay more going in
    exit_adj  = exit_price  * (1 - total_bps / 10_000)   # receive less going out
    return entry_adj, exit_adj


# ──────────────────────────────────────────────────────────────────────────────
# SCAN
# ──────────────────────────────────────────────────────────────────────────────

def scan_pair(pair: str, path: Path, lookback: int) -> list[dict]:
    bars = pd.read_parquet(path)
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    bars.columns = [c.lower() for c in bars.columns]

    trades = []
    entry_idx = lookback
    n = len(bars)

    while entry_idx < n - 2:
        window = bars.iloc[entry_idx - lookback: entry_idx + 1]
        fired, structure_high = check_entry(window)
        if not fired:
            entry_idx += 1
            continue

        exit_info   = find_exit(bars, entry_idx, structure_high)
        raw_entry   = bars.iloc[entry_idx]['close']
        raw_exit    = exit_info['exit_price']
        entry_price, exit_price = apply_costs(raw_entry, raw_exit)

        exit_idx_abs = entry_idx + exit_info['bars_to_exit']
        exit_time    = bars.index[min(exit_idx_abs, n - 1)]
        pnl          = exit_price - entry_price
        pnl_pct      = pnl / entry_price * 100

        trades.append(dict(
            pair           = pair,
            entry_time     = bars.index[entry_idx],
            exit_time      = exit_time,
            entry_price    = round(entry_price, 8),
            exit_price     = round(exit_price, 8),
            peak_price     = round(exit_info['peak_price'], 8),
            structure_high = round(structure_high, 8),
            pnl            = round(pnl, 8),
            pnl_pct        = round(pnl_pct, 4),
            bars_to_exit   = exit_info['bars_to_exit'],
            peak_capture   = round(exit_info['peak_capture'], 2),
            exit_reason    = exit_info['exit_reason'],
        ))

        # No new entries while this trade is "open" — jump straight past the
        # exit bar before scanning for the next signal. This is what makes
        # trades mutually exclusive instead of overlapping/correlated.
        entry_idx = exit_idx_abs + 1

    return trades


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def remove_outlier_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops entire pairs whose average PnL is an IQR outlier relative to the
    rest of the pair universe. This is a blunt tool — a pair with absurd
    avg PnL is just as likely to be a data artifact (broken decimals, a
    bad print from the source API, a mispriced illiquid pool) as it is to
    be a real edge. Both the raw and post-removal numbers are shown so
    nothing is hidden by this step.
    """
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

    print(f"\n{'═' * 56}")
    print(f"  RAW  —  {len(df)} trades across {df['pair'].nunique()} pairs (before IQR removal)")
    _print_stats_block(df)

    clean_df, outliers = remove_outlier_pairs(df)
    print(f"\n{'═' * 56}")
    if outliers:
        print(f"  IQR outlier pairs removed ({len(outliers)}): {', '.join(outliers)}")
        removed = df[df['pair'].isin(outliers)]
        print(f"  (their stats — inspect these before trusting them as real edge:)")
        print(removed.groupby('pair')['pnl_pct'].agg(['mean', 'count']).to_string())
    else:
        print("  No pairs flagged as IQR outliers.")
    print(f"{'═' * 56}")

    if clean_df.empty:
        print("  No trades remaining after outlier removal.")
        return

    print(f"\n  CLEAN  —  {len(clean_df)} trades across {clean_df['pair'].nunique()} pairs (after IQR removal)")
    _print_stats_block(clean_df)


def _print_stats_block(df: pd.DataFrame) -> None:
    winners = df[df['pnl'] > 0]
    losers  = df[df['pnl'] <= 0]
    gp = winners['pnl_pct'].sum()
    gl = abs(losers['pnl_pct'].sum())
    pf = gp / gl if gl > 0 else float('inf')

    print(f"  Costs modeled: {'YES (' + str(TAKER_FEE_BPS + SLIPPAGE_BPS) + ' bps round-trip)' if APPLY_COSTS else 'NO — see APPLY_COSTS'}")
    print(f"  Win Rate       : {len(winners)/len(df)*100:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Avg PnL        : {df['pnl_pct'].mean():.3f}%")
    print(f"  Median PnL     : {df['pnl_pct'].median():.3f}%")
    print(f"  Total PnL      : {df['pnl_pct'].sum():.2f}%")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Avg Bars Held  : {df['bars_to_exit'].mean():.1f}")
    print(f"  Avg Peak Cap.  : {df['peak_capture'].mean():.1f}%")
    print(f"\n  Exit reasons (in priority order):")
    for reason in ["case1_divergence_fulfilled", "case2_bearish_obv_divergence",
                    "case3_bearish_volume", "end_of_data"]:
        sub = df[df['exit_reason'] == reason]
        if len(sub) == 0:
            continue
        print(f"    {reason:<32} {len(sub):>4}   avg pnl={sub['pnl_pct'].mean():+.3f}%"
              f"   win rate={ (sub['pnl'] > 0).mean()*100:.1f}%")
    print(f"\n  Per-pair avg PnL (all pairs, no filtering):")
    by_pair = df.groupby('pair')['pnl_pct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(by_pair.to_string())
    print(f"{'═' * 56}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',   default='backtest_results.csv')
    parser.add_argument('--lookback', type=int, default=LOOKBACK)
    parser.add_argument('--pairs',    nargs='*', default=None)
    parser.add_argument('--costs',    action='store_true', help='apply fee+slippage costs')
    args = parser.parse_args()

    global APPLY_COSTS
    if args.costs:
        APPLY_COSTS = True

    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files in {CACHE_DIR.resolve()}/ — run dex_fetch.py first.")
        return

    pair_map = {p.stem: p for p in parquet_files}

    def is_stable(pair_name: str) -> bool:
        sym = pair_name.split("_")[0].upper()
        return sym in STABLE_SYMBOLS

    n_before = len(pair_map)
    pair_map = {k: v for k, v in pair_map.items() if not is_stable(k)}
    n_dropped = n_before - len(pair_map)
    if n_dropped:
        print(f"Skipping {n_dropped} stablecoin-prefixed cached pair(s) "
              f"(stale from before the fetch-script symbol fix).")

    pairs    = args.pairs or list(pair_map.keys())
    print(f"Scanning {len(pairs)} pairs  (lookback={args.lookback} bars @ 1m, swing_n={SWING_N})")
    print(f"Costs: {'ON' if APPLY_COSTS else 'OFF'}\n")

    all_trades = []
    for i, pair in enumerate(pairs, 1):
        path = pair_map.get(pair)
        if path is None:
            print(f"  [{i}/{len(pairs)}] {pair:<20} — not in cache, skipping")
            continue
        trades = scan_pair(pair, path, args.lookback)
        print(f"  [{i}/{len(pairs)}] {pair:<20} — {len(trades)} signals")
        all_trades.extend(trades)

    if not all_trades:
        print("No signals found across any pair.")
        return

    df = pd.DataFrame(all_trades)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} trades → {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()