"""
apnl_crypto.py  —  Alt P&L Backtest  [Price Compression + Bullish Volume Divergence]
                    CRYPTO VERSION — reads from crypto_cache/ (5m native bars)

Parallel to apnl.py but adapted for:
  - crypto_cache/ parquets (covariatec.py format: timestamped OHLCV)
  - No session filter — crypto trades 24/7
  - Native 5m bars with in-memory resampling to coarser timeframes
  - Symbol names contain '/' (e.g. INJ/USDT) — safe-name handling for file I/O

SIGNAL DEFINITION
-----------------
  Price Compression  : ATR(atr_window) / Close < compress_thresh
                       AND bar range (High-Low) < range_mult * ATR
                       — price is coiling, volatility contracting

  Bullish Vol Div    : Volume > vol_mult * vol_ma(vol_window)   [volume expanding]
                       AND Close > Open                          [bar is green]
                       AND Close > prev_Close                    [price still rising]

  Signal fires when BOTH conditions are true on the same bar.

TRADE MECHANICS
---------------
  Entry  : Open of bar t+1
  Exit   : first bar where Close > entry * (1 + profit_target)   [profit target]
           OR first bar where Close < entry * (1 - stop_loss)     [stop loss]
           OR Close of bar t+horizon                              [time stop]
  Return : (exit_price - entry_price) / entry_price * 100  (%)
  Sizing : equal weight, arithmetic — no compounding

OUTPUTS
-------
  crypto_apnl_trades.csv
  crypto_apnl_by_symbol.csv
  crypto_apnl_by_threshold.csv
  crypto_apnl_summary.txt
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path("crypto_cache")
EPS          = 1e-12

# signal defaults
DEFAULT_ATR_WINDOW    = 10      # bars for ATR / range reference
DEFAULT_VOL_WINDOW    = 20      # bars for volume MA
DEFAULT_COMPRESS_MULT = 0.75    # bar range must be < this * ATR  (compression)
DEFAULT_ATR_THRESH    = 0.03    # ATR/Close < 3%  (absolute volatility low)
DEFAULT_VOL_MULT      = 1.5     # volume must be > 1.5x vol MA

# trade defaults
DEFAULT_HORIZON       = 5       # bars time stop
DEFAULT_PROFIT_TARGET = 0.03    # 3% profit target  (0 = disabled)
DEFAULT_STOP_LOSS     = 0.02    # 2% stop loss      (0 = disabled)

TAU_STEPS = np.round(np.arange(1.0, 3.51, 0.25), 2)  # vol_mult sweep

# Crypto: 5m native; resample options
_RESAMPLE_RULES = {
    "5m":  "5min",  "15m": "15min", "30m": "30min", "45m": "45min",
    "1h":  "1h",    "2h":  "2h",    "3h":  "3h",
    "4h":  "4h",    "1d":  "1D",
}
_BAR_MINUTES = {
    "5m": 5, "15m": 15, "30m": 30, "45m": 45,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "1d": 1440,
}

def log(msg=""): print(msg, flush=True)


# ── price loader ──────────────────────────────────────────────────────────────

def _safe(symbol):
    """INJ/USDT → INJ_USDT for filename matching."""
    return symbol.replace("/", "_")


def load_prices(resample_to=None, symbols=None):
    """
    Load full OHLCV from crypto_cache/ parquets.
    covariatec.py writes files named: {SYMBOL}_{TF}_{DATE_RANGE}.parquet
    e.g.  INJ_USDT_5m_20240622_20250622.parquet

    No session filter — crypto is 24/7.
    Optionally resamples to a coarser timeframe in memory.
    Returns dict: ticker -> DataFrame(Open, High, Low, Close, Volume)
    """
    prices = {}
    if not CACHE_DIR.exists():
        log(f"  WARNING: {CACHE_DIR}/ not found")
        return prices

    # Collect one (latest / largest) parquet per symbol
    sym_files = {}
    for p in sorted(CACHE_DIR.glob("*.parquet")):
        if ".partial." in p.name:
            continue
        # derive symbol from filename: first token before the timeframe token
        parts = p.stem.split("_")
        # find the timeframe token position
        tf_tokens = set(_RESAMPLE_RULES.keys())
        tf_idx = None
        for i, part in enumerate(parts):
            if part in tf_tokens:
                tf_idx = i
                break
        if tf_idx is None:
            continue
        sym_key = "/".join(parts[:tf_idx])   # e.g. "INJ/USDT"
        # keep largest file per symbol (most data)
        if sym_key not in sym_files or p.stat().st_size > sym_files[sym_key].stat().st_size:
            sym_files[sym_key] = p

    if symbols:
        sym_files = {k: v for k, v in sym_files.items() if k in symbols}

    for sym, path in sym_files.items():
        try:
            df = pd.read_parquet(path)
            # flatten MultiIndex columns if present
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            needed = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                      if c in df.columns]
            if len(needed) < 5:
                log(f"  [{sym}] missing OHLCV columns in {path.name}, skipping")
                continue

            df = df[needed].dropna()

            # Ensure UTC-aware index
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]

            # Resample if requested
            if resample_to and resample_to != "5m":
                rule = _RESAMPLE_RULES.get(resample_to)
                if rule:
                    df = df.resample(rule, closed="left", label="left").agg(
                        {"Open": "first", "High": "max",
                         "Low": "min",   "Close": "last", "Volume": "sum"}
                    ).dropna()

            if len(df) >= 20:
                prices[sym] = df

        except Exception as e:
            log(f"  [{sym}] load error: {e}")
            continue

    return prices


# ── signal computation ────────────────────────────────────────────────────────

def compute_signals(df, atr_window=DEFAULT_ATR_WINDOW,
                    vol_window=DEFAULT_VOL_WINDOW,
                    compress_mult=DEFAULT_COMPRESS_MULT,
                    atr_thresh=DEFAULT_ATR_THRESH,
                    vol_mult=DEFAULT_VOL_MULT):
    """
    Compute compression and bullish volume divergence signals bar by bar.

    Returns DataFrame with added columns:
      atr, atr_pct, bar_range, vol_ma,
      is_compressed, is_bullish_vol, signal
    """
    df = df.copy()
    hi, lo, cl, op, vol = (df["High"].values, df["Low"].values,
                            df["Close"].values, df["Open"].values,
                            df["Volume"].values)
    T = len(df)

    # ── ATR (Wilder) ──────────────────────────────────────────────────────────
    tr = np.zeros(T)
    for t in range(1, T):
        tr[t] = max(hi[t] - lo[t],
                    abs(hi[t] - cl[t-1]),
                    abs(lo[t] - cl[t-1]))
    tr[0] = hi[0] - lo[0]

    atr = np.zeros(T)
    atr[atr_window-1] = tr[:atr_window].mean()
    for t in range(atr_window, T):
        atr[t] = (atr[t-1] * (atr_window - 1) + tr[t]) / atr_window

    # ── volume MA ─────────────────────────────────────────────────────────────
    vol_ma = pd.Series(vol).rolling(vol_window, min_periods=max(5, vol_window//4)).mean().values

    # ── bar range ─────────────────────────────────────────────────────────────
    bar_range = hi - lo

    # ── conditions ───────────────────────────────────────────────────────────
    atr_pct       = np.where(cl > 0, atr / cl, np.inf)
    is_compressed = (atr_pct < atr_thresh) & (bar_range < compress_mult * atr)

    prev_close    = np.roll(cl, 1)
    prev_close[0] = cl[0]
    is_bullish_vol = (
        (vol > vol_mult * (vol_ma + EPS)) &   # volume expanding
        (cl > op) &                            # green bar
        (cl > prev_close)                      # closing higher
    )

    # need enough history for both windows
    warm_up = max(atr_window, vol_window)
    signal  = np.zeros(T, dtype=bool)
    signal[warm_up:] = is_compressed[warm_up:] & is_bullish_vol[warm_up:]

    df["atr"]            = atr
    df["atr_pct"]        = atr_pct
    df["bar_range"]      = bar_range
    df["vol_ma"]         = vol_ma
    df["is_compressed"]  = is_compressed
    df["is_bullish_vol"] = is_bullish_vol
    df["signal"]         = signal

    return df


# ── trade simulator ───────────────────────────────────────────────────────────

def run_trades(prices, horizon=DEFAULT_HORIZON,
               profit_target=DEFAULT_PROFIT_TARGET,
               stop_loss=DEFAULT_STOP_LOSS,
               atr_window=DEFAULT_ATR_WINDOW,
               vol_window=DEFAULT_VOL_WINDOW,
               compress_mult=DEFAULT_COMPRESS_MULT,
               atr_thresh=DEFAULT_ATR_THRESH,
               vol_mult=DEFAULT_VOL_MULT,
               train_frac=0.60):
    """
    Run backtest across all tickers. Evaluates on TEST set only
    (bars after the train_frac split point, same convention as covariatec.py).
    """
    trade_rows = []

    for ticker, df in prices.items():
        df = compute_signals(df, atr_window=atr_window, vol_window=vol_window,
                             compress_mult=compress_mult, atr_thresh=atr_thresh,
                             vol_mult=vol_mult)

        T         = len(df)
        split_idx = int(T * train_frac)
        df_test   = df.iloc[split_idx:].reset_index()
        nt        = len(df_test)

        op  = df_test["Open"].values
        hi  = df_test["High"].values
        lo  = df_test["Low"].values
        cl  = df_test["Close"].values
        sig = df_test["signal"].values
        ts  = df_test.iloc[:, 0].values   # timestamp column after reset_index

        for t in range(nt - 1):
            if not sig[t]:
                continue

            entry_bar = t + 1
            if entry_bar >= nt:
                continue

            entry_price = op[entry_bar]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            pt_price = entry_price * (1 + profit_target) if profit_target > 0 else np.inf
            sl_price = entry_price * (1 - stop_loss)     if stop_loss    > 0 else -np.inf

            exit_bar    = min(t + horizon, nt - 1)
            exit_reason = "time_stop"

            for h in range(1, horizon + 1):
                check = entry_bar + h - 1
                if check >= nt:
                    break
                if profit_target > 0 and hi[check] >= pt_price:
                    exit_bar    = check
                    exit_reason = "profit_target"
                    break
                if stop_loss > 0 and lo[check] <= sl_price:
                    exit_bar    = check
                    exit_reason = "stop_loss"
                    break

            if exit_reason == "profit_target":
                exit_price = pt_price
            elif exit_reason == "stop_loss":
                exit_price = sl_price
            else:
                exit_price = cl[exit_bar]

            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            pct_return = (exit_price - entry_price) / (entry_price + EPS) * 100.0

            trade_rows.append({
                "ticker":       ticker,
                "signal_date":  ts[t],
                "entry_date":   ts[entry_bar],
                "entry_price":  round(float(entry_price), 6),
                "exit_date":    ts[exit_bar],
                "exit_price":   round(float(exit_price), 6),
                "exit_reason":  exit_reason,
                "bars_held":    int(exit_bar - entry_bar + 1),
                "atr_pct_sig":  round(float(df_test["atr_pct"].iloc[t]), 6),
                "vol_mult_sig": round(float(df_test["Volume"].iloc[t] /
                                           (df_test["vol_ma"].iloc[t] + EPS)), 3),
                "pct_return":   round(float(pct_return), 4),
            })

    if not trade_rows:
        return pd.DataFrame()
    return pd.DataFrame(trade_rows).sort_values("entry_date").reset_index(drop=True)


# ── stats ─────────────────────────────────────────────────────────────────────

def trade_stats(trades):
    if trades.empty:
        return {}
    r    = trades["pct_return"]
    wins = r[r > 0]
    loss = r[r <= 0]
    cum  = r.cumsum()
    return {
        "n_trades":      len(trades),
        "n_win":         len(wins),
        "n_loss":        len(loss),
        "win_rate":      round(len(wins) / len(trades), 4),
        "mean_ret":      round(float(r.mean()),    4),
        "median_ret":    round(float(r.median()),  4),
        "mean_win":      round(float(wins.mean()), 4) if len(wins) else 0.0,
        "mean_loss":     round(float(loss.mean()), 4) if len(loss) else 0.0,
        "profit_factor": round(float(wins.sum() / (-loss.sum() + EPS)), 3)
                         if len(loss) and loss.sum() < 0 else float("inf"),
        "total_pct":     round(float(r.sum()),    4),
        "cum_peak":      round(float(cum.max()),   4),
        "max_drawdown":  round(float((cum - cum.cummax()).min()), 4),
        "sharpe_proxy":  round(float(r.mean() / (r.std() + EPS)), 4),
    }


def by_symbol(trades):
    rows = []
    for ticker, grp in trades.groupby("ticker"):
        s = trade_stats(grp)
        s["ticker"] = ticker
        rows.append(s)
    return (pd.DataFrame(rows)
              .set_index("ticker")
              .sort_values("total_pct", ascending=False))


def vol_mult_sweep(prices, horizon, profit_target, stop_loss,
                   atr_window, vol_window, compress_mult, atr_thresh, train_frac):
    rows = []
    for vm in TAU_STEPS:
        t = run_trades(prices, horizon=horizon, profit_target=profit_target,
                       stop_loss=stop_loss, atr_window=atr_window,
                       vol_window=vol_window, compress_mult=compress_mult,
                       atr_thresh=atr_thresh, vol_mult=vm, train_frac=train_frac)
        s = trade_stats(t)
        s["vol_mult"] = vm
        rows.append(s)
    return pd.DataFrame(rows).set_index("vol_mult")


# ── print ─────────────────────────────────────────────────────────────────────

def print_summary(stats, horizon, vol_mult, tf, profit_target, stop_loss):
    bm = _BAR_MINUTES.get(tf, 5)
    log(f"\n{'='*65}")
    log("  CRYPTO APNL SUMMARY  [TEST SET ONLY]")
    log(f"{'='*65}")
    if not stats:
        log("  No trades generated."); return
    log(f"  tf={tf}  vol_mult={vol_mult:.2f}  horizon={horizon} bars ({horizon*bm}min)")
    log(f"  profit_target={profit_target*100:.1f}%  stop_loss={stop_loss*100:.1f}%")
    log(f"  Trades       : {stats['n_trades']}  "
        f"W {stats['n_win']}  L {stats['n_loss']}  "
        f"WR {stats['win_rate']*100:.1f}%")
    log(f"  Mean return  : {stats['mean_ret']:+.3f}%   "
        f"(W {stats['mean_win']:+.3f}%  L {stats['mean_loss']:+.3f}%)")
    log(f"  Median ret   : {stats['median_ret']:+.3f}%")
    log(f"  Profit factor: {stats['profit_factor']:.3f}")
    log(f"  Total P&L    : {stats['total_pct']:+.2f}%  (arithmetic sum)")
    log(f"  Peak cum P&L : {stats['cum_peak']:+.2f}%")
    log(f"  Max drawdown : {stats['max_drawdown']:+.2f}%")
    log(f"  Sharpe proxy : {stats['sharpe_proxy']:+.4f}  (mean/std per trade)")


def print_by_symbol(sym_df):
    log(f"\n{'='*65}")
    log("  PER-SYMBOL BREAKDOWN")
    log(f"{'='*65}")
    log(f"  {'Symbol':<14} {'N':>5} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'MaxDD':>8}")
    log(f"  {'─'*62}")
    for ticker, row in sym_df.iterrows():
        log(f"  {str(ticker):<14} {int(row['n_trades']):>5} "
            f"{row['win_rate']*100:>5.1f}% "
            f"{row['mean_ret']:>+9.3f}% "
            f"{row['total_pct']:>+10.2f}% "
            f"{row['profit_factor']:>7.3f} "
            f"{row['max_drawdown']:>+8.2f}%")


def print_vol_sweep(sweep_df, tf):
    bm = _BAR_MINUTES.get(tf, 5)
    log(f"\n{'='*65}")
    log("  VOL MULTIPLIER SWEEP")
    log(f"{'='*65}")
    log(f"  {'vol_mult':>8} {'N':>6} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'Sharpe':>8}")
    log(f"  {'─'*58}")
    for vm, row in sweep_df.iterrows():
        n = row.get("n_trades", 0)
        if not n or not np.isfinite(float(n if n else 0)):
            log(f"  {vm:>8.2f}  —"); continue
        log(f"  {vm:>8.2f} {int(n):>6} "
            f"{row['win_rate']*100:>5.1f}% "
            f"{row['mean_ret']:>+9.3f}% "
            f"{row['total_pct']:>+10.2f}% "
            f"{row['profit_factor']:>7.3f} "
            f"{row['sharpe_proxy']:>+8.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crypto alt backtest: price compression + bullish volume divergence")
    parser.add_argument("--horizon",        type=int,   default=DEFAULT_HORIZON)
    parser.add_argument("--vol-mult",       type=float, default=DEFAULT_VOL_MULT)
    parser.add_argument("--vol-window",     type=int,   default=DEFAULT_VOL_WINDOW)
    parser.add_argument("--atr-window",     type=int,   default=DEFAULT_ATR_WINDOW)
    parser.add_argument("--compress-mult",  type=float, default=DEFAULT_COMPRESS_MULT)
    parser.add_argument("--atr-thresh",     type=float, default=DEFAULT_ATR_THRESH)
    parser.add_argument("--profit-target",  type=float, default=DEFAULT_PROFIT_TARGET)
    parser.add_argument("--stop-loss",      type=float, default=DEFAULT_STOP_LOSS)
    parser.add_argument("--train-frac",     type=float, default=0.60)
    parser.add_argument("--resample-to",    default=None,
                        help=f"Resample 5m bars to: {list(_RESAMPLE_RULES.keys())}")
    parser.add_argument("--symbols",        nargs="*",  default=None,
                        help="e.g. INJ/USDT SUI/USDT  (default: all cached)")
    parser.add_argument("--no-sweep",       action="store_true")
    args = parser.parse_args()

    tf       = args.resample_to if args.resample_to in _RESAMPLE_RULES else "5m"
    bar_mins = _BAR_MINUTES.get(tf, 5)

    log("=" * 65)
    log("  apnl_crypto.py  —  Price Compression + Bullish Volume Divergence")
    log("  CRYPTO / 24-7 / native 5m")
    log("=" * 65)
    log(f"  tf={tf}  horizon={args.horizon} bars ({args.horizon*bar_mins}min)")
    log(f"  vol_mult={args.vol_mult:.2f}x  vol_window={args.vol_window}")
    log(f"  atr_window={args.atr_window}  compress_mult={args.compress_mult:.2f}  "
        f"atr_thresh={args.atr_thresh*100:.1f}%")
    log(f"  profit_target={args.profit_target*100:.1f}%  "
        f"stop_loss={args.stop_loss*100:.1f}%")
    log(f"  train_frac={args.train_frac*100:.0f}%  test on remaining "
        f"{(1-args.train_frac)*100:.0f}%")
    log(f"  cache={CACHE_DIR}/  (no session filter — 24/7)")
    log("=" * 65)

    log(f"\n  Loading prices from {CACHE_DIR}/...")
    prices = load_prices(resample_to=args.resample_to, symbols=args.symbols)
    log(f"  Loaded {len(prices)} symbols")
    if not prices:
        log("  ERROR: no data found. Check crypto_cache/ exists and contains parquets.")
        sys.exit(1)

    if args.symbols:
        missing = [s for s in args.symbols if s not in prices]
        if missing:
            log(f"  WARNING: not found in cache: {missing}")

    log(f"\n  Simulating trades...")
    trades = run_trades(
        prices, horizon=args.horizon,
        profit_target=args.profit_target, stop_loss=args.stop_loss,
        atr_window=args.atr_window, vol_window=args.vol_window,
        compress_mult=args.compress_mult, atr_thresh=args.atr_thresh,
        vol_mult=args.vol_mult, train_frac=args.train_frac,
    )

    stats = trade_stats(trades)
    print_summary(stats, args.horizon, args.vol_mult, tf,
                  args.profit_target, args.stop_loss)

    if not trades.empty:
        sym_df = by_symbol(trades)
        print_by_symbol(sym_df)

        er = trades["exit_reason"].value_counts()
        log(f"\n  Exit reasons:")
        for reason, cnt in er.items():
            log(f"    {reason:<14}: {cnt:>5}  ({100*cnt/len(trades):.1f}%)")

        log(f"\n  Mean return by bars held:")
        for bh, grp in sorted(trades.groupby("bars_held")):
            log(f"    {bh} bar{'s' if bh!=1 else '':<4} "
                f"n={len(grp):>4}  mean={grp['pct_return'].mean():>+7.3f}%  "
                f"({bh*bar_mins}min hold)")

    if not args.no_sweep:
        log(f"\n  Running vol_mult sweep...")
        sweep_df = vol_mult_sweep(
            prices, horizon=args.horizon,
            profit_target=args.profit_target, stop_loss=args.stop_loss,
            atr_window=args.atr_window, vol_window=args.vol_window,
            compress_mult=args.compress_mult, atr_thresh=args.atr_thresh,
            train_frac=args.train_frac,
        )
        print_vol_sweep(sweep_df, tf)
    else:
        sweep_df = pd.DataFrame()

    # ── save ──────────────────────────────────────────────────────────────────
    log(f"\n{'─'*65}")
    if not trades.empty:
        trades.to_csv("crypto_apnl_trades.csv", index=False)
        log("  Saved → crypto_apnl_trades.csv")
        sym_df.to_csv("crypto_apnl_by_symbol.csv")
        log("  Saved → crypto_apnl_by_symbol.csv")
    if not sweep_df.empty:
        sweep_df.to_csv("crypto_apnl_by_threshold.csv")
        log("  Saved → crypto_apnl_by_threshold.csv")

    with open("crypto_apnl_summary.txt", "w") as f:
        f.write("CRYPTO ALT P&L BACKTEST SUMMARY\n")
        f.write(f"strategy       : price_compression + bullish_vol_divergence\n")
        f.write(f"asset_class    : crypto (24/7, no session filter)\n")
        f.write(f"timeframe      : {tf}\n")
        f.write(f"bar_minutes    : {bar_mins}\n")
        f.write(f"vol_mult       : {args.vol_mult}\n")
        f.write(f"vol_window     : {args.vol_window}\n")
        f.write(f"atr_window     : {args.atr_window}\n")
        f.write(f"compress_mult  : {args.compress_mult}\n")
        f.write(f"atr_thresh     : {args.atr_thresh}\n")
        f.write(f"profit_target  : {args.profit_target}\n")
        f.write(f"stop_loss      : {args.stop_loss}\n")
        f.write(f"horizon        : {args.horizon} bars\n")
        f.write(f"train_frac     : {args.train_frac}\n")
        f.write(f"evaluation_set : TEST ONLY\n")
        for k, v in stats.items():
            f.write(f"{k:<16}: {v}\n")
    log("  Saved → crypto_apnl_summary.txt")
    log("\n  [DONE]")


if __name__ == "__main__":
    main()