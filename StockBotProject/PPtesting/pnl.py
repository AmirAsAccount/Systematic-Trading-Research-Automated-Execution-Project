"""
pnl.py  —  P&L Backtest for Stock Signal  [TEST SET ONLY]

Reads covariate.py outputs. Prices come from cache/ parquets (15m bars).

TRADE MECHANICS
---------------
  Signal   : p_burst > τ  AND  hmm_state < 2  at close of bar t
             AND signal bar is a green candle (Close > Open)
  Entry    : Open of bar t+1
  Exit     : Close of event bar if event fires in [t+1, t+horizon]
             else Close of bar t+horizon  (time stop)
  Return   : (exit_price - entry_price) / entry_price × 100  (%)
  Sizing   : equal weight, arithmetic accumulation — no compounding

BAR DURATION
------------
  15-minute bars. Default horizon=5 bars = 75 minutes.
  Adjust with --horizon N.

OUTPUTS
-------
  pnl_trades.csv
  pnl_by_symbol.csv
  pnl_by_threshold.csv
  pnl_summary.txt
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
STATES_CSV      = "covariate_states.csv"
EVENTS_CSV      = "process_id_15m_events.csv"
CACHE_DIR       = Path("cache")

DEFAULT_HORIZON = 5
DEFAULT_TAU     = 0.40
TAU_STEPS       = np.round(np.arange(0.10, 0.91, 0.05), 2)
EPS             = 1e-12

# Session hours (UTC) — same as covariate.py
SESSION_START_UTC = 14 * 60 + 30   # 09:30 ET
SESSION_END_UTC   = 21 * 60        # 16:00 ET

def log(msg=""): print(msg, flush=True)


# ── price loader ──────────────────────────────────────────────────────────────

def _is_session(ts):
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    m   = utc.hour * 60 + utc.minute
    return SESSION_START_UTC <= m < SESSION_END_UTC


def load_prices():
    """
    Load 15m OHLCV from cache/ parquets (same files covariate.py uses).
    Merges 1h historical + 15m recent exactly as load_bars() does.
    Returns dict: ticker -> DataFrame(index=timestamp, Open, Close)
    """
    prices = {}
    if not CACHE_DIR.exists():
        log(f"  WARNING: {CACHE_DIR}/ not found — no price data available")
        return prices

    tickers = set()
    for p in CACHE_DIR.glob("*_1h.parquet"):
        tickers.add(p.stem.replace("_1h", ""))
    for p in CACHE_DIR.glob("*_15m_recent.parquet"):
        tickers.add(p.stem.replace("_15m_recent", ""))

    from datetime import datetime, timedelta

    for ticker in tickers:
        path_1h  = CACHE_DIR / f"{ticker}_1h.parquet"
        path_15m = CACHE_DIR / f"{ticker}_15m_recent.parquet"

        chunks = []

        for path in [path_1h, path_15m]:
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                needed = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                          if c in df.columns]
                if len(needed) < 4:
                    continue
                df = df[needed].dropna()
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df = df[df.index.map(_is_session)]
                if not df.empty:
                    chunks.append(df)
            except Exception:
                continue

        if not chunks:
            continue

        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        if len(combined) >= 20:
            prices[ticker] = combined[["Open", "Close"]].sort_index()

    return prices


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    missing = [f for f in [STATES_CSV, EVENTS_CSV] if not Path(f).exists()]
    if missing:
        log(f"ERROR: missing files: {missing}")
        log("Run covariate.py first.")
        sys.exit(1)

    states = pd.read_csv(STATES_CSV, index_col=0, parse_dates=True)
    events = pd.read_csv(EVENTS_CSV)

    states.columns = [c.strip() for c in states.columns]
    events.columns = [c.strip() for c in events.columns]
    events["date"] = pd.to_datetime(events["date"])

    if not isinstance(states.index, pd.DatetimeIndex):
        states.index = pd.to_datetime(states.index)

    if "p_burst" not in states.columns:
        log("CRITICAL ERROR: p_burst column missing. Re-run covariate.py.")
        sys.exit(1)

    p_max, p_min = states["p_burst"].max(), states["p_burst"].min()
    if p_max > 1.0001 or p_min < -0.0001:
        log(f"INTEGRITY ERROR: p_burst out of [0,1] [{p_min:.4f}, {p_max:.4f}].")
        sys.exit(1)

    if "split" in states.columns:
        n_total = len(states)
        states  = states[states["split"] == "test"].copy()
        log(f"  Filtered to test split: {len(states):,} / {n_total:,} bars "
            f"({100*len(states)/n_total:.1f}%)")
    else:
        log("  WARNING: no 'split' column — using all bars (may be optimistic)")

    states = states.sort_index()
    events["bar_idx"] = events["bar_idx"].astype(int)

    log(f"  {states['ticker'].nunique()} symbols  |  "
        f"{len(states):,} state bars  |  {len(events):,} events")

    return states, events


# ── core trade simulator ──────────────────────────────────────────────────────

def run_trades(states, events, prices, tau=DEFAULT_TAU, horizon=DEFAULT_HORIZON):
    """
    Simulate every trade triggered by p_burst > tau on the test set.

    Entry  : Open[t+1]
    Exit   : Close of first event bar in [t+1 .. t+horizon]
             OR Close[t+horizon] if no event fires  (time stop)

    Directional filter: signal bar must close bullish (green candle).
    """
    trade_rows = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty:
            continue

        # ── resolve price series ──────────────────────────────────────────────
        if ticker in prices:
            px = prices[ticker]
            s_idx = s.index
            if s_idx.tz is None:
                s_idx = s_idx.tz_localize("UTC")
            elif str(s_idx.tz) != "UTC":
                s_idx = s_idx.tz_convert("UTC")
            px = px.reindex(s_idx, method=None)
            open_prices  = px["Open"].values
            close_prices = px["Close"].values
        elif "Close" in s.columns and "Open" in s.columns:
            open_prices  = s["Open"].values
            close_prices = s["Close"].values
        elif "Close" in s.columns:
            open_prices  = s["Close"].values
            close_prices = s["Close"].values
        else:
            log(f"  [{ticker}] no price data — skipping")
            continue

        p_burst   = s["p_burst"].values
        hmm_state = s["hmm_state"].values if "hmm_state" in s.columns \
                    else np.zeros(len(s))
        dates     = s.index
        n         = len(s)

        time_to_pos = {dt: i for i, dt in enumerate(dates)}
        event_bars  = set(
            time_to_pos[dt] for dt in ev_grp["date"] if dt in time_to_pos
        )

        for t in range(n - 1):
            if p_burst[t] <= tau:
                continue
            if hmm_state[t] >= 2:
                continue

            # Directional filter: signal bar must be a green candle
            if not (np.isfinite(open_prices[t]) and np.isfinite(close_prices[t])
                    and open_prices[t] > 0):
                continue
            bar_ret = (close_prices[t] - open_prices[t]) / (open_prices[t] + EPS)
            if bar_ret <= 0:
                continue

            entry_bar = t + 1
            if entry_bar >= n:
                continue

            entry_price = open_prices[entry_bar]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            # Find exit
            exit_bar  = min(t + horizon, n - 1)   # time stop
            event_hit = False

            for h in range(horizon):
                check = entry_bar + h
                if check >= n:
                    break
                if check in event_bars:
                    exit_bar  = check
                    event_hit = True
                    break

            exit_price = close_prices[exit_bar]
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            pct_return = (exit_price - entry_price) / (entry_price + EPS) * 100.0

            trade_rows.append({
                "ticker":        ticker,
                "signal_date":   dates[t],
                "entry_date":    dates[entry_bar],
                "entry_price":   round(float(entry_price), 6),
                "exit_date":     dates[exit_bar],
                "exit_price":    round(float(exit_price), 6),
                "exit_reason":   "event" if event_hit else "time_stop",
                "bars_held":     int(exit_bar - entry_bar + 1),
                "p_burst_sig":   round(float(p_burst[t]), 4),
                "pct_return":    round(float(pct_return), 4),
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
        "n_trades":     len(trades),
        "n_win":        len(wins),
        "n_loss":       len(loss),
        "win_rate":     round(len(wins) / len(trades), 4),
        "mean_ret":     round(float(r.mean()),    4),
        "median_ret":   round(float(r.median()),  4),
        "mean_win":     round(float(wins.mean()), 4) if len(wins)  else 0.0,
        "mean_loss":    round(float(loss.mean()), 4) if len(loss)  else 0.0,
        "profit_factor":round(float(wins.sum() / (-loss.sum() + EPS)), 3)
                        if len(loss) and loss.sum() < 0 else float("inf"),
        "total_pct":    round(float(r.sum()),   4),
        "cum_peak":     round(float(cum.max()),  4),
        "max_drawdown": round(float((cum - cum.cummax()).min()), 4),
        "sharpe_proxy": round(float(r.mean() / (r.std() + EPS)), 4),
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


def tau_sweep(states, events, prices, horizon):
    rows = []
    for tau in TAU_STEPS:
        t = run_trades(states, events, prices, tau=tau, horizon=horizon)
        s = trade_stats(t)
        s["tau"] = tau
        rows.append(s)
    return pd.DataFrame(rows).set_index("tau")


# ── print ─────────────────────────────────────────────────────────────────────

def print_summary(stats, horizon, tau):
    log(f"\n{'='*65}")
    log("  P&L SUMMARY  [TEST SET ONLY]")
    log(f"{'='*65}")
    if not stats:
        log("  No trades generated."); return
    log(f"  τ={tau:.2f}  horizon={horizon} bars (={horizon*15}min)")
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
    log(f"  {'Symbol':<8} {'N':>5} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'MaxDD':>8}")
    log(f"  {'─'*56}")
    for ticker, row in sym_df.iterrows():
        log(f"  {str(ticker):<8} {int(row['n_trades']):>5} "
            f"{row['win_rate']*100:>5.1f}% "
            f"{row['mean_ret']:>+9.3f}% "
            f"{row['total_pct']:>+10.2f}% "
            f"{row['profit_factor']:>7.3f} "
            f"{row['max_drawdown']:>+8.2f}%")


def print_tau_sweep(tau_df):
    log(f"\n{'='*65}")
    log("  τ SWEEP")
    log(f"{'='*65}")
    log(f"  {'τ':>5} {'N':>6} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'Sharpe':>8}")
    log(f"  {'─'*55}")
    for tau, row in tau_df.iterrows():
        n = row.get("n_trades", 0)
        if not n or not np.isfinite(float(n if n else 0)):
            log(f"  {tau:>5.2f}  —"); continue
        log(f"  {tau:>5.2f} {int(n):>6} "
            f"{row['win_rate']*100:>5.1f}% "
            f"{row['mean_ret']:>+9.3f}% "
            f"{row['total_pct']:>+10.2f}% "
            f"{row['profit_factor']:>7.3f} "
            f"{row['sharpe_proxy']:>+8.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",      type=int,   default=DEFAULT_HORIZON)
    parser.add_argument("--tau",          type=float, default=DEFAULT_TAU)
    parser.add_argument("--symbols",      nargs="*",  default=None)
    parser.add_argument("--no-tau-sweep", action="store_true")
    args = parser.parse_args()

    log("=" * 65)
    log("  pnl.py  —  P&L Backtest  [STOCKS / 15m bars / TEST SET]")
    log("=" * 65)
    log(f"  τ={args.tau:.2f}  horizon={args.horizon} bars ({args.horizon*15}min)")
    log(f"  Entry : Open[t+1]  |  Exit: Close[event] or Close[t+{args.horizon}]")
    log(f"  Filter: green signal bar required (bullish directional filter)")
    log("=" * 65)

    states, events = load_data()

    log(f"\n  Loading prices from {CACHE_DIR}/...")
    prices = load_prices()
    log(f"  Loaded price data for {len(prices)} symbols from cache")

    if args.symbols:
        states = states[states["ticker"].isin(args.symbols)]
        events = events[events["ticker"].isin(args.symbols)]

    log(f"\n  Simulating trades (τ={args.tau:.2f})...")
    trades = run_trades(states, events, prices, tau=args.tau, horizon=args.horizon)

    stats = trade_stats(trades)
    print_summary(stats, args.horizon, args.tau)

    if not trades.empty:
        sym_df = by_symbol(trades)
        print_by_symbol(sym_df)

        er = trades["exit_reason"].value_counts()
        log(f"\n  Exit reasons:")
        for reason, cnt in er.items():
            log(f"    {reason:<12}: {cnt:>5}  ({100*cnt/len(trades):.1f}%)")

        log(f"\n  Mean return by bars held:")
        for bh, grp in sorted(trades.groupby("bars_held")):
            log(f"    {bh} bar{'s' if bh!=1 else '':<4} "
                f"n={len(grp):>4}  mean={grp['pct_return'].mean():>+7.3f}%  "
                f"({bh*15}min hold)")

    if not args.no_tau_sweep:
        log(f"\n  Running τ sweep...")
        tau_df = tau_sweep(states, events, prices, args.horizon)
        print_tau_sweep(tau_df)
    else:
        tau_df = pd.DataFrame()

    # ── save ──────────────────────────────────────────────────────────────────
    log(f"\n{'─'*65}")
    if not trades.empty:
        trades.to_csv("pnl_trades.csv", index=False)
        log("  Saved → pnl_trades.csv")
        sym_df.to_csv("pnl_by_symbol.csv")
        log("  Saved → pnl_by_symbol.csv")
    if not tau_df.empty:
        tau_df.to_csv("pnl_by_threshold.csv")
        log("  Saved → pnl_by_threshold.csv")

    with open("pnl_summary.txt", "w") as f:
        f.write("STOCK P&L BACKTEST SUMMARY\n")
        f.write(f"bar_duration   : 15 minutes\n")
        f.write(f"tau            : {args.tau}\n")
        f.write(f"horizon        : {args.horizon} bars ({args.horizon*15}min)\n")
        f.write(f"entry          : Open[t+1]\n")
        f.write(f"exit           : Close[event] OR Close[t+horizon]\n")
        f.write(f"filter         : green signal bar (bullish only)\n")
        f.write(f"evaluation set : TEST ONLY\n")
        for k, v in stats.items():
            f.write(f"{k:<16}: {v}\n")
    log("  Saved → pnl_summary.txt")
    log("\n  [DONE]")


if __name__ == "__main__":
    main()