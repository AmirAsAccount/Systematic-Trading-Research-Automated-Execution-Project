"""
pnlc.py  —  P&L Backtest for Crypto Signal  [TEST SET ONLY]

Reads covariatec.py v4 outputs (rich 5-feature HMM + Cox pipeline).
Prices come from crypto_cache/ parquets.

TRADE MECHANICS
---------------
  Signal   : p_burst > τ  AND  hmm_state < 2  at close of bar t
             + optional rich-feature filters (--sparse-entry)
  Entry    : Open of bar t+1
  Exit (standard) : Close of first event bar in [t+1, t+horizon]
                    else Close[t+horizon]  (time stop)
  Exit (hmm_exit) : Close when hmm_state drops below 2
                    OR event fires OR max-bars cap
  Return   : (exit_price - entry_price) / entry_price × 100  (%)
  Sizing   : equal weight, arithmetic accumulation — no compounding

SIGNAL FEATURES (from covariatec.py v4 states CSV)
---------------------------------------------------
  p_burst     — filtered forward P(Burst state)
  hmm_state   — 0=Quiet  1=Accum  2=Burst
  vol_ratio   — volume / 8h rolling mean
  vol_zscore  — vol_ratio deviation in std units
  RVA         — volume acceleration (diff of vol_ratio)
  RVA_vel     — acceleration of acceleration
  ret_signed  — directional bar return (bullish = positive)
  persist     — bars continuously in current state

ENTRY FILTERS (--sparse-entry)
-------------------------------
  hmm_state == 1            strictly Accumulation
  vol_ratio  > --vol-ratio-min   (default 3.0)
  vol_zscore > --vol-zscore-min  (default 1.0)
  RVA_vel    > --rva-vel-min     (default 0.0, i.e. accelerating)
  ret_signed > --ret-signed-min  (default 0.0, i.e. green bar)

OUTPUTS
-------
  crypto_pnl_trades.csv
  crypto_pnl_by_symbol.csv
  crypto_pnl_by_threshold.csv
  crypto_pnl_summary.txt
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
STATES_CSV      = "crypto_covariate_states.csv"
EVENTS_CSV      = "crypto_events.csv"
CACHE_DIR       = Path("crypto_cache")

DEFAULT_HORIZON = 5
DEFAULT_TAU     = 0.40
TAU_STEPS       = np.round(np.arange(0.10, 0.91, 0.05), 2)
EPS             = 1e-12

# Pandas resample rules for each supported timeframe
TF_RESAMPLE = {
    "5m":  "5min",  "10m": "10min", "15m": "15min",
    "30m": "30min", "45m": "45min", "1h":  "1h",
    "2h":  "2h",    "3h":  "3h",    "4h":  "4h",
    "1d":  "1D",
}

def log(msg=""): print(msg, flush=True)


# ── price loader ──────────────────────────────────────────────────────────────

def load_prices(resample_to=None, source_tf="5m"):
    prices = {}
    if not CACHE_DIR.exists():
        return prices

    source_tag = f"_{source_tf}_"
    candidates = [p for p in CACHE_DIR.glob("*.parquet")
                  if source_tag in p.name and ".partial." not in p.name]

    if not candidates:
        candidates = [p for p in CACHE_DIR.glob("*.parquet")
                      if ".partial." not in p.name]

    for path in candidates:
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            needed = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                      if c in df.columns]
            if len(needed) < 2:
                continue
            df = df[needed].dropna()

            if resample_to and resample_to != source_tf:
                rule = TF_RESAMPLE.get(resample_to)
                if rule is None:
                    log(f"  WARNING: unknown resample target '{resample_to}' — skipping resample")
                else:
                    agg = {}
                    if "Open"   in df.columns: agg["Open"]   = "first"
                    if "High"   in df.columns: agg["High"]   = "max"
                    if "Low"    in df.columns: agg["Low"]    = "min"
                    if "Close"  in df.columns: agg["Close"]  = "last"
                    if "Volume" in df.columns: agg["Volume"] = "sum"
                    df = df.resample(rule).agg(agg).dropna()

            parts = path.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0] + "/" + parts[1]
            else:
                continue

            if len(df) >= 10:
                prices[symbol] = df.sort_index()

        except Exception:
            continue

    return prices


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(tf_tag=None):
    def _resolve(base, tag):
        if tag:
            tagged = Path(base.replace(".csv", f"_{tag}.csv"))
            if tagged.exists():
                return tagged
        return Path(base)

    states_path = _resolve(STATES_CSV, tf_tag)
    events_path = _resolve(EVENTS_CSV, tf_tag)

    missing = [str(f) for f in [states_path, events_path] if not f.exists()]
    if missing:
        log(f"ERROR: missing files: {missing}")
        log("Run covariatec.py first.")
        sys.exit(1)

    states = pd.read_csv(states_path, index_col=0, parse_dates=True)
    events = pd.read_csv(events_path)

    states.columns = [c.strip() for c in states.columns]
    events.columns = [c.strip() for c in events.columns]
    events["date"] = pd.to_datetime(events["date"])

    if not isinstance(states.index, pd.DatetimeIndex):
        states.index = pd.to_datetime(states.index)

    if "p_burst" not in states.columns:
        log("CRITICAL ERROR: p_burst column missing. Re-run covariatec.py.")
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

    if "symbol" in states.columns and "ticker" not in states.columns:
        states = states.rename(columns={"symbol": "ticker"})
    if "symbol" in events.columns and "ticker" not in events.columns:
        events = events.rename(columns={"symbol": "ticker"})

    states = states.sort_index()
    events["bar_idx"] = events["bar_idx"].astype(int)

    log(f"  {states['ticker'].nunique()} symbols  |  "
        f"{len(states):,} state bars  |  {len(events):,} events")

    return states, events


# ── core trade simulator ──────────────────────────────────────────────────────

def run_trades(states, events, prices, tau=DEFAULT_TAU, horizon=DEFAULT_HORIZON,
               sparse_entry=False, hmm_exit=False,
               vol_ratio_min=3.0, vol_zscore_min=1.0,
               rva_vel_min=0.0, ret_signed_min=0.0,
               hmm_exit_max_bars=None):
    """
    Simulate every trade triggered by p_burst > tau.

    Reads rich feature columns from covariatec.py v4 states CSV:
      vol_ratio, vol_zscore, RVA, RVA_vel, ret_signed, persist

    Entry  : Open[t+1]

    Exit (standard) : Close of first event bar in [t+1 .. t+horizon]
                      OR Close[t+horizon]  (time stop)

    Exit (hmm_exit) : Close when hmm_state drops below 2  (burst over)
                      OR event fires  OR hmm_exit_max_bars safety cap

    sparse_entry=True — all rich-feature filters applied:
      hmm_state == 1         (strictly Accumulation)
      vol_ratio  > vol_ratio_min
      vol_zscore > vol_zscore_min
      RVA_vel    > rva_vel_min   (volume acceleration still rising)
      ret_signed > ret_signed_min  (directional bar return)
    """
    if hmm_exit_max_bars is None:
        hmm_exit_max_bars = max(horizon * 4, 20)

    trade_rows = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty:
            continue

        cache_sym = ticker
        if cache_sym in prices:
            px = prices[cache_sym]
            s_idx = s.index
            if s_idx.tz is None:
                s_idx = s_idx.tz_localize("UTC")
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
            log(f"  [{ticker}] no price data available — skipping")
            continue

        p_burst   = s["p_burst"].values
        hmm_state = s["hmm_state"].values if "hmm_state" in s.columns                     else np.zeros(len(s))
        dates     = s.index
        n         = len(s)

        # Rich feature arrays — fall back gracefully if column absent
        def _col(name, default):
            return s[name].values if name in s.columns                    else np.full(n, default)

        vol_ratio  = _col("vol_ratio",  np.inf)
        vol_zscore = _col("vol_zscore", np.inf)
        rva_vel    = _col("RVA_vel",    np.inf)
        ret_signed = _col("ret_signed", np.inf)

        time_to_pos = {dt: i for i, dt in enumerate(dates)}
        event_bars  = set(
            time_to_pos[dt] for dt in ev_grp["date"] if dt in time_to_pos
        )

        for t in range(n - 1):
            if p_burst[t] <= tau:
                continue

            # ── sparse / high-conviction entry (rich feature filters) ──────
            if sparse_entry:
                if hmm_state[t] != 1:           continue  # must be Accum
                if vol_ratio[t]  < vol_ratio_min:  continue
                if vol_zscore[t] < vol_zscore_min: continue
                if rva_vel[t]    < rva_vel_min:    continue  # still accelerating
                if ret_signed[t] <= ret_signed_min: continue  # green bar
            else:
                if hmm_state[t] >= 2:
                    continue

            # Base directional filter: ret_signed from states CSV (preferred)
            # falls back to price-computed bar return if col absent
            if "ret_signed" in s.columns:
                if ret_signed[t] <= 0:
                    continue
            else:
                bar_ret = (close_prices[t] - open_prices[t]) / (open_prices[t] + EPS)
                if bar_ret <= 0:
                    continue

            entry_bar = t + 1
            if entry_bar >= n:
                continue

            entry_price = open_prices[entry_bar]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            # ── Path 2A: HMM-state-aware exit ─────────────────────────────
            if hmm_exit:
                exit_bar  = min(t + hmm_exit_max_bars, n - 1)   # safety cap
                event_hit = False
                for h in range(1, hmm_exit_max_bars + 1):
                    check = entry_bar + h - 1
                    if check >= n:
                        exit_bar = n - 1
                        break
                    # Event fires → exit immediately at that bar's close
                    if check in event_bars:
                        exit_bar  = check
                        event_hit = True
                        break
                    # HMM burst regime ended → exit at close of this bar
                    if hmm_state[check] < 2:
                        exit_bar = check
                        break
                exit_reason = "event" if event_hit else \
                              ("hmm_exit" if exit_bar < min(t + hmm_exit_max_bars, n - 1)
                               else "max_bars_cap")
            else:
                # ── Standard fixed-horizon exit ────────────────────────────
                exit_bar  = min(t + horizon, n - 1)
                event_hit = False
                for h in range(horizon):
                    check = entry_bar + h
                    if check >= n:
                        break
                    if check in event_bars:
                        exit_bar  = check
                        event_hit = True
                        break
                exit_reason = "event" if event_hit else "time_stop"

            exit_price = close_prices[exit_bar]
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            pct_return = (exit_price - entry_price) / (entry_price + EPS) * 100.0

            trade_rows.append({
                "ticker":        ticker,
                "signal_date":   dates[t],
                "entry_date":    dates[entry_bar],
                "entry_price":   round(float(entry_price), 8),
                "exit_date":     dates[exit_bar],
                "exit_price":    round(float(exit_price), 8),
                "exit_reason":   exit_reason,
                "bars_held":     int(exit_bar - entry_bar + 1),
                "p_burst_sig":   round(float(p_burst[t]), 4),
                "hmm_state_sig": int(hmm_state[t]),
                # rich feature snapshot at signal bar
                "vol_ratio_sig":  round(float(vol_ratio[t]),  4)
                                  if np.isfinite(vol_ratio[t])  else None,
                "vol_zscore_sig": round(float(vol_zscore[t]), 4)
                                  if np.isfinite(vol_zscore[t]) else None,
                "rva_vel_sig":    round(float(rva_vel[t]),    6)
                                  if np.isfinite(rva_vel[t])    else None,
                "ret_signed_sig": round(float(ret_signed[t]), 6)
                                  if np.isfinite(ret_signed[t]) else None,
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
        "total_pct":    round(float(r.sum()),  4),
        "cum_peak":     round(float(cum.max()), 4),
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


def tau_sweep(states, events, prices, horizon, sparse_entry=False,
              hmm_exit=False, vol_ratio_min=3.0, vol_zscore_min=1.0,
              rva_vel_min=0.0, ret_signed_min=0.0, hmm_exit_max_bars=None):
    rows = []
    for tau in TAU_STEPS:
        t = run_trades(states, events, prices, tau=tau, horizon=horizon,
                       sparse_entry=sparse_entry, hmm_exit=hmm_exit,
                       vol_ratio_min=vol_ratio_min,
                       vol_zscore_min=vol_zscore_min,
                       rva_vel_min=rva_vel_min,
                       ret_signed_min=ret_signed_min,
                       hmm_exit_max_bars=hmm_exit_max_bars)
        s = trade_stats(t)
        s["tau"] = tau
        rows.append(s)
    return pd.DataFrame(rows).set_index("tau")


# ── print ─────────────────────────────────────────────────────────────────────

def print_summary(stats, horizon, tau, tf_label="", mode_label=""):
    log(f"\n{'='*65}")
    log(f"  P&L SUMMARY  [TEST SET ONLY]{tf_label}{mode_label}")
    log(f"{'='*65}")
    if not stats:
        log("  No trades generated."); return
    log(f"  τ={tau:.2f}  horizon={horizon} bars")
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


def print_tau_sweep(tau_df):
    log(f"\n{'='*65}")
    log("  τ SWEEP")
    log(f"{'='*65}")
    log(f"  {'τ':>5} {'N':>6} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'Sharpe':>8}")
    log(f"  {'─'*55}")
    for tau, row in tau_df.iterrows():
        n = row.get("n_trades", 0)
        if not n or not np.isfinite(n):
            log(f"  {tau:>5.2f}  —"); continue
        log(f"  {tau:>5.2f} {int(n):>6} "
            f"{row['win_rate']*100:>5.1f}% "
            f"{row['mean_ret']:>+9.3f}% "
            f"{row['total_pct']:>+10.2f}% "
            f"{row['profit_factor']:>7.3f} "
            f"{row['sharpe_proxy']:>+8.4f}")


# ── multi-timeframe sweep ─────────────────────────────────────────────────────

def run_tf_sweep(states, events, source_tf, timeframes, tau, horizon,
                 sparse_entry=False, hmm_exit=False,
                 vol_ratio_min=3.0, vol_zscore_min=1.0,
                 rva_vel_min=0.0, ret_signed_min=0.0,
                 hmm_exit_max_bars=None):
    log(f"\n{'='*65}")
    log(f"  TIMEFRAME SWEEP  (states from {source_tf} | τ={tau:.2f} | horizon={horizon} bars)")
    log(f"{'='*65}")
    log(f"  {'TF':<6} {'N':>6} {'WR%':>6} {'MeanRet':>9} "
        f"{'TotalPct':>10} {'PF':>7} {'Sharpe':>8} {'MaxDD':>9}")
    log(f"  {'─'*60}")

    tf_rows = []
    for tf in timeframes:
        prices_tf = load_prices(resample_to=tf, source_tf=source_tf)
        trades_tf = run_trades(states, events, prices_tf, tau=tau, horizon=horizon,
                               sparse_entry=sparse_entry, hmm_exit=hmm_exit,
                               vol_ratio_min=vol_ratio_min,
                               vol_zscore_min=vol_zscore_min,
                               rva_vel_min=rva_vel_min,
                               ret_signed_min=ret_signed_min,
                               hmm_exit_max_bars=hmm_exit_max_bars)
        s = trade_stats(trades_tf)
        if not s:
            log(f"  {tf:<6}  — (no trades)")
            continue
        log(f"  {tf:<6} {int(s['n_trades']):>6} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{s['mean_ret']:>+9.3f}% "
            f"{s['total_pct']:>+10.2f}% "
            f"{s['profit_factor']:>7.3f} "
            f"{s['sharpe_proxy']:>+8.4f} "
            f"{s['max_drawdown']:>+9.2f}%")
        tf_rows.append({"tf": tf, **s})

    return pd.DataFrame(tf_rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",          type=int,   default=DEFAULT_HORIZON)
    parser.add_argument("--tau",              type=float, default=DEFAULT_TAU)
    parser.add_argument("--symbols",          nargs="*",  default=None)
    parser.add_argument("--no-tau-sweep",     action="store_true")
    parser.add_argument("--source-tf",        default="5m")
    parser.add_argument("--resample-to",      default=None)
    parser.add_argument("--tf-sweep",         action="store_true")
    parser.add_argument("--tf-tag",           default=None)
    # Path 1 — sparse/high-conviction entry (rich feature filters)
    parser.add_argument("--sparse-entry",     action="store_true",
                        help="Require hmm_state==1 + all rich-feature thresholds.")
    parser.add_argument("--vol-ratio-min",    type=float, default=3.0,
                        help="Min vol_ratio for --sparse-entry (default: 3.0).")
    parser.add_argument("--vol-zscore-min",   type=float, default=1.0,
                        help="Min vol_zscore for --sparse-entry (default: 1.0).")
    parser.add_argument("--rva-vel-min",      type=float, default=0.0,
                        help="Min RVA_vel (vol accel still rising) for --sparse-entry "
                             "(default: 0.0).")
    parser.add_argument("--ret-signed-min",   type=float, default=0.0,
                        help="Min ret_signed for --sparse-entry (default: 0.0 = any green).")
    # Path 2A — HMM-state-aware exit
    parser.add_argument("--hmm-exit",         action="store_true",
                        help="Exit when hmm_state drops below 2 (burst over) "
                             "instead of a fixed time stop.")
    parser.add_argument("--hmm-exit-max-bars", type=int, default=None,
                        help="Safety cap for --hmm-exit (default: max(horizon*4, 20)).")
    args = parser.parse_args()

    active_tf  = args.resample_to or args.source_tf
    tf_label   = f"  [{active_tf} bars]" if args.resample_to else ""

    mode_parts = []
    if args.sparse_entry:
        mode_parts.append(
            f"sparse-entry("
            f"vr>{args.vol_ratio_min} "
            f"vz>{args.vol_zscore_min} "
            f"rv>{args.rva_vel_min} "
            f"rs>{args.ret_signed_min})")
    if args.hmm_exit:
        cap = args.hmm_exit_max_bars or max(args.horizon * 4, 20)
        mode_parts.append(f"hmm-exit(cap={cap})")
    mode_label = ("  [" + " + ".join(mode_parts) + "]") if mode_parts else ""

    log("=" * 65)
    log("  pnlc.py  —  P&L Backtest  [CRYPTO / TEST SET]")
    log("=" * 65)
    log(f"  τ={args.tau:.2f}  horizon={args.horizon} bars")
    log(f"  Entry: Open[t+1]  |  Exit: Close[event] or Close[t+{args.horizon}]")
    if args.sparse_entry:
        log(f"  SPARSE ENTRY: hmm_state==1 "
            f"vol_ratio>{args.vol_ratio_min} "
            f"vol_zscore>{args.vol_zscore_min} "
            f"RVA_vel>{args.rva_vel_min} "
            f"ret_signed>{args.ret_signed_min}")
    if args.hmm_exit:
        cap = args.hmm_exit_max_bars or max(args.horizon * 4, 20)
        log(f"  Path 2A HMM EXIT: hold while hmm_state==2, cap={cap} bars")
    if args.resample_to:
        log(f"  Source TF: {args.source_tf}  →  Resampled to: {args.resample_to}")
    if args.tf_sweep:
        log(f"  Mode: TIMEFRAME SWEEP across all supported TFs")
    log("=" * 65)

    states, events = load_data(tf_tag=args.tf_tag)

    if args.symbols:
        states = states[states["ticker"].isin(args.symbols)]
        events = events[events["ticker"].isin(args.symbols)]

    kw = dict(sparse_entry=args.sparse_entry, hmm_exit=args.hmm_exit,
              vol_ratio_min=args.vol_ratio_min,
              vol_zscore_min=args.vol_zscore_min,
              rva_vel_min=args.rva_vel_min,
              ret_signed_min=args.ret_signed_min,
              hmm_exit_max_bars=args.hmm_exit_max_bars)

    if args.tf_sweep:
        all_tfs = list(TF_RESAMPLE.keys())
        run_tf_sweep(states, events, args.source_tf, all_tfs,
                     args.tau, args.horizon, **kw)
        log("\n  [DONE]")
        return

    log(f"\n  Loading prices from {CACHE_DIR}/  "
        f"[source: {args.source_tf}"
        f"{' → ' + args.resample_to if args.resample_to else ''}]...")
    prices = load_prices(resample_to=args.resample_to, source_tf=args.source_tf)
    log(f"  Loaded price data for {len(prices)} symbols")

    log(f"\n  Simulating trades (τ={args.tau:.2f})...")
    trades = run_trades(states, events, prices, tau=args.tau, horizon=args.horizon,
                        **kw)

    stats = trade_stats(trades)
    print_summary(stats, args.horizon, args.tau, tf_label, mode_label)

    out_tag = f"_{active_tf}" if args.resample_to else ""

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
                f"n={len(grp):>4}  mean={grp['pct_return'].mean():>+7.3f}%")

    if not args.no_tau_sweep:
        log(f"\n  Running τ sweep...")
        tau_df = tau_sweep(states, events, prices, args.horizon, **kw)
        print_tau_sweep(tau_df)
    else:
        tau_df = pd.DataFrame()

    log(f"\n{'─'*65}")
    if not trades.empty:
        trades.to_csv(f"crypto_pnl_trades{out_tag}.csv", index=False)
        log(f"  Saved → crypto_pnl_trades{out_tag}.csv")
        sym_df.to_csv(f"crypto_pnl_by_symbol{out_tag}.csv")
        log(f"  Saved → crypto_pnl_by_symbol{out_tag}.csv")
    if not tau_df.empty:
        tau_df.to_csv(f"crypto_pnl_by_threshold{out_tag}.csv")
        log(f"  Saved → crypto_pnl_by_threshold{out_tag}.csv")

    with open(f"crypto_pnl_summary{out_tag}.txt", "w") as f:
        f.write("CRYPTO P&L BACKTEST SUMMARY\n")
        f.write(f"source_tf      : {args.source_tf}\n")
        f.write(f"active_tf      : {active_tf}\n")
        f.write(f"tau            : {args.tau}\n")
        f.write(f"horizon        : {args.horizon} bars\n")
        f.write(f"entry          : Open[t+1]\n")
        f.write(f"exit           : Close[event] OR Close[t+horizon]\n")
        f.write(f"sparse_entry   : {args.sparse_entry}\n")
        f.write(f"vol_ratio_min  : {args.vol_ratio_min}\n")
        f.write(f"vol_zscore_min : {args.vol_zscore_min}\n")
        f.write(f"rva_vel_min    : {args.rva_vel_min}\n")
        f.write(f"ret_signed_min : {args.ret_signed_min}\n")
        f.write(f"hmm_exit       : {args.hmm_exit}\n")
        f.write(f"hmm_exit_cap   : {args.hmm_exit_max_bars or max(args.horizon*4,20)}\n")
        f.write(f"evaluation set : TEST ONLY\n")
        for k, v in stats.items():
            f.write(f"{k:<16}: {v}\n")
    log(f"  Saved → crypto_pnl_summary{out_tag}.txt")
    log("\n  [DONE]")


if __name__ == "__main__":
    main()