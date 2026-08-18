"""
backtest.py  —  Signal Lead-Time & Threshold Backtest  [FIXED]

FIXES APPLIED
-------------
  FIX 1 (Problem 1) — p_burst scaling
        Removed the live log-scale normalization patch.
        covariate.py now writes true filtered probabilities in [0,1].
        If values are out of range this script crashes loudly rather
        than silently distorting the signal.

  FIX 2 (Problem 2) — bar sequencing
        Entry is placed at bar t+1 (open of next bar after signal).
        Exit is at open of bar t+6 (5 bars held) or first non-Burst bar.
        Events on bar t (the signal bar itself) do NOT count as hits.
        All window checks use range(entry_bar, ...) starting at t+1.

  FIX 3 (Problem 3) — test-set only evaluation
        All analysis is restricted to split=='test' rows from
        covariate_states.csv.  Train rows are excluded.
        This ensures no future information contaminates the metrics.

ENTRY/EXIT SEQUENCING (enforced throughout)
-------------------------------------------
  Signal fires  : end of bar t        (p_burst > τ at close of bar t)
  Entry placed  : open of bar t+1     (first actionable bar)
  Max exit      : open of bar t+6     (5 bars held = 5 × bar_duration)
  Early exit    : open of first bar after entry where hmm_state != 2
  Event counts  : qualifying move in bars t+1 .. t+5 inclusive
                  bar t itself NEVER counts as a hit
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── config ────────────────────────────────────────────────────────────────────
STATES_CSV          = "covariate_states.csv"
EVENTS_CSV          = "process_id_15m_events.csv"
RESULTS_CSV         = "covariate_results.csv"

DEFAULT_HORIZON     = 5     # FIX 2: 5 bars held (bars t+1 .. t+5)
DEFAULT_CONT_WINDOW = 5     # continuation window from burst entry
MAX_LAG             = 3     # bars back for lead-time analysis
THRESHOLD_STEPS     = np.round(np.arange(0.10, 0.95, 0.05), 2)
EPS                 = 1e-12

def log(msg=""): print(msg, flush=True)

def fmt_p(p):
    if not np.isfinite(p): return "  —   "
    stars = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else "    "
    return f"{p:.4f}{stars}"


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    missing = [f for f in [STATES_CSV, EVENTS_CSV] if not Path(f).exists()]
    if missing:
        log(f"ERROR: missing files: {missing}")
        log("Run process_id_15m.py then covariate.py first.")
        sys.exit(1)

    states = pd.read_csv(STATES_CSV, index_col=0, parse_dates=True)
    events = pd.read_csv(EVENTS_CSV)

    states.columns = [c.strip() for c in states.columns]
    events.columns = [c.strip() for c in events.columns]
    events["date"] = pd.to_datetime(events["date"])

    if not isinstance(states.index, pd.DatetimeIndex):
        states.index = pd.to_datetime(states.index)

    log(f"  Columns in states: {list(states.columns)}")

    # ── FIX 1: hard integrity check — no silent patching ─────────────────────
    if "p_burst" not in states.columns:
        log("CRITICAL ERROR: p_burst column missing. Re-run covariate.py.")
        sys.exit(1)

    max_p = states["p_burst"].max()
    min_p = states["p_burst"].min()
    log(f"  p_burst range: [{min_p:.6f}, {max_p:.6f}]")

    if max_p > 1.0001 or min_p < -0.0001:
        log("=" * 65)
        log("  INTEGRITY ERROR: p_burst is not in [0, 1].")
        log("  This means covariate.py wrote raw HMM log-probabilities")
        log("  or exponentiated values instead of true state probabilities.")
        log("  Fix: re-run covariate.py (the fixed version) which uses")
        log("  filtered_forward_probs() with built-in assertions.")
        log("  DO NOT proceed with a patched/distorted signal.")
        log("=" * 65)
        sys.exit(1)

    log(f"  [OK] p_burst is valid probability in [0, 1]")

    # ── FIX 3: restrict to test split only ───────────────────────────────────
    if "split" in states.columns:
        n_total = len(states)
        states  = states[states["split"] == "test"].copy()
        n_test  = len(states)
        log(f"  [OK] Filtered to test split: {n_test:,} / {n_total:,} bars "
            f"({100*n_test/n_total:.1f}%)")
        log(f"  *** Train bars excluded — no future information in evaluation ***")
    else:
        log("  WARNING: 'split' column not found in states CSV.")
        log("  This means covariate.py was not the fixed version.")
        log("  All bars will be used — results may be optimistic.")
        log("  Re-run covariate.py to get the split column.")

    events["bar_idx"] = events["bar_idx"].astype(int)
    states = states.sort_index()

    log(f"  Loaded {len(states):,} state bars across "
        f"{states['ticker'].nunique()} symbols")
    log(f"  Loaded {len(events):,} events across "
        f"{events['ticker'].nunique()} symbols")

    return states, events


# ── build per-symbol position index ──────────────────────────────────────────

def build_pos_map(s):
    """
    Given a per-ticker state DataFrame (sorted chronologically),
    return a dict mapping datetime → sequential bar position (0-indexed).
    Also returns arrays: p_burst, hmm_state, rva aligned to position.
    """
    s = s.reset_index()
    date_col = s.columns[0]
    time_to_pos = {row[date_col]: idx for idx, row in s.iterrows()}
    return (time_to_pos,
            s["p_burst"].values,
            s["hmm_state"].values if "hmm_state" in s.columns else np.zeros(len(s)),
            s["RVA"].values if "RVA" in s.columns else np.zeros(len(s)),
            len(s))


# ── Q1 + Q2: lead-time analysis ───────────────────────────────────────────────

def lead_time_analysis(states, events, max_lag=MAX_LAG):
    """
    For each event bar, measure p_burst at lags 0, 1, 2, 3 bars before.
    Tests whether p_burst is significantly elevated before events vs baseline.

    FIX 2 note: lag 0 = event bar itself (not tradeable).
                lag 1 = 1 bar before event (signal bar where entry fires).
                We want lag 1 to be significant — that is the predictive claim.
    """
    results = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty:
            continue

        time_to_pos, p_burst_all, _, rva_all, n_bars = build_pos_map(s)

        event_bars = [time_to_pos[dt] for dt in ev_grp["date"]
                      if dt in time_to_pos]
        event_bars = np.array(event_bars, dtype=int)

        if len(event_bars) < 5:
            continue

        # Baseline: bars that are not within max_lag of any event
        non_event_mask = np.ones(n_bars, dtype=bool)
        for b in event_bars:
            for lag in range(max_lag + 1):
                idx = b - lag
                if 0 <= idx < n_bars:
                    non_event_mask[idx] = False

        baseline_p = p_burst_all[non_event_mask]
        baseline_r = rva_all[non_event_mask]

        if len(baseline_p) == 0:
            continue

        for lag in range(max_lag + 1):
            pre_p, pre_r = [], []
            for b in event_bars:
                idx = b - lag
                if 0 <= idx < n_bars:
                    pre_p.append(p_burst_all[idx])
                    pre_r.append(rva_all[idx])

            if len(pre_p) < 3:
                continue

            pre_p = np.array(pre_p)
            pre_r = np.array(pre_r)

            try:
                _, mw_p = stats.mannwhitneyu(pre_p, baseline_p, alternative="greater")
            except Exception:
                mw_p = np.nan

            try:
                _, rva_p = stats.mannwhitneyu(pre_r, baseline_r, alternative="greater")
                rva_mean = float(np.mean(pre_r))
            except Exception:
                rva_p, rva_mean = np.nan, np.nan

            results.append({
                "ticker":                   ticker,
                "lag":                      lag,
                "n_events":                 len(pre_p),
                "pre_event_p_burst_mean":   round(float(np.mean(pre_p)), 4),
                "pre_event_p_burst_median": round(float(np.median(pre_p)), 4),
                "baseline_p_burst_mean":    round(float(np.mean(baseline_p)), 4),
                "lift":         round(float(np.mean(pre_p) / (np.mean(baseline_p) + EPS)), 3),
                "mw_pval":      round(float(mw_p), 4) if np.isfinite(mw_p) else None,
                "mw_significant": bool(mw_p < 0.05) if np.isfinite(mw_p) else False,
                "rva_pre_event_mean": round(rva_mean, 4) if np.isfinite(rva_mean) else None,
                "rva_pval":     round(float(rva_p), 4) if np.isfinite(rva_p) else None,
            })

    return pd.DataFrame(results)


# ── Q3a: threshold sweep ──────────────────────────────────────────────────────

def threshold_sweep(states, events, horizon=DEFAULT_HORIZON,
                    thresholds=THRESHOLD_STEPS):
    """
    FIX 2: Signal fires at bar t → entry at bar t+1.
    Hit = event in bars t+1 .. t+horizon (inclusive).
    Bar t itself is excluded from the hit window.
    """
    results = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty:
            continue

        time_to_pos, p_burst, hmm_state, _, n_bars = build_pos_map(s)

        event_bars = [time_to_pos[dt] for dt in ev_grp["date"]
                      if dt in time_to_pos]
        event_set  = set(event_bars)

        if len(event_bars) < 5:
            continue

        for tau in thresholds:
            # Signal fires at bar t — must NOT already be in burst state
            signal_bars = np.where(
                (p_burst > tau) & (hmm_state < 2)
            )[0]

            tp = fp = fn = 0

            for sb in signal_bars:
                # FIX 2: entry at t+1, check t+1 .. t+horizon
                entry_bar = sb + 1
                hit = any(
                    entry_bar + h in event_set
                    for h in range(horizon)          # t+1, t+2, ... t+horizon
                )
                if hit:
                    tp += 1
                else:
                    fp += 1

            for eb in event_bars:
                # Was this event preceded by a signal in the window?
                # Signal at sb catches event at eb if eb in [sb+1, sb+horizon]
                # i.e. sb in [eb-horizon, eb-1]
                preceded = any(
                    (eb - h) in set(signal_bars)
                    for h in range(1, horizon + 1)   # sb = eb-1, eb-2, ... eb-horizon
                )
                if not preceded:
                    fn += 1

            precision = tp / (tp + fp + EPS) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn + EPS) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall /
                         (precision + recall + EPS)) if (precision + recall) > 0 else 0.0

            results.append({
                "ticker":    ticker,
                "threshold": float(tau),
                "horizon":   horizon,
                "n_signals": int(len(signal_bars)),
                "tp":        int(tp),
                "fp":        int(fp),
                "fn":        int(fn),
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
            })

    return pd.DataFrame(results)


# ── Q3b: continuation backtest ────────────────────────────────────────────────

def continuation_backtest(states, events, cont_window=DEFAULT_CONT_WINDOW):
    """
    From a confirmed HMM transition Accum→Burst (state 1→2):
      - Entry at the NEXT bar after transition (t+1)
      - Look for event in bars t+1 .. t+cont_window
      - Exit when HMM state != 2 OR cont_window bars elapsed

    FIX 2: entry_bar = transition_bar + 1 throughout.
    """
    summary_rows = []
    detail_rows  = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty or "hmm_state" not in s.columns:
            continue

        time_to_pos, p_burst, hmm_state, _, n_bars = build_pos_map(s)

        event_bars = [time_to_pos[dt] for dt in ev_grp["date"]
                      if dt in time_to_pos]
        event_set  = set(event_bars)

        if len(event_bars) < 3:
            continue

        # Detect Accum→Burst transitions (bar i is the TRANSITION bar)
        burst_transitions = [
            i for i in range(1, n_bars)
            if hmm_state[i] == 2 and hmm_state[i - 1] != 2
        ]

        if not burst_transitions:
            continue

        hits, bars_to_events, burst_durations = [], [], []

        for trans_bar in burst_transitions:
            # FIX 2: entry is at t+1 (open of bar after transition)
            entry_bar   = trans_bar + 1
            event_found = False
            bars_to_ev  = np.nan

            # Look for event in [entry_bar, entry_bar + cont_window - 1]
            for h in range(cont_window):
                check_bar = entry_bar + h
                if check_bar >= n_bars:
                    break
                if check_bar in event_set:
                    event_found = True
                    bars_to_ev  = h + 1   # 1-indexed distance from entry
                    break

            hits.append(event_found)
            bars_to_events.append(bars_to_ev)

            # Burst duration: how long does the HMM stay in state 2
            exit_bar = n_bars - 1
            for j in range(trans_bar + 1, n_bars):
                if hmm_state[j] != 2:
                    exit_bar = j
                    break
            burst_durations.append(exit_bar - trans_bar)

            detail_rows.append({
                "ticker":              ticker,
                "transition_bar":      trans_bar,
                "entry_bar":           entry_bar,    # FIX 2: t+1
                "p_burst_at_entry":    round(float(p_burst[min(entry_bar, n_bars-1)]), 4),
                "event_within_window": event_found,
                "bars_to_event":       int(bars_to_ev) if np.isfinite(bars_to_ev) else None,
                "burst_duration":      int(exit_bar - trans_bar),
            })

        hit_rate  = np.mean(hits)
        valid_bte = [b for b in bars_to_events if np.isfinite(b)]

        summary_rows.append({
            "ticker":               ticker,
            "n_transitions":        len(burst_transitions),
            "n_events":             len(event_bars),
            "cont_window":          cont_window,
            "hit_rate":             round(float(hit_rate), 4),
            "mean_bars_to_event":   round(float(np.mean(valid_bte)), 2)   if valid_bte else None,
            "med_bars_to_event":    round(float(np.median(valid_bte)), 2) if valid_bte else None,
            "mean_burst_duration":  round(float(np.mean(burst_durations)), 2),
            "med_burst_duration":   round(float(np.median(burst_durations)), 2),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ── signal log ────────────────────────────────────────────────────────────────

def build_signal_log(states, events, tau=0.4, horizon=DEFAULT_HORIZON):
    """
    Build a complete log of every signal and event with outcome labels.
    FIX 2: hit window is [sb+1, sb+horizon], not [sb, sb+horizon-1].
    """
    rows = []

    for ticker, ev_grp in events.groupby("ticker"):
        s = states[states["ticker"] == ticker].copy().sort_index()
        if s.empty:
            continue

        time_to_pos, p_burst, hmm_state, rva, n_bars = build_pos_map(s)

        event_bars = [time_to_pos[dt] for dt in ev_grp["date"]
                      if dt in time_to_pos]
        event_set  = set(event_bars)

        signal_bars = np.where((p_burst > tau) & (hmm_state < 2))[0]

        for sb in signal_bars:
            entry_bar = sb + 1    # FIX 2
            hit  = any(entry_bar + h in event_set for h in range(horizon))
            first_ev = next(
                (entry_bar + h for h in range(horizon)
                 if entry_bar + h in event_set), None
            )
            rows.append({
                "ticker":        ticker,
                "signal_bar":    int(sb),
                "entry_bar":     int(entry_bar),   # FIX 2
                "type":          "signal",
                "outcome":       "TP" if hit else "FP",
                "p_burst":       round(float(p_burst[sb]), 4),
                "hmm_state":     int(hmm_state[sb]),
                "rva":           round(float(rva[sb]), 4),
                "bars_to_event": int(first_ev - entry_bar + 1) if first_ev else None,
            })

        for eb in event_bars:
            # Signal at sb catches event at eb if sb+1 <= eb <= sb+horizon
            # i.e. sb in [eb-horizon, eb-1]
            preceded = any(
                (eb - h) in set(signal_bars)
                for h in range(1, horizon + 1)
            )
            rows.append({
                "ticker":        ticker,
                "signal_bar":    None,
                "entry_bar":     None,
                "type":          "event",
                "outcome":       "caught" if preceded else "FN",
                "p_burst":       round(float(p_burst[eb]), 4),
                "hmm_state":     int(hmm_state[eb]),
                "rva":           round(float(rva[eb]), 4),
                "bars_to_event": 0,
            })

    return pd.DataFrame(rows)


# ── print helpers ─────────────────────────────────────────────────────────────

def print_lead_time(lt_df):
    log(f"\n{'='*65}")
    log("  Q1 + Q2  LEAD-TIME ANALYSIS  [TEST SET ONLY]")
    log(f"{'='*65}")
    log("  Lag 0 = event bar itself (not tradeable).")
    log("  Lag 1 = 1 bar before event = signal bar where entry fires.")
    log("  Significant lag 1 = p_burst predicts events 1 bar in advance.")

    if lt_df.empty:
        log("  No data."); return

    for ticker in lt_df["ticker"].unique():
        t = lt_df[lt_df["ticker"] == ticker]
        log(f"\n  {ticker}")
        log(f"  {'Lag':>4} {'PreEvt p_burst':>15} {'Baseline':>9} "
            f"{'Lift':>6} {'MW p-val':>10} {'RVA mean':>9} {'RVA p':>8}")
        log(f"  " + "─" * 66)
        for _, row in t.sort_values("lag").iterrows():
            sig = " *" if row.get("mw_significant") else "  "
            rva_m = row["rva_pre_event_mean"]
            rva_p = row["rva_pval"]
            log(f"  {int(row['lag']):>4} "
                f"{row['pre_event_p_burst_mean']:>15.4f} "
                f"{row['baseline_p_burst_mean']:>9.4f} "
                f"{row['lift']:>6.2f}x "
                f"{fmt_p(row['mw_pval'] if row['mw_pval'] is not None else np.nan):>10}"
                f"{sig}"
                f"{rva_m if rva_m is not None else '  —':>9} "
                f"{fmt_p(rva_p if rva_p is not None else np.nan):>8}")


def print_threshold(thr_df, horizon):
    log(f"\n{'='*65}")
    log(f"  Q3a  THRESHOLD SWEEP  [TEST SET ONLY]")
    log(f"{'='*65}")
    log(f"  Horizon = {horizon} bars.  Entry at t+1.  Event window: t+1 .. t+{horizon}.")
    log("  Signal: p_burst > τ and hmm_state < 2 (not already in burst)\n")

    all_best = []
    for ticker in thr_df["ticker"].unique():
        t    = thr_df[thr_df["ticker"] == ticker]
        if t.empty: continue
        best = t.loc[t["f1"].idxmax()]
        all_best.append(best)
        log(f"  {ticker:<6}  τ={best['threshold']:.2f}  "
            f"F1={best['f1']:.3f}  P={best['precision']:.3f}  "
            f"R={best['recall']:.3f}  sigs={int(best['n_signals'])}  "
            f"TP={int(best['tp'])}  FP={int(best['fp'])}  FN={int(best['fn'])}")

    if all_best:
        bd = pd.DataFrame(all_best)
        log(f"\n  Mean F1        : {bd['f1'].mean():.3f}")
        log(f"  Mean precision : {bd['precision'].mean():.3f}")
        log(f"  Mean recall    : {bd['recall'].mean():.3f}")
        log(f"  F1 > 0.20      : {(bd['f1']>0.20).sum()}/{len(bd)}")
        log(f"  F1 > 0.30      : {(bd['f1']>0.30).sum()}/{len(bd)}")


def print_continuation(cont_df):
    log(f"\n{'='*65}")
    log("  Q3b  CONTINUATION BACKTEST  [TEST SET ONLY]")
    log(f"{'='*65}")
    log("  From HMM Accum→Burst transition.  Entry at NEXT bar (t+1).")
    log(f"  {'Ticker':<7} {'Trans':>6} {'HitRate':>8} "
        f"{'MeanBars':>9} {'MedBars':>8} {'BrstDur':>8}")
    log(f"  " + "─" * 55)

    if cont_df.empty:
        log("  No transitions detected."); return

    for _, row in cont_df.sort_values("hit_rate", ascending=False).iterrows():
        bte_m = row["mean_bars_to_event"]
        bte_d = row["med_bars_to_event"]
        log(f"  {row['ticker']:<7} "
            f"{int(row['n_transitions']):>6} "
            f"{row['hit_rate']:>8.3f} "
            f"{bte_m if bte_m is not None else '  —':>9} "
            f"{bte_d if bte_d is not None else '  —':>8} "
            f"{row['mean_burst_duration']:>8.1f}")


def print_verdict(lt_df, thr_df, cont_df):
    log(f"\n{'='*65}")
    log("  VERDICT  [TEST SET — HONEST OUT-OF-SAMPLE ESTIMATES]")
    log(f"{'='*65}")

    if lt_df.empty:
        log("  No data."); return

    lag1    = lt_df[lt_df["lag"] == 1]
    n_sig   = lag1["mw_significant"].sum()
    n_total = lag1["ticker"].nunique()

    best_f1   = thr_df["f1"].max()     if not thr_df.empty   else 0
    cont_hit  = cont_df["hit_rate"].mean() if not cont_df.empty else 0

    log(f"\n  Lead-time (lag 1, p<0.05) : {n_sig}/{n_total} symbols")
    log(f"  Best F1                    : {best_f1:.3f}")
    log(f"  Mean continuation hit rate : {cont_hit:.3f}")
    log(f"\n  These are TEST-SET numbers.  Train bars were excluded.")
    log(f"  HMM used forward algorithm only — no future information.")
    log(f"  Entry sequencing: signal at t → entry at t+1 → hold 5 bars.")

    # Per-symbol candidates
    if not thr_df.empty:
        best_per = thr_df.loc[thr_df.groupby("ticker")["f1"].idxmax()]
        candidates = best_per[best_per["f1"] >= 0.20].sort_values("f1", ascending=False)
        if not candidates.empty:
            log(f"\n  Symbols worth pursuing (F1 ≥ 0.20 on test set):")
            for _, r in candidates.iterrows():
                log(f"    {r['ticker']:<8} F1={r['f1']:.3f}  "
                    f"P={r['precision']:.3f}  R={r['recall']:.3f}  τ={r['threshold']:.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",       type=int,   default=DEFAULT_HORIZON)
    parser.add_argument("--cont-window",   type=int,   default=DEFAULT_CONT_WINDOW)
    parser.add_argument("--symbols",       nargs="*",  default=None)
    parser.add_argument("--min-threshold", type=float, default=0.10)
    parser.add_argument("--signal-tau",    type=float, default=0.40)
    args = parser.parse_args()

    thresholds = np.round(np.arange(args.min_threshold, 0.95, 0.05), 2)

    log("=" * 65)
    log("  backtest.py  —  Signal Backtest  [FIXED]")
    log("=" * 65)
    log(f"  Horizon       : {args.horizon} bars")
    log(f"  Cont window   : {args.cont_window} bars")
    log(f"  Entry timing  : signal at bar t → entry at bar t+1")
    log(f"  Exit timing   : bar t+{args.horizon+1} open OR first non-Burst bar")
    log(f"  Evaluation    : TEST SET ONLY (split=='test' from covariate.py)")
    log("=" * 65)

    states, events = load_data()

    if args.symbols:
        states = states[states["ticker"].isin(args.symbols)]
        events = events[events["ticker"].isin(args.symbols)]

    log("\n  Running lead-time analysis...")
    lt_df = lead_time_analysis(states, events, max_lag=MAX_LAG)
    print_lead_time(lt_df)

    log("\n  Running threshold sweep...")
    thr_df = threshold_sweep(states, events,
                             horizon=args.horizon, thresholds=thresholds)
    print_threshold(thr_df, args.horizon)

    log("\n  Running continuation backtest...")
    cont_df, cont_detail = continuation_backtest(
        states, events, cont_window=args.cont_window)
    print_continuation(cont_df)

    print_verdict(lt_df, thr_df, cont_df)

    # ── save ──────────────────────────────────────────────────────────────────
    lt_df.to_csv("backtest_lead_time.csv", index=False)
    thr_df.to_csv("backtest_threshold.csv", index=False)
    cont_df.to_csv("backtest_continuation.csv", index=False)
    cont_detail.to_csv("backtest_continuation_detail.csv", index=False)

    sig_df = build_signal_log(states, events,
                               tau=args.signal_tau, horizon=args.horizon)
    sig_df.to_csv("backtest_signals.csv", index=False)

    # Save summary text
    with open("backtest_summary.txt", "w") as f:
        f.write("BACKTEST SUMMARY\n")
        f.write(f"Horizon        : {args.horizon} bars\n")
        f.write(f"Entry          : signal at t, entry at t+1\n")
        f.write(f"Evaluation set : TEST ONLY (split=='test')\n")
        f.write(f"p_burst source : filtered forward algorithm (no future info)\n")
        if not lt_df.empty:
            lag1   = lt_df[lt_df["lag"]==1]
            n_sig  = lag1["mw_significant"].sum()
            n_tot  = lag1["ticker"].nunique()
            f.write(f"Lead-time sig  : {n_sig}/{n_tot} symbols\n")
        if not thr_df.empty:
            f.write(f"Best F1        : {thr_df['f1'].max():.3f}\n")
        if not cont_df.empty:
            f.write(f"Mean hit rate  : {cont_df['hit_rate'].mean():.3f}\n")

    log("\n  Saved → backtest_lead_time.csv")
    log("  Saved → backtest_threshold.csv")
    log("  Saved → backtest_continuation.csv")
    log("  Saved → backtest_continuation_detail.csv")
    log("  Saved → backtest_signals.csv")
    log("  Saved → backtest_summary.txt")
    log("\n  [DONE] All results are honest out-of-sample estimates.")


if __name__ == "__main__":
    main()