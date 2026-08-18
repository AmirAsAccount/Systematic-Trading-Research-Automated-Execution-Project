"""
backtest.py  —  Online Walk-Forward Simulator for the Dynamic
            Cox-Modulated Poisson / 2-Layer Latent Markov model.

ARCHITECTURE (matches covariate.py's DYNAMIC model)
-----------------------------------------------------
  covariate.py is an OFFLINE FITTER: it produces an initial model bundle
  per ticker (covariate_models/covariate_model_<TICKER>.npz) containing:
    - emission params (means, covars, startprob) for the Gaussian HMM
    - the static transmat_ (used as the W initialization reference)
    - W            : (n_states, n_states, n_features) dynamic transition weights
    - betas        : (n_states, n_features) per-state Cox intensity coefficients
    - x_mean_dyn / x_std_dyn   : normalization for the 5-dim dynamic feature vector
    - x_mean_emit / x_std_emit : normalization for the 2-dim emission vector
                                 (vol_ratio, abs_ret) — BUG 4 FIX from covariate.py,
                                 MUST be used here, not recomputed.
    - perm, split_idx, n_states, vol_window

  backtest.py is the ONLINE SIMULATOR. For each ticker, it:
    1. Loads the model bundle.
    2. Loads the full feature sequence (train+test) from covariate_states.csv
       (train portion only used for context/continuity of persist counters —
       see WALK-FORWARD LOOP below) and recomputes features directly from
       cached bars if the dynamic feature columns are missing/stale, falling
       back gracefully.
    3. Walks the TEST split bar-by-bar in chronological order:
         a. Runs ONE step of the forward algorithm using the CURRENT
            (possibly already-updated) W and emission params to get
            p_burst(t), p_accum(t), p_quiet(t) — this is what gets recorded
            as the signal at bar t.
         b. Records the signal (exactly the existing p_burst array format
            that lead_time_analysis / threshold_sweep / continuation_backtest
            / build_signal_log already expect — those functions are NOT
            modified, just fed this online-generated p_burst sequence).
         c. AFTER locking in the prediction, observes whether an event
            occurred at bar t (this is legitimate — bar t has already
            closed by the time we use this information to update for t+1).
         d. Performs ONE online gradient step updating W (the row for the
            currently-departed source state) and beta_{s_t} (the currently
            active state's Cox coefficients), using ONLY this single
            observation. No future bar is ever touched.
         e. Advances to t+1, carrying forward the updated W and betas.
    4. Logs the full parameter trajectory (W_norm, beta norms, lambda_t,
       event_occurred per bar) to backtest_param_trajectory.csv — this is
       the diagnostic file that lets you check whether the online updates
       are converging, oscillating, or drifting.
    5. Feeds the resulting full (train-smoothed + test-online) p_burst
       sequence into the UNCHANGED lead_time_analysis / threshold_sweep /
       continuation_backtest / build_signal_log functions, preserving their
       exact entry/exit bar sequencing (signal at t -> entry at t+1 -> hold
       up to horizon bars; bar t itself never counts as a hit).
    6. Writes all the original output CSVs with identical schemas, plus the
       new backtest_param_trajectory.csv.

ENTRY/EXIT SEQUENCING (unchanged from the static reference backtest.py —
do not modify; this logic lives in lead_time_analysis / threshold_sweep /
continuation_backtest / build_signal_log verbatim)
-------------------------------------------------------------------------
  Signal fires  : end of bar t        (p_burst > tau at close of bar t)
  Entry placed  : open of bar t+1     (first actionable bar)
  Max exit      : open of bar t+6     (5 bars held = 5 x bar_duration)
  Early exit    : open of first bar after entry where hmm_state != 2
  Event counts  : qualifying move in bars t+1 .. t+5 inclusive
                  bar t itself NEVER counts as a hit

WHY THE ONLINE UPDATE IS NOT LEAKAGE
-------------------------------------
  The update at step (d) uses only the outcome of bar t, observed once bar
  t has closed — exactly the information a live system would have before
  generating bar t+1's prediction. No future bar is ever used to tune the
  model that produced bar t's own prediction. This is standard sequential
  online learning, not retroactive parameter tuning.
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# config
STATES_CSV           = "covariate_states.csv"
EVENTS_CSV            = "process_id_15m_events.csv"
RESULTS_CSV           = "covariate_results.csv"
MODEL_DIR             = Path("covariate_models")

DEFAULT_HORIZON       = 5     # 5 bars held (bars t+1 .. t+5)
DEFAULT_CONT_WINDOW   = 5     # continuation window from burst entry
MAX_LAG               = 3     # bars back for lead-time analysis
THRESHOLD_STEPS       = np.round(np.arange(0.10, 0.95, 0.05), 2)
EPS                   = 1e-12
LAMBDA_CLIP           = (1e-6, 100.0)
X_T_CLIP              = 8.0   # clip standardized dynamic features to +/-8 std devs
                              # (BUG 6 fix — see run_online_walkforward)

DYN_FEATURE_COLS = ["RVA", "RVA_vel", "persist", "vol_zscore", "ret_signed"]
EMIT_FEATURE_COLS = ["vol_ratio", "abs_ret"]

# Default online learning rates — UNVALIDATED, see spec §9. Exposed as CLI
# args (--lr-w, --lr-beta, --lr-decay) so they can be tuned without code
# changes. Use backtest_param_trajectory.csv to diagnose:
#   W_norm / beta_norm exploding  -> lower the relevant lr
#   W_norm / beta_norm flat       -> raise the relevant lr
DEFAULT_LR_W      = 0.01
DEFAULT_LR_BETA   = 0.005
DEFAULT_LR_DECAY  = 0.001
PERSIST_CAP       = 50

def log(msg=""): print(msg, flush=True)

def fmt_p(p):
    if not np.isfinite(p): return "  -   "
    stars = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else "    "
    return f"{p:.4f}{stars}"


# model bundle loading

def load_model_bundle(ticker):
    """
    Load the .npz model bundle written by covariate.py.
    Returns a dict of plain numpy arrays / python scalars, or None if
    the bundle doesn't exist (caller should skip the ticker).
    """
    path = MODEL_DIR / f"covariate_model_{ticker}.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        bundle = {k: data[k] for k in data.files}
        # unwrap 0-d arrays back to python scalars
        for key in ("split_idx", "n_states", "vol_window"):
            if key in bundle:
                bundle[key] = int(bundle[key])
        return bundle
    except Exception as e:
        log(f"  [{ticker}] failed to load model bundle: {e}")
        return None


# data loading

def load_data():
    missing = [f for f in [STATES_CSV, EVENTS_CSV] if not Path(f).exists()]
    if missing:
        log(f"ERROR: missing files: {missing}")
        log("Run process_id_15m.py then covariate.py first.")
        sys.exit(1)

    if not MODEL_DIR.exists():
        log(f"ERROR: {MODEL_DIR}/ not found.")
        log("Run covariate.py first to generate model bundles.")
        sys.exit(1)

    states = pd.read_csv(STATES_CSV, index_col=0, parse_dates=True)
    events = pd.read_csv(EVENTS_CSV)

    states.columns = [c.strip() for c in states.columns]
    events.columns = [c.strip() for c in events.columns]
    events["date"] = pd.to_datetime(events["date"])

    if not isinstance(states.index, pd.DatetimeIndex):
        states.index = pd.to_datetime(states.index)

    log(f"  Columns in states: {list(states.columns)}")

    required_dyn_cols = DYN_FEATURE_COLS + EMIT_FEATURE_COLS
    missing_cols = [c for c in required_dyn_cols if c not in states.columns]
    if missing_cols:
        log(f"  WARNING: states CSV missing columns {missing_cols}.")
        log(f"  These are required for the online walk-forward loop.")
        log(f"  Re-run the dynamic covariate.py (writes RVA_vel, persist,")
        log(f"  vol_zscore, ret_signed) if this list is non-empty.")

    if "split" not in states.columns:
        log("  ERROR: 'split' column not found in states CSV.")
        log("  Re-run covariate.py to get the split column.")
        sys.exit(1)

    n_total = len(states)
    n_test  = (states["split"] == "test").sum()
    log(f"  Loaded {n_total:,} total bars ({n_test:,} test bars) across "
        f"{states['ticker'].nunique()} symbols")
    log(f"  Loaded {len(events):,} events across "
        f"{events['ticker'].nunique()} symbols")

    states = states.sort_index()
    return states, events


# online walk-forward primitives

def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / (e.sum() + EPS)


def _emission_log_likelihood(obs_std, means, covars):
    """
    Diagonal-Gaussian log-likelihood of a single standardized observation
    (2-dim: vol_ratio, abs_ret) under each of n_states emission distributions.
    means, covars: (n_states, 2). Returns (n_states,) log-likelihoods.
    """
    n_states = means.shape[0]
    ll = np.zeros(n_states)
    for k in range(n_states):
        var = np.clip(covars[k], EPS, None)
        diff = obs_std - means[k]
        ll[k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var)
    return ll


def _dynamic_transition_row(W_i, x_t):
    """A[i, :](t) = softmax(W_i . x_t).  W_i: (n_states, n_features)."""
    logits = W_i @ x_t
    return _softmax(logits)


def run_online_walkforward(ticker, bundle, df_feat_full, event_bars_set,
                           lr_w=DEFAULT_LR_W, lr_beta=DEFAULT_LR_BETA,
                           lr_decay=DEFAULT_LR_DECAY):
    """
    Walk the TEST split bar-by-bar, running the dynamic forward algorithm
    and online parameter updates exactly as specified.

    Parameters
    ----------
    ticker          : str
    bundle          : dict from load_model_bundle()
    df_feat_full    : DataFrame, full train+test features for this ticker,
                      sorted chronologically, containing at minimum
                      EMIT_FEATURE_COLS + DYN_FEATURE_COLS + 'hmm_state'
                      (train-split hmm_state, from covariate.py, used to
                      seed the online state belief at the train/test
                      boundary).
    event_bars_set  : set of int bar indices (within df_feat_full's full
                      0-indexed range) that are events.

    Returns
    -------
    p_burst_full    : ndarray (T,) — train portion copied from the
                      covariate.py-computed smoothed values (unchanged,
                      offline), test portion filled with the online
                      filtered values generated by this loop.
    p_accum_full, p_quiet_full : same shape, same convention.
    hmm_state_full  : ndarray (T,) int — decoded state per bar (train as
                      given, test as argmax of the online filtered probs).
    trajectory_rows : list of dicts, one per TEST bar, for
                      backtest_param_trajectory.csv.
    """
    n_states    = bundle["n_states"]
    split_idx   = bundle["split_idx"]
    emit_means  = bundle["emit_means"].copy()     # (n_states, 2), standardized space
    emit_covars = bundle["emit_covars"].copy()    # (n_states, 2)
    startprob   = bundle["startprob"].copy()
    W           = bundle["W"].copy()              # (n_states, n_states, n_features)
    betas       = bundle["betas"].copy()           # (n_states, n_features)
    x_mean_dyn  = bundle["x_mean_dyn"]
    x_std_dyn   = bundle["x_std_dyn"]
    x_mean_emit = bundle["x_mean_emit"]            # BUG 4 FIX consumed here
    x_std_emit  = bundle["x_std_emit"]

    T = len(df_feat_full)
    if split_idx >= T:
        log(f"  [{ticker}] split_idx >= T, no test bars to simulate")
        return (np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan),
                np.full(T, -1), [])

    p_burst_full = np.full(T, np.nan)
    p_accum_full = np.full(T, np.nan) if n_states == 3 else None
    p_quiet_full = np.full(T, np.nan)
    hmm_state_full = np.full(T, -1, dtype=int)

    # train portion: copy through whatever covariate.py already computed
    # (smoothed posteriors) — unchanged, this is offline-fit territory.
    if "p_burst" in df_feat_full.columns:
        p_burst_full[:split_idx] = df_feat_full["p_burst"].values[:split_idx]
    if "p_quiet" in df_feat_full.columns:
        p_quiet_full[:split_idx] = df_feat_full["p_quiet"].values[:split_idx]
    if n_states == 3 and "p_accum" in df_feat_full.columns:
        p_accum_full[:split_idx] = df_feat_full["p_accum"].values[:split_idx]
    if "hmm_state" in df_feat_full.columns:
        hmm_state_full[:split_idx] = df_feat_full["hmm_state"].values[:split_idx].astype(int)

    emit_arr = df_feat_full[EMIT_FEATURE_COLS].values.astype(float)
    emit_std_arr = (emit_arr - x_mean_emit) / x_std_emit

    # dynamic feature array — recomputed online for persist (capped counter,
    # reset on state change) so it is NOT taken from covariate.py's offline
    # column for the test split (that column is diagnostic-only per BUG 2
    # note in covariate.py). RVA / RVA_vel / vol_zscore / ret_signed ARE
    # taken from the precomputed columns since those are purely backward-
    # looking rolling statistics and causal by construction at every bar.
    rva_arr        = df_feat_full["RVA"].values.astype(float)
    rva_vel_arr    = df_feat_full["RVA_vel"].values.astype(float)
    vol_zscore_arr = df_feat_full["vol_zscore"].values.astype(float)
    ret_signed_arr = df_feat_full["ret_signed"].values.astype(float)

    # seed forward-algorithm belief at the train/test boundary using the
    # last train-split filtered/smoothed distribution if available, else
    # the model's startprob.
    if split_idx > 0 and "p_quiet" in df_feat_full.columns:
        if n_states == 3:
            prev_belief = np.array([
                df_feat_full["p_quiet"].values[split_idx - 1],
                df_feat_full["p_accum"].values[split_idx - 1],
                df_feat_full["p_burst"].values[split_idx - 1],
            ])
        else:
            prev_belief = np.array([
                df_feat_full["p_quiet"].values[split_idx - 1],
                df_feat_full["p_burst"].values[split_idx - 1],
            ])
        prev_belief = prev_belief / (prev_belief.sum() + EPS)
    else:
        prev_belief = startprob.copy()

    prev_state = hmm_state_full[split_idx - 1] if split_idx > 0 else int(np.argmax(prev_belief))
    if prev_state < 0:
        prev_state = int(np.argmax(prev_belief))

    # online persist counter — incremented/reset purely from causally
    # decoded states, independent of covariate.py's offline persist column.
    persist_count = 0.0
    if split_idx > 0:
        # walk backwards through train hmm_state to seed persist correctly
        k = split_idx - 1
        s_ref = hmm_state_full[k]
        cnt = 0
        while k >= 0 and hmm_state_full[k] == s_ref:
            cnt += 1
            k -= 1
        persist_count = min(float(cnt), PERSIST_CAP)

    trajectory_rows = []
    n_updates = 0

    for t in range(split_idx, T):
        # build dynamic feature vector x(t) using ONLINE persist
        x_t_raw = np.array([
            rva_arr[t], rva_vel_arr[t], persist_count,
            vol_zscore_arr[t], ret_signed_arr[t],
        ])
        x_t = (x_t_raw - x_mean_dyn) / x_std_dyn

        # BUG 6 FIX: on illiquid/microcap tickers (e.g. HQ), vol_ratio can
        # be near-constant for long stretches, making the rolling std used
        # in vol_zscore collapse toward zero. A subsequent volume spike then
        # produces an astronomically large vol_zscore (observed: ~5e13),
        # which after standardization can still be enormous, overflowing
        # W_i @ x_t in _dynamic_transition_row even with the max-subtraction
        # softmax trick (two logits can both saturate to -inf, giving a
        # 0/0 = NaN softmax output), which crashes the belief integrity
        # assertion. Clip the standardized feature vector to a generous but
        # finite range — this affects only extreme outlier bars and leaves
        # normal-range observations untouched.
        x_t = np.clip(x_t, -X_T_CLIP, X_T_CLIP)

        # ---- step a: one forward-algorithm step using CURRENT W, emissions ----
        emit_ll = _emission_log_likelihood(emit_std_arr[t], emit_means, emit_covars)

        # dynamic transition row from the PREVIOUS state, evaluated at x(t)
        # (the transition INTO bar t depends on the state at t-1 and the
        # observable context at t, consistent with the offline fit's
        # convention of x(t) driving the transition out of state s_{t-1}).
        A_row = _dynamic_transition_row(W[prev_state], x_t)   # (n_states,)

        # combine prior (from transition) with emission likelihood -> filtered belief
        log_prior = np.log(A_row + EPS)
        log_post  = log_prior + emit_ll
        log_post -= np.logaddexp.reduce(log_post)
        belief    = np.exp(log_post)

        belief = belief / (belief.sum() + EPS)

        # BUG 6 FIX: a hard crash here kills the entire ticker's walk-forward
        # run over a single degenerate bar (e.g. one extreme outlier on an
        # illiquid microcap). Recover gracefully: log loudly, reset belief
        # to uniform for this bar only, and continue — this is preferable
        # to silently producing a bad belief (which the assertion already
        # prevents) AND preferable to losing all downstream test bars for
        # this symbol over one numerical edge case. The X_T_CLIP fix above
        # should make this fallback rare; if it fires often for a given
        # ticker, that is itself a signal worth investigating (check
        # backtest_param_trajectory.csv for that ticker around this bar).
        if not (np.isfinite(belief).all() and (belief >= -EPS).all() and (belief <= 1 + EPS).all()):
            log(f"  [{ticker}] WARNING: degenerate belief at bar {t} "
                f"(x_t={x_t}, A_row={A_row}, emit_ll={emit_ll}) — "
                f"resetting to uniform belief for this bar and continuing")
            belief = np.full(n_states, 1.0 / n_states)

        p_quiet_full[t] = belief[0]
        if n_states == 3:
            p_accum_full[t] = belief[1]
        p_burst_full[t] = belief[n_states - 1]

        s_t = int(np.argmax(belief))
        hmm_state_full[t] = s_t

        # Cox intensity at the CURRENTLY decoded state, using x(t)
        lam_t = float(np.clip(np.exp(betas[s_t] @ x_t), *LAMBDA_CLIP))

        # ---- step b: signal already implicitly recorded via p_burst_full ----
        # (lead_time_analysis / threshold_sweep / etc. consume p_burst_full
        # downstream — nothing further needed here for "recording" beyond
        # having written belief into the *_full arrays above.)

        # ---- step c: observe ground truth for bar t (now legitimate) ----
        event_occurred = 1 if t in event_bars_set else 0

        # ---- step d: online updates (only affect predictions for t+1 onward) ----
        eta_w    = lr_w    / (1.0 + lr_decay * n_updates)
        eta_beta = lr_beta / (1.0 + lr_decay * n_updates)

        # Layer 1 update: nudge W[prev_state] toward the transition that
        # was actually observed (prev_state -> s_t), using x(t) as predictor.
        target_onehot = np.zeros(n_states)
        target_onehot[s_t] = 1.0
        grad_w = np.outer(target_onehot - A_row, x_t)   # (n_states, n_features)
        W[prev_state] += eta_w * grad_w

        # Layer 2 update: nudge beta_{s_t} using the Poisson GLM gradient
        # for this single observation.
        grad_beta = x_t * (event_occurred - lam_t)
        betas[s_t] += eta_beta * grad_beta

        n_updates += 1

        # update persist counter causally
        if s_t == prev_state:
            persist_count = min(persist_count + 1.0, PERSIST_CAP)
        else:
            persist_count = 0.0

        trajectory_rows.append({
            "ticker":         ticker,
            "bar_idx":        t,
            "datetime":       df_feat_full.index[t],
            "hmm_state":      s_t,
            "p_burst":        round(float(p_burst_full[t]), 6),
            "p_accum":        round(float(p_accum_full[t]), 6) if n_states == 3 else None,
            "p_quiet":        round(float(p_quiet_full[t]), 6),
            "lambda_t":       round(lam_t, 6),
            "W_norm":         round(float(np.linalg.norm(W)), 6),
            "beta_quiet_norm": round(float(np.linalg.norm(betas[0])), 6),
            "beta_accum_norm": round(float(np.linalg.norm(betas[1])), 6) if n_states == 3 else None,
            "beta_burst_norm": round(float(np.linalg.norm(betas[n_states - 1])), 6),
            "event_occurred": event_occurred,
        })

        prev_state = s_t

    return p_burst_full, p_accum_full, p_quiet_full, hmm_state_full, trajectory_rows


# build per-symbol position index (UNCHANGED from reference backtest.py)

def build_pos_map(s):
    """
    Given a per-ticker state DataFrame (sorted chronologically),
    return a dict mapping datetime -> sequential bar position (0-indexed).
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


# Q1 + Q2: lead-time analysis (UNCHANGED logic from reference backtest.py)

def lead_time_analysis(states, events, max_lag=MAX_LAG):
    """
    For each event bar, measure p_burst at lags 0, 1, 2, 3 bars before.
    Tests whether p_burst is significantly elevated before events vs baseline.

    Lag 0 = event bar itself (not tradeable).
    Lag 1 = 1 bar before event = signal bar where entry fires.
    Significant lag 1 = p_burst predicts events 1 bar in advance.
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


# Q3a: threshold sweep (UNCHANGED logic from reference backtest.py)

def threshold_sweep(states, events, horizon=DEFAULT_HORIZON,
                    thresholds=THRESHOLD_STEPS):
    """
    Signal fires at bar t -> entry at bar t+1.
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
            signal_bars = np.where(
                (p_burst > tau) & (hmm_state < 2)
            )[0]

            tp = fp = fn = 0

            for sb in signal_bars:
                entry_bar = sb + 1
                hit = any(
                    entry_bar + h in event_set
                    for h in range(horizon)
                )
                if hit:
                    tp += 1
                else:
                    fp += 1

            for eb in event_bars:
                preceded = any(
                    (eb - h) in set(signal_bars)
                    for h in range(1, horizon + 1)
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


# Q3b: continuation backtest (UNCHANGED logic from reference backtest.py)

def continuation_backtest(states, events, cont_window=DEFAULT_CONT_WINDOW):
    """
    From a confirmed HMM transition Accum->Burst (state 1->2):
      - Entry at the NEXT bar after transition (t+1)
      - Look for event in bars t+1 .. t+cont_window
      - Exit when HMM state != 2 OR cont_window bars elapsed
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

        burst_transitions = [
            i for i in range(1, n_bars)
            if hmm_state[i] == 2 and hmm_state[i - 1] != 2
        ]

        if not burst_transitions:
            continue

        hits, bars_to_events, burst_durations = [], [], []

        for trans_bar in burst_transitions:
            entry_bar   = trans_bar + 1
            event_found = False
            bars_to_ev  = np.nan

            for h in range(cont_window):
                check_bar = entry_bar + h
                if check_bar >= n_bars:
                    break
                if check_bar in event_set:
                    event_found = True
                    bars_to_ev  = h + 1
                    break

            hits.append(event_found)
            bars_to_events.append(bars_to_ev)

            exit_bar = n_bars - 1
            for j in range(trans_bar + 1, n_bars):
                if hmm_state[j] != 2:
                    exit_bar = j
                    break
            burst_durations.append(exit_bar - trans_bar)

            detail_rows.append({
                "ticker":              ticker,
                "transition_bar":      trans_bar,
                "entry_bar":           entry_bar,
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


# signal log (UNCHANGED logic from reference backtest.py)

def build_signal_log(states, events, tau=0.4, horizon=DEFAULT_HORIZON):
    """
    Build a complete log of every signal and event with outcome labels.
    Hit window is [sb+1, sb+horizon].
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
            entry_bar = sb + 1
            hit  = any(entry_bar + h in event_set for h in range(horizon))
            first_ev = next(
                (entry_bar + h for h in range(horizon)
                 if entry_bar + h in event_set), None
            )
            rows.append({
                "ticker":        ticker,
                "signal_bar":    int(sb),
                "entry_bar":     int(entry_bar),
                "type":          "signal",
                "outcome":       "TP" if hit else "FP",
                "p_burst":       round(float(p_burst[sb]), 4),
                "hmm_state":     int(hmm_state[sb]),
                "rva":           round(float(rva[sb]), 4),
                "bars_to_event": int(first_ev - entry_bar + 1) if first_ev else None,
            })

        for eb in event_bars:
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


# print helpers (UNCHANGED from reference backtest.py)

def print_lead_time(lt_df):
    log(f"\n{'='*65}")
    log("  Q1 + Q2  LEAD-TIME ANALYSIS  [TEST SET, ONLINE-UPDATED p_burst]")
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
        log(f"  " + "-" * 66)
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
                f"{rva_m if rva_m is not None else '  -':>9} "
                f"{fmt_p(rva_p if rva_p is not None else np.nan):>8}")


def print_threshold(thr_df, horizon):
    log(f"\n{'='*65}")
    log(f"  Q3a  THRESHOLD SWEEP  [TEST SET, ONLINE-UPDATED p_burst]")
    log(f"{'='*65}")
    log(f"  Horizon = {horizon} bars.  Entry at t+1.  Event window: t+1 .. t+{horizon}.")
    log("  Signal: p_burst > tau and hmm_state < 2 (not already in burst)\n")

    all_best = []
    for ticker in thr_df["ticker"].unique():
        t    = thr_df[thr_df["ticker"] == ticker]
        if t.empty: continue
        best = t.loc[t["f1"].idxmax()]
        all_best.append(best)
        log(f"  {ticker:<6}  tau={best['threshold']:.2f}  "
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
    log("  Q3b  CONTINUATION BACKTEST  [TEST SET, ONLINE-UPDATED p_burst]")
    log(f"{'='*65}")
    log("  From HMM Accum->Burst transition.  Entry at NEXT bar (t+1).")
    log(f"  {'Ticker':<7} {'Trans':>6} {'HitRate':>8} "
        f"{'MeanBars':>9} {'MedBars':>8} {'BrstDur':>8}")
    log(f"  " + "-" * 55)

    if cont_df.empty:
        log("  No transitions detected."); return

    for _, row in cont_df.sort_values("hit_rate", ascending=False).iterrows():
        bte_m = row["mean_bars_to_event"]
        bte_d = row["med_bars_to_event"]
        log(f"  {row['ticker']:<7} "
            f"{int(row['n_transitions']):>6} "
            f"{row['hit_rate']:>8.3f} "
            f"{bte_m if bte_m is not None else '  -':>9} "
            f"{bte_d if bte_d is not None else '  -':>8} "
            f"{row['mean_burst_duration']:>8.1f}")


def print_verdict(lt_df, thr_df, cont_df):
    log(f"\n{'='*65}")
    log("  VERDICT  [TEST SET, ONLINE WALK-FORWARD — HONEST OOS ESTIMATES]")
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
    log(f"\n  These are TEST-SET numbers from the ONLINE walk-forward loop.")
    log(f"  At every bar t, predictions used only model state updated from")
    log(f"  bars 0..t-1 — no future information at any point.")
    log(f"  Entry sequencing: signal at t -> entry at t+1 -> hold 5 bars.")

    if not thr_df.empty:
        best_per = thr_df.loc[thr_df.groupby("ticker")["f1"].idxmax()]
        candidates = best_per[best_per["f1"] >= 0.20].sort_values("f1", ascending=False)
        if not candidates.empty:
            log(f"\n  Symbols worth pursuing (F1 >= 0.20 on test set):")
            for _, r in candidates.iterrows():
                log(f"    {r['ticker']:<8} F1={r['f1']:.3f}  "
                    f"P={r['precision']:.3f}  R={r['recall']:.3f}  tau={r['threshold']:.2f}")


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",       type=int,   default=DEFAULT_HORIZON)
    parser.add_argument("--cont-window",   type=int,   default=DEFAULT_CONT_WINDOW)
    parser.add_argument("--symbols",       nargs="*",  default=None)
    parser.add_argument("--min-threshold", type=float, default=0.10)
    parser.add_argument("--signal-tau",    type=float, default=0.40)
    parser.add_argument("--lr-w",          type=float, default=DEFAULT_LR_W,
                        help="online learning rate for dynamic transition weights W "
                             "(unvalidated default — tune via backtest_param_trajectory.csv)")
    parser.add_argument("--lr-beta",       type=float, default=DEFAULT_LR_BETA,
                        help="online learning rate for state-stratified Cox betas "
                             "(unvalidated default — tune via backtest_param_trajectory.csv)")
    parser.add_argument("--lr-decay",      type=float, default=DEFAULT_LR_DECAY,
                        help="inverse-time decay rate applied to both lr-w and lr-beta")
    args = parser.parse_args()

    thresholds = np.round(np.arange(args.min_threshold, 0.95, 0.05), 2)

    log("=" * 65)
    log("  backtest.py  -  Online Walk-Forward Simulator  [DYNAMIC]")
    log("=" * 65)
    log(f"  Horizon       : {args.horizon} bars")
    log(f"  Cont window   : {args.cont_window} bars")
    log(f"  Entry timing  : signal at bar t -> entry at bar t+1")
    log(f"  Exit timing   : bar t+{args.horizon+1} open OR first non-Burst bar")
    log(f"  lr_w          : {args.lr_w}  (decay={args.lr_decay})")
    log(f"  lr_beta       : {args.lr_beta}  (decay={args.lr_decay})")
    log(f"  Evaluation    : TEST SET, online walk-forward (no future info at any bar)")
    log("=" * 65)

    states, events = load_data()

    if args.symbols:
        states = states[states["ticker"].isin(args.symbols)]
        events = events[events["ticker"].isin(args.symbols)]

    tickers = sorted(states["ticker"].unique())

    log("\n  Running online walk-forward simulation per symbol...")
    all_trajectory_rows = []
    updated_state_frames = []

    for ticker in tickers:
        bundle = load_model_bundle(ticker)
        if bundle is None:
            log(f"  [{ticker}] no model bundle found in {MODEL_DIR}/ — skipping "
                f"(re-run covariate.py for this symbol)")
            continue

        df_t = states[states["ticker"] == ticker].copy().sort_index()

        required = DYN_FEATURE_COLS + EMIT_FEATURE_COLS
        missing = [c for c in required if c not in df_t.columns]
        if missing:
            log(f"  [{ticker}] missing required columns {missing} — skipping")
            continue

        ev_t = events[events["ticker"] == ticker]
        time_to_pos = {dt: i for i, dt in enumerate(df_t.index)}
        event_bars_set = set(
            time_to_pos[dt] for dt in ev_t["date"] if dt in time_to_pos
        )

        p_burst, p_accum, p_quiet, hmm_state, traj_rows = run_online_walkforward(
            ticker, bundle, df_t, event_bars_set,
            lr_w=args.lr_w, lr_beta=args.lr_beta, lr_decay=args.lr_decay,
        )

        split_idx = bundle["split_idx"]
        n_test = len(df_t) - split_idx
        log(f"  [{ticker}] online walk-forward complete: {n_test} test bars, "
            f"{len(traj_rows)} updates logged")

        # overwrite test-split p_burst / p_accum / p_quiet / hmm_state with
        # the online-generated values; train-split values are left as the
        # offline-fit (smoothed) values already present in df_t.
        df_t["p_burst"] = p_burst
        if p_accum is not None:
            df_t["p_accum"] = p_accum
        df_t["p_quiet"] = p_quiet
        df_t["hmm_state"] = hmm_state

        assert df_t["p_burst"].between(0.0 - EPS, 1.0 + EPS).all(), \
            f"INTEGRITY ERROR: online p_burst out of [0,1] range for {ticker}"

        updated_state_frames.append(df_t)
        all_trajectory_rows.extend(traj_rows)

    if not updated_state_frames:
        log("\nERROR: no symbols produced a valid online walk-forward run. Exiting.")
        sys.exit(1)

    states_online = pd.concat(updated_state_frames).sort_index()

    # restrict everything downstream to TEST split only, exactly as the
    # static reference backtest.py did — train bars never enter the metrics.
    states_test = states_online[states_online["split"] == "test"].copy()
    log(f"\n  [OK] Restricted to test split: {len(states_test):,} / "
        f"{len(states_online):,} bars "
        f"({100*len(states_test)/max(len(states_online),1):.1f}%)")
    log(f"  *** All p_burst values on test split came from the online ***")
    log(f"  *** walk-forward loop — no future information used at any bar ***")

    events_eval = events[events["ticker"].isin(states_test["ticker"].unique())]

    log("\n  Running lead-time analysis...")
    lt_df = lead_time_analysis(states_test, events_eval, max_lag=MAX_LAG)
    print_lead_time(lt_df)

    log("\n  Running threshold sweep...")
    thr_df = threshold_sweep(states_test, events_eval,
                             horizon=args.horizon, thresholds=thresholds)
    print_threshold(thr_df, args.horizon)

    log("\n  Running continuation backtest...")
    cont_df, cont_detail = continuation_backtest(
        states_test, events_eval, cont_window=args.cont_window)
    print_continuation(cont_df)

    print_verdict(lt_df, thr_df, cont_df)

    # save
    lt_df.to_csv("backtest_lead_time.csv", index=False)
    thr_df.to_csv("backtest_threshold.csv", index=False)
    cont_df.to_csv("backtest_continuation.csv", index=False)
    cont_detail.to_csv("backtest_continuation_detail.csv", index=False)

    sig_df = build_signal_log(states_test, events_eval,
                              tau=args.signal_tau, horizon=args.horizon)
    sig_df.to_csv("backtest_signals.csv", index=False)

    traj_df = pd.DataFrame(all_trajectory_rows)
    traj_df.to_csv("backtest_param_trajectory.csv", index=False)

    with open("backtest_summary.txt", "w") as f:
        f.write("BACKTEST SUMMARY [ONLINE WALK-FORWARD, DYNAMIC MODEL]\n")
        f.write(f"Horizon        : {args.horizon} bars\n")
        f.write(f"Entry          : signal at t, entry at t+1\n")
        f.write(f"Evaluation set : TEST ONLY (split=='test'), online-updated p_burst\n")
        f.write(f"lr_w           : {args.lr_w}  (decay={args.lr_decay})\n")
        f.write(f"lr_beta        : {args.lr_beta}  (decay={args.lr_decay})\n")
        if not lt_df.empty:
            lag1   = lt_df[lt_df["lag"]==1]
            n_sig  = lag1["mw_significant"].sum()
            n_tot  = lag1["ticker"].nunique()
            f.write(f"Lead-time sig  : {n_sig}/{n_tot} symbols\n")
        if not thr_df.empty:
            f.write(f"Best F1        : {thr_df['f1'].max():.3f}\n")
        if not cont_df.empty:
            f.write(f"Mean hit rate  : {cont_df['hit_rate'].mean():.3f}\n")

    log("\n  Saved -> backtest_lead_time.csv")
    log("  Saved -> backtest_threshold.csv")
    log("  Saved -> backtest_continuation.csv")
    log("  Saved -> backtest_continuation_detail.csv")
    log("  Saved -> backtest_signals.csv")
    log("  Saved -> backtest_param_trajectory.csv  (diagnostic: check W_norm /")
    log("           beta_*_norm columns for convergence vs oscillation vs drift)")
    log("  Saved -> backtest_summary.txt")
    log("\n  [DONE] All results are honest out-of-sample estimates from an")
    log("  online walk-forward simulation — verify backtest_param_trajectory.csv")
    log("  before trusting these numbers (see spec section 9 on unvalidated")
    log("  learning rate defaults).")


if __name__ == "__main__":
    main()