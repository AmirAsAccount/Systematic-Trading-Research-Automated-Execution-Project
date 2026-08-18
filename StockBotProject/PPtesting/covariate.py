"""
covariate.py  —  Dynamic Cox-Modulated Poisson Process
            with 2-Layer Latent Markov Regime
ARCHITECTURE (DYNAMIC)
-----------------------
  Layer 1  Input-Coupled HMM (latent intensity)
  Layer 2  State-Stratified Cox Intensity
  Feature vector x(t)  [5-dimensional]
           [RVA, RVA_vel, persist, vol_zscore, ret_signed]
OUTPUTS
-------
  covariate_states.csv
  covariate_results.csv
  covariate_transitions.csv
  covariate_burst_prob.csv
  covariate_model_<TICKER>.npz
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats, optimize
from hmmlearn import hmm
try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

# ── constants ──────────────────────────────────────────────────────────────────
EPS                 = 1e-12
CACHE_DIR           = Path("cache")
RESULTS_CSV         = "process_id_15m_results.csv"
EVENTS_CSV          = "process_id_15m_events.csv"
TRAIN_FRAC          = 0.60
MIN_BARS_PER_STATE  = 30
PERSIST_CAP         = 50
MODEL_DIR           = Path("covariate_models")
TICKERS = [
    "AHMA", "ARQQ", "BRUN", "CAST", "CUPR", "ELVR", "FAMI", "GLXG",
    "HQ",   "HUBC", "LUD",  "MTEN", "PRFX", "QUCY", "RDGT", "RGNT",
    "RTB",  "SDOT", "SLMT", "UBXG", "VSME", "WLDS", "WYFI",
]
KNOWN_FLOATS = {
    "AHMA": 49_415_181,  "ARQQ": 3_014_767,   "BRUN": 2_460_142,
    "CAST": 156_691_120, "CUPR": 48_097_092,   "ELVR": 614_442,
    "FAMI": 236_629,     "GLXG": 54_589_644,   "HQ":   6_210_698,
    "HUBC": 42_242_063,  "LUD":  221_233,       "MTEN": 35_787_793,
    "PRFX": 43_312_350,  "QUCY": 11_321_027,   "RDGT": 436_974,
    "RGNT": 179_204_775, "RTB":  291_824,       "SDOT": 1_012_647,
    "SLMT": 515_055,     "UBXG": 3_515_685,    "VSME": 64_283_127,
    "WLDS": 22_910_171,  "WYFI": 1_963_121,
}
SESSION_START_UTC = 14 * 60 + 30
SESSION_END_UTC   = 21 * 60
DYN_FEATURE_COLS  = ["RVA", "RVA_vel", "persist", "vol_zscore", "ret_signed"]
EMIT_FEATURE_COLS = ["vol_ratio", "abs_ret"]

# Pandas resample rules — same as pnlc.py / covariatec.py
TF_RESAMPLE = {
    "5m":  "5min",  "10m": "10min", "15m": "15min",
    "30m": "30min", "45m": "45min", "1h":  "1h",
    "2h":  "2h",    "3h":  "3h",    "4h":  "4h",
    "1d":  "1D",
}

def log(msg=""): print(msg, flush=True)

# ── data loading ──────────────────────────────────────────────────────────────
def _is_session(ts):
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    m = utc.hour * 60 + utc.minute
    return SESSION_START_UTC <= m < SESSION_END_UTC

def _session_position(ts):
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    m = utc.hour * 60 + utc.minute
    pos = (m - SESSION_START_UTC) / (SESSION_END_UTC - SESSION_START_UTC)
    return float(np.clip(pos, 0.0, 1.0))

def _read_parquet_paginated(path, chunk_rows=50_000):
    """
    Read a parquet file in row-chunks to avoid loading the entire file into
    RAM at once. Uses pyarrow's batch reader if available; falls back to a
    single pd.read_parquet call for small files or missing pyarrow.
    Returns a concatenated DataFrame, or None on failure.
    """
    try:
        import pyarrow.parquet as pq
        pf     = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        if n_rows <= chunk_rows:
            # Small file — single read is fine
            return pd.read_parquet(path)
        chunks = []
        for batch in pf.iter_batches(batch_size=chunk_rows):
            chunks.append(batch.to_pandas())
        return pd.concat(chunks, ignore_index=False)
    except ImportError:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path)   # last-resort fallback

def load_bars(ticker, use_cache=True, resample_to=None):
    """
    Load OHLCV bars from cache/ parquets for a single ticker (stocks only).
    PAGINATION / CHUNKED READING
    ----------------------------
    Large parquets are read in row-chunks of 50k rows via _read_parquet_paginated()
    so memory usage stays bounded when processing multiple tickers.
    RESAMPLE
    --------
    If resample_to is set (e.g. "1h"), bars are resampled to that timeframe.
    OHLCV aggregation: Open=first, High=max, Low=min, Close=last, Volume=sum.
    """
    cache_1h  = CACHE_DIR / f"{ticker}_1h.parquet"
    cache_15m = CACHE_DIR / f"{ticker}_15m_recent.parquet"

    def read(path):
        if not path.exists():
            return None
        try:
            df = _read_parquet_paginated(path)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            needed = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            if len(needed) < 5:
                return None
            df = df[needed].dropna()
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df = df[df.index.map(_is_session)]
            return df if not df.empty else None
        except Exception:
            return None

    df_1h  = read(cache_1h)
    df_15m = read(cache_15m)

    chunks = []
    if df_1h is not None and not df_1h.empty:
        cutoff  = pd.Timestamp(datetime.now() - timedelta(days=58), tz="UTC")
        trimmed = df_1h[df_1h.index < cutoff]
        if not trimmed.empty:
            chunks.append(trimmed)
    if df_15m is not None and not df_15m.empty:
        chunks.append(df_15m)
    if not chunks:
        return None

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < 50:
        return None

    # ── optional resample ─────────────────────────────────────────────────────
    if resample_to and resample_to != "15m":
        rule = TF_RESAMPLE.get(resample_to)
        if rule is None:
            log(f"  [{ticker}] WARNING: unknown --resample-to '{resample_to}' — "
                f"valid options: {list(TF_RESAMPLE.keys())}")
        else:
            agg = {"Open": "first", "High": "max",
                   "Low": "min",   "Close": "last", "Volume": "sum"}
            df = df.resample(rule).agg(
                {k: v for k, v in agg.items() if k in df.columns}
            ).dropna()
            if len(df) < 20:
                log(f"  [{ticker}] too few bars after resample to {resample_to} "
                    f"({len(df)}) — skipping")
                return None

    return df if len(df) >= 20 else None

# ── feature engineering ───────────────────────────────────────────────────────
def compute_features(df, vol_window=130):
    df = df.copy()
    df["vol_ma"]     = df["Volume"].rolling(vol_window, min_periods=max(10, vol_window // 4)).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma"] + EPS)
    df["abs_ret"]    = ((df["Close"] - df["Open"]) / (df["Open"] + EPS)).abs()
    df["RVA"]        = df["vol_ratio"].diff().fillna(0.0)

    # session_pos: only meaningful for intraday; set to 0.5 for daily bars
    if hasattr(df.index, 'hour') and df.index.dtype.tz is not None:
        try:
            df["session_pos"] = [_session_position(ts) for ts in df.index]
        except Exception:
            df["session_pos"] = 0.5
    else:
        df["session_pos"] = 0.5

    df["RVA_vel"]    = df["RVA"].diff().fillna(0.0)
    df["vol_zscore"] = (
        (df["vol_ratio"] - df["vol_ratio"].rolling(vol_window).mean())
        / (df["vol_ratio"].rolling(vol_window).std() + EPS)
    ).fillna(0.0)
    df["ret_signed"] = (df["Close"] - df["Open"]) / (df["Open"] + EPS)
    df["persist"]    = 0.0
    df = df.dropna(subset=["vol_ma", "vol_ratio", "abs_ret"])
    return df

def _fill_persist_from_states(df, state_seq):
    persist = np.zeros(len(state_seq), dtype=float)
    count   = 0
    for t in range(len(state_seq)):
        if t == 0:
            count = 0
        elif state_seq[t] == state_seq[t - 1]:
            count += 1
        else:
            count = 0
        persist[t] = min(count, PERSIST_CAP)
    df = df.copy()
    df["persist"] = persist
    return df

# ── filtered forward probabilities ────────────────────────────────────────────
def filtered_forward_probs(model, X_std):
    n_samples = len(X_std)
    n_states  = model.n_components
    framelogprob = model._compute_log_likelihood(X_std)
    log_alpha    = np.full((n_samples, n_states), -np.inf)
    log_alpha[0] = np.log(model.startprob_ + EPS) + framelogprob[0]
    log_transmat = np.log(model.transmat_ + EPS)

    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, j])
                + framelogprob[t, j]
            )

    filtered = np.zeros((n_samples, n_states))
    for t in range(n_samples):
        log_sum    = np.logaddexp.reduce(log_alpha[t])
        filtered[t] = np.exp(log_alpha[t] - log_sum)

    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6), \
        "INTEGRITY ERROR: filtered probabilities do not sum to 1 at every bar"
    assert (filtered >= 0).all() and (filtered <= 1 + 1e-9).all(), \
        "INTEGRITY ERROR: filtered probabilities contain values outside [0, 1]"

    return filtered

# ── HMM layer ─────────────────────────────────────────────────────────────────
def fit_hmm(df_feat, n_states=3, n_iter=200, random_seed=42, train_frac=TRAIN_FRAC):
    obs_cols = EMIT_FEATURE_COLS
    X = df_feat[obs_cols].values.astype(float)
    X[:, 0] = np.clip(X[:, 0], 0, 10.0)
    X[:, 1] = np.clip(X[:, 1], 0, 0.5)

    T         = len(X)
    split_idx = int(T * train_frac)

    if split_idx < 30:
        return None, None, None, split_idx, -np.inf, {}, [], None, None

    X_train_raw = X[:split_idx]
    mu_x  = X_train_raw.mean(axis=0)
    std_x = X_train_raw.std(axis=0) + EPS
    X_std = (X - mu_x) / std_x
    X_train = X_std[:split_idx]
    X_test  = X_std[split_idx:]

    best_model, best_logL = None, -np.inf

    for seed in [random_seed, random_seed + 1, random_seed + 7]:
        try:
            model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="diag",
                n_iter=n_iter, tol=1e-5, random_state=seed,
                init_params="stmc", params="stmc",
            )
            model.fit(X_train)
            logL = model.score(X_train)
            if logL > best_logL:
                best_logL  = logL
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return None, None, None, split_idx, -np.inf, {}, [], None, None

    train_smoothed = best_model.predict_proba(X_train)
    train_states   = best_model.predict(X_train)

    if len(X_test) > 0:
        test_filtered = filtered_forward_probs(best_model, X_test)
        test_states   = np.argmax(test_filtered, axis=1)
    else:
        test_filtered = np.empty((0, n_states))
        test_states   = np.array([], dtype=int)

    state_means = []
    for s in range(n_states):
        mask    = train_states == s
        mean_vr = X[:split_idx][mask, 0].mean() if mask.sum() > 0 else 0.0
        state_means.append((mean_vr, s))
    state_means.sort()

    perm       = [orig for _, orig in state_means]
    old_to_new = {old: new for new, (_, old) in enumerate(state_means)}
    train_states   = np.array([old_to_new[s] for s in train_states])
    train_smoothed = train_smoothed[:, perm]

    if len(test_states) > 0:
        test_states   = np.array([old_to_new[s] for s in test_states])
        test_filtered = test_filtered[:, perm]

    state_seq = np.concatenate([train_states, test_states]).astype(int)
    fwd_probs = (np.vstack([train_smoothed, test_filtered])
                 if len(test_filtered) > 0 else train_smoothed)

    # Note: state_seq and fwd_probs are already permuted above.
    # Skip model internal permutation to avoid hmmlearn validation issues.
    # The permuted outputs are what we use downstream, not the model internals.

    labels = {0: "Quiet", 1: "Accum", 2: "Burst"} if n_states == 3 \
        else {0: "Quiet", 1: "Burst"}

    return (best_model, state_seq, fwd_probs, split_idx, best_logL, labels, perm,
            mu_x, std_x)

# ── dynamic HMM initial fit ────────────────────────────────────────────────────
def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / (e.sum() + EPS)

def _init_W_from_transmat(transmat, x_mean, n_states, n_features):
    W = np.zeros((n_states, n_states, n_features))
    for i in range(n_states):
        log_target  = np.log(transmat[i] + EPS)
        log_target -= log_target.mean()
        norm_sq     = np.dot(x_mean, x_mean) + EPS
        for j in range(n_states):
            W[i, j] = log_target[j] * x_mean / norm_sq
    return W

def fit_dynamic_hmm_initial(model, df_feat_train, train_states, n_states,
                            n_iter_w=50, lr_w_init=0.005):
    n_features = len(DYN_FEATURE_COLS)
    X_dyn      = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    x_mean_dyn = X_dyn.mean(axis=0)
    x_std_dyn  = X_dyn.std(axis=0) + EPS
    X_norm     = (X_dyn - x_mean_dyn) / x_std_dyn

    W = _init_W_from_transmat(model.transmat_, x_mean_dyn / x_std_dyn,
                              n_states, n_features)

    T = len(train_states)

    if HAS_SM:
        for i in range(n_states):
            src_mask = np.array([t for t in range(T - 1) if train_states[t] == i])
            if len(src_mask) < MIN_BARS_PER_STATE:
                continue

            X_i = X_norm[src_mask]
            y_i = train_states[src_mask + 1]

            try:
                mnl    = sm.MNLogit(y_i, sm.add_constant(X_i, prepend=True))
                res    = mnl.fit(disp=False, maxiter=200)
                params = res.params
                base_state  = np.bincount(y_i, minlength=n_states).argmax()
                all_params  = np.zeros((X_norm.shape[1] + 1, n_states))
                col = 0
                for j in range(n_states):
                    if j == base_state:
                        continue
                    all_params[:, j] = params[:, col]
                    col += 1
                W[i] = all_params[1:, :].T
            except Exception as e:
                log(f"      MNLogit failed for state {i} ({e}), using gradient fallback")

    lr = lr_w_init
    log_loss = 0.0
    for _ in range(n_iter_w):
        total_grad = np.zeros_like(W)
        log_loss   = 0.0
        for t in range(T - 1):
            i  = train_states[t]
            j  = train_states[t + 1]
            xt = X_norm[t]
            logits = W[i] @ xt
            probs  = _softmax(logits)
            total_grad[i] += np.outer((np.eye(n_states)[j] - probs), xt)
            log_loss      += np.log(probs[j] + EPS)

        W    += lr * total_grad / (T + EPS)
        lr   *= 0.995

    log(f"      Dynamic W fit: final log-loss/bar = {log_loss / max(T-1, 1):.4f}")
    return W, x_mean_dyn, x_std_dyn

# ── state-stratified Cox fit ──────────────────────────────────────────────────
def _fit_poisson_glm_beta(X_dyn_norm, y_binary, fallback_beta=None):
    n_features = X_dyn_norm.shape[1]
    y = y_binary.astype(float)

    if HAS_SM:
        try:
            res = sm.GLM(y, X_dyn_norm,
                         family=sm.families.Poisson()).fit(disp=False, maxiter=200)
            return res.params.astype(float), True
        except Exception:
            pass

    def neg_logL(beta):
        lam = np.clip(np.exp(X_dyn_norm @ beta), 1e-6, 100.0)
        return -np.sum(y * np.log(lam + EPS) - lam)

    def neg_logL_grad(beta):
        lam = np.clip(np.exp(X_dyn_norm @ beta), 1e-6, 100.0)
        return -(X_dyn_norm.T @ (y - lam))

    x0 = fallback_beta if fallback_beta is not None else np.zeros(n_features)

    try:
        res = optimize.minimize(neg_logL, x0=x0, jac=neg_logL_grad, method="L-BFGS-B")
        return res.x.astype(float), res.success
    except Exception:
        return np.zeros(n_features), False

def fit_stratified_cox_initial(event_bars_train, train_states, df_feat_train,
                               n_states, x_mean_dyn, x_std_dyn):
    T      = len(train_states)
    X_dyn  = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    X_norm = (X_dyn - x_mean_dyn) / x_std_dyn

    event_set   = set(int(b) for b in event_bars_train if 0 <= int(b) < T)
    y_all       = np.array([1.0 if t in event_set else 0.0 for t in range(T)])

    beta_pooled, pooled_ok = _fit_poisson_glm_beta(X_norm, y_all)
    if not pooled_ok:
        beta_pooled = np.zeros(len(DYN_FEATURE_COLS))

    betas = np.zeros((n_states, len(DYN_FEATURE_COLS)))
    for k in range(n_states):
        mask = train_states == k
        if mask.sum() < MIN_BARS_PER_STATE or y_all[mask].sum() < 1:
            betas[k] = beta_pooled
            continue

        beta_k, ok = _fit_poisson_glm_beta(X_norm[mask], y_all[mask],
                                           fallback_beta=beta_pooled)
        betas[k] = beta_k if ok else beta_pooled

    return betas

# ── bar-level Cox diagnostics ─────────────────────────────────────────────────
def cox_diagnostics_from_betas(event_bars_train, train_states, df_feat_train,
                               betas, x_mean_dyn, x_std_dyn, n_states, labels):
    """
    Compute diagnostics from the bar-level stratified betas.
    Replaces fit_cox_with_covariates for reporting purposes.
    Evaluates whether Cox intensity truly separates events from non-events
    and discriminates across states.
    """
    T      = len(train_states)
    X_dyn  = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    X_norm = (X_dyn - x_mean_dyn) / x_std_dyn
    event_set = set(int(b) for b in event_bars_train if 0 <= int(b) < T)
    y_all  = np.array([1.0 if t in event_set else 0.0 for t in range(T)])

    # Compute per-bar intensity from stratified betas
    lambda_t = np.zeros(T)
    for k in range(n_states):
        mask = train_states == k
        if mask.sum() > 0:
            lambda_t[mask] = np.exp(
                np.clip(X_norm[mask] @ betas[k], -10, 10)
            )

    # Pseudo R² — does intensity predict events better than mean rate?
    mean_rate   = y_all.mean()
    log_null    = np.sum(y_all * np.log(mean_rate + EPS) +
                         (1 - y_all) * np.log(1 - mean_rate + EPS))
    lam_clipped = np.clip(lambda_t / (1 + lambda_t), EPS, 1 - EPS)
    log_model   = np.sum(y_all * np.log(lam_clipped) +
                         (1 - y_all) * np.log(1 - lam_clipped))
    pseudo_r2   = 1 - (log_model / (log_null + EPS))

    # Mean intensity at event bars vs non-event bars
    event_mask     = y_all == 1
    mean_lam_event = lambda_t[event_mask].mean() if event_mask.sum() > 0 else np.nan
    mean_lam_noise = lambda_t[~event_mask].mean() if (~event_mask).sum() > 0 else np.nan
    intensity_lift = mean_lam_event / (mean_lam_noise + EPS)

    # Per-state mean intensity
    state_intensity = {}
    for k in range(n_states):
        mask = train_states == k
        state_intensity[labels[k]] = float(lambda_t[mask].mean()) if mask.sum() > 0 else np.nan

    return {
        "pseudo_r2":        round(float(pseudo_r2), 4),
        "intensity_lift":   round(float(intensity_lift), 3),
        "mean_lam_event":   round(float(mean_lam_event), 5),
        "mean_lam_noise":   round(float(mean_lam_noise), 5),
        "state_intensity":  state_intensity,
        "converged":        True,
    }

# ── per-state diagnostics ─────────────────────────────────────────────────────
def state_event_rates(event_bars, state_seq, n_states, T):
    event_set  = set(int(b) for b in event_bars if 0 <= int(b) < len(state_seq))
    rows       = []
    quiet_rate = None

    for s in range(n_states):
        mask     = state_seq == s
        n_bars_s = mask.sum()
        n_events = sum(1 for b in event_set if state_seq[int(b)] == s)
        rate     = n_events / (n_bars_s + EPS)

        rows.append({
            "state": s, "n_bars": int(n_bars_s),
            "time_frac":    round(float(n_bars_s / (T + EPS)), 4),
            "n_events":     int(n_events),
            "event_frac":   round(float(n_events / (len(event_set) + EPS)), 4),
            "rate_per_bar": round(float(rate), 6),
        })

        if s == 0:
            quiet_rate = rate

    for row in rows:
        row["rate_ratio_vs_quiet"] = round(row["rate_per_bar"] / (quiet_rate + EPS), 2)

    return rows

# ── burst probability signal ──────────────────────────────────────────────────
def compute_burst_prob_signal(df_feat, fwd_probs, state_seq, n_states, split_idx):
    df_out      = df_feat.copy()
    burst_state = n_states - 1

    df_out["hmm_state"] = state_seq
    df_out["p_burst"]   = fwd_probs[:, burst_state]

    if n_states == 3:
        df_out["p_quiet"] = fwd_probs[:, 0]
        df_out["p_accum"] = fwd_probs[:, 1]
    else:
        df_out["p_quiet"] = fwd_probs[:, 0]

    split_labels    = ["train"] * split_idx + ["test"] * (len(df_feat) - split_idx)
    df_out["split"] = split_labels

    assert df_out["p_burst"].between(0.0 - EPS, 1.0 + EPS).all(), \
        "p_burst out of [0,1] range — check HMM output"

    return df_out

# ── model bundle serialization ────────────────────────────────────────────────
def save_model_bundle(ticker, model, W, betas, perm, split_idx,
                      x_mean_dyn, x_std_dyn, n_states, vol_window,
                      x_mean_emit, x_std_emit, tf_tag="15m"):
    MODEL_DIR.mkdir(exist_ok=True)
    tag      = f"_{tf_tag}" if tf_tag and tf_tag != "15m" else ""
    out_path = MODEL_DIR / f"covariate_model_{ticker}{tag}.npz"

    np.savez(
        out_path,
        W            = W,
        betas        = betas,
        emit_means   = model.means_.copy(),
        emit_covars  = model.covars_.copy(),
        startprob    = model.startprob_,
        transmat     = model.transmat_,
        perm         = np.array(perm),
        x_mean_dyn   = x_mean_dyn,
        x_std_dyn    = x_std_dyn,
        x_mean_emit  = x_mean_emit,
        x_std_emit   = x_std_emit,
        split_idx    = np.array(split_idx),
        n_states     = np.array(n_states),
        vol_window   = np.array(vol_window),
    )
    log(f"    Saved model bundle → {out_path}")
    return out_path

# ── transition matrix ─────────────────────────────────────────────────────────
def empirical_transition_matrix(state_seq, n_states):
    A = np.zeros((n_states, n_states))
    for i in range(len(state_seq) - 1):
        A[state_seq[i], state_seq[i + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    return A / (row_sums + EPS)

# ── main per-symbol routine ───────────────────────────────────────────────────
def process_symbol(ticker, event_bars, df_bars, n_states=3,
                   vol_window=130, n_windows=20, train_frac=TRAIN_FRAC,
                   tf_tag="15m"):
    log(f"\n{'─'*65}")
    log(f"  {ticker}  ({len(event_bars)} events, {len(df_bars)} bars)")
    log(f"{'─'*65}")

    df_feat = compute_features(df_bars, vol_window=vol_window)
    T = len(df_feat)

    if T < 50:
        log(f"    [{ticker}] insufficient bars ({T})")
        return None, None, None

    log(f"    Features: {T} bars  vol_ratio mean={df_feat['vol_ratio'].mean():.2f}  "
        f"RVA std={df_feat['RVA'].std():.3f}")

    (model, state_seq, fwd_probs, split_idx, hmm_logL, labels, perm,
     x_mean_emit, x_std_emit) = fit_hmm(
        df_feat, n_states=n_states, train_frac=train_frac)

    if model is None:
        log(f"    [{ticker}] HMM fit failed")
        return None, None, None

    n_train      = split_idx
    n_test       = T - split_idx
    train_states = state_seq[:split_idx]
    df_feat      = _fill_persist_from_states(df_feat, state_seq)

    log(f"    HMM logL (train): {hmm_logL:.2f}  |  "
        f"Split: {n_train} train / {n_test} test bars")

    state_counts = np.bincount(state_seq, minlength=n_states)
    for s, name in labels.items():
        log(f"      State {s} ({name:<10}): {state_counts[s]:>5} bars "
            f"({100*state_counts[s]/T:>5.1f}%)")

    A = empirical_transition_matrix(state_seq, n_states)

    if n_states == 3:
        p_entry, p_persist = A[1, 2], A[2, 2]
        log(f"    P(Burst|Accum)={p_entry:.4f}  P(Burst|Burst)={p_persist:.4f}")
    else:
        p_entry, p_persist = A[0, 1], A[1, 1]

    feat_len           = len(df_feat)
    event_bars_clamped = [min(int(b), feat_len-1) for b in event_bars if int(b) < feat_len]
    event_bars_train   = [b for b in event_bars_clamped if b < split_idx]

    rate_rows  = state_event_rates(event_bars_clamped, state_seq, n_states, T)
    log(f"    Per-state event rates:")
    for row in rate_rows:
        log(f"      State {row['state']} ({labels[row['state']]:<10}): "
            f"{row['n_bars']:>6} bars  {row['n_events']:>5} events  "
            f"{row['rate_ratio_vs_quiet']:>6.2f}x vs Quiet")

    log(f"    Fitting dynamic transition matrix W...")
    df_feat_train = df_feat.iloc[:split_idx].copy()

    try:
        W, x_mean_dyn, x_std_dyn = fit_dynamic_hmm_initial(
            model, df_feat_train, train_states, n_states)
    except Exception as e:
        log(f"      Dynamic W fit failed ({e}) — using zeros")
        n_features = len(DYN_FEATURE_COLS)
        W          = np.zeros((n_states, n_states, n_features))
        x_mean_dyn = df_feat_train[DYN_FEATURE_COLS].mean().values
        x_std_dyn  = df_feat_train[DYN_FEATURE_COLS].std().values + EPS

    log(f"    Fitting state-stratified Cox intensity...")
    try:
        betas = fit_stratified_cox_initial(
            event_bars_train, train_states, df_feat_train,
            n_states, x_mean_dyn, x_std_dyn)

        for k in range(n_states):
            log(f"      β_{k} ({labels[k]}): {np.round(betas[k], 4)}")
    except Exception as e:
        log(f"      Stratified Cox fit failed ({e}) — using zeros")
        betas = np.zeros((n_states, len(DYN_FEATURE_COLS)))

    # ── Compute bar-level diagnostics from stratified betas ────
    log(f"    Computing bar-level Cox diagnostics...")
    try:
        diag = cox_diagnostics_from_betas(
            event_bars_train, train_states, df_feat_train,
            betas, x_mean_dyn, x_std_dyn, n_states, labels)

        log(f"    Cox diagnostics (bar-level):")
        log(f"      Pseudo R²       : {diag['pseudo_r2']:.4f}")
        log(f"      Intensity lift  : {diag['intensity_lift']:.2f}x  "
            f"(event bars vs noise bars)")
        for state_name, lam in diag['state_intensity'].items():
            log(f"      λ mean [{state_name:<10}]: {lam:.5f}")
    except Exception as e:
        log(f"      Cox diagnostics failed ({e})")
        diag = {
            "pseudo_r2": np.nan,
            "intensity_lift": np.nan,
            "mean_lam_event": np.nan,
            "mean_lam_noise": np.nan,
            "state_intensity": {name: np.nan for name in labels.values()},
            "converged": False,
        }

    save_model_bundle(ticker, model, W, betas, perm, split_idx,
                      x_mean_dyn, x_std_dyn, n_states, vol_window,
                      x_mean_emit, x_std_emit, tf_tag=tf_tag)

    df_signal = compute_burst_prob_signal(
        df_feat, fwd_probs, state_seq, n_states, split_idx)
    df_signal["ticker"] = ticker

    burst_state_idx  = n_states - 1
    burst_rate_ratio = next(
        (r["rate_ratio_vs_quiet"] for r in rate_rows if r["state"] == burst_state_idx), np.nan)

    result = {
        "ticker":           ticker,
        "n_events":         len(event_bars),
        "n_bars":           T,
        "n_train":          n_train,
        "n_test":           n_test,
        "n_states":         n_states,
        "tf_tag":           tf_tag,
        "hmm_logL":         round(float(hmm_logL), 2),
        "burst_rate_ratio": round(float(burst_rate_ratio), 2),
        "p_burst_entry":    round(float(p_entry), 4),
        "p_burst_persist":  round(float(p_persist), 4),
        "pseudo_r2":        diag["pseudo_r2"],
        "intensity_lift":   diag["intensity_lift"],
        "lam_event":        diag["mean_lam_event"],
        "lam_noise":        diag["mean_lam_noise"],
        "glm_converged":    diag.get("converged", False),
        "W_norm_initial":   round(float(np.linalg.norm(W)), 4),
    }

    trans_row = {"ticker": ticker}
    for i in range(n_states):
        for j in range(n_states):
            key = f"A_{labels[i][:2].lower()}_{labels[j][:2].lower()}"
            trans_row[key] = round(float(A[i, j]), 6)

    return result, df_signal, trans_row

# ── aggregate summary ─────────────────────────────────────────────────────────
def print_summary(results):
    log(f"\n{'='*65}")
    log("  AGGREGATE SUMMARY")
    log(f"{'='*65}")

    valid = [r for r in results if r.get("glm_converged")]
    log(f"  Symbols fitted: {len(valid)} / {len(results)}")

    brr = [r["burst_rate_ratio"] for r in valid if r.get("burst_rate_ratio")]
    if brr:
        log(f"  Burst rate ratio — Mean: {np.mean(brr):.2f}x  Max: {np.max(brr):.2f}x")

    # New pseudo-R² and intensity lift summary
    psr = [r["pseudo_r2"] for r in valid if np.isfinite(r.get("pseudo_r2", np.nan))]
    if psr:
        log(f"  Pseudo R² — Mean: {np.mean(psr):.4f}  Median: {np.median(psr):.4f}")

    ilift = [r["intensity_lift"] for r in valid if np.isfinite(r.get("intensity_lift", np.nan))]
    if ilift:
        log(f"  Intensity lift — Mean: {np.mean(ilift):.2f}x  Median: {np.median(ilift):.2f}x")

    log(f"\n  {'Ticker':<7} {'N':>5} {'Train':>6} {'Test':>6} "
        f"{'BrstRatio':>10} {'PseudoR²':>9} {'IntLift':>9} {'LamEvent':>10}")
    log(f"  " + "─" * 72)

    for r in results:
        if not r.get("glm_converged"):
            log(f"  {r['ticker']:<7} fit failed")
            continue

        log(f"  {r['ticker']:<7} {r['n_events']:>5} "
            f"{r['n_train']:>6} {r['n_test']:>6} "
            f"{r['burst_rate_ratio']:>10.2f}x "
            f"{r['pseudo_r2']:>+9.4f} "
            f"{r['intensity_lift']:>9.2f}x "
            f"{r['lam_event']:>10.5f}")

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",     nargs="*", default=None)
    parser.add_argument("--states",      type=int,  default=3)
    parser.add_argument("--vol-window",  type=int,  default=130)
    parser.add_argument("--windows",     type=int,  default=20)
    parser.add_argument("--train-frac",  type=float, default=TRAIN_FRAC)
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--resample-to", default=None,
                        help="Resample 15m cache bars to a larger timeframe before "
                             "fitting HMM/Cox. E.g. --resample-to 1h. "
                             f"Valid values: {list(TF_RESAMPLE.keys())}")

    args = parser.parse_args()
    tf_tag = args.resample_to if args.resample_to else "15m"

    log("=" * 65)
    log("  covariate.py  —  Dynamic HMM + Cox  [Stocks / 15m + resample]")
    log("=" * 65)
    log(f"  HMM states    : {args.states}")
    log(f"  Vol window    : {args.vol_window} bars")
    log(f"  Train frac    : {args.train_frac*100:.0f}%")
    log(f"  Source TF     : 15m  →  Active TF: {tf_tag}")

    if args.resample_to:
        log(f"  Resampling    : 15m bars → {args.resample_to} "
            f"(rule: {TF_RESAMPLE.get(args.resample_to, '?')})")

    log(f"  Pagination    : parquets read in 50k-row chunks via _read_parquet_paginated()")
    log(f"  statsmodels   : {'available' if HAS_SM else 'NOT available'}")
    log(f"  Cox diagnostics: bar-level intensity metrics (pseudo R², intensity lift)")
    log("=" * 65)

    cox_tickers = set(TICKERS)

    if not args.all and Path(RESULTS_CSV).exists():
        try:
            res_df      = pd.read_csv(RESULTS_CSV)
            cox_winners = res_df[res_df["winner"] == "Cox"]["ticker"].tolist()
            if cox_winners:
                cox_tickers = set(cox_winners)
                log(f"\n  Cox winners: {', '.join(sorted(cox_tickers))}")
        except Exception as e:
            log(f"  WARNING: {e}")

    events_lookup = {}
    if Path(EVENTS_CSV).exists():
        try:
            ev_df = pd.read_csv(EVENTS_CSV)
            for ticker, grp in ev_df.groupby("ticker"):
                events_lookup[ticker] = grp["bar_idx"].tolist()
        except Exception as e:
            log(f"  WARNING: {e}")

    target_tickers = args.symbols if args.symbols else sorted(cox_tickers)
    n_total        = len(target_tickers)
    all_results   = []
    all_state_dfs = []
    all_trans     = []

    for idx, ticker in enumerate(target_tickers, 1):
        log(f"\n  [{idx}/{n_total}] {ticker}")

        if ticker not in events_lookup:
            log(f"    no events — skipping")
            continue

        df_bars = load_bars(ticker, use_cache=not args.no_cache,
                            resample_to=args.resample_to)

        if df_bars is None:
            log(f"    no bar data — skipping")
            continue

        log(f"    Loaded {len(df_bars)} bars  "
            f"({df_bars.index.min().date()} → {df_bars.index.max().date()})")

        result, df_signal, trans_row = process_symbol(
            ticker, events_lookup[ticker], df_bars,
            n_states=args.states, vol_window=args.vol_window,
            n_windows=args.windows, train_frac=args.train_frac,
            tf_tag=tf_tag,
        )

        if result    is not None: all_results.append(result)
        if df_signal is not None: all_state_dfs.append(df_signal)
        if trans_row is not None: all_trans.append(trans_row)

    print_summary(all_results)

    # Output filenames tagged by active timeframe
    tag = f"_{tf_tag}" if tf_tag != "15m" else ""
    log(f"\n{'─'*65}")

    if all_results:
        pd.DataFrame(all_results).to_csv(f"covariate_results{tag}.csv", index=False)
        log(f"  Saved → covariate_results{tag}.csv")

    if all_state_dfs:
        state_df = pd.concat(all_state_dfs)
        keep = [c for c in ["ticker","Volume","vol_ratio","abs_ret","RVA","RVA_vel",
                            "persist","vol_zscore","ret_signed","session_pos",
                            "hmm_state","p_burst","p_quiet","p_accum","split"]
                if c in state_df.columns]
        state_df[keep].to_csv(f"covariate_states{tag}.csv")
        log(f"  Saved → covariate_states{tag}.csv  ({len(state_df)} bars)")
        log(f"  *** Downstream: filter to split=='test' before backtesting ***")

    if all_trans:
        pd.DataFrame(all_trans).to_csv(f"covariate_transitions{tag}.csv", index=False)
        log(f"  Saved → covariate_transitions{tag}.csv")

    if all_state_dfs:
        burst_df = pd.concat(all_state_dfs)
        cols = ["ticker","p_burst","split"] + (["p_accum"] if args.states == 3 else [])
        burst_df[[c for c in cols if c in burst_df.columns]].to_csv(
            f"covariate_burst_prob{tag}.csv")
        log(f"  Saved → covariate_burst_prob{tag}.csv")

    log(f"\n{'='*65}")
    log("  [DONE]")
    log(f"{'='*65}")

if __name__ == "__main__":
    main()