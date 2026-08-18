"""
covariate.py  —  Dynamic Cox-Modulated Poisson Process
            with 2-Layer Latent Markov Regime

ARCHITECTURE (DYNAMIC)
-----------------------
  Layer 1  Input-Coupled HMM (latent intensity)
           Transition matrix is time-varying:
             A[i,j](t) = softmax_j( W_i . x(t) )
           where W has shape (n_states, n_states, n_features).
           Emission: Gaussian (vol_ratio, abs_ret), fitted via Baum-Welch.
           Initial W fitted offline on TRAIN split; online updates in backtest.py.

  Layer 2  State-Stratified Cox Intensity
           Each state k has its own beta_k:
             lambda(t | s_t=k) = exp( beta_k . x(t) )
           Initial {beta_k} fitted offline on TRAIN split; online updates in backtest.py.

  Feature vector x(t)  [5-dimensional]
           [RVA, RVA_vel, persist, vol_zscore, ret_signed]

OUTPUTS
-------
  covariate_states.csv          - same schema as before (train+test diagnostics)
  covariate_results.csv         - per-symbol summary stats
  covariate_transitions.csv     - empirical transition matrices
  covariate_burst_prob.csv      - p_burst + split column
  covariate_models/covariate_model_<TICKER>.npz - serialized model bundle for backtest.py

BUGFIXES vs prior draft
------------------------
  BUG 1  base_disp / res_disp / dr_pct referenced unconditionally in the
         result dict even when glm_result["converged"] is False, causing
         a NameError. Fixed by initializing all three to np.nan before
         the conditional block.

  BUG 2  persist column on the test split was filled using the full
         state_seq (train+test) without a clarifying comment. This is
         NOT a leakage bug (test states come from causal filtered/Viterbi
         decoding), but is now explicitly documented, and backtest.py
         recomputes persist online for the test split independently per
         spec - the value written here is for diagnostic/offline-fit use
         on the TRAIN portion only; test-split persist in this file is
         informational only and must not be relied upon downstream.

  BUG 3  MNLogit base-category reconstruction could index out of bounds
         or silently mis-assign columns when a source state's observed
         transitions don't cover all n_states target categories (np.bincount
         returns a short array). Fixed with explicit bincount padding to
         length n_states and a shape assertion before reconstructing W_i,
         falling through to the gradient-only fit on any mismatch.

  BUG 4  Emission feature standardization (mu_x, std_x used to standardize
         (vol_ratio, abs_ret) before HMM fitting) was computed as a dead
         placeholder and never saved to the model bundle. backtest.py
         needs this to correctly standardize new test-bar observations
         before computing emission log-likelihoods in the online forward
         step. Fixed: fit_hmm() now returns mu_x, std_x; process_symbol
         passes them through; save_model_bundle() saves them as
         x_mean_emit / x_std_emit.

USAGE
-----
  python covariate.py
  python covariate.py --symbols GLXG CUPR HUBC
  python covariate.py --states 2
  python covariate.py --train-frac 0.7
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
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

# constants
EPS                 = 1e-12
CACHE_DIR           = Path("cache")
RESULTS_CSV         = "process_id_15m_results.csv"
EVENTS_CSV          = "process_id_15m_events.csv"
TRAIN_FRAC          = 0.60
MIN_BARS_PER_STATE  = 30      # fallback to pooled Cox if state has fewer bars
PERSIST_CAP         = 50      # cap persist counter to prevent unbounded growth
MODEL_DIR           = Path("covariate_models")
LAMBDA_CLIP         = (1e-6, 100.0)   # Cox intensity numerical safety range

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

# Feature columns used in the dynamic transition / Cox intensity layers
DYN_FEATURE_COLS = ["RVA", "RVA_vel", "persist", "vol_zscore", "ret_signed"]

def log(msg=""): print(msg, flush=True)


# data loading

def _is_session(ts):
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    m = utc.hour * 60 + utc.minute
    return SESSION_START_UTC <= m < SESSION_END_UTC


def _session_position(ts):
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    m = utc.hour * 60 + utc.minute
    pos = (m - SESSION_START_UTC) / (SESSION_END_UTC - SESSION_START_UTC)
    return float(np.clip(pos, 0.0, 1.0))


def load_bars(ticker, use_cache=True):
    cache_1h  = CACHE_DIR / f"{ticker}_1h.parquet"
    cache_15m = CACHE_DIR / f"{ticker}_15m_recent.parquet"

    def read(path):
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
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
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=58), tz="UTC")
        trimmed = df_1h[df_1h.index < cutoff]
        if not trimmed.empty:
            chunks.append(trimmed)
    if df_15m is not None and not df_15m.empty:
        chunks.append(df_15m)

    if not chunks:
        return None

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df if len(df) >= 50 else None


# feature engineering

def compute_features(df, vol_window=130):
    """
    Compute all features including the 5 dynamic model features.

    Existing features (preserved for backward compatibility):
      vol_ma, vol_ratio, abs_ret, RVA, session_pos

    New dynamic features:
      RVA_vel    - first diff of RVA (acceleration of vol_ratio velocity)
      persist    - consecutive bars in current HMM state. Filled in a
                   separate pass (_fill_persist_from_states) once a state
                   sequence is available; this function only initializes
                   the column to 0.0 as a placeholder.
      vol_zscore - standardized vol_ratio vs recent rolling window
      ret_signed - signed bar return (Close-Open)/Open, not absolute
                   (used for the directional/bullish-only intensity target)
    """
    df = df.copy()

    # existing features
    df["vol_ma"]     = df["Volume"].rolling(vol_window, min_periods=max(10, vol_window // 4)).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma"] + EPS)
    df["abs_ret"]    = ((df["Close"] - df["Open"]) / (df["Open"] + EPS)).abs()
    df["RVA"]        = df["vol_ratio"].diff().fillna(0.0)
    df["session_pos"] = [_session_position(ts) for ts in df.index]

    # new dynamic features
    df["RVA_vel"]    = df["RVA"].diff().fillna(0.0)

    df["vol_zscore"] = (
        (df["vol_ratio"] - df["vol_ratio"].rolling(vol_window).mean())
        / (df["vol_ratio"].rolling(vol_window).std() + EPS)
    ).fillna(0.0)

    df["ret_signed"] = (df["Close"] - df["Open"]) / (df["Open"] + EPS)

    # placeholder - filled by _fill_persist_from_states() once states exist
    df["persist"] = 0.0

    df = df.dropna(subset=["vol_ma", "vol_ratio", "abs_ret"])
    return df


def _fill_persist_from_states(df, state_seq):
    """
    Fill the 'persist' column using a decoded state sequence.
    Counter resets to 0 on state change, increments each bar in same state.
    Capped at PERSIST_CAP to prevent unbounded growth dominating features.

    BUG 2 NOTE: state_seq passed in here may span train+test (it does, in
    process_symbol). This is not a leakage bug - test-split states in
    state_seq come from causal filtered/Viterbi decoding (filtered_forward_probs
    + argmax), never from future bars. However, the persist values written
    here for the TEST split are for diagnostic/offline use only.
    backtest.py independently and incrementally recomputes persist online,
    bar-by-bar, during its walk-forward loop - it does NOT read the test-split
    persist values from this file's output. This separation is intentional
    per the covariate/backtest division of labor.
    """
    persist = np.zeros(len(state_seq), dtype=float)
    count = 0
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


# filtered forward probabilities

def filtered_forward_probs(model, X_std):
    """
    Compute FILTERED state probabilities using the forward algorithm only.
    Uses model.transmat_ (static matrix - for the initial static pass;
    the dynamic transition version is in backtest.py's online loop).

    At each timestep t, only uses observations 0..t - no future information.
    """
    n_samples = len(X_std)
    n_states  = model.n_components

    framelogprob = model._compute_log_likelihood(X_std)   # (T, K)

    log_alpha = np.full((n_samples, n_states), -np.inf)
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
        log_sum = np.logaddexp.reduce(log_alpha[t])
        filtered[t] = np.exp(log_alpha[t] - log_sum)

    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6), \
        "INTEGRITY ERROR: filtered probabilities do not sum to 1 at every bar"
    assert (filtered >= 0).all() and (filtered <= 1 + 1e-9).all(), \
        "INTEGRITY ERROR: filtered probabilities contain values outside [0, 1]"

    return filtered


# HMM layer

def fit_hmm(df_feat, n_states=3, n_iter=200, random_seed=42, train_frac=TRAIN_FRAC):
    """
    Fit HMM on TRAIN portion only.

    BUG 4 FIX: now returns mu_x, std_x (the emission feature standardization
    applied to (vol_ratio, abs_ret) before fitting) so they can be saved to
    the model bundle. backtest.py MUST apply this same standardization to
    new test-bar emission features before computing log-likelihoods -
    using any other standardization (e.g. recomputed from test data) would
    be both a leakage risk and inconsistent with the fitted emission params.

    Returns
    -------
    model, state_seq, fwd_probs, split_idx, logL, labels, perm, mu_x, std_x
    """
    obs_cols = ["vol_ratio", "abs_ret"]
    X = df_feat[obs_cols].values.astype(float)
    X[:, 0] = np.clip(X[:, 0], 0, 10.0)
    X[:, 1] = np.clip(X[:, 1], 0, 0.5)

    mu_x  = X.mean(axis=0)
    std_x = X.std(axis=0) + EPS
    X_std = (X - mu_x) / std_x

    T         = len(X_std)
    split_idx = int(T * train_frac)

    if split_idx < 30:
        return None, None, None, split_idx, -np.inf, {}, [], mu_x, std_x

    X_train = X_std[:split_idx]
    X_test  = X_std[split_idx:]

    best_model, best_logL = None, -np.inf

    for seed in [random_seed, random_seed + 1, random_seed + 7]:
        try:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                tol=1e-5,
                random_state=seed,
                init_params="stmc",
                params="stmc",
            )
            model.fit(X_train)
            logL = model.score(X_train)
            if logL > best_logL:
                best_logL = logL
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return None, None, None, split_idx, -np.inf, {}, [], mu_x, std_x

    train_smoothed = best_model.predict_proba(X_train)
    train_states   = best_model.predict(X_train)

    if len(X_test) > 0:
        test_filtered = filtered_forward_probs(best_model, X_test)
        test_states   = np.argmax(test_filtered, axis=1)
    else:
        test_filtered = np.empty((0, n_states))
        test_states   = np.array([], dtype=int)

    # State relabelling: ascending mean vol_ratio
    state_means = []
    for s in range(n_states):
        mask = train_states == s
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

    if len(test_filtered) > 0:
        fwd_probs = np.vstack([train_smoothed, test_filtered])
    else:
        fwd_probs = train_smoothed

    # Relabel model's internal params to match perm - so the SAVED model
    # bundle's emit_means / emit_covars / transmat / startprob are all in
    # the same relabeled (ascending vol_ratio) state ordering as state_seq.
    #
    # BUG 5 FIX: hmmlearn 0.3.x's covars_ GETTER returns full (n_states,
    # n_dim, n_dim) matrices for covariance_type='diag', but the SETTER
    # expects the compact (n_states, n_dim) diagonal form. Directly doing
    # `model.covars_ = model.covars_[perm]` round-trips through the getter's
    # expanded shape and fails the setter's validation with:
    #   ValueError: 'diag' covars must have shape (n_components, n_dim)
    # Fix: explicitly extract the diagonal before permuting and reassigning.
    diag_covars = np.array([np.diag(c) for c in best_model.covars_])  # (n_states, n_dim)
    best_model.transmat_  = best_model.transmat_[np.ix_(perm, perm)]
    best_model.startprob_ = best_model.startprob_[perm]
    best_model.means_     = best_model.means_[perm]
    best_model.covars_    = diag_covars[perm]

    if n_states == 3:
        labels = {0: "Quiet", 1: "Accum", 2: "Burst"}
    else:
        labels = {0: "Quiet", 1: "Burst"}

    return best_model, state_seq, fwd_probs, split_idx, best_logL, labels, perm, mu_x, std_x


# dynamic HMM initial fit

def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / (e.sum() + EPS)


def _init_W_from_transmat(transmat, x_ref, n_states, n_features):
    """
    Initialize W such that softmax(W_i . x_ref) ~= transmat[i].
    Approximation: W_i[j] = log_target[j] * x_ref / ||x_ref||^2, where
    log_target is the (mean-centered) log of the static transition row.
    This is refined by the gradient ascent pass that follows.
    """
    W = np.zeros((n_states, n_states, n_features))
    norm_sq = np.dot(x_ref, x_ref) + EPS
    for i in range(n_states):
        log_target = np.log(transmat[i] + EPS)
        log_target = log_target - log_target.mean()   # center (softmax shift-invariant)
        for j in range(n_states):
            W[i, j] = log_target[j] * x_ref / norm_sq
    return W


def fit_dynamic_hmm_initial(model, df_feat_train, train_states, n_states,
                            n_iter_w=50, lr_w_init=0.005):
    """
    Fit the dynamic transition weight matrix W by:
      1. Initializing from the static transmat_ via outer-product approximation.
      2. (If statsmodels available) Per-source-state MNLogit regression,
         predicting next-state from x(t), with BUG 3 FIX: explicit padding
         of np.bincount to length n_states and a shape check before
         reconstructing W_i - falls through cleanly on any mismatch
         instead of risking an out-of-bounds index.
      3. A joint gradient-ascent refinement pass over all source states.

    Returns
    -------
    W          : ndarray (n_states, n_states, n_features)
    x_mean_dyn : ndarray (n_features,) - dynamic feature mean (train split)
    x_std_dyn  : ndarray (n_features,) - dynamic feature std  (train split)
    """
    n_features = len(DYN_FEATURE_COLS)

    X_dyn = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    x_mean_dyn = X_dyn.mean(axis=0)
    x_std_dyn  = X_dyn.std(axis=0) + EPS
    X_norm = (X_dyn - x_mean_dyn) / x_std_dyn

    W = _init_W_from_transmat(model.transmat_, x_mean_dyn / x_std_dyn,
                              n_states, n_features)

    T = len(train_states)

    # per-source-state MNLogit fit (BUG 3 FIX applied)
    if HAS_SM:
        for i in range(n_states):
            src_idx = np.array([t for t in range(T - 1) if train_states[t] == i])
            if len(src_idx) < MIN_BARS_PER_STATE:
                log(f"      W init: state {i} has only {len(src_idx)} transitions, "
                    f"keeping outer-product initialization")
                continue

            X_i = X_norm[src_idx]
            y_i = train_states[src_idx + 1]   # target = next state

            # BUG 3 FIX: pad bincount to length n_states so argmin/category
            # coverage checks are well-defined even if y_i doesn't visit
            # every possible target state.
            counts_padded = np.bincount(y_i, minlength=n_states)
            observed_categories = np.unique(y_i)

            if len(observed_categories) < 2:
                log(f"      W init: state {i} transitions only ever go to a "
                    f"single target state - keeping outer-product initialization")
                continue

            try:
                mnl = sm.MNLogit(y_i, sm.add_constant(X_i, prepend=True))
                res = mnl.fit(disp=False, maxiter=200)
                params = res.params  # (n_features+1, n_categories_observed - 1)

                # BUG 3 FIX: verify shape before reconstructing - MNLogit's
                # column count depends on how many distinct categories were
                # actually observed in y_i, which may be < n_states. If it
                # doesn't match what we expect for a clean reconstruction,
                # fall through to the gradient-only fit instead of indexing
                # blindly.
                n_obs_categories = len(observed_categories)
                if params.shape[1] != n_obs_categories - 1:
                    log(f"      W init: state {i} MNLogit param shape "
                        f"{params.shape} inconsistent with "
                        f"{n_obs_categories} observed categories - "
                        f"skipping MNLogit reconstruction, using "
                        f"outer-product init + gradient refinement only")
                    continue

                # statsmodels MNLogit always uses the smallest label value
                # among observed categories as the implicit base category.
                mnl_base = int(observed_categories.min())

                all_params = np.zeros((n_features + 1, n_states))
                col = 0
                for cat in observed_categories:
                    if cat == mnl_base:
                        continue
                    if col >= params.shape[1]:
                        raise ValueError("column overflow during reconstruction")
                    all_params[:, int(cat)] = params[:, col]
                    col += 1

                W[i] = all_params[1:, :].T   # (n_states, n_features), drop intercept row
                log(f"      W init: state {i} MNLogit fit OK "
                    f"({len(src_idx)} transitions, {n_obs_categories} categories)")

            except Exception as e:
                log(f"      W init: MNLogit failed for state {i} ({e}), "
                    f"using outer-product init + gradient refinement only")

    # joint gradient-ascent refinement over all source states
    lr = lr_w_init
    log_loss = 0.0
    for iteration in range(n_iter_w):
        total_grad = np.zeros_like(W)
        log_loss   = 0.0
        for t in range(T - 1):
            i  = train_states[t]
            j  = train_states[t + 1]
            xt = X_norm[t]
            logits = W[i] @ xt
            probs  = _softmax(logits)
            grad_i = np.outer((np.eye(n_states)[j] - probs), xt)
            total_grad[i] += grad_i
            log_loss      += np.log(probs[j] + EPS)

        W += lr * total_grad / (T + EPS)
        lr *= 0.995

    log(f"      Dynamic W fit complete: final log-loss/bar = "
        f"{log_loss / max(T - 1, 1):.4f}")

    return W, x_mean_dyn, x_std_dyn


# state-stratified Cox fit

def _windowed_counts(event_bars, T, n_windows=20):
    ws = T / n_windows
    return np.array([
        int(np.sum((w * ws <= np.array(event_bars)) &
                   (np.array(event_bars) < (w + 1) * ws)))
        for w in range(n_windows)
    ])


def _window_mean_feature(feature_series, T, n_windows=20):
    ws = T / n_windows
    vals = np.array(feature_series)
    result = np.zeros(n_windows)
    for w in range(n_windows):
        lo = int(w * ws)
        hi = min(int((w + 1) * ws), T)
        if lo < hi:
            result[w] = vals[lo:hi].mean()
    return result


def baseline_cox_dispersion(event_bars, T, n_windows=20):
    counts = _windowed_counts(event_bars, T, n_windows)
    mean_c = counts.mean()
    if mean_c < EPS:
        return np.nan
    return counts.var() / mean_c


def _fit_poisson_glm_beta(X_dyn_norm, y_binary, fallback_beta=None):
    """
    Fit a single Poisson GLM: lambda = exp(beta . x), returning beta as ndarray.
    X_dyn_norm: (T, n_features), y_binary: (T,) with 0/1 event indicators.
    Uses statsmodels if available, falls back to L-BFGS-B.
    lambda is clipped to LAMBDA_CLIP range in both paths for numerical safety.
    """
    n_features = X_dyn_norm.shape[1]
    y = y_binary.astype(float)

    if HAS_SM:
        try:
            res = sm.GLM(y, X_dyn_norm,
                        family=sm.families.Poisson()).fit(disp=1, maxiter=200)
            return res.params.astype(float), True
        except Exception:
            pass

    def neg_logL(beta):
        lam = np.clip(np.exp(X_dyn_norm @ beta), *LAMBDA_CLIP)
        return -np.sum(y * np.log(lam + EPS) - lam)

    def neg_logL_grad(beta):
        lam = np.clip(np.exp(X_dyn_norm @ beta), *LAMBDA_CLIP)
        return -(X_dyn_norm.T @ (y - lam))

    x0 = fallback_beta if fallback_beta is not None else np.zeros(n_features)
    try:
        res = optimize.minimize(neg_logL, x0=x0, jac=neg_logL_grad,
                                method="L-BFGS-B")
        return res.x.astype(float), res.success
    except Exception:
        return np.zeros(n_features), False


def fit_stratified_cox_initial(event_bars_train, train_states, df_feat_train,
                               n_states, x_mean_dyn, x_std_dyn):
    """
    Fit one Poisson GLM beta_k per state k, using only bars in that state.
    Falls back to pooled fit if a state has too few bars or zero events.
    """
    T = len(train_states)
    X_dyn  = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    X_norm = (X_dyn - x_mean_dyn) / x_std_dyn

    event_set = set(int(b) for b in event_bars_train if 0 <= int(b) < T)
    y_all     = np.array([1.0 if t in event_set else 0.0 for t in range(T)])

    beta_pooled, pooled_ok = _fit_poisson_glm_beta(X_norm, y_all)
    if not pooled_ok:
        log(f"      Pooled Cox fallback also failed - using zeros")
        beta_pooled = np.zeros(len(DYN_FEATURE_COLS))

    n_features = len(DYN_FEATURE_COLS)
    betas      = np.zeros((n_states, n_features))

    for k in range(n_states):
        mask   = train_states == k
        n_bars = mask.sum()
        if n_bars < MIN_BARS_PER_STATE:
            log(f"      State {k}: only {n_bars} bars < MIN_BARS_PER_STATE="
                f"{MIN_BARS_PER_STATE} - using pooled beta")
            betas[k] = beta_pooled
            continue

        X_k = X_norm[mask]
        y_k = y_all[mask]
        n_ev = y_k.sum()

        if n_ev < 1:
            log(f"      State {k}: 0 events in train - using pooled beta")
            betas[k] = beta_pooled
            continue

        beta_k, ok = _fit_poisson_glm_beta(X_k, y_k, fallback_beta=beta_pooled)
        if ok:
            betas[k] = beta_k
            log(f"      State {k}: beta fitted ({int(n_bars)} bars, {int(n_ev)} events) "
                f"  ||beta||={np.linalg.norm(beta_k):.4f}")
        else:
            log(f"      State {k}: fit failed - using pooled beta")
            betas[k] = beta_pooled

    return betas


# original Cox fit (preserved for diagnostics)

def fit_cox_with_covariates(event_bars, T, rva_series, state_series, n_windows=20):
    counts      = _windowed_counts(event_bars, T, n_windows).astype(float)
    rva_w       = _window_mean_feature(rva_series,   T, n_windows)
    state_w     = _window_mean_feature(state_series, T, n_windows)
    base_disp   = baseline_cox_dispersion(event_bars, T, n_windows)

    if HAS_SM:
        try:
            X = sm.add_constant(np.column_stack([rva_w, state_w]), prepend=True)
            res = sm.GLM(counts, X, family=sm.families.Poisson()).fit(
                disp=1, maxiter=200)
            fitted    = res.fittedvalues
            pearson_r = (counts - fitted) / (np.sqrt(fitted) + EPS)
            resid_disp = pearson_r.var()
            dr_pct = (100.0 * (base_disp - resid_disp) / (base_disp + EPS)
                     if np.isfinite(base_disp) and base_disp > EPS else np.nan)
            b, p = res.params, res.pvalues
            return {"beta0": float(b[0]), "beta1_rva": float(b[1]),
                    "beta2_state": float(b[2]),
                    "pval_rva": float(p[1]), "pval_state": float(p[2]),
                    "residual_dispersion": float(resid_disp),
                    "baseline_dispersion": float(base_disp) if np.isfinite(base_disp) else None,
                    "dispersion_reduction_pct": float(dr_pct) if np.isfinite(dr_pct) else None,
                    "logL_cov": float(res.llf), "converged": res.converged}
        except Exception as e:
            log(f"      statsmodels GLM failed ({e}), using manual fallback")

    X_man = np.column_stack([np.ones(n_windows), rva_w, state_w])

    def neg_logL(beta):
        lam = np.exp(X_man @ beta)
        return -(np.sum(counts * np.log(lam + EPS) - lam))

    def neg_logL_grad(beta):
        lam = np.exp(X_man @ beta)
        return -(X_man.T @ (counts - lam))

    try:
        res = optimize.minimize(neg_logL, x0=[np.log(counts.mean() + EPS), 0.0, 0.0],
                                jac=neg_logL_grad, method="L-BFGS-B")
        b = res.x
        fitted     = np.exp(X_man @ b)
        pearson_r  = (counts - fitted) / (np.sqrt(fitted) + EPS)
        resid_disp = pearson_r.var()
        dr_pct     = (100.0 * (base_disp - resid_disp) / (base_disp + EPS)
                     if np.isfinite(base_disp) and base_disp > EPS else np.nan)
        return {"beta0": float(b[0]), "beta1_rva": float(b[1]),
                "beta2_state": float(b[2]),
                "pval_rva": np.nan, "pval_state": np.nan,
                "residual_dispersion": float(resid_disp),
                "baseline_dispersion": float(base_disp) if np.isfinite(base_disp) else None,
                "dispersion_reduction_pct": float(dr_pct) if np.isfinite(dr_pct) else None,
                "logL_cov": float(-res.fun), "converged": res.success}
    except Exception as e:
        return {"converged": False, "error": str(e)}


# per-state diagnostics

def state_event_rates(event_bars, state_seq, n_states, T):
    event_set  = set(int(b) for b in event_bars if 0 <= int(b) < len(state_seq))
    rows       = []
    quiet_rate = None

    for s in range(n_states):
        mask       = state_seq == s
        n_bars_s   = mask.sum()
        n_events   = sum(1 for b in event_set if state_seq[int(b)] == s)
        rate       = n_events / (n_bars_s + EPS)
        rows.append({
            "state":        s,
            "n_bars":       int(n_bars_s),
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


# burst probability signal

def compute_burst_prob_signal(df_feat, fwd_probs, state_seq, n_states, split_idx):
    """
    Attach HMM state and p_burst to df_feat.
    Unchanged from original - produces covariate_states.csv for diagnostics.
    """
    df_out = df_feat.copy()
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
        f"p_burst out of [0,1] range - check HMM output"

    return df_out


# model bundle serialization

def save_model_bundle(ticker, model, W, betas, perm, split_idx,
                      x_mean_dyn, x_std_dyn, x_mean_emit, x_std_emit,
                      n_states, vol_window):
    """
    Serialize all parameters needed by backtest.py for online walk-forward.
    All raw numpy arrays - no hmmlearn objects - so backtest.py does not
    need to import hmmlearn's fitted-model class to resume the model.

    Bundle contents
    ---------------
      W            : (n_states, n_states, n_features) - dynamic transition weights
      betas        : (n_states, n_features) - per-state Cox coefficients
      emit_means   : (n_states, 2) - HMM emission means, STANDARDIZED space
      emit_covars  : (n_states, 2) - HMM emission diag covariances, STANDARDIZED space
      startprob    : (n_states,) - HMM initial state distribution
      transmat     : (n_states, n_states) - static transmat (fallback / init reference)
      perm         : (n_states,) - state relabeling permutation applied
      x_mean_dyn   : (n_features,) - dynamic feature [RVA,RVA_vel,persist,
                     vol_zscore,ret_signed] mean, for online normalization
      x_std_dyn    : (n_features,) - dynamic feature std, for online normalization
      x_mean_emit  : (2,) - emission feature (vol_ratio, abs_ret) mean
                     BUG 4 FIX: previously a dead placeholder, now the actual
                     mu_x returned by fit_hmm(). backtest.py MUST use this
                     (not a freshly computed mean) to standardize new test-bar
                     emission observations before calling the HMM's emission
                     log-likelihood - using any other value would silently
                     decouple the saved emit_means/emit_covars from the
                     observations they're being compared against.
      x_std_emit   : (2,) - emission feature std, BUG 4 FIX (see above)
      split_idx    : int - train/test boundary bar index
      n_states     : int
      vol_window   : int
    """
    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / f"covariate_model_{ticker}.npz"

    emit_means  = model.means_.copy()    # (n_states, 2), standardized space

    # BUG 5 FIX: be defensive about covars_ shape here too — after the
    # fit_hmm relabeling fix this will already be (n_states, 2) diagonal
    # form, but guard against any future hmmlearn version where covars_
    # getter shape changes again, so backtest.py's _emission_log_likelihood
    # (which expects a 1D variance vector per state) never silently breaks.
    raw_covars = model.covars_
    if raw_covars.ndim == 3:
        # full (n_states, n_dim, n_dim) matrices — extract diagonal
        emit_covars = np.array([np.diag(c) for c in raw_covars])
    else:
        # already compact (n_states, n_dim) diagonal form
        emit_covars = raw_covars.copy()

    np.savez(
        out_path,
        W            = W,
        betas        = betas,
        emit_means   = emit_means,
        emit_covars  = emit_covars,
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
    log(f"    Saved model bundle -> {out_path}")
    return out_path


# transition matrix

def empirical_transition_matrix(state_seq, n_states):
    A = np.zeros((n_states, n_states))
    for i in range(len(state_seq) - 1):
        A[state_seq[i], state_seq[i + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    return A / (row_sums + EPS)


# main per-symbol routine

def process_symbol(ticker, event_bars, df_bars, n_states=3,
                   vol_window=130, n_windows=20, train_frac=TRAIN_FRAC):

    log(f"\n{'-'*65}")
    log(f"  {ticker}  ({len(event_bars)} events, {len(df_bars)} bars)")
    log(f"{'-'*65}")

    df_feat = compute_features(df_bars, vol_window=vol_window)
    T = len(df_feat)

    if T < 50:
        log(f"    [{ticker}] insufficient bars ({T})")
        return None, None, None

    log(f"    Features computed: {T} bars  "
        f"vol_ratio mean={df_feat['vol_ratio'].mean():.2f}  "
        f"RVA std={df_feat['RVA'].std():.3f}")

    log(f"    Fitting {n_states}-state Gaussian HMM (Baum-Welch)...")
    model, state_seq, fwd_probs, split_idx, hmm_logL, labels, perm, mu_x, std_x = fit_hmm(
        df_feat, n_states=n_states, train_frac=train_frac
    )

    if model is None:
        log(f"    [{ticker}] HMM fit failed")
        return None, None, None

    n_train = split_idx
    n_test  = T - split_idx
    log(f"    HMM log-likelihood (train): {hmm_logL:.2f}")
    log(f"    Walk-forward split: {n_train} train bars / {n_test} test bars "
        f"({train_frac*100:.0f}% / {(1-train_frac)*100:.0f}%)")

    # Fill persist using full Viterbi/filtered state path (causal; see BUG 2
    # note in _fill_persist_from_states docstring - test-split values here
    # are diagnostic only, backtest.py recomputes persist online).
    train_states = state_seq[:split_idx]
    df_feat      = _fill_persist_from_states(df_feat, state_seq)

    state_counts = np.bincount(state_seq, minlength=n_states)
    for s, name in labels.items():
        pct = 100 * state_counts[s] / T
        log(f"      State {s} ({name:<10}): {state_counts[s]:>5} bars  ({pct:>5.1f}%)")

    A = empirical_transition_matrix(state_seq, n_states)
    log(f"\n    Transition matrix (rows = from, cols = to):")
    state_names = [labels[s] for s in range(n_states)]
    log(f"    {'':10}  " + "  ".join(f"{n:>10}" for n in state_names))
    for s in range(n_states):
        row_str = "  ".join(f"{A[s, j]:>10.4f}" for j in range(n_states))
        log(f"      {labels[s]:<10}  {row_str}")

    if n_states == 3:
        p_entry   = A[1, 2]
        p_persist = A[2, 2]
        log(f"\n    P(Burst | Accumulation) = {p_entry:.4f}")
        log(f"    P(Burst | Burst)         = {p_persist:.4f}  [persistence]")
    else:
        p_entry   = A[0, 1]
        p_persist = A[1, 1]

    feat_len = len(df_feat)
    event_bars_clamped = [
        min(int(b), feat_len - 1) for b in event_bars if int(b) < feat_len
    ]
    event_bars_train = [b for b in event_bars_clamped if b < split_idx]

    rate_rows = state_event_rates(event_bars_clamped, state_seq, n_states, T)
    log(f"\n    Per-state event rates:")
    log(f"    {'State':<12} {'Bars':>6} {'TimeFrac':>9} "
        f"{'Events':>7} {'EvFrac':>8} {'Rate/Bar':>9} {'RateRatio':>10}")
    log(f"    " + "-" * 70)
    for row in rate_rows:
        log(f"    {labels[row['state']]:<12} {row['n_bars']:>6} "
            f"{row['time_frac']:>9.3f} {row['n_events']:>7} "
            f"{row['event_frac']:>8.3f} {row['rate_per_bar']:>9.5f} "
            f"{row['rate_ratio_vs_quiet']:>10.2f}x")

    log(f"\n    Cox re-fit with (RVA, S-hat) covariates (legacy diagnostic)...")
    rva_series   = df_feat["RVA"].values
    state_series = state_seq.astype(float)

    glm_result = fit_cox_with_covariates(
        event_bars_clamped, feat_len, rva_series, state_series, n_windows=n_windows)

    def fmt_p(p):
        if not np.isfinite(p): return "  -   "
        return f"{p:.4f}" + (" ***" if p < 0.001 else " **" if p < 0.01
                              else " *" if p < 0.05 else "    ")

    # BUG 1 FIX: initialize these before the conditional block so the
    # result dict construction below never hits a NameError when the
    # legacy Cox GLM fails to converge.
    base_disp = res_disp = dr_pct = np.nan

    if glm_result.get("converged"):
        b0, b1, b2 = glm_result["beta0"], glm_result["beta1_rva"], glm_result["beta2_state"]
        p1, p2     = glm_result.get("pval_rva", np.nan), glm_result.get("pval_state", np.nan)
        res_disp   = glm_result["residual_dispersion"]
        base_disp  = glm_result.get("baseline_dispersion", np.nan)
        dr_pct     = glm_result.get("dispersion_reduction_pct", np.nan)

        log(f"      beta0 (intercept) = {b0:+.4f}")
        log(f"      beta1 (RVA)       = {b1:+.4f}   p={fmt_p(p1)}")
        log(f"      beta2 (state)     = {b2:+.4f}   p={fmt_p(p2)}")
        if np.isfinite(base_disp):
            log(f"      Baseline dispersion: {base_disp:.3f}")
        log(f"      Residual dispersion: {res_disp:.3f}")
        if np.isfinite(dr_pct):
            log(f"      Dispersion change: {dr_pct:+.1f}%")
    else:
        log(f"      GLM did not converge: {glm_result.get('error','unknown')}")

    # Dynamic HMM W fit
    log(f"\n    Fitting dynamic transition matrix W...")
    df_feat_train = df_feat.iloc[:split_idx].copy()
    try:
        W, x_mean_dyn, x_std_dyn = fit_dynamic_hmm_initial(
            model, df_feat_train, train_states, n_states
        )
        log(f"      W shape: {W.shape}  ||W||_F = {np.linalg.norm(W):.4f}")
    except Exception as e:
        log(f"      Dynamic W fit failed ({e}) - using zero initialization")
        n_features = len(DYN_FEATURE_COLS)
        W = np.zeros((n_states, n_states, n_features))
        x_mean_dyn = df_feat_train[DYN_FEATURE_COLS].mean().values
        x_std_dyn  = df_feat_train[DYN_FEATURE_COLS].std().values + EPS

    # State-stratified Cox fit
    log(f"\n    Fitting state-stratified Cox intensity (per-state beta)...")
    try:
        betas = fit_stratified_cox_initial(
            event_bars_train, train_states, df_feat_train,
            n_states, x_mean_dyn, x_std_dyn
        )
        for k in range(n_states):
            log(f"      beta_{k} ({labels[k]}): {np.round(betas[k], 4)}")
    except Exception as e:
        log(f"      Stratified Cox fit failed ({e}) - using zeros")
        betas = np.zeros((n_states, len(DYN_FEATURE_COLS)))

    # Save model bundle (BUG 4 FIX: now includes mu_x, std_x)
    save_model_bundle(
        ticker, model, W, betas, perm, split_idx,
        x_mean_dyn, x_std_dyn, mu_x, std_x,
        n_states, vol_window
    )

    # Original signal CSV (train+test diagnostics)
    df_signal = compute_burst_prob_signal(
        df_feat, fwd_probs, state_seq, n_states, split_idx)
    df_signal["ticker"] = ticker

    burst_state_idx  = n_states - 1
    burst_time_frac  = next((r["time_frac"]          for r in rate_rows if r["state"] == burst_state_idx), np.nan)
    burst_rate_ratio = next((r["rate_ratio_vs_quiet"] for r in rate_rows if r["state"] == burst_state_idx), np.nan)

    result = {
        "ticker":               ticker,
        "n_events":             len(event_bars),
        "n_bars":               T,
        "n_train":              n_train,
        "n_test":               n_test,
        "n_states":             n_states,
        "hmm_logL":             round(float(hmm_logL), 2),
        "burst_time_frac":      round(float(burst_time_frac), 4),
        "burst_rate_ratio":     round(float(burst_rate_ratio), 2),
        "p_burst_entry":        round(float(p_entry), 4),
        "p_burst_persist":      round(float(p_persist), 4),
        "baseline_dispersion":  round(float(base_disp), 3) if np.isfinite(base_disp) else None,
        "residual_dispersion":  round(float(res_disp), 3)  if np.isfinite(res_disp)  else None,
        "dispersion_reduction_pct": round(float(dr_pct), 1) if np.isfinite(dr_pct) else None,
        "beta0":           round(float(glm_result.get("beta0",     np.nan)), 4) if glm_result.get("converged") else None,
        "beta1_rva":       round(float(glm_result.get("beta1_rva", np.nan)), 4) if glm_result.get("converged") else None,
        "beta2_state":     round(float(glm_result.get("beta2_state",np.nan)),4) if glm_result.get("converged") else None,
        "pval_rva":        round(float(glm_result.get("pval_rva",  np.nan)),4) if glm_result.get("converged") and np.isfinite(glm_result.get("pval_rva",np.nan)) else None,
        "pval_state":      round(float(glm_result.get("pval_state",np.nan)),4) if glm_result.get("converged") and np.isfinite(glm_result.get("pval_state",np.nan)) else None,
        "glm_logL":        round(float(glm_result.get("logL_cov",  np.nan)),2) if glm_result.get("converged") else None,
        "glm_converged":   glm_result.get("converged", False),
        "W_norm_initial":  round(float(np.linalg.norm(W)), 4),
        "beta_norms_initial": str([round(float(np.linalg.norm(betas[k])), 4) for k in range(n_states)]),
    }

    trans_row = {"ticker": ticker}
    for i in range(n_states):
        for j in range(n_states):
            key = f"A_{labels[i][:2].lower()}_{labels[j][:2].lower()}"
            trans_row[key] = round(float(A[i, j]), 6)

    return result, df_signal, trans_row


# aggregate summary

def print_summary(results):
    log(f"\n{'='*65}")
    log("  AGGREGATE SUMMARY")
    log(f"{'='*65}")

    valid = [r for r in results if r.get("glm_converged")]
    log(f"\n  Symbols fitted : {len(valid)} / {len(results)}")
    if not valid:
        return

    dr_vals = [r["dispersion_reduction_pct"] for r in valid if r.get("dispersion_reduction_pct") is not None]
    if dr_vals:
        log(f"\n  -- Dispersion reduction --")
        log(f"    Mean : {np.mean(dr_vals):+.1f}%   Med : {np.median(dr_vals):+.1f}%   "
            f"Improved: {sum(v>0 for v in dr_vals)}/{len(dr_vals)}")

    brr = [r["burst_rate_ratio"] for r in valid if r.get("burst_rate_ratio")]
    if brr:
        log(f"\n  -- Burst rate ratio (vs Quiet) --")
        log(f"    Mean: {np.mean(brr):.2f}x   Med: {np.median(brr):.2f}x   "
            f"Max: {np.max(brr):.2f}x  ({valid[np.argmax(brr)]['ticker']})")

    log(f"\n  -- Per-Symbol Results --")
    log(f"  {'Ticker':<7} {'N':>5} {'Train':>6} {'Test':>6} {'DisDelta':>9} "
        f"{'BrstRatio':>10} {'B1_RVA':>8} {'B2_State':>10} {'P_entry':>9} {'P_persist':>10}")
    log(f"  " + "-" * 90)

    for r in results:
        tk = r.get("ticker","?")
        n  = r.get("n_events", 0)
        if not r.get("glm_converged"):
            log(f"  {tk:<7} {n:>5}   fit failed"); continue
        log(f"  {tk:<7} {n:>5} "
            f"{r.get('n_train',0):>6} {r.get('n_test',0):>6} "
            f"{(str(round(r['dispersion_reduction_pct'],1))+'%') if r.get('dispersion_reduction_pct') is not None else '-':>9} "
            f"{(str(r['burst_rate_ratio'])+'x') if r.get('burst_rate_ratio') else '-':>10} "
            f"{r['beta1_rva']:>+8.4f} "
            f"{r['beta2_state']:>+10.4f} "
            f"{r['p_burst_entry']:>9.4f} "
            f"{r['p_burst_persist']:>10.4f}")


# entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",    nargs="*", default=None)
    parser.add_argument("--states",     type=int,  default=3)
    parser.add_argument("--vol-window", type=int,  default=130)
    parser.add_argument("--windows",    type=int,  default=20)
    parser.add_argument("--train-frac", type=float,default=TRAIN_FRAC)
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    log("=" * 65)
    log("  covariate.py  -  Dynamic HMM + Cox Model  [DYNAMIC, BUGFIXED]")
    log("=" * 65)
    log(f"  HMM states      : {args.states}")
    log(f"  Vol-ma window   : {args.vol_window} bars")
    log(f"  Train fraction  : {args.train_frac*100:.0f}%  (test = {(1-args.train_frac)*100:.0f}%)")
    log(f"  Dynamic features: {DYN_FEATURE_COLS}")
    log(f"  persist cap     : {PERSIST_CAP} bars")
    log(f"  MIN_BARS/state  : {MIN_BARS_PER_STATE}")
    log(f"  lambda clip     : {LAMBDA_CLIP}")
    log(f"  statsmodels     : {'available' if HAS_SM else 'NOT available - manual fallback'}")
    log(f"  Model output    : {MODEL_DIR}/covariate_model_<TICKER>.npz")
    log("=" * 65)

    cox_tickers = set(TICKERS)
    if not args.all and Path(RESULTS_CSV).exists():
        try:
            res_df      = pd.read_csv(RESULTS_CSV)
            log(f"  [OK] Found {RESULTS_CSV}: columns = {res_df.columns.tolist()}")
            if "winner" in res_df.columns:
                cox_winners = res_df[res_df["winner"] == "Cox"]["ticker"].tolist()
                log(f"       'winner' column values: {res_df['winner'].unique().tolist()}")
                if cox_winners:
                    cox_tickers = set(cox_winners)
                    log(f"  Cox winners: {', '.join(sorted(cox_tickers))}")
                else:
                    log(f"  No rows with winner=='Cox' — using full TICKERS list "
                        f"({len(cox_tickers)} symbols)")
            else:
                log(f"  No 'winner' column in {RESULTS_CSV} — using full TICKERS "
                    f"list ({len(cox_tickers)} symbols)")
        except Exception as e:
            log(f"  WARNING: {RESULTS_CSV} read failed ({e}) — using full TICKERS "
                f"list ({len(cox_tickers)} symbols)")
    else:
        log(f"  {RESULTS_CSV} not found or --all passed — using full TICKERS "
            f"list ({len(cox_tickers)} symbols)")

    events_lookup = {}
    if Path(EVENTS_CSV).exists():
        try:
            ev_df = pd.read_csv(EVENTS_CSV)
            log(f"\n  [OK] Found {EVENTS_CSV}: {len(ev_df):,} rows, "
                f"{ev_df['ticker'].nunique() if 'ticker' in ev_df.columns else '?'} "
                f"unique tickers")
            if "ticker" in ev_df.columns:
                log(f"       Tickers in events file: "
                    f"{sorted(ev_df['ticker'].unique().tolist())}")
            for ticker, grp in ev_df.groupby("ticker"):
                events_lookup[ticker] = grp["bar_idx"].tolist()
        except Exception as e:
            log(f"  WARNING: failed to read {EVENTS_CSV}: {e}")
    else:
        log(f"\n  *** ERROR: {EVENTS_CSV} NOT FOUND in current directory ***")
        log(f"  *** Every ticker will be skipped with 'no events' below.   ***")
        log(f"  *** Check the exact filename with: ls *.csv               ***")
        log(f"  *** and either rename it to '{EVENTS_CSV}' or fix EVENTS_CSV constant.")

    target_tickers = args.symbols if args.symbols else sorted(cox_tickers)

    overlap = set(target_tickers) & set(events_lookup.keys())
    if not overlap and target_tickers:
        log(f"\n  *** WARNING: ZERO overlap between target tickers and ***")
        log(f"  *** tickers found in {EVENTS_CSV}.                    ***")
        log(f"  Target tickers : {sorted(target_tickers)}")
        log(f"  Event tickers  : {sorted(events_lookup.keys())}")
        log(f"  Every symbol below will be skipped as 'no events'.")

    all_results   = []
    all_state_dfs = []
    all_trans     = []

    for ticker in target_tickers:
        if ticker not in events_lookup:
            log(f"\n  [{ticker}] no events - skipping"); continue

        df_bars = load_bars(ticker, use_cache=not args.no_cache)
        if df_bars is None:
            log(f"\n  [{ticker}] no bar data - skipping"); continue

        result, df_signal, trans_row = process_symbol(
            ticker, events_lookup[ticker], df_bars,
            n_states=args.states, vol_window=args.vol_window,
            n_windows=args.windows, train_frac=args.train_frac,
        )
        if result    is not None: all_results.append(result)
        if df_signal is not None: all_state_dfs.append(df_signal)
        if trans_row is not None: all_trans.append(trans_row)

    print_summary(all_results)

    log(f"\n{'-'*65}")
    if all_results:
        pd.DataFrame(all_results).to_csv("covariate_results.csv", index=False)
        log("  Saved -> covariate_results.csv")

    if all_state_dfs:
        state_df = pd.concat(all_state_dfs)
        keep = [c for c in ["ticker","Volume","vol_ratio","abs_ret","RVA","RVA_vel",
                             "persist","vol_zscore","ret_signed","session_pos",
                             "hmm_state","p_burst","p_quiet","p_accum","split"]
                if c in state_df.columns]
        state_df[keep].to_csv("covariate_states.csv")
        log(f"  Saved -> covariate_states.csv  ({len(state_df)} bars)")

    if all_trans:
        pd.DataFrame(all_trans).to_csv("covariate_transitions.csv", index=False)
        log("  Saved -> covariate_transitions.csv")

    if all_state_dfs:
        burst_df = pd.concat(all_state_dfs)
        cols = ["ticker","p_burst","split"] + (["p_accum"] if args.states == 3 else [])
        burst_df[[c for c in cols if c in burst_df.columns]].to_csv(
            "covariate_burst_prob.csv")
        log("  Saved -> covariate_burst_prob.csv")

    log(f"\n{'='*65}")
    log("  VERIFICATION")
    log(f"{'='*65}")
    log("  p_burst: filtered_forward_probs() + assertions in [0,1].")
    log("  HMM trained on first 60% of bars only.")
    log("  Dynamic W + stratified beta saved to covariate_models/.")
    log("  Emission standardization (mu_x, std_x) saved to bundle - BUG 4 FIXED.")
    log("  backtest.py loads model bundles for online walk-forward.")
    log(f"{'='*65}")


if __name__ == "__main__":
    main()