"""
covariatec.py  —  HMM + Cox Latent Regime Model for Crypto
              [v5 — Forward Price Event + Small-Cap Alts + 5m Native]

EVENT DEFINITION (v5)
---------------------
  Event = abnormal price increase:
    close[t + fwd_horizon] / close[t] - 1  >  fwd_thresh
  e.g. 10% gain over the next 5 bars (25 minutes at 5m).

  This makes the Cox GLM directly model the probability of the thing
  the backtest profits from.  No volume proxy, no abs_ret ambiguity —
  pure directional forward-return labelling.

  The HMM emission features (ret_signed, abs_ret, vol_ratio) are
  unchanged — the model still detects latent regimes from price/vol
  dynamics.  Only the event definition changes.

  IMPORTANT: events are labelled at bar t (the bar whose forward
  return exceeds the threshold), NOT at t+fwd_horizon.  The HMM state
  at bar t is what predicts the upcoming move — no future leakage.

ARCHITECTURE
------------
  - 5-dim dynamic feature vector: ret_signed, abs_ret, RVA, RVA_vel,
                                   persist  (volume kept as context)
  - HMM emission: ret_signed, abs_ret  (price-centric, no vol in emit)
  - Input-coupled dynamic transition matrix W
  - State-stratified Cox intensity (separate betas per state)
  - p_burst: filtered forward algorithm — no future information

PAGINATION
----------
  OKX spot → swap → Binance fallback.
  Checkpointed every CHECKPOINT_EVERY_PAGES pages.
  Resumable from last checkpoint bar.

OUTPUTS
-------
  crypto_covariate_states.csv
  crypto_covariate_results.csv
  crypto_covariate_transitions.csv
  crypto_events.csv
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from scipy import stats, optimize
from hmmlearn import hmm

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("ERROR: ccxt not installed. Run: pip install ccxt")

# ── universe — small-cap alts with thin order books ───────────────────────────
# Selected for low circulating supply, episodic volume surges, illiquid books.
# These behave like low-float stocks: a single buyer visibly moves price over
# several bars, creating the Accum→Burst sequence the HMM is designed to detect.
# Excludes BTC/ETH/BNB/SOL — too liquid, no float constraint.
DEFAULT_SYMBOLS = [
    # micro-cap / thin book alts
    "MINA/USDT",   "ZIL/USDT",    "ONE/USDT",    "ASTR/USDT",
    "KAVA/USDT",   "REN/USDT",    "BNT/USDT",    "UMA/USDT",
    "BAL/USDT",    "KNC/USDT",    "LRC/USDT",    "ZRX/USDT",
    "OMG/USDT",    "SNX/USDT",    "COMP/USDT",
    # higher-vol small-caps with episodic bursts
    "INJ/USDT",    "SUI/USDT",    "APT/USDT",
    "GMX/USDT",    "DYDX/USDT",
]

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR           = Path("crypto_cache")
TIMEFRAME           = "5m"
LOOKBACK_DAYS       = 365
OKX_PAGE_SIZE       = 300
MIN_EVENTS          = 10
MIN_BARS            = 500          # raised — 5m needs more bars for vol_window
MIN_BURST_BARS      = 5
TRAIN_FRAC          = 0.60
EPS                 = 1e-12
RATE_LIMIT_MS       = 300
CHECKPOINT_EVERY_PAGES = 20
PERSIST_CAP         = 50
MIN_BARS_PER_STATE  = 30
MODEL_DIR           = Path("crypto_models")

# Feature columns
# Emission: price-centric — HMM detects regimes from directional + magnitude
# Dynamic:  full 5-dim vector for transition matrix W and Cox intensity
DYN_FEATURE_COLS  = ["ret_signed", "abs_ret", "RVA", "RVA_vel", "persist"]
EMIT_FEATURE_COLS = ["ret_signed", "abs_ret"]

TF_RESAMPLE = {
    "5m":  "5min",  "10m": "10min", "15m": "15min",
    "30m": "30min", "45m": "45min", "1h":  "1h",
    "2h":  "2h",    "3h":  "3h",    "4h":  "4h",
    "1d":  "1D",
}

# ── forward-return event thresholds ──────────────────────────────────────────
# Event = price increase > FWD_THRESH over the next FWD_HORIZON bars.
# At 5m bars: FWD_HORIZON=5 means 25 minutes, FWD_THRESH=0.10 means 10%.
# A 10% gain in 25 minutes on a small-cap alt is a genuine burst event.
# VOL_WINDOW kept for HMM feature computation only (vol_ratio context).
FWD_HORIZON = 5        # bars ahead to measure forward return
FWD_THRESH  = 0.10     # 10% forward price increase = event
VOL_WINDOW  = 96       # 8h rolling window for vol_ratio feature (HMM only)

def log(msg=""): print(msg, flush=True)


# ── pagination (extended from v3) ─────────────────────────────────────────────

def _fetch_paginated(exchange, symbol, timeframe, since_ms, end_ms,
                     on_checkpoint=None, checkpoint_every=CHECKPOINT_EVERY_PAGES):
    all_bars           = []
    cursor             = since_ms
    got_full_page      = False
    page_count         = 0
    consecutive_errors = 0

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe,
                                        since=cursor, limit=OKX_PAGE_SIZE)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                log(f"      [{symbol}] 5 consecutive errors, stopping ({e})")
                break
            time.sleep(min(2 ** consecutive_errors, 30))
            continue

        if not bars:
            break

        bars = [b for b in bars if since_ms <= b[0] <= end_ms]
        if not bars:
            break

        last_seen = all_bars[-1][0] if all_bars else -1
        new_bars  = [b for b in bars if b[0] > last_seen]
        if not new_bars:
            break

        all_bars.extend(new_bars)
        page_count += 1

        if on_checkpoint is not None and page_count % checkpoint_every == 0:
            on_checkpoint(all_bars)

        if len(bars) >= OKX_PAGE_SIZE:
            got_full_page = True

        if len(bars) < OKX_PAGE_SIZE and got_full_page:
            break

        next_cursor = new_bars[-1][0] + 1
        if next_cursor > end_ms:
            break

        cursor = next_cursor
        time.sleep(RATE_LIMIT_MS / 1000)

    return all_bars


def fetch_ohlcv(symbol, timeframe=TIMEFRAME, lookback_days=LOOKBACK_DAYS,
                start=None, end=None, use_cache=True):
    CACHE_DIR.mkdir(exist_ok=True)
    safe_sym = symbol.replace("/", "_")

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    if end:
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    else:
        end_ms = now_ms
    if start:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    else:
        start_ms = end_ms - lookback_days * 24 * 3600 * 1000

    range_tag  = (f"{pd.Timestamp(start_ms, unit='ms').strftime('%Y%m%d')}_"
                  f"{pd.Timestamp(end_ms,   unit='ms').strftime('%Y%m%d')}")
    cache_path   = CACHE_DIR / f"{safe_sym}_{timeframe}_{range_tag}.parquet"
    partial_path = CACHE_DIR / f"{safe_sym}_{timeframe}_{range_tag}.partial.parquet"

    if use_cache and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if len(df) >= MIN_BARS:
                return df
        except Exception:
            pass

    if not HAS_CCXT:
        return None

    resume_df      = None
    fetch_start_ms = start_ms
    if use_cache and partial_path.exists():
        try:
            resume_df = pd.read_parquet(partial_path)
            if len(resume_df) > 0:
                last_ts_ms     = int(resume_df.index.max().timestamp() * 1000)
                fetch_start_ms = last_ts_ms + 1
                log(f"      [{symbol}] resuming from checkpoint: "
                    f"{len(resume_df)} bars cached, "
                    f"continuing from {pd.Timestamp(last_ts_ms, unit='ms')}")
        except Exception:
            resume_df      = None
            fetch_start_ms = start_ms

    attempts = [
        ("okx",     "spot", symbol),
        ("okx",     "swap", symbol.replace("/USDT", "/USDT:USDT")),
        ("binance", "spot", symbol),
    ]

    for exc_name, market_type, sym in attempts:
        try:
            if exc_name == "okx":
                exchange = ccxt.okx({
                    "enableRateLimit": True,
                    "options": {"defaultType": market_type},
                })
            else:
                exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": market_type},
                })

            def _checkpoint(bars_so_far, _resume_df=resume_df, _path=partial_path):
                try:
                    df_ck = pd.DataFrame(bars_so_far,
                                         columns=["timestamp","Open","High","Low",
                                                   "Close","Volume"])
                    df_ck["timestamp"] = pd.to_datetime(df_ck["timestamp"],
                                                        unit="ms", utc=True)
                    df_ck = df_ck.set_index("timestamp").sort_index()
                    if _resume_df is not None and len(_resume_df) > 0:
                        df_ck = pd.concat([_resume_df, df_ck])
                    df_ck = df_ck[~df_ck.index.duplicated(keep="first")].dropna()
                    df_ck.to_parquet(_path)
                except Exception:
                    pass

            bars = _fetch_paginated(exchange, sym, timeframe,
                                    fetch_start_ms, end_ms,
                                    on_checkpoint=_checkpoint)

            if not bars and resume_df is None:
                continue
            if not bars and resume_df is not None:
                df = resume_df.sort_index()
                df = df[~df.index.duplicated(keep="first")].dropna()
            else:
                df = pd.DataFrame(bars,
                                   columns=["timestamp","Open","High","Low",
                                             "Close","Volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                if resume_df is not None and len(resume_df) > 0:
                    df = pd.concat([resume_df, df])
                df = df[~df.index.duplicated(keep="first")].dropna()

            if len(df) >= MIN_BARS:
                df.to_parquet(cache_path)
                if partial_path.exists():
                    partial_path.unlink()
                actual_days = (df.index.max() - df.index.min()).days
                log(f"    Fetched {len(df):,} bars for {symbol} "
                    f"[{exc_name}/{market_type}]  "
                    f"({df.index.min().date()} → {df.index.max().date()}, "
                    f"~{actual_days}d)")
                return df

        except Exception as e:
            log(f"    [{symbol}] {exc_name}/{market_type} error: {e}")
            continue

    log(f"    [{symbol}] could not fetch enough bars from any source")
    return None


# ── event extraction — forward price increase ────────────────────────────────

def extract_events(df, symbol, fwd_horizon=FWD_HORIZON, fwd_thresh=FWD_THRESH):
    """
    Label bar t as an event if:
        close[t + fwd_horizon] / close[t] - 1  >  fwd_thresh

    Event is stamped at bar t — the bar whose HMM state we use to predict
    the upcoming move.  No future leakage: the HMM sees nothing beyond t.

    mark = actual forward return (useful for Cox GLM intensity scaling).
    """
    df = df.copy()
    df["fwd_ret"] = df["Close"].shift(-fwd_horizon) / df["Close"] - 1

    mask   = df["fwd_ret"] > fwd_thresh
    events = df[mask].copy()

    if events.empty:
        return None

    events["mark"] = events["fwd_ret"]
    result = events.reset_index()[["timestamp", "Volume", "fwd_ret", "mark"]]
    result.columns = ["date", "volume", "fwd_ret", "mark"]
    result["bar_idx"] = -1
    return result.reset_index(drop=True)


def realign_events(events, df_feat):
    feat_index  = pd.Index(df_feat.index)
    date_to_pos = {dt: i for i, dt in enumerate(feat_index)}
    aligned = []
    for _, row in events.iterrows():
        pos = date_to_pos.get(row["date"])
        if pos is not None:
            r = row.copy()
            r["bar_idx"] = pos
            aligned.append(r)
    if not aligned:
        return None
    return pd.DataFrame(aligned).reset_index(drop=True)


# ── feature engineering ──────────────────────────────────────────────────────

def compute_features(df, vol_window=VOL_WINDOW):
    """
    Compute per-bar features.

    HMM emission (EMIT_FEATURE_COLS): ret_signed, abs_ret
      — price-centric; HMM detects bullish/bearish regime from
        directional return magnitude, no volume in emission.

    Dynamic features (DYN_FEATURE_COLS): ret_signed, abs_ret, RVA, RVA_vel, persist
      — vol context kept via RVA/RVA_vel for W matrix and Cox betas.

    vol_ratio / vol_zscore also written to states CSV for pnlc.py
    sparse-entry filters (informational, not used in HMM fitting).
    """
    df = df.copy()
    min_p = max(5, vol_window // 4)

    # ── price features ────────────────────────────────────────────────────────
    df["ret_signed"] = (df["Close"] - df["Open"]) / (df["Open"] + EPS)
    df["abs_ret"]    = df["ret_signed"].abs()

    # ── price momentum / rolling context ─────────────────────────────────────
    df["ret_ma"]     = df["ret_signed"].rolling(vol_window,
                                                min_periods=min_p).mean()
    df["ret_zscore"] = (
        (df["ret_signed"] - df["ret_ma"])
        / (df["ret_signed"].rolling(vol_window).std() + EPS)
    ).fillna(0.0)

    # ── volume context (not in emit, kept for RVA and output) ────────────────
    df["vol_ma"]     = df["Volume"].rolling(vol_window, min_periods=min_p).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma"] + EPS)
    df["vol_zscore"] = (
        (df["vol_ratio"] - df["vol_ratio"].rolling(vol_window).mean())
        / (df["vol_ratio"].rolling(vol_window).std() + EPS)
    ).fillna(0.0)

    # RVA: volume acceleration — first diff of vol_ratio
    df["RVA"]        = df["vol_ratio"].diff().fillna(0.0)
    # RVA_vel: acceleration of acceleration
    df["RVA_vel"]    = df["RVA"].diff().fillna(0.0)

    # persist: filled after HMM state assignment
    df["persist"]    = 0.0

    df = df.dropna(subset=["vol_ma", "ret_signed", "abs_ret"])
    return df


def _fill_persist(df, state_seq):
    """Fill persist column: bars continuously in same state, capped."""
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


# ── filtered forward probabilities ───────────────────────────────────────────

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
        log_sum     = np.logaddexp.reduce(log_alpha[t])
        filtered[t] = np.exp(log_alpha[t] - log_sum)

    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6), \
        "INTEGRITY ERROR: filtered probs do not sum to 1"
    assert (filtered >= 0).all() and (filtered <= 1 + 1e-9).all(), \
        "INTEGRITY ERROR: filtered probs outside [0,1]"
    return filtered


# ── HMM fit ───────────────────────────────────────────────────────────────────

def fit_hmm(df_feat, n_states=3, n_iter=200, random_seed=42,
            train_frac=TRAIN_FRAC):
    obs_cols = EMIT_FEATURE_COLS   # ret_signed, abs_ret for emission
    X = df_feat[obs_cols].values.astype(float)
    X[:, 0] = np.clip(X[:, 0], -0.5, 0.5)   # ret_signed: ±50% bar
    X[:, 1] = np.clip(X[:, 1],  0.0, 0.5)   # abs_ret: 0–50%

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

    # Relabel states by ascending mean vol_ratio: Quiet=0, Accum=1, Burst=2
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

    # Do NOT permute model internals — only outputs are relabelled.
    # (matches covariate.py behaviour exactly)

    state_seq = np.concatenate([train_states, test_states]).astype(int)
    fwd_probs = (np.vstack([train_smoothed, test_filtered])
                 if len(test_filtered) > 0 else train_smoothed)

    labels = {0: "Quiet", 1: "Accum", 2: "Burst"} if n_states == 3 \
             else {0: "Quiet", 1: "Burst"}

    return (best_model, state_seq, fwd_probs, split_idx, best_logL,
            labels, perm, mu_x, std_x)


# ── dynamic transition matrix W (input-coupled) ───────────────────────────────

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

    W  = _init_W_from_transmat(model.transmat_,
                                x_mean_dyn / x_std_dyn,
                                n_states, n_features)
    T  = len(train_states)

    # MNLogit initialisation per source state
    if HAS_SM:
        for i in range(n_states):
            src_mask = np.array([t for t in range(T - 1)
                                  if train_states[t] == i])
            if len(src_mask) < MIN_BARS_PER_STATE:
                continue
            X_i = X_norm[src_mask]
            y_i = train_states[src_mask + 1]
            # Guard: MNLogit needs all n_states classes present in y_i.
            # If some next-states are never observed from state i, skip
            # and let gradient refinement handle it.
            if len(np.unique(y_i)) < n_states:
                continue
            try:
                mnl        = sm.MNLogit(y_i, sm.add_constant(X_i, prepend=True))
                res        = mnl.fit(disp=False, maxiter=200)
                params     = res.params   # shape (n_features+1, n_states-1)
                base_state = np.bincount(y_i, minlength=n_states).argmax()
                all_params = np.zeros((X_norm.shape[1] + 1, n_states))
                col = 0
                for j in range(n_states):
                    if j == base_state:
                        continue
                    if col < params.shape[1]:   # safety: skip if col OOB
                        all_params[:, j] = params[:, col]
                    col += 1
                W[i] = all_params[1:, :].T
            except Exception as e:
                log(f"      MNLogit state {i} failed ({e}), gradient fallback")

    # Gradient refinement
    lr = lr_w_init
    log_loss = 0.0
    for _ in range(n_iter_w):
        total_grad = np.zeros_like(W)
        log_loss   = 0.0
        for t in range(T - 1):
            i      = train_states[t]
            j      = train_states[t + 1]
            xt     = X_norm[t]
            logits = W[i] @ xt
            probs  = _softmax(logits)
            total_grad[i] += np.outer((np.eye(n_states)[j] - probs), xt)
            log_loss      += np.log(probs[j] + EPS)
        W    += lr * total_grad / (T + EPS)
        lr   *= 0.995

    log(f"      Dynamic W fit: log-loss/bar = {log_loss / max(T-1,1):.4f}")
    return W, x_mean_dyn, x_std_dyn


# ── state-stratified Cox intensity ────────────────────────────────────────────

def _fit_poisson_glm_beta(X_norm, y_binary, fallback_beta=None):
    n_features = X_norm.shape[1]
    y = y_binary.astype(float)
    if HAS_SM:
        try:
            res = sm.GLM(y, X_norm,
                         family=sm.families.Poisson()).fit(disp=1, maxiter=200)
            return res.params.astype(float), True
        except Exception:
            pass
    def neg_logL(beta):
        lam = np.clip(np.exp(X_norm @ beta), 1e-6, 100.0)
        return -np.sum(y * np.log(lam + EPS) - lam)
    def grad(beta):
        lam = np.clip(np.exp(X_norm @ beta), 1e-6, 100.0)
        return -(X_norm.T @ (y - lam))
    x0 = fallback_beta if fallback_beta is not None else np.zeros(n_features)
    try:
        res = optimize.minimize(neg_logL, x0=x0, jac=grad, method="L-BFGS-B")
        return res.x.astype(float), res.success
    except Exception:
        return np.zeros(n_features), False


def fit_stratified_cox(event_bars_train, train_states, df_feat_train,
                       n_states, x_mean_dyn, x_std_dyn):
    T      = len(train_states)
    X_dyn  = df_feat_train[DYN_FEATURE_COLS].values.astype(float)
    X_norm = (X_dyn - x_mean_dyn) / x_std_dyn
    event_set   = set(int(b) for b in event_bars_train if 0 <= int(b) < T)
    y_all       = np.array([1.0 if t in event_set else 0.0 for t in range(T)])
    beta_pooled, _ = _fit_poisson_glm_beta(X_norm, y_all)
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


# ── windowed Cox GLM (aggregate diagnostics) ─────────────────────────────────

def _windowed_counts(event_bars, T, n_windows=20):
    ws = T / n_windows
    return np.array([
        int(np.sum((w * ws <= np.array(event_bars)) &
                   (np.array(event_bars) < (w + 1) * ws)))
        for w in range(n_windows)
    ])


def _window_mean_feature(feature_series, T, n_windows=20):
    ws   = T / n_windows
    vals = np.array(feature_series)
    out  = np.zeros(n_windows)
    for w in range(n_windows):
        lo = int(w * ws)
        hi = min(int((w + 1) * ws), T)
        if lo < hi:
            out[w] = vals[lo:hi].mean()
    return out


def baseline_cox_dispersion(event_bars, T, n_windows=20):
    counts = _windowed_counts(event_bars, T, n_windows)
    mean_c = counts.mean()
    return np.nan if mean_c < EPS else counts.var() / mean_c


def fit_cox_with_covariates(event_bars, T, rva_series, state_series, n_windows=20):
    counts    = _windowed_counts(event_bars, T, n_windows).astype(float)
    rva_w     = _window_mean_feature(rva_series,   T, n_windows)
    state_w   = _window_mean_feature(state_series, T, n_windows)
    base_disp = baseline_cox_dispersion(event_bars, T, n_windows)

    if HAS_SM:
        try:
            X   = sm.add_constant(np.column_stack([rva_w, state_w]), prepend=True)
            res = sm.GLM(counts, X, family=sm.families.Poisson()).fit(
                disp=1, maxiter=200)
            fitted     = res.fittedvalues
            pearson_r  = (counts - fitted) / (np.sqrt(fitted) + EPS)
            resid_disp = pearson_r.var()
            dr_pct     = (100.0 * (base_disp - resid_disp) / (base_disp + EPS)
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
            log(f"      GLM failed ({e}), fallback")

    X_man = np.column_stack([np.ones(n_windows), rva_w, state_w])
    def neg_logL(b):
        lam = np.exp(X_man @ b)
        return -(np.sum(counts * np.log(lam + EPS) - lam))
    def grad(b):
        lam = np.exp(X_man @ b)
        return -(X_man.T @ (counts - lam))
    try:
        res = optimize.minimize(neg_logL,
                                x0=[np.log(counts.mean() + EPS), 0.0, 0.0],
                                jac=grad, method="L-BFGS-B")
        b          = res.x
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


# ── per-state event rates ─────────────────────────────────────────────────────

def state_event_rates(event_bars, state_seq, n_states, T):
    event_set  = set(int(b) for b in event_bars if 0 <= int(b) < len(state_seq))
    total_rate = len(event_set) / (T + EPS)
    rows       = []
    quiet_rate = None
    for s in range(n_states):
        mask     = state_seq == s
        n_bars_s = mask.sum()
        n_events = sum(1 for b in event_set if state_seq[int(b)] == s)
        rate     = n_events / (n_bars_s + EPS)
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
    baseline = quiet_rate if (quiet_rate is not None and quiet_rate > EPS) \
               else total_rate
    for row in rows:
        row["rate_ratio_vs_quiet"] = round(row["rate_per_bar"] / (baseline + EPS), 2)
    return rows


# ── burst probability signal ──────────────────────────────────────────────────

def compute_burst_prob_signal(df_feat, fwd_probs, state_seq, n_states,
                               split_idx, symbol):
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
    df_out["symbol"] = symbol
    assert df_out["p_burst"].between(0.0 - EPS, 1.0 + EPS).all(), \
        f"p_burst out of [0,1] for {symbol}"
    return df_out


# ── model bundle save ─────────────────────────────────────────────────────────

def save_model_bundle(symbol, model, W, betas, perm, split_idx,
                      x_mean_dyn, x_std_dyn, x_mean_emit, x_std_emit,
                      n_states, vol_window):
    MODEL_DIR.mkdir(exist_ok=True)
    safe = symbol.replace("/", "_")
    out  = MODEL_DIR / f"crypto_model_{safe}_5m.npz"
    np.savez(out,
             W=W, betas=betas,
             emit_means=model.means_.copy(),
             emit_covars=model.covars_.copy(),
             startprob=model.startprob_,
             transmat=model.transmat_,
             perm=np.array(perm),
             x_mean_dyn=x_mean_dyn, x_std_dyn=x_std_dyn,
             x_mean_emit=x_mean_emit, x_std_emit=x_std_emit,
             split_idx=np.array(split_idx),
             n_states=np.array(n_states),
             vol_window=np.array(vol_window))
    log(f"    Saved model → {out}")


# ── empirical transition matrix ───────────────────────────────────────────────

def empirical_transition_matrix(state_seq, n_states):
    A = np.zeros((n_states, n_states))
    for i in range(len(state_seq) - 1):
        A[state_seq[i], state_seq[i + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    return A / (row_sums + EPS)


# ── main per-symbol routine ───────────────────────────────────────────────────

def process_symbol(symbol, df_bars, n_states=3, n_windows=None,
                   train_frac=TRAIN_FRAC, fwd_horizon=FWD_HORIZON,
                   fwd_thresh=FWD_THRESH, vol_window=VOL_WINDOW):

    log(f"\n{'─'*65}")
    log(f"  {symbol}")
    log(f"{'─'*65}")

    events_raw = extract_events(df_bars, symbol,
                                fwd_horizon=fwd_horizon,
                                fwd_thresh=fwd_thresh)
    if events_raw is None or len(events_raw) < MIN_EVENTS:
        n_ev = 0 if events_raw is None else len(events_raw)
        log(f"    [{symbol}] only {n_ev} events — skipping (min {MIN_EVENTS})")
        return None, None, None

    df_feat = compute_features(df_bars, vol_window=vol_window)
    T = len(df_feat)
    if T < MIN_BARS:
        log(f"    [{symbol}] insufficient bars ({T})")
        return None, None, None

    events = realign_events(events_raw, df_feat)
    if events is None or len(events) < MIN_EVENTS:
        n_ev = 0 if events is None else len(events)
        log(f"    [{symbol}] only {n_ev} events after alignment — skipping")
        return None, None, None

    log(f"    {len(events)} forward-return events  |  {T:,} bars  "
        f"[fwd_ret>{fwd_thresh*100:.0f}%  horizon={fwd_horizon} bars]")

    if n_windows is None:
        n_windows = max(10, min(40, len(events) * 2))

    log(f"    Fitting {n_states}-state HMM (emit: ret_signed, abs_ret)...")
    (model, state_seq, fwd_probs, split_idx, hmm_logL,
     labels, perm, mu_emit, std_emit) = fit_hmm(
        df_feat, n_states=n_states, train_frac=train_frac)

    if model is None:
        log(f"    [{symbol}] HMM fit failed")
        return None, None, None

    n_train      = split_idx
    n_test       = T - split_idx
    train_states = state_seq[:split_idx]

    # Fill persist now that we have state assignments
    df_feat = _fill_persist(df_feat, state_seq)

    log(f"    HMM logL={hmm_logL:.2f}  |  "
        f"train={n_train:,}  test={n_test:,}")

    state_counts = np.bincount(state_seq, minlength=n_states)
    for s, name in labels.items():
        log(f"      State {s} ({name:<10}): {state_counts[s]:>6,} bars "
            f"({100*state_counts[s]/T:>5.1f}%)")

    burst_bar_count = int(state_counts[n_states - 1])
    if burst_bar_count < MIN_BURST_BARS:
        log(f"    [{symbol}] Burst state only {burst_bar_count} bars — collapsed, skipping")
        return None, None, None

    A = empirical_transition_matrix(state_seq, n_states)
    if n_states == 3:
        p_entry   = A[1, 2]
        p_persist = A[2, 2]
        log(f"    P(Burst|Accum)={p_entry:.4f}  P(Burst|Burst)={p_persist:.4f}")
    else:
        p_entry, p_persist = A[0, 1], A[1, 1]

    feat_len           = len(df_feat)
    event_bars         = [min(int(b), feat_len-1)
                          for b in events["bar_idx"].tolist() if int(b) < feat_len]
    event_bars_train   = [b for b in event_bars if b < split_idx]

    rate_rows = state_event_rates(event_bars, state_seq, n_states, T)
    log(f"    Per-state event rates:")
    log(f"    {'State':<12} {'Bars':>7} {'Events':>7} "
        f"{'Rate/Bar':>10} {'RateRatio':>11}")
    for row in rate_rows:
        log(f"    {labels[row['state']]:<12} {row['n_bars']:>7,} "
            f"{row['n_events']:>7} {row['rate_per_bar']:>10.6f} "
            f"{row['rate_ratio_vs_quiet']:>10.2f}x")

    glm = fit_cox_with_covariates(
        event_bars, feat_len,
        df_feat["RVA"].values, state_seq.astype(float),
        n_windows=n_windows)

    def fp(p):
        if not np.isfinite(p): return "  —  "
        return (f"{p:.4f}" +
                (" ***" if p < 0.001 else " **" if p < 0.01
                 else " *" if p < 0.05 else "    "))

    if glm.get("converged"):
        b0, b1, b2 = glm["beta0"], glm["beta1_rva"], glm["beta2_state"]
        log(f"    Cox GLM: β₀={b0:+.4f}  β₁(RVA)={b1:+.4f} "
            f"p={fp(glm.get('pval_rva', np.nan))}  "
            f"β₂(state)={b2:+.4f} p={fp(glm.get('pval_state', np.nan))}")
        if np.isfinite(glm.get("dispersion_reduction_pct", np.nan)):
            log(f"    Dispersion reduction: "
                f"{glm['dispersion_reduction_pct']:+.1f}%")

    log(f"    Fitting dynamic transition matrix W...")
    df_feat_train = df_feat.iloc[:split_idx].copy()
    try:
        W, x_mean_dyn, x_std_dyn = fit_dynamic_hmm_initial(
            model, df_feat_train, train_states, n_states)
    except Exception as e:
        log(f"      W fit failed ({e}) — zeros fallback")
        n_features = len(DYN_FEATURE_COLS)
        W          = np.zeros((n_states, n_states, n_features))
        x_mean_dyn = df_feat_train[DYN_FEATURE_COLS].mean().values
        x_std_dyn  = df_feat_train[DYN_FEATURE_COLS].std().values + EPS

    log(f"    Fitting state-stratified Cox intensity...")
    try:
        betas = fit_stratified_cox(event_bars_train, train_states,
                                   df_feat_train, n_states,
                                   x_mean_dyn, x_std_dyn)
        for k in range(n_states):
            log(f"      β_{k} ({labels[k]}): {np.round(betas[k], 4)}")
    except Exception as e:
        log(f"      Stratified Cox failed ({e}) — zeros fallback")
        betas = np.zeros((n_states, len(DYN_FEATURE_COLS)))

    save_model_bundle(symbol, model, W, betas, perm, split_idx,
                      x_mean_dyn, x_std_dyn, mu_emit, std_emit,
                      n_states, vol_window)

    df_signal = compute_burst_prob_signal(
        df_feat, fwd_probs, state_seq, n_states, split_idx, symbol)

    burst_state_idx  = n_states - 1
    burst_rate_ratio = next(
        (r["rate_ratio_vs_quiet"] for r in rate_rows
         if r["state"] == burst_state_idx), np.nan)

    result = {
        "symbol":             symbol,
        "n_events":           len(events),
        "n_bars":             T,
        "n_train":            n_train,
        "n_test":             n_test,
        "hmm_logL":           round(float(hmm_logL), 2),
        "burst_rate_ratio":   round(float(burst_rate_ratio), 2),
        "p_burst_entry":      round(float(p_entry), 4),
        "p_burst_persist":    round(float(p_persist), 4),
        "dispersion_reduction_pct":
            round(float(glm.get("dispersion_reduction_pct") or np.nan), 1),
        "beta0":        round(float(glm.get("beta0",       np.nan)), 4),
        "beta1_rva":    round(float(glm.get("beta1_rva",   np.nan)), 4),
        "beta2_state":  round(float(glm.get("beta2_state", np.nan)), 4),
        "W_norm":       round(float(np.linalg.norm(W)), 4),
        "glm_converged": glm.get("converged", False),
    }

    trans_row = {"symbol": symbol}
    for i in range(n_states):
        for j in range(n_states):
            key = f"A_{labels[i][:2].lower()}_{labels[j][:2].lower()}"
            trans_row[key] = round(float(A[i, j]), 6)

    # Attach vol_ratio to signal df so pnlc.py --sparse-entry can use it
    if "vol_ratio" not in df_signal.columns and "vol_ratio" in df_feat.columns:
        df_signal["vol_ratio"] = df_feat["vol_ratio"].values

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
        log(f"  Burst rate ratio — Mean: {np.mean(brr):.2f}x  "
            f"Max: {np.max(brr):.2f}x "
            f"({valid[int(np.argmax(brr))]['symbol']})")

    dr = [r["dispersion_reduction_pct"] for r in valid
          if np.isfinite(r.get("dispersion_reduction_pct") or np.nan)]
    if dr:
        log(f"  Dispersion reduction — Mean: {np.mean(dr):+.1f}%  "
            f"Improved: {sum(v>0 for v in dr)}/{len(dr)}")

    log(f"\n  {'Symbol':<14} {'N':>5} {'Train':>7} {'Test':>7} "
        f"{'BrstRatio':>10} {'β₂State':>9} {'P_entry':>9} {'P_persist':>10}")
    log(f"  {'─'*78}")
    for r in results:
        if not r.get("glm_converged"):
            log(f"  {r['symbol']:<14} fit failed")
            continue
        log(f"  {r['symbol']:<14} {r['n_events']:>5} "
            f"{r['n_train']:>7,} {r['n_test']:>7,} "
            f"{r['burst_rate_ratio']:>10.2f}x "
            f"{r['beta2_state']:>+9.4f} "
            f"{r['p_burst_entry']:>9.4f} "
            f"{r['p_burst_persist']:>10.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="*", default=None)
    parser.add_argument("--all-symbols",   action="store_true")
    parser.add_argument("--timeframe",     default=TIMEFRAME)
    parser.add_argument("--lookback",      type=int,   default=LOOKBACK_DAYS)
    parser.add_argument("--start",         default=None)
    parser.add_argument("--end",           default=None)
    parser.add_argument("--states",        type=int,   default=3)
    parser.add_argument("--train-frac",    type=float, default=TRAIN_FRAC)
    parser.add_argument("--fwd-horizon",   type=int,   default=FWD_HORIZON,
                        help="Bars ahead for forward return event (default: 5).")
    parser.add_argument("--fwd-thresh",    type=float, default=FWD_THRESH,
                        help="Forward return threshold for event (default: 0.10 = 10%%).")
    parser.add_argument("--vol-window",    type=int,   default=VOL_WINDOW,
                        help="Rolling vol window for RVA features (default: 96 bars).")
    parser.add_argument("--no-cache",      action="store_true")
    args = parser.parse_args()

    if args.all_symbols:
        args.symbols = DEFAULT_SYMBOLS
    elif not args.symbols:
        log("ERROR: no symbols given.")
        log("  Pass symbols:      --symbols INJ/USDT SUI/USDT ...")
        log("  Or full universe:  --all-symbols")
        sys.exit(1)

    log("=" * 65)
    log("  covariatec.py  —  HMM + Cox  [Crypto / Small-Cap Alts / v5]")
    log("=" * 65)
    log(f"  Symbols      : {len(args.symbols)}")
    log(f"  Timeframe    : {args.timeframe}  (native 5m fitting)")
    log(f"  Lookback     : {args.lookback} days")
    log(f"  Event        : fwd_ret > {args.fwd_thresh*100:.0f}%  "
        f"over {args.fwd_horizon} bars  "
        f"({args.fwd_horizon * 5}min at 5m)")
    log(f"  Vol window   : {args.vol_window} bars  "
        f"({args.vol_window * 5 // 60}h — RVA context only)")
    log(f"  HMM states   : {args.states}  Train: {args.train_frac*100:.0f}%")
    log(f"  Emit features: {EMIT_FEATURE_COLS}")
    log(f"  Dyn features : {DYN_FEATURE_COLS}")
    log(f"  Dynamic W    : input-coupled transition matrix")
    log(f"  Cox          : state-stratified intensity")
    log(f"  p_burst      : filtered forward algorithm — no future info")
    log("=" * 65)

    all_results   = []
    all_state_dfs = []
    all_trans     = []
    all_events    = []

    for sym in args.symbols:
        df = fetch_ohlcv(sym,
                         timeframe=args.timeframe,
                         lookback_days=args.lookback,
                         start=args.start,
                         end=args.end,
                         use_cache=not args.no_cache)
        if df is None or len(df) < MIN_BARS:
            log(f"  [{sym}] insufficient data ({0 if df is None else len(df)} bars)")
            continue

        # Collect events for output CSV before process_symbol
        ev_raw = extract_events(df, sym,
                                 fwd_horizon=args.fwd_horizon,
                                 fwd_thresh=args.fwd_thresh)
        if ev_raw is not None and len(ev_raw) >= MIN_EVENTS:
            df_feat_tmp = compute_features(df, vol_window=args.vol_window)
            ev_aligned  = realign_events(ev_raw, df_feat_tmp)
            if ev_aligned is not None and len(ev_aligned) >= MIN_EVENTS:
                ev_aligned["symbol"]      = sym
                ev_aligned["fwd_horizon"] = args.fwd_horizon
                ev_aligned["fwd_thresh"]  = args.fwd_thresh
                all_events.append(ev_aligned)

        result, df_signal, trans_row = process_symbol(
            sym, df,
            n_states=args.states,
            train_frac=args.train_frac,
            fwd_horizon=args.fwd_horizon,
            fwd_thresh=args.fwd_thresh,
            vol_window=args.vol_window,
        )
        if result    is not None: all_results.append(result)
        if df_signal is not None: all_state_dfs.append(df_signal)
        if trans_row is not None: all_trans.append(trans_row)

    print_summary(all_results)

    log(f"\n{'─'*65}")
    if all_results:
        pd.DataFrame(all_results).to_csv("crypto_covariate_results.csv", index=False)
        log("  Saved → crypto_covariate_results.csv")

    if all_state_dfs:
        state_df = pd.concat(all_state_dfs)
        keep = [c for c in ["symbol", "Volume", "vol_ratio", "abs_ret",
                             "ret_signed", "ret_zscore", "RVA", "RVA_vel",
                             "persist", "vol_zscore", "hmm_state", "p_burst",
                             "p_quiet", "p_accum", "split"]
                if c in state_df.columns]
        state_df[keep].to_csv("crypto_covariate_states.csv")
        log(f"  Saved → crypto_covariate_states.csv  ({len(state_df):,} bars)")
        log(f"  *** Downstream: filter to split=='test' before backtesting ***")

    if all_trans:
        pd.DataFrame(all_trans).to_csv("crypto_covariate_transitions.csv",
                                        index=False)
        log("  Saved → crypto_covariate_transitions.csv")

    if all_events:
        ev_df = pd.concat(all_events)
        ev_df.to_csv("crypto_events.csv", index=False)
        log(f"  Saved → crypto_events.csv  "
            f"({len(ev_df):,} events  "
            f"{ev_df['symbol'].nunique()} symbols)")

    log(f"\n{'='*65}")
    log("  Next: python pnlc.py")
    log("        python pnlc.py --resample-to 15m")
    log("        python pnlc.py --tf-sweep")
    log("        python pnlc.py --sparse-entry --hmm-exit")
    log(f"{'='*65}")


if __name__ == "__main__":
    main()