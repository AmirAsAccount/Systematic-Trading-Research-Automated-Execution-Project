"""
process_id_15m.py  —  Point Process Tournament for Low Float Stocks
                       15-Minute Bar Edition

CHANGES FROM DAILY VERSION
--------------------------
  - Chunked yfinance intraday fetch (59-day windows) with parquet cache
  - Session filter: 09:30–16:00 ET only (14:30–21:00 UTC)
  - Session-contiguous bar index (overnight gaps removed)
  - Event detection: 130-bar (~1 week) rolling volume mean, 2% return threshold
  - Hawkes beta initialized at 0.05 (tuned for intraday bar units)
  - Ogata residual test now has power (50–400 events expected)
  - QQ compensator data written to CSV for visual inspection
  - Bivariate Hawkes for correlated symbol pairs (mutual excitation matrix)
  - Intraday regime covariates: VIX proxy + session position (first/mid/last hour)
  - Additional AIC margin and Weibull shape columns in output CSV

PROCESS HIERARCHY
-----------------
  Level 0  HPP            — constant rate, memoryless
  Level 1  NHPP           — rate varies with regime
  Level 2  Renewal        — Weibull inter-arrivals
  Level 3  Cox            — latent stochastic intensity (NegBin windows)
  Level 4  Hawkes         — self-excitation (exponential kernel)
  Level 5  MarkedHawkes   — self-excitation weighted by float rotation
  Level 6  CoxHawkes      — regime baseline + self-excitation

BIVARIATE HAWKES (pairs)
------------------------
  Fits mutual excitation matrix for correlated symbol pairs.
  Off-diagonal terms α_AB, α_BA identify cross-excitation.

OUTPUT
------
  process_id_15m_results.csv   — one row per symbol, winner + metrics
  process_id_15m_events.csv    — all extracted events with marks
  process_id_15m_compensator.csv — Ogata residual sequences for QQ inspection
  process_id_15m_bivariate.csv — bivariate Hawkes results for pairs
  cache/                        — parquet cache of raw 15m bars

USAGE
-----
  pip install yfinance scipy numpy pandas pyarrow
  python process_id_15m.py
  python process_id_15m.py --vol-thresh 2.0 --ret-thresh 0.02 --no-cache
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
from scipy.special import gamma as gamma_fn

# ── universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    # --- Your Original 23 Tickers ---
    "AHMA", "ARQQ", "BRUN", "CAST", "CUPR", "ELVR", "FAMI", "GLXG",
    "HQ",   "HUBC", "LUD",  "MTEN", "PRFX", "QUCY", "RDGT", "RGNT",
    "RTB",  "SDOT", "SLMT", "UBXG", "VSME", "WLDS", "WYFI", 

    # --- Expanded Additions (24 - 75) ---
    "ADTX", "AEHL", "AENT", "ALBT", "AMV",  "APCX", "ASST", "BETS", 
    "BFRG", "BLBX", "CDIO", "CEI",  "CETY", "CJET", "CRKN", "CXAI", 
    "DRUG", "EDBL", "GDC",  "GDHG", "GGE",  "GPUS", "GWAV", "IKT", 
    "INVO", "IVA",  "JAKK", "KPLT", "LBBB", "LIFW", "MGIH", "MGRM", 
    "MRAI", "MULN", "OBLG", "OMH",  "PEV",  "PHUN", "PLUR", "REVB", 
    "SMFL", "SNAL", "SOXS", "SRXH", "TCBP", "TENX", "TOP",  "TRKA", 
    "UXIN", "VCI",  "VFS",  "WAVS",
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

# Correlated pairs for bivariate Hawkes
BIVARIATE_PAIRS = [
    ("FAMI", "RTB"),
    ("ARQQ", "UBXG"),
    ("SLMT", "RDGT"),
    ("HUBC", "VSME"),
]

# ── config ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS    = 365 * 2       # total history to fetch
VOLUME_ROLL      = 130           # bars (~1 week of 15m bars)
VOLUME_THRESH    = 2.0           # volume > N × rolling mean
RETURN_THRESH    = 0.02          # |bar return| > 2%
MIN_EVENTS       = 30            # minimum events for model fitting
MARKET_TICKER    = "IWM"
CACHE_DIR        = Path("cache")
EPS              = 1e-12

# Session hours in UTC (ET 09:30–16:00 = UTC 14:30–21:00)
SESSION_START_UTC = 14 * 60 + 30   # minutes since midnight UTC
SESSION_END_UTC   = 21 * 60        # exclusive

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

def minutes_utc(ts):
    """Return minutes since midnight UTC for a timezone-aware timestamp."""
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts
    return utc.hour * 60 + utc.minute

def is_session(ts):
    m = minutes_utc(ts)
    return SESSION_START_UTC <= m < SESSION_END_UTC

def session_position(ts):
    """
    Normalised position within the trading session [0, 1].
    0 = open, 1 = close. Used as an intraday regime covariate.
    """
    m = minutes_utc(ts)
    pos = (m - SESSION_START_UTC) / (SESSION_END_UTC - SESSION_START_UTC)
    return float(np.clip(pos, 0, 1))


# ── dual-resolution fetch: 1h (full history) + 15m (last 60 days) ────────────
def _clean_df(raw):
    """Standardise columns, localise index to UTC, apply session filter."""
    if raw is None or (hasattr(raw, 'empty') and raw.empty):
        return None
    df = raw.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(needed) < 5:
        return None
    df = df[needed].dropna()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[df.index.map(is_session)]
    return df if not df.empty else None


def fetch_15m(ticker, lookback_days=LOOKBACK_DAYS, use_cache=True):
    """
    Dual-resolution fetch strategy:

      1h bars  — yfinance serves up to 730 days; covers full 2-year history.
      15m bars — yfinance hard limit is last 60 days; replaces 1h in that window.

    The two are merged: 1h bars are trimmed to end 58 days ago, then 15m bars
    for the recent window are appended. This gives the maximum available history
    at the highest available resolution throughout.

    yfinance limitation: 15m intraday data is only available for the last 60
    calendar days regardless of start/end parameters. Attempts to fetch older
    15m data silently return empty or error — this is a Yahoo Finance API
    restriction, not a script bug.
    """
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    end_dt    = datetime.now()
    start_dt  = end_dt - timedelta(days=lookback_days)
    cutoff_dt = end_dt - timedelta(days=58)

    # ── 1h bars: full 2-year lookback ────────────────────────────────────────
    cache_1h = CACHE_DIR / f"{ticker}_1h.parquet"
    df_1h    = None

    if use_cache and cache_1h.exists():
        try:
            df_1h = pd.read_parquet(cache_1h)
            log(f"    [{ticker}] 1h: {len(df_1h)} bars from cache")
        except Exception:
            df_1h = None

    if df_1h is None:
        try:
            raw = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                auto_adjust=True,
            )
            df_1h = _clean_df(raw)
            if df_1h is not None and use_cache:
                df_1h.to_parquet(cache_1h)
            log(f"    [{ticker}] 1h: fetched {len(df_1h) if df_1h is not None else 0} bars")
        except Exception as e:
            log(f"    [{ticker}] 1h fetch error: {e}")
        time.sleep(0.4)

    # ── 15m bars: last 58 days only ───────────────────────────────────────────
    cache_15m = CACHE_DIR / f"{ticker}_15m_recent.parquet"
    df_15m    = None

    if use_cache and cache_15m.exists():
        try:
            df_15m = pd.read_parquet(cache_15m)
            if df_15m is not None and not df_15m.empty:
                newest = df_15m.index.max().replace(tzinfo=None)
                if (end_dt - newest).days > 3:
                    df_15m = None   # stale cache — re-fetch
                else:
                    log(f"    [{ticker}] 15m: {len(df_15m)} bars from cache")
        except Exception:
            df_15m = None

    if df_15m is None:
        try:
            raw = yf.download(
                ticker,
                start=cutoff_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="15m",
                progress=False,
                auto_adjust=True,
            )
            df_15m = _clean_df(raw)
            if df_15m is not None and use_cache:
                df_15m.to_parquet(cache_15m)
            log(f"    [{ticker}] 15m: fetched {len(df_15m) if df_15m is not None else 0} bars")
        except Exception as e:
            log(f"    [{ticker}] 15m fetch error: {e}")
        time.sleep(0.4)

    # ── merge: 15m replaces 1h in the overlapping window ─────────────────────
    chunks = []

    if df_1h is not None and not df_1h.empty:
        cutoff_utc    = pd.Timestamp(cutoff_dt, tz="UTC")
        df_1h_trimmed = df_1h[df_1h.index < cutoff_utc]
        if not df_1h_trimmed.empty:
            chunks.append(df_1h_trimmed)

    if df_15m is not None and not df_15m.empty:
        chunks.append(df_15m)

    if not chunks:
        return None

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df if len(df) > 100 else None


# ── session-contiguous bar index ──────────────────────────────────────────────
def build_session_index(df):
    """
    Assign contiguous integer bar indices within trading sessions only.
    Overnight gaps are not counted. Returns Series aligned to df.index.
    """
    idx = np.arange(len(df))
    return pd.Series(idx, index=df.index)


# ── intraday regime covariates ────────────────────────────────────────────────
def build_intraday_regimes(df, market_df):
    """
    Build a regime Series aligned to df.index with two components:
      1. IWM 30-day momentum (daily, forward-filled to intraday)
      2. Session position (normalised 0→1 within each session)

    Returns DataFrame with columns: mkt_regime, session_pos, combined
    """
    # daily IWM momentum
    if market_df is not None:
        daily_ret30 = market_df["Close"].pct_change(30)
        daily_reg   = (daily_ret30 - daily_ret30.rolling(60).mean()) / \
                      (daily_ret30.rolling(60).std() + EPS)
        daily_reg   = daily_reg.fillna(0)

        # reindex: align by date
        df_dates     = df.index.normalize()
        mkt_reg_vals = []
        for ts in df.index:
            date = ts.normalize() if hasattr(ts, 'normalize') else ts.date()
            try:
                v = daily_reg.asof(date) if hasattr(daily_reg.index, 'asof') else 0.0
                mkt_reg_vals.append(float(v) if pd.notna(v) else 0.0)
            except Exception:
                mkt_reg_vals.append(0.0)
        mkt_reg = pd.Series(mkt_reg_vals, index=df.index)
    else:
        mkt_reg = pd.Series(0.0, index=df.index)

    sess_pos = pd.Series(
        [session_position(ts) for ts in df.index],
        index=df.index
    )

    combined = 0.6 * mkt_reg + 0.4 * sess_pos
    combined = (combined - combined.mean()) / (combined.std() + EPS)

    return pd.DataFrame({
        "mkt_regime":  mkt_reg,
        "session_pos": sess_pos,
        "combined":    combined,
    })


# ── event extraction ──────────────────────────────────────────────────────────
def extract_events(df, ticker, bar_index):
    """
    Extract volume surge + price move events from 15-minute bars.

    Event criteria:
      1. Volume > VOLUME_THRESH × VOLUME_ROLL-bar rolling mean
      2. |Close/Open - 1| > RETURN_THRESH

    Mark = volume / float_shares  (float rotation)
           or vol_ratio if float unknown.

    Returns DataFrame: date, bar_idx, volume, ret, mark, session_pos
    """
    df = df.copy()
    df["vol_ma"]    = df["Volume"].rolling(VOLUME_ROLL, min_periods=20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_ma"] + EPS)
    df["ret"]       = (df["Close"] - df["Open"]) / (df["Open"] + EPS)
    df["abs_ret"]   = df["ret"].abs()
    df["bar_idx"]   = bar_index.values

    mask   = (df["vol_ratio"] > VOLUME_THRESH) & (df["abs_ret"] > RETURN_THRESH)
    events = df[mask].copy()

    if events.empty:
        return None

    float_shares = KNOWN_FLOATS.get(ticker)
    events["mark"] = (events["Volume"] / float_shares
                      if float_shares and float_shares > 0
                      else events["vol_ratio"])

    events["session_pos"] = [session_position(ts) for ts in events.index]
    events["date"]        = events.index

    result = events[["date", "bar_idx", "Volume", "ret", "mark", "session_pos"]].copy()
    result.columns = ["date", "bar_idx", "volume", "ret", "mark", "session_pos"]
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL FITTING  (each returns logL, n_params, params_dict)
# ══════════════════════════════════════════════════════════════════════════════

def fit_hpp(event_times, T):
    N  = len(event_times)
    if N == 0: return -np.inf, 1, {}
    mu = N / T
    return N * np.log(mu + EPS) - mu * T, 1, {"mu": mu}


def fit_nhpp(event_times, T, regimes):
    N = len(event_times)
    if N < 5 or regimes is None:
        logL, _, p = fit_hpp(event_times, T)
        return logL, 2, {"gamma0": np.log(N/T+EPS), "gamma1": 0.0}

    reg_ev  = regimes.iloc[
        [min(int(t), len(regimes)-1) for t in event_times]
    ].values
    reg_all = regimes.values

    def neg_logL(params):
        g0, g1 = params
        lam_ev  = np.exp(g0 + g1 * reg_ev)
        lam_all = np.exp(g0 + g1 * reg_all)
        return -(np.sum(np.log(lam_ev + EPS)) - lam_all.sum())

    try:
        res = optimize.minimize(neg_logL, [np.log(N/T+EPS), 0.0],
                                method="Nelder-Mead",
                                options={"maxiter": 2000, "xatol": 1e-7})
        g0, g1 = res.x
        return -res.fun, 2, {"gamma0": g0, "gamma1": g1}
    except Exception:
        logL, _, p = fit_hpp(event_times, T)
        return logL, 2, {}


def fit_renewal_weibull(inter_arrivals):
    if len(inter_arrivals) < 10:
        return -np.inf, 2, {}
    try:
        k, loc, scale = stats.weibull_min.fit(inter_arrivals, floc=0)
        logL = np.sum(stats.weibull_min.logpdf(inter_arrivals, k, loc=0, scale=scale))
        return logL, 2, {"k": k, "scale": scale,
                         "shape_interpretation": (
                             "increasing hazard (k>1)" if k > 1.05 else
                             "decreasing hazard (k<1)" if k < 0.95 else
                             "near-exponential (k≈1)")}
    except Exception:
        return -np.inf, 2, {}


def fit_cox_gamma(event_times, T, n_windows=20):
    if len(event_times) < 10:
        return -np.inf, 2, {}
    try:
        window_size = T / n_windows
        counts = np.array([
            sum(w*window_size <= t < (w+1)*window_size for t in event_times)
            for w in range(n_windows)
        ])
        mean_c = counts.mean()
        var_c  = counts.var()
        if var_c <= mean_c + EPS:
            logL, _, _ = fit_hpp(event_times, T)
            return logL - 0.5, 2, {"r": np.inf, "p": 0,
                                    "dispersion": var_c/(mean_c+EPS)}
        r = mean_c**2 / (var_c - mean_c + EPS)
        p = mean_c / (var_c + EPS)
        logL = np.sum(stats.nbinom.logpmf(counts, r, p))
        return logL, 2, {"r": r, "p": p,
                          "dispersion": var_c/(mean_c+EPS)}
    except Exception:
        return -np.inf, 2, {}


def fit_hawkes_exp(event_times, T, beta_inits=None):
    """
    Hawkes with exponential kernel. Beta inits tuned for intraday bar units
    (typical inter-arrivals 10-200 bars → beta in 0.005-0.2 range).
    """
    N = len(event_times)
    if N < 15:
        return -np.inf, 3, {}

    t = np.array(event_times, dtype=float)

    if beta_inits is None:
        beta_inits = [0.005, 0.02, 0.05, 0.1, 0.2]

    def neg_logL(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10
        if alpha / beta >= 0.999:
            return 1e10
        A = np.zeros(N)
        for i in range(1, N):
            A[i] = np.exp(-beta * (t[i] - t[i-1])) * (1 + A[i-1])
        lam = mu + alpha * A
        if np.any(lam <= 0):
            return 1e10
        return -(np.sum(np.log(lam)) -
                 mu * T -
                 (alpha/beta) * np.sum(1 - np.exp(-beta * (T - t))))

    mu0    = N / T
    best_logL   = -np.inf
    best_params = {}

    for b_init in beta_inits:
        for a_frac in [0.3, 0.6, 0.9]:
            try:
                res = optimize.minimize(
                    neg_logL, [mu0, mu0*a_frac*b_init, b_init],
                    method="Nelder-Mead",
                    options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-9}
                )
                if -res.fun > best_logL:
                    mu_f, a_f, b_f = res.x
                    if mu_f > 0 and a_f > 0 and b_f > 0 and a_f/b_f < 1:
                        best_logL = -res.fun
                        best_params = {
                            "mu": mu_f, "alpha": a_f, "beta": b_f,
                            "branching_ratio": a_f/b_f,
                            "halflife_bars": np.log(2)/b_f,
                        }
            except Exception:
                continue

    return best_logL, 3, best_params


def fit_marked_hawkes(event_times, marks, T):
    N = len(event_times)
    if N < 20:
        return -np.inf, 4, {}

    t      = np.array(event_times, dtype=float)
    M_norm = (np.array(marks) - np.mean(marks)) / (np.std(marks) + EPS)

    def neg_logL(params):
        mu, alpha, beta, gamma = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10
        eff_alpha = alpha * np.exp(gamma * M_norm)
        if np.any(eff_alpha / beta >= 0.999):
            return 1e10
        A = np.zeros(N)
        for i in range(1, N):
            A[i] = np.exp(-beta * (t[i] - t[i-1])) * (1 + A[i-1])
        lam = mu + eff_alpha * A
        if np.any(lam <= 0):
            return 1e10
        return -(np.sum(np.log(lam)) - mu*T -
                 np.sum((eff_alpha/beta)*(1 - np.exp(-beta*(T-t)))))

    hwk_logL, _, hwk_p = fit_hawkes_exp(event_times, T)
    if not hwk_p:
        return -np.inf, 4, {}

    mu0, a0, b0 = hwk_p["mu"], hwk_p["alpha"], hwk_p["beta"]
    best_logL, best_params = -np.inf, {}

    for g_init in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        try:
            res = optimize.minimize(
                neg_logL, [mu0, a0, b0, g_init],
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-9}
            )
            if -res.fun > best_logL:
                mu_f, a_f, b_f, gm_f = res.x
                if mu_f > 0 and a_f > 0 and b_f > 0:
                    best_logL = -res.fun
                    best_params = {
                        "mu": mu_f, "alpha": a_f, "beta": b_f,
                        "gamma": gm_f, "branching_ratio": a_f/b_f,
                        "halflife_bars": np.log(2)/b_f,
                        "mark_sensitivity": gm_f,
                    }
        except Exception:
            continue

    return best_logL, 4, best_params


def fit_cox_hawkes(event_times, marks, T, regimes):
    N = len(event_times)
    if N < 25 or regimes is None:
        return fit_marked_hawkes(event_times, marks, T)

    t      = np.array(event_times, dtype=float)
    M_norm = (np.array(marks) - np.mean(marks)) / (np.std(marks) + EPS)
    reg_all = regimes.values
    reg_ev  = regimes.iloc[
        [min(int(ti), len(regimes)-1) for ti in t]
    ].values

    def neg_logL(params):
        g0, g1, alpha, beta, gamma = params
        if alpha <= 0 or beta <= 0:
            return 1e10
        eff_alpha = alpha * np.exp(gamma * M_norm)
        A = np.zeros(N)
        for i in range(1, N):
            A[i] = np.exp(-beta * (t[i] - t[i-1])) * (1 + A[i-1])
        cox_ev  = np.exp(g0 + g1 * reg_ev)
        lam     = cox_ev + eff_alpha * A
        if np.any(lam <= 0):
            return 1e10
        cox_all  = np.exp(g0 + g1 * reg_all)
        integral = cox_all.sum() + np.sum((eff_alpha/beta)*(1 - np.exp(-beta*(T-t))))
        return -(np.sum(np.log(lam + EPS)) - integral)

    mh_logL, _, mh_p = fit_marked_hawkes(event_times, marks, T)
    if not mh_p:
        return fit_marked_hawkes(event_times, marks, T)

    mu0, a0, b0, gm0 = mh_p["mu"], mh_p["alpha"], mh_p["beta"], mh_p.get("gamma", 0.0)
    g0_0 = np.log(mu0 + EPS)
    best_logL, best_params = -np.inf, {}

    for g1_init in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        try:
            res = optimize.minimize(
                neg_logL, [g0_0, g1_init, a0, b0, gm0],
                method="Nelder-Mead",
                options={"maxiter": 15000, "xatol": 1e-8, "fatol": 1e-8}
            )
            if -res.fun > best_logL:
                g0_f, g1_f, a_f, b_f, gm_f = res.x
                if a_f > 0 and b_f > 0:
                    best_logL = -res.fun
                    best_params = {
                        "gamma0": g0_f, "gamma1": g1_f,
                        "alpha": a_f, "beta": b_f,
                        "gamma_mark": gm_f,
                        "branching_ratio": a_f/b_f,
                        "halflife_bars": np.log(2)/b_f,
                    }
        except Exception:
            continue

    return best_logL, 5, best_params


# ── information criteria ──────────────────────────────────────────────────────
def aic(logL, k):       return 2*k - 2*logL
def bic(logL, k, n):    return k*np.log(n+EPS) - 2*logL
def lr_test(logL_r, logL_f, df):
    if logL_f <= logL_r or df <= 0: return 0.0, 1.0
    lr  = 2*(logL_f - logL_r)
    return lr, 1 - stats.chi2.cdf(lr, df)


# ── Ogata residual test ───────────────────────────────────────────────────────
def ogata_residual_test(event_times, T, model_type, params):
    """
    Compute compensator Λ(tᵢ) and test residual inter-arrivals against Exp(1).
    Returns (ks_stat, p_value, passed, compensator_sequence).
    """
    t = np.array(sorted(event_times), dtype=float)
    N = len(t)
    if N < 20 or not params:
        return np.nan, np.nan, False, []

    try:
        if model_type == "HPP":
            mu  = params.get("mu", N/T)
            lam = np.array([mu * ti for ti in t])

        elif model_type in ("Hawkes", "MarkedHawkes", "CoxHawkes"):
            mu    = params.get("mu", N/T)
            alpha = params.get("alpha", 0.01)
            beta  = params.get("beta", 0.05)

            lam = np.zeros(N)
            for i in range(N):
                base = mu * t[i]
                kern = sum(
                    (alpha/beta) * (1 - np.exp(-beta*(t[i]-t[j])))
                    for j in range(i)
                )
                lam[i] = base + kern

        elif model_type == "Renewal":
            k     = params.get("k", 1.0)
            scale = params.get("scale", np.mean(np.diff(t)) if N > 1 else 1.0)
            lam   = np.array([
                stats.weibull_min.cdf(t[i], k, loc=0, scale=scale) * N
                for i in range(N)
            ])

        else:
            return np.nan, np.nan, False, []

        intervals = np.diff(np.concatenate([[0], lam]))
        intervals = intervals[intervals > 0]

        if len(intervals) < 10:
            return np.nan, np.nan, False, list(intervals)

        ks, p = stats.kstest(intervals, "expon", args=(0, 1))
        return ks, p, p > 0.05, list(intervals)

    except Exception:
        return np.nan, np.nan, False, []


# ── dispersion index ──────────────────────────────────────────────────────────
def dispersion_index(event_times, T, n_windows=30):
    if len(event_times) < 10:
        return np.nan
    ws = T / n_windows
    counts = np.array([
        sum(w*ws <= et < (w+1)*ws for et in event_times)
        for w in range(n_windows)
    ])
    mean_c = counts.mean()
    return np.nan if mean_c < EPS else counts.var() / mean_c


# ══════════════════════════════════════════════════════════════════════════════
#  BIVARIATE HAWKES
# ══════════════════════════════════════════════════════════════════════════════

def fit_bivariate_hawkes(times_A, times_B, T):
    """
    Bivariate Hawkes with mutual excitation:
      λ_A(t) = μ_A + α_AA·Σ exp(-β(t-tᵢ)) + α_BA·Σ exp(-β(t-tⱼ))
      λ_B(t) = μ_B + α_BB·Σ exp(-β(t-tⱼ)) + α_AB·Σ exp(-β(t-tᵢ))

    Returns fitted matrix and log-likelihood.
    α_AB: does A excite B?
    α_BA: does B excite A?
    """
    tA = np.array(sorted(times_A), dtype=float)
    tB = np.array(sorted(times_B), dtype=float)
    NA, NB = len(tA), len(tB)

    if NA < 20 or NB < 20:
        return None

    def neg_logL(params):
        muA, muB, aAA, aBB, aAB, aBA, beta = params
        if any(p <= 0 for p in [muA, muB, beta]) or any(p < 0 for p in [aAA, aBB, aAB, aBA]):
            return 1e10
        if (aAA + aBA)/beta >= 0.999 or (aBB + aAB)/beta >= 0.999:
            return 1e10

        # intensities at A events
        AA_contrib = np.zeros(NA)
        BA_contrib = np.zeros(NA)
        for i in range(NA):
            for j in range(i):
                AA_contrib[i] += np.exp(-beta*(tA[i]-tA[j]))
            for j in range(NB):
                if tB[j] < tA[i]:
                    BA_contrib[i] += np.exp(-beta*(tA[i]-tB[j]))
        lamA = muA + aAA*AA_contrib + aBA*BA_contrib

        # intensities at B events
        BB_contrib = np.zeros(NB)
        AB_contrib = np.zeros(NB)
        for i in range(NB):
            for j in range(i):
                BB_contrib[i] += np.exp(-beta*(tB[i]-tB[j]))
            for j in range(NA):
                if tA[j] < tB[i]:
                    AB_contrib[i] += np.exp(-beta*(tB[i]-tA[j]))
        lamB = muB + aBB*BB_contrib + aAB*AB_contrib

        if np.any(lamA <= 0) or np.any(lamB <= 0):
            return 1e10

        logLA = np.sum(np.log(lamA)) - muA*T - \
                (aAA/beta)*np.sum(1-np.exp(-beta*(T-tA))) - \
                (aBA/beta)*np.sum(1-np.exp(-beta*(T-tB)))
        logLB = np.sum(np.log(lamB)) - muB*T - \
                (aBB/beta)*np.sum(1-np.exp(-beta*(T-tB))) - \
                (aAB/beta)*np.sum(1-np.exp(-beta*(T-tA)))

        return -(logLA + logLB)

    muA0 = NA / T
    muB0 = NB / T
    best_logL, best_params = -np.inf, None

    for b_init in [0.01, 0.05, 0.1]:
        for cross_init in [0.001, 0.01]:
            try:
                x0 = [muA0, muB0,
                       muA0*0.3*b_init, muB0*0.3*b_init,
                       cross_init, cross_init, b_init]
                res = optimize.minimize(
                    neg_logL, x0, method="Nelder-Mead",
                    options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8}
                )
                if -res.fun > best_logL:
                    p = res.x
                    if all(v > 0 for v in p):
                        best_logL = -res.fun
                        best_params = p
            except Exception:
                continue

    if best_params is None:
        return None

    muA, muB, aAA, aBB, aAB, aBA, beta = best_params
    return {
        "logL":      best_logL,
        "muA":       muA, "muB":   muB,
        "alpha_AA":  aAA, "alpha_BB": aBB,
        "alpha_AB":  aAB, "alpha_BA": aBA,
        "beta":      beta,
        "halflife":  np.log(2)/beta,
        "cross_AB_sig": aAB > 0.01 * muB,
        "cross_BA_sig": aBA > 0.01 * muA,
    }


# ── main tournament ───────────────────────────────────────────────────────────
def run_tournament(ticker, df, regimes):
    events = extract_events(df, ticker, build_session_index(df))
    if events is None or len(events) < MIN_EVENTS:
        n = 0 if events is None else len(events)
        log(f"    [{ticker}] only {n} events — below minimum {MIN_EVENTS}")
        return {"ticker": ticker, "winner": "INSUFFICIENT_DATA", "n_events": n}, events

    N          = len(events)
    T          = len(df)
    event_bars = events["bar_idx"].values.tolist()
    marks      = events["mark"].values.tolist()
    inter_arr  = np.diff(event_bars).tolist() if N > 1 else []

    log(f"\n  [{ticker}] {N} events over {T} bars "
        f"({T/26:.0f} sessions)")
    log(f"    Marks: min={min(marks):.2f}  mean={np.mean(marks):.2f}  "
        f"max={max(marks):.2f}")

    D = dispersion_index(event_bars, T)
    log(f"    Dispersion: {D:.3f}  "
        f"({'clustered' if D > 1.5 else 'near-Poisson' if D > 0.8 else 'regular'})")

    # regime series aligned to bars
    reg_series = None
    if regimes is not None:
        try:
            reg_series = regimes["combined"].reset_index(drop=True)
            if len(reg_series) != T:
                reg_series = reg_series.reindex(range(T), method="ffill").fillna(0)
        except Exception:
            reg_series = None

    log(f"    Fitting models...")

    logL0, k0, p0 = fit_hpp(event_bars, T)
    logL1, k1, p1 = fit_nhpp(event_bars, T, reg_series)
    logL2, k2, p2 = fit_renewal_weibull(inter_arr) if len(inter_arr) >= 10 else (-np.inf, 2, {})
    logL3, k3, p3 = fit_cox_gamma(event_bars, T)
    logL4, k4, p4 = fit_hawkes_exp(event_bars, T)
    logL5, k5, p5 = fit_marked_hawkes(event_bars, marks, T)
    logL6, k6, p6 = fit_cox_hawkes(event_bars, marks, T, reg_series)

    models = {
        "HPP":          (logL0, k0, p0),
        "NHPP":         (logL1, k1, p1),
        "Renewal":      (logL2, k2, p2),
        "Cox":          (logL3, k3, p3),
        "Hawkes":       (logL4, k4, p4),
        "MarkedHawkes": (logL5, k5, p5),
        "CoxHawkes":    (logL6, k6, p6),
    }

    log(f"\n    {'Model':<14} {'logL':>10} {'k':>4} {'AIC':>11} {'BIC':>11}")
    log(f"    " + "-"*55)

    aic_scores, bic_scores = {}, {}
    for name, (logL, k, _) in models.items():
        if np.isfinite(logL):
            a = aic(logL, k)
            b = bic(logL, k, N)
            aic_scores[name] = a
            bic_scores[name] = b
            log(f"    {name:<14} {logL:>10.2f} {k:>4} {a:>11.2f} {b:>11.2f}")
        else:
            log(f"    {name:<14} {'—':>10} {k:>4} {'—':>11} {'—':>11}")

    if not aic_scores:
        return {"ticker": ticker, "winner": "FIT_FAILED", "n_events": N}, events

    winner_aic = min(aic_scores, key=aic_scores.get)
    winner_bic = min(bic_scores, key=bic_scores.get)
    winner     = winner_aic

    # AIC margin over HPP
    aic_margin_vs_hpp = (aic_scores.get("HPP", np.nan) -
                         aic_scores.get(winner, np.nan))

    log(f"\n    LR Tests:")
    nested_pairs = [
        ("HPP",    "NHPP",         k1-k0),
        ("HPP",    "Hawkes",        k4-k0),
        ("NHPP",   "CoxHawkes",     k6-k1),
        ("Hawkes", "MarkedHawkes",  k5-k4),
        ("MarkedHawkes","CoxHawkes",k6-k5),
        ("Renewal","Hawkes",        k4-k2),
    ]
    lr_results = {}
    for r_name, f_name, df_lr in nested_pairs:
        lL_r = models[r_name][0]
        lL_f = models[f_name][0]
        if np.isfinite(lL_r) and np.isfinite(lL_f) and df_lr > 0:
            lr, p = lr_test(lL_r, lL_f, df_lr)
            sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            log(f"    {r_name:<14} vs {f_name:<14}: LR={lr:>7.2f}  p={p:.4f} {sig}")
            lr_results[f"{r_name}_vs_{f_name}"] = {"LR": lr, "p": p}

    # Ogata test on winner
    winner_params = models[winner][2]
    ks, p_ks, passed, compensator = ogata_residual_test(
        event_bars, T, winner, winner_params
    )

    if np.isfinite(ks):
        log(f"\n    Ogata ({winner}): KS={ks:.4f}  p={p_ks:.4f}  "
            f"{'PASS' if passed else 'FAIL'}")
        if not passed and np.isfinite(p_ks):
            log(f"    → Residuals reject Exp(1) — model may be misspecified "
                f"or a mixed process is needed")
    else:
        log(f"\n    Ogata: insufficient data for {winner}")

    # Weibull shape interpretation
    weibull_k = models["Renewal"][2].get("k", np.nan) if models["Renewal"][2] else np.nan
    if np.isfinite(weibull_k):
        log(f"    Weibull shape k={weibull_k:.3f}  →  "
            f"{models['Renewal'][2].get('shape_interpretation','')}")

    branching_ratio = winner_params.get("branching_ratio", np.nan)
    if np.isfinite(branching_ratio):
        stab = ("sub-critical" if branching_ratio < 0.8 else
                "near-critical" if branching_ratio < 0.95 else
                "near-unstable")
        log(f"    Branching ratio n={branching_ratio:.4f}  →  {stab}")

    halflife = winner_params.get("halflife_bars", np.nan)
    if np.isfinite(halflife):
        log(f"    Kernel half-life: {halflife:.1f} bars "
            f"({halflife/26:.1f} sessions)")

    log(f"\n    ► WINNER (AIC): {winner}  "
        f"[ΔAIC vs HPP = {aic_margin_vs_hpp:.1f}]")
    if winner_bic != winner_aic:
        log(f"    ► WINNER (BIC): {winner_bic}  (disagreement)")

    interp = {
        "HPP":          "Random, independent events. No model adds value.",
        "NHPP":         "Market regime drives rate. Build regime classifier.",
        "Renewal":      "Inter-arrival gap is the primary signal. Use Weibull hazard model.",
        "Cox":          "Latent burst regime. Identify external driver with covariates.",
        "Hawkes":       "Self-excitation confirmed. Intensity is the predictive signal.",
        "MarkedHawkes": "Self-excitation + rotation magnitude matters. Use ETAS form.",
        "CoxHawkes":    "Regime + self-excitation both significant. Full hybrid justified.",
    }.get(winner, "Unknown")
    log(f"    → {interp}")

    result = {
        "ticker":           ticker,
        "n_events":         N,
        "n_bars":           T,
        "n_sessions":       round(T/26),
        "dispersion":       round(D, 4) if np.isfinite(D) else None,
        "winner_aic":       winner_aic,
        "winner_bic":       winner_bic,
        "winner":           winner,
        "aic_margin_hpp":   round(aic_margin_vs_hpp, 2) if np.isfinite(aic_margin_vs_hpp) else None,
        "aic_hpp":          round(aic_scores.get("HPP",    np.nan), 2),
        "aic_nhpp":         round(aic_scores.get("NHPP",   np.nan), 2),
        "aic_renewal":      round(aic_scores.get("Renewal",np.nan), 2),
        "aic_cox":          round(aic_scores.get("Cox",    np.nan), 2),
        "aic_hawkes":       round(aic_scores.get("Hawkes", np.nan), 2),
        "aic_mhawkes":      round(aic_scores.get("MarkedHawkes", np.nan), 2),
        "aic_coxhawkes":    round(aic_scores.get("CoxHawkes",    np.nan), 2),
        "weibull_k":        round(weibull_k, 4) if np.isfinite(weibull_k) else None,
        "branching_ratio":  round(branching_ratio, 4) if np.isfinite(branching_ratio) else None,
        "halflife_bars":    round(halflife, 2) if np.isfinite(halflife) else None,
        "halflife_sessions":round(halflife/26, 2) if np.isfinite(halflife) else None,
        "ogata_ks":         round(ks, 4) if np.isfinite(ks) else None,
        "ogata_p":          round(p_ks, 4) if np.isfinite(p_ks) else None,
        "ogata_pass":       passed,
        "interpretation":   interp,
        "params":           winner_params,
        "_compensator":     compensator,
    }

    return result, events


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global VOLUME_THRESH, RETURN_THRESH, MIN_EVENTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",    nargs="*", default=TICKERS)
    parser.add_argument("--vol-thresh", type=float, default=VOLUME_THRESH)
    parser.add_argument("--ret-thresh", type=float, default=RETURN_THRESH)
    parser.add_argument("--min-events", type=int,   default=MIN_EVENTS)
    parser.add_argument("--market",     default=MARKET_TICKER)
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--pairs-only", action="store_true")
    args = parser.parse_args()

    VOLUME_THRESH = args.vol_thresh
    RETURN_THRESH = args.ret_thresh
    MIN_EVENTS    = args.min_events
    use_cache     = not args.no_cache

    log("=" * 65)
    log("  Low Float — Point Process Tournament  [15-minute bars]")
    log("=" * 65)
    log(f"  Universe      : {len(args.tickers)} tickers")
    log(f"  Lookback      : {LOOKBACK_DAYS} days (~{LOOKBACK_DAYS//365}y)")
    log(f"  Bar resolution: 1h (history) + 15m (last 60 days), session 09:30-16:00 ET")
    log(f"  Expected bars : ~{6*252*2:,} per symbol (1h) + recent 15m overlay")
    log(f"  Event filter  : vol > {VOLUME_THRESH}x MA{VOLUME_ROLL}  AND  |ret| > {RETURN_THRESH*100:.0f}%")
    log(f"  Min events    : {MIN_EVENTS}")
    log(f"  Cache         : {'disabled' if not use_cache else str(CACHE_DIR)}")
    log("=" * 65)

    # fetch market regime
    log(f"\n  Fetching market regime ({args.market})...")
    market_df = fetch_15m(args.market, use_cache=use_cache)
    if market_df is None:
        log("  WARNING: market data unavailable")

    all_results    = []
    all_events     = []
    all_compensators = []
    ticker_dfs     = {}
    winner_tally   = {}

    if not args.pairs_only:
        for ticker in args.tickers:
            log(f"\n{'─'*65}")
            log(f"  {ticker}")
            log(f"{'─'*65}")

            df = fetch_15m(ticker, use_cache=use_cache)
            if df is None or len(df) < 500:
                log(f"  [{ticker}] insufficient intraday data")
                all_results.append({"ticker": ticker, "winner": "NO_DATA", "n_events": 0})
                continue

            ticker_dfs[ticker] = df

            regimes = None
            if market_df is not None:
                try:
                    regimes = build_intraday_regimes(df, market_df)
                except Exception as e:
                    log(f"    regime build failed: {e}")

            result, events = run_tournament(ticker, df, regimes)
            all_results.append(result)

            if events is not None:
                events["ticker"] = ticker
                all_events.append(events)

            # collect compensator
            comp = result.pop("_compensator", [])
            if comp:
                comp_df = pd.DataFrame({
                    "ticker": ticker,
                    "idx":    range(len(comp)),
                    "residual_interval": comp,
                })
                all_compensators.append(comp_df)

            winner = result.get("winner", "UNKNOWN")
            winner_tally[winner] = winner_tally.get(winner, 0) + 1

    # ── bivariate Hawkes ──────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("  BIVARIATE HAWKES — MUTUAL EXCITATION PAIRS")
    log(f"{'='*65}")

    bivariate_results = []
    events_lookup = {}
    for ev_df in all_events:
        tk = ev_df["ticker"].iloc[0]
        events_lookup[tk] = ev_df["bar_idx"].values.tolist()

    for tickA, tickB in BIVARIATE_PAIRS:
        if tickA not in events_lookup or tickB not in events_lookup:
            log(f"\n  [{tickA}/{tickB}] one or both symbols missing events — skip")
            continue

        tA = events_lookup[tickA]
        tB = events_lookup[tickB]
        T  = max(
            len(ticker_dfs.get(tickA, pd.DataFrame())),
            len(ticker_dfs.get(tickB, pd.DataFrame())),
            1
        )

        log(f"\n  [{tickA} ↔ {tickB}]  |A|={len(tA)}  |B|={len(tB)}  T={T}")
        biv = fit_bivariate_hawkes(tA, tB, T)

        if biv is None:
            log(f"    Insufficient events for bivariate fit")
            continue

        log(f"    logL         = {biv['logL']:.2f}")
        log(f"    α_AA = {biv['alpha_AA']:.5f}   α_BB = {biv['alpha_BB']:.5f}  "
            f"(self-excitation)")
        log(f"    α_AB = {biv['alpha_AB']:.5f}   α_BA = {biv['alpha_BA']:.5f}  "
            f"(cross-excitation)")
        log(f"    Kernel half-life: {biv['halflife']:.1f} bars  "
            f"({biv['halflife']/26:.1f} sessions)")
        log(f"    {tickA}→{tickB} cross-excitation significant: {biv['cross_AB_sig']}")
        log(f"    {tickB}→{tickA} cross-excitation significant: {biv['cross_BA_sig']}")

        bivariate_results.append({
            "pair": f"{tickA}/{tickB}",
            "ticker_A": tickA, "ticker_B": tickB,
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in biv.items()}
        })

    # ── aggregate summary ─────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("  AGGREGATE SUMMARY")
    log(f"{'='*65}")

    valid = [r for r in all_results if r.get("winner") not in
             ("NO_DATA", "INSUFFICIENT_DATA", "FIT_FAILED", "UNKNOWN")]

    log(f"\n  Symbols processed   : {len(all_results)}")
    log(f"  Symbols with events : {len(valid)}")
    log(f"\n  ── Winner Distribution ──")
    for w, c in sorted(winner_tally.items(), key=lambda x: -x[1]):
        log(f"    {w:<16} : {c:>3}  {'█'*c}")

    hawkes_r = [r for r in valid if "Hawkes" in r.get("winner","")]
    if hawkes_r:
        brs = [r["branching_ratio"] for r in hawkes_r if r.get("branching_ratio")]
        hls = [r["halflife_sessions"] for r in hawkes_r if r.get("halflife_sessions")]
        if brs:
            log(f"\n  ── Hawkes Branching Ratios ──")
            log(f"    Mean={np.mean(brs):.4f}  Med={np.median(brs):.4f}  "
                f"Max={np.max(brs):.4f}")
        if hls:
            log(f"  ── Kernel Half-life (sessions) ──")
            log(f"    Mean={np.mean(hls):.2f}  Med={np.median(hls):.2f}")

    renewal_r = [r for r in valid if r.get("winner") == "Renewal"]
    if renewal_r:
        ks_vals = [r["weibull_k"] for r in renewal_r if r.get("weibull_k")]
        margins = [r["aic_margin_hpp"] for r in renewal_r if r.get("aic_margin_hpp")]
        ogata_p = [r["ogata_p"] for r in renewal_r if r.get("ogata_p") is not None]
        log(f"\n  ── Renewal Weibull k Distribution ──")
        if ks_vals:
            log(f"    Mean={np.mean(ks_vals):.3f}  "
                f">1 (incr hazard): {sum(k>1 for k in ks_vals)}/{len(ks_vals)}  "
                f"<1 (decr hazard): {sum(k<1 for k in ks_vals)}/{len(ks_vals)}")
        if margins:
            log(f"    Mean ΔAIC vs HPP: {np.mean(margins):.1f}  "
                f"(strong>6: {sum(m>6 for m in margins)}/{len(margins)})")
        if ogata_p:
            log(f"    Ogata pass rate: "
                f"{sum(p>0.05 for p in ogata_p)}/{len(ogata_p)}")

    log(f"\n  ── Per-Symbol Results ──")
    log(f"  {'Ticker':<7} {'N':>5} {'Winner':<14} {'k':>7} {'BR':>7} "
        f"{'HL':>5} {'D':>6} {'ΔAIC':>7} {'OgataP':>8} {'OgataOK':>8}")
    log(f"  " + "-"*85)
    for r in all_results:
        tk   = r.get("ticker","?")
        n    = r.get("n_events",0)
        w    = r.get("winner","?")[:13]
        wk   = f"{r['weibull_k']:.3f}" if r.get("weibull_k") else "  —  "
        br   = f"{r['branching_ratio']:.4f}" if r.get("branching_ratio") else "  —   "
        hl   = f"{r['halflife_sessions']:.1f}" if r.get("halflife_sessions") else " — "
        d    = f"{r['dispersion']:.2f}" if r.get("dispersion") else " — "
        da   = f"{r['aic_margin_hpp']:.1f}" if r.get("aic_margin_hpp") else "  —  "
        op   = f"{r['ogata_p']:.4f}" if r.get("ogata_p") is not None else "   —   "
        ok   = ("PASS" if r.get("ogata_pass") else
                "FAIL" if r.get("ogata_p") is not None else " — ")
        log(f"  {tk:<7} {n:>5} {w:<14} {wk:>7} {br:>7} "
            f"{hl:>5} {d:>6} {da:>7} {op:>8} {ok:>8}")

    # ── save outputs ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df = results_df.drop(columns=["params","_compensator"], errors="ignore")
    results_df.to_csv("process_id_15m_results.csv", index=False)
    log(f"\n  Saved → process_id_15m_results.csv")

    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            "process_id_15m_events.csv", index=False)
        log(f"  Saved → process_id_15m_events.csv")

    if all_compensators:
        pd.concat(all_compensators, ignore_index=True).to_csv(
            "process_id_15m_compensator.csv", index=False)
        log(f"  Saved → process_id_15m_compensator.csv  "
            f"(Ogata residuals for QQ inspection)")

    if bivariate_results:
        pd.DataFrame(bivariate_results).to_csv(
            "process_id_15m_bivariate.csv", index=False)
        log(f"  Saved → process_id_15m_bivariate.csv")

    log(f"\n{'='*65}")
    log("  INTERPRETATION GUIDE")
    log(f"{'='*65}")
    log("  HPP          → random events; no model justified")
    log("  NHPP         → regime drives rate; build regime classifier")
    log("  Renewal k>1  → hazard rises with time; momentum in timing")
    log("  Renewal k<1  → hazard falls; events cluster then go quiet")
    log("  Cox          → latent burst driver; identify with covariates")
    log("  Hawkes       → self-excitation; intensity IS the signal")
    log("  MarkedHawkes → self-excitation + float rotation magnitude")
    log("  CoxHawkes    → full model; regime + self-excitation")
    log(f"\n  Bivariate α_AB/α_BA > 0 → cross-symbol excitation detected")
    log(f"  Ogata PASS  → model correctly specified")
    log(f"  Ogata FAIL  → consider mixed or regime-switching extension")
    log(f"{'='*65}")


if __name__ == "__main__":
    main()