"""
Ψ̂(t) — Optuna Bayesian Optimization Wrapper  [CRYPTO / ccxt]
=================================================================
Optimizes 7 hyperparameters of the Psi signal using TPE (Tree-structured
Parzen Estimator) — the industry-standard Bayesian optimizer in Optuna.

Search space:
  DELTA_WINDOW       [1, 7]      differencing window for Δ̈ and ILLIQ̇
  AMIHUD_WINDOW      [5, 30]     smoothing window for ILLIQ
  NORM_WINDOW        [20, 120]   rolling std window for z-scoring both components
  PSI_ENTRY_LONG     [0.55, 0.85] entry threshold
  PSI_ENTRY_SHORT    [0.10, 0.50] exhaustion exit threshold (drop-relative also tried)
  HOLD_BARS          [3, 15]     max holding period in bars
  RELVOL_THRESHOLD   disabled    (removed per user instruction)

Train / Validation / Test split (temporal, no leakage):
  Train      : first 60% of each symbol's data  → optimizer sees this
  Validation : next 20%                          → objective function scored here
  Test       : final 20%                         → touched ONCE at the end

Objective : Sharpe ratio on the VALIDATION set
            (not train — prevents overfitting to historical noise)

Usage:
  pip install optuna ccxt pandas numpy
  python psi_optuna.py

  Outputs:
    optuna_best_params.json     — best params found
    optuna_study.csv            — all trial results
    psi_test_results.csv        — final out-of-sample test trades
    psi_test_stats.csv          — per-symbol test stats
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ccxt
import optuna
import json
import time
import sys
from datetime import datetime, timedelta
from copy import deepcopy

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── universe (same as backtest) ───────────────────────────────────────────────
CANDIDATE_SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT","AVAX/USDT","DOT/USDT","LTC/USDT",
    "LINK/USDT","UNI/USDT","ATOM/USDT","BCH/USDT","ICP/USDT","HBAR/USDT","ETC/USDT",
    "NEAR/USDT","APT/USDT","OP/USDT","ARB/USDT","GRT/USDT","FTM/USDT","MKR/USDT",
    "TIA/USDT","SUI/USDT","INJ/USDT","STX/USDT","SEI/USDT","AAVE/USDT",
    "GALA/USDT","SAND/USDT","MANA/USDT","ALGO/USDT","AXS/USDT","PEPE/USDT",
    "DOGE/USDT","BONK/USDT","WIF/USDT","JUP/USDT","PYTH/USDT",
    "FET/USDT","OCEAN/USDT","RUNE/USDT","CRV/USDT","SUSHI/USDT","DYDX/USDT",
    "ENS/USDT","GMT/USDT","ONE/USDT","ANKR/USDT","BAT/USDT","KAVA/USDT",
    "SNX/USDT","ZEC/USDT","ICX/USDT","KAS/USDT","MAGIC/USDT","BLUR/USDT",
    "WLD/USDT","TON/USDT","EIGEN/USDT","VIRTUAL/USDT","TRUMP/USDT","GRASS/USDT",
]

# ── fixed settings ────────────────────────────────────────────────────────────
LOOKBACK_DAYS     = 730
TIMEFRAME         = "1d"
MIN_BARS          = 120    # minimum bars after dropna to be included
EXCHANGE_PRIORITY = ["okx", "kraken", "binanceus"]
REQUEST_DELAY     = 0.35

# ── optimization settings ─────────────────────────────────────────────────────
N_TRIALS          = 100    # number of Optuna trials (more = better, slower)
N_STARTUP_TRIALS  = 20     # random exploration before TPE kicks in

# Train/Val/Test split ratios (temporal — no shuffling)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# TEST_RATIO  = 0.20 (implicit remainder)


# ── signal functions (parameterized) ─────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def compute_psi(close, volume, delta_window, amihud_window, norm_window):
    eps = 1e-12

    # Δ(t) — tick-rule flow delta
    sign  = np.sign(close.diff()).replace(0, np.nan).ffill().fillna(1)
    delta = sign * volume

    # Δ̈(t) — flow acceleration
    d1        = delta.diff(delta_window)
    delta_acc = d1.diff(delta_window)

    # ILLIQ(t) — Amihud illiquidity
    illiq     = (close.pct_change().abs() / (volume + 1)).rolling(amihud_window).mean()

    # ILLIQ̇(t) — rate of change
    illiq_dot = illiq.diff(delta_window)

    # NLI — log-scaled liquidity weight (original form retained)
    nli      = -np.sign(illiq_dot) * np.log(illiq_dot.abs() + eps)

    # Normalise both components
    da_std        = delta_acc.rolling(norm_window).std().replace(0, np.nan)
    da_norm       = delta_acc / (da_std + eps)

    nli_std       = nli.rolling(norm_window).std().replace(0, np.nan)
    nli_norm      = nli / (nli_std + eps)

    psi = (da_norm * nli_norm).apply(sigmoid)
    return psi

def compute_relvol(close, window=20):
    ret = close.pct_change()
    return ret.abs() / (ret.rolling(window).std() + 1e-12)

def run_backtest_on_slice(data_slices, params):
    """
    Run backtest on a dict of {symbol: df} slices using the given params.
    Returns list of trade dicts.
    """
    delta_window    = params["delta_window"]
    amihud_window   = params["amihud_window"]
    norm_window     = params["norm_window"]
    psi_entry_long  = params["psi_entry_long"]
    psi_entry_short = params["psi_entry_short"]
    hold_bars       = params["hold_bars"]

    all_trades = []

    for symbol, df in data_slices.items():
        if len(df) < norm_window + 30:
            continue

        close  = df["Close"]
        volume = df["Volume"]

        psi = compute_psi(close, volume, delta_window, amihud_window, norm_window)

        aligned = pd.DataFrame({
            "close": close,
            "psi":   psi,
        }).dropna()

        if len(aligned) < hold_bars + 1:
            continue

        idx = aligned.index.tolist()

        for dt in aligned.index:
            pos = idx.index(dt)
            if pos + hold_bars >= len(idx):
                continue

            psi_val = aligned.loc[dt, "psi"]
            if psi_val <= psi_entry_long:
                continue

            # Exit: signal exhaustion or time stop
            exit_pos = pos + hold_bars
            for j in range(pos + 1, min(pos + hold_bars + 1, len(idx))):
                if aligned.loc[idx[j], "psi"] < psi_entry_short:
                    exit_pos = j
                    break

            entry = aligned.loc[idx[pos],      "close"]
            exit_ = aligned.loc[idx[exit_pos], "close"]
            ret   = (exit_ - entry) / entry * 100

            all_trades.append({
                "symbol":    symbol,
                "date":      dt,
                "psi":       float(psi_val),
                "bars_held": exit_pos - pos,
                "exhausted": (exit_pos - pos) < hold_bars,
                "ret_pct":   round(float(ret), 4),
            })

    return all_trades

def sharpe_from_trades(trades, annual_factor=365, hold_bars=5):
    if len(trades) < 10:
        return -999.0
    rets = pd.Series([t["ret_pct"] for t in trades])
    std  = rets.std()
    if std == 0:
        return -999.0
    return float(rets.mean() / std * np.sqrt(annual_factor / hold_bars))


# ── exchange + data ───────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def make_exchange(exchange_id):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    markets = ex.load_markets()
    test = ex.fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, limit=5)
    if not test:
        raise RuntimeError("empty test fetch")
    return ex, set(markets.keys())

def connect_exchange():
    for eid in EXCHANGE_PRIORITY:
        try:
            log(f"  Trying {eid} ...")
            ex, avail = make_exchange(eid)
            log(f"  ✓ {eid} connected")
            return ex, avail, eid
        except Exception as e:
            log(f"  ✗ {eid}: {e}")
    log("FATAL: no exchange available"); sys.exit(1)

def fetch_all_data(exchange, available, since_ms):
    """
    Fetch all symbols once upfront. Returns dict {symbol: df}.
    Data fetched once and reused across all Optuna trials — critical for speed.
    """
    log(f"\n  Fetching data for {len(CANDIDATE_SYMBOLS)} symbols ...")
    data = {}
    for i, sym in enumerate(CANDIDATE_SYMBOLS):
        if sym not in available:
            continue
        time.sleep(REQUEST_DELAY)
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, since=since_ms, limit=1000)
            if not ohlcv or len(ohlcv) < MIN_BARS:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=1000)
            if not ohlcv or len(ohlcv) < MIN_BARS:
                continue
            df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[["Open","High","Low","Close","Volume"]].astype(float).dropna()
            if len(df) >= MIN_BARS:
                data[sym] = df
                log(f"  [{i+1:>3}] {sym:<14} — {len(df)} bars")
        except Exception:
            pass
    log(f"\n  Fetched {len(data)} symbols with sufficient data")
    return data

def split_data(all_data):
    """
    Temporal train/val/test split per symbol. No shuffling — preserves time order.
    """
    train, val, test = {}, {}, {}
    for sym, df in all_data.items():
        n      = len(df)
        t_end  = int(n * TRAIN_RATIO)
        v_end  = int(n * (TRAIN_RATIO + VAL_RATIO))
        train[sym] = df.iloc[:t_end]
        val[sym]   = df.iloc[t_end:v_end]
        test[sym]  = df.iloc[v_end:]
    return train, val, test


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(train_data, val_data):
    def objective(trial):
        params = {
            "delta_window":    trial.suggest_int("delta_window",    1,   7),
            "amihud_window":   trial.suggest_int("amihud_window",   5,  30),
            "norm_window":     trial.suggest_int("norm_window",     20, 120),
            "psi_entry_long":  trial.suggest_float("psi_entry_long",  0.55, 0.85),
            "psi_entry_short": trial.suggest_float("psi_entry_short", 0.10, 0.50),
            "hold_bars":       trial.suggest_int("hold_bars",       3,  15),
        }

        # Score on TRAIN first — prune obviously bad configs fast
        train_trades = run_backtest_on_slice(train_data, params)
        train_sharpe = sharpe_from_trades(train_trades, hold_bars=params["hold_bars"])
        if train_sharpe < -2.0:
            raise optuna.exceptions.TrialPruned()

        # Score on VALIDATION — this is what the optimizer actually maximizes
        val_trades  = run_backtest_on_slice(val_data, params)
        val_sharpe  = sharpe_from_trades(val_trades, hold_bars=params["hold_bars"])

        n_val = len(val_trades)
        wr    = sum(1 for t in val_trades if t["ret_pct"] > 0) / max(n_val, 1) * 100
        avg_r = np.mean([t["ret_pct"] for t in val_trades]) if val_trades else 0

        log(f"  Trial {trial.number:>3} | "
            f"dw={params['delta_window']} aw={params['amihud_window']} "
            f"nw={params['norm_window']} el={params['psi_entry_long']:.2f} "
            f"es={params['psi_entry_short']:.2f} hb={params['hold_bars']} | "
            f"val_sharpe={val_sharpe:+.3f} WR={wr:.1f}% n={n_val}")

        return val_sharpe

    return objective


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(start_dt.timestamp() * 1000)

    log("=" * 65)
    log("  Ψ̂(t) — Optuna Bayesian Optimization  [TPE / Sharpe objective]")
    log("=" * 65)
    log(f"  Trials         : {N_TRIALS}  ({N_STARTUP_TRIALS} random warmup)")
    log(f"  Split          : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/20  train/val/test")
    log(f"  Objective      : Validation Sharpe (annualized, 365-day crypto)")
    log(f"  NLI formula    : -sign(ILLIQ̇) × log(|ILLIQ̇|+ε)  [log form]")
    log(f"  RelVol gate    : DISABLED — all Ψ̂ > threshold bars eligible")
    log("=" * 65)

    # ── connect and fetch data once ───────────────────────────────────────────
    exchange, available, exchange_id = connect_exchange()
    all_data = fetch_all_data(exchange, available, since_ms)

    if len(all_data) < 5:
        log("FATAL: fewer than 5 symbols fetched — cannot optimize"); sys.exit(1)

    train_data, val_data, test_data = split_data(all_data)
    log(f"\n  Split sizes:")
    log(f"    Train symbols : {len(train_data)}")
    log(f"    Val   symbols : {len(val_data)}")
    log(f"    Test  symbols : {len(test_data)}")

    # ── run optimization ──────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log(f"  Running {N_TRIALS} Optuna trials ...")
    log(f"{'='*65}")

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=42,
        multivariate=True,   # models parameter correlations — more accurate than independent TPE
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="psi_hat_optimization",
    )
    study.optimize(make_objective(train_data, val_data), n_trials=N_TRIALS)

    # ── results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    best_params = best.params

    log(f"\n{'='*65}")
    log(f"  OPTIMIZATION COMPLETE")
    log(f"{'='*65}")
    log(f"  Best val Sharpe   : {best.value:+.4f}")
    log(f"  Best params:")
    for k, v in best_params.items():
        log(f"    {k:<20} : {v}")

    # ── evaluate on test set (touched once, right now) ────────────────────────
    log(f"\n{'='*65}")
    log(f"  OUT-OF-SAMPLE TEST RESULTS  (never seen during optimization)")
    log(f"{'='*65}")

    test_trades = run_backtest_on_slice(test_data, best_params)

    if not test_trades:
        log("  No trades on test set.")
    else:
        td   = pd.DataFrame(test_trades)
        rets = td["ret_pct"]
        n    = len(rets)
        wins = (rets > 0).sum()
        aw   = rets[rets > 0].mean() if wins > 0 else 0
        al_  = rets[rets <= 0].mean() if (n - wins) > 0 else 0
        pf   = abs(aw * wins) / (abs(al_ * (n - wins)) + 1e-12)
        sh   = sharpe_from_trades(test_trades, hold_bars=best_params["hold_bars"])
        mdd  = (rets.cumsum() - rets.cumsum().cummax()).min()

        log(f"  Symbols in test   : {td['symbol'].nunique()}")
        log(f"  Total trades      : {n}")
        log(f"  Win Rate          : {wins/n*100:.2f}%")
        log(f"  Avg Return/Trade  : {rets.mean():+.4f}%")
        log(f"  Median Return     : {rets.median():+.4f}%")
        log(f"  Avg Win           : {aw:+.4f}%")
        log(f"  Avg Loss          : {al_:+.4f}%")
        log(f"  Profit Factor     : {pf:.3f}")
        log(f"  Sharpe (annlzd)   : {sh:.4f}")
        log(f"  Max Drawdown      : {mdd:.4f}%")
        log(f"  Best Trade        : {rets.max():+.4f}%")
        log(f"  Worst Trade       : {rets.min():+.4f}%")

        # Ψ̂ quantile on test
        log(f"\n  ── Ψ̂ Quantile Breakdown (test) ──────────────────────")
        td["psi_bin"] = pd.cut(td["psi"],
            bins=[0,.55,.60,.70,.80,1.],
            labels=[".55-.60",".60-.70",".70-.80",">.80"])
        for lbl, grp in td.groupby("psi_bin", observed=True):
            wr_g = (grp["ret_pct"]>0).mean()*100
            log(f"    Ψ̂ {str(lbl):<10}: {len(grp):>5} trades | "
                f"WR {wr_g:>5.1f}% | AvgRet {grp['ret_pct'].mean():>+6.3f}%")

        # Exit type on test
        log(f"\n  ── Exit Type Breakdown (test) ────────────────────────")
        ex  = td[td["exhausted"]==True]
        tim = td[td["exhausted"]==False]
        for lbl, grp in [("Signal reversal", ex), ("Time stop", tim)]:
            if len(grp)==0: continue
            log(f"    {lbl:<20}: {len(grp):>5} trades | "
                f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
                f"AvgRet {grp['ret_pct'].mean():>+6.3f}%")

        td.to_csv("psi_test_results.csv", index=False)
        log(f"\n  Test trades saved → psi_test_results.csv")

    # ── save study and params ─────────────────────────────────────────────────
    with open("optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    study_df = study.trials_dataframe()
    study_df.to_csv("optuna_study.csv", index=False)

    log(f"  Best params saved → optuna_best_params.json")
    log(f"  All trials saved  → optuna_study.csv")
    log("=" * 65)
    log(f"\n  To run your backtest with optimized params,")
    log(f"  update psi_backtest_ccxt_v4.py constants to:")
    for k, v in best_params.items():
        log(f"    {k.upper():<22} = {v}")
    log("=" * 65)


if __name__ == "__main__":
    main()
    