
"""
backtest.py  —  Swing Level Gate Backtest  [CRYPTO]  —  v10

THESIS
------
Price swing areas are market-generated supply/demand imprints — points
where the market has already proven it cannot sustain a move. A swing
high is where sellers overcame buyers; a swing low is where buyers
overcame sellers. These are actual structural levels, not calendar
references. Combined with Ψ̂ confirmation on breakout, they define
genuine inflection points where absorption can be measured.

ARCHITECTURE
------------
  RESAMPLE  1m cache → 5m bars.

  SWING DETECTION
            A swing high is a bar whose high is strictly greater than
            the N bars on each side (SWING_BARS = 5 or 10).
            A swing low is a bar whose low is strictly less than the
            N bars on each side.
            Only swings formed within the last NORM_WINDOW bars are
            considered active — stale structure beyond that window
            carries no microstructure relevance.
            Two runs: SWING_BARS = 5 and SWING_BARS = 10.

  ZONE      swing_price ± touch_band  (TOUCH_PCT × price)

  Ψ̂(t)     sigmoid( sign(Δ̈(t)) × z(NLI(t)) )
              sign(Δ̈) — directional component, always −1/0/+1
              z(NLI)  — liquidity confirmation magnitude
              NLI     = −sign(İLLIQ) × log(|İLLIQ| + ε)
              z-scored over NORM_WINDOW (same window as swing staleness)

  ENTRY     1. Price was in swing zone previous bar
            2. Current bar closes ABOVE zone (bullish breakout)
            3. Ψ̂ > PSI_ENTRY_LONG on breakout bar

  EXIT      Price structure only — no Ψ̂:
            Price arrives at any active swing zone AND closes BELOW
            that zone (bearish breakdown of structure)

  TIME      Safety cap: HOLD_BARS × 5m

Usage:
    python backtest.py
    python backtest.py --symbols BTC/USDT ETH/USDT
    python backtest.py --touch 0.001 --psi-long 0.65
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR       = Path("cache")

RESAMPLE_TF     = "5min"
MIN_BARS        = 200

TOUCH_PCT       = 0.001        # zone half-width as fraction of price

# Ψ̂ parameters
DELTA_WINDOW    = 3
AMIHUD_WINDOW   = 14
NORM_WINDOW     = 60           # NLI z-score window AND swing staleness cutoff

PSI_ENTRY_LONG  = 0.65

HOLD_BARS       = 60           # 60 × 5m = 5 hours max hold

# Swing detection — two runs
SWING_BARS_LIST = [5, 10]      # bars each side required to qualify

EPS = 1e-12

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

# ── resample ──────────────────────────────────────────────────────────────────
def resample_ohlcv(df, rule=RESAMPLE_TF):
    return df.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()

# ── Ψ̂ signal ──────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def compute_psi(close, volume):
    """
    Ψ̂(t) = sigmoid( sign(Δ̈(t)) × z(NLI(t)) )

    sign(Δ̈) — regime-independent direction: always −1, 0, +1
    z(NLI)  — liquidity confirmation magnitude, single well-behaved series
    """
    # tick-rule signed volume
    sign  = np.sign(close.diff()).replace(0, np.nan).ffill().fillna(1)
    delta = sign * volume

    # sign of flow acceleration
    delta_acc_sign = np.sign(delta.diff(DELTA_WINDOW).diff(DELTA_WINDOW))

    # Amihud illiquidity smoothed
    illiq     = (close.pct_change().abs() / (volume + 1)).rolling(AMIHUD_WINDOW).mean()
    illiq_dot = illiq.diff(DELTA_WINDOW)

    # NLI — log-scaled liquidity weight, sign of İLLIQ preserved
    nli      = -np.sign(illiq_dot) * np.log(illiq_dot.abs() + EPS)
    nli_std  = nli.rolling(NORM_WINDOW).std().replace(0, np.nan)
    nli_norm = nli / (nli_std + EPS)

    return (delta_acc_sign * nli_norm).apply(sigmoid)

# ── ATR ───────────────────────────────────────────────────────────────────────
def compute_atr(high, low, close, window=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low  - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# ── swing detection ───────────────────────────────────────────────────────────
def detect_swings(df, swing_bars):
    """
    Detect swing highs and lows using a rolling window approach.
    A swing high at bar i: high[i] > high[i-n] and high[i] > high[i+n]
                           for all n in 1..swing_bars
    A swing low  at bar i: low[i]  < low[i-n]  and low[i]  < low[i+n]
                           for all n in 1..swing_bars

    Returns two boolean Series: is_swing_high, is_swing_low
    Both are shifted forward by swing_bars so they only become
    visible AFTER the right-side confirmation bars have formed —
    no lookahead.
    """
    highs = df["High"]
    lows  = df["Low"]
    n     = len(df)

    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)

    sb = swing_bars
    for i in range(sb, n - sb):
        left_h  = highs.iloc[i - sb : i]
        right_h = highs.iloc[i + 1 : i + sb + 1]
        if highs.iloc[i] > left_h.max() and highs.iloc[i] > right_h.max():
            is_sh[i] = True

        left_l  = lows.iloc[i - sb : i]
        right_l = lows.iloc[i + 1 : i + sb + 1]
        if lows.iloc[i] < left_l.min() and lows.iloc[i] < right_l.min():
            is_sl[i] = True

    # shift forward by swing_bars — right-side bars must close before
    # the swing is confirmed. This is the lookahead prevention.
    is_sh_s = pd.Series(is_sh, index=df.index).shift(sb).fillna(False).astype(bool)
    is_sl_s = pd.Series(is_sl, index=df.index).shift(sb).fillna(False).astype(bool)

    return is_sh_s, is_sl_s

def get_active_swings(pos, swing_highs, swing_lows, high_arr, low_arr, norm_window):
    """
    Return active swing levels visible at bar `pos`.
    Active = confirmed swing formed within last norm_window bars.
    Returns list of (level_price, level_type).
    """
    levels = []
    lookback_start = max(0, pos - norm_window)

    for i in range(lookback_start, pos):
        if swing_highs[i]:
            levels.append((high_arr[i], "SwingHigh"))
        if swing_lows[i]:
            levels.append((low_arr[i], "SwingLow"))

    return levels

# ── zone checks ───────────────────────────────────────────────────────────────
def in_zone(price, level, touch_band):
    return abs(price - level) <= touch_band

def broke_above(close, level, touch_band):
    return close > level + touch_band

def broke_below(close, level, touch_band):
    return close < level - touch_band

# ── per-symbol backtest ───────────────────────────────────────────────────────
def backtest_symbol(symbol, df, swing_bars):
    df = resample_ohlcv(df)

    if len(df) < MIN_BARS + swing_bars * 2:
        return None, 0

    df["psi"] = compute_psi(df["Close"], df["Volume"])
    df["atr"] = compute_atr(df["High"], df["Low"], df["Close"])
    df.dropna(subset=["psi", "atr"], inplace=True)

    if len(df) < MIN_BARS:
        return None, 0

    # detect swings — confirmed after right-side bars close
    sh_series, sl_series = detect_swings(df, swing_bars)
    sh_arr = sh_series.values
    sl_arr = sl_series.values

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    psi_arr   = df["psi"].values
    n         = len(df)

    trades   = []
    in_trade = False

    for pos in range(swing_bars * 2 + NORM_WINDOW, n - HOLD_BARS - 1):

        if in_trade:
            continue

        price      = close_arr[pos]
        prev_price = close_arr[pos - 1]
        psi_val    = psi_arr[pos]
        touch_band = price * TOUCH_PCT

        # Ψ̂ gate — must pass before checking levels
        if psi_val <= PSI_ENTRY_LONG:
            continue

        # get active swing levels within norm_window
        active = get_active_swings(
            pos, sh_arr, sl_arr, high_arr, low_arr, NORM_WINDOW
        )
        if not active:
            continue

        # ── ENTRY ─────────────────────────────────────────────────────────────
        entry_level = None
        entry_type  = None

        for level, ltype in active:
            if in_zone(prev_price, level, touch_band) and \
               broke_above(price, level, touch_band):
                entry_level = level
                entry_type  = ltype
                break

        if entry_level is None:
            continue

        entry    = price
        in_trade = True
        exit_pos  = pos + HOLD_BARS
        exit_type = "time_stop"
        exit_lvl_type = None

        # ── EXIT LOOP ─────────────────────────────────────────────────────────
        for j in range(pos + 1, min(pos + HOLD_BARS + 1, n)):
            curr_price = close_arr[j]
            prev_close = close_arr[j - 1]
            curr_touch = curr_price * TOUCH_PCT

            # refresh active swings at exit bar
            exit_active = get_active_swings(
                j, sh_arr, sl_arr, high_arr, low_arr, NORM_WINDOW
            )

            for level, ltype in exit_active:
                if in_zone(prev_close, level, curr_touch) and \
                   broke_below(curr_price, level, curr_touch):
                    exit_pos      = j
                    exit_type     = "bearish_break"
                    exit_lvl_type = ltype
                    break

            if exit_type == "bearish_break":
                break

        in_trade = False

        ex  = close_arr[exit_pos]
        ret = (ex - entry) / entry * 100

        trades.append({
            "symbol":          symbol,
            "swing_bars":      swing_bars,
            "date":            df.index[pos],
            "entry":           round(float(entry), 8),
            "entry_level":     round(float(entry_level), 8),
            "entry_lvl_type":  entry_type,
            "exit_lvl_type":   exit_lvl_type,
            "psi_in":          round(float(psi_val), 6),
            "bars_held":       exit_pos - pos,
            "exit_type":       exit_type,
            "ret_pct":         round(float(ret), 6),
        })

    return trades, len(df)

# ── reporting ─────────────────────────────────────────────────────────────────
def report(tdf, ticker_stats, swing_bars, passed):
    rets = tdf["ret_pct"]
    n    = len(tdf)
    if n == 0:
        log("  No trades."); return

    wins = (rets > 0).sum()
    loss = (rets <= 0).sum()
    aw   = rets[rets > 0].mean() if wins > 0 else 0
    al   = rets[rets <= 0].mean() if loss > 0 else 0
    pf   = abs(aw * wins) / abs(al * loss + EPS)

    avg_hold = tdf["bars_held"].mean()
    sh       = rets.mean() / (rets.std() + EPS) * np.sqrt(525_600 / max(avg_hold * 5, 1))
    dd       = (rets.cumsum() - rets.cumsum().cummax()).min()

    log(f"  Symbols passed      : {passed}")
    log(f"  Total trades        : {n}")
    log(f"  Win Rate            : {wins/n*100:.2f}%")
    log(f"  Avg Return/Trade    : {rets.mean():+.6f}%")
    log(f"  Median Return       : {rets.median():+.6f}%")
    log(f"  Avg Win             : {aw:+.6f}%")
    log(f"  Avg Loss            : {al:+.6f}%")
    log(f"  Profit Factor       : {pf:.4f}")
    log(f"  Sharpe (annlzd)     : {sh:.4f}")
    log(f"  Max Drawdown        : {dd:.6f}%")
    log(f"  Skew / Kurt         : {rets.skew():.3f} / {rets.kurt():.3f}")
    log(f"  Best / Worst        : {rets.max():+.6f}% / {rets.min():+.6f}%")
    log(f"  Avg Bars Held (5m)  : {avg_hold:.1f}  ({avg_hold*5:.0f} min)")

    log(f"\n  ── Exit Type ─────────────────────────────────────────")
    for etype in ["bearish_break", "time_stop"]:
        grp = tdf[tdf["exit_type"] == etype]
        if not len(grp): continue
        log(f"    {etype:<20} : {len(grp):>5} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}% | "
            f"Avg Hold {grp['bars_held'].mean():.1f} bars")

    log(f"\n  ── Entry Level Type ──────────────────────────────────")
    for ltype, grp in tdf.groupby("entry_lvl_type", observed=True):
        log(f"    {str(ltype):<12} : {len(grp):>5} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    log(f"\n  ── Exit Level Type ───────────────────────────────────")
    for ltype, grp in tdf.groupby("exit_lvl_type", observed=True):
        if pd.isna(ltype): continue
        log(f"    {str(ltype):<12} : {len(grp):>5} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    log(f"\n  ── Ψ̂ at Entry (quintiles) ────────────────────────────")
    try:
        p_bins = pd.qcut(tdf["psi_in"], q=5, duplicates="drop")
        for label, grp in tdf.groupby(p_bins, observed=True):
            log(f"    Ψ̂ {str(label):<22} : {len(grp):>5} trades | "
                f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
                f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")
    except Exception:
        log("    (insufficient data for quintile split)")

    log(f"\n  ── Hold Duration ─────────────────────────────────────")
    tdf["hold_bin"] = pd.cut(tdf["bars_held"],
        bins=[0, 2, 5, 10, 20, 40, 60],
        labels=["1-2", "3-5", "6-10", "11-20", "21-40", "41-60"])
    for label, grp in tdf.groupby("hold_bin", observed=True):
        if not len(grp): continue
        lo = int(str(label).split("-")[0]) * 5
        hi = int(str(label).split("-")[-1]) * 5
        log(f"    {str(label):<8} bars ({lo}-{hi}min) : "
            f"{len(grp):>5} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    stat_df = pd.DataFrame(ticker_stats).sort_values("avg_ret", ascending=False)
    log(f"\n  ── Top 10 Symbols by Avg Return ──────────────────────")
    for _, r in stat_df.head(10).iterrows():
        log(f"    {r['symbol']:<14} : {r['n_trades']:>5} trades | "
            f"WR {r['win_rate']:>5.1f}% | AvgRet {r['avg_ret']:>+7.4f}%")

    tag = f"v10_swing{swing_bars}"
    tdf.to_csv(f"swing_trades_{tag}.csv", index=False)
    stat_df.to_csv(f"swing_stats_{tag}.csv", index=False)
    log(f"\n  Saved → swing_trades_{tag}.csv / swing_stats_{tag}.csv")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global TOUCH_PCT, PSI_ENTRY_LONG

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="*", default=None)
    parser.add_argument("--touch",    type=float, default=TOUCH_PCT)
    parser.add_argument("--psi-long", type=float, default=PSI_ENTRY_LONG)
    args = parser.parse_args()

    TOUCH_PCT      = args.touch
    PSI_ENTRY_LONG = args.psi_long

    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        log("  ERROR: no cached data found. Run fetch.py first.")
        return

    if args.symbols:
        wanted = {s.replace("/", "_") for s in args.symbols}
        parquet_files = [p for p in parquet_files if p.stem in wanted]

    # load all parquet files once — reused across both swing runs
    raw_data = {}
    for pf in parquet_files:
        symbol = pf.stem.replace("_", "/", 1)
        try:
            raw_data[symbol] = pd.read_parquet(pf)
        except Exception as e:
            log(f"  [{symbol}] load error: {e}")

    log(f"  Loaded {len(raw_data)} symbols from cache\n")

    # ── run both swing_bars configurations ────────────────────────────────────
    for swing_bars in SWING_BARS_LIST:

        log("=" * 65)
        log(f"  Swing Level Gate Backtest  —  v10 / swing_bars={swing_bars}")
        log("=" * 65)
        log(f"  RESAMPLE    : 1m → {RESAMPLE_TF}")
        log(f"  ZONE        : swing ± {TOUCH_PCT*100:.2f}%")
        log(f"  SWING DEF   : {swing_bars} bars each side, confirmed after right-side closes")
        log(f"  STALENESS   : active only within last {NORM_WINDOW} bars ({NORM_WINDOW*5}min)")
        log(f"  ENTRY       : prev bar in zone + close above zone + Ψ̂ > {PSI_ENTRY_LONG}")
        log(f"  EXIT        : price at any swing zone + close below zone")
        log(f"  TIME STOP   : {HOLD_BARS} bars ({HOLD_BARS*5}min)")
        log(f"  Ψ̂           : sigmoid( sign(Δ̈) × z(NLI) )")
        log("=" * 65)
        log("")

        all_trades   = []
        ticker_stats = []
        passed = 0

        for symbol, df in raw_data.items():
            trades, n_bars = backtest_symbol(symbol, df.copy(), swing_bars)
            if not trades:
                log(f"  [{symbol}] {n_bars} bars | 0 trades")
                continue

            rets = [t["ret_pct"] for t in trades]
            wr   = sum(1 for r in rets if r > 0) / len(trades) * 100
            ar   = np.mean(rets)
            log(f"  [{symbol:<14}] {n_bars:>5} bars | {len(trades):>4} trades | "
                f"WR {wr:>5.1f}% | AvgRet {ar:>+7.4f}%")

            all_trades.extend(trades)
            ticker_stats.append({
                "symbol":   symbol,
                "n_bars":   n_bars,
                "n_trades": len(trades),
                "win_rate": round(wr, 2),
                "avg_ret":  round(ar, 6),
            })
            passed += 1

        log("\n" + "=" * 65)
        log(f"  AGGREGATE RESULTS  —  swing_bars={swing_bars}")
        log("=" * 65)

        if all_trades:
            tdf = pd.DataFrame(all_trades)
            report(tdf, ticker_stats, swing_bars, passed)

        log("=" * 65 + "\n")

if __name__ == "__main__":
    main()