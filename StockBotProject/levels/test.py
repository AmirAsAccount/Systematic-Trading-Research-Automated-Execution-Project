"""
psi_validate_v2.py  —  Ψ̂ Variant Comparison  [CRYPTO]

Three signal variants tested side by side against raw forward returns.
No entry zones, no exits, no hold logic — pure directional signal test.

VARIANTS
--------
  A  sigmoid( sign(Δ̇) )
       Direction only. Binary ±1 input to sigmoid → {~0.27, 0.50, ~0.73}.
       Baseline: does flow direction alone predict anything?

  B  sigmoid( sign(Δ̇) × |z(Δ̇)| )
       Direction × flow magnitude (z-scored Δ̇, absolute value).
       Tests whether bigger flow changes = stronger signal,
       without the Amihud path.

  C  sigmoid( sign(Δ̇) × z(NLI) )   ← current equation
       Direction × liquidity-weighted magnitude.
       NLI = -sign(İLLIQ) × log(|İLLIQ| + ε)
       Kept as baseline. Prior results show quintile inversion at high Ψ̂.

KEY DIAGNOSTIC: quintile accuracy table.
  If monotonically increasing → magnitude is adding real information.
  If flat                     → only direction matters (use A).
  If monotonically decreasing → magnitude is inverting the signal (NLI problem).

Usage:
    python psi_validate_v2.py
    python psi_validate_v2.py --symbols BTC/USDT ETH/USDT SOL/USDT
    python psi_validate_v2.py --horizons 1 3 6 12 24
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR       = Path("cache")
RESAMPLE_TF     = "5min"
MIN_BARS        = 200

DELTA_WINDOW    = 3
AMIHUD_WINDOW   = 14
NORM_WINDOW     = 60

PSI_LONG_THRESH  = 0.65
PSI_SHORT_THRESH = 0.35

HORIZONS    = [1, 3, 6, 12, 24]
CLUSTER_GAP = 3
EPS         = 1e-12

VARIANTS = ["A_direction", "B_flow_mag", "C_nli"]
VARIANT_LABELS = {
    "A_direction": "A  sigmoid(sign(Δ̇))",
    "B_flow_mag":  "B  sigmoid(sign(Δ̇) × |z(Δ̇)|)",
    "C_nli":       "C  sigmoid(sign(Δ̇) × z(NLI))  ← current",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

def resample_ohlcv(df, rule=RESAMPLE_TF):
    return df.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

# ── three variants ────────────────────────────────────────────────────────────
def compute_psi_variants(close, volume):
    sign      = np.sign(close.diff()).replace(0, np.nan).ffill().fillna(1)
    delta     = sign * volume
    delta_dot = delta.diff(DELTA_WINDOW)
    dd_sign   = np.sign(delta_dot)

    # A — direction only
    psi_a = dd_sign.apply(sigmoid)

    # B — direction × |z(Δ̇)|
    dd_mean = delta_dot.rolling(NORM_WINDOW).mean()
    dd_std  = delta_dot.rolling(NORM_WINDOW).std().replace(0, np.nan)
    dd_z    = (delta_dot - dd_mean) / (dd_std + EPS)
    psi_b   = (dd_sign * dd_z.abs()).apply(sigmoid)

    # C — direction × z(NLI)  [current equation]
    illiq     = (close.pct_change().abs() / (volume + 1)).rolling(AMIHUD_WINDOW).mean()
    illiq_dot = illiq.diff(DELTA_WINDOW)
    nli       = -np.sign(illiq_dot) * np.log(illiq_dot.abs() + EPS)
    nli_std   = nli.rolling(NORM_WINDOW).std().replace(0, np.nan)
    nli_norm  = nli / (nli_std + EPS)
    psi_c     = (dd_sign * nli_norm).apply(sigmoid)

    return {"A_direction": psi_a, "B_flow_mag": psi_b, "C_nli": psi_c}

# ── per-symbol validation ─────────────────────────────────────────────────────
def validate_symbol(symbol, df, long_thresh, short_thresh, horizons):
    df = resample_ohlcv(df)
    if len(df) < MIN_BARS:
        return None

    variants = compute_psi_variants(df["Close"], df["Volume"])

    # drop rows where any variant is NaN
    psi_df = pd.DataFrame(variants, index=df.index)
    df = df.join(psi_df)
    df.dropna(subset=list(variants.keys()), inplace=True)
    if len(df) < MIN_BARS:
        return None

    close  = df["Close"].values
    n      = len(df)
    max_h  = max(horizons)

    fwd = {}
    for h in horizons:
        fwd[h] = pd.Series(close).pct_change(h).shift(-h).values * 100

    # one record per bar per variant
    all_records = {v: [] for v in VARIANTS}

    for i in range(NORM_WINDOW, n - max_h - 1):
        for vname in VARIANTS:
            p        = df[vname].iloc[i]
            is_long  = p > long_thresh
            is_short = p < short_thresh
            if not (is_long or is_short):
                continue

            row = {
                "symbol":    symbol,
                "bar":       i,
                "psi":       round(float(p), 6),
                "direction": "long" if is_long else "short",
            }
            for h in horizons:
                ret = fwd[h][i]
                if np.isnan(ret):
                    continue
                signed_ret = ret if is_long else -ret
                row[f"fwd_{h}"]     = round(float(ret), 6)
                row[f"correct_{h}"] = 1 if signed_ret > 0 else 0

            all_records[vname].append(row)

    return {
        "symbol":  symbol,
        "n_bars":  n,
        "records": {v: pd.DataFrame(all_records[v]) for v in VARIANTS},
    }

# ── quintile table for one variant ───────────────────────────────────────────
def quintile_table(long_df, h_mid, variant_label):
    col_c = f"correct_{h_mid}"
    col_r = f"fwd_{h_mid}"
    if col_c not in long_df.columns or len(long_df.dropna(subset=[col_c])) < 50:
        log("    (insufficient data)")
        return
    sub = long_df.dropna(subset=[col_c]).copy()
    try:
        sub["psi_q"] = pd.qcut(sub["psi"], q=5, duplicates="drop")
        for label, grp in sub.groupby("psi_q", observed=True):
            acc = grp[col_c].mean() * 100
            mr  = grp[col_r].mean()
            log(f"    Ψ̂ {str(label):<25} : {len(grp):>5} sigs | "
                f"Acc {acc:>5.1f}% | MeanRet {mr:>+7.4f}%")
    except Exception as e:
        log(f"    (quintile failed: {e})")

# ── aggregate report for one variant ─────────────────────────────────────────
def report_variant(vname, all_records, horizons):
    label = VARIANT_LABELS[vname]
    log(f"\n{'='*65}")
    log(f"  VARIANT {label}")
    log(f"{'='*65}")

    if not all_records:
        log("  No data."); return

    df       = pd.DataFrame(all_records)
    long_df  = df[df["direction"] == "long"]
    short_df = df[df["direction"] == "short"]
    log(f"  Signals: {len(df)}  (long: {len(long_df)}, short: {len(short_df)})")

    # accuracy by horizon
    log(f"\n  {'Horizon':>8}  {'Dir':>6}  {'N':>6}  {'Accuracy':>10}  "
        f"{'MeanRet':>9}  {'p-val':>8}")
    log(f"  " + "-" * 58)
    for h in horizons:
        col_c = f"correct_{h}"
        col_r = f"fwd_{h}"
        for direction, sub in [("long", long_df), ("short", short_df)]:
            sub_h = sub.dropna(subset=[col_c, col_r]) if col_c in sub.columns else sub
            if len(sub_h) < 10:
                continue
            acc  = sub_h[col_c].mean() * 100
            mr   = sub_h[col_r].mean() if direction == "long" else -sub_h[col_r].mean()
            rets = sub_h[col_r].values if direction == "long" else -sub_h[col_r].values
            _, pv = stats.ttest_1samp(rets[~np.isnan(rets)], 0)
            sig  = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            log(f"  {h*5:>6}min  {direction:>6}  {len(sub_h):>6}  "
                f"{acc:>9.2f}%  {mr:>+8.4f}%  p={pv:.4f}{sig}")

    # quintile table — the key diagnostic
    h_mid = horizons[min(2, len(horizons)-1)]
    log(f"\n  ── Quintile Accuracy (long, {h_mid*5}min) — INVERSION DIAGNOSTIC ──")
    log(f"  Monotone UP=good  FLAT=direction-only  DOWN=magnitude inverting")
    quintile_table(long_df, h_mid, label)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    global PSI_LONG_THRESH, PSI_SHORT_THRESH, HORIZONS

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",      nargs="*", default=None)
    parser.add_argument("--long-thresh",  type=float, default=PSI_LONG_THRESH)
    parser.add_argument("--short-thresh", type=float, default=PSI_SHORT_THRESH)
    parser.add_argument("--horizons",     nargs="*", type=int, default=HORIZONS)
    args = parser.parse_args()

    PSI_LONG_THRESH  = args.long_thresh
    PSI_SHORT_THRESH = args.short_thresh
    HORIZONS         = sorted(args.horizons)

    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        log("  ERROR: no cached data. Run fetch.py first."); return

    if args.symbols:
        wanted = {s.replace("/", "_") for s in args.symbols}
        parquet_files = [p for p in parquet_files if p.stem in wanted]

    log("=" * 65)
    log("  Ψ̂ Variant Comparison — Three Equations Side by Side")
    log("=" * 65)
    log(f"  A  sigmoid( sign(Δ̇) )")
    log(f"  B  sigmoid( sign(Δ̇) × |z(Δ̇)| )")
    log(f"  C  sigmoid( sign(Δ̇) × z(NLI) )  ← current")
    log(f"  RESAMPLE : 1m → {RESAMPLE_TF}  |  LONG > {PSI_LONG_THRESH}  |  SHORT < {PSI_SHORT_THRESH}")
    log(f"  HORIZONS : {[h*5 for h in HORIZONS]} min  |  NO entry/exit logic")
    log("=" * 65)

    collected = {v: [] for v in VARIANTS}
    processed = 0

    for pf in parquet_files:
        symbol = pf.stem.replace("_", "/", 1)
        try:
            raw = pd.read_parquet(pf)
        except Exception as e:
            log(f"  [{symbol}] load error: {e}"); continue

        result = validate_symbol(symbol, raw, PSI_LONG_THRESH, PSI_SHORT_THRESH, HORIZONS)
        if result is None:
            continue

        # per-symbol summary line uses variant C (current) as reference
        h_mid = HORIZONS[min(2, len(HORIZONS)-1)]
        col_c = f"correct_{h_mid}"
        accs  = {}
        for v in VARIANTS:
            recs = result["records"][v]
            if recs.empty or col_c not in recs.columns:
                accs[v] = 0.0
            else:
                l = recs[recs["direction"]=="long"]
                accs[v] = l[col_c].mean() * 100 if len(l) > 0 else 0.0
            collected[v].extend(recs.to_dict("records"))

        log(f"  [{symbol:<14}] {result['n_bars']:>5} bars | "
            f"A:{accs['A_direction']:>5.1f}%  "
            f"B:{accs['B_flow_mag']:>5.1f}%  "
            f"C:{accs['C_nli']:>5.1f}%  "
            f"(L-acc@{h_mid*5}m)")
        processed += 1

    log(f"\n  Processed {processed} symbols")

    for v in VARIANTS:
        report_variant(v, collected[v], HORIZONS)

    log(f"\n{'='*65}")
    log("  SUMMARY — Quintile Monotonicity")
    log(f"{'='*65}")
    log("  Check output above:")
    log("  A flat quintile  → direction is the whole signal, magnitude adds noise")
    log("  B rising quintile → |z(Δ̇)| is real confirmation, use B")
    log("  C falling quintile → NLI is inverting at extremes (confirmed)")
    log(f"{'='*65}")


if __name__ == "__main__":
    main()