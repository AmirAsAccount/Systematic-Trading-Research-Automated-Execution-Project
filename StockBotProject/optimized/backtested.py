"""
Ψ̂(t) — Flow Direction × Liquidity Regime Backtest  [CRYPTO / ccxt]
Signal: CKS flow direction gating log-scaled Amihud illiquidity derivative, sigmoid-normalized

v5 changes from v4:
  - Removed delta acceleration (Δ̈) entirely
  - CKS delta now contributes only its sign — pure aggressor direction (+1 buy / -1 sell)
  - NLI_norm provides all magnitude — how strongly is the liquidity regime shifting
  - Ψ̂(t) = sigmoid( sign(Δ) × NLI_norm )
  - Interpretation: signal is high when buyers are aggressing into a deepening book,
    low when sellers are aggressing or book is thinning regardless of flow direction
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import sys

# ── universe ──────────────────────────────────────────────────────────────────
CANDIDATE_SYMBOLS = [
  "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
  "LINK/USDT", "UNI/USDT", "TRX/USDT", "ATOM/USDT", "BCH/USDT", "XLM/USDT", "ICP/USDT", "FIL/USDT", "HBAR/USDT", "ETC/USDT",
  "NEAR/USDT", "VET/USDT", "APT/USDT", "OP/USDT", "ARB/USDT", "GRT/USDT", "RNDR/USDT", "THETA/USDT", "FTM/USDT", "MKR/USDT",
  "LDO/USDT", "TIA/USDT", "SUI/USDT", "INJ/USDT", "STX/USDT", "SEI/USDT", "IMX/USDT", "EGLD/USDT", "AAVE/USDT", "FLOW/USDT",
  "QNT/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT", "CHZ/USDT", "ALGO/USDT", "MINA/USDT", "AXS/USDT", "APE/USDT", "PEPE/USDT",
  "SHIB/USDT", "DOGE/USDT", "BONK/USDT", "FLOKI/USDT", "WIF/USDT", "MEME/USDT", "BOME/USDT", "JUP/USDT", "PYTH/USDT", "ONDO/USDT",
  "PENDLE/USDT", "FET/USDT", "AGIX/USDT", "OCEAN/USDT", "AKT/USDT", "W/USDT", "ENA/USDT", "STRK/USDT", "DYM/USDT", "MANTA/USDT",
  "ALT/USDT", "XAI/USDT", "NFP/USDT", "ACE/USDT", "JTO/USDT", "ORDI/USDT", "SATS/USDT",
  "NOT/USDT", "BB/USDT", "REZ/USDT", "ETHFI/USDT", "AEVO/USDT", "METIS/USDT", "RON/USDT",
  "PIXEL/USDT", "PORTAL/USDT", "ZETA/USDT", "OMNI/USDT", "TAO/USDT", "TNSR/USDT", "CRV/USDT", "SUSHI/USDT", "1INCH/USDT", "DYDX/USDT",
  "ENS/USDT", "WOO/USDT", "GMT/USDT", "JASMY/USDT", "CKB/USDT", "ZIL/USDT", "ENJ/USDT", "HOT/USDT", "ONE/USDT", "ANKR/USDT",
  "RVN/USDT", "BAT/USDT", "LRC/USDT", "KAVA/USDT", "COMP/USDT", "YFI/USDT", "SNX/USDT", "ZEC/USDT", "XMR/USDT", "DASH/USDT",
  "WAVES/USDT", "QTUM/USDT", "OMG/USDT", "ONT/USDT", "NEO/USDT", "IOTA/USDT", "ICX/USDT", "LSK/USDT",
  "NANO/USDT", "DGB/USDT", "SC/USDT", "BTT/USDT", "WIN/USDT", "DENT/USDT",
  "STMX/USDT", "COTI/USDT", "CTSI/USDT", "HIVE/USDT",
  "XDC/USDT", "FLR/USDT", "CORE/USDT", "KAS/USDT", "ALICE/USDT", "ILV/USDT", "TLM/USDT",
  "YGG/USDT", "SUPER/USDT", "AUDIO/USDT", "PERP/USDT", "RUNE/USDT", "BAL/USDT", "NMR/USDT",
  "UMA/USDT", "BAND/USDT", "KNC/USDT", "STORJ/USDT",
  "POWR/USDT", "REQ/USDT",
  "AGLD/USDT", "FXS/USDT", "CELO/USDT", "AR/USDT", "ZEN/USDT", "GLMR/USDT", "ASTR/USDT",
  "SKL/USDT", "NKN/USDT", "BEAM/USDT", "TRB/USDT",
  "CAKE/USDT", "XVS/USDT",
  "UNFI/USDT", "CELR/USDT",
  "HOOK/USDT", "MAGIC/USDT", "HFT/USDT", "STG/USDT", "SSV/USDT",
  "LQTY/USDT", "JOE/USDT", "RDNT/USDT", "ID/USDT", "EDU/USDT", "ARKM/USDT", "WLD/USDT", "CYBER/USDT",
  "NTRN/USDT", "BLUR/USDT", "AXL/USDT", "SAGA/USDT", "IO/USDT", "ZRO/USDT", "ZK/USDT",
  "RENDER/USDT", "TON/USDT", "DOGS/USDT", "GTC/USDT",
  "RLC/USDT", "VTHO/USDT", "WAN/USDT",
  "ZRX/USDT", "CLV/USDT",
  "KSM/USDT", "PHA/USDT",
  "CFG/USDT", "REGEN/USDT", "KUJI/USDT", "EVMOS/USDT",
  "CRO/USDT", "MNT/USDT",
  "HYPE/USDT", "ME/USDT", "BERA/USDT", "EIGEN/USDT", "POPCAT/USDT", "MOG/USDT", "BRETT/USDT", "NEIRO/USDT",
  "COW/USDT", "SCR/USDT", "DRIFT/USDT",
  "GOAT/USDT", "PENGU/USDT", "VIRTUAL/USDT", "AIXBT/USDT",
  "TRUMP/USDT", "MELANIA/USDT", "GIGA/USDT", "PNUT/USDT", "ACT/USDT",
  "BIO/USDT", "FARTCOIN/USDT", "GRASS/USDT",
]

# ── parameters ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS     = 730
TIMEFRAME         = "1m"
RELVOL_WINDOW     = 20
HOLD_BARS         = 5
MIN_BARS          = 30
PSI_ENTRY_LONG    = 0.55
PSI_ENTRY_SHORT   = 0.45
AMIHUD_WINDOW     = 10
DELTA_WINDOW      = 3
SYMBOL_LIMIT      = 10000
REQUEST_DELAY     = 0.35

EXCHANGE_PRIORITY = ["okx", "kraken", "binanceus"]


# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_delta_sign(high, low, close, volume):
    """
    CKS flow imbalance direction only.
    buy_vol  = volume × (close - low) / (high - low + ε)
    sell_vol = volume - buy_vol
    sign(Δ)  = sign(buy_vol - sell_vol) → +1 aggressors buying, -1 selling
    """
    hl_range = (high - low).replace(0, np.nan)
    buy_vol  = volume * (close - low) / (hl_range + 1e-12)
    sell_vol = volume - buy_vol
    delta    = buy_vol - sell_vol
    return np.sign(delta).replace(0, np.nan).ffill().fillna(1)

def compute_amihud(close, volume, window=AMIHUD_WINDOW):
    ret       = close.pct_change().abs()
    illiq_raw = ret / (volume + 1)
    illiq     = illiq_raw.rolling(window).mean()
    return illiq

def compute_illiq_dot(illiq, window=DELTA_WINDOW):
    return illiq.diff(window)

def compute_psi(high, low, close, volume):
    """
    Ψ̂(t) = sigmoid( sign(Δ) × NLI_norm )

    sign(Δ)   : CKS flow direction — +1 if buy pressure dominates, -1 if sell
    NLI_norm  : -sign(ILLIQ̇) × log(|ILLIQ̇| + ε), z-scored over rolling 60 bars
                magnitude of liquidity regime shift, signed so deepening → positive

    Product interpretation:
      +1 × positive NLI → buyers aggressing into deepening book → Ψ̂ > 0.5 (bullish)
      +1 × negative NLI → buyers aggressing into thinning book  → Ψ̂ < 0.5 (suppressed)
      -1 × positive NLI → sellers aggressing, book deepening    → Ψ̂ < 0.5 (bearish)
      -1 × negative NLI → sellers aggressing into thinning book → Ψ̂ > 0.5 (short signal)
    """
    delta_sign = compute_delta_sign(high, low, close, volume)
    illiq      = compute_amihud(close, volume)
    illiq_dot  = compute_illiq_dot(illiq)

    eps = 1e-12

    neg_log_illiq_dot = -np.sign(illiq_dot) * np.log(illiq_dot.abs() + eps)
    nli_std           = neg_log_illiq_dot.rolling(60).std().replace(0, np.nan)
    nli_norm          = neg_log_illiq_dot / (nli_std + eps)

    raw_signal = delta_sign * nli_norm
    psi        = raw_signal.apply(sigmoid)
    return psi, illiq, illiq_dot

def compute_relvol(close, window=RELVOL_WINDOW):
    ret         = close.pct_change()
    rolling_std = ret.rolling(window).std()
    relvol      = ret.abs() / (rolling_std + 1e-12)
    return relvol


# ── exchange setup ─────────────────────────────────────────────────────────────

def make_exchange(exchange_id):
    log(f"  Connecting to {exchange_id} ...")
    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    markets = exchange.load_markets()
    log(f"  {exchange_id}: {len(markets)} markets loaded")

    test_sym = "BTC/USDT"
    if test_sym not in markets:
        raise RuntimeError(f"{exchange_id}: {test_sym} not in market list")

    log(f"  Running connectivity test ({test_sym}, 10 bars) ...")
    test_ohlcv = exchange.fetch_ohlcv(test_sym, timeframe=TIMEFRAME, limit=10)
    if not test_ohlcv or len(test_ohlcv) == 0:
        raise RuntimeError(f"{exchange_id}: fetch_ohlcv returned empty. Likely geo-blocked.")

    log(f"  ✓ {exchange_id} OK — {len(test_ohlcv)} test bars received")
    return exchange, set(markets.keys())


def connect_exchange():
    for exchange_id in EXCHANGE_PRIORITY:
        try:
            exchange, available = make_exchange(exchange_id)
            return exchange, available, exchange_id
        except Exception as e:
            log(f"  ✗ {exchange_id} failed: {e}")

    log("\n  FATAL: All exchanges failed. Check your network or VPN.")
    log("  Exchanges tried: " + ", ".join(EXCHANGE_PRIORITY))
    sys.exit(1)


# ── data fetch ────────────────────────────────────────────────────────────────

def fetch_symbol(exchange, symbol, since_ms, limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=limit)
        if not ohlcv or len(ohlcv) < 60:
            time.sleep(REQUEST_DELAY)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
        if not ohlcv or len(ohlcv) < 60:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None


# ── per-symbol backtest ───────────────────────────────────────────────────────

def backtest_symbol(symbol, df):
    high   = df["High"]
    low    = df["Low"]
    close  = df["Close"]
    volume = df["Volume"]

    psi, illiq, illiq_dot = compute_psi(high, low, close, volume)
    relvol = compute_relvol(close)

    aligned = pd.DataFrame({
        "close":  close,
        "psi":    psi,
        "relvol": relvol,
        "illiq":  illiq,
    }).dropna()

    if len(aligned) < MIN_BARS:
        return None, 0

    idx    = aligned.index.tolist()
    trades = []

    for dt in aligned.index:
        pos = idx.index(dt)
        if pos + HOLD_BARS >= len(idx):
            continue

        psi_val = aligned.loc[dt, "psi"]
        if psi_val <= PSI_ENTRY_LONG:
            continue

        exit_pos = pos + HOLD_BARS
        for j in range(pos + 1, min(pos + HOLD_BARS + 1, len(idx))):
            if aligned.loc[idx[j], "psi"] < PSI_ENTRY_SHORT:
                exit_pos = j
                break

        entry_price = aligned.loc[idx[pos],      "close"]
        exit_price  = aligned.loc[idx[exit_pos], "close"]
        bars_held   = exit_pos - pos
        exhausted   = bars_held < HOLD_BARS
        trade_ret   = (exit_price - entry_price) / entry_price

        trades.append({
            "symbol":    symbol,
            "date":      dt,
            "direction": "LONG",
            "psi":       round(float(psi_val), 4),
            "entry":     round(float(entry_price), 6),
            "exit":      round(float(exit_price), 6),
            "bars_held": bars_held,
            "exhausted": exhausted,
            "ret_pct":   round(float(trade_ret * 100), 4),
            "relvol":    round(float(aligned.loc[dt, "relvol"]), 4),
        })

    return trades, len(aligned)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(start_dt.timestamp() * 1000)

    log("=" * 65)
    log("  Ψ̂(t) Backtest [CRYPTO / ccxt]  —  v5: sign(Δ) × NLI")
    log("=" * 65)
    log(f"  Signal  : sigmoid( sign(Δ_CKS) × NLI_norm )")
    log(f"  Delta   : sign(buy_vol - sell_vol)  [direction only]")
    log(f"  NLI     : -sign(ILLIQ̇) × log(|ILLIQ̇|+ε) / rolling_std(60)  [magnitude]")
    log(f"  Period  : {start_dt.date()} → {end_dt.date()}")
    log(f"  Hold    : {HOLD_BARS} bars | exit on Ψ̂ < {PSI_ENTRY_SHORT}")
    log(f"  Long if : Ψ̂ > {PSI_ENTRY_LONG}")
    log("=" * 65)

    log("\n  Selecting exchange ...")
    exchange, available_symbols, exchange_id = connect_exchange()
    log(f"\n  Active exchange : {exchange_id}")
    log("=" * 65)

    symbols_to_test = CANDIDATE_SYMBOLS[:SYMBOL_LIMIT]
    all_trades   = []
    ticker_stats = []
    passed = failed = skipped_market = 0

    for i, symbol in enumerate(symbols_to_test):
        log(f"[{i+1:>3}/{len(symbols_to_test)}] {symbol:<14}")

        if symbol not in available_symbols:
            log(f"  → SKIP (not listed on {exchange_id})")
            skipped_market += 1
            failed += 1
            continue

        time.sleep(REQUEST_DELAY)
        df = fetch_symbol(exchange, symbol, since_ms)

        if df is None:
            log(f"  → SKIP (insufficient data)")
            failed += 1
            continue

        trades, n_bars = backtest_symbol(symbol, df)

        if trades is None:
            log(f"  → SKIP ({n_bars} bars < {MIN_BARS})")
            failed += 1
            continue

        n_trades = len(trades)
        if n_trades == 0:
            log(f"  → SKIP (0 trades generated)")
            failed += 1
            continue

        rets  = [t["ret_pct"] for t in trades]
        wins  = sum(1 for r in rets if r > 0)
        avg_r = np.mean(rets)
        wr    = wins / n_trades * 100

        log(f"  → {n_bars:>4} bars | {n_trades:>4} trades | "
            f"WR {wr:>5.1f}% | AvgRet {avg_r:>+6.3f}%")

        all_trades.extend(trades)
        ticker_stats.append({
            "symbol":   symbol,
            "n_bars":   n_bars,
            "n_trades": n_trades,
            "win_rate": round(wr, 2),
            "avg_ret":  round(avg_r, 4),
        })
        passed += 1

    # ── aggregate ──────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  AGGREGATE RESULTS  [v5 — sign(Δ) × NLI]")
    log("=" * 65)

    if not all_trades:
        log("  No trades generated.")
        log(f"  ({skipped_market} symbols not listed on {exchange_id})")
        return

    trade_df   = pd.DataFrame(all_trades)
    rets       = trade_df["ret_pct"]
    wins       = (rets > 0).sum()
    losses     = (rets <= 0).sum()
    n          = len(trade_df)

    win_rate   = wins / n * 100
    avg_ret    = rets.mean()
    med_ret    = rets.median()
    avg_win    = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss   = rets[rets <= 0].mean() if losses > 0 else 0
    profit_fac = abs(avg_win * wins) / abs(avg_loss * losses + 1e-12)
    sharpe     = rets.mean() / (rets.std() + 1e-12) * np.sqrt(365 / HOLD_BARS)
    max_dd     = (rets.cumsum() - rets.cumsum().cummax()).min()

    log(f"  Exchange          : {exchange_id}")
    log(f"  Symbols attempted : {len(symbols_to_test)}")
    log(f"  Not listed        : {skipped_market}")
    log(f"  Symbols passed    : {passed}")
    log(f"  Total trades      : {n}")
    log(f"")
    log(f"  ── Performance ──────────────────────────────────────")
    log(f"  Win Rate          : {win_rate:.2f}%")
    log(f"  Avg Return/Trade  : {avg_ret:+.4f}%")
    log(f"  Median Return     : {med_ret:+.4f}%")
    log(f"  Avg Win           : {avg_win:+.4f}%")
    log(f"  Avg Loss          : {avg_loss:+.4f}%")
    log(f"  Profit Factor     : {profit_fac:.3f}")
    log(f"  Sharpe (annlzd)   : {sharpe:.3f}")
    log(f"  Max Drawdown      : {max_dd:.4f}%  (on cumulative ret series)")
    log(f"")
    log(f"  ── Distribution ─────────────────────────────────────")
    log(f"  Ret Std Dev       : {rets.std():.4f}%")
    log(f"  Ret Skew          : {rets.skew():.4f}")
    log(f"  Ret Kurt          : {rets.kurt():.4f}")
    log(f"  Best Trade        : {rets.max():+.4f}%")
    log(f"  Worst Trade       : {rets.min():+.4f}%")

    log(f"\n  ── Ψ̂ Quantile Breakdown ─────────────────────────────")
    trade_df["psi_bin"] = pd.cut(
        trade_df["psi"],
        bins=[0, .45, .5, .55, .6, .7, .8, 1.0],
        labels=["<.45",".45-.50",".50-.55",".55-.60",".60-.70",".70-.80",">.80"]
    )
    for label, grp in trade_df.groupby("psi_bin", observed=True):
        wr_g = (grp["ret_pct"] > 0).mean() * 100
        ar_g = grp["ret_pct"].mean()
        log(f"    Ψ̂ {str(label):<10} : {len(grp):>5} trades | WR {wr_g:>5.1f}% | AvgRet {ar_g:>+6.3f}%")

    log(f"\n  ── Exit Type Breakdown ───────────────────────────────")
    ex  = trade_df[trade_df["exhausted"] == True]
    tim = trade_df[trade_df["exhausted"] == False]
    for label, grp in [("Signal reversal", ex), ("Time stop (5-bar)", tim)]:
        if len(grp) == 0: continue
        wr_g = (grp["ret_pct"] > 0).mean() * 100
        ar_g = grp["ret_pct"].mean()
        log(f"    {label:<20} : {len(grp):>5} trades | WR {wr_g:>5.1f}% | AvgRet {ar_g:>+6.3f}%")

    log(f"\n  ── RelVol Episode Breakdown ──────────────────────────")
    trade_df["rv_bin"] = pd.cut(
        trade_df["relvol"],
        bins=[1, 1.5, 2.0, 3.0, 5.0, 999],
        labels=["1-1.5x","1.5-2x","2-3x","3-5x",">5x"]
    )
    for label, grp in trade_df.groupby("rv_bin", observed=True):
        wr_g = (grp["ret_pct"] > 0).mean() * 100
        ar_g = grp["ret_pct"].mean()
        log(f"    RelVol {str(label):<8} : {len(grp):>5} trades | WR {wr_g:>5.1f}% | AvgRet {ar_g:>+6.3f}%")

    log(f"\n  ── Top 10 Symbols by Trade Count ────────────────────")
    stat_df = pd.DataFrame(ticker_stats).sort_values("n_trades", ascending=False)
    for _, row in stat_df.head(10).iterrows():
        log(f"    {row['symbol']:<14} : {row['n_trades']:>4} trades | "
            f"WR {row['win_rate']:>5.1f}% | AvgRet {row['avg_ret']:>+6.3f}%")

    trade_df.to_csv("psi_crypto_trades_v5.csv", index=False)
    stat_df.to_csv("psi_crypto_stats_v5.csv", index=False)
    log(f"\n  Trades saved → psi_crypto_trades_v5.csv")
    log(f"  Stats saved  → psi_crypto_stats_v5.csv")
    log("=" * 65)


if __name__ == "__main__":
    main()