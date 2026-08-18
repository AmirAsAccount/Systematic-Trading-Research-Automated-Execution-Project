"""VWAP Execution Drip Backtest  [CRYPTO / ccxt]  —  v5

Architecture: predict institutional VWAP execution drip toward technical anchors

  WHAT WE ARE PREDICTING
  Institutional VWAP algos leak predictable directional flow across time windows.
  They execute heaviest near known reference points (prior day high/low, weekly open,
  round numbers, ATR-based levels) because that is where their benchmark risk is lowest.
  Price drifts persistently on one side of VWAP during active execution windows.
  We ride that drift from deviation back toward the anchor.

  NO liquidity gate. NO acceleration. NO second derivative. NO sigmoid on flow.

  PROCESS 1 — VWAP DEVIATION GATE
    Compute session VWAP (anchored at daily open, resets each UTC day)
    Measure deviation: how far is price from VWAP as % of ATR
    Gate fires when deviation is large enough that a VWAP algo
    has meaningful benchmark risk and MUST continue executing

  PROCESS 2 — INSTITUTIONAL LEVEL PROXIMITY
    Levels computed fresh each bar from prior day OHLC + weekly open + round numbers + ATR bands
    Proximity: is price within ATR-scaled distance of a level in the direction of the trade?
    A level in the direction of VWAP reversion = the execution target
    Gate fires when a valid target level exists between current price and VWAP

  PROCESS 3 — DELTA CONFIRMATION  Δ(t)
    CKS raw buy/sell flow imbalance, EMA-smoothed
    Confirms execution flow is actually present and one-sided
    Positive Δ = buy-side aggressor dominant (VWAP algo still dripping)
    Gate fires when Δ > 0 in the direction of the expected reversion

  ENTRY  : all three conditions met simultaneously
  RIDE   : Δ(t) stays positive (algo still executing)
  EXIT   : price reaches target level OR Δ reverts for DECEL_CONFIRM bars OR time-stop
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
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
    "LINK/USDT", "UNI/USDT", "TRX/USDT", "ATOM/USDT", "BCH/USDT",
    "XLM/USDT", "ICP/USDT", "FIL/USDT", "HBAR/USDT", "ETC/USDT",
    "NEAR/USDT", "VET/USDT", "APT/USDT", "OP/USDT", "ARB/USDT",
    "GRT/USDT", "RNDR/USDT", "THETA/USDT", "FTM/USDT", "MKR/USDT",
    "LDO/USDT", "TIA/USDT", "SUI/USDT", "INJ/USDT", "STX/USDT",
    "SEI/USDT", "IMX/USDT", "EGLD/USDT", "AAVE/USDT", "FLOW/USDT",
    "QNT/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT", "CHZ/USDT",
    "ALGO/USDT", "MINA/USDT", "AXS/USDT", "APE/USDT", "PEPE/USDT",
    "SHIB/USDT", "DOGE/USDT", "BONK/USDT", "FLOKI/USDT", "WIF/USDT",
    "MEME/USDT", "BOME/USDT", "JUP/USDT", "PYTH/USDT", "ONDO/USDT",
    "PENDLE/USDT", "FET/USDT", "AGIX/USDT", "OCEAN/USDT", "AKT/USDT",
    "W/USDT", "ENA/USDT", "STRK/USDT", "DYM/USDT", "MANTA/USDT",
    "ALT/USDT", "XAI/USDT", "NFP/USDT", "ACE/USDT", "JTO/USDT",
    "ORDI/USDT", "SATS/USDT", "NOT/USDT", "BB/USDT", "REZ/USDT",
    "ETHFI/USDT", "AEVO/USDT", "METIS/USDT", "RON/USDT", "PIXEL/USDT",
    "PORTAL/USDT", "ZETA/USDT", "OMNI/USDT", "TAO/USDT", "TNSR/USDT",
    "CRV/USDT", "SUSHI/USDT", "1INCH/USDT", "DYDX/USDT", "ENS/USDT",
    "WOO/USDT", "GMT/USDT", "JASMY/USDT", "CKB/USDT", "ZIL/USDT",
    "ENJ/USDT", "HOT/USDT", "ONE/USDT", "ANKR/USDT", "RVN/USDT",
    "BAT/USDT", "LRC/USDT", "KAVA/USDT", "COMP/USDT", "YFI/USDT",
    "SNX/USDT", "ZEC/USDT", "XMR/USDT", "DASH/USDT", "WAVES/USDT",
    "QTUM/USDT", "OMG/USDT", "ONT/USDT", "NEO/USDT", "IOTA/USDT",
    "ICX/USDT", "LSK/USDT", "NANO/USDT", "DGB/USDT", "SC/USDT",
    "BTT/USDT", "WIN/USDT", "DENT/USDT", "STMX/USDT", "COTI/USDT",
    "CTSI/USDT", "HIVE/USDT", "XDC/USDT", "FLR/USDT", "CORE/USDT",
    "KAS/USDT", "ALICE/USDT", "ILV/USDT", "TLM/USDT", "YGG/USDT",
    "SUPER/USDT", "AUDIO/USDT", "PERP/USDT", "RUNE/USDT", "BAL/USDT",
    "NMR/USDT", "UMA/USDT", "BAND/USDT", "KNC/USDT", "STORJ/USDT",
    "POWR/USDT", "REQ/USDT", "AGLD/USDT", "FXS/USDT", "CELO/USDT",
    "AR/USDT", "ZEN/USDT", "GLMR/USDT", "ASTR/USDT", "SKL/USDT",
    "NKN/USDT", "BEAM/USDT", "TRB/USDT", "CAKE/USDT", "XVS/USDT",
    "UNFI/USDT", "CELR/USDT", "HOOK/USDT", "MAGIC/USDT", "HFT/USDT",
    "STG/USDT", "SSV/USDT", "LQTY/USDT", "JOE/USDT", "RDNT/USDT",
    "ID/USDT", "EDU/USDT", "ARKM/USDT", "WLD/USDT", "CYBER/USDT",
    "NTRN/USDT", "BLUR/USDT", "AXL/USDT", "SAGA/USDT", "IO/USDT",
    "ZRO/USDT", "ZK/USDT", "RENDER/USDT", "TON/USDT", "DOGS/USDT",
    "GTC/USDT", "RLC/USDT", "VTHO/USDT", "WAN/USDT", "ZRX/USDT",
    "CLV/USDT", "KSM/USDT", "PHA/USDT", "CRO/USDT", "MNT/USDT",
    "HYPE/USDT", "ME/USDT", "BERA/USDT", "EIGEN/USDT", "POPCAT/USDT",
    "MOG/USDT", "BRETT/USDT", "NEIRO/USDT", "COW/USDT", "SCR/USDT",
    "DRIFT/USDT", "GOAT/USDT", "PENGU/USDT", "VIRTUAL/USDT", "AIXBT/USDT",
    "TRUMP/USDT", "MELANIA/USDT", "GIGA/USDT", "PNUT/USDT", "ACT/USDT",
    "BIO/USDT", "FARTCOIN/USDT", "GRASS/USDT",
]

# ── parameters ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS       = 14        # need prior day levels — 14d gives stable history

TIMEFRAME           = "1m"
HOLD_BARS           = 60        # safety time-stop only

MIN_BARS            = 500       # need full prior day + current session

# Process 1 — VWAP deviation gate
VWAP_DEV_ATR_MULT   = 0.5      # min deviation from VWAP in ATR units to fire gate
ATR_WINDOW          = 14        # ATR period (1m bars)

# Process 2 — institutional level proximity
LEVEL_ATR_BAND      = 2.0      # levels within this many ATRs of current price are "active"
ROUND_PCT_STEPS     = [0.005, 0.01]  # 0.5% and 1% round number grid

# Process 3 — delta confirmation
DELTA_EMA_SPAN      = 5        # EMA smoothing on raw CKS delta
DECEL_CONFIRM       = 2        # consecutive bars of falling Δ to trigger exit

# Infra
SYMBOL_LIMIT        = 10000
REQUEST_DELAY       = 0.4
BARS_PER_FETCH      = 1000
EXCHANGE_PRIORITY   = ["okx", "kraken", "binanceus"]

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)

# ── Process 1: Session VWAP (anchored, resets at UTC midnight) ────────────────
def compute_session_vwap(df):
    """
    Anchored VWAP: resets at the start of each UTC calendar day.
    This is the standard institutional intraday VWAP reference.
    Returns VWAP series and ATR series.
    """
    df = df.copy()
    df["date"]     = df.index.date
    df["tp"]       = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_vol"]   = df["tp"] * df["Volume"]

    # anchored cumulative sum — resets each day
    df["cum_tpv"]  = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]  = df.groupby("date")["Volume"].cumsum()
    df["vwap"]     = df["cum_tpv"] / (df["cum_vol"] + 1e-12)

    # ATR
    hl   = df["High"] - df["Low"]
    hc   = (df["High"] - df["Close"].shift()).abs()
    lc   = (df["Low"]  - df["Close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_WINDOW).mean()

    # VWAP deviation in ATR units — signed (positive = price above VWAP)
    df["vwap_dev"] = (df["Close"] - df["vwap"]) / (df["atr"] + 1e-12)

    return df["vwap"], df["atr"], df["vwap_dev"]

# ── Process 2: Institutional levels ──────────────────────────────────────────
def compute_institutional_levels(df):
    """
    For each bar, compute the set of active institutional reference levels:
      - Prior day high (PDH)
      - Prior day low  (PDL)
      - Prior day close (PDC)
      - Prior day VWAP  (PDV) — rolling last-day VWAP value
      - Weekly open     (WO)  — Monday UTC open of current week
      - ATR bands above/below current price (1x and 2x ATR)
      - Round number grid at 0.5% and 1% intervals

    Returns a DataFrame where each row has the nearest level ABOVE and BELOW price,
    and which type it is — used by the entry gate to find the execution target.
    """
    df = df.copy()
    df["date"]     = df.index.date
    df["weekday"]  = df.index.dayofweek   # 0=Monday

    # ── prior day OHLC ────────────────────────────────────────────────────────
    daily = df.groupby("date").agg(
        day_high  = ("High",  "max"),
        day_low   = ("Low",   "min"),
        day_close = ("Close", "last"),
        day_vwap  = ("vwap",  "last"),   # last VWAP value of the day = PDV
    ).shift(1)                            # shift(1) = prior day

    df = df.join(daily, on="date")

    # ── weekly open ───────────────────────────────────────────────────────────
    # first bar of the week (Monday) open — carry forward through the week
    df["is_week_open"] = df["weekday"] == 0
    week_opens = df[df["is_week_open"]].groupby(
        df[df["is_week_open"]].index.to_period("W")
    )["Open"].first()

    # map each bar to its week period and fill weekly open
    df["week"] = df.index.to_period("W")
    df["weekly_open"] = df["week"].map(week_opens)
    df["weekly_open"] = df["weekly_open"].ffill()

    return df

def get_nearest_level(price, atr, row, direction):
    """
    Given current price, ATR, and the row of institutional levels,
    find the nearest institutional level in `direction` (+1 = above, -1 = below)
    within LEVEL_ATR_BAND ATRs of price.

    Returns (level_price, level_type) or (None, None) if none found.
    """
    band      = LEVEL_ATR_BAND * atr
    candidates = []

    # named levels
    named = {
        "PDH":         row.get("day_high"),
        "PDL":         row.get("day_low"),
        "PDC":         row.get("day_close"),
        "PDV":         row.get("day_vwap"),
        "WeeklyOpen":  row.get("weekly_open"),
    }
    for name, lvl in named.items():
        if lvl is None or np.isnan(lvl):
            continue
        if direction == 1 and price < lvl <= price + band:
            candidates.append((lvl, name))
        elif direction == -1 and price - band <= lvl < price:
            candidates.append((lvl, name))

    # ATR bands (1x and 2x)
    for mult in [1.0, 2.0]:
        lvl_up   = price + mult * atr
        lvl_down = price - mult * atr
        if direction == 1 and lvl_up <= price + band:
            candidates.append((lvl_up, f"ATR+{mult}x"))
        if direction == -1 and lvl_down >= price - band:
            candidates.append((lvl_down, f"ATR-{mult}x"))

    # round number grid
    for step in ROUND_PCT_STEPS:
        grid_up   = price * (1 + step - (price % (price * step)) / price)
        grid_down = price - (price % (price * step))
        step_label = f"{int(step*100)}pct"
        if direction == 1 and price < grid_up <= price + band:
            candidates.append((grid_up, f"Round_{step_label}"))
        if direction == -1 and price - band <= grid_down < price:
            candidates.append((grid_down, f"Round_{step_label}"))

    if not candidates:
        return None, None

    # nearest level in the direction
    if direction == 1:
        candidates.sort(key=lambda x: x[0])
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0]

# ── Process 3: CKS raw flow delta ─────────────────────────────────────────────
def compute_delta(high, low, close, volume):
    """
    Raw CKS buy/sell imbalance, EMA-smoothed.
    No second derivative. No sigmoid. No normalization beyond EMA.
    Positive = buy-side aggressor dominant.
    """
    hl      = (high - low).replace(0, np.nan)
    buy_vol = volume * (close - low) / (hl + 1e-12)
    delta   = buy_vol - (volume - buy_vol)
    return delta.ewm(span=DELTA_EMA_SPAN, adjust=False).mean()

# ── exchange setup ─────────────────────────────────────────────────────────────
def make_exchange(exchange_id):
    log(f"  Connecting to {exchange_id} ...")
    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    markets = exchange.load_markets()
    log(f"  {exchange_id}: {len(markets)} markets loaded")
    test = exchange.fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, limit=5)
    if not test:
        raise RuntimeError("fetch_ohlcv empty")
    log(f"  ✓ {exchange_id} OK")
    return exchange, set(markets.keys())

def connect_exchange():
    for eid in EXCHANGE_PRIORITY:
        try:
            return (*make_exchange(eid), eid)
        except Exception as e:
            log(f"  ✗ {eid}: {e}")
    log("FATAL: all exchanges failed")
    sys.exit(1)

# ── data fetch ────────────────────────────────────────────────────────────────
def fetch_symbol_1m(exchange, symbol, since_ms):
    all_rows = {}
    cursor   = since_ms
    max_ts   = int(datetime.utcnow().timestamp() * 1000)
    stall    = 0
    while cursor < max_ts:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=TIMEFRAME,
                since=cursor, limit=BARS_PER_FETCH
            )
        except Exception as e:
            log(f"    fetch error: {e}")
            break
        if not batch:
            break
        new = 0
        for row in batch:
            if row[0] not in all_rows:
                all_rows[row[0]] = row
                new += 1
        last_ts = batch[-1][0]
        if new == 0 or last_ts <= cursor:
            stall += 1
            if stall >= 2:
                break
        else:
            stall = 0
        cursor = last_ts + 60_000
        time.sleep(REQUEST_DELAY)

    if len(all_rows) < MIN_BARS:
        return None
    rows = sorted(all_rows.values(), key=lambda r: r[0])
    df   = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["Open","High","Low","Close","Volume"]].astype(float)
    df.dropna(inplace=True)
    return df

# ── per-symbol backtest ───────────────────────────────────────────────────────
def backtest_symbol(symbol, df):
    # compute all three series
    vwap, atr, vwap_dev = compute_session_vwap(df)
    df["vwap"]     = vwap
    df["atr"]      = atr
    df["vwap_dev"] = vwap_dev

    df = compute_institutional_levels(df)
    df["delta"] = compute_delta(df["High"], df["Low"], df["Close"], df["Volume"])

    # drop warmup bars (need prior day data + ATR window)
    df.dropna(subset=["vwap", "atr", "delta", "day_high", "day_low",
                       "day_close", "weekly_open"], inplace=True)

    if len(df) < MIN_BARS // 2:
        return None, 0

    close_arr   = df["Close"].values
    vwap_dev_arr= df["vwap_dev"].values
    atr_arr     = df["atr"].values
    delta_arr   = df["delta"].values
    n           = len(df)
    df_vals     = df.to_dict("records")   # for level lookup per bar

    trades = []

    for pos in range(n - HOLD_BARS - 1):
        price     = close_arr[pos]
        dev       = vwap_dev_arr[pos]
        atr_val   = atr_arr[pos]
        delta_val = delta_arr[pos]
        row       = df_vals[pos]

        # ── PROCESS 1: VWAP deviation gate ───────────────────────────────────
        # price must be meaningfully extended from VWAP
        # and delta must point back TOWARD VWAP (reversion trade)
        # long setup: price below VWAP (dev < 0), delta positive (buying into VWAP)
        # short setup: price above VWAP (dev > 0), delta negative (selling into VWAP)
        # here we only model the long side (price below VWAP, buy-side delta)

        if dev >= -VWAP_DEV_ATR_MULT:
            continue   # price not far enough below VWAP

        # ── PROCESS 3: delta must be positive (buy-side dominant) ─────────────
        if delta_val <= 0:
            continue

        # ── PROCESS 2: institutional level must exist above price as target ───
        target_price, level_type = get_nearest_level(
            price, atr_val, row, direction=1   # level above = VWAP reversion target
        )
        if target_price is None:
            continue

        # target must be between price and VWAP (or at/above VWAP)
        vwap_val = row["vwap"]
        if target_price > vwap_val * 1.005:    # target too far above VWAP
            continue

        # ── ENTRY confirmed — all three processes agree ───────────────────────
        exit_pos    = pos + HOLD_BARS
        exit_type   = "time_stop"
        consec_down = 0

        for j in range(pos + 1, min(pos + HOLD_BARS + 1, n)):
            # exit 1: price reached target level
            if close_arr[j] >= target_price:
                exit_pos  = j
                exit_type = "target_hit"
                break

            # exit 2: price reached VWAP (full reversion)
            if close_arr[j] >= df_vals[j]["vwap"]:
                exit_pos  = j
                exit_type = "vwap_hit"
                break

            # exit 3: delta reverts (flow exhausted)
            if delta_arr[j] <= 0:
                exit_pos  = j
                exit_type = "delta_zero"
                break

            # exit 4: delta decelerating for DECEL_CONFIRM bars
            if delta_arr[j] < delta_arr[j - 1]:
                consec_down += 1
                if consec_down >= DECEL_CONFIRM:
                    exit_pos  = j
                    exit_type = "decel_exit"
                    break
            else:
                consec_down = 0

        entry = close_arr[pos]
        ex    = close_arr[exit_pos]
        ret   = (ex - entry) / entry * 100

        trades.append({
            "symbol":       symbol,
            "date":         df.index[pos],
            "entry":        round(float(entry), 6),
            "target":       round(float(target_price), 6),
            "level_type":   level_type,
            "vwap_dev":     round(float(dev), 4),
            "delta_in":     round(float(delta_val), 4),
            "bars_held":    exit_pos - pos,
            "exit_type":    exit_type,
            "ret_pct":      round(float(ret), 6),
        })

    return trades, len(df)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(start_dt.timestamp() * 1000)

    log("=" * 65)
    log("  VWAP Execution Drip Backtest  —  v5 / 1m bars")
    log("=" * 65)
    log(f"  WHAT    : Institutional VWAP algo execution drip")
    log(f"  PROCESS 1 : VWAP deviation gate  (>{VWAP_DEV_ATR_MULT} ATR below VWAP)")
    log(f"  PROCESS 2 : Institutional level  (PDH/PDL/PDC/PDV/WeeklyOpen/ATR/Round)")
    log(f"  PROCESS 3 : Δ(t) CKS flow        (EMA-{DELTA_EMA_SPAN}, must be positive)")
    log(f"  EXIT      : target hit | VWAP hit | Δ=0 | decel {DECEL_CONFIRM}b | stop {HOLD_BARS}b")
    log(f"  Period    : {start_dt.date()} → {end_dt.date()}  ({LOOKBACK_DAYS}d)")
    log(f"  Min bars  : {MIN_BARS}")
    log("=" * 65)

    exchange, available, exchange_id = connect_exchange()
    log(f"\n  Active exchange : {exchange_id}\n")

    symbols      = [s for s in CANDIDATE_SYMBOLS[:SYMBOL_LIMIT] if s in available]
    all_trades   = []
    ticker_stats = []
    passed = failed = 0

    for i, symbol in enumerate(symbols):
        log(f"[{i+1:>3}/{len(symbols)}] {symbol:<14}")
        time.sleep(REQUEST_DELAY)

        df = fetch_symbol_1m(exchange, symbol, since_ms)
        if df is None:
            log(f"  → SKIP (< {MIN_BARS} bars)")
            failed += 1
            continue

        trades, n_bars = backtest_symbol(symbol, df)
        if not trades:
            log(f"  → {n_bars} bars | 0 trades")
            failed += 1
            continue

        rets = [t["ret_pct"] for t in trades]
        wr   = sum(1 for r in rets if r > 0) / len(trades) * 100
        ar   = np.mean(rets)
        log(f"  → {n_bars:>6} bars | {len(trades):>5} trades | "
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

    # ── aggregate ──────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  AGGREGATE RESULTS")
    log("=" * 65)

    if not all_trades:
        log("  No trades generated.")
        return

    tdf  = pd.DataFrame(all_trades)
    rets = tdf["ret_pct"]
    n    = len(tdf)

    wins = (rets > 0).sum()
    loss = (rets <= 0).sum()
    aw   = rets[rets > 0].mean() if wins > 0 else 0
    al   = rets[rets <= 0].mean() if loss > 0 else 0
    pf   = abs(aw * wins) / abs(al * loss + 1e-12)

    avg_hold = tdf["bars_held"].mean()
    sh       = rets.mean() / (rets.std() + 1e-12) * np.sqrt(525_600 / max(avg_hold, 1))
    dd       = (rets.cumsum() - rets.cumsum().cummax()).min()

    log(f"  Symbols passed    : {passed}")
    log(f"  Total trades      : {n}")
    log(f"  Win Rate          : {wins/n*100:.2f}%")
    log(f"  Avg Return/Trade  : {rets.mean():+.6f}%")
    log(f"  Median Return     : {rets.median():+.6f}%")
    log(f"  Avg Win           : {aw:+.6f}%")
    log(f"  Avg Loss          : {al:+.6f}%")
    log(f"  Profit Factor     : {pf:.4f}")
    log(f"  Sharpe (annlzd)   : {sh:.4f}")
    log(f"  Max Drawdown      : {dd:.6f}%")
    log(f"  Skew / Kurt       : {rets.skew():.3f} / {rets.kurt():.3f}")
    log(f"  Best / Worst      : {rets.max():+.6f}% / {rets.min():+.6f}%")
    log(f"  Avg Bars Held     : {avg_hold:.1f}")

    # ── VWAP deviation at entry ────────────────────────────────────────────────
    log(f"\n  ── VWAP Deviation at Entry (ATR units) ───────────────")
    tdf["dev_bin"] = pd.cut(tdf["vwap_dev"],
        bins=[-10, -3, -2, -1.5, -1, -0.5, 0],
        labels=["<-3", "-3:-2", "-2:-1.5", "-1.5:-1", "-1:-0.5", "-0.5:0"])
    for label, grp in tdf.groupby("dev_bin", observed=True):
        if not len(grp): continue
        log(f"    Dev {str(label):<12} : {len(grp):>6} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    # ── level type breakdown ───────────────────────────────────────────────────
    log(f"\n  ── Target Level Type ─────────────────────────────────")
    for ltype, grp in tdf.groupby("level_type", observed=True):
        if not len(grp): continue
        log(f"    {str(ltype):<18} : {len(grp):>6} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    # ── exit type breakdown ────────────────────────────────────────────────────
    log(f"\n  ── Exit Type Breakdown ───────────────────────────────")
    for etype in ["target_hit", "vwap_hit", "delta_zero", "decel_exit", "time_stop"]:
        grp = tdf[tdf["exit_type"] == etype]
        if not len(grp): continue
        log(f"    {etype:<14} : {len(grp):>6} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}% | "
            f"Avg Hold {grp['bars_held'].mean():.1f} bars")

    # ── hold duration ──────────────────────────────────────────────────────────
    log(f"\n  ── Hold Duration Distribution ────────────────────────")
    tdf["hold_bin"] = pd.cut(tdf["bars_held"],
        bins=[0, 2, 5, 10, 20, 40, 60],
        labels=["1-2", "3-5", "6-10", "11-20", "21-40", "41-60"])
    for label, grp in tdf.groupby("hold_bin", observed=True):
        if not len(grp): continue
        log(f"    Hold {str(label):<8} bars : {len(grp):>6} trades | "
            f"WR {(grp['ret_pct']>0).mean()*100:>5.1f}% | "
            f"AvgRet {grp['ret_pct'].mean():>+7.4f}%")

    # ── top 10 symbols ─────────────────────────────────────────────────────────
    stat_df = pd.DataFrame(ticker_stats).sort_values("avg_ret", ascending=False)
    log(f"\n  ── Top 10 by Avg Return ──────────────────────────────")
    for _, r in stat_df.head(10).iterrows():
        log(f"    {r['symbol']:<14} : {r['n_trades']:>5} trades | "
            f"WR {r['win_rate']:>5.1f}% | AvgRet {r['avg_ret']:>+7.4f}%")

    tdf.to_csv("vwap_drip_trades_v5.csv", index=False)
    stat_df.to_csv("vwap_drip_stats_v5.csv", index=False)
    log(f"\n  Saved → vwap_drip_trades_v5.csv / vwap_drip_stats_v5.csv")
    log("=" * 65)

if __name__ == "__main__":
    main()