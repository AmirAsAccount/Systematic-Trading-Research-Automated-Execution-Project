"""
liveT.py  —  Live Test Implementation
======================================
Paper trades with a $20 starting balance. No orders are placed.

Pipeline (runs in a continuous loop):
  Stage 1 — Market screen (CoinGecko)
    • High 24h volume AND high 24h price change (bullish only)

  Stage 2 — Signal filter (swing-point bullish volume divergence)
    • Same logic as backtest.py
    • If multiple signals fire simultaneously, rank by divergence strength
      (volume spike ratio at second swing low) and take the strongest

  Trade management
    • One active trade at a time
    • Exit on bearish volume divergence (swing highs, OBV declining)
    • Track running PnL across all trades from $20 base

Dependencies:
    pip install ccxt requests pandas numpy
"""

import time
import logging
import requests
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

STARTING_BALANCE     = 20.0
QUOTE_CURRENCY       = "USDT"
EXCHANGES            = ["okx"]

# ── Stage 1: CoinGecko market screen ─────────────────────────────────────────
CG_BASE              = "https://api.coingecko.com/api/v3"
CG_PAGES             = 4
CG_DELAY             = 1.5
MIN_VOLUME_USDT      = 500_000     # 24h volume floor
MIN_PRICE_CHANGE_24H = 8.0         # % — bullish movers only (positive)

# ── Stage 2 / Signal: swing-point volume divergence ──────────────────────────
LOOKBACK             = 60          # bars fed into the entry window
SWING_N              = 3           # bars each side to confirm a swing
VOL_SPIKE_MULT       = 2.5         # volume at 2nd swing low vs 1st (was 6.0)
AVG_VOL_RATIO        = 1.2         # context avg volume around 2nd vs 1st swing (was 1.5)
CONTEXT_BARS         = 3

# ── Exit ──────────────────────────────────────────────────────────────────────
OBV_DROP_MULT        = 0.97        # OBV at 2nd swing high < this × 1st OBV

# ── Loop timing ───────────────────────────────────────────────────────────────
SCREEN_INTERVAL_SEC  = 300         # re-screen market every 5 min
SIGNAL_INTERVAL_SEC  = 60          # check signals on candidates every 1 min
REQUEST_DELAY        = 0.25

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# EXCHANGE POOL
# ──────────────────────────────────────────────────────────────────────────────

def build_pool(exchange_ids: list[str]) -> list[ccxt.Exchange]:
    pool = []
    for eid in exchange_ids:
        try:
            ex = getattr(ccxt, eid)({"enableRateLimit": True, "timeout": 20_000})
            ex.load_markets()
            pool.append(ex)
            log.info("  ✓ %s (%d markets)", eid, len(ex.markets))
        except Exception as e:
            log.warning("  ✗ %s: %s", eid, e)
    return pool


def find_exchange(symbol: str, quote: str, pool: list) -> tuple:
    # Exact match first
    sym_upper  = symbol.upper()
    pair_exact = f"{sym_upper}/{quote}"
    for ex in pool:
        if pair_exact in ex.markets:
            return ex, pair_exact
    # Fuzzy: CoinGecko symbols often differ from OKX tickers (rndr vs RENDER etc.)
    # scan all USDT markets for a base that matches case-insensitively
    for ex in pool:
        for market_id in ex.markets:
            if not market_id.endswith(f"/{quote}"):
                continue
            if market_id.split("/")[0].upper() == sym_upper:
                return ex, market_id
    return None, None


def fetch_ohlcv(ex: ccxt.Exchange, pair: str, limit: int) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(pair, timeframe="1m", limit=limit)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"]).set_index("datetime").sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — COINGECKO MARKET SCREEN
# ──────────────────────────────────────────────────────────────────────────────

def cg_get(path: str, params: dict = None):
    r = requests.get(CG_BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def stage1_screen() -> list[dict]:
    """
    CoinGecko screen: high volume + meaningful bullish 24h move. That's it.
    Sorted by volume descending so the most liquid movers come first.
    """
    log.info("Stage 1 — CoinGecko market screen …")
    rows = []
    for page in range(1, CG_PAGES + 1):
        data = cg_get("/coins/markets", params={
            "vs_currency":             "usd",
            "order":                   "volume_desc",
            "per_page":                250,
            "page":                    page,
            "sparkline":               "false",
            "price_change_percentage": "24h",
        })
        if not data:
            break
        rows.extend(data)
        time.sleep(CG_DELAY)

    if not rows:
        return []

    df = pd.DataFrame(rows)[["symbol", "total_volume", "price_change_percentage_24h"]].dropna()
    df.columns = ["symbol", "volume_24h", "pct_change_24h"]

    mask = (
        (df["pct_change_24h"] >  0) &
        (df["pct_change_24h"] >= MIN_PRICE_CHANGE_24H) &
        (df["volume_24h"]     >= MIN_VOLUME_USDT)
    )
    filtered = df[mask].sort_values("volume_24h", ascending=False)
    log.info("  %d / %d coins passed Stage 1 (>+%.0f%% bullish, vol >=$%.0f)",
             len(filtered), len(df), MIN_PRICE_CHANGE_24H, MIN_VOLUME_USDT)
    return filtered[["symbol", "pct_change_24h", "volume_24h"]].to_dict("records")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  (same as backtest.py)
# ──────────────────────────────────────────────────────────────────────────────

def find_swing_lows(series: pd.Series, n: int) -> list[int]:
    vals = series.values
    return [
        i for i in range(n, len(vals) - n)
        if vals[i] == min(vals[i-n: i+n+1])
        and list(vals[i-n: i+n+1]).count(vals[i]) == 1
    ]


def find_swing_highs(series: pd.Series, n: int) -> list[int]:
    vals = series.values
    return [
        i for i in range(n, len(vals) - n)
        if vals[i] == max(vals[i-n: i+n+1])
        and list(vals[i-n: i+n+1]).count(vals[i]) == 1
    ]


def compute_obv(df: pd.DataFrame) -> pd.Series:
    pc  = df["close"].diff()
    obv = [0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + df["volume"].iloc[i] * np.sign(pc.iloc[i]))
    return pd.Series(obv, index=df.index)


def check_entry_signal(df: pd.DataFrame, debug: bool = False, label: str = "") -> dict | None:
    """
    Bullish volume divergence:
      - price makes lower low at swing 2 vs swing 1
      - volume at swing 2 spikes >= VOL_SPIKE_MULT × swing 1 volume
      - context average volume around swing 2 >= AVG_VOL_RATIO × swing 1 context
    Returns signal dict with vol_ratio (divergence strength) or None.
    """
    window = df.iloc[-LOOKBACK:] if len(df) >= LOOKBACK else df
    close  = window["close"]
    vol    = window["volume"]
    swings = find_swing_lows(close, SWING_N)

    if len(swings) < 2:
        if debug:
            log.info("    [%s] ✗ only %d swing low(s) found (need 2)", label, len(swings))
        return None

    sw1, sw2 = swings[-2], swings[-1]

    price_ll  = close.iloc[sw2] < close.iloc[sw1]
    vol_ratio = vol.iloc[sw2] / vol.iloc[sw1] if vol.iloc[sw1] > 0 else 0
    vol_spike = vol_ratio >= VOL_SPIKE_MULT

    def ctx_vol(idx):
        lo = max(0, idx - CONTEXT_BARS)
        hi = min(len(vol), idx + CONTEXT_BARS + 1)
        return vol.iloc[lo:hi].mean()

    ctx1, ctx2 = ctx_vol(sw1), ctx_vol(sw2)
    avg_vol_up = ctx2 > ctx1 * AVG_VOL_RATIO

    if debug:
        log.info("    [%s] swings=%d  price_ll=%s  vol_ratio=%.2f× (need %.1f×)  avg_ctx=%.2f× (need %.1f×)",
                 label, len(swings), price_ll, vol_ratio, VOL_SPIKE_MULT,
                 ctx2 / ctx1 if ctx1 > 0 else 0, AVG_VOL_RATIO)

    if price_ll and vol_spike and avg_vol_up:
        return {
            "entry_price": close.iloc[-1],
            "vol_ratio":   round(vol_ratio, 2),
            "sw1_price":   close.iloc[sw1],
            "sw2_price":   close.iloc[sw2],
        }
    return None


def check_exit_signal(df: pd.DataFrame, entry_bar: int) -> bool:
    """
    Bearish volume divergence on swing highs after entry:
      price makes higher high but OBV is declining.
    """
    obv_full   = compute_obv(df)
    bars_after = df.iloc[entry_bar:]
    highs      = bars_after["high"].values
    obv_vals   = obv_full.iloc[entry_bar:].values
    confirmed  = []

    for i in range(SWING_N, len(bars_after) - SWING_N):
        lo = i - SWING_N
        hi = i + SWING_N + 1
        wh = highs[lo:hi]
        if highs[i] == wh.max() and list(wh).count(highs[i]) == 1:
            confirmed.append((i, highs[i], obv_vals[i]))

        if len(confirmed) >= 2:
            sh1, sh2 = confirmed[-2], confirmed[-1]
            if sh2[1] > sh1[1] and sh2[2] < sh1[2] * OBV_DROP_MULT:
                return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# PAPER TRADER
# ──────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self, starting_balance: float):
        self.balance      = starting_balance
        self.total_pnl    = 0.0
        self.trade_count  = 0
        self.active_trade = None
        self.history      = []

    def enter(self, pair: str, price: float, signal: dict, df: pd.DataFrame):
        size_usd = self.balance
        qty      = size_usd / price
        self.active_trade = {
            "pair":        pair,
            "entry_price": price,
            "qty":         qty,
            "size_usd":    size_usd,
            "entry_time":  datetime.now(timezone.utc),
            "vol_ratio":   signal["vol_ratio"],
        }
        log.info("  ▶ ENTER  %s @ %.6f  (size $%.2f  vol_ratio=%.1f×)",
                 pair, price, size_usd, signal["vol_ratio"])

    def exit(self, price: float, reason: str):
        t   = self.active_trade
        pnl = (price - t["entry_price"]) / t["entry_price"] * t["size_usd"]
        pct = (price - t["entry_price"]) / t["entry_price"] * 100
        self.balance     += pnl
        self.total_pnl   += pnl
        self.trade_count += 1

        self.history.append({
            "trade":       self.trade_count,
            "pair":        t["pair"],
            "entry_time":  t["entry_time"],
            "exit_time":   datetime.now(timezone.utc),
            "entry_price": t["entry_price"],
            "exit_price":  price,
            "pnl_usd":     round(pnl, 4),
            "pnl_pct":     round(pct, 4),
            "balance":     round(self.balance, 4),
            "total_pnl":   round(self.total_pnl, 4),
            "reason":      reason,
        })
        self.active_trade = None

        emoji = "✅" if pnl >= 0 else "❌"
        log.info("  %s EXIT   %s @ %.6f  PnL %+.2f$  (%+.2f%%)  Balance $%.2f  Total PnL $%+.2f",
                 emoji, t["pair"], price, pnl, pct, self.balance, self.total_pnl)
        self._print_summary()

    def _print_summary(self):
        if not self.history:
            return
        df   = pd.DataFrame(self.history)
        wins = (df["pnl_usd"] > 0).sum()
        log.info("  ─── Running summary: %d trades  %dW/%dL  Balance $%.2f  Total PnL $%+.2f ───",
                 len(df), wins, len(df) - wins, self.balance, self.total_pnl)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("══════════════════════════════════════════")
    log.info("  liveT.py  —  Paper trading from $%.2f", STARTING_BALANCE)
    log.info("══════════════════════════════════════════")

    log.info("Loading exchange pool …")
    pool = build_pool(EXCHANGES)
    if not pool:
        log.error("No exchanges loaded. Exiting.")
        return

    okx_ex     = pool[0]
    trader     = PaperTrader(STARTING_BALANCE)
    candidates = []
    last_screen = 0

    while True:
        now = time.time()

        # ── Re-screen market every SCREEN_INTERVAL_SEC ───────────────────────
        if now - last_screen >= SCREEN_INTERVAL_SEC:
            stage1     = stage1_screen()
            candidates = []

            not_on_okx = []
            for row in stage1:
                sym = row["symbol"].upper()
                ex, pair = find_exchange(sym, QUOTE_CURRENCY, pool)
                if ex is None:
                    not_on_okx.append(sym)
                    continue
                candidates.append({
                    "sym":        sym,
                    "pair":       pair,
                    "ex":         ex,
                    "pct_change": row["pct_change_24h"],
                    "volume":     row["volume_24h"],
                })
                log.info("  ✓ candidate: %-14s  24h %+.1f%%  vol $%.0f",
                         pair, row["pct_change_24h"], row["volume_24h"])
                time.sleep(REQUEST_DELAY)

            if not_on_okx:
                log.info("  ✗ not on OKX (%d): %s", len(not_on_okx), ", ".join(not_on_okx[:15])
                         + (" …" if len(not_on_okx) > 15 else ""))
            log.info("Candidates after Stage 1: %d", len(candidates))
            last_screen = now

        # ── Signal pass over all candidates ──────────────────────────────────
        if candidates:
            signals_found = []
            log.info("  — Scanning %d candidates for entry signal —", len(candidates))

            for c in candidates:
                if trader.active_trade and trader.active_trade["pair"] == c["pair"]:
                    log.info("    [%s] skipped — active trade", c["pair"])
                    continue
                try:
                    df = fetch_ohlcv(c["ex"], c["pair"], LOOKBACK + 20)
                    if df.empty:
                        log.info("    [%s] no OHLCV data", c["pair"])
                        continue
                    sig = check_entry_signal(df, debug=True, label=c["pair"])
                    if sig:
                        log.info("    [%s] ✅ SIGNAL  vol_ratio=%.2f×  entry=%.6f",
                                 c["pair"], sig["vol_ratio"], sig["entry_price"])
                        signals_found.append({**c, **sig, "df": df})
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    log.info("    [%s] error: %s", c["pair"], e)

            if signals_found and not trader.active_trade:
                signals_found.sort(key=lambda x: x["vol_ratio"], reverse=True)
                best = signals_found[0]
                if len(signals_found) > 1:
                    log.info("  %d simultaneous signals — taking strongest: %s (%.1f×)",
                             len(signals_found), best["pair"], best["vol_ratio"])
                trader.enter(best["pair"], best["entry_price"], best, best["df"])

        # ── Monitor active trade for exit ─────────────────────────────────────
        if trader.active_trade:
            t = trader.active_trade
            try:
                df_live = fetch_ohlcv(okx_ex, t["pair"], LOOKBACK + 20)
                if not df_live.empty:
                    diffs     = abs(df_live.index.tz_convert("UTC") -
                                    pd.Timestamp(t["entry_time"]))
                    entry_bar = int(diffs.argmin())

                    if check_exit_signal(df_live, entry_bar):
                        trader.exit(df_live["close"].iloc[-1], "bearish_vol_divergence")
                    else:
                        current = df_live["close"].iloc[-1]
                        unreal  = (current - t["entry_price"]) / t["entry_price"] * t["size_usd"]
                        log.info("  ↻ Holding %-14s  current %.6f  unrealised $%+.2f",
                                 t["pair"], current, unreal)
            except Exception as e:
                log.warning("  exit check failed for %s: %s", t["pair"], e)

        time.sleep(SIGNAL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nStopped by user.")