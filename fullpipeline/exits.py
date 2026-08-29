'exits.py — ATR stop, CVD/OBV secondary exit, position monitor loop'

import numpy as np
import pandas as pd
import threading
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
import os

import config

if TYPE_CHECKING:
    from execution import Position
from config import (
    CVD_BEARISH_THRESHOLD,
    CVD_LOOKBACK_BARS,
    ENTRY_STOP_ATR_MULTIPLE,
    ENTRY_STOP_ATR_PERIOD,
    EXIT_LOOKBACK_BARS,
    HEARTBEAT_SEC,
    MAX_LOSS_PCT,
    MIN_STOP_PCT,
    NY_TZ,
    POLL_INTERVAL_SEC,
    PT_TZ,
    SECONDARY_EXIT_MOVE_TRIGGER_PCT,
    SECONDARY_EXIT_PST_HOUR,
    SWING_WINDOW_EXIT,
    TOTAL_BUY_OPPORTUNITIES,
    _acked_lock,
    _positions_lock,
    _watchlist_lock,
    acked_news_symbols,
    open_positions,
    opportunities_left,
    trading_paused,
    watchlist,
)


                                                                            
                                                                          
                                                                            

def find_swing_highs(values: np.ndarray, window: int) -> list:
    idx = []
    n = len(values)
    for i in range(window, n - window):
        left, right = values[i - window:i], values[i + 1:i + window + 1]
        if len(left) and len(right) and values[i] > left.max() and values[i] > right.max():
            idx.append(i)
    return idx



def find_swing_lows(values: np.ndarray, window: int) -> list:
    idx = []
    n = len(values)
    for i in range(window, n - window):
        left, right = values[i - window:i], values[i + 1:i + window + 1]
        if len(left) and len(right) and values[i] < left.min() and values[i] < right.min():
            idx.append(i)
    return idx



def compute_obv_series(df: pd.DataFrame) -> np.ndarray:
    priceChange = df["Close"].diff()
    return (np.sign(priceChange) * df["Volume"]).fillna(0).cumsum().values



def bearish_swing_divergence(df: pd.DataFrame, window: int) -> bool:
    "Classic divergence: price makes a higher swing high while OBV's value at that same bar is lower than OBV's value at the prior swing high."
    close = df["Close"].values
    if len(close) < (2 * window + 3):
        return False
    obv = compute_obv_series(df)
    highs = find_swing_highs(close, window)
    if len(highs) < 2:
        return False
    i_new, i_old = highs[-1], highs[-2]
    return bool(close[i_new] > close[i_old] and obv[i_new] < obv[i_old])



def most_recent_swing_low_price(df: pd.DataFrame, window: int) -> Optional[float]:
    close = df["Close"].values
    if len(close) < (2 * window + 1):
        return None
    lows = find_swing_lows(close, window)
    if not lows:
        return None
    return float(close[lows[-1]])


# ---- SET entry-volatility stop (measured once at entry, never trailed) -----


def compute_atr_series(df: pd.DataFrame, period: int) -> np.ndarray:
    "Wilder's ATR (True Range smoothed with Wilder's RMA)."
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(close)
    atr = np.full(n, np.nan)
    if n == 0:
        return atr
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr



def compute_entry_volatility_stop(df: pd.DataFrame, entry_price: float,
                                   atr_period: int = ENTRY_STOP_ATR_PERIOD,
                                   atr_multiple: float = ENTRY_STOP_ATR_MULTIPLE,
                                   max_loss_pct: float = MAX_LOSS_PCT,
                                   min_stop_pct: float = MIN_STOP_PCT) -> dict:
    'Measures volatility ONCE, at entry, to size a SET (non-trailing) stop-loss — replaces the volume-confirmed Chandelier ATR TRAIL (see the...'
    atr_series = compute_atr_series(df, atr_period)
    atr = float(atr_series[-1]) if len(atr_series) and not np.isnan(atr_series[-1]) else None

    if atr is not None and entry_price > 0:
        vol_implied_pct = (atr_multiple * atr) / entry_price
    else:
        vol_implied_pct = max_loss_pct   # not enough bars yet at entry — fall back to the flat cap

    stopPct = max(min_stop_pct, min(vol_implied_pct, max_loss_pct))
    capped = vol_implied_pct >= max_loss_pct
    stopPrice = entry_price * (1 - stopPct)

    return {"stop_price": stopPrice, "stop_pct": stopPct, "atr": atr, "capped": capped}



def compute_session_vwap(df: pd.DataFrame) -> Optional[float]:
    "Simple session VWAP = Sum(price*volume) / Sum(volume) over today's 1-minute bars (see get_price_history_1m) — used by the VWAP entry gate..."
    if df is None or df.empty or "Close" not in df or "Volume" not in df:
        return None
    volume = df["Volume"].astype(float)
    totalVolume = volume.sum()
    if totalVolume <= 0:
        return None
    price = df["Close"].astype(float)
    return float((price * volume).sum() / totalVolume)


                                                                              
                                                                            
                                                                     


def compute_cvd_series(df: pd.DataFrame) -> np.ndarray:
    "Cumulative Volume Delta: running sum of each bar's volume signed by that bar's own price direction (up bar -> +volume, down bar -> -volume)."
    priceChange = df["Close"].diff()
    return (np.sign(priceChange) * df["Volume"]).fillna(0).cumsum().values



def significant_cvd_bearish(df: pd.DataFrame, lookback_bars: int = CVD_LOOKBACK_BARS,
                             threshold: float = CVD_BEARISH_THRESHOLD) -> dict:
    "'Significant CVD bearish' — cumulative volume delta over the last `lookback_bars` bars actively dominated by down-volume, independent of..."
    close = df["Close"].values
    volume = df["Volume"].values
    tailClose = close[-(lookback_bars + 1):]
    tailVolume = volume[-(lookback_bars + 1):]
    if len(tailClose) < 2:
        return {"net_cvd_ratio": 0.0, "significant": False}

    priceChange = np.diff(tailClose)
    bar_volume = tailVolume[1:]   # volume attributed to the bar that produced each price_change
    downVolume = float(bar_volume[priceChange < 0].sum())
    upVolume = float(bar_volume[priceChange >= 0].sum())
    totalVolume = downVolume + upVolume
    netCvdRatio = ((upVolume - downVolume) / totalVolume) if totalVolume > 0 else 0.0

    return {"net_cvd_ratio": netCvdRatio, "significant": netCvdRatio <= threshold}



def _secondary_exit_armed(pos: "Position", current_price: float) -> bool:
    'The secondary exit (significant CVD-bearish / bearish swing divergence) does NOT check from the moment of entry — it only arms once EITHE...'
    now_pt = datetime.now(timezone.utc).astimezone(PT_TZ)
    time_armed = now_pt.hour >= SECONDARY_EXIT_PST_HOUR
    movePct = abs(current_price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price else 0.0
    move_armed = movePct >= SECONDARY_EXIT_MOVE_TRIGGER_PCT
    return bool(time_armed or move_armed)



def _check_position(ticker: str):
    with _positions_lock:
        pos = open_positions.get(ticker)
    if pos is None:
        return

    try:
        df = config.SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] PnL poll failed for {ticker}: {e}")
        return
    if df.empty:
        return

    current_price = float(df["Close"].iloc[-1])

    with _positions_lock:
        pos.peak_price = max(pos.peak_price, current_price)
        peak = pos.peak_price
        entry_price = pos.entry_price
        stopPrice = pos.stopPrice
        stopPct = pos.stopPct
        stopCapped = pos.stopCapped

    pnl_pct = (current_price - entry_price) / entry_price * 100.0 if entry_price else 0.0

    exit_reason = None

                                                                         
                                                                        
                                                                         
                                                                            
    if current_price <= stopPrice:
        cap_note = " [10% cap]" if stopCapped else ""
        exit_reason = (f"SET entry-volatility stop break (stop=${stopPrice:.2f}, "
                        f"{stopPct*100:.1f}% off entry{cap_note})")

                                                                            
                                                                       
                                                                           
                                                                     
                                                                           
                                                                          
                                     
    armed = _secondary_exit_armed(pos, current_price)
    if exit_reason is None and armed:
        since_entry = df[df.index >= pd.Timestamp(pos.entry_time)].tail(EXIT_LOOKBACK_BARS)
        if len(since_entry) >= 2:
            divergence = bearish_swing_divergence(since_entry, window=SWING_WINDOW_EXIT)
            cvd = significant_cvd_bearish(since_entry, lookback_bars=CVD_LOOKBACK_BARS,
                                           threshold=CVD_BEARISH_THRESHOLD)
            if divergence:
                exit_reason = "Bearish OBV/price swing divergence (secondary exit, armed)"
            elif cvd["significant"]:
                exit_reason = (f"Significant bearish CVD (secondary exit, armed; "
                                f"net_cvd_ratio={cvd['net_cvd_ratio']:.2f} <= {CVD_BEARISH_THRESHOLD})")

    if exit_reason:
        from execution import exit_position  # deferred: breaks exits<->execution circular import

        exit_position(ticker, current_price, pnl_pct, exit_reason)
    else:
        print(f"  [PNL] {ticker:<6} px=${current_price:.2f}  pnl={pnl_pct:+.2f}%  peak=${peak:.2f}  "
              f"stop=${stopPrice:.2f}  secondary_armed={armed}")



def monitor_positions_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        with _positions_lock:
            tickers = list(open_positions.keys())
        for ticker in tickers:
            try:
                _check_position(ticker)
            except Exception as e:
                print(f"  [WARN] monitor error on {ticker}: {e}")
        stop_event.wait(POLL_INTERVAL_SEC)



def heartbeat_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        with _watchlist_lock:
            wl = set(watchlist)
        with _positions_lock:
            pos_count = len(open_positions)
        with config._news_lock:
            last_news = config._last_news_at
        with _acked_lock:
            acked = set(acked_news_symbols)

        now = datetime.now(timezone.utc)
        last_news_str = "none yet" if last_news is None else f"{int((now - last_news).total_seconds())}s ago"

                                                                            
                                                                           
                                                                           
                                                                          
                                                    
        missing = wl - acked
        acked_str = f"{len(acked)}/{len(wl)}"
        if missing and len(missing) <= 10:
            acked_str += f"  UNCONFIRMED={sorted(missing)}"
        elif missing:
            acked_str += f"  UNCONFIRMED={len(missing)} tickers (e.g. {sorted(missing)[:5]}...)"

                                                                             
                                                                      
                                                                          
                                                                          
        now_et = now.astimezone(NY_TZ)
                                                                           
                                                                           
                                                                          
                                                                           
                                                                        
        trading_str = "PAUSED (position open)" if pos_count > 0 else "active"
        print(f"  [HEARTBEAT] {now_et.strftime('%H:%M:%S')} ET  alive  "
              f"watchlist={len(wl)}  server_acked={acked_str}  "
              f"open_positions={pos_count}  opportunities_left={opportunities_left()}/{TOTAL_BUY_OPPORTUNITIES}  "
              f"trading={trading_str}  last_news={last_news_str}")
        stop_event.wait(HEARTBEAT_SEC)



