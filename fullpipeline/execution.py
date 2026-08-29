'execution.py — Schwab OAuth client + order placement (enter/exit)'

from datetime import datetime, timedelta, timezone
import csv
import time
import pandas as pd
import requests
import threading
from typing import Optional
import base64
import queue
import random
import sys
import os
from dataclasses import dataclass, field

import config
from config import (
    ALLOW_BUYING_POWER_FIELD_FALLBACK,
    AVG_DAILY_VOLUME,
    ENTRY_FILL_CHECK_SEC,
    ENTRY_FILL_MAX_ATTEMPTS,
    ENTRY_FILL_POLL_INTERVAL_SEC,
    ENTRY_LIMIT_WIDEN_STEP_PCT,
    ENTRY_STOP_ATR_PERIOD,
    ENTRY_VWAP_GATE_ENABLED,
    EXIT_FILL_CHECK_SEC,
    EXIT_FILL_MAX_ATTEMPTS,
    EXIT_FILL_POLL_INTERVAL_SEC,
    EXIT_LIMIT_WIDEN_STEP_PCT,
    EXPECTED_BUYING_POWER_FIELD,
    LIMIT_ORDER_SLIPPAGE_PCT,
    MAX_LOSS_PCT,
    MAX_SLOTS_PER_TRADE,
    MIN_SLOTS_PER_TRADE,
    NY_TZ,
    POLL_INTERVAL_SEC,
    PREMARKET_OPEN_ET,
    PRICE_CEILING_USD,
    QUOTE_BATCH_CHUNK_SIZE,
    SCHWAB_ACCOUNT_HASH,
    SCHWAB_MARKETDATA_CLIENT_ID,
    SCHWAB_MARKETDATA_CLIENT_SECRET,
    SCHWAB_MARKETDATA_REDIRECT_URI,
    SCHWAB_SEAMLESS_SESSION_START_ET,
    SCHWAB_TRADER_CLIENT_ID,
    SCHWAB_TRADER_CLIENT_SECRET,
    SCHWAB_TRADER_REDIRECT_URI,
    SCHWAB_TRADER_REFRESH_TOKEN,
    TOTAL_BUY_OPPORTUNITIES,
    TRADES_LOG,
    _exiting_lock,
    _exiting_tickers,
    _positions_lock,
    claim_fda_all_remaining,
    detect_capital_pool,
    open_positions,
    opportunities_left,
    release_opportunity,
    request_shutdown,
    run_is_complete,
    trading_paused,
    try_claim_shares,
    watchlist,
)

from exits import compute_entry_volatility_stop, compute_session_vwap
from entry_gates import passes_liquidity_gate

                                                                            
                                                                          
                                                                            

class SchwabClient:
    TOKEN_URL   = "https://api.schwabapi.com/v1/oauth/token"
    BASE_URL    = "https://api.schwabapi.com/marketdata/v1"
    TRADER_URL  = "https://api.schwabapi.com/trader/v1"   # accounts + order placement

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _request_with_retry(self, method: str, url: str, *, max_attempts: int = 3,
                             base_delay: float = 0.25, max_delay: float = 1.5, **kwargs):
        'SHORT, bounded retry with jittered exponential backoff for transient connection failures (RemoteDisconnected, connection reset, read time...'
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt == max_attempts:
                        return resp   # let the caller's own raise_for_status() surface the real error
                else:
                    return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_exc = e
                if attempt == max_attempts:
                    raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.15)
            time.sleep(delay)
        if last_exc:
            raise last_exc

    def _refresh(self):
        creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        resp = self._request_with_retry("POST", self.TOKEN_URL, headers=headers, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 1800) - 60

    def _auth_headers(self) -> dict:
        with self._lock:
            if self._access_token is None or time.time() >= self._expires_at:
                self._refresh()
            return {"Authorization": f"Bearer {self._access_token}"}

    def _price_history(self, symbol: str, period_days: int, extended_hours: bool) -> pd.DataFrame:
                                                                           
                                                                             
                                                                          
                                                                          
                                                                           
                                                                       
                                                                           
                                                                          
                       
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=period_days)
        params = {
            "symbol": symbol,
            "periodType": "day",
            "frequencyType": "minute",
            "frequency": 1,
            "needExtendedHoursData": str(extended_hours).lower(),
            "startDate": int(start_dt.timestamp() * 1000),
            "endDate": int(end_dt.timestamp() * 1000),
        }
        resp = self._request_with_retry("GET", f"{self.BASE_URL}/pricehistory", headers=self._auth_headers(),
                                         params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("timestamp").rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
        return df.dropna()

    def get_price_history_1m(self, symbol: str, extended_hours: bool = True) -> pd.DataFrame:
        "Today's 1-minute OHLCV bars (pre/post market included)."
        return self._price_history(symbol, period_days=1, extended_hours=extended_hours)

    def get_quote(self, symbol: str) -> Optional[float]:
        'Simple last-price lookup.'
        full = self.get_quote_full(symbol)
        return full.get("last") if full else None

    def get_quote_full(self, symbol: str) -> dict:
        'Returns last/bid/ask/security-status in one call, used for the liquidity gate and the halt check.'
        resp = self._request_with_retry("GET", f"{self.BASE_URL}/quotes", headers=self._auth_headers(),
                                         params={"symbols": symbol}, timeout=15)
        resp.raise_for_status()
        node = resp.json().get(symbol, {})
        quote = node.get("quote", {})
        return {
            "last": quote.get("lastPrice") or quote.get("mark") or quote.get("closePrice"),
            "bid": quote.get("bidPrice"),
            "ask": quote.get("askPrice"),
                                                                         
                                                                        
                                                                 
                                                                             
                               
            "total_volume": quote.get("totalVolume"),
                                                                             
                                                                           
                                                                          
                                                                       
                                                                              
            "previous_close": quote.get("closePrice"),
            "security_status": quote.get("securityStatus"),
        }

    def get_quotes_batch(self, symbols: list) -> dict:
        "Batched quote fetch for MANY symbols in ONE request — Schwab's /quotes endpoint accepts a comma-separated symbols param."
        if not symbols:
            return {}
        resp = self._request_with_retry("GET", f"{self.BASE_URL}/quotes", headers=self._auth_headers(),
                                         params={"symbols": ",".join(symbols)}, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        out = {}
        for sym in symbols:
            node = raw.get(sym, {})
            quote = node.get("quote", {})
            if not quote:
                continue
            out[sym] = {
                "last": quote.get("lastPrice") or quote.get("mark") or quote.get("closePrice"),
                "bid": quote.get("bidPrice"),
                "ask": quote.get("askPrice"),
                "total_volume": quote.get("totalVolume"),
                "previous_close": quote.get("closePrice"),
                "security_status": quote.get("securityStatus"),
            }
        return out

    def get_avg_daily_volumes_batch(self, symbols: list) -> dict:
        "Startup-only, ONE-TIME batched fetch of each symbol's average daily volume, used to build AVG_DAILY_VOLUME (see build_avg_daily_volume_ca..."
        if not symbols:
            return {}
        resp = self._request_with_retry(
            "GET", f"{self.BASE_URL}/quotes", headers=self._auth_headers(),
            params={"symbols": ",".join(symbols), "fields": "fundamental"}, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        out = {}
        for sym in symbols:
            node = raw.get(sym, {})
            fundamental = node.get("fundamental", {})
            if not fundamental:
                continue
            avg_vol = fundamental.get("avg10DaysVolume") or fundamental.get("avg1YearVolume")
            if avg_vol:
                out[sym] = float(avg_vol)
        return out

    def get_account_hashes(self) -> list:
        "One-time lookup: returns [{'accountNumber':, 'hashValue':}, ...]."
        resp = self._request_with_retry("GET", f"{self.TRADER_URL}/accounts/accountNumbers",
                                         headers=self._auth_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_account_buying_power(self, account_hash: str) -> Optional[float]:
        "Fetches the account's currently available trading capital in USD."
        try:
            resp = self._request_with_retry("GET", f"{self.TRADER_URL}/accounts/{account_hash}",
                                             headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            balances = resp.json().get("securitiesAccount", {}).get("currentBalances", {})
        except Exception as e:
            print(f"  [WARN] Could not fetch account buying power: {e}")
            return None

        for field in ("buyingPower", "cashAvailableForWithdrawal", "cashBalance"):
            raw_value = balances.get(field)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            if field == EXPECTED_BUYING_POWER_FIELD:
                print(f"  [CAPITAL] Buying power detected from currentBalances.{field} = "
                      f"${value:,.2f} (expected field — the run will size its capital pool off this).")
            elif ALLOW_BUYING_POWER_FIELD_FALLBACK:
                print(f"  [WARN] Buying power detected from currentBalances.{field} = ${value:,.2f} "
                      f"— NOT the expected field (currentBalances.{EXPECTED_BUYING_POWER_FIELD} was "
                      f"missing/None on this account). Proceeding anyway because "
                      f"ALLOW_BUYING_POWER_FIELD_FALLBACK=True.")
            else:
                raise RuntimeError(
                    f"Buying-power detection found currentBalances.{field}=${value:,.2f}, but "
                    f"currentBalances.{EXPECTED_BUYING_POWER_FIELD} (the expected field — see "
                    f"EXPECTED_BUYING_POWER_FIELD) was missing or null on this account. Refusing "
                    f"to silently size this run's ENTIRE capital pool off a lower-priority "
                    f"fallback field the account type wasn't expected to need. Either this isn't "
                    f"the account type you thought it was, or Schwab's schema doesn't match what "
                    f"this script assumes — check the account's actual /accounts response before "
                    f"proceeding. If currentBalances.{field} really is what you want this run to "
                    f"size off of, set ALLOW_BUYING_POWER_FIELD_FALLBACK=True and rerun."
                )
            return value
        return None

    def place_equity_order(self, account_hash: str, symbol: str, side: str, quantity: float,
                            limit_price: float) -> dict:
        'Places a live LIMIT equity order, eligible for execution in pre-market, regular hours, AND after-hours within the current day.'
        if not account_hash:
            raise RuntimeError("SCHWAB_ACCOUNT_HASH is not set — run --schwab-account-hash first.")
        body = {
            "orderType": "LIMIT",
            "price": f"{limit_price:.2f}",
            "session": "SEAMLESS",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": side.upper(),
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"},
            }],
        }
                                                                          
                                                                   
                                                                       
                                                                        
                                                                          
                                                                           
                                                                         
                                                 
        assert body["session"] == "SEAMLESS" and body["duration"] == "DAY", (
            f"place_equity_order() must submit Day + extended-hours "
            f"(session=SEAMLESS, duration=DAY) for every order — entries AND "
            f"exits both rely on this to fill in premarket/after-hours. Got "
            f"session={body['session']!r} duration={body['duration']!r} for "
            f"{side.upper()} {symbol}."
        )
        resp = requests.post(f"{self.TRADER_URL}/accounts/{account_hash}/orders",
                              headers={**self._auth_headers(), "Content-Type": "application/json"},
                              json=body, timeout=15)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
                                                                            
                                                                          
                                                                             
                                                                         
                                                                           
                                                                             
                                                                              
                                                                              
                                                                        
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise RuntimeError(
                f"Schwab order rejected (HTTP {resp.status_code}) for {side.upper()} "
                f"{quantity} {symbol} @ limit ${limit_price:.2f}: {detail}"
            ) from e
        location = resp.headers.get("Location")
                                                                             
                                                                           
                                                                             
                          
        order_id = location.rstrip("/").split("/")[-1] if location else None
        return {"status_code": resp.status_code, "order_location": location, "order_id": order_id}

    def get_order_status(self, account_hash: str, order_id: str) -> Optional[str]:
        "Returns the order's current status string (e.g."
        details = self.get_order_fill_details(account_hash, order_id)
        return details.get("status") if details else None

    def get_order_fill_details(self, account_hash: str, order_id: str) -> Optional[dict]:
        'Returns {status, filled_quantity, avg_fill_price} for an order, or None if the lookup fails / the order id is falsy.'
        if not order_id:
            return None
        try:
            resp = self._request_with_retry(
                "GET", f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}",
                headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            order = resp.json()
        except Exception as e:
            print(f"  [WARN] Could not fetch order status for order {order_id}: {e}")
            return None

        status = order.get("status")
        filled_quantity = order.get("filledQuantity") or 0

        avgFillPrice = None
        total_shares = 0.0
        total_notional = 0.0
        for activity in order.get("orderActivityCollection", []) or []:
            for leg in activity.get("executionLegs", []) or []:
                qty = leg.get("quantity")
                px = leg.get("price")
                if qty and px:
                    total_shares += float(qty)
                    total_notional += float(qty) * float(px)
        if total_shares > 0:
            avgFillPrice = total_notional / total_shares
        elif order.get("price") and status == "FILLED":
                                                                             
                                                                        
            avgFillPrice = float(order["price"])

        return {"status": status, "filled_quantity": float(filled_quantity), "avg_fill_price": avgFillPrice}

    def cancel_order(self, account_hash: str, order_id: str) -> bool:
        'Cancels a still-working order.'
        if not order_id:
            return False
        try:
            resp = requests.delete(f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}",
                                    headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"  [WARN] Could not cancel order {order_id} (may have already filled/closed): {e}")
            return False



class _Tee:
    'Mirrors every write to BOTH the original stream (console) and a log file.'

    _SENTINEL = object()

    def __init__(self, original, log_file, console_queue_maxsize: int = 4000):
        self._original = original
        self._log_file = log_file
        self._console_queue: "queue.Queue" = queue.Queue(maxsize=console_queue_maxsize)
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._console_writer_loop, name="tee-console-writer", daemon=True)
        self._writer_thread.start()

    def _console_writer_loop(self):
        'Runs on its own thread.'
        while True:
            data = self._console_queue.get()
            if data is self._SENTINEL:
                return
            try:
                self._original.write(data)
                self._original.flush()
            except Exception:
                                                                          
                                                                          
                                                                     
                pass

    def write(self, data):
                                                                         
                                                          
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except Exception:
            pass
        # Console echo is best-effort and NON-BLOCKING for the caller.
        try:
            self._console_queue.put_nowait(data)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1

    def flush(self):
                                                                            
                                                                          
                                                                          
                                      
        try:
            self._log_file.flush()
        except Exception:
            pass

    def close(self):
        'Optional clean shutdown: stop the writer thread.'
        try:
            self._console_queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass



def schwab_oauth_bootstrap(app: str):
    "One-time interactive helper to mint the first refresh token for whichever app ('marketdata' or 'trader') is selected."
    if app == "trader":
        client_id, client_secret, redirect_uri = (
            SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET, SCHWAB_TRADER_REDIRECT_URI)
        env_prefix = "config.SCHWAB_TRADER"
    else:
        client_id, client_secret, redirect_uri = (
            SCHWAB_MARKETDATA_CLIENT_ID, SCHWAB_MARKETDATA_CLIENT_SECRET, SCHWAB_MARKETDATA_REDIRECT_URI)
        env_prefix = "SCHWAB_MARKETDATA"

    if not client_id or not client_secret:
        print(f"[ERROR] Set {env_prefix}_CLIENT_ID / {env_prefix}_CLIENT_SECRET first.")
        return
    authUrl = (f"https://api.schwabapi.com/v1/oauth/authorize?"
                f"client_id={client_id}&redirect_uri={redirect_uri}")
    print(f"[{app.upper()} APP] 1) Open this URL, log in with your Schwab account, and approve access:\n   {authUrl}")
    print("2) You'll land on a blank/error page after approving — that's expected.")
    print("   Copy the FULL resulting URL from your browser's address bar.")
    returnedUrl = input("Paste the full redirected URL here: ").strip()
    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(returnedUrl).query)
    code = query.get("code", [None])[0]
    if not code:
        print("[ERROR] Couldn't find a 'code=' parameter in that URL.")
        return
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}
    resp = requests.post(SchwabClient.TOKEN_URL, headers=headers, data=payload, timeout=15)
    resp.raise_for_status()
    tokens = resp.json()
    print(f"\nSuccess. Export this (refresh token is valid ~7 days):\n")
    print(f"  export {env_prefix}_REFRESH_TOKEN='{tokens['refresh_token']}'")



def schwab_account_hash_bootstrap():
    'One-time helper: prints the account hash(es) needed for order placement.'
    missing = [n for n, v in [
        ("SCHWAB_TRADER_CLIENT_ID", SCHWAB_TRADER_CLIENT_ID),
        ("SCHWAB_TRADER_CLIENT_SECRET", SCHWAB_TRADER_CLIENT_SECRET),
        ("SCHWAB_TRADER_REFRESH_TOKEN", SCHWAB_TRADER_REFRESH_TOKEN),
    ] if not v]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}. Run --schwab-login --app trader first if needed.")
        return
    client = SchwabClient(SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET, SCHWAB_TRADER_REFRESH_TOKEN)
    try:
        accounts = client.get_account_hashes()
    except Exception as e:
        print(f"[ERROR] Could not fetch account hashes: {e}")
        return
    if not accounts:
        print("[ERROR] No accounts returned — check the account is enabled for API trading.")
        return
    print("\nFound account(s):")
    for a in accounts:
        print(f"  accountNumber=...{str(a.get('accountNumber'))[-4:]}  hashValue={a.get('hashValue')}")
    print("\nExport the hashValue for the account you want this script to trade in:")
    print(f"  export SCHWAB_ACCOUNT_HASH='{accounts[0].get('hashValue')}'")



                                                                            
                                                                             
                                                                            

@dataclass
class Position:
    ticker: str
    entry_time: datetime
    entry_price: float
    peak_price: float
    reason: str
    quantity: float = 0.0
    news_time: Optional[datetime] = None
                                                                 
                                                                          
                                                                         
                                                                     
                                                                          
                                                                        
                              
    stop_price: float = 0.0
    stop_pct: float = MAX_LOSS_PCT
    stop_capped: bool = True
    entry_atr: Optional[float] = None



def log_trade_row(row: dict):
    write_header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "ticker", "time", "price", "value", "reason", "headline"])
        if write_header:
            w.writeheader()
        w.writerow(row)



def _seamless_session_gap_warning() -> Optional[str]:
    "Returns a warning string if it's currently before SCHWAB_SEAMLESS_SESSION_START_ET (7:00 AM ET) — the real start of Schwab's Day+extended..."
    now_et = datetime.now(timezone.utc).astimezone(NY_TZ)
    if now_et.time() < SCHWAB_SEAMLESS_SESSION_START_ET:
        return (f"placed at {now_et.strftime('%H:%M:%S')} ET, before Schwab's "
                f"{SCHWAB_SEAMLESS_SESSION_START_ET.strftime('%H:%M')} ET seamless-session start — "
                f"may sit accepted-but-not-yet-eligible-to-fill until then")
    return None



def enter_position(ticker: str, reason: str, headline: str, news_time: Optional[datetime] = None,
                    bypass_gates: bool = False, desired_slots: Optional[int] = None):
    'bypass_gates=True is the FDA-approval fast path ONLY — the sole exception that skips the halt check, the liquidity check, and the price-c...'
    with _positions_lock:
        if ticker in open_positions:
            print(f"  [SKIP] {ticker} already has an open position — ignoring duplicate signal.")
            return

    if bypass_gates:
        slots = claim_fda_all_remaining()
        if slots <= 0:
            print(f"  [SKIP] {ticker}: FDA fast-path fired but 0 capital slots remain — not entering.")
            return
    else:
        requested = desired_slots if desired_slots else MIN_SLOTS_PER_TRADE
        requested = max(MIN_SLOTS_PER_TRADE, min(int(requested), MAX_SLOTS_PER_TRADE))
        slots = try_claim_shares(requested)
        if slots <= 0:
            print(f"  [SKIP] {ticker}: no capital slots remain this run ({opportunities_left()} left) — not entering.")
            return
        if slots < requested:
            print(f"  [SIZING] {ticker}: magnitude score requested {requested} slot(s) but only "
                  f"{slots} remained in the pool — claimed {slots}.")

        if not passes_liquidity_gate(ticker):
            release_opportunity(slots)
            return

    quote = config.SCHWAB.get_quote_full(ticker)
    entry_price = quote.get("last")
    if not entry_price:
        try:
            df = config.SCHWAB.get_price_history_1m(ticker)
            if not df.empty:
                entry_price = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"  [WARN] Could not fetch entry price for {ticker} from price history: {e}")
    if not entry_price:
        print(f"  [ABORT] {ticker}: no usable price — cannot place the order.")
        release_opportunity(slots)
        return

                                                                           
                                                                          
                      
    if not bypass_gates and entry_price > PRICE_CEILING_USD:
        print(f"  [ABORT] {ticker}: price ${entry_price:.2f} exceeds ${PRICE_CEILING_USD:.2f}/share ceiling — entry blocked.")
        release_opportunity(slots)
        return

                                                                              
                                                                        
                                                                           
                                                                      
    if not bypass_gates and ENTRY_VWAP_GATE_ENABLED:
        try:
            vwap_df = config.SCHWAB.get_price_history_1m(ticker)
        except Exception as e:
            vwap_df = None
            print(f"  [WARN] {ticker}: couldn't fetch bars for the VWAP gate ({e}) — "
                  f"gate can't be evaluated, treating as a pass.")
        session_vwap = compute_session_vwap(vwap_df) if vwap_df is not None else None
        if session_vwap is not None:
            if entry_price <= session_vwap:
                print(f"  [ABORT] {ticker}: price ${entry_price:.2f} is at/below session VWAP "
                      f"${session_vwap:.2f} — entry blocked.")
                release_opportunity(slots)
                return
            print(f"  [GATE] {ticker}: price ${entry_price:.2f} above session VWAP ${session_vwap:.2f} — VWAP gate passed.")
        else:
            print(f"  [WARN] {ticker}: session VWAP unavailable (no usable bar/volume data yet) — "
                  f"gate can't be evaluated, treating as a pass.")

                                                                             
                                                                                   
                                                                          
                                                                          
                                                                           
                                                                        
                                                                        
                                                                              
    claimedDollars = slots * config.CAPITAL_PER_SLOT_USD
    quantity = int(claimedDollars // entry_price) if entry_price > 0 else 0

    if quantity < 1 and not bypass_gates:
        while quantity < 1 and slots < MAX_SLOTS_PER_TRADE:
            extra = try_claim_shares(1)
            if extra <= 0:
                break
            slots += extra
            claimedDollars = slots * config.CAPITAL_PER_SLOT_USD
            quantity = int(claimedDollars // entry_price)

    if quantity < 1:
        print(f"  [ABORT] {ticker}: allocated capital (${claimedDollars:.2f} across {slots} "
              f"slot(s) of ${config.CAPITAL_PER_SLOT_USD:.2f} each) can't buy even 1 share at "
              f"${entry_price:.2f} — entry blocked.")
        release_opportunity(slots)
        return

                                                                           
                                                                              
                                                                           
                                                                             
    buyRef = quote.get("ask") or entry_price
    limit_price = round(buyRef * (1 + LIMIT_ORDER_SLIPPAGE_PCT), 2)

    gapWarning = _seamless_session_gap_warning()
    if gapWarning:
        print(f"  [WARN] {ticker}: BUY order {gapWarning}. If it doesn't fill by 7:00 AM ET, "
              f"it's still working at Schwab, not lost — check the account, don't assume it failed.")

                                                                              
                                                                             
                                                                             
                                                                            
                                                                        
                                                                           
                                                                        
                                                                         
    order_id = None
    filled = False
    fillDetails = None
    for attempt in range(1, ENTRY_FILL_MAX_ATTEMPTS + 1):
        try:
            result = config.SCHWAB_TRADER.place_equity_order(SCHWAB_ACCOUNT_HASH, ticker, "BUY",
                                                        quantity, limit_price)
            order_id = result.get("order_id")
        except Exception as e:
                                                                       
                                                                      
                                                                           
                                                                         
                                                                             
            print(f"  [ABORT] {ticker}: order rejected — {e}")
            print(f"          {quantity} whole share(s), limit=${limit_price:.2f}/share "
                  f"(notional ~${limit_price*quantity:.2f}) was rejected by the API. "
                  f"If this is a broker-assisted-only security, Schwab's error body above "
                  f"should say so explicitly — this order will NOT show up as filled/working "
                  f"in the account.")
            break

        deadline = time.monotonic() + ENTRY_FILL_CHECK_SEC
        status = None
        while time.monotonic() < deadline:
            fillDetails = config.SCHWAB_TRADER.get_order_fill_details(SCHWAB_ACCOUNT_HASH, order_id)
            status = fillDetails.get("status") if fillDetails else None
            if status == "FILLED":
                filled = True
                break
            if status in ("CANCELED", "REJECTED", "EXPIRED"):
                break
            time.sleep(ENTRY_FILL_POLL_INTERVAL_SEC)

        if filled:
            print(f"  [FILLED] {ticker}: entry order {order_id} confirmed filled within "
                  f"{ENTRY_FILL_CHECK_SEC}s (attempt {attempt}/{ENTRY_FILL_MAX_ATTEMPTS}).")
            break

                                                                          
                                                                            
                                                                         
                                                                
        if status not in ("CANCELED", "REJECTED", "EXPIRED"):
            config.SCHWAB_TRADER.cancel_order(SCHWAB_ACCOUNT_HASH, order_id)

        if attempt < ENTRY_FILL_MAX_ATTEMPTS:
            try:
                buyRef = config.SCHWAB.get_quote_full(ticker).get("ask") or buyRef
            except Exception:
                pass
            widen = ENTRY_LIMIT_WIDEN_STEP_PCT * attempt
            limit_price = round(buyRef * (1 + LIMIT_ORDER_SLIPPAGE_PCT + widen), 2)
            print(f"  [RETRY] {ticker}: entry order not filled within {ENTRY_FILL_CHECK_SEC}s "
                  f"(last status={status}) — widening limit to ${limit_price:.2f} and resubmitting "
                  f"(attempt {attempt + 1}/{ENTRY_FILL_MAX_ATTEMPTS}).")
        else:
            print(f"  [WARN] {ticker}: entry still unfilled after {ENTRY_FILL_MAX_ATTEMPTS} widened "
                  f"attempts (last status={status}, last limit=${limit_price:.2f}).")

    if not filled:
                                                                           
                                                                           
        print(f"  [UNFILLED] {ticker}: entry NOT confirmed — releasing {slots} claimed slot(s) "
              f"back to the pool. No position opened.")
        release_opportunity(slots)
        log_trade_row({
            "event": "ENTRY_ATTEMPT_UNFILLED", "ticker": ticker, "time": datetime.now(timezone.utc).isoformat(),
            "price": f"{entry_price:.4f}", "value": "unfilled", "reason": reason, "headline": headline,
        })
        return

                                                                    
                                                                    
                                        
    avgFillPrice = fillDetails.get("avg_fill_price") if fillDetails else None
    if avgFillPrice is not None:
        actualEntryPrice = avgFillPrice
        priceNote = ""
    else:
        actualEntryPrice = entry_price
        priceNote = " [UNCONFIRMED — fill price unavailable, using reference price]"

                                                                             
                                                                        
                                                                          
                                                                            
                                                                          
                                                                         
    try:
        entry_df = config.SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] {ticker}: couldn't fetch bars for the entry-volatility stop ({e}) — "
              f"using the flat {MAX_LOSS_PCT*100:.0f}% stop instead.")
        entry_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    stop = compute_entry_volatility_stop(entry_df, actualEntryPrice)
    cap_note = " [10% cap — ATR implied wider]" if stop["capped"] else ""
    atrNote = "ATR unavailable" if stop["atr"] is None else f"ATR={stop['atr']:.4f}"
    print(f"  [STOP] {ticker}: SET entry-volatility stop = ${stop['stop_price']:.2f} "
          f"({stop['stop_pct']*100:.1f}% off entry{cap_note}, {atrNote}) — "
          f"fixed for the entire trade, does not trail.")

    pos = Position(ticker=ticker, entry_time=datetime.now(timezone.utc),
                    entry_price=actualEntryPrice, peak_price=actualEntryPrice, reason=reason,
                    quantity=quantity, news_time=news_time,
                    stop_price=stop["stop_price"], stop_pct=stop["stop_pct"],
                    stop_capped=stop["capped"], entry_atr=stop["atr"])
    with _positions_lock:
        open_positions[ticker] = pos
                                                                              
                                                                      
                                                                          
                                                                      
                                                                        
                                                           

    print(f"\n  \U0001F7E2 ENTRY  {ticker:<6}  fill=${actualEntryPrice:.2f}{priceNote}  qty={quantity}  "
          f"notional=${actualEntryPrice*quantity:.2f}  slots={slots}/{TOTAL_BUY_OPPORTUNITIES} "
          f"(${claimedDollars:.2f} allocated)  reason={reason}")
    print(f"     \u2514\u2500 {headline}\n")
    print(f"  [OPPORTUNITIES] {opportunities_left()}/{TOTAL_BUY_OPPORTUNITIES} capital slot(s) remaining this run.\n")
    print(f"  [PAUSE] Global entry consideration is now paused (position open) until {ticker} closes.\n")

    log_trade_row({
        "event": "ENTRY", "ticker": ticker, "time": pos.entry_time.isoformat(),
        "price": f"{actualEntryPrice:.4f}", "value": f"qty={quantity}", "reason": reason, "headline": headline,
    })



def exit_position(ticker: str, price: float, pnl_pct: float, reason: str):
    '`price`/`pnl_pct` are the REFERENCE values from the signal that triggered this call (current market price at the moment the stop/exit con...'
    with _positions_lock:
        pos = open_positions.get(ticker)   # PEEK, don't remove yet — removed only on confirmed fill
    if pos is None:
        return

    with _exiting_lock:
        if ticker in _exiting_tickers:
            print(f"  [SKIP] {ticker}: an exit is already in flight for this ticker — not "
                  f"starting a second concurrent SELL attempt.")
            return
        _exiting_tickers.add(ticker)
    try:
        return _exit_position_inner(ticker, price, pnl_pct, reason, pos)
    finally:
        with _exiting_lock:
            _exiting_tickers.discard(ticker)



def _exit_position_inner(ticker: str, price: float, pnl_pct: float, reason: str, pos: "Position"):
    gapWarning = _seamless_session_gap_warning()
    if gapWarning:
        print(f"  [WARN] {ticker}: SELL (exit) order {gapWarning}. A signal firing in this "
              f"window can be accepted without being fillable yet — that gap, not a broken "
              f"exit signal, is the most likely explanation if a triggered exit doesn't "
              f"promptly show up as filled.")

                                                                            
                                                                            
                                                                         
                                                                        
                            
    try:
        sellRef = config.SCHWAB.get_quote_full(ticker).get("bid") or price
    except Exception:
        sellRef = price
    limit_price = round(sellRef * (1 - LIMIT_ORDER_SLIPPAGE_PCT), 2)

                                                                               
                                                                            
                                                                 
                                                                        
                                                                         
                                                                          
                                                 
    order_id = None
    filled = False
    fillDetails = None
    for attempt in range(1, EXIT_FILL_MAX_ATTEMPTS + 1):
        try:
            result = config.SCHWAB_TRADER.place_equity_order(SCHWAB_ACCOUNT_HASH, ticker, "SELL",
                                                        pos.quantity, limit_price)
            order_id = result.get("order_id")
        except Exception as e:
                                                                              
                                                                              
                                                                     
            print(f"  [ERROR] SELL order failed for {ticker} (qty={pos.quantity}, limit=${limit_price:.2f}): {e}")
            print(f"          Position remains in open_positions and under active monitoring — "
                  f"this exit attempt will be retried on the next poll if the exit condition "
                  f"still holds. If Schwab's error body above mentions a broker-assisted-only / "
                  f"restricted security, this symbol cannot be closed via the Trader API and "
                  f"needs a phone/rep order.")
            break

        deadline = time.monotonic() + EXIT_FILL_CHECK_SEC
        status = None
        while time.monotonic() < deadline:
            fillDetails = config.SCHWAB_TRADER.get_order_fill_details(SCHWAB_ACCOUNT_HASH, order_id)
            status = fillDetails.get("status") if fillDetails else None
            if status == "FILLED":
                filled = True
                break
            if status in ("CANCELED", "REJECTED", "EXPIRED"):
                break
            time.sleep(EXIT_FILL_POLL_INTERVAL_SEC)

        if filled:
            print(f"  [FILLED] {ticker}: exit order {order_id} confirmed filled within "
                  f"{EXIT_FILL_CHECK_SEC}s (attempt {attempt}/{EXIT_FILL_MAX_ATTEMPTS}).")
            break

                                                                          
                                                                           
                                                                         
        config.SCHWAB_TRADER.cancel_order(SCHWAB_ACCOUNT_HASH, order_id)

        if attempt < EXIT_FILL_MAX_ATTEMPTS:
            try:
                sellRef = config.SCHWAB.get_quote_full(ticker).get("bid") or sellRef
            except Exception:
                pass
            widen = EXIT_LIMIT_WIDEN_STEP_PCT * attempt
            limit_price = round(sellRef * (1 - LIMIT_ORDER_SLIPPAGE_PCT - widen), 2)
            print(f"  [RETRY] {ticker}: exit order not filled within {EXIT_FILL_CHECK_SEC}s "
                  f"(last status={status}) — widening limit to ${limit_price:.2f} and resubmitting "
                  f"(attempt {attempt + 1}/{EXIT_FILL_MAX_ATTEMPTS}).")
        else:
            print(f"  [WARN] {ticker}: exit still unfilled after {EXIT_FILL_MAX_ATTEMPTS} widened "
                  f"attempts (last status={status}, last limit=${limit_price:.2f}).")

    if not filled:
                                                                          
                                                                          
                                                                         
                                                                           
                                                                    
                                                                    
        print(f"  [UNFILLED] {ticker}: exit NOT confirmed — position remains open and under "
              f"active monitoring (SET stop + secondary exit both still apply). This will be "
              f"retried automatically on the next poll if the exit condition still holds.")
        log_trade_row({
            "event": "EXIT_ATTEMPT_UNFILLED", "ticker": ticker, "time": datetime.now(timezone.utc).isoformat(),
            "price": f"{price:.4f}", "value": "unfilled", "reason": reason, "headline": "",
        })
        return

                                                                          
                                                                       
                                                                     
    with _positions_lock:
        pos = open_positions.pop(ticker, None)
    if pos is None:
        return   # shouldn't happen (we still held the lock the whole time above), but don't crash

    avgFillPrice = fillDetails.get("avg_fill_price") if fillDetails else None
    if avgFillPrice is not None:
        actual_price = avgFillPrice
        priceNote = ""
    else:
        actual_price = price
        priceNote = " [UNCONFIRMED — fill price unavailable, using reference price]"
    actualPnlPct = (actual_price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price else 0.0

    icon = "\U0001F534" if actualPnlPct < 0 else "\U0001F7E2"
    dollarPnl = (actual_price - pos.entry_price) * pos.quantity
    print(f"\n  {icon} EXIT   {ticker:<6}  fill=${actual_price:.2f}{priceNote}  qty={pos.quantity}  "
          f"pnl={actualPnlPct:+.2f}% (${dollarPnl:+.4f})  reason={reason}\n")
    log_trade_row({
        "event": "EXIT", "ticker": ticker, "time": datetime.now(timezone.utc).isoformat(),
        "price": f"{actual_price:.4f}", "value": f"{actualPnlPct:+.2f}", "reason": reason, "headline": "",
    })

    if run_is_complete():
        print("=" * 60)
        print(f"  RUN COMPLETE — all {TOTAL_BUY_OPPORTUNITIES} buying opportunities used "
              f"and all positions closed.")
        print("=" * 60)
        request_shutdown("All buying opportunities used and all positions closed.")


