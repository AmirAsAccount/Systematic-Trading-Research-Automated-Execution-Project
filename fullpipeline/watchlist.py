'watchlist.py — Finviz screening, avg-volume cache'

import requests
import time
from lxml import html
import re
from typing import Optional
import sys

import config
from config import (
    AVG_DAILY_VOLUME,
    CACHE_DIR,
    FINVIZ_BATCH_SIZE,
    FINVIZ_COOLDOWN_SEC,
    FINVIZ_PAGE_DELAY_SEC,
    FUNDAMENTALS_CACHE_TTL_SEC,
    QUOTE_BATCH_CHUNK_SIZE,
    TICK_VOLUME_MULTIPLE,
    _avg_daily_volume_lock,
    _fundamentals_cache,
    _fundamentals_lock,
    watchlist,
)


                                                                            
                                                                 
                                                                            

def _finviz_url() -> str:
                                                                               
                                                                        
     
                                                                             
                                                                         
                                                                             
                                                                           
                                                                           
                                                                         
                                                                             
                                                                            
                                          
    floatCode   = "sh_float_u10"     # Float: Under 10M
    marketCap = "cap_microunder"     # MICRO CAP
    avgVolCode  = "sh_avgvol_o100"   # Average Volume: Over 100K
    return (f"https://finviz.com/screener.ashx?v=111&"
            f"f={floatCode},{marketCap},{avgVolCode}&ft=4&o=ticker")



def _fetch_finviz_page(session: requests.Session, page_url: str):
    'Fetch one screener page through a persistent session with real browser-like headers.'
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/screener.ashx",
    }
    resp = session.get(page_url, headers=headers, timeout=15)
    resp.raise_for_status()
    if resp.text.strip() == "Too many requests.":
        raise RuntimeError("Too many requests.")
    return resp


_TICKER_LINK_RE = re.compile(r'[?&]t=([A-Za-z][A-Za-z\.\-]{0,5})(?=[&"\'])')



def _extract_tickers_from_html(html_text: str) -> list:
    "Pull ticker symbols out of '...t=TICKER...' query-string fragments anywhere in the raw page HTML, in document order, de-duplicated."
    seenLocal = set()
    out = []
    for t in _TICKER_LINK_RE.findall(html_text):
        t = t.upper()
        if t not in seenLocal:
            seenLocal.add(t)
            out.append(t)
    return out



def _dump_debug_html(page_url: str, response) -> str:
    'Saves the raw response so we can see exactly what came back instead of guessing at the link format again.'
    debugPath = CACHE_DIR / "finviz_debug_page.html"
    try:
        debugPath.write_text(response.text, encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [DEBUG] Couldn't write debug HTML: {e}")
    text = response.text
    print(f"  [DEBUG] {page_url}")
    print(f"  [DEBUG] status={response.status_code}  content-length={len(text)}")
    print(f"  [DEBUG] contains 'quote.ashx'? {'quote.ashx' in text}   "
          f"contains 't='? {'t=' in text}   contains 'screener'? {'screener' in text.lower()}")
    print(f"  [DEBUG] full HTML saved to {debugPath.resolve()} — open it and search for one of "
          f"your watchlist tickers' text to see what markup actually wraps it, then send me that snippet.")
    return debugPath.as_posix()



def _scrape_finviz_tickers(url: str, label: str) -> list:
    'The actual scrape/paginate/parse loop, factored out of get_all_tickers() so it can be reused against a DIFFERENT Finviz screener URL (see...'
    try:
        from finviz.screener import Screener
        from finviz.helper_functions import scraper_functions as scrape
    except ImportError:
        print("[ERROR] finviz not installed: pip install finviz")
        sys.exit(1)

                                                                               
                                                                       
    probe = Screener.init_from_url(url, rows=20)
    totalRows = getattr(probe, "_total_rows", 0)
    if totalRows <= 0:
        return _extract_tickers_from_html(str(html.tostring(probe._page_content)))

    pageUrls = scrape.get_page_urls(probe._page_content, totalRows, probe._url)
    print(f"  [FINVIZ:{label}] {totalRows} total matches across {len(pageUrls)} pages — fetching all in paced batches.")

    tickers: list = []
    seen: set = set()
    session = requests.Session()

    for i, page_url in enumerate(pageUrls, start=1):
        if i > 1:
            time.sleep(FINVIZ_PAGE_DELAY_SEC)

        response = None

        def _try_fetch():
            nonlocal response
            response = _fetch_finviz_page(session, page_url)
            pageTickers = _extract_tickers_from_html(response.text)
            if not pageTickers:
                raise RuntimeError("page returned 0 tickers (likely throttled/blocked, or link format changed)")
            return pageTickers

        try:
            pageTickers = _try_fetch()
        except Exception as e:
            print(f"  [WARN] Finviz:{label} page {i} failed ({e}) — "
                  f"cooling down {FINVIZ_COOLDOWN_SEC}s and retrying once.")
            time.sleep(FINVIZ_COOLDOWN_SEC)
            try:
                pageTickers = _try_fetch()
            except Exception as e2:
                print(f"  [WARN] Page {i} failed again ({e2}) — "
                      f"stopping pagination early with {len(tickers)}/{totalRows} rows collected.")
                if response is not None:
                    _dump_debug_html(page_url, response)
                break

        newOnPage = 0
        for t in pageTickers:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
                newOnPage += 1
        print(f"  [FINVIZ:{label}] page {i}/{len(pageUrls)}: {len(pageTickers)} tickers on page, "
              f"{newOnPage} new (running total {len(tickers)}/{totalRows})")

        if i % FINVIZ_BATCH_SIZE == 0 and i < len(pageUrls):
            time.sleep(FINVIZ_COOLDOWN_SEC)

    if len(tickers) < totalRows:
        print(f"  [FINVIZ:{label}] WARNING: collected {len(tickers)}/{totalRows} — "
              f"some pages came back short or failed. See [WARN] lines above.")

    return tickers



def get_all_tickers() -> list:
    return _scrape_finviz_tickers(_finviz_url(), label="watchlist")


# ---- Per-ticker fundamentals (float / short-float), fetched lazily ---------


def _parse_share_count(s) -> Optional[float]:
    "Parses Finviz-style strings like '5.20M', '800.00K', '12.5%' -> float."
    if s is None:
        return None
    s = str(s).strip().replace("%", "")
    if not s or s == "-":
        return None
    mult = 1.0
    if s[-1] in ("K", "k"):
        mult, s = 1_000.0, s[:-1]
    elif s[-1] in ("M", "m"):
        mult, s = 1_000_000.0, s[:-1]
    elif s[-1] in ("B", "b"):
        mult, s = 1_000_000_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None



def get_ticker_fundamentals(ticker: str) -> dict:
    "Lazily scrapes float + short-float for a single ticker via the `finviz` package's per-stock fundamentals lookup, cached for FUNDAMENTALS_..."
    now = time.time()
    with _fundamentals_lock:
        cached = _fundamentals_cache.get(ticker)
        if cached and (now - cached["fetched_at"]) < FUNDAMENTALS_CACHE_TTL_SEC:
            return cached["data"]

    try:
        import finviz
        raw = finviz.get_stock(ticker)   # dict of fundamental fields keyed by label
    except Exception as e:
        print(f"  [WARN] Could not fetch fundamentals for {ticker}: {e}")
        raw = {}

    data = {
        "float_shares": _parse_share_count(raw.get("Shs Float")),
        "short_float_pct": _parse_share_count(raw.get("Short Float")),
                                                                             
                                                                            
                                                                         
                                                                             
                            
        "days_to_cover": _parse_share_count(raw.get("Short Ratio")),
                                                                              
                                                                          
        "avg_volume": _parse_share_count(raw.get("Avg Volume")),
    }
    with _fundamentals_lock:
        _fundamentals_cache[ticker] = {"data": data, "fetched_at": now}
    return data



def build_avg_daily_volume_cache(tickers: list):
    'Populates AVG_DAILY_VOLUME ONCE at startup — a single batched pass across the whole watchlist via SchwabClient.get_avg_daily_volumes_batc...'
    if config.SCHWAB is None or not tickers:
        return
    fetched = 0
    for i in range(0, len(tickers), QUOTE_BATCH_CHUNK_SIZE):
        chunk = tickers[i:i + QUOTE_BATCH_CHUNK_SIZE]
        try:
            batch = config.SCHWAB.get_avg_daily_volumes_batch(chunk)
        except Exception as e:
            print(f"  [WARN] Avg-daily-volume fetch failed for chunk starting {chunk[0]}: {e}")
            continue
        with _avg_daily_volume_lock:
            AVG_DAILY_VOLUME.update(batch)
        fetched += len(batch)
    print(f"[INIT] Avg daily volume cache built: {fetched}/{len(tickers)} tickers "
          f"(RVOL checks for the rest of this run are O(1) lookups against this — "
          f"no Finviz, no per-check Schwab history fetch).")



def get_avg_daily_volume(ticker: str) -> Optional[float]:
    'O(1) read of the startup-built cache — see build_avg_daily_volume_cache.'
    with _avg_daily_volume_lock:
        return AVG_DAILY_VOLUME.get(ticker)