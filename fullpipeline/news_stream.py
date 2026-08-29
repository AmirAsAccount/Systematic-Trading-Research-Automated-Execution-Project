'news_stream.py — Alpaca news websocket, dedup, stale-headline filter (Path A)'

from datetime import datetime, timedelta, timezone
import asyncio
import time
import difflib
import re
import threading
from typing import Optional
import logging
from html import unescape as _html_unescape
from lxml import html
import traceback
from alpaca.data.live.news import NewsDataStream
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import config
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    MAX_SLOTS_PER_TRADE,
    NEGATIVE_PROB_MAX,
    NEWS_SUBSCRIBE_CHUNK_SIZE,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    POSITIVE_PROB_MIN,
    STALE_RECAP_HEADLINE_PATTERNS,
    TICK_UP_VOLUME_RATIO_MIN,
    TICK_VOLUME_MULTIPLE,
    TICK_WINDOW_SEC,
    _SEEN_NEWS_CACHE_SIZE,
    _acked_lock,
    _positions_lock,
    _seen_news_ids,
    _seen_news_lock,
    _watchlist_lock,
    acked_news_symbols,
    open_positions,
    opportunities_left,
    trading_paused,
    watchlist,
)

from execution import enter_position
from entry_gates import instant_tick_volume_check
from watchlist import get_avg_daily_volume, get_ticker_fundamentals
from sentiment import call_ollama_catalyst_check, passes_sentiment_gate
from fda_fastpath import call_ollama_fda_check, contains_fda_fastpath
from sizing import compute_magnitude_score

def _already_seen_news(news_id) -> bool:
    'Returns True (and does nothing further) if news_id was already processed; otherwise records it and returns False.'
    if news_id is None:
        return False  # can't dedup without an id; let it through
    with _seen_news_lock:
        if news_id in _seen_news_ids:
            return True
        _seen_news_ids[news_id] = None
        if len(_seen_news_ids) > _SEEN_NEWS_CACHE_SIZE:
            _seen_news_ids.popitem(last=False)
        return False


                                                                               
                                                                          
                                                                           
                                                                         
                                                                       
                                                                         
                                                                       
                                                                       
                                                                   
                                                             
NEWS_DEDUP_SIMILARITY_THRESHOLD = 0.85   # ratio >= this counts as "recycled"
NEWS_DEDUP_HISTORY_PER_TICKER   = 20     # how many recent headlines to keep, per ticker

_recent_headlines_lock = threading.Lock()
_recent_headlines: "dict[str, list[str]]" = {}   # ticker -> [normalized headline, ...] (oldest first)


                                                                               
                                                                           
                                                                            
                                                                     
                                                                          
                                                                           
                                                                            
                                                                             
                                                                             
                                                                            
_bullish_news_history_lock = threading.Lock()
_bullish_news_history: "dict[str, list[datetime]]" = {}   # ticker -> [confirmed-bullish timestamp, ...]



def _is_recycled_headline(ticker: str, headline: str,
                           threshold: float = NEWS_DEDUP_SIMILARITY_THRESHOLD) -> bool:
    'Returns True if `headline` is a near-duplicate of a headline already seen for this ticker (fuzzy match, not exact/id match — see module c...'
    normalized = re.sub(r"\s+", " ", headline).strip().lower()
    if not normalized:
        return False
    with _recent_headlines_lock:
        history = _recent_headlines.setdefault(ticker, [])
        isDupe = any(
            difflib.SequenceMatcher(None, normalized, prior).ratio() >= threshold
            for prior in history
        )
        history.append(normalized)
        if len(history) > NEWS_DEDUP_HISTORY_PER_TICKER:
            del history[0]
    return isDupe



                                                                            
                                                                        
                                                                          
                                                                           
                                                     
                                                                            

def _is_stale_recap_headline(headline: str) -> bool:
    "True if the headline matches a known 'reporting on an already-happened move' pattern (see STALE_RECAP_HEADLINE_PATTERNS above) rather tha..."
    text = headline.lower()
    return any(pat in text for pat in STALE_RECAP_HEADLINE_PATTERNS)



                                                                            
                                                                               
                                                                        
                                                                            

def handle_news_event(ticker: str, headline: str, summary: str, news_time: Optional[datetime] = None):
    if opportunities_left() <= 0:
        return   # pool exhausted — no more entries this run (FDA can't claim 0 either)

    if trading_paused():
                                                                            
                                                                        
                                                                          
        return

    with _positions_lock:
        alreadyOpen = ticker in open_positions
    if alreadyOpen:
        print(f"  [SKIP] {ticker} already has an open position — not re-screening this news.")
        return

    fullText = f"{headline} {summary}".strip()
                                                                         
                                                                           
                                                                           
                                                                          
                                                                       
    if news_time is None:
        news_time = datetime.now(timezone.utc)
    elif news_time.tzinfo is None:
        news_time = news_time.replace(tzinfo=timezone.utc)
    else:
        news_time = news_time.astimezone(timezone.utc)

    if contains_fda_fastpath(fullText):
                                                                             
                                                                       
                                                                             
                                                                          
                                                                          
                                                                      
                                                                           
                                                                         
                                                                               
        fda_check = None
        if OLLAMA_ENABLED:
            fda_check = call_ollama_fda_check(headline, summary)
        if fda_check is not None and fda_check["is_explicit_approval"] and fda_check["confidence"] != "low":
            print(f"  [FDA FAST-PATH] {ticker}: keyword match confirmed by {OLLAMA_MODEL} "
                  f"(confidence={fda_check['confidence']}) as a genuine explicit approval — "
                  f"bypassing every gate, claiming all remaining opportunities, entering directly. "
                  f"Reasoning: {fda_check['reasoning'][:120]}")
            enter_position(ticker, reason="FDA_APPROVAL_FASTPATH", headline=headline,
                            news_time=news_time, bypass_gates=True)
            return
        elif fda_check is not None:
            print(f"  [FDA FAST-PATH] {ticker}: keyword match but {OLLAMA_MODEL} did NOT confirm an "
                  f"explicit granted approval (is_explicit_approval={fda_check['is_explicit_approval']}, "
                  f"confidence={fda_check['confidence']}) — no bypass, falling through to the normal "
                  f"gated flow. Reasoning: {fda_check['reasoning'][:120]}")
        else:
            print(f"  [FDA FAST-PATH] {ticker}: keyword match but qwen confirmation unavailable — "
                  f"no bypass (fail-closed on the fast path only), falling through to the normal "
                  f"gated flow instead of dropping this headline entirely.")

    if _is_recycled_headline(ticker, headline):
        print(f"  [NEWS-DROP] {ticker}: headline is a near-duplicate of a recent headline for "
              f"this ticker (fuzzy similarity >= {NEWS_DEDUP_SIMILARITY_THRESHOLD}, independent of "
              f"article id) — treating as recycled/republished, skipping. {headline[:70]}")
        return

    if _is_stale_recap_headline(headline):
        print(f"  [NEWS-DROP] {ticker}: headline matches a known 'reporting on an already-happened "
              f"move' recap pattern (e.g. 'why shares are up') rather than delivering fresh "
              f"information — skipping before spending an LLM call on it.  {headline[:70]}")
        return

                                                                              
                                                                          
                                                                             
                                                                          
                                                                            
    if not OLLAMA_ENABLED:
        print(f"  [GATE] {ticker}: blocked — LLM escalation is disabled (OLLAMA_ENABLED=False), "
              f"and tier-1 alone is no longer sufficient for entry. No entry.  {headline[:70]}")
        return

    llm = call_ollama_catalyst_check(headline, summary)
    if llm is None:
        print(f"  [ESCALATE] {ticker}: local LLM unavailable/unparseable — fail-closed, no entry "
              f"(tier-1 is never consulted without a tier-2 read first).  {headline[:70]}")
        return

    if llm["is_stale_or_secondhand"]:
                                                                        
                                                                         
                                                                       
                                                                        
                                                                 
        print(f"  [ESCALATE] {ticker}: {OLLAMA_MODEL} flagged this as secondhand/recap/listicle "
              f"reporting rather than a primary catalyst — no entry. Reasoning: {llm['reasoning'][:120]}")
        return

    if llm["catalyst"] != "positive":
        print(f"  [ESCALATE] {ticker}: {OLLAMA_MODEL} says catalyst={llm['catalyst']} "
              f"confidence={llm['confidence']} — no entry. Reasoning: {llm['reasoning'][:120]}")
        return

                                                                             
                                                                             
                                                                             
                                                                         
                                                                          
    from breakout import _record_bullish_news  # deferred: breaks news_stream<->breakout circular import

    _record_bullish_news(ticker, news_time or datetime.now(timezone.utc))

    passed, detail = passes_sentiment_gate(headline, summary)

    if llm["confidence"] == "high":
                                                                          
                                                                          
                                                                           
                                                            
        print(f"  [ESCALATE-PASS] {ticker}: {OLLAMA_MODEL} confirms a POSITIVE catalyst "
              f"(confidence=high) — tier-1 confirmation not required at this confidence "
              f"(tier-1 read: P(neg)={detail['p_negative_worst']:.3f} P(pos)={detail['p_positive_worst']:.3f}, "
              f"pass={passed}) — firing instant RVOL check now. Reasoning: {llm['reasoning'][:120]}")
        _instant_rvol_check_and_maybe_enter(ticker, headline, summary, news_time, llm)
        return

    # medium/low confidence: tier 1 (VADER+LM) must ALSO confirm positive.
    if not passed:
        print(f"  [GATE] {ticker}: {OLLAMA_MODEL} said POSITIVE (confidence={llm['confidence']}) but "
              f"tier-1 secondary confirmation failed — P(negative)={detail['p_negative_worst']:.3f} "
              f"(need <{NEGATIVE_PROB_MAX}) / P(positive)={detail['p_positive_worst']:.3f} "
              f"(need >{POSITIVE_PROB_MIN}). At sub-high LLM confidence both tiers must agree — no entry. "
              f"{headline[:70]}")
        return

    print(f"  [ESCALATE-PASS] {ticker}: {OLLAMA_MODEL} confirms a POSITIVE catalyst "
          f"(confidence={llm['confidence']}), tier-1 secondary confirmation also passed "
          f"(P(neg)={detail['p_negative_worst']:.3f} P(pos)={detail['p_positive_worst']:.3f}) — "
          f"firing instant RVOL check now. Reasoning: {llm['reasoning'][:120]}")
    _instant_rvol_check_and_maybe_enter(ticker, headline, summary, news_time, llm)



def _instant_rvol_check_and_maybe_enter(ticker: str, headline: str, summary: str,
                                         news_time: datetime, llm: dict):
    'The second (and final) entry GATE.'
    with _positions_lock:
        if ticker in open_positions:
            return

    avgVolume = get_avg_daily_volume(ticker)

    result = instant_tick_volume_check(ticker, news_time, avgVolume)

    print(f"  [TICK-CHECK] {ticker:<6} volume_since_news={result['volume_since_news']:.0f} "
          f"(need >= {TICK_VOLUME_MULTIPLE}x scaled baseline="
          f"{result['baseline'] if result['baseline'] is None else round(result['baseline'], 0)}) "
          f"up_ratio={result['up_ratio']:.2f} (need >= {TICK_UP_VOLUME_RATIO_MIN}) "
          f"elapsed={result['elapsed_sec']:.1f}s pass={result['pass']} "
          f"data_quality={result['data_quality']} "
          f"(polls_ok={result['quote_polls_ok']} polls_failed={result['quote_polls_failed']})")

    if result["data_quality"] == "NO_DATA":
        print(f"  [WARN] {ticker}: volume_since_news={result['volume_since_news']:.0f} is NOT a "
              f"confirmed real reading — every quote poll during this window failed or returned "
              f"no total_volume field ({result['quote_polls_failed']} failed poll(s), 0 usable). "
              f"Treat this as a data outage, not a genuine zero-volume reaction. Check Schwab "
              f"connectivity / the totalVolume field name (see module docstring VERIFY note).")

    if result["pass"]:
                                                                             
                                                                              
                                                                               
                                                                           
                                                                                   
        fundamentals = get_ticker_fundamentals(ticker)
        magnitude = compute_magnitude_score(ticker, headline, summary, llm, news_time, fundamentals)
        print(f"  [MAGNITUDE] {ticker:<6} composite={magnitude['composite']:.2f} -> "
              f"slots={magnitude['slots']}/{MAX_SLOTS_PER_TRADE} "
              f"(~${magnitude['slots']*config.CAPITAL_PER_SLOT_USD:.2f} requested)  "
              f"float_score={magnitude['float_score']:.2f}"
              f"(float={magnitude['float_shares']})  "
              f"price_action_score={magnitude['price_action_score']:.2f}"
              f"(pct={magnitude['price_action_pct']})  "
              f"llm_score={magnitude['llm_score']:.2f}"
              f"(confidence={llm.get('confidence')}, reasoning_words={magnitude['llm_reasoning_words']})  "
              f"news_category_score={magnitude['news_category_score']:.2f}"
              f"(category={magnitude['news_category']})")
        enter_position(ticker, reason="SENTIMENT_AND_TICK_VOLUME_CONFIRMED", headline=headline,
                        news_time=news_time, desired_slots=magnitude["slots"])
    else:
        print(f"  [TICK-CHECK] {ticker}: volume/up-down split not confirmed within "
              f"{TICK_WINDOW_SEC}s of news drop — no entry.")



                                                                            
                               
                                                                           
                                                                            
                                                                            
                                                                            
                                                                           
                                                                          
                                                                            
                                                                           
                                                                      
                                                                           
                                                                       
                                                                            
                                                                     
                                                                          
                                                                          
                                               
                                                                            

                                                                          
                                                                           
                                           
_COMPANY_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z&'.\-]*(?:\s+[A-Z][A-Za-z&'.\-]*){0,4}\s+"
    r"(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|Holdings?|"
    r"Group|Technologies|Pharmaceuticals|Therapeutics|Biotechnology|Biotech|"
    r"Solutions|International|Global|Industries|Systems)\.?)\b"
)

                                                                           
                                                                            
                                                                             
_MARKET_WIDE_OPENERS = (
    "nasdaq", "dow", "s&p", "s&p 500", "russell", "market", "markets",
    "stocks", "wall street", "futures",
)


def _extract_named_company_phrase(headline: str) -> Optional[str]:
    'Best-effort extraction of a specific company name the headline is ANCHORED to (e.g.'
    if not headline:
        return None
                                                                       
                                                                          
                                                                         
                                                                            
                                                                           
                                                                             
    for m in re.finditer(r"([A-Z][A-Za-z0-9&'.\-]*(?:\s+[A-Z][A-Za-z0-9&'.\-]*){0,5})\s+[Ss]hares\b", headline):
        candidate = m.group(1).strip()
                                                                          
                                                                  
        candidateClean = re.split(r"[;:,]\s*", candidate)[-1].strip()
        if candidateClean.lower() not in _MARKET_WIDE_OPENERS and candidateClean:
            return candidateClean
    m = _COMPANY_SUFFIX_RE.search(headline)
    if m:
        return m.group(1).strip()
    return None



def _ticker_relevant_to_headline(ticker: str, headline: str, summary: str, num_symbols: int) -> bool:
    'Returns False only when we have a strong positive reason to believe this ticker is an incidental tag on a roundup article about a DIFFERE...'
    if num_symbols < 3:
        return True   # low fan-out — not the roundup pattern this guards against

    anchor = _extract_named_company_phrase(headline)
    if not anchor:
        return True   # no clear single-company anchor found — don't guess, keep it

    tickerLower = ticker.lower()
    combinedLower = f"{headline} {summary}".lower()

                                                                           
                                                                      
                                                                          
                                               
    if re.search(rf"\b{re.escape(tickerLower)}\b", combinedLower):
        return True

    anchorLower = anchor.lower()
    if tickerLower in anchorLower or anchorLower in tickerLower:
        return True   # crude but safe: covers cases where they do overlap

                                                                            
                                                                           
                                              
    return False



def _strip_html(raw: str) -> str:
    "Alpaca/Benzinga's News.content field 'might contain HTML' per the model docstring — strip tags down to plain text for sentiment/LLM input."
    if not raw:
        return ""
    try:
        text = html.fromstring(raw).text_content()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()



def _process_news_article(news, tag: str = "NEWS-IN"):
    'Shared processing path for a single News object, regardless of whether it arrived via the live websocket (on_news) or the REST reconcilia...'
    news_id = getattr(news, "id", None)
    symbols = list(getattr(news, "symbols", None) or [])
    headline = getattr(news, "headline", "") or ""
    summary = getattr(news, "summary", "") or ""
                                                                       
                                                                             
                                                                       
                                                                     
                                                                   
                                                                          
                                                                   
                                                                            
                                                                      
                           
    headline = _html_unescape(headline)
    summary = _html_unescape(summary)
                                                                           
                                                                          
                                                                           
                                                                           
                                                                          
                                                                           
    rawContent = getattr(news, "content", "") or ""
    contentText = _strip_html(rawContent)
    if contentText and len(contentText) > 4000:
        contentText = contentText[:4000]  # bound prompt size/latency, not the whole article is needed
    if contentText and contentText not in summary:
        summary = f"{summary}\n\n{contentText}".strip()
                                                                          
                                                                             
                                                 
    articleTime = getattr(news, "created_at", None) or datetime.now(timezone.utc)

    if _already_seen_news(news_id):
        return

    with config._news_lock:
        config._last_news_at = datetime.now(timezone.utc)

                                                                      
                                                                      
                                                                          
                                                                      
                                                                         
                                                                       
    print(f"  [{tag}] id={news_id} symbols={symbols}  {headline[:80]}")

    if not symbols:
        print(f"  [NEWS-DROP] no symbols tagged on this article — skipped.  {headline[:70]}")
        return

    with _watchlist_lock:
        wl = set(watchlist)
    relevant = [s for s in symbols if s in wl]
    if not relevant:
        print(f"  [NEWS-DROP] {symbols} not on current watchlist ({len(wl)} tickers) — skipped.")
        return

    for ticker in relevant:
        if not _ticker_relevant_to_headline(ticker, headline, summary, len(symbols)):
            print(f"  [NEWS-DROP] {ticker}: article tags {len(symbols)} symbols but headline text "
                  f"is anchored to a different named company — skipping {ticker} for this article "
                  f"instead of scoring it against unrelated text.  {headline[:70]}")
            continue
        handle_news_event(ticker, headline, summary, articleTime)


async def on_news(news):
    try:
        await asyncio.to_thread(_process_news_article, news, "NEWS-IN")
    except Exception:
                                                                          
                                                                            
                                                                     
                                                                        
                                                                        
                                                                        
        news_id = getattr(news, "id", "?")
        print(f"  [PROCESS-ERROR] on_news crashed processing article id={news_id} — "
              f"logged and skipped, connection kept alive. Traceback:")
        traceback.print_exc()



def news_reconciliation_loop(stop_event: threading.Event, lookback_minutes: int = 10,
                              interval_seconds: int = 180):
    "Safety net that runs independently of the live websocket's health."
    client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    while not stop_event.is_set():
        if stop_event.wait(interval_seconds):
            break
        with _watchlist_lock:
            wl = sorted(watchlist)
        if not wl:
            continue
        start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                                                                        
                                                               
        for i in range(0, len(wl), 100):
            chunk = wl[i:i + 100]
            try:
                req = NewsRequest(symbols=",".join(chunk), start=start, limit=50,
                                   include_content=True, exclude_contentless=False)
                newsset = client.get_news(req)
                articles = newsset.data.get("news", [])
            except Exception as e:
                print(f"  [RECONCILE-ERROR] chunk starting {chunk[0]}: {e}")
                continue
            for article in articles:
                                                                           
                                                                            
                                                                    
                                                                      
                                                                          
                                                                            
                                                                            
                                                                      
                                                                             
                                                                          
                                                                    
                try:
                    _process_news_article(article, tag="RECONCILE")
                except Exception:
                    article_id = getattr(article, "id", "?")
                    print(f"  [PROCESS-ERROR] reconciliation loop crashed processing "
                          f"article id={article_id} — logged and skipped, loop kept alive. Traceback:")
                    traceback.print_exc()



def chunked_subscribe_news_loop(stream: "VerboseNewsDataStream", symbols: list, stop_event: threading.Event):
    'Subscribes to `symbols` in small batches (NEWS_SUBSCRIBE_CHUNK_SIZE at a time), waiting for the stream to actually be connected/running b...'
    remaining = list(symbols)
    total = len(remaining)
    sent = 0

                                                                            
                                                                           
    while not stop_event.is_set() and not getattr(stream, "_running", False):
        time.sleep(0.25)

    while remaining and not stop_event.is_set():
        batch = remaining[:NEWS_SUBSCRIBE_CHUNK_SIZE]
        remaining = remaining[NEWS_SUBSCRIBE_CHUNK_SIZE:]
        try:
            stream.subscribe_news(on_news, *batch)
        except Exception as e:
            print(f"  [SUB-ERROR] Batch subscribe crashed for {len(batch)} symbol(s) "
                  f"(e.g. {batch[:3]}...): {e} — will not retry this batch automatically.")
            continue
        sent += len(batch)
        print(f"  [SUB-SEND] Sent subscribe batch of {len(batch)} symbol(s) "
              f"({sent}/{total} sent so far) — waiting briefly for ack before next batch.")
                                                                     
                                                                         
                                                              
        time.sleep(1.0)

    if not stop_event.is_set():
        print(f"  [SUB-SEND] All {total} symbol(s) sent in batches of "
              f"{NEWS_SUBSCRIBE_CHUNK_SIZE}. Check heartbeat server_acked for confirmation.")



class VerboseNewsDataStream(NewsDataStream):
    "Same as NewsDataStream, except it also records which symbols Alpaca's server actually acknowledged for the news channel, and prints that..."

    async def _dispatch(self, msg):
        msg_type = msg.get("T")
        if msg_type == "subscription":
            acked = list(msg.get("news", []))
            with _acked_lock:
                acked_news_symbols.update(acked)
                total_acked = len(acked_news_symbols)
            print(f"  [SUB-ACK] Alpaca confirmed {len(acked)} news symbol(s) in this ack "
                  f"(cumulative server-confirmed total: {total_acked})")
        elif msg_type == "error":
            print(f"  [SUB-ERROR] Alpaca rejected/errored on subscription: "
                  f"{msg.get('msg')} (code {msg.get('code')})")
        await super()._dispatch(msg)


