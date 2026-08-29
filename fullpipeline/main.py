'main.py — CLI entry point, thread orchestration, reconnect loop'

import threading
from pathlib import Path
import sys
import argparse
from datetime import datetime, timedelta, timezone
from alpaca.data.live.news import NewsDataStream

import config
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    AVG_DAILY_VOLUME,
    CVD_BEARISH_THRESHOLD,
    ENTRY_STOP_ATR_MULTIPLE,
    ENTRY_STOP_ATR_PERIOD,
    FDA_APPROVAL_KEYWORDS,
    FLOAT_CEILING_SHARES,
    GAP_PCT_MIN,
    MAGNITUDE_WEIGHT_FLOAT,
    MAGNITUDE_WEIGHT_LLM,
    MAGNITUDE_WEIGHT_NEWS_CATEGORY,
    MAGNITUDE_WEIGHT_PRICE_ACTION,
    MARKET_SCAN_INTERVAL_SEC,
    MAX_LOSS_PCT,
    MAX_SLOTS_PER_TRADE,
    MIN_AVG_VOLUME_SHARES,
    MIN_SLOTS_PER_TRADE,
    MIN_STOP_PCT,
    NEGATIVE_PROB_MAX,
    NEWS_SUBSCRIBE_CHUNK_SIZE,
    NEWS_TIER_REFRESH_SEC,
    OLD_BULLISH_NEWS_LOOKBACK_DAYS,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_URL,
    POSITIVE_PROB_MIN,
    PREMARKET_MIN_VOLUME_SHARES,
    PRICE_CEILING_USD,
    ROLLING_HIGH_POLL_INTERVAL_SEC,
    RVOL_MIN,
    SCHWAB_ACCOUNT_HASH,
    SCHWAB_MARKETDATA_CLIENT_ID,
    SCHWAB_MARKETDATA_CLIENT_SECRET,
    SCHWAB_MARKETDATA_REFRESH_TOKEN,
    SCHWAB_TRADER_CLIENT_ID,
    SCHWAB_TRADER_CLIENT_SECRET,
    SCHWAB_TRADER_REFRESH_TOKEN,
    SECONDARY_EXIT_MOVE_TRIGGER_PCT,
    SECONDARY_EXIT_PST_HOUR,
    TICK_POLL_INTERVAL_SEC,
    TICK_UP_VOLUME_RATIO_MIN,
    TICK_VOLUME_MULTIPLE,
    TICK_WINDOW_SEC,
    TOTAL_BUY_OPPORTUNITIES,
    WS_RECONNECT_BACKOFF_SEC,
    _watchlist_lock,
    detect_capital_pool,
    watchlist,
)

from execution import SchwabClient, _Tee, schwab_oauth_bootstrap, schwab_account_hash_bootstrap
from watchlist import get_all_tickers, build_avg_daily_volume_cache
from sentiment import get_vader, get_lm, call_ollama_catalyst_check
from exits import monitor_positions_loop, heartbeat_loop
from news_stream import (
    news_reconciliation_loop, chunked_subscribe_news_loop,
    VerboseNewsDataStream, on_news,
)
from breakout import market_scan_loop, bullish_news_tier_loop, rolling_high_breakout_loop


                                                                            
      
                                                                            

def main():

                                                                          
                                                                            
                                                                          
                                                                          
                                                                          
    logPath = Path(__file__).resolve().parent / "output.txt"
    logFile = open(logPath, "a", buffering=1, encoding="utf-8")
    logFile.write(f"\n{'=' * 80}\n[RUN START] {datetime.now(timezone.utc).isoformat()}\n{'=' * 80}\n")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout_tee = _Tee(original_stdout, logFile)
    stderr_tee = _Tee(original_stderr, logFile)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    print(f"[LOG] Mirroring all console output to {logPath} "
          f"(console echo is async/best-effort — see _Tee docstring)")

    try:
        _main_body()
    finally:
        logFile.write(f"[RUN END] {datetime.now(timezone.utc).isoformat()}\n")
        sys.stdout, sys.stderr = original_stdout, original_stderr
                                                                         
                                                                           
                                                                          
                                 
        stdout_tee.close()
        stderr_tee.close()
        logFile.close()



def _main_body():

    parser = argparse.ArgumentParser(description="Live catalyst entry-signal pipeline")
    parser.add_argument("--schwab-login", action="store_true",
                         help="Run the one-time interactive Schwab OAuth bootstrap and exit.")
    parser.add_argument("--app", choices=["marketdata", "trader"], default="marketdata",
                         help="Which app's credentials --schwab-login should bootstrap "
                              "(default: marketdata).")
    parser.add_argument("--schwab-account-hash", action="store_true",
                         help="Fetch and print the account hash needed for order placement, then exit. "
                              "Always uses the trader app's credentials.")
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument("--pathA", action="store_true",
                             help="Run ONLY the news-driven pipeline (websocket + reconciliation). "
                                  "The rolling-high breakout path (Tiers 1-4) is not started. "
                                  "Default (neither flag given): both paths run in parallel.")
    path_group.add_argument("--pathB", action="store_true",
                             help="Run ONLY the rolling-high breakout path (Tiers 1-4). The news "
                                  "websocket is not connected and no news-driven entries can fire. "
                                  "Default (neither flag given): both paths run in parallel.")
    args = parser.parse_args()

    if args.schwab_login:
        schwab_oauth_bootstrap(args.app)
        return
    if args.schwab_account_hash:
        schwab_account_hash_bootstrap()
        return

    required = [
        ("ALPACA_API_KEY", ALPACA_API_KEY), ("ALPACA_API_SECRET", ALPACA_API_SECRET),
        ("SCHWAB_MARKETDATA_CLIENT_ID", SCHWAB_MARKETDATA_CLIENT_ID),
        ("SCHWAB_MARKETDATA_CLIENT_SECRET", SCHWAB_MARKETDATA_CLIENT_SECRET),
        ("SCHWAB_MARKETDATA_REFRESH_TOKEN", SCHWAB_MARKETDATA_REFRESH_TOKEN),
        ("SCHWAB_TRADER_CLIENT_ID", SCHWAB_TRADER_CLIENT_ID),
        ("SCHWAB_TRADER_CLIENT_SECRET", SCHWAB_TRADER_CLIENT_SECRET),
        ("SCHWAB_TRADER_REFRESH_TOKEN", SCHWAB_TRADER_REFRESH_TOKEN),
    ]
    required.append(("SCHWAB_ACCOUNT_HASH", SCHWAB_ACCOUNT_HASH))   # orders are always live now
    missing = [n for n, v in required if not v]
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("        Run with --schwab-login --app marketdata / --app trader if you don't have refresh tokens yet.")
        print("        Run with --schwab-account-hash first if you don't have SCHWAB_ACCOUNT_HASH yet.")
        sys.exit(1)

    config.SCHWAB = SchwabClient(SCHWAB_MARKETDATA_CLIENT_ID, SCHWAB_MARKETDATA_CLIENT_SECRET,
                          SCHWAB_MARKETDATA_REFRESH_TOKEN)
    config.SCHWAB_TRADER = SchwabClient(SCHWAB_TRADER_CLIENT_ID, SCHWAB_TRADER_CLIENT_SECRET,
                                SCHWAB_TRADER_REFRESH_TOKEN)

    print("[INIT] Detecting account buying power...")
    detectedCapital = detect_capital_pool(config.SCHWAB_TRADER, SCHWAB_ACCOUNT_HASH)
    print(f"[INIT] Detected ${detectedCapital:.2f} buying power -> "
          f"{TOTAL_BUY_OPPORTUNITIES} slots of ${config.CAPITAL_PER_SLOT_USD:.2f} each.")

    print("=" * 80)
    print("  LIVE CATALYST ENTRY PIPELINE")
    print(f"  *** LIVE ORDERS — ${config.ACCOUNT_BUYING_POWER_USD:.2f} detected buying power split into "
          f"{TOTAL_BUY_OPPORTUNITIES} capital slots (${config.CAPITAL_PER_SLOT_USD:.2f}/slot). A normal "
          f"entry claims {MIN_SLOTS_PER_TRADE}-{MAX_SLOTS_PER_TRADE} slots (magnitude-scaled), "
          f"converted to shares at entry price (${PRICE_CEILING_USD:.2f}/share ceiling). "
          f"FDA-approval fast path claims ALL slots (the entire remaining pool) at that instant. ***")
    print(f"  Watchlist: float<{FLOAT_CEILING_SHARES/1e6:.0f}M  MICROCAP(SUB 300M) "
          f"avg-vol>{MIN_AVG_VOLUME_SHARES/1e3:.0f}K")
    print(f"  Rolling-high breakout path (parallel to news, 3-tier funnel + halt sub-path): "
          f"Tier1 gap>={GAP_PCT_MIN:.0f}% AND RVOL>={RVOL_MIN:.1f}x AND vol>={PREMARKET_MIN_VOLUME_SHARES:,} "
          f"(batched Schwab quotes across the whole watchlist, refreshed every "
          f"{MARKET_SCAN_INTERVAL_SEC}s) -> Tier2 confirmed-bullish catalyst within "
          f"{OLD_BULLISH_NEWS_LOOKBACK_DAYS} days (Alpaca historical news + qwen, refreshed every "
          f"{NEWS_TIER_REFRESH_SEC//60}min) -> Tier3 premarket/session-high break confirmed by the "
          f"same tick-volume gate as news (>= {TICK_VOLUME_MULTIPLE}x baseline, "
          f">= {TICK_UP_VOLUME_RATIO_MIN*100:.0f}% up-ratio, within {TICK_WINDOW_SEC}s, polled every "
          f"{ROLLING_HIGH_POLL_INTERVAL_SEC}s).  HALT SUB-PATH (independent, no news requirement): "
          f"any halt->reopen detected in the Tier-1 scan is confirmed by the same tick-volume gate "
          f"before entering.")
    print(f"  FDA fast-path candidate keywords (must ALSO be confirmed by {OLLAMA_MODEL} as an "
          f"explicit granted approval before bypassing every other gate): {FDA_APPROVAL_KEYWORDS}")
    print(f"  Sentiment gate tier 2 (PRIMARY, local LLM): {OLLAMA_MODEL} via Ollama, "
          f"{'ENABLED' if OLLAMA_ENABLED else 'DISABLED'} — runs FIRST on every headline; "
          f"tier-1 alone can no longer approve entry")
    print(f"  Sentiment gate tier 1 (SECONDARY confirmation, VADER+LM composite): "
          f"P(negative) < {NEGATIVE_PROB_MAX} AND P(positive) > {POSITIVE_PROB_MIN} — "
          f"only consulted after tier-2 says positive; skipped entirely (not required) "
          f"when tier-2 confidence=high")
    print(f"  Instant tick-volume gate (no wait, no bars): volume-since-news >= "
          f"{TICK_VOLUME_MULTIPLE}x a flat session-scaled baseline (AVG_DAILY_VOLUME, "
          f"fetched once at startup, scaled to a {TICK_WINDOW_SEC}s window — O(1), no "
          f"per-check fetch) AND up-volume ratio >= {TICK_UP_VOLUME_RATIO_MIN*100:.0f}% "
          f"within {TICK_WINDOW_SEC}s of the drop (polled every {TICK_POLL_INTERVAL_SEC}s)")
    print(f"  Exits: SET entry-volatility stop (measured once at entry, "
          f"{ENTRY_STOP_ATR_MULTIPLE}x ATR{ENTRY_STOP_ATR_PERIOD}, capped at "
          f"{MAX_LOSS_PCT*100:.0f}%, floored at {MIN_STOP_PCT*100:.0f}% — never trails)  |  "
          f"secondary (armed after {SECONDARY_EXIT_PST_HOUR}am PT or a "
          f"{SECONDARY_EXIT_MOVE_TRIGGER_PCT:.0f}% move from entry): significant bearish CVD "
          f"(net ratio <= {CVD_BEARISH_THRESHOLD}) or bearish OBV/price divergence")
    print(f"  Magnitude scoring (SIZING ONLY, never gates entry): float "
          f"(w={MAGNITUDE_WEIGHT_FLOAT}) + pre-news price action "
          f"(w={MAGNITUDE_WEIGHT_PRICE_ACTION}) + LLM confidence/reasoning "
          f"(w={MAGNITUDE_WEIGHT_LLM}) + news category (w={MAGNITUDE_WEIGHT_NEWS_CATEGORY}) "
          f"-> {MIN_SLOTS_PER_TRADE}-{MAX_SLOTS_PER_TRADE} slots/entry "
          f"(${config.CAPITAL_PER_SLOT_USD:.2f} each)")
    print("=" * 80)

    print("[INIT] Building initial watchlist from Finviz...")
    initial = get_all_tickers()
    with _watchlist_lock:
        watchlist.update(initial)
    print(f"[INIT] Watchlist size: {len(watchlist)}")

    print("[INIT] Building avg-daily-volume cache (Schwab, one batched pass — "
          "no Finviz, no per-check history fetch)...")
    build_avg_daily_volume_cache(sorted(watchlist))

    get_vader()
    get_lm()

    if OLLAMA_ENABLED:
        print(f"[INIT] Checking Ollama tier-2 escalation ({OLLAMA_MODEL} at {OLLAMA_URL})...")
        test = call_ollama_catalyst_check(
            "Company Reports Record Revenue, Beats Estimates, Raises Full-Year Guidance",
            "The company beat analyst estimates and raised its full-year outlook.")
        if test is None:
            print(f"  [WARN] Ollama did not respond at startup. Tier-2 escalation will fail-closed "
                  f"(no entry) on every headline it would otherwise have rescued until this is fixed. "
                  f"Check: is `ollama serve` running? Is `{OLLAMA_MODEL}` pulled (`ollama pull {OLLAMA_MODEL}`)?")
        else:
            print(f"  [OK] Ollama responded: catalyst={test['catalyst']} confidence={test['confidence']}")

    stop_event = threading.Event()
    config._stop_event_ref = stop_event

    runPathA = not args.pathB   # news pipeline (websocket + reconciliation)
    runPathB = not args.pathA   # rolling-high breakout funnel (Tiers 1-4)
    print(f"[RUN-MODE] Path A (news): {'ON' if runPathA else 'OFF'}   "
          f"Path B (rolling-high breakout): {'ON' if runPathB else 'OFF'}")

    sortedWatchlist = sorted(watchlist)

    threading.Thread(target=monitor_positions_loop, args=(stop_event,), daemon=True).start()
    threading.Thread(target=heartbeat_loop, args=(stop_event,), daemon=True).start()
    if runPathA:
        threading.Thread(target=news_reconciliation_loop, args=(stop_event,), daemon=True).start()
    if runPathB:
        threading.Thread(target=market_scan_loop, args=(stop_event,), daemon=True).start()
        threading.Thread(target=bullish_news_tier_loop, args=(stop_event,), daemon=True).start()
        threading.Thread(target=rolling_high_breakout_loop, args=(stop_event,), daemon=True).start()

    if not runPathA:
                                                                           
                                                                             
        print("[RUN] Path A disabled (--pathB) — no news websocket connected. "
              "Rolling-high breakout path running standalone. (Ctrl+C to stop)\n")
        try:
            while not stop_event.is_set():
                stop_event.wait(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            print("\n[SHUTDOWN] Stopped.")
        return

    print("[RUN] Listening for live news on the watchlist... (Ctrl+C to stop)\n")

                                                                           
                                                                            
                                                                    
                                                                           
                                                                        
                                                                            
                                                                       
                                                                          
                                                                          


                                                                              
                                                                              
                                                                    
                                                                     
                                                                           
                                                                          
                                                                           
                                                                     
                                                                          
                                                                             
                                                                     
                                                                            
    reconnectAttempt = 0
    try:
        while not stop_event.is_set():
            stream = VerboseNewsDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)
            config._stream_ref = stream

            seedBatch = sortedWatchlist[:NEWS_SUBSCRIBE_CHUNK_SIZE]
            rest_batch = sortedWatchlist[NEWS_SUBSCRIBE_CHUNK_SIZE:]
            if seedBatch:
                stream.subscribe_news(on_news, *seedBatch)
                print(f"  [SUB-SEND] Seeded first subscribe batch of {len(seedBatch)} symbol(s) "
                      f"pre-run so the connection has something to open with.")
            if rest_batch:
                threading.Thread(
                    target=chunked_subscribe_news_loop,
                    args=(stream, rest_batch, stop_event),
                    daemon=True,
                ).start()

            reconnectAttempt += 1
            if reconnectAttempt > 1:
                print(f"  [RECONNECT] Attempt #{reconnectAttempt} — reopening the news websocket "
                      f"and resubscribing {len(sortedWatchlist)} symbol(s).")

            try:
                stream.run()
                                                                        
                                                                      
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if stop_event.is_set():
                    break
                print(f"  [WARN] News websocket dropped/timed out: {e!r} — "
                      f"reconnecting in {WS_RECONNECT_BACKOFF_SEC}s.")
                stop_event.wait(WS_RECONNECT_BACKOFF_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\n[SHUTDOWN] Stopped.")


if __name__ == "__main__":
    main()