import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullpipeline import get_all_tickers

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_API_SECRET = os.environ["ALPACA_API_SECRET"]

print("[DIAG] Rebuilding watchlist from Finviz...")
wl = sorted(set(get_all_tickers()))
print(f"[DIAG] Watchlist size: {len(wl)}")

client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
# Narrow window: just the period the live run was supposedly listening but silent.
start = datetime.now(timezone.utc) - timedelta(minutes=30)
print(f"[DIAG] Querying from {start.isoformat()} to now...")

total_found = 0
for i in range(0, len(wl), 100):
    chunk = wl[i:i + 100]
    req = NewsRequest(symbols=",".join(chunk), start=start, limit=50,
                       include_content=True, exclude_contentless=False)
    try:
        data = client.get_news(req)
        articles = data.data.get("news", [])
    except Exception as e:
        print(f"[DIAG] chunk starting {chunk[0]}: ERROR {e}")
        continue
    if articles:
        print(f"[DIAG] chunk {i//100 + 1} ({chunk[0]}..{chunk[-1]}): {len(articles)} articles")
        for a in articles:
            print(f"         id={a.id} created={getattr(a, 'created_at', '?')} symbols={a.symbols} {a.headline[:70]}")
    total_found += len(articles)

print(f"[DIAG] TOTAL articles found in last 30 min for full watchlist: {total_found}")