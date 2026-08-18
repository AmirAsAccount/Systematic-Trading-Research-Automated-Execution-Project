#!/usr/bin/env python3
"""
screener_news.py
----------------
1. Screen Finviz: float < 5M, avg vol > 50K
2. For each ticker, fetch last 10 headlines (no date filter)
3. Save → screener_news.csv

Requires: pip install finviz
"""

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import finviz
from finviz.screener import Screener

OUTPUT_CSV = "screener_news.csv"


def fetch_news_for_ticker(ticker: str) -> list[dict]:
    try:
        news = finviz.get_news(ticker)
    except Exception:
        return []

    return [
        {
            "Ticker":    ticker,
            "Timestamp": timestamp,
            "Headline":  headline,
            "Source":    source,
            "URL":       url,
        }
        for timestamp, headline, url, source in news[:10]
    ]


def main():
    print("Screening...")
    stock_list = Screener(
        filters=["sh_float_u5", "sh_avgvol_o50"],
        order="ticker",
        rows=None,
    )
    tickers = [row["Ticker"] for row in stock_list.data]
    print(f"  {len(tickers)} tickers found: {', '.join(tickers)}")

    print("Fetching news in parallel...")
    csv_rows = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch_news_for_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            csv_rows.extend(future.result())

    csv_rows.sort(key=lambda r: (r["Ticker"], r["Timestamp"]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ticker", "Timestamp", "Headline", "Source", "URL"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Done. {len(csv_rows)} headlines across {len(tickers)} tickers → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()