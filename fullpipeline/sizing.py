'sizing.py — magnitude scoring, capital-slot allocation'

from datetime import datetime

import config
from config import (
    BREAKOUT_WEIGHT_CONFIRMATION,
    BREAKOUT_WEIGHT_GAP_SCANNER,
    FLOAT_CEILING_SHARES,
    GAP_PCT_SCORE_CLIP,
    LLM_CONFIDENCE_SCORE,
    LLM_CONFIDENCE_VS_RICHNESS_MIX,
    LLM_REASONING_RICHNESS_WORD_CAP,
    MAGNITUDE_NEWS_CATEGORY_TIERS,
    MAGNITUDE_WEIGHT_FLOAT,
    MAGNITUDE_WEIGHT_LLM,
    MAGNITUDE_WEIGHT_NEWS_CATEGORY,
    MAGNITUDE_WEIGHT_PRICE_ACTION,
    MAX_SLOTS_PER_TRADE,
    MIN_SLOTS_PER_TRADE,
    NEWS_CATEGORY_BASELINE_SCORE,
    PRICE_ACTION_CLIP_PCT,
    PRICE_ACTION_LOOKBACK_MIN,
    RVOL_SCORE_CLIP,
    TICK_UP_VOLUME_RATIO_MIN,
    TICK_VOLUME_MULTIPLE,
    detect_capital_pool,
    watchlist,
)


def float_score(fundamentals: dict) -> tuple:
    'Lower float -> higher score, scaled against the same FLOAT_CEILING_SHARES the watchlist screen already enforces (so every candidate here...'
    floatShares = fundamentals.get("float_shares")
    if not floatShares or floatShares <= 0:
        return 0.5, None
    score = 1.0 - (floatShares / FLOAT_CEILING_SHARES)
    return max(0.0, min(1.0, score)), floatShares



def price_action_score(ticker: str, news_time: datetime) -> tuple:
    'Scores the PRICE_ACTION_LOOKBACK_MIN minutes of price action strictly BEFORE news_time.'
    try:
        df = config.SCHWAB.get_price_history_1m(ticker)
    except Exception as e:
        print(f"  [WARN] Magnitude price-action fetch failed for {ticker}: {e}")
        return 0.5, None
    if df.empty:
        return 0.5, None

    preNews = df[df.index < news_time]
    if len(preNews) < 2:
        return 0.5, None

    lookback = preNews.tail(PRICE_ACTION_LOOKBACK_MIN)
    startClose = float(lookback["Close"].iloc[0])
    endClose = float(lookback["Close"].iloc[-1])
    if startClose <= 0:
        return 0.5, None

    pctChange = (endClose - startClose) / startClose * 100.0
    clipped = max(-PRICE_ACTION_CLIP_PCT, min(PRICE_ACTION_CLIP_PCT, pctChange))
    score = (clipped + PRICE_ACTION_CLIP_PCT) / (2 * PRICE_ACTION_CLIP_PCT)
    return score, pctChange



def llm_score(llm: dict) -> tuple:
    "Combines qwen3's own confidence label with a crude length-based proxy for how substantive its reasoning was."
    confidenceComponent = LLM_CONFIDENCE_SCORE.get(llm.get("confidence"), 0.3)
    reasoning = llm.get("reasoning") or ""
    wordCount = len(reasoning.split())
    richnessComponent = min(wordCount / LLM_REASONING_RICHNESS_WORD_CAP, 1.0)
    score = (LLM_CONFIDENCE_VS_RICHNESS_MIX * confidenceComponent
             + (1 - LLM_CONFIDENCE_VS_RICHNESS_MIX) * richnessComponent)
    return score, confidenceComponent, richnessComponent, wordCount



def news_category_score(headline: str, summary: str) -> tuple:
    'Highest-scoring matched tier from MAGNITUDE_NEWS_CATEGORY_TIERS, else NEWS_CATEGORY_BASELINE_SCORE.'
    text = f"{headline} {summary}".lower()
    bestScore, bestCat = NEWS_CATEGORY_BASELINE_SCORE, "uncategorized"
    for category, score, keywords in MAGNITUDE_NEWS_CATEGORY_TIERS:
        if score > bestScore and any(kw in text for kw in keywords):
            bestScore, bestCat = score, category
    return bestScore, bestCat



def magnitude_score_to_slots(composite: float) -> int:
    'Linear map from a composite score in [0,1] to an integer CAPITAL SLOT count in [MIN_SLOTS_PER_TRADE, MAX_SLOTS_PER_TRADE].'
    raw = MIN_SLOTS_PER_TRADE + composite * (MAX_SLOTS_PER_TRADE - MIN_SLOTS_PER_TRADE)
    return int(round(max(MIN_SLOTS_PER_TRADE, min(MAX_SLOTS_PER_TRADE, raw))))



def compute_magnitude_score(ticker: str, headline: str, summary: str, llm: dict,
                             news_time: datetime, fundamentals: dict) -> dict:
    'Composite magnitude score -> desired CAPITAL SLOT count for a NORMAL (non-FDA) entry that has already cleared both gates.'
    f_score, floatShares = float_score(fundamentals)
    p_score, pctChange = price_action_score(ticker, news_time)
    l_score, conf_c, rich_c, wordCount = llm_score(llm)
    n_score, category = news_category_score(headline, summary)

    composite = (MAGNITUDE_WEIGHT_FLOAT * f_score
                 + MAGNITUDE_WEIGHT_PRICE_ACTION * p_score
                 + MAGNITUDE_WEIGHT_LLM * l_score
                 + MAGNITUDE_WEIGHT_NEWS_CATEGORY * n_score)
    composite = max(0.0, min(1.0, composite))

    return {
        "composite": composite, "slots": magnitude_score_to_slots(composite),
        "float_score": f_score, "float_shares": floatShares,
        "price_action_score": p_score, "price_action_pct": pctChange,
        "llm_score": l_score, "llm_confidence_component": conf_c,
        "llm_richness_component": rich_c, "llm_reasoning_words": wordCount,
        "news_category_score": n_score, "news_category": category,
    }



def confirmation_score(tick_result: dict) -> float:
    'Path B sizing, component 1/2.'
    baseline = tick_result.get("baseline")
    volumeSince = tick_result.get("volume_since_news", 0.0)
    upRatio = tick_result.get("up_ratio", 0.0)

    if not baseline or baseline <= 0:
        volumeComponent = 0.5   # unknown baseline -> neutral, same convention as float_score
    else:
        multipleAchieved = volumeSince / baseline
        clipped = max(0.0, min(multipleAchieved, TICK_VOLUME_MULTIPLE * 3))
        volumeComponent = clipped / (TICK_VOLUME_MULTIPLE * 3)

    if upRatio >= TICK_UP_VOLUME_RATIO_MIN:
        upRatioComponent = (upRatio - TICK_UP_VOLUME_RATIO_MIN) / (1.0 - TICK_UP_VOLUME_RATIO_MIN)
    else:
        upRatioComponent = 0.0   # shouldn't happen if caller already checked result['pass']

    return max(0.0, min(1.0, 0.5 * volumeComponent + 0.5 * upRatioComponent))



def gap_scanner_score(gap_pct: float, rvol: float) -> float:
    'Path B sizing, component 2/2.'
    gapComponent = max(0.0, min(gap_pct, GAP_PCT_SCORE_CLIP)) / GAP_PCT_SCORE_CLIP
    rvolComponent = max(0.0, min(rvol, RVOL_SCORE_CLIP)) / RVOL_SCORE_CLIP
    return max(0.0, min(1.0, 0.5 * gapComponent + 0.5 * rvolComponent))



def compute_breakout_magnitude_score(tick_result: dict, gap_pct: float, rvol: float) -> dict:
    "Path B's equivalent of compute_magnitude_score — used by BOTH the RVOL/gap breakout path and the halt-reopen path (per request), since bo..."
    c_score = confirmation_score(tick_result)
    g_score = gap_scanner_score(gap_pct, rvol)
    composite = max(0.0, min(1.0, BREAKOUT_WEIGHT_CONFIRMATION * c_score
                                    + BREAKOUT_WEIGHT_GAP_SCANNER * g_score))
    return {
        "composite": composite, "slots": magnitude_score_to_slots(composite),
        "confirmation_score": c_score, "gap_scanner_score": g_score,
        "gap_pct": gap_pct, "rvol": rvol,
    }



