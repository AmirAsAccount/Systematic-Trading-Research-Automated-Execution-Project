'sentiment.py — VADER + Loughran-McDonald + Ollama tier-2 sentiment'

import requests
from typing import Optional
import logging
import json

from config import (
    FDA_APPROVAL_KEYWORDS,
    NEGATIVE_PROB_MAX,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    POSITIVE_PROB_MIN,
    RED_FLAG_KEYWORDS,
    STALE_RECAP_HEADLINE_PATTERNS,
    _lm,
    _vader,
)                                                  

def get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("[INIT] Loading VADER...")
        _vader = SentimentIntensityAnalyzer()
    return _vader



def get_lm():
    global _lm
    if _lm is None:
        import pysentiment2 as ps
        print("[INIT] Loading Loughran-McDonald dictionary...")
        _lm = ps.LM()
    return _lm



def score_text_full(text: str) -> dict:
    "Returns {'label':, 'confidence':, 'p_negative':, 'p_positive':, 'p_neutral':, 'vader_compound':, 'lm_polarity':, 'lm_pos_words':, 'lm_neg..."
    empty = {"label": "neutral", "confidence": 0.0, "p_negative": 0.5, "p_positive": 0.5, "p_neutral": 1.0,
             "vader_compound": 0.0, "lm_polarity": 0.0, "lm_pos_words": 0, "lm_neg_words": 0, "red_flag_hit": None}
    if not text or not text.strip():
        return empty
    try:
        compound = get_vader().polarity_scores(text)["compound"]   # -1..1

        lm = get_lm()
        lmScore = lm.get_score(lm.tokenize(text))
        lmPos, lmNeg = int(lmScore["Positive"]), int(lmScore["Negative"])
        lmPolarity = float(lmScore["Polarity"])   # -1..1, 0.0 when no LM-dictionary words are found

        # VADER compound rescaled to a complementary 0..1 positive/negative pair.
        vaderPositiveLike = (compound + 1.0) / 2.0
        vaderNegativeLike = 1.0 - vaderPositiveLike

                                                                           
                                                                            
                                                                       
                                                               
        textLower = text.lower()
        redFlagHit = next((k for k in RED_FLAG_KEYWORDS if k in textLower), None)

        pNegative = 1.0 if redFlagHit else vaderNegativeLike
        pPositive = vaderPositiveLike
                                                                              
                                                                               
        if not redFlagHit and lmPos >= 2 and lmNeg == 0:
            pPositive = min(1.0, pPositive + 0.05)

        pNegative = round(min(max(pNegative, 0.0), 1.0), 4)
        pPositive = round(min(max(pPositive, 0.0), 1.0), 4)
        pNeutral = round(max(0.0, 1.0 - max(pNegative, pPositive)), 4)
        label = ("negative" if pNegative >= 0.5 and pNegative >= pPositive
                  else "positive" if pPositive >= 0.5 else "neutral")

        return {
            "label": label, "confidence": round(max(pNegative, pPositive, pNeutral), 4),
            "p_negative": pNegative, "p_positive": pPositive, "p_neutral": pNeutral,
            "vader_compound": round(compound, 4), "lm_polarity": round(lmPolarity, 4),
            "lm_pos_words": lmPos, "lm_neg_words": lmNeg, "red_flag_hit": redFlagHit,
        }
    except Exception as e:
        print(f"  [WARN] VADER/Loughran-McDonald scoring failed: {e}")
        return empty



def passes_sentiment_gate(headline: str, summary: str) -> tuple:
    'COMPOSITE gate: entry requires BOTH negative and positive halves to pass, checked on the worse of headline/summary in each direction — P(...'
    hdl = score_text_full(headline)
    sm = score_text_full(summary if summary and summary.strip() else headline)
    worstPNegative = max(hdl["p_negative"], sm["p_negative"])
    worstPPositive = min(hdl["p_positive"], sm["p_positive"])
    negativeOk = worstPNegative < NEGATIVE_PROB_MAX
    positiveOk = worstPPositive > POSITIVE_PROB_MIN
    passed = negativeOk and positiveOk
    return passed, {
        "headline": hdl, "summary": sm,
        "p_negative_worst": worstPNegative, "p_positive_worst": worstPPositive,
        "negative_ok": negativeOk, "positive_ok": positiveOk,
        "red_flag_hit": hdl.get("red_flag_hit") or sm.get("red_flag_hit"),
    }



                                                                            
                                                                  
                                                                            
                                                                        
                                                                              
                                                                            

_OLLAMA_CATALYST_PROMPT = """You are screening a stock-news headline for a low-float catalyst trading strategy. Decide ONLY whether this is a genuine, real, price-positive catalyst for the stock — NOT how big the move will be, NOT general tone. Routine/procedural news (routine filings, generic conference appearances, analyst coverage initiations with no rating, etc.) is NOT a catalyst even if worded neutrally-to-positively. Numeric beats, guidance raises, contract wins, approvals, and similar concrete positive business developments ARE catalysts even if the wording is flat/factual.

Also decide whether this headline is SECONDHAND/RECAP reporting rather than a primary, quantitative disclosure — for example, an article that reports on a price move that ALREADY happened ("Why Shares Of X Are Up Today", "X Surges: Here's What's Driving It"), or a roundup/listicle piece covering many unrelated tickers at once ("20 Stocks That Have Gone Up Today And Why"). This is true even if the headline doesn't literally contain those words — judge the underlying pattern (reporting ABOUT a move / aggregating many stocks) not just the phrasing. A single-company article about a concrete new business development (earnings, a contract, an approval, guidance) is NOT secondhand/recap just because a reporter wrote it.

Headline: {headline}
Summary: {summary}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"catalyst": "positive" or "negative" or "neutral", "confidence": "high" or "medium" or "low", "reasoning": "one short sentence", "is_stale_or_secondhand": true or false}}"""


def call_ollama_catalyst_check(headline: str, summary: str) -> Optional[dict]:
    "Returns {'catalyst':, 'confidence':, 'reasoning':, 'is_stale_or_secondhand':} or None if Ollama is unreachable/times out/returns somethin..."
    prompt = _OLLAMA_CATALYST_PROMPT.format(headline=headline, summary=summary or headline)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",          # ask Ollama to constrain output to valid JSON
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
        parsed = json.loads(raw_text)
        catalyst = str(parsed.get("catalyst", "")).strip().lower()
        confidence = str(parsed.get("confidence", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
        is_stale_or_secondhand = bool(parsed.get("is_stale_or_secondhand", False))
        if catalyst not in ("positive", "negative", "neutral") or confidence not in ("high", "medium", "low"):
            print(f"  [WARN] Ollama returned an unexpected shape, treating as no-override: {raw_text[:200]}")
            return None
        return {"catalyst": catalyst, "confidence": confidence, "reasoning": reasoning,
                 "is_stale_or_secondhand": is_stale_or_secondhand}
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Ollama unreachable at {OLLAMA_URL} (is `ollama serve` running?) — no LLM escalation this event.")
        return None
    except requests.exceptions.Timeout:
        print(f"  [WARN] Ollama call timed out after {OLLAMA_TIMEOUT_SEC}s — no LLM escalation this event.")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [WARN] Could not parse Ollama's response as the expected JSON shape: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Ollama escalation call failed: {e}")
        return None






