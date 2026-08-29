'fda_fastpath.py — FDA-approval fast path (bypasses normal gates)'

import requests
from typing import Optional
import json

from config import (
    FDA_APPROVAL_KEYWORDS,
    FDA_EXCLUDE_KEYWORDS,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
)


def contains_fda_fastpath(text: str) -> bool:
    'CANDIDATE pre-filter only (see the FDA_APPROVAL_KEYWORDS config note above) — a True here means "worth asking qwen to confirm", not "bypa...'
    t = text.lower()
    if any(bad in t for bad in FDA_EXCLUDE_KEYWORDS):
        return False
    return any(k in t for k in FDA_APPROVAL_KEYWORDS)



                                                                               
                                                                              
                                                                             
                                                                           
                                                                      
                                                                             
                                                          
_OLLAMA_FDA_PROMPT = """You are confirming whether a stock-news headline is reporting a genuine, ALREADY-GRANTED, explicit FDA marketing approval for a drug/device — the single most reliable, first-look catalyst this strategy trades, so this must be conservative.

Answer YES only for an approval that has actually been decided and granted right now (e.g. FDA approval, NDA/BLA approved, FDA clearance, accelerated approval granted).
Answer NO for anything short of that, even if it sounds similar or uses the word "approval" — for example: breakthrough therapy / fast track / orphan drug DESIGNATIONS, priority review grants, a PDUFA date being set, the company merely "seeking", "expecting", or "up for" approval, or an FDA advisory committee recommendation (not yet an actual agency approval decision).

Headline: {headline}
Summary: {summary}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"is_explicit_approval": true or false, "confidence": "high" or "medium" or "low", "reasoning": "one short sentence"}}"""


def call_ollama_fda_check(headline: str, summary: str) -> Optional[dict]:
    "Returns {'is_explicit_approval':, 'confidence':, 'reasoning':} or None if Ollama is unreachable/times out/returns something unparseable."
    prompt = _OLLAMA_FDA_PROMPT.format(headline=headline, summary=summary or headline)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        rawText = resp.json().get("response", "")
        parsed = json.loads(rawText)
        isExplicitApproval = bool(parsed.get("is_explicit_approval", False))
        confidence = str(parsed.get("confidence", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
        if confidence not in ("high", "medium", "low"):
            print(f"  [WARN] Ollama FDA-check returned an unexpected shape, treating as no-override: {rawText[:200]}")
            return None
        return {"is_explicit_approval": isExplicitApproval, "confidence": confidence, "reasoning": reasoning}
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Ollama unreachable at {OLLAMA_URL} (is `ollama serve` running?) — no FDA fast-path confirmation this event.")
        return None
    except requests.exceptions.Timeout:
        print(f"  [WARN] Ollama FDA-check call timed out after {OLLAMA_TIMEOUT_SEC}s — no FDA fast-path confirmation this event.")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [WARN] Could not parse Ollama's FDA-check response as the expected JSON shape: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Ollama FDA-check call failed: {e}")
        return None

