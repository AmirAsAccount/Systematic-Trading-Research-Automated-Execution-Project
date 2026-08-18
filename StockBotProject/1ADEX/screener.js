// ==UserScript==
// @name         Jupiter OBV Divergence Panel
// @namespace    local.voldiv
// @version      1.0
// @description  Watches which Solana mint you're viewing on jup.ag and pops up a live OBV-divergence chart (price vs on-balance-volume) in a side panel.
// @match        https://jup.ag/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

/*
 * HOW MINT DETECTION WORKS (read this before trusting it blindly)
 * -----------------------------------------------------------------
 * Jupiter is a single-page app — the URL changes without a full reload,
 * and there's no public "give me the mint I'm looking at" API. This script
 * detects the mint two ways, in order:
 *
 *   1. URL path: https://jup.ag/tokens/<mint> — clean, exact.
 *   2. Fallback: scans the URL and visible page text for anything that
 *      matches the shape of a Solana address (base58, 32-44 chars) and
 *      isn't a known non-mint (SOL/USDC/etc program IDs). This is a
 *      heuristic against Jupiter's current DOM/URL scheme — if Jupiter
 *      changes their page structure, this can silently pick the wrong
 *      string or nothing at all.
 *
 * There's always a manual override box in the panel itself. If detection
 * ever looks wrong, just paste the mint there directly — it takes
 * priority over auto-detection until you clear it.
 *
 * DATA SOURCE
 * -----------------------------------------------------------------
 * OHLCV comes from GeckoTerminal's public API (no key, browser-CORS
 * friendly, same source your Python screener uses). Nothing from
 * Birdeye/Jupiter/RugCheck is called here — don't paste private API keys
 * into a script that runs in a public page's JS context; anything in here
 * is visible in devtools.
 */

(function () {
  'use strict';

  // ── CONFIG ────────────────────────────────────────────────────────────
  const GT_BASE          = 'https://api.geckoterminal.com/api/v2';
  const GT_NETWORK       = 'solana';
  const OHLCV_LIMIT      = 300;   // 1m bars pulled per refresh
  const REFRESH_MS       = 30_000; // how often the panel refetches while a mint is active
  const URL_POLL_MS      = 1000;   // how often we check "did the URL change" (SPA nav has no reliable event)
  const PIVOT_LEFT       = 4;      // swing-point detection window (bars each side)
  const PIVOT_RIGHT      = 4;
  const KNOWN_NON_MINTS  = new Set([
    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
    'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB', // USDT
    'So11111111111111111111111111111111111111112', // wSOL — usually the OTHER side of a swap, not the token of interest
  ]);
  const BASE58_RE = /[1-9A-HJ-NP-Za-km-z]{32,44}/g;

  // ── STATE ────────────────────────────────────────────────────────────
  let currentMint   = null;
  let manualMint    = null;   // overrides detection when set
  let lastUrl       = '';
  let refreshTimer  = null;
  let poolCache     = new Map(); // mint -> pool address

  // ── MINT DETECTION ──────────────────────────────────────────────────
  function extractMintFromUrl(url) {
    const m = url.match(/\/tokens\/([1-9A-HJ-NP-Za-km-z]{32,44})/);
    if (m) return m[1];
    const matches = url.match(BASE58_RE) || [];
    const candidates = matches.filter(a => !KNOWN_NON_MINTS.has(a));
    return candidates.length ? candidates[candidates.length - 1] : null;
  }

  function extractMintFromPage() {
    // Best-effort fallback: look for a base58-shaped string in visible text,
    // preferring ones near "contract"/"address"/"mint" wording.
    const bodyText = document.body.innerText || '';
    const all = bodyText.match(BASE58_RE) || [];
    const filtered = all.filter(a => !KNOWN_NON_MINTS.has(a));
    if (!filtered.length) return null;
    // Most repeated candidate wins — the mint tends to appear more than once
    // (header, copy button, links) vs incidental lookalike strings.
    const counts = new Map();
    for (const a of filtered) counts.set(a, (counts.get(a) || 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  }

  function detectMint() {
    if (manualMint) return manualMint;
    return extractMintFromUrl(location.href) || extractMintFromPage();
  }

  // ── GECKOTERMINAL FETCH ─────────────────────────────────────────────
  async function resolvePool(mint) {
    if (poolCache.has(mint)) return poolCache.get(mint);
    const res = await fetch(`${GT_BASE}/networks/${GT_NETWORK}/tokens/${mint}/pools`);
    if (!res.ok) throw new Error(`pool lookup ${res.status}`);
    const json = await res.json();
    const items = json.data || [];
    if (!items.length) { poolCache.set(mint, null); return null; }
    const best = items.reduce((a, b) => {
      const ra = parseFloat(a.attributes?.reserve_in_usd || 0);
      const rb = parseFloat(b.attributes?.reserve_in_usd || 0);
      return rb > ra ? b : a;
    });
    const addr = best.attributes?.address || best.id?.split('_').pop();
    poolCache.set(mint, addr);
    return addr;
  }

  async function fetchOhlcv(poolAddress) {
    const url = `${GT_BASE}/networks/${GT_NETWORK}/pools/${poolAddress}/ohlcv/minute?aggregate=1&limit=${OHLCV_LIMIT}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`ohlcv ${res.status}`);
    const json = await res.json();
    const list = json.data?.attributes?.ohlcv_list || [];
    // GeckoTerminal returns newest-first — flip to chronological order.
    return list.slice().reverse().map(([ts, o, h, l, c, v]) => ({
      t: ts * 1000, o, h, l, c, v,
    }));
  }

  async function fetchTrades(poolAddress) {
    // Free tier: recent trades only (roughly the last ~300), no deep
    // pagination. `kind` is the actual taker side of the swap — real
    // buy/sell tagging, not inferred from price direction like OBV does.
    const url = `${GT_BASE}/networks/${GT_NETWORK}/pools/${poolAddress}/trades`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`trades ${res.status}`);
    const json = await res.json();
    const items = json.data || [];
    return items.map(t => ({
      t: new Date(t.attributes.block_timestamp).getTime(),
      kind: t.attributes.kind, // 'buy' | 'sell'
      usd: parseFloat(t.attributes.volume_in_usd || 0),
    }));
  }

  function computeCvd(bars, trades) {
    // Bucket tagged trades into the same 1m bars as OHLCV, signed
    // (+buy / -sell), then cumulative-sum — this is Cumulative Volume
    // Delta, the closest honest analog to order-flow imbalance available
    // without a real limit order book.
    const bucketMs = 60_000;
    const perBar = new Array(bars.length).fill(0);
    let covered = 0;
    for (const tr of trades) {
      const idx = bars.findIndex(b => tr.t >= b.t && tr.t < b.t + bucketMs);
      if (idx === -1) continue;
      covered++;
      perBar[idx] += tr.kind === 'buy' ? tr.usd : -tr.usd;
    }
    let cum = 0;
    const cvd = perBar.map(v => (cum += v));
    return { cvd, coverage: trades.length ? covered / trades.length : 0 };
  }

  // ── OBV + DIVERGENCE ─────────────────────────────────────────────────
  function computeObv(bars) {
    let obv = 0;
    const out = [0];
    for (let i = 1; i < bars.length; i++) {
      if (bars[i].c > bars[i - 1].c) obv += bars[i].v;
      else if (bars[i].c < bars[i - 1].c) obv -= bars[i].v;
      out.push(obv);
    }
    return out;
  }

  function findPivots(series, left, right, mode) {
    // mode: 'high' or 'low'. Returns array of indices.
    const pivots = [];
    for (let i = left; i < series.length - right; i++) {
      const window = series.slice(i - left, i + right + 1);
      const v = series[i];
      const isHigh = mode === 'high' && v === Math.max(...window);
      const isLow  = mode === 'low'  && v === Math.min(...window);
      if (isHigh || isLow) pivots.push(i);
    }
    return pivots;
  }

  function detectDivergences(closes, series, sourceLabel) {
    const highs = findPivots(closes, PIVOT_LEFT, PIVOT_RIGHT, 'high');
    const lows  = findPivots(closes, PIVOT_LEFT, PIVOT_RIGHT, 'low');
    const markers = []; // {idx, type: 'bearish'|'bullish', source}

    for (let i = 1; i < highs.length; i++) {
      const [a, b] = [highs[i - 1], highs[i]];
      if (closes[b] > closes[a] && series[b] < series[a]) {
        markers.push({ idx: b, type: 'bearish', source: sourceLabel });
      }
    }
    for (let i = 1; i < lows.length; i++) {
      const [a, b] = [lows[i - 1], lows[i]];
      if (closes[b] < closes[a] && series[b] > series[a]) {
        markers.push({ idx: b, type: 'bullish', source: sourceLabel });
      }
    }
    return markers;
  }

  // ── RENDERING ────────────────────────────────────────────────────────
  function normalize(arr) {
    const min = Math.min(...arr), max = Math.max(...arr);
    const range = max - min || 1;
    return arr.map(v => (v - min) / range);
  }

  function drawChart(canvas, closes, obv, cvd, markers) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const nClose = normalize(closes);
    const nObv   = normalize(obv);
    const nCvd   = cvd ? normalize(cvd) : null;
    const n = closes.length;
    const x = i => (i / (n - 1)) * (W - 20) + 10;
    const y = v => H - 10 - v * (H - 40);

    const drawLine = (series, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      series.forEach((v, i) => i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)));
      ctx.stroke();
    };

    drawLine(nClose, '#e8a33d', 1.6);
    drawLine(nObv, '#5aa9e6', 1.1);
    if (nCvd) drawLine(nCvd, '#b073e8', 1.1);

    for (const m of markers) {
      const px = x(m.idx), py = y(nClose[m.idx]);
      const color = m.type === 'bearish' ? '#e5484d' : '#3ecf6a';
      ctx.fillStyle = color;
      ctx.beginPath();
      if (m.source === 'CVD') {
        // diamond marker for CVD-sourced divergence
        ctx.moveTo(px, py - 5); ctx.lineTo(px + 5, py); ctx.lineTo(px, py + 5); ctx.lineTo(px - 5, py);
        ctx.closePath(); ctx.fill();
      } else {
        // circle marker for OBV-sourced divergence
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.font = '10px monospace';
    ctx.fillStyle = '#e8a33d'; ctx.fillText('price', 10, 12);
    ctx.fillStyle = '#5aa9e6'; ctx.fillText('OBV', 50, 12);
    if (nCvd) { ctx.fillStyle = '#b073e8'; ctx.fillText('CVD (●=OBV ♦=CVD)', 82, 12); }
  }

  // ── PANEL UI ─────────────────────────────────────────────────────────
  let panel, canvas, statusEl, mintEl, inputEl;

  function buildPanel() {
    panel = document.createElement('div');
    panel.style.cssText = `
      position:fixed; top:80px; right:16px; width:320px; z-index:999999;
      background:#14151a; border:1px solid #2a2c34; border-radius:10px;
      box-shadow:0 8px 24px rgba(0,0,0,.4); font-family:monospace;
      color:#d8d9de; font-size:12px; padding:10px; cursor:default;
    `;

    const header = document.createElement('div');
    header.style.cssText = 'display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; cursor:move;';
    header.innerHTML = `<b style="color:#e8a33d;">OBV Divergence</b>`;
    const closeBtn = document.createElement('span');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = 'cursor:pointer; opacity:.6;';
    closeBtn.onclick = () => { panel.style.display = 'none'; };
    header.appendChild(closeBtn);
    panel.appendChild(header);
    makeDraggable(panel, header);

    mintEl = document.createElement('div');
    mintEl.style.cssText = 'opacity:.7; word-break:break-all; margin-bottom:4px;';
    panel.appendChild(mintEl);

    inputEl = document.createElement('input');
    inputEl.placeholder = 'manual mint override (optional)';
    inputEl.style.cssText = 'width:100%; box-sizing:border-box; margin-bottom:6px; background:#0d0e12; border:1px solid #2a2c34; color:#d8d9de; padding:4px; border-radius:4px;';
    inputEl.addEventListener('change', () => {
      manualMint = inputEl.value.trim() || null;
      refresh();
    });
    panel.appendChild(inputEl);

    canvas = document.createElement('canvas');
    canvas.width = 300; canvas.height = 160;
    canvas.style.cssText = 'width:100%; background:#0d0e12; border-radius:6px;';
    panel.appendChild(canvas);

    statusEl = document.createElement('div');
    statusEl.style.cssText = 'margin-top:6px; opacity:.6; font-size:11px;';
    panel.appendChild(statusEl);

    document.body.appendChild(panel);
  }

  function makeDraggable(el, handle) {
    let dx, dy, dragging = false;
    handle.addEventListener('mousedown', e => {
      dragging = true;
      dx = e.clientX - el.offsetLeft;
      dy = e.clientY - el.offsetTop;
    });
    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      el.style.left = (e.clientX - dx) + 'px';
      el.style.top  = (e.clientY - dy) + 'px';
      el.style.right = 'auto';
    });
    document.addEventListener('mouseup', () => dragging = false);
  }

  // ── REFRESH LOOP ─────────────────────────────────────────────────────
  async function refresh() {
    const mint = detectMint();
    if (!mint) {
      statusEl.textContent = 'no mint detected on this page';
      mintEl.textContent = '—';
      return;
    }
    if (mint !== currentMint) {
      currentMint = mint;
    }
    mintEl.textContent = mint;
    statusEl.textContent = 'loading…';

    try {
      const pool = await resolvePool(mint);
      if (!pool) { statusEl.textContent = 'no pool found for this mint'; return; }
      const [bars, trades] = await Promise.all([
        fetchOhlcv(pool),
        fetchTrades(pool).catch(() => []), // trades endpoint is a bonus signal — don't fail the whole panel if it's unavailable
      ]);
      if (bars.length < (PIVOT_LEFT + PIVOT_RIGHT + 2)) {
        statusEl.textContent = `only ${bars.length} bars — too little history yet`;
        return;
      }
      const closes = bars.map(b => b.c);
      const obv = computeObv(bars);
      const obvMarkers = detectDivergences(closes, obv, 'OBV');

      let cvd = null, cvdMarkers = [], coverage = 0;
      if (trades.length) {
        const result = computeCvd(bars, trades);
        cvd = result.cvd;
        coverage = result.coverage;
        cvdMarkers = detectDivergences(closes, cvd, 'CVD');
      }

      const markers = [...obvMarkers, ...cvdMarkers];
      drawChart(canvas, closes, obv, cvd, markers);

      const last = markers[markers.length - 1];
      const coverageNote = trades.length
        ? ` · trades cover ~${Math.round(coverage * 100)}% of window`
        : ' · trades endpoint empty/unavailable';
      statusEl.textContent = (last
        ? `last: ${last.source} ${last.type} divergence, ${bars.length - last.idx} bar(s) ago`
        : 'no divergence in view') + coverageNote + ` — ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      statusEl.textContent = `error: ${err.message}`;
    }
  }

  function watchUrl() {
    setInterval(() => {
      if (location.href !== lastUrl) {
        lastUrl = location.href;
        refresh();
      }
    }, URL_POLL_MS);
  }

  // ── INIT ─────────────────────────────────────────────────────────────
  function init() {
    buildPanel();
    lastUrl = location.href;
    refresh();
    watchUrl();
    refreshTimer = setInterval(refresh, REFRESH_MS);
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(init, 500);
  } else {
    window.addEventListener('DOMContentLoaded', () => setTimeout(init, 500));
  }
})();