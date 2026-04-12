"""
Black Swan Dashboard — premium single-file FastAPI app.

Reads from coordinator.get_status() and the world model's thought ring buffer.
Polls every 2 seconds. Localhost only.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

logger = structlog.get_logger()


def create_app(coordinator: Any) -> FastAPI:
    app = FastAPI(title="Black Swan", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @app.get("/api/status")
    async def api_status():
        try:
            status = coordinator.get_status()
            thoughts = list(getattr(coordinator.world, "_thought_buffer", []))
            return JSONResponse({"ok": True, "status": status, "thoughts": thoughts})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    return app


async def run_dashboard(coordinator: Any, port: int = 8000):
    app = create_app(coordinator)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    print(f"  [DASHBOARD] http://localhost:{port}", flush=True)
    await server.serve()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Black Swan</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg-base: #07090d;
  --bg-panel: #0d1117;
  --bg-elevated: #131923;
  --bg-hover: #1a2130;
  --border: #1e2533;
  --border-strong: #2a3447;
  --text-primary: #e6edf3;
  --text-secondary: #8b96a8;
  --text-dim: #4d5666;
  --text-faint: #2d3444;
  --accent: #a78bfa;
  --accent-glow: rgba(167, 139, 250, 0.15);
  --cyan: #67e8f9;
  --cyan-glow: rgba(103, 232, 249, 0.12);
  --green: #4ade80;
  --green-dim: #166534;
  --red: #f87171;
  --red-dim: #7f1d1d;
  --amber: #fbbf24;
  --amber-dim: #78350f;
  --pink: #f472b6;
  --blue: #60a5fa;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg-base);
  color: var(--text-primary);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.01em;
}

body {
  background:
    radial-gradient(ellipse 1200px 600px at 20% -10%, rgba(167, 139, 250, 0.06), transparent),
    radial-gradient(ellipse 800px 400px at 80% 100%, rgba(103, 232, 249, 0.04), transparent),
    var(--bg-base);
  min-height: 100vh;
  padding: 24px 32px 40px;
}

.mono { font-family: 'JetBrains Mono', monospace; font-feature-settings: 'tnum' 1, 'zero' 1; }
.tabular { font-variant-numeric: tabular-nums; }

/* ── HEADER ─────────────────────────────────────── */

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-mark {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--cyan));
  border-radius: 9px;
  position: relative;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 24px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.1);
}
.brand-mark::after {
  content: '';
  position: absolute;
  width: 14px; height: 14px;
  background: var(--bg-base);
  border-radius: 50%;
}

.brand-text h1 {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}
.brand-text .tagline {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 1px;
  letter-spacing: 0.02em;
}

.status-badge {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 12px;
  background: rgba(74, 222, 128, 0.08);
  border: 1px solid rgba(74, 222, 128, 0.2);
  border-radius: 999px;
  font-size: 11px;
  color: var(--green);
  font-weight: 500;
  letter-spacing: 0.02em;
}
.live-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}

/* ── KPI STRIP ──────────────────────────────────── */

.kpi-strip {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 24px;
}

.kpi {
  background: var(--bg-panel);
  padding: 18px 22px;
  position: relative;
  transition: background 0.2s;
}
.kpi:hover { background: var(--bg-elevated); }

.kpi-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  font-weight: 500;
  margin-bottom: 8px;
}
.kpi-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
.kpi-value.green { color: var(--green); }
.kpi-value.red { color: var(--red); }
.kpi-value.accent { color: var(--accent); }
.kpi-sub {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
}
.kpi-sub.green { color: var(--green); }
.kpi-sub.red { color: var(--red); }

/* ── LAYOUT ─────────────────────────────────────── */

.main-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel-header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(180deg, rgba(255,255,255,0.015), transparent);
}
.panel-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 10px;
}
.panel-title-accent {
  width: 3px;
  height: 12px;
  background: var(--accent);
  border-radius: 2px;
}
.panel-meta {
  font-size: 10px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.05em;
}

.panel-body {
  padding: 8px 0;
  flex: 1;
  overflow-y: auto;
  max-height: 540px;
}
.panel-body::-webkit-scrollbar { width: 6px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
.panel-body::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── THOUGHT FEED ───────────────────────────────── */

.thought {
  padding: 12px 20px;
  border-bottom: 1px solid rgba(30, 37, 51, 0.5);
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 14px;
  align-items: flex-start;
  transition: background 0.15s;
  animation: slideIn 0.35s cubic-bezier(0.2, 0.9, 0.3, 1);
}
.thought:hover { background: rgba(255,255,255,0.015); }
.thought:last-child { border-bottom: none; }

@keyframes slideIn {
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
}

.thought-agent-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.02em;
  white-space: nowrap;
  border: 1px solid;
}

.thought-body { min-width: 0; }
.thought-question {
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.thought-meta {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
  font-family: 'JetBrains Mono', monospace;
  display: flex;
  gap: 14px;
  align-items: center;
}
.thought-meta .sep { color: var(--text-faint); }
.thought-meta .edge-pos { color: var(--green); font-weight: 600; }
.thought-meta .edge-neg { color: var(--red); font-weight: 600; }
.thought-meta .conf { color: var(--text-secondary); }

.thought-prob-bar {
  width: 64px;
  align-self: center;
}
.prob-bar-track {
  width: 100%;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
  position: relative;
}
.prob-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  border-radius: 2px;
  transition: width 0.4s cubic-bezier(0.2, 0.9, 0.3, 1);
}
.prob-bar-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-secondary);
  margin-top: 4px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.thought.surprise {
  background: linear-gradient(90deg, rgba(251, 191, 36, 0.05), transparent 60%);
  border-left: 2px solid var(--amber);
}
.thought.surprise .thought-question {
  color: var(--amber);
}

/* Agent color coding */
.agent-news_scout       { color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); background: rgba(96, 165, 250, 0.08); }
.agent-market_scanner   { color: #67e8f9; border-color: rgba(103, 232, 249, 0.3); background: rgba(103, 232, 249, 0.08); }
.agent-probability_estimator { color: #a78bfa; border-color: rgba(167, 139, 250, 0.3); background: rgba(167, 139, 250, 0.08); }
.agent-correlation_detective { color: #f472b6; border-color: rgba(244, 114, 182, 0.3); background: rgba(244, 114, 182, 0.08); }
.agent-contrarian       { color: #fb7185; border-color: rgba(251, 113, 133, 0.3); background: rgba(251, 113, 133, 0.08); }
.agent-momentum_detector{ color: #4ade80; border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.08); }
.agent-edge_stacker     { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.08); }
.agent-social_signals   { color: #c084fc; border-color: rgba(192, 132, 252, 0.3); background: rgba(192, 132, 252, 0.08); }
.agent-sports_intelligence { color: #34d399; border-color: rgba(52, 211, 153, 0.3); background: rgba(52, 211, 153, 0.08); }
.agent-odds_arbitrage   { color: #fcd34d; border-color: rgba(252, 211, 77, 0.3); background: rgba(252, 211, 77, 0.08); }
.agent-web_researcher   { color: #818cf8; border-color: rgba(129, 140, 248, 0.3); background: rgba(129, 140, 248, 0.08); }
.agent-adversarial      { color: #f87171; border-color: rgba(248, 113, 113, 0.3); background: rgba(248, 113, 113, 0.08); }
.agent-belief_propagation { color: #94a3b8; border-color: rgba(148, 163, 184, 0.3); background: rgba(148, 163, 184, 0.08); }

/* ── EDGES TABLE ────────────────────────────────── */

.row {
  display: grid;
  padding: 12px 20px;
  gap: 12px;
  align-items: center;
  border-bottom: 1px solid rgba(30, 37, 51, 0.5);
  font-size: 12px;
  transition: background 0.15s;
}
.row:hover { background: rgba(255,255,255,0.012); }
.row:last-child { border-bottom: none; }
.row.head {
  padding: 10px 20px;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  font-weight: 600;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.01);
}

.edge-row { grid-template-columns: 1fr 60px 60px 60px 70px; }
.market-row { grid-template-columns: 1fr 50px 50px 60px 50px; }

.row .question {
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
}
.row .num {
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary);
}

.tag {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  text-align: center;
  font-family: 'Inter', sans-serif;
  border: 1px solid;
}
.tag.green { color: var(--green); border-color: rgba(74, 222, 128, 0.3); background: rgba(74, 222, 128, 0.1); }
.tag.red { color: var(--red); border-color: rgba(248, 113, 113, 0.3); background: rgba(248, 113, 113, 0.1); }
.tag.amber { color: var(--amber); border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.1); }
.tag.dim { color: var(--text-dim); border-color: var(--border-strong); background: transparent; }

.edge-pct.pos { color: var(--green); font-weight: 600; }
.edge-pct.neg { color: var(--red); font-weight: 600; }

/* ── HEALTH GRID ────────────────────────────────── */

.health-grid {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 12px;
}
.health-cell {
  background: var(--bg-elevated);
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  position: relative;
  overflow: hidden;
  transition: all 0.2s;
}
.health-cell:hover {
  border-color: var(--border-strong);
  transform: translateY(-1px);
}
.health-cell::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.3;
}
.health-label {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  font-weight: 500;
  margin-bottom: 6px;
}
.health-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}

/* ── AGENT GRID ─────────────────────────────────── */

.agent-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  padding: 16px 20px;
}
.agent-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  position: relative;
  transition: all 0.2s;
}
.agent-card:hover {
  border-color: var(--border-strong);
  background: var(--bg-hover);
}
.agent-card.idle { opacity: 0.5; }
.agent-card.error { border-color: rgba(248, 113, 113, 0.3); }

.agent-card-name {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  font-family: 'JetBrains Mono', monospace;
}
.agent-card-stats {
  display: flex;
  gap: 10px;
  font-size: 10px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}
.agent-card-stats .runs { color: var(--text-secondary); }
.agent-card-stats .errs.has { color: var(--red); }
.agent-status-dot {
  position: absolute;
  top: 12px; right: 12px;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
}
.agent-card.idle .agent-status-dot { background: var(--text-faint); box-shadow: none; }
.agent-card.error .agent-status-dot { background: var(--red); box-shadow: 0 0 6px var(--red); }

.empty {
  padding: 40px 20px;
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
  font-style: italic;
}

/* ── BOTTOM SECTION ─────────────────────────────── */

.bottom-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="brand-mark"></div>
    <div class="brand-text">
      <h1>BLACK SWAN</h1>
      <div class="tagline">Multi-agent prediction market intelligence</div>
    </div>
  </div>
  <div class="status-badge">
    <div class="live-dot"></div>
    <span id="status-text">SYSTEM LIVE</span>
  </div>
</div>

<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Uptime</div>
    <div class="kpi-value" id="uptime">--</div>
    <div class="kpi-sub" id="mode">--</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">P&amp;L</div>
    <div class="kpi-value" id="pnl">--</div>
    <div class="kpi-sub" id="pnl-sub">paper trading</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value" id="winrate">--</div>
    <div class="kpi-sub" id="winrate-sub">--</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Trades</div>
    <div class="kpi-value" id="trades">--</div>
    <div class="kpi-sub" id="trades-sub">executed</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Markets</div>
    <div class="kpi-value accent" id="markets-kpi">--</div>
    <div class="kpi-sub" id="beliefs-sub">-- beliefs</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Agents</div>
    <div class="kpi-value" id="agents">--</div>
    <div class="kpi-sub" id="agents-sub">active</div>
  </div>
</div>

<div class="main-grid">
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <div class="panel-title-accent"></div>
        Live Agent Thoughts
      </div>
      <div class="panel-meta" id="thought-count">0 events</div>
    </div>
    <div class="panel-body" id="thoughts">
      <div class="empty">Waiting for agents to think…</div>
    </div>
  </div>

  <div style="display: flex; flex-direction: column; gap: 20px;">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="panel-title-accent" style="background: var(--green);"></div>
          Top Edges
        </div>
        <div class="panel-meta">composite score</div>
      </div>
      <div class="row head edge-row">
        <div>Market</div>
        <div class="num">Edge</div>
        <div class="num">Conv</div>
        <div class="num">Score</div>
        <div class="num">Risk</div>
      </div>
      <div class="panel-body" id="edges" style="max-height: 240px;">
        <div class="empty">No edges yet</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="panel-title-accent" style="background: var(--cyan);"></div>
          Top Markets
        </div>
        <div class="panel-meta">by attention</div>
      </div>
      <div class="row head market-row">
        <div>Market</div>
        <div class="num">Price</div>
        <div class="num">Fair</div>
        <div class="num">Regime</div>
        <div class="num">Score</div>
      </div>
      <div class="panel-body" id="markets" style="max-height: 240px;">
        <div class="empty">No markets yet</div>
      </div>
    </div>
  </div>
</div>

<div class="bottom-grid">
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <div class="panel-title-accent" style="background: var(--cyan);"></div>
        Agent Status
      </div>
      <div class="panel-meta" id="agent-meta">--</div>
    </div>
    <div id="agent-grid" class="agent-grid"></div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <div class="panel-title-accent" style="background: var(--amber);"></div>
        System Health
      </div>
      <div class="panel-meta">world model</div>
    </div>
    <div style="padding: 16px 20px;">
      <div class="health-grid" style="grid-template-columns: repeat(4, 1fr);">
        <div class="health-cell">
          <div class="health-label">Markets</div>
          <div class="health-value" id="h_markets">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Beliefs</div>
          <div class="health-value" id="h_beliefs">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Correlations</div>
          <div class="health-value" id="h_corr">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">News (1h)</div>
          <div class="health-value" id="h_news">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Events (5m)</div>
          <div class="health-value" id="h_events">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Surprises</div>
          <div class="health-value" id="h_surprises">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Conflicts</div>
          <div class="health-value" id="h_conflicts">--</div>
        </div>
        <div class="health-cell">
          <div class="health-label">Edges</div>
          <div class="health-value" id="h_edges">--</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
function fmt(n, d=2) { return n == null ? "--" : Number(n).toFixed(d); }
function fmtPct(n, d=1) { return n == null ? "--" : Number(n).toFixed(d) + "%"; }
function shorten(s, n=50) { return !s ? "" : (s.length > n ? s.slice(0,n)+"…" : s); }

async function refresh() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    if (!d.ok) return;
    const s = d.status, perf = s.performance || {}, port = s.portfolio || {};

    // KPIs
    document.getElementById("uptime").textContent = (perf.uptime_hours || 0).toFixed(1) + "h";
    document.getElementById("mode").textContent = (s.mode || "--").toLowerCase();

    const pnlEl = document.getElementById("pnl");
    const pnl = port.pnl || 0;
    pnlEl.textContent = (pnl >= 0 ? "+$" : "-$") + Math.abs(pnl).toFixed(0);
    pnlEl.className = "kpi-value " + (pnl >= 0 ? "green" : "red");

    document.getElementById("winrate").textContent = (perf.win_rate || 0).toFixed(0) + "%";
    document.getElementById("winrate-sub").textContent =
      (perf.consecutive_wins ? perf.consecutive_wins + "W streak" :
       perf.consecutive_losses ? perf.consecutive_losses + "L streak" : "—");

    document.getElementById("trades").textContent = perf.total_trades || 0;

    const wm = s.world_model || {};
    document.getElementById("markets-kpi").textContent = wm.markets_tracked || 0;
    document.getElementById("beliefs-sub").textContent = (wm.total_beliefs || 0) + " beliefs";

    const allAgents = s.agents || [];
    const activeAgents = allAgents.filter(a => a.runs > 0).length;
    document.getElementById("agents").textContent = activeAgents;
    document.getElementById("agents-sub").textContent = "of " + allAgents.length + " total";

    // Health cells
    document.getElementById("h_markets").textContent = wm.markets_tracked || 0;
    document.getElementById("h_beliefs").textContent = wm.total_beliefs || 0;
    document.getElementById("h_corr").textContent = wm.correlations || 0;
    document.getElementById("h_news").textContent = wm.recent_news || 0;
    document.getElementById("h_events").textContent = (s.event_bus || {}).recent_events_5min || 0;
    document.getElementById("h_surprises").textContent = wm.surprises || 0;
    document.getElementById("h_conflicts").textContent = wm.conflicts || 0;
    document.getElementById("h_edges").textContent = wm.active_edges || 0;

    // Thoughts
    const thoughts = (d.thoughts || []).slice().reverse();
    document.getElementById("thought-count").textContent = thoughts.length + " events";
    if (thoughts.length > 0) {
      document.getElementById("thoughts").innerHTML = thoughts.map(t => {
        const cls = t.surprise ? "thought surprise" : "thought";
        const edgeCls = (t.edge || 0) > 0 ? "edge-pos" : "edge-neg";
        const edgeStr = (t.edge != null ? (t.edge > 0 ? "+" : "") + t.edge.toFixed(1) + "%" : "—");
        const probPct = (t.probability || 0) * 100;
        return `<div class="${cls}">
          <span class="thought-agent-tag agent-${t.agent}">${t.agent}</span>
          <div class="thought-body">
            <div class="thought-question">${shorten(t.question, 60)}</div>
            <div class="thought-meta">
              <span>market <span class="tabular">${(t.market_price*100).toFixed(0)}%</span></span>
              <span class="sep">·</span>
              <span class="${edgeCls}">${edgeStr}</span>
              <span class="sep">·</span>
              <span class="conf">conf <span class="tabular">${(t.confidence*100).toFixed(0)}%</span></span>
            </div>
          </div>
          <div class="thought-prob-bar">
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width:${probPct}%"></div>
            </div>
            <div class="prob-bar-label">${probPct.toFixed(0)}%</div>
          </div>
        </div>`;
      }).join("");
    }

    // Markets
    const attn = s.attention || {};
    const topMarkets = (attn.top_attention_markets || []).slice(0, 8);
    if (topMarkets.length > 0) {
      document.getElementById("markets").innerHTML = topMarkets.map(m => {
        return `<div class="row market-row">
          <div class="question">${shorten(m.question || "", 36)}</div>
          <div class="num">${m.price != null ? (m.price*100).toFixed(0)+"%" : "—"}</div>
          <div class="num">${m.consensus != null ? (m.consensus*100).toFixed(0)+"%" : "—"}</div>
          <div class="num"><span class="tag dim">${shorten(m.regime || "—", 6)}</span></div>
          <div class="num">${fmt(m.total_score, 1)}</div>
        </div>`;
      }).join("");
    }

    // Edges
    const recentTheses = s.recent_theses || [];
    if (recentTheses.length > 0) {
      document.getElementById("edges").innerHTML = recentTheses.slice(0, 8).map(t => {
        const edgeCls = (t.edge_pct || 0) > 0 ? "pos" : "neg";
        const rating = t.adversarial_rating || "—";
        const riskCls = rating === "safe" ? "green"
                      : (rating === "dangerous" || rating === "abort") ? "red" : "amber";
        return `<div class="row edge-row">
          <div class="question">${shorten(t.market_question || "", 36)}</div>
          <div class="num edge-pct ${edgeCls}">${fmtPct(t.edge_pct)}</div>
          <div class="num">${fmt(t.conviction, 0)}</div>
          <div class="num">${fmt(t.composite_score, 1)}</div>
          <div class="num"><span class="tag ${riskCls}">${shorten(rating, 6)}</span></div>
        </div>`;
      }).join("");
    }

    // Agent grid
    if (allAgents.length > 0) {
      document.getElementById("agent-meta").textContent = activeAgents + " / " + allAgents.length + " active";
      document.getElementById("agent-grid").innerHTML = allAgents.map(a => {
        const cls = a.errors > 0 ? "agent-card error"
                  : a.runs === 0 ? "agent-card idle"
                  : "agent-card";
        const errCls = a.errors > 0 ? "errs has" : "errs";
        return `<div class="${cls}">
          <div class="agent-status-dot"></div>
          <div class="agent-card-name">${a.name}</div>
          <div class="agent-card-stats">
            <span class="runs">${a.runs} runs</span>
            <span class="${errCls}">${a.errors} err</span>
          </div>
        </div>`;
      }).join("");
    }
  } catch (e) { console.error(e); }
}

refresh();
setInterval(refresh, 2000);
</script>

</body>
</html>"""