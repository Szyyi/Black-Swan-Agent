"""
Black Swan Dashboard — single-file FastAPI app with embedded HTML.

Reads from coordinator.get_status() and the world model's thought ring buffer.
Polls every 2 seconds. Localhost only.

Run via the coordinator (started automatically by agent.main).
Open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

logger = structlog.get_logger()


def create_app(coordinator: Any) -> FastAPI:
    app = FastAPI(title="Black Swan Dashboard", docs_url=None, redoc_url=None)

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
    """Start the dashboard server in the background. Localhost only."""
    app = create_app(coordinator)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    print(f"  [DASHBOARD] http://localhost:{port}", flush=True)
    await server.serve()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Black Swan</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0b0e14; color: #cbd5e1; padding: 20px; min-height: 100vh;
}
.mono { font-family: "SF Mono", "Consolas", monospace; }
h1 { color: #f8fafc; font-size: 22px; margin-bottom: 4px; letter-spacing: 0.5px; }
.subtitle { color: #64748b; font-size: 12px; margin-bottom: 20px; }
.header {
  display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px;
  background: #111827; border: 1px solid #1f2937; border-radius: 10px;
  padding: 16px 20px; margin-bottom: 20px;
}
.stat { text-align: center; }
.stat .label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.stat .value { font-size: 22px; color: #f1f5f9; font-weight: 600; margin-top: 4px; }
.stat .value.green { color: #4ade80; }
.stat .value.red { color: #f87171; }
.grid {
  display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px;
}
.panel {
  background: #111827; border: 1px solid #1f2937; border-radius: 10px;
  padding: 16px 20px; margin-bottom: 20px;
}
.panel h2 {
  font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px;
  margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;
}
.pulse {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #4ade80; box-shadow: 0 0 8px #4ade80;
  animation: pulse 1.6s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.thought {
  padding: 10px 12px; margin-bottom: 8px; background: #0f172a;
  border-left: 3px solid #334155; border-radius: 4px; font-size: 13px;
  animation: slideIn 0.3s ease;
}
@keyframes slideIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.thought .agent { color: #60a5fa; font-weight: 600; }
.thought .question { color: #e2e8f0; }
.thought .meta { color: #64748b; font-size: 11px; margin-top: 4px; }
.thought.surprise { border-left-color: #fbbf24; }
.thought.surprise .agent { color: #fbbf24; }

.edge-row, .market-row {
  display: grid; gap: 10px; padding: 8px 4px;
  border-bottom: 1px solid #1f2937; font-size: 12px; align-items: center;
}
.edge-row { grid-template-columns: 2fr 60px 70px 70px 80px; }
.market-row { grid-template-columns: 2fr 60px 60px 70px 70px; }
.edge-row:last-child, .market-row:last-child { border-bottom: none; }
.edge-row .question, .market-row .question {
  color: #e2e8f0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.num { text-align: right; }
.green { color: #4ade80; }
.red { color: #f87171; }
.amber { color: #fbbf24; }
.dim { color: #64748b; }
.tablehead { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.empty { color: #475569; font-style: italic; padding: 12px; text-align: center; font-size: 12px; }

.health {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 12px;
}
.health div { background: #0f172a; padding: 10px; border-radius: 6px; }
.health .label { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
.health .value { color: #e2e8f0; font-size: 16px; margin-top: 3px; }
.feed { max-height: 600px; overflow-y: auto; padding-right: 8px; }
.feed::-webkit-scrollbar { width: 6px; }
.feed::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
</style>
</head>
<body>
<h1>BLACK SWAN <span class="pulse"></span></h1>
<div class="subtitle mono">Multi-agent prediction market intelligence</div>

<div class="header">
  <div class="stat"><div class="label">Uptime</div><div class="value mono" id="uptime">--</div></div>
  <div class="stat"><div class="label">Mode</div><div class="value mono" id="mode" style="font-size:14px;">--</div></div>
  <div class="stat"><div class="label">P&amp;L</div><div class="value mono" id="pnl">--</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value mono" id="winrate">--</div></div>
  <div class="stat"><div class="label">Trades</div><div class="value mono" id="trades">--</div></div>
  <div class="stat"><div class="label">Agents</div><div class="value mono" id="agents">--</div></div>
</div>

<div class="grid">
  <div>
    <div class="panel">
      <h2>LIVE AGENT THOUGHTS <span class="pulse"></span></h2>
      <div class="feed" id="thoughts"><div class="empty">Waiting for agents to think...</div></div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>TOP EDGES</h2>
      <div class="edge-row tablehead">
        <div>MARKET</div><div class="num">EDGE</div><div class="num">CONV</div><div class="num">SCORE</div><div class="num">RISK</div>
      </div>
      <div id="edges"><div class="empty">No edges yet</div></div>
    </div>
    <div class="panel">
      <h2>TOP MARKETS</h2>
      <div class="market-row tablehead">
        <div>MARKET</div><div class="num">PRICE</div><div class="num">FAIR</div><div class="num">REGIME</div><div class="num">CONV</div>
      </div>
      <div id="markets"><div class="empty">No markets yet</div></div>
    </div>
  </div>
</div>

<div class="panel">
  <h2>SYSTEM HEALTH</h2>
  <div class="health">
    <div><div class="label">Markets Tracked</div><div class="value mono" id="h_markets">--</div></div>
    <div><div class="label">Beliefs</div><div class="value mono" id="h_beliefs">--</div></div>
    <div><div class="label">Correlations</div><div class="value mono" id="h_corr">--</div></div>
    <div><div class="label">Recent News</div><div class="value mono" id="h_news">--</div></div>
    <div><div class="label">Events 5min</div><div class="value mono" id="h_events">--</div></div>
    <div><div class="label">Surprises</div><div class="value mono" id="h_surprises">--</div></div>
    <div><div class="label">Conflicts</div><div class="value mono" id="h_conflicts">--</div></div>
    <div><div class="label">Recorded Beliefs</div><div class="value mono" id="h_rec">--</div></div>
  </div>
</div>

<script>
let lastThoughtIds = new Set();

function fmtNum(n, digits=2) { return n == null ? "--" : Number(n).toFixed(digits); }
function fmtPct(n, digits=1) { return n == null ? "--" : Number(n).toFixed(digits) + "%"; }
function shorten(s, n=50) { return !s ? "" : (s.length > n ? s.slice(0,n)+"..." : s); }

async function refresh() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    if (!d.ok) return;
    const s = d.status, perf = s.performance || {}, port = s.portfolio || {};

    document.getElementById("uptime").textContent = (perf.uptime_hours || 0).toFixed(1) + "h";
    document.getElementById("mode").textContent = (s.mode || "--").toUpperCase();
    const pnlEl = document.getElementById("pnl");
    pnlEl.textContent = "$" + (port.pnl || 0).toFixed(0);
    pnlEl.className = "value mono " + ((port.pnl || 0) >= 0 ? "green" : "red");
    document.getElementById("winrate").textContent = (perf.win_rate || 0).toFixed(0) + "%";
    document.getElementById("trades").textContent = perf.total_trades || 0;
    const activeAgents = (s.agents || []).filter(a => a.runs > 0).length;
    document.getElementById("agents").textContent = activeAgents + "/" + (s.agents || []).length;

    const wm = s.world_model || {};
    document.getElementById("h_markets").textContent = wm.markets_tracked || 0;
    document.getElementById("h_beliefs").textContent = wm.total_beliefs || 0;
    document.getElementById("h_corr").textContent = wm.correlations || 0;
    document.getElementById("h_news").textContent = wm.recent_news || 0;
    document.getElementById("h_events").textContent = (s.event_bus || {}).recent_events_5min || 0;
    document.getElementById("h_surprises").textContent = wm.surprises || 0;
    document.getElementById("h_conflicts").textContent = wm.conflicts || 0;

    const thoughts = (d.thoughts || []).slice().reverse();
    if (thoughts.length > 0) {
      document.getElementById("thoughts").innerHTML = thoughts.map(t => {
        const cls = t.surprise ? "thought surprise" : "thought";
        const edgeStr = t.edge != null ? (t.edge > 0 ? "+" : "") + t.edge.toFixed(1) + "%" : "";
        return `<div class="${cls}">
          <span class="agent">${t.agent}</span>
          <span class="question">→ ${shorten(t.question, 55)}</span>
          <div class="meta mono">thinks ${(t.probability*100).toFixed(0)}% · market ${(t.market_price*100).toFixed(0)}% · edge ${edgeStr} · conf ${(t.confidence*100).toFixed(0)}%</div>
        </div>`;
      }).join("");
    }

    const attn = s.attention || {};
    const topMarkets = (attn.top_attention_markets || []).slice(0, 8);
    if (topMarkets.length > 0) {
      document.getElementById("markets").innerHTML = topMarkets.map(m => {
        return `<div class="market-row">
          <div class="question">${shorten(m.question || "", 38)}</div>
          <div class="num mono">${m.price != null ? (m.price*100).toFixed(0)+"%" : "--"}</div>
          <div class="num mono">${m.consensus != null ? (m.consensus*100).toFixed(0)+"%" : "--"}</div>
          <div class="num mono dim" style="font-size:10px;">${shorten(m.regime || "", 8)}</div>
          <div class="num mono">${fmtNum(m.total_score, 1)}</div>
        </div>`;
      }).join("");
    }

    const recentTheses = s.recent_theses || [];
    if (recentTheses.length > 0) {
      document.getElementById("edges").innerHTML = recentTheses.slice(0, 8).map(t => {
        const edgeCls = (t.edge_pct || 0) > 0 ? "green" : "red";
        const riskCls = t.adversarial_rating === "safe" ? "green"
                      : t.adversarial_rating === "dangerous" ? "red"
                      : t.adversarial_rating === "abort" ? "red" : "amber";
        return `<div class="edge-row">
          <div class="question">${shorten(t.market_question || "", 38)}</div>
          <div class="num mono ${edgeCls}">${fmtPct(t.edge_pct)}</div>
          <div class="num mono">${fmtNum(t.conviction, 0)}</div>
          <div class="num mono">${fmtNum(t.composite_score, 1)}</div>
          <div class="num mono ${riskCls}" style="font-size:10px;">${shorten(t.adversarial_rating || "--", 7)}</div>
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