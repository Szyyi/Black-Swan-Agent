"""
Terminal dashboard v2: live-updating, informative, easy to read.
Uses Rich library for beautiful terminal rendering.
Auto-refreshes every status cycle with color-coded everything.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from agent.swarm.world_model import WorldModel
    from agent.risk.manager import RiskManager

console = Console()

# ── Helpers ────────────────────────────────────────────

def _pnl_text(value: float, prefix: str = "$") -> Text:
    if value > 0:
        return Text(f"+{prefix}{value:,.2f}", style="bold green")
    elif value < 0:
        return Text(f"-{prefix}{abs(value):,.2f}", style="bold red")
    return Text(f"{prefix}0.00", style="dim")


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values or len(values) < 2:
        return "  " + "─" * width
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    # Sample to fit width
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    chars = []
    for v in sampled:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        chars.append(blocks[idx])
    color = "green" if values[-1] >= values[0] else "red"
    return f"[{color}]{''.join(chars)}[/]"


def _time_ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m ago"
    return f"{seconds/3600:.1f}h ago"


def _regime_badge(regime: str) -> Text:
    colors = {
        "stable": "green",
        "trending_up": "cyan",
        "trending_down": "yellow",
        "volatile": "red",
        "unknown": "dim",
    }
    symbols = {
        "stable": "● STABLE",
        "trending_up": "▲ TREND UP",
        "trending_down": "▼ TREND DOWN",
        "volatile": "◆ VOLATILE",
        "unknown": "○ UNKNOWN",
    }
    return Text(symbols.get(regime, regime), style=colors.get(regime, "dim"))


# ── Main Dashboard ─────────────────────────────────────

def print_dashboard(agents: list, world, risk, performance,
                    trades_executed: int = 0, signals_seen: int = 0,
                    signals_approved: int = 0):
    """Render the full dashboard to terminal."""
    console.clear()
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # ═══ HEADER ═══
    header = Table(show_header=False, box=None, expand=True, padding=0)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text("  AI TRADING SWARM", style="bold bright_white"),
        Text(f"Smarkets · Paper Mode · {now}  ", style="dim"),
    )
    console.print(Panel(header, style="bright_blue", box=box.DOUBLE))

    # ═══ TOP ROW: Portfolio + Performance ═══
    top = Table.grid(expand=True, padding=(0, 1))
    top.add_column(ratio=1)
    top.add_column(ratio=1)
    top.add_row(
        _portfolio_panel(risk, performance),
        _performance_panel(performance, trades_executed, signals_seen, signals_approved),
    )
    console.print(top)

    # ═══ MIDDLE ROW: Agents + World Model ═══
    mid = Table.grid(expand=True, padding=(0, 1))
    mid.add_column(ratio=1)
    mid.add_column(ratio=1)
    mid.add_row(
        _agents_panel(agents),
        _world_model_panel(world),
    )
    console.print(mid)

    # ═══ BOTTOM: Edges ═══
    console.print(_edges_panel(world))

    # ═══ FOOTER ═══
    console.print(
        Text("  Ctrl+C to stop · Refreshes every 60s · Agents run on independent schedules",
             style="dim"),
    )


# ── Panels ─────────────────────────────────────────────

def _portfolio_panel(risk, performance) -> Panel:
    snap = risk.get_snapshot()
    perf = performance.get_summary()

    t = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    t.add_column("label", style="dim", width=14)
    t.add_column("value", justify="right")

    # Capital section
    t.add_row("Capital", Text(f"${snap.total_capital:,.0f}", style="bold"))
    t.add_row("Available", Text(f"${snap.available_capital:,.0f}", style="bright_white"))
    t.add_row("Exposure", Text(f"${snap.total_exposure:,.0f}", style="yellow"))
    t.add_row("", "")  # spacer

    # P&L section
    t.add_row("Total P&L", _pnl_text(snap.total_pnl))
    t.add_row("Daily P&L", _pnl_text(snap.daily_pnl))
    t.add_row("Max Drawdown", Text(f"{snap.max_drawdown:.1f}%",
              style="red" if snap.max_drawdown > 5 else "dim"))
    t.add_row("", "")

    # Positions
    t.add_row("Positions", Text(str(len(snap.positions)), style="bright_white"))
    t.add_row("Trades", Text(str(snap.total_trades), style="bright_white"))

    return Panel(t, title="[bold]Portfolio[/]", border_style="blue", box=box.ROUNDED)


def _performance_panel(performance, trades: int, seen: int, approved: int) -> Panel:
    perf = performance.get_summary()

    t = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    t.add_column("label", style="dim", width=14)
    t.add_column("value", justify="right")

    # Session info
    t.add_row("Uptime", Text(f"{perf['uptime_hours']:.1f} hours", style="bright_white"))
    t.add_row("Win Rate", Text(
        f"{perf['win_rate']}%",
        style="green" if perf['win_rate'] > 50 else "red" if perf['win_rate'] > 0 else "dim"
    ))
    t.add_row("", "")

    # Streak
    if perf["consecutive_wins"] > 0:
        streak = Text(f"W{perf['consecutive_wins']}", style="bold green")
    elif perf["consecutive_losses"] > 0:
        streak = Text(f"L{perf['consecutive_losses']}", style="bold red")
    else:
        streak = Text("—", style="dim")
    t.add_row("Streak", streak)

    # Size multiplier
    mult = perf["confidence_multiplier"]
    mult_style = "green" if mult > 1 else "red" if mult < 1 else "dim"
    t.add_row("Size Mult", Text(f"{mult:.2f}x", style=mult_style))
    t.add_row("", "")

    # Signals
    approval = f"{approved}/{seen}" if seen > 0 else "0/0"
    rate = f" ({approved/seen*100:.0f}%)" if seen > 0 else ""
    t.add_row("Signals", Text(f"{approval}{rate}", style="bright_white"))

    # Best agent
    rankings = perf.get("agent_ranking", [])
    if rankings:
        best_name = rankings[0][0].replace("_", " ").title()
        t.add_row("Best Agent", Text(best_name, style="cyan"))

    return Panel(t, title="[bold]Performance[/]", border_style="green", box=box.ROUNDED)


def _agents_panel(agents: list) -> Panel:
    t = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    t.add_column("Agent", width=22)
    t.add_column("Status", width=6, justify="center")
    t.add_column("Runs", width=4, justify="right")
    t.add_column("Last", width=7, justify="right")

    for agent in agents:
        s = agent.get_status()
        last = s.get("last_run_ago")

        # Status indicator
        if last is None:
            status = Text("WAIT", style="yellow")
        elif last < agent.interval_seconds * 2:
            status = Text("● OK", style="green")
        elif last < agent.interval_seconds * 5:
            status = Text("● SLOW", style="yellow")
        else:
            status = Text("● DOWN", style="red")

        # Clean name
        name = s["name"].replace("_", " ").title()
        if len(name) > 20:
            name = name[:18] + ".."

        # Error indicator
        if s["errors"] > 0:
            name_text = Text(f"{name} ", style="bright_white")
            name_text.append(f"({s['errors']}!)", style="red")
        else:
            name_text = Text(name, style="bright_white")

        t.add_row(
            name_text,
            status,
            str(s["runs"]),
            _time_ago(last) if last else "—",
        )

    return Panel(t, title="[bold]Agents[/]", border_style="cyan", box=box.ROUNDED)


def _world_model_panel(world) -> Panel:
    status = world.get_status()

    t = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    t.add_column("label", style="dim", width=16)
    t.add_column("value", justify="right")

    markets = status.get("markets_tracked", 0)
    t.add_row("Markets", Text(str(markets), style="bold bright_white"))
    t.add_row("Beliefs", Text(str(status.get("total_beliefs", 0)),
              style="cyan" if status.get("total_beliefs", 0) > 0 else "dim"))
    t.add_row("Active Edges", Text(str(status.get("active_edges", 0)),
              style="green bold" if status.get("active_edges", 0) > 0 else "dim"))
    t.add_row("Correlations", Text(str(status.get("correlations", 0)), style="bright_white"))
    t.add_row("News (1h)", Text(str(status.get("recent_news", 0)), style="bright_white"))
    t.add_row("Timing Signals", Text(str(status.get("active_timing_signals", 0)),
              style="yellow" if status.get("active_timing_signals", 0) > 0 else "dim"))
    t.add_row("", "")

    # Conflicts
    conflicts = status.get("conflicts", 0)
    if conflicts > 0:
        t.add_row("Conflicts", Text(f"{conflicts} markets", style="red bold"))

    # Regimes
    regimes = status.get("regimes", {})
    if regimes:
        regime_parts = []
        for regime, count in regimes.items():
            r = regime.replace("_", " ")
            regime_parts.append(f"{r}: {count}")
        t.add_row("Regimes", Text(" · ".join(regime_parts), style="dim"))

    # Trust rankings
    rankings = status.get("agent_trust_rankings", [])
    if rankings:
        top = rankings[0]
        t.add_row("Top Analyst",
                   Text(f"{top['agent'].replace('_',' ').title()} ({top['avg_accuracy']:.0%})",
                        style="cyan"))

    return Panel(t, title="[bold]World Model[/]", border_style="magenta", box=box.ROUNDED)


def _edges_panel(world) -> Panel:
    edges = world._edges[:8]

    if not edges:
        content = Table(show_header=False, box=None, expand=True)
        content.add_column()
        content.add_row(
            Text("\n  Warming up — agents are gathering data and building beliefs.\n"
                 "  Edges will appear once the probability estimator completes its first cycle.\n"
                 "  This typically takes 3-5 minutes after startup.\n",
                 style="dim italic"),
        )
        return Panel(content, title="[bold]Top Opportunities[/]",
                     border_style="yellow", box=box.ROUNDED)

    t = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    t.add_column("Market", width=36, no_wrap=True)
    t.add_column("Action", width=8, justify="center")
    t.add_column("Edge", width=6, justify="right")
    t.add_column("Conf", width=5, justify="right")
    t.add_column("Quality", width=7, justify="right")
    t.add_column("Regime", width=10)
    t.add_column("Agents", width=6, justify="center")

    for edge in edges:
        # Market name
        q = edge.market_question[:35] if edge.market_question else edge.market_id[:12]

        # Action badge
        if edge.direction == "BUY_YES":
            action = Text(" YES ", style="bold white on green")
        else:
            action = Text("  NO  ", style="bold white on red")

        # Edge color
        edge_val = edge.edge_pct
        if edge_val > 10:
            edge_style = "bold green"
        elif edge_val > 5:
            edge_style = "green"
        else:
            edge_style = "yellow"

        # Quality score
        quality_style = "bold green" if edge.quality_score > 20 else "bright_white"

        t.add_row(
            Text(q, style="bright_white"),
            action,
            Text(f"{edge_val:.1f}%", style=edge_style),
            Text(f"{edge.confidence:.0%}", style="bright_white"),
            Text(f"{edge.quality_score:.0f}", style=quality_style),
            _regime_badge(edge.regime),
            Text(str(len(edge.contributing_agents)), style="cyan"),
        )

    return Panel(t, title="[bold]Top Opportunities[/]", border_style="yellow", box=box.ROUNDED)