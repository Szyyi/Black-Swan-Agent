"""
Meta coordinator v3: fully fixed for Windows, no hangs, continuous output.

Fixes:
- 5 min refresh interval (was 60s, causing infinite reload loop)
- stdout.flush() on all prints (Windows buffers output)
- Agents are NOT killed/restarted on refresh
- Status prints every 20 seconds
- Protection against overlapping refreshes
"""

from __future__ import annotations

import asyncio
import datetime
import sys
import time
from collections import deque
from statistics import mean

import structlog

from agent.config import AgentConfig
from agent.data.smarkets_client import SmarketsClient
from agent.execution.engine import ExecutionEngine, signal_to_order
from agent.models import Market, MarketOutcome, OrderStatus, Signal, Side, StrategyType
from agent.risk.manager import RiskManager
from agent.swarm.agents import (
    ContrarianAgent,
    CorrelationDetectiveAgent,
    EdgeStackerAgent,
    MarketScannerAgent,
    MomentumAgent,
    NewsScoutAgent,
    OddsArbitrageAgent,
    ProbabilityEstimatorAgent,
    SocialSignalsAgent,
    SportsIntelligenceAgent,
    SwarmAgent,
    WebResearchAgent,
)
from agent.swarm.world_model import WorldModel
from agent.swarm.adversarial import AdversarialAgent
from agent.swarm.attention import AttentionAllocationEngine
from agent.swarm.event_bus import (
    EventBus, create_price_spike_detector,
    create_news_detector, create_surprise_detector,
)
from agent.swarm.metalearning import MetalearningSystem, TradeOutcome
from agent.swarm.thesis import ThesisGenerator
from agent.backtest.recorder import SessionRecorder
from agent.dashboard.server import run_dashboard


logger = structlog.get_logger()


def out(msg: str):
    """Print with flush — critical for Windows terminal."""
    print(msg)
    sys.stdout.flush()


class PerformanceTracker:
    def __init__(self, window: int = 50):
        self.recent_trades: deque[dict] = deque(maxlen=window)
        self.session_start: float = time.time()
        self.total_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.best_agents: dict[str, list[float]] = {}

    def record_trade(self, pnl: float, agents: list[str], market: str):
        self.recent_trades.append({
            "pnl": pnl, "agents": agents, "market": market, "timestamp": time.time(),
        })
        self.total_pnl += pnl
        self.peak_pnl = max(self.peak_pnl, self.total_pnl)
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        for agent in agents:
            if agent not in self.best_agents:
                self.best_agents[agent] = []
            self.best_agents[agent].append(pnl)

    @property
    def win_rate(self) -> float:
        if not self.recent_trades:
            return 0.0
        return sum(1 for t in self.recent_trades if t["pnl"] > 0) / len(self.recent_trades)

    @property
    def confidence_multiplier(self) -> float:
        if self.consecutive_losses >= 5:
            return 0.3
        elif self.consecutive_losses >= 3:
            return 0.5
        elif self.consecutive_losses >= 2:
            return 0.7
        elif self.consecutive_wins >= 5:
            return 1.3
        elif self.consecutive_wins >= 3:
            return 1.15
        return 1.0

    def get_agent_ranking(self) -> list[tuple[str, float]]:
        rankings = []
        for agent, pnls in self.best_agents.items():
            if pnls:
                rankings.append((agent, mean(pnls)))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_summary(self) -> dict:
        uptime = time.time() - self.session_start
        return {
            "uptime_hours": round(uptime / 3600, 2),
            "total_trades": len(self.recent_trades),
            "total_pnl": round(self.total_pnl, 2),
            "peak_pnl": round(self.peak_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "confidence_multiplier": self.confidence_multiplier,
            "agent_ranking": self.get_agent_ranking(),
        }


class MetaCoordinator:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.running = False

        self.world = WorldModel()
        self.smarkets = SmarketsClient()
        self._smarkets_username = config.smarkets_username
        self._smarkets_password = config.smarkets_password

        self.risk = RiskManager(config.risk, total_capital=config.risk.max_total_exposure_usd)
        self.executor = ExecutionEngine(config)
        self.performance = PerformanceTracker()
        self.agents: list[SwarmAgent] = self._create_agents()
        self.event_bus = EventBus(cooldown_seconds=30)
        self.attention = AttentionAllocationEngine(self.world)
        from agent.swarm import agents as agents_module
        agents_module.set_attention_engine(self.attention)
        self.metalearning = MetalearningSystem(
            persist_path="./data/metalearning.json"
        )
        self.thesis_gen = ThesisGenerator(self.world)
        self.recorder = SessionRecorder()

        # Give thesis generator access to adversarial agent
        # Cache adversarial agent reference (used by decision cycle and thesis gen)
        self._adversarial_ref = next(
            (a for a in self.agents if getattr(a, 'name', '') == 'adversarial'),
            None
        )
        if self._adversarial_ref:
            self.thesis_gen = ThesisGenerator(self.world, self._adversarial_ref)
        
        # Wire up event-driven triggers
        self.world.register_event_callback(
            "price_update",
            create_price_spike_detector(self.event_bus, threshold_pct=5.0)
        )
        self.world.register_event_callback(
            "news_impact",
            create_news_detector(self.event_bus, urgency_threshold=0.7)
        )
        self.world.register_event_callback(
            "surprise",
            create_surprise_detector(self.event_bus, shift_threshold=0.10)
        )

        # Register all agents with event bus
        for agent in self.agents:
            self.event_bus.register_agent(agent.name, agent)


        self._markets: list[Market] = []
        self._last_market_refresh: float = 0
        self._market_refresh_interval: float = 300  # 5 minutes — not 60!
        self._decision_interval: float = 30
        self._last_decision: float = 0
        self._trades_executed: int = 0
        self._signals_seen: int = 0
        self._signals_approved: int = 0
        self._tick_count: int = 0
        self._refreshing: bool = False  # Prevent overlapping refreshes

    def _create_agents(self) -> list[SwarmAgent]:
        agents: list[SwarmAgent] = []
        cfg = self.config
        agents.append(MarketScannerAgent(self.world, self.smarkets))
        agents.append(EdgeStackerAgent(self.world))
        agents.append(MomentumAgent(self.world))
        if cfg.anthropic_api_key:
            agents.append(NewsScoutAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
                news_sources=cfg.sentiment.news_sources if cfg.sentiment.enabled else [],
            ))
            agents.append(ProbabilityEstimatorAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
                ensemble_size=cfg.event_probability.ensemble_models,
            ))
            agents.append(CorrelationDetectiveAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
            ))
            agents.append(ContrarianAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
            ))
            agents.append(SocialSignalsAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
            ))
            agents.append(SportsIntelligenceAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
                football_data_key=getattr(cfg, 'football_data_api_key', ''),
            ))
            agents.append(WebResearchAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
            ))
        odds_key = getattr(cfg, 'odds_api_key', '')
        if odds_key:
            agents.append(OddsArbitrageAgent(self.world, odds_api_key=odds_key))
        if cfg.anthropic_api_key:
            agents.append(AdversarialAgent(
                self.world, cfg.anthropic_api_key, cfg.anthropic_model,
            ))
        out(f"  [INIT] {len(agents)} agents created")
        return agents

    # ── Main Loop ──────────────────────────────────────

    async def run(self):
        self.running = True
        await self.executor.initialize()
        await self.recorder.start()

        out("")
        out("=" * 60)
        out("  AI TRADING SWARM — Starting up")
        out(f"  Mode: {self.config.mode} | Platform: {self.config.platform}")
        out(f"  Capital: ${self.config.risk.max_total_exposure_usd:,.0f}")
        out(f"  Agents: {len(self.agents)}")
        out("=" * 60)
        out("")
        out("  [1/3] Logging into Smarkets...")

        await self._refresh_markets()

        out(f"  [2/3] Loaded {len(self._markets)} markets with prices")
        out("  [3/3] Launching agents...")
        out("")

        # Start agents ONCE — don't restart them on every refresh
        agent_tasks = self._start_agents()

        dashboard_task = asyncio.create_task(run_dashboard(self, port=8000))


        out("=" * 60)
        out("  SYSTEM LIVE — agents are running")
        out("  Status updates every 20 seconds")
        out("  Press Ctrl+C to stop")
        out("=" * 60)
        out("")

        try:
            while self.running:
                self._tick_count += 1

                # Refresh markets every 5 min (but don't kill agents)
                if (time.time() - self._last_market_refresh > self._market_refresh_interval
                        and not self._refreshing):
                    self._refreshing = True
                    try:
                        await self._refresh_markets()
                    finally:
                        self._refreshing = False

                # Decision cycle
                if time.time() - self._last_decision >= self._decision_interval:
                    try:
                        await self._decision_cycle()
                    except Exception as e:
                        out(f"  [WARN] Decision cycle error: {e}")
                    self._last_decision = time.time()

                # Status every 20 seconds
                if self._tick_count % 20 == 0:
                    self._log_status()

                await asyncio.sleep(1)

        except KeyboardInterrupt:
            out("\n  Shutting down gracefully...")
        except Exception as e:
            out(f"\n  [CRITICAL] {e}")
            raise
        finally:
            for agent in self.agents:
                agent.stop()
            for task in agent_tasks:
                task.cancel()
            dashboard_task.cancel()
            await self.shutdown()

    def _start_agents(self) -> list[asyncio.Task]:
        return [asyncio.create_task(agent.start(self._markets)) for agent in self.agents]

    # ── Decision Cycle ─────────────────────────────────

    async def _decision_cycle(self):
        edges = self.world.compute_edges(min_edge_pct=3.0)
        if not edges:
            return

        for edge in edges[:5]:
            self._signals_seen += 1
            timing = self.world.get_timing_signals(edge.market_id)
            urgency = max((s.urgency for s in timing), default=0.5)

            belief_summary = self.world.get_belief_summary(edge.market_id)
            belief_ages = [b.get("age_seconds", 999) for b in belief_summary.get("beliefs", [])]
            avg_age = mean(belief_ages) if belief_ages else 999
            freshness = max(0.3, 1.0 - (avg_age / 600))

            num_agents = len(edge.contributing_agents)
            agreement_bonus = min(1.5, 0.7 + (num_agents * 0.15))

            correlated = self.world.get_correlated_markets(edge.market_id)
            corr_penalty = 0.0
            for corr in correlated:
                other_id = corr.market_b_id if corr.market_a_id == edge.market_id else corr.market_a_id
                if other_id in self.risk.positions:
                    corr_penalty += corr.strength * 0.3

            # Metalearning weight: boost proven combos, dampen bad ones
            meta_weight = self.metalearning.get_combo_weight(
                agents=edge.contributing_agents,
                category=self.world._market_categories.get(edge.market_id, ""),
                regime=edge.regime,
                hour=time.localtime().tm_hour,
            )

            # Adversarial gate: reduce score if risk is high
            adversarial_mult = 1.0
            if self._adversarial_ref:
                risk = self._adversarial_ref.get_risk_rating(edge.market_id)
                if risk == "dangerous":
                    adversarial_mult = 0.4
                elif risk == "abort":
                    adversarial_mult = 0.0  # Block the trade entirely
                elif risk == "proceed_with_caution":
                    adversarial_mult = 0.7

            composite = (
                edge.edge_pct * edge.confidence * urgency * freshness
                * agreement_bonus * (1 - corr_penalty)
                * self.performance.confidence_multiplier
                * meta_weight
                * adversarial_mult
            )

            # Record every computed edge — even ones that don't trade.
            # This is critical for measuring whether the threshold is correct.
            self.recorder.record_edge(edge, composite_score=composite)

            if composite < 1.5:
                continue

            market = next((m for m in self._markets if m.condition_id == edge.market_id), None)
            if not market or len(market.token_ids) < 2:
                continue

            if edge.direction == "BUY_YES":
                token_id, outcome = market.token_ids[0], MarketOutcome.YES
            else:
                token_id, outcome = market.token_ids[1], MarketOutcome.NO

            signal = Signal(
                strategy=StrategyType.EVENT_PROBABILITY,
                market_id=edge.market_id, token_id=token_id,
                side=Side.BUY, outcome=outcome,
                confidence=edge.confidence, edge_pct=edge.edge_pct,
                fair_value=edge.fair_value, market_price=edge.market_price,
                suggested_size_usd=self._kelly_size(edge, urgency),
                reasoning=f"Swarm [{', '.join(edge.contributing_agents)}] score={composite:.1f}",
                metadata={"composite_score": composite, "contributing_agents": edge.contributing_agents},
            )

            approved, reason = self.risk.check_signal(signal)
            if not approved:
                continue
            self._signals_approved += 1

            sized = self.risk.size_order(signal, 1.0) * self.performance.confidence_multiplier
            sized = max(0, min(sized, self.risk.available_capital * 0.15))
            if sized < 1.0:
                continue

            order = signal_to_order(signal, sized)
            result = await self.executor.submit_order(order)

            if result.status == OrderStatus.FILLED:
                try:
                    self.risk.on_fill(result, edge.market_question)
                except Exception:
                    pass
                self._trades_executed += 1
                estimated_pnl = sized * (edge.edge_pct / 100)
                self.performance.record_trade(
                    estimated_pnl, edge.contributing_agents, edge.market_question
                )

                # Record trade for later analysis
                adv_rating = ""
                if self._adversarial_ref:
                    try:
                        adv_rating = self._adversarial_ref.get_risk_rating(edge.market_id) or ""
                    except Exception:
                        pass
                self.recorder.record_trade(
                    trade_id=result.exchange_order_id or order.id,
                    market_id=edge.market_id,
                    market_question=edge.market_question,
                    side=signal.side.value if hasattr(signal.side, "value") else str(signal.side),
                    size_usd=sized,
                    entry_price=edge.market_price,
                    edge_pct=edge.edge_pct,
                    composite_score=composite,
                    contributing_agents=edge.contributing_agents,
                    regime=edge.regime,
                    adversarial_rating=adv_rating,
                    kelly_fraction=self._kelly_size(edge, urgency),
                    confidence_multiplier=self.performance.confidence_multiplier,
                )

                # Terminal output with thesis
                out(thesis.to_terminal_summary())

                # === Record for metalearning ===
                import datetime
                # === Record for metalearning ===
                self.metalearning.record_outcome(TradeOutcome(
                    market_id=edge.market_id,
                    market_question=edge.market_question,
                    market_category=self.world._market_categories.get(edge.market_id, ""),
                    contributing_agents=edge.contributing_agents,
                    edge_pct=edge.edge_pct,
                    conviction=edge.conviction,
                    composite_score=composite,
                    regime=edge.regime,
                    hour_of_day=datetime.datetime.now().hour,
                    pnl=estimated_pnl,
                    was_profitable=estimated_pnl > 0,
                ))

    def _kelly_size(self, edge, urgency: float) -> float:
        win_prob = edge.fair_value
        loss_prob = 1 - win_prob
        if loss_prob <= 0 or edge.market_price <= 0:
            return 0
        b = (1 / edge.market_price) - 1
        if b <= 0:
            return 0
        kelly = (b * win_prob - loss_prob) / b
        kelly = max(0, kelly * 0.25)
        kelly *= (0.5 + urgency * 0.5)
        max_bet = self.risk.available_capital * kelly
        return min(max_bet, self.config.risk.max_total_exposure_usd * 0.10)

    # ── Market Refresh ─────────────────────────────────

    async def _refresh_markets(self):
        try:
            if not self.smarkets.session_token and self._smarkets_username:
                out("  [LOGIN] Connecting to Smarkets...")
                await self.smarkets.login(self._smarkets_username, self._smarkets_password)

            domains = ["politics", "current_affairs", "entertainment",
                       "football", "horse_racing", "tennis"]
            self._markets = await self.smarkets.get_markets_as_agent_models(
                domains=domains, limit=100
            )
            self._last_market_refresh = time.time()

            prices_loaded = 0
            for market in self._markets:
                price_found = None
                for cid in market.token_ids:
                    price_found = self.smarkets.get_initial_price(cid)
                    if price_found is not None:
                        break

                # Only register markets we have real prices for.
                # A 0.5 placeholder poisons every belief computed against it
                # because agents would compute fake "edges" vs a fake midpoint.
                if price_found is None:
                    continue

                self.world.update_market_price(
                    market.condition_id,
                    price_found,
                    question=market.question,
                    volume=market.volume,
                    category=market.category,
                )
                prices_loaded += 1

            out(f"  [DATA] {len(self._markets)} markets, {prices_loaded} with prices")

        except Exception as e:
            out(f"  [ERROR] Market refresh failed: {e}")
            if not self._markets:
                self._markets = []

    # ── Status ─────────────────────────────────────────

    def _log_status(self):
        world = self.world.get_status()
        perf = self.performance.get_summary()
        portfolio = self.risk.get_snapshot()

        mt = world.get('markets_tracked', 0)
        bl = world.get('total_beliefs', 0)
        ed = world.get('active_edges', 0)
        cr = world.get('correlations', 0)
        nw = world.get('recent_news', 0)

        out("")
        out(f"  --- STATUS ({perf['uptime_hours']:.1f}h) ---")
        out(f"  Markets: {mt} | Beliefs: {bl} | Edges: {ed} | Corr: {cr} | News: {nw}")
        out(f"  Trades: {self._trades_executed} | PnL: ${portfolio.total_pnl:+.2f} | Win: {perf['win_rate']}%")

        # Show agent activity on one line each
        active = 0
        for a in self.agents:
            s = a.get_status()
            if s["runs"] > 0:
                active += 1
        out(f"  Active agents: {active}/{len(self.agents)}")

        # Show any agents with errors
        for a in self.agents:
            s = a.get_status()
            if s["errors"] > 0:
                out(f"    {s['name']}: {s['errors']} errors, {s['runs']} runs")

        # Event bus status
        bus_status = self.event_bus.get_status()
        if bus_status["recent_events_5min"] > 0:
            out(f"  Events (5min): {bus_status['recent_events_5min']} | "
                f"Types: {bus_status['event_types']}")

        # Attention highlights
        attn = self.attention.get_attention_report()
        if attn["top_attention_markets"]:
            top = attn["top_attention_markets"][0]
            out(f"  Top attention: '{top['question']}' (score: {top['total_score']:.1f})")

        # Recorder status
        rec = self.recorder.get_status()
        s = rec["stats"]
        out(f"  Recorded: {s['markets']} mkts | {s['beliefs']} beliefs | "
            f"{s['edges']} edges | {s['trades']} trades")

        out("")

    def get_status(self) -> dict:
        portfolio = self.risk.get_snapshot()
        return {
            "running": self.running,
            "event_bus": self.event_bus.get_status(),
            "attention": self.attention.get_attention_report(),
            "metalearning": self.metalearning.get_report(),
            "recent_theses": self.thesis_gen.get_recent_theses(5),
            "decay_profiles": self.world.decay_engine.get_status(),
            "mode": self.config.mode,
            "platform": self.config.platform,
            "performance": self.performance.get_summary(),
            "agents": [a.get_status() for a in self.agents],
            "world_model": self.world.get_status(),
            "portfolio": {
                "capital": portfolio.total_capital,
                "available": portfolio.available_capital,
                "exposure": portfolio.total_exposure,
                "positions": len(portfolio.positions),
                "pnl": portfolio.total_pnl,
                "drawdown": portfolio.max_drawdown,
            },
        }

    async def shutdown(self):
        out("  [SHUTDOWN] Cleaning up...")
        await self.recorder.stop()
        await self.executor.cancel_all()
        await self.smarkets.close()
        await self.executor.close()
        perf = self.performance.get_summary()
        snapshot = self.risk.get_snapshot()
        out("")
        out("=" * 60)
        out("  SESSION REPORT")
        out("=" * 60)
        out(f"  Uptime: {perf['uptime_hours']:.1f} hours")
        out(f"  Trades: {perf['total_trades']}")
        out(f"  P&L: ${perf['total_pnl']:+.2f}")
        out(f"  Win Rate: {perf['win_rate']}%")
        out(f"  Max Drawdown: {snapshot.max_drawdown:.1f}%")
        rankings = perf['agent_ranking'][:3]
        if rankings:
            out(f"  Top agents: {rankings}")
        out("=" * 60)
        out("")
