"""
Natural Language Trade Thesis — human-readable trade explanations.

For every trade the swarm executes, generates a structured thesis:
- WHAT: What are we trading and in which direction
- WHY: Which agents contributed and what evidence they used
- HOW: Position sizing logic, Kelly fraction, confidence multiplier
- RISK: Adversarial pre-mortem, correlated positions, kill conditions
- CONTEXT: Market regime, recent news, related markets

Output format is designed for:
1. Terminal display (compact summary)
2. Web dashboard (full structured thesis)
3. Trade log (archived for metalearning)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from agent.swarm.world_model import WorldModel

logger = structlog.get_logger()


@dataclass
class TradeThesis:
    """Complete human-readable thesis for a trade."""
    # === Identity ===
    trade_id: str
    market_id: str
    market_question: str

    # === Action ===
    direction: str           # BUY_YES / BUY_NO
    entry_price: float
    fair_value: float
    edge_pct: float
    size_usd: float

    # === Intelligence ===
    contributing_agents: list[str]
    agent_views: list[dict]  # [{agent, probability, confidence, reasoning}]
    consensus_probability: float
    consensus_confidence: float
    conviction_score: float

    # === Evidence ===
    key_evidence: list[str]
    news_context: list[str]
    correlation_context: list[str]

    # === Risk ===
    risk_rating: str         # from adversarial
    pre_mortem_scenario: str  # most likely failure
    kill_condition: str       # when to exit
    correlated_positions: list[str]
    kelly_fraction: float
    confidence_multiplier: float

    # === Context ===
    market_regime: str
    entropy: float
    surprise_factor: float
    composite_score: float

    # === Meta ===
    timestamp: float = field(default_factory=time.time)

    def to_terminal_summary(self) -> str:
        """Compact 4-line summary for terminal output."""
        agents_str = ", ".join(self.contributing_agents[:4])
        risk_icon = {
            "safe": "OK", "proceed_with_caution": "!",
            "dangerous": "!!", "abort": "XXX", "unreviewed": "?"
        }.get(self.risk_rating, "?")

        lines = [
            f"  [TRADE] {self.direction} \"{self.market_question[:50]}\" @ {self.entry_price:.0%}",
            f"    Edge: {self.edge_pct:.1f}% | Fair: {self.fair_value:.0%} | "
            f"Size: ${self.size_usd:.2f} | Conv: {self.conviction_score:.0f}/100",
            f"    Agents: {agents_str} | Risk: [{risk_icon}] {self.risk_rating}",
            f"    Thesis: {self._generate_one_liner()}",
        ]

        if self.pre_mortem_scenario:
            lines.append(f"    Pre-mortem: {self.pre_mortem_scenario[:70]}")

        return "\n".join(lines)

    def to_full_thesis(self) -> str:
        """
        Full structured thesis for dashboard/logging.
        This is the core output — the "why" behind every trade.
        """
        sections = []

        # Header
        sections.append(f"{'='*60}")
        sections.append(f"TRADE THESIS: {self.direction}")
        sections.append(f"Market: {self.market_question}")
        sections.append(f"{'='*60}")

        # Action
        sections.append(f"\n--- ACTION ---")
        sections.append(f"Direction: {self.direction}")
        sections.append(f"Entry Price: {self.entry_price:.2%}")
        sections.append(f"Fair Value:  {self.fair_value:.2%}")
        sections.append(f"Edge:        {self.edge_pct:.1f}%")
        sections.append(f"Size:        ${self.size_usd:.2f}")
        sections.append(f"Kelly:       {self.kelly_fraction:.1%}")

        # Agent contributions
        sections.append(f"\n--- AGENT INTELLIGENCE ---")
        sections.append(f"Consensus: {self.consensus_probability:.1%} "
                        f"(confidence: {self.consensus_confidence:.0%})")
        sections.append(f"Conviction: {self.conviction_score:.0f}/100")
        sections.append(f"Contributing agents ({len(self.contributing_agents)}):")
        for view in self.agent_views:
            sections.append(
                f"  - {view['agent']}: {view['probability']:.0%} "
                f"(conf: {view['confidence']:.0%})"
            )
            if view.get("reasoning"):
                sections.append(f"    {view['reasoning'][:80]}")

        # Evidence
        if self.key_evidence:
            sections.append(f"\n--- KEY EVIDENCE ---")
            for ev in self.key_evidence[:5]:
                sections.append(f"  * {ev[:80]}")

        if self.news_context:
            sections.append(f"\n--- NEWS CONTEXT ---")
            for news in self.news_context[:3]:
                sections.append(f"  * {news[:80]}")

        if self.correlation_context:
            sections.append(f"\n--- CORRELATED MARKETS ---")
            for corr in self.correlation_context[:3]:
                sections.append(f"  * {corr[:80]}")

        # Risk
        sections.append(f"\n--- RISK ASSESSMENT ---")
        sections.append(f"Risk Rating:     {self.risk_rating.upper()}")
        sections.append(f"Market Regime:   {self.market_regime}")
        sections.append(f"Entropy:         {self.entropy:.2f}")
        sections.append(f"Surprise Factor: {self.surprise_factor:.2f}")
        if self.pre_mortem_scenario:
            sections.append(f"Pre-mortem:      {self.pre_mortem_scenario}")
        if self.kill_condition:
            sections.append(f"Kill Condition:  {self.kill_condition}")
        if self.correlated_positions:
            sections.append(f"Corr. Positions: {', '.join(self.correlated_positions[:3])}")

        sections.append(f"\n--- SCORING ---")
        sections.append(f"Composite Score:        {self.composite_score:.1f}")
        sections.append(f"Confidence Multiplier:  {self.confidence_multiplier:.2f}")

        sections.append(f"\n{'='*60}")

        return "\n".join(sections)

    def to_dict(self) -> dict:
        """Serialisable dict for storage and API responses."""
        return {
            "trade_id": self.trade_id,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "fair_value": self.fair_value,
            "edge_pct": self.edge_pct,
            "size_usd": self.size_usd,
            "contributing_agents": self.contributing_agents,
            "agent_views": self.agent_views,
            "consensus_probability": self.consensus_probability,
            "consensus_confidence": self.consensus_confidence,
            "conviction_score": self.conviction_score,
            "key_evidence": self.key_evidence,
            "news_context": self.news_context,
            "correlation_context": self.correlation_context,
            "risk_rating": self.risk_rating,
            "pre_mortem_scenario": self.pre_mortem_scenario,
            "kill_condition": self.kill_condition,
            "correlated_positions": self.correlated_positions,
            "kelly_fraction": self.kelly_fraction,
            "confidence_multiplier": self.confidence_multiplier,
            "market_regime": self.market_regime,
            "entropy": self.entropy,
            "surprise_factor": self.surprise_factor,
            "composite_score": self.composite_score,
            "one_liner": self._generate_one_liner(),
            "timestamp": self.timestamp,
        }

    def _generate_one_liner(self) -> str:
        """Generate a concise natural language summary."""
        agent_count = len(self.contributing_agents)
        top_agents = self.contributing_agents[:2]

        # Find the agent with the strongest view
        strongest = max(self.agent_views, key=lambda v: v.get("confidence", 0)) \
            if self.agent_views else None

        direction_word = "bullish" if "YES" in self.direction else "bearish"

        parts = [f"{agent_count} agents {direction_word}"]

        if strongest:
            parts.append(
                f"led by {strongest['agent']} ({strongest.get('reasoning', '')[:40]})"
            )

        if self.edge_pct > 10:
            parts.append(f"with strong {self.edge_pct:.0f}% edge")
        elif self.edge_pct > 5:
            parts.append(f"with {self.edge_pct:.0f}% edge")

        if self.risk_rating == "dangerous":
            parts.append("BUT adversarial flags risk")

        return " — ".join(parts)


class ThesisGenerator:
    """
    Generates trade theses from swarm state.

    Usage:
        gen = ThesisGenerator(world, adversarial_agent)
        thesis = gen.generate(edge, composite_score, size, confidence_mult)
        print(thesis.to_terminal_summary())
        # Store thesis.to_dict() for dashboard
    """

    def __init__(self, world: WorldModel, adversarial=None):
        self.world = world
        self.adversarial = adversarial
        self._theses: list[TradeThesis] = []

    def generate(self, edge, composite_score: float,
                  size_usd: float, kelly_fraction: float,
                  confidence_multiplier: float,
                  trade_id: str = "") -> TradeThesis:
        """Generate a complete thesis for a trade about to be executed."""

        market_id = edge.market_id
        beliefs = self.world.get_belief_summary(market_id)
        news = self.world.get_news_for_market(market_id)
        correlations = self.world.get_correlated_markets(market_id)

        # Agent views
        agent_views = [
            {
                "agent": b["agent"],
                "probability": b["probability"],
                "confidence": b["confidence"],
                "reasoning": b["reasoning"][:100],
            }
            for b in beliefs.get("beliefs", [])
        ]

        # Key evidence (flatten from beliefs)
        key_evidence = []
        for b in beliefs.get("beliefs", []):
            if b.get("reasoning"):
                key_evidence.append(f"[{b['agent']}] {b['reasoning'][:80]}")

        # News context
        news_context = [
            f"{n.headline} (urgency: {n.urgency:.0%})"
            for n in (news or [])[:3]
        ]

        # Correlation context
        correlation_context = [
            f"{c.description} ({c.correlation_type}, strength: {c.strength:.0%})"
            for c in (correlations or [])[:3]
        ]

        # Correlated existing positions
        correlated_positions = []
        for c in (correlations or []):
            other_id = c.market_b_id if c.market_a_id == market_id else c.market_a_id
            other_q = self.world._market_questions.get(other_id, "")
            if other_q:
                correlated_positions.append(f"{other_q[:40]} ({c.correlation_type})")

        # Adversarial assessment
        risk_rating = "unreviewed"
        pre_mortem_scenario = ""
        kill_condition = ""
        if self.adversarial:
            pm = self.adversarial.get_pre_mortem(market_id)
            if pm:
                risk_rating = pm.risk_rating
                if pm.failure_scenarios:
                    pre_mortem_scenario = pm.failure_scenarios[0].scenario
                    kill_condition = pm.failure_scenarios[0].kill_condition

        if not trade_id:
            import uuid
            trade_id = str(uuid.uuid4())[:8]

        thesis = TradeThesis(
            trade_id=trade_id,
            market_id=market_id,
            market_question=edge.market_question,
            direction=edge.direction,
            entry_price=edge.market_price,
            fair_value=edge.fair_value,
            edge_pct=edge.edge_pct,
            size_usd=size_usd,
            contributing_agents=edge.contributing_agents,
            agent_views=agent_views,
            consensus_probability=beliefs.get("consensus_probability", edge.fair_value),
            consensus_confidence=beliefs.get("consensus_confidence", edge.confidence),
            conviction_score=edge.conviction,
            key_evidence=key_evidence,
            news_context=news_context,
            correlation_context=correlation_context,
            risk_rating=risk_rating,
            pre_mortem_scenario=pre_mortem_scenario,
            kill_condition=kill_condition,
            correlated_positions=correlated_positions,
            kelly_fraction=kelly_fraction,
            confidence_multiplier=confidence_multiplier,
            market_regime=edge.regime,
            entropy=edge.entropy,
            surprise_factor=edge.surprise_factor,
            composite_score=composite_score,
        )

        self._theses.append(thesis)
        self._theses = self._theses[-100:]  # Keep last 100

        return thesis

    def get_recent_theses(self, n: int = 10) -> list[dict]:
        """Get recent theses for dashboard."""
        return [t.to_dict() for t in self._theses[-n:]]

    def get_thesis_by_id(self, trade_id: str) -> TradeThesis | None:
        """Look up a specific thesis."""
        for t in reversed(self._theses):
            if t.trade_id == trade_id:
                return t
        return None