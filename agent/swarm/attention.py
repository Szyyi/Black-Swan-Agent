"""
Attention Allocation Engine — intelligent agent scheduling.

Instead of all agents analysing all markets on fixed timers,
this engine dynamically allocates agent attention using UCB1
(Upper Confidence Bound) multi-armed bandit.

Markets compete for agent cycles based on:
- Information entropy (agent disagreement = needs more analysis)
- Surprise recency (recent shocks need immediate attention)
- Time-to-expiry urgency (approaching deadlines)
- Staleness (hasn't been analysed recently)
- Edge magnitude (bigger edges deserve more scrutiny)
- Volatility (volatile markets change fast, need frequent updates)

Each agent gets a prioritised market queue instead of
scanning everything on a fixed timer.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from agent.swarm.world_model import WorldModel

logger = structlog.get_logger()


@dataclass
class MarketAttentionScore:
    """Computed attention score for a single market."""
    market_id: str
    question: str
    entropy_score: float = 0.0      # Agent disagreement
    surprise_score: float = 0.0     # Recent shocks
    urgency_score: float = 0.0      # Time to expiry
    staleness_score: float = 0.0    # Time since last analysis
    edge_score: float = 0.0         # Magnitude of detected edge
    volatility_score: float = 0.0   # Regime volatility
    exploration_bonus: float = 0.0  # UCB exploration term
    total_score: float = 0.0
    last_analysed_by: dict[str, float] = field(default_factory=dict)  # agent -> timestamp


class AttentionAllocationEngine:
    """
    Multi-armed bandit scheduler for agent attention.

    Usage:
        engine = AttentionAllocationEngine(world)
        # Each agent asks for its priority queue:
        queue = engine.get_priority_markets("probability_estimator", all_markets, top_n=3)
        # After analysis, record the result:
        engine.record_analysis("probability_estimator", market_id, value_gained)
    """

    # How much each factor matters (tunable weights)
    WEIGHTS = {
        "entropy": 3.0,       # High disagreement = needs attention
        "surprise": 4.0,      # Recent shocks = urgent
        "urgency": 2.5,       # Approaching expiry = time-sensitive
        "staleness": 2.0,     # Hasn't been looked at = exploration
        "edge": 1.5,          # Bigger edges = more scrutiny needed
        "volatility": 1.5,    # Fast-moving markets = frequent updates
        "exploration": 1.0,   # UCB1 exploration bonus
    }

    # Agent specialisation — which factors matter MORE for each agent
    AGENT_FACTOR_WEIGHTS = {
        "news_scout": {"surprise": 2.0, "urgency": 1.5},
        "probability_estimator": {"entropy": 1.5, "edge": 1.5},
        "contrarian": {"entropy": 2.0, "staleness": 1.5},
        "correlation_detective": {"edge": 1.5, "volatility": 1.5},
        "momentum_detector": {"volatility": 2.0, "surprise": 1.5},
        "sports_intelligence": {"staleness": 1.5, "urgency": 1.5},
        "web_researcher": {"staleness": 2.0, "edge": 1.5},
        "odds_arbitrage": {"edge": 2.0, "staleness": 1.0},
        "social_signals": {"surprise": 1.5, "entropy": 1.5},
        "edge_stacker": {"edge": 2.0, "entropy": 1.5},
        "adversarial": {"entropy": 2.0, "edge": 2.0, "surprise": 1.5},
    }

    def __init__(self, world: WorldModel, exploration_constant: float = 1.4):
        self.world = world
        self.c = exploration_constant  # UCB1 exploration parameter

        # Per-agent, per-market tracking
        self._analysis_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_analyses: dict[str, int] = defaultdict(int)
        self._value_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._last_analysis_time: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Global market scores (cached, recomputed periodically)
        self._market_scores: dict[str, MarketAttentionScore] = {}
        self._last_score_update: float = 0
        self._score_update_interval: float = 15  # Recompute every 15 seconds

    def get_priority_markets(self, agent_name: str, markets: list,
                              top_n: int = 5, category_filter: str | None = None) -> list:
        """
        Get the top-N markets this agent should focus on RIGHT NOW.

        Returns markets sorted by attention score, personalised for this agent.
        """
        self._maybe_refresh_scores(markets)

        agent_weights = self.AGENT_FACTOR_WEIGHTS.get(agent_name, {})
        scored = []

        for market in markets:
            if not market.active:
                continue
            if category_filter and market.category != category_filter:
                continue

            mid = market.condition_id
            base = self._market_scores.get(mid)
            if not base:
                continue

            # Personalise score for this agent
            score = self._personalise_score(base, agent_name, agent_weights)

            # Add UCB1 exploration bonus
            n_total = max(1, self._total_analyses[agent_name])
            n_this = max(1, self._analysis_count[agent_name][mid])
            ucb_bonus = self.c * math.sqrt(math.log(n_total) / n_this)

            score += ucb_bonus * self.WEIGHTS["exploration"]
            scored.append((score, market))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_n]]

    def record_analysis(self, agent_name: str, market_id: str,
                         value_gained: float = 0.0):
        """
        Record that an agent analysed a market and the value it produced.

        value_gained: positive if the analysis was useful (found edge, changed belief, etc.)
                     This feeds back into the bandit to learn which markets are worth analysing.
        """
        self._analysis_count[agent_name][market_id] += 1
        self._total_analyses[agent_name] += 1
        self._last_analysis_time[agent_name][market_id] = time.time()

        if value_gained != 0:
            self._value_history[agent_name][market_id].append(value_gained)
            # Keep last 20 values per market per agent
            self._value_history[agent_name][market_id] = \
                self._value_history[agent_name][market_id][-20:]

    def get_agent_dynamic_interval(self, agent_name: str,
                                     base_interval: float) -> float:
        """
        Dynamically adjust an agent's cycle interval based on how much
        value it's producing. Agents finding lots of edges run faster;
        agents finding nothing slow down to conserve API credits.

        Returns adjusted interval in seconds.
        """
        recent_values = []
        for market_vals in self._value_history[agent_name].values():
            recent_values.extend(market_vals[-5:])

        if not recent_values:
            return base_interval  # No data yet, use default

        avg_value = sum(recent_values) / len(recent_values)

        # High value → run faster (down to 50% of base interval)
        # Low value → run slower (up to 200% of base interval)
        if avg_value > 0.5:
            multiplier = max(0.5, 1.0 - avg_value * 0.5)
        elif avg_value < 0.1:
            multiplier = min(2.0, 1.0 + (0.1 - avg_value) * 10)
        else:
            multiplier = 1.0

        return base_interval * multiplier

    def _maybe_refresh_scores(self, markets: list):
        """Recompute base scores if stale."""
        now = time.time()
        if now - self._last_score_update < self._score_update_interval:
            return

        for market in markets:
            if not market.active:
                continue
            mid = market.condition_id
            self._market_scores[mid] = self._compute_base_score(market)

        self._last_score_update = now

    def _compute_base_score(self, market) -> MarketAttentionScore:
        """Compute the attention score for a market (agent-agnostic)."""
        mid = market.condition_id
        score = MarketAttentionScore(market_id=mid, question=market.question)

        # 1. Entropy — agent disagreement
        entropy = self.world.get_entropy(mid)
        score.entropy_score = entropy  # Already 0-1 normalised

        # 2. Surprise — recent belief shocks
        score.surprise_score = self.world.get_surprise_score(mid)

        # 3. Urgency — time to expiry
        if market.end_date:
            try:
                from datetime import datetime
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                now = datetime.now(end.tzinfo) if end.tzinfo else datetime.now()
                hours_left = max(0, (end - now).total_seconds() / 3600)
                if hours_left < 1:
                    score.urgency_score = 1.0
                elif hours_left < 6:
                    score.urgency_score = 0.8
                elif hours_left < 24:
                    score.urgency_score = 0.5
                elif hours_left < 72:
                    score.urgency_score = 0.3
                else:
                    score.urgency_score = 0.1
            except (ValueError, TypeError):
                score.urgency_score = 0.2

        # 4. Staleness — time since ANY agent last analysed
        beliefs = self.world._get_valid_beliefs(mid)
        if beliefs:
            newest = max(b.timestamp for b in beliefs)
            age_minutes = (time.time() - newest) / 60
            score.staleness_score = min(1.0, age_minutes / 30)  # Max at 30 min stale
        else:
            score.staleness_score = 1.0  # Never analysed = max staleness

        # 5. Edge magnitude
        price = self.world.get_market_price(mid)
        if price is not None and beliefs:
            consensus, conf, _ = self.world.get_consensus(mid)
            if consensus is not None:
                edge = abs(consensus - price) * 100
                score.edge_score = min(1.0, edge / 15)  # Normalise: 15% edge = max
            else:
                score.edge_score = 0.0
        else:
            score.edge_score = 0.0

        # 6. Volatility
        regime = self.world._regimes.get(mid)
        if regime:
            if regime.regime in ("volatile",):
                score.volatility_score = 0.8
            elif regime.regime in ("trending_up", "trending_down"):
                score.volatility_score = 0.5
            elif regime.regime == "stable":
                score.volatility_score = 0.1
            else:
                score.volatility_score = 0.3
        else:
            score.volatility_score = 0.3

        # Weighted total
        score.total_score = (
            score.entropy_score * self.WEIGHTS["entropy"]
            + score.surprise_score * self.WEIGHTS["surprise"]
            + score.urgency_score * self.WEIGHTS["urgency"]
            + score.staleness_score * self.WEIGHTS["staleness"]
            + score.edge_score * self.WEIGHTS["edge"]
            + score.volatility_score * self.WEIGHTS["volatility"]
        )

        return score

    def _personalise_score(self, base: MarketAttentionScore,
                            agent_name: str,
                            agent_weights: dict[str, float]) -> float:
        """Apply agent-specific factor weights to the base score."""
        score = 0.0
        components = {
            "entropy": base.entropy_score,
            "surprise": base.surprise_score,
            "urgency": base.urgency_score,
            "staleness": base.staleness_score,
            "edge": base.edge_score,
            "volatility": base.volatility_score,
        }
        for factor, value in components.items():
            weight = self.WEIGHTS[factor] * agent_weights.get(factor, 1.0)
            score += value * weight

        # Bonus for markets this agent hasn't seen recently
        last_seen = self._last_analysis_time[agent_name].get(base.market_id, 0)
        agent_staleness = min(1.0, (time.time() - last_seen) / 1800)  # 30 min max
        score += agent_staleness * self.WEIGHTS["staleness"]

        return score

    def get_attention_report(self) -> dict:
        """Status report for the dashboard."""
        top_markets = sorted(
            self._market_scores.values(),
            key=lambda s: s.total_score,
            reverse=True,
        )[:10]

        return {
            "top_attention_markets": [
                {
                    "question": s.question[:50],
                    "total_score": round(s.total_score, 2),
                    "entropy": round(s.entropy_score, 2),
                    "surprise": round(s.surprise_score, 2),
                    "urgency": round(s.urgency_score, 2),
                    "staleness": round(s.staleness_score, 2),
                    "edge": round(s.edge_score, 2),
                    "volatility": round(s.volatility_score, 2),
                }
                for s in top_markets
            ],
            "agent_analysis_counts": {
                agent: sum(counts.values())
                for agent, counts in self._analysis_count.items()
            },
        }