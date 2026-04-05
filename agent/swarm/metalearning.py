"""
Metalearning System — learning which agent combinations work.

Tracks performance across dimensions:
- Which agent combinations produce the best predictions
- Which market types (sports, politics, entertainment) each combo excels at
- Time-of-day patterns (agents may be better at different times)
- Market regime performance (stable vs volatile)

Feeds back into:
- Consensus weighting (boost proven combos, dampen weak ones)
- Attention allocation (send the right agents to the right markets)
- Dynamic agent intervals (run effective agents more often)

This is essentially a contextual bandit over agent combinations.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import structlog

logger = structlog.get_logger()


@dataclass
class TradeOutcome:
    """Recorded outcome of a trade for metalearning."""
    market_id: str
    market_question: str
    market_category: str
    contributing_agents: list[str]
    edge_pct: float
    conviction: float
    composite_score: float
    regime: str
    hour_of_day: int
    pnl: float
    was_profitable: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentComboStats:
    """Performance statistics for a specific agent combination."""
    agents: tuple[str, ...]
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_edge: float = 0.0
    avg_conviction: float = 0.0
    pnl_history: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if not self.pnl_history:
            return 0.0
        return mean(self.pnl_history[-20:])

    @property
    def sharpe(self) -> float:
        if len(self.pnl_history) < 5:
            return 0.0
        recent = self.pnl_history[-20:]
        avg = mean(recent)
        sd = stdev(recent) if len(recent) >= 2 else 1.0
        return avg / sd if sd > 0 else 0.0


class MetalearningSystem:
    """
    Learns which agent combinations work best in which contexts.

    The key insight: a single agent's accuracy is less important
    than the COMBINATION of agents that contributed to a trade.

    "probability_estimator + sports_intelligence on football at 3pm"
    is a much richer signal than just "probability_estimator accuracy = 65%".
    """

    def __init__(self, persist_path: str | None = None):
        # === Performance tracking ===
        self._outcomes: list[TradeOutcome] = []

        # === Combo performance (the core learning) ===
        # Key: frozenset of agent names -> stats
        self._combo_stats: dict[frozenset, AgentComboStats] = {}

        # === Contextual performance ===
        # Key: (combo_key, category) -> stats
        self._category_stats: dict[tuple, AgentComboStats] = {}
        # Key: (combo_key, regime) -> stats
        self._regime_stats: dict[tuple, AgentComboStats] = {}
        # Key: (combo_key, hour_bucket) -> stats
        self._time_stats: dict[tuple, AgentComboStats] = {}

        # === Individual agent tracking ===
        self._agent_contribution: dict[str, list[float]] = defaultdict(list)

        # === Persistence ===
        self._persist_path = persist_path

        # Load previous data if available
        if persist_path:
            self._load()

    def record_outcome(self, outcome: TradeOutcome):
        """Record the outcome of a trade for learning."""
        self._outcomes.append(outcome)
        self._outcomes = self._outcomes[-500:]  # Keep last 500

        agents = tuple(sorted(outcome.contributing_agents))
        combo_key = frozenset(agents)

        # === Update combo stats ===
        if combo_key not in self._combo_stats:
            self._combo_stats[combo_key] = AgentComboStats(agents=agents)
        stats = self._combo_stats[combo_key]
        stats.total_trades += 1
        stats.pnl_history.append(outcome.pnl)
        stats.pnl_history = stats.pnl_history[-50:]
        stats.total_pnl += outcome.pnl
        if outcome.was_profitable:
            stats.wins += 1
        else:
            stats.losses += 1

        # === Update contextual stats ===
        cat_key = (combo_key, outcome.market_category)
        if cat_key not in self._category_stats:
            self._category_stats[cat_key] = AgentComboStats(agents=agents)
        self._update_stats(self._category_stats[cat_key], outcome)

        regime_key = (combo_key, outcome.regime)
        if regime_key not in self._regime_stats:
            self._regime_stats[regime_key] = AgentComboStats(agents=agents)
        self._update_stats(self._regime_stats[regime_key], outcome)

        hour_bucket = (outcome.hour_of_day // 4) * 4  # 4-hour buckets
        time_key = (combo_key, hour_bucket)
        if time_key not in self._time_stats:
            self._time_stats[time_key] = AgentComboStats(agents=agents)
        self._update_stats(self._time_stats[time_key], outcome)

        # === Individual agent contribution ===
        for agent in outcome.contributing_agents:
            self._agent_contribution[agent].append(outcome.pnl)
            self._agent_contribution[agent] = self._agent_contribution[agent][-100:]

        # Narrate significant learnings
        if stats.total_trades % 10 == 0 and stats.total_trades >= 10:
            print(
                f"  [METALEARNING] Combo {'+'.join(agents[:3])} "
                f"after {stats.total_trades} trades: "
                f"win={stats.win_rate:.0%}, avg_pnl=${stats.avg_pnl:.2f}, "
                f"sharpe={stats.sharpe:.2f}",
                flush=True,
            )

        # Persist periodically
        if self._persist_path and len(self._outcomes) % 20 == 0:
            self._save()

    def _update_stats(self, stats: AgentComboStats, outcome: TradeOutcome):
        """Helper to update any stats object."""
        stats.total_trades += 1
        stats.pnl_history.append(outcome.pnl)
        stats.pnl_history = stats.pnl_history[-50:]
        stats.total_pnl += outcome.pnl
        if outcome.was_profitable:
            stats.wins += 1
        else:
            stats.losses += 1

    # ── Query methods (used by coordinator and attention engine) ──

    def get_combo_weight(self, agents: list[str],
                          category: str = "",
                          regime: str = "",
                          hour: int = -1) -> float:
        """
        Get a performance-based weight multiplier for an agent combination.
        Returns: float multiplier (0.5 = bad track record, 1.0 = neutral, 1.5 = great)

        This is the PRIMARY feedback mechanism. The coordinator uses this
        to adjust the composite score for each edge.
        """
        combo_key = frozenset(sorted(agents))
        weights = []

        # Base combo performance
        stats = self._combo_stats.get(combo_key)
        if stats and stats.total_trades >= 5:
            combo_w = 0.7 + stats.win_rate * 0.6  # Range: 0.7-1.3
            if stats.sharpe > 0.5:
                combo_w *= 1.1
            elif stats.sharpe < -0.5:
                combo_w *= 0.8
            weights.append(combo_w)

        # Category-specific performance
        if category:
            cat_stats = self._category_stats.get((combo_key, category))
            if cat_stats and cat_stats.total_trades >= 3:
                cat_w = 0.7 + cat_stats.win_rate * 0.6
                weights.append(cat_w)

        # Regime-specific performance
        if regime:
            regime_stats = self._regime_stats.get((combo_key, regime))
            if regime_stats and regime_stats.total_trades >= 3:
                regime_w = 0.7 + regime_stats.win_rate * 0.6
                weights.append(regime_w)

        # Time-specific performance
        if hour >= 0:
            hour_bucket = (hour // 4) * 4
            time_stats = self._time_stats.get((combo_key, hour_bucket))
            if time_stats and time_stats.total_trades >= 3:
                time_w = 0.7 + time_stats.win_rate * 0.6
                weights.append(time_w)

        if not weights:
            return 1.0  # No data, neutral weight

        return sum(weights) / len(weights)

    def get_best_agents_for_context(self, category: str = "",
                                      regime: str = "",
                                      top_n: int = 3) -> list[tuple[str, float]]:
        """
        Get the best-performing individual agents for a given context.
        Used by attention allocation to prioritise agents.
        """
        agent_scores: dict[str, list[float]] = defaultdict(list)

        for outcome in self._outcomes[-200:]:
            if category and outcome.market_category != category:
                continue
            if regime and outcome.regime != regime:
                continue
            for agent in outcome.contributing_agents:
                agent_scores[agent].append(1.0 if outcome.was_profitable else 0.0)

        ranked = []
        for agent, scores in agent_scores.items():
            if len(scores) >= 3:
                ranked.append((agent, mean(scores)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def get_agent_consensus_weight(self, agent_name: str,
                                     category: str = "") -> float:
        """
        Weight adjustment for a specific agent's beliefs in consensus calculation.
        Agents with good track records get amplified, bad ones dampened.

        Used by world model's get_consensus() to weight beliefs.
        """
        contributions = self._agent_contribution.get(agent_name, [])
        if len(contributions) < 5:
            return 1.0  # Not enough data

        recent = contributions[-20:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)

        # Range: 0.6 (terrible) to 1.4 (excellent)
        return 0.6 + win_rate * 0.8

    def should_skip_combo(self, agents: list[str],
                            category: str = "") -> bool:
        """
        Returns True if this agent combination has a proven negative record
        in this context. Saves API credits by not running bad combos.
        """
        combo_key = frozenset(sorted(agents))

        if category:
            stats = self._category_stats.get((combo_key, category))
        else:
            stats = self._combo_stats.get(combo_key)

        if stats and stats.total_trades >= 10:
            if stats.win_rate < 0.3 and stats.avg_pnl < 0:
                return True

        return False

    def get_report(self) -> dict:
        """Full metalearning report for dashboard."""
        # Best combos overall
        best_combos = sorted(
            [(k, v) for k, v in self._combo_stats.items() if v.total_trades >= 5],
            key=lambda x: x[1].sharpe,
            reverse=True,
        )[:5]

        # Best agents overall
        agent_perf = []
        for agent, pnls in self._agent_contribution.items():
            if len(pnls) >= 5:
                agent_perf.append({
                    "agent": agent,
                    "trades": len(pnls),
                    "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 2),
                    "avg_pnl": round(mean(pnls[-20:]), 3),
                })
        agent_perf.sort(key=lambda x: x["avg_pnl"], reverse=True)

        return {
            "total_outcomes_tracked": len(self._outcomes),
            "unique_combos": len(self._combo_stats),
            "best_combos": [
                {
                    "agents": list(combo),
                    "trades": stats.total_trades,
                    "win_rate": round(stats.win_rate, 2),
                    "sharpe": round(stats.sharpe, 2),
                    "total_pnl": round(stats.total_pnl, 2),
                }
                for combo, stats in best_combos
            ],
            "agent_performance": agent_perf[:8],
        }

    # ── Persistence ───────────────────────────────────

    def _save(self):
        """Save outcomes to disk."""
        if not self._persist_path:
            return
        try:
            data = {
                "outcomes": [
                    {
                        "market_id": o.market_id,
                        "market_question": o.market_question,
                        "market_category": o.market_category,
                        "contributing_agents": o.contributing_agents,
                        "edge_pct": o.edge_pct,
                        "conviction": o.conviction,
                        "composite_score": o.composite_score,
                        "regime": o.regime,
                        "hour_of_day": o.hour_of_day,
                        "pnl": o.pnl,
                        "was_profitable": o.was_profitable,
                        "timestamp": o.timestamp,
                    }
                    for o in self._outcomes
                ]
            }
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug("metalearning_save_error", error=str(e))

    def _load(self):
        """Load outcomes from disk and rebuild stats."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            if not path.exists():
                return
            with open(path) as f:
                data = json.load(f)

            for raw in data.get("outcomes", []):
                outcome = TradeOutcome(**raw)
                # Rebuild stats without narration
                self._outcomes.append(outcome)
                agents = tuple(sorted(outcome.contributing_agents))
                combo_key = frozenset(agents)
                if combo_key not in self._combo_stats:
                    self._combo_stats[combo_key] = AgentComboStats(agents=agents)
                self._update_stats(self._combo_stats[combo_key], outcome)

            logger.info("metalearning_loaded", outcomes=len(self._outcomes))
        except Exception as e:
            logger.debug("metalearning_load_error", error=str(e))