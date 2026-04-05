"""
Information Decay Curves — type-specific belief aging.

Different information types should decay at different rates:
- Breaking news: 30 min half-life (already stale if market hasn't moved)
- Sports form data: 6 hour half-life (valid until next match)
- Odds consensus: 30 min half-life (bookmakers update constantly)
- Probability estimates: 1 hour half-life (baseline analysis)
- Deep research: 4 hour half-life (thorough work stays relevant)
- Contrarian analysis: 2 hour half-life (bias corrections are medium-term)
- Structural/political: 24 hour half-life (slow-moving factors)
- Correlation-derived: 15 min half-life (weak signal, decays fast)
- Base rates: 7 day half-life (historical patterns barely change)

This replaces the flat TTL system in the world model.
Instead of beliefs simply expiring, they FADE — their weight
in consensus calculation decreases smoothly over time.

The decay curve also depends on MARKET REGIME:
- Volatile markets: all decay rates increase by 50%
- Stable markets: decay rates decrease by 30%
- Trending markets: momentum signals decay slower, contrarian faster
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class DecayProfile:
    """Decay parameters for a specific information type."""
    half_life_seconds: float    # Time for weight to drop to 50%
    min_weight: float = 0.05   # Below this, belief is effectively dead
    max_age_seconds: float = 0  # Hard expiry (0 = use calculated from half-life)

    def __post_init__(self):
        if self.max_age_seconds == 0:
            # Default hard expiry: when weight drops below min_weight
            # weight = e^(-0.693 * age / half_life) = min_weight
            # age = -half_life * ln(min_weight) / 0.693
            if self.min_weight > 0:
                self.max_age_seconds = -self.half_life_seconds * math.log(self.min_weight) / 0.693
            else:
                self.max_age_seconds = self.half_life_seconds * 10

    def compute_weight(self, age_seconds: float) -> float:
        """
        Exponential decay weight for a belief of this age.
        Returns 1.0 at age=0, 0.5 at age=half_life, approaches 0.
        """
        if age_seconds >= self.max_age_seconds:
            return 0.0
        if self.half_life_seconds <= 0:
            return 1.0
        weight = math.exp(-0.693 * age_seconds / self.half_life_seconds)
        return max(self.min_weight, weight)


# ── Default decay profiles per agent/source ──────────

AGENT_DECAY_PROFILES: dict[str, DecayProfile] = {
    # Breaking news decays fastest — if the market hasn't moved, it's priced in
    "news_scout": DecayProfile(
        half_life_seconds=1800,      # 30 minutes
        min_weight=0.05,
    ),

    # Odds consensus from bookmakers — updates constantly
    "odds_arbitrage": DecayProfile(
        half_life_seconds=1800,      # 30 minutes
        min_weight=0.05,
    ),

    # Momentum signals — fleeting by nature
    "momentum_detector": DecayProfile(
        half_life_seconds=600,       # 10 minutes
        min_weight=0.05,
    ),

    # Probability estimates — solid baseline
    "probability_estimator": DecayProfile(
        half_life_seconds=3600,      # 1 hour
        min_weight=0.05,
    ),

    # Contrarian analysis — medium-term bias corrections
    "contrarian": DecayProfile(
        half_life_seconds=7200,      # 2 hours
        min_weight=0.05,
    ),

    # Social signals — sentiment shifts moderately fast
    "social_signals": DecayProfile(
        half_life_seconds=2400,      # 40 minutes
        min_weight=0.05,
    ),

    # Sports intelligence — valid until next match
    "sports_intelligence": DecayProfile(
        half_life_seconds=21600,     # 6 hours
        min_weight=0.05,
    ),

    # Deep research — thorough analysis stays relevant
    "web_researcher": DecayProfile(
        half_life_seconds=14400,     # 4 hours
        min_weight=0.05,
    ),

    # Edge stacking — derived from other beliefs, decays fast
    "edge_stacker": DecayProfile(
        half_life_seconds=900,       # 15 minutes
        min_weight=0.05,
    ),

    # Correlation-derived beliefs — weakest signal
    "belief_propagation": DecayProfile(
        half_life_seconds=900,       # 15 minutes
        min_weight=0.05,
    ),

    # Adversarial — warnings stay relevant for a while
    "adversarial": DecayProfile(
        half_life_seconds=5400,      # 90 minutes
        min_weight=0.05,
    ),

    # Correlation detective — structural insights last long
    "correlation_detective": DecayProfile(
        half_life_seconds=28800,     # 8 hours
        min_weight=0.05,
    ),

    # Market scanner — price snapshots decay fast
    "market_scanner": DecayProfile(
        half_life_seconds=300,       # 5 minutes
        min_weight=0.05,
    ),
}

# Fallback for unknown agents
DEFAULT_DECAY_PROFILE = DecayProfile(
    half_life_seconds=3600,  # 1 hour default
    min_weight=0.05,
)


# ── Regime-adjusted decay ────────────────────────────

REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
    # In volatile markets, most information decays faster
    "volatile": {
        "news_scout": 0.6,           # News in volatile markets → very fast decay
        "probability_estimator": 0.7,
        "momentum_detector": 0.5,    # Momentum even more fleeting
        "contrarian": 0.8,
        "sports_intelligence": 1.0,  # Sports data doesn't care about volatility
        "web_researcher": 0.8,
        "adversarial": 0.7,
        "_default": 0.7,
    },

    # In stable markets, information stays relevant longer
    "stable": {
        "news_scout": 1.3,
        "probability_estimator": 1.5,
        "momentum_detector": 0.5,    # No momentum in stable markets anyway
        "contrarian": 1.5,
        "sports_intelligence": 1.0,
        "web_researcher": 1.5,
        "adversarial": 1.3,
        "_default": 1.3,
    },

    # In trending markets, momentum signals last longer
    "trending_up": {
        "momentum_detector": 2.0,    # Momentum is king in trends
        "contrarian": 0.6,           # Contrarian views decay fast in trends
        "probability_estimator": 1.0,
        "_default": 1.0,
    },
    "trending_down": {
        "momentum_detector": 2.0,
        "contrarian": 0.6,
        "probability_estimator": 1.0,
        "_default": 1.0,
    },
}


class DecayEngine:
    """
    Computes belief freshness weights using type-specific decay curves.

    Drop-in replacement for the flat freshness calculation in
    WorldModel._get_consensus_unlocked().

    Usage:
        engine = DecayEngine()
        weight = engine.compute_freshness(
            agent_name="news_scout",
            belief_age_seconds=900,  # 15 min old
            regime="volatile"
        )
        # Returns ~0.5 for 15-min-old breaking news in volatile market
    """

    def __init__(self, custom_profiles: dict[str, DecayProfile] | None = None):
        self.profiles = {**AGENT_DECAY_PROFILES}
        if custom_profiles:
            self.profiles.update(custom_profiles)

    def compute_freshness(self, agent_name: str,
                           belief_age_seconds: float,
                           regime: str = "unknown") -> float:
        """
        Compute the freshness weight for a belief.

        Returns: float 0.0-1.0 where 1.0 is brand new, 0.0 is expired.
        """
        profile = self.profiles.get(agent_name, DEFAULT_DECAY_PROFILE)

        # Adjust half-life based on market regime
        regime_mults = REGIME_MULTIPLIERS.get(regime, {})
        multiplier = regime_mults.get(agent_name, regime_mults.get("_default", 1.0))

        adjusted_half_life = profile.half_life_seconds * multiplier
        adjusted_profile = DecayProfile(
            half_life_seconds=adjusted_half_life,
            min_weight=profile.min_weight,
        )

        return adjusted_profile.compute_weight(belief_age_seconds)

    def is_belief_alive(self, agent_name: str,
                         belief_age_seconds: float,
                         regime: str = "unknown") -> bool:
        """Check if a belief should still be considered at all."""
        return self.compute_freshness(agent_name, belief_age_seconds, regime) > 0.05

    def get_effective_ttl(self, agent_name: str,
                           regime: str = "unknown") -> float:
        """Get the effective TTL for this agent's beliefs in this regime."""
        profile = self.profiles.get(agent_name, DEFAULT_DECAY_PROFILE)
        regime_mults = REGIME_MULTIPLIERS.get(regime, {})
        multiplier = regime_mults.get(agent_name, regime_mults.get("_default", 1.0))
        return profile.max_age_seconds * multiplier

    def get_status(self) -> dict:
        """Report current decay profiles for dashboard."""
        return {
            agent: {
                "half_life_min": round(p.half_life_seconds / 60, 1),
                "max_age_min": round(p.max_age_seconds / 60, 1),
            }
            for agent, p in sorted(self.profiles.items())
        }