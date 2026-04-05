"""
Event Bus — event-driven agent triggers.

Instead of agents only running on fixed timers, this system
detects critical events and immediately wakes relevant agents.

Events that trigger immediate analysis:
- Price moves > 5% in a short window
- Breaking news with high urgency
- Surprise events (belief shifts > 10%)
- New correlation discovered
- Market approaching expiry (< 1 hour)
- Adversarial alert (dangerous risk rating)

Each event type maps to specific agents that should respond.
The bus is non-blocking — it queues triggers that agents
check on their next iteration, or can force an immediate wake.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    PRICE_SPIKE = "price_spike"
    BREAKING_NEWS = "breaking_news"
    SURPRISE_BELIEF = "surprise_belief"
    NEW_CORRELATION = "new_correlation"
    EXPIRY_APPROACHING = "expiry_approaching"
    ADVERSARIAL_ALERT = "adversarial_alert"
    REGIME_CHANGE = "regime_change"
    EDGE_DETECTED = "edge_detected"


@dataclass
class SwarmEvent:
    """An event that should trigger agent analysis."""
    event_type: EventType
    market_id: str
    severity: float          # 0-1, how urgent
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    handled_by: list[str] = field(default_factory=list)


# Which agents should respond to which event types
EVENT_AGENT_MAP: dict[EventType, list[str]] = {
    EventType.PRICE_SPIKE: [
        "probability_estimator", "momentum_detector", "adversarial",
        "odds_arbitrage", "edge_stacker",
    ],
    EventType.BREAKING_NEWS: [
        "news_scout", "probability_estimator", "web_researcher",
        "social_signals", "adversarial",
    ],
    EventType.SURPRISE_BELIEF: [
        "probability_estimator", "contrarian", "adversarial",
        "correlation_detective",
    ],
    EventType.NEW_CORRELATION: [
        "edge_stacker", "adversarial", "probability_estimator",
    ],
    EventType.EXPIRY_APPROACHING: [
        "probability_estimator", "odds_arbitrage", "adversarial",
    ],
    EventType.ADVERSARIAL_ALERT: [
        "probability_estimator", "web_researcher", "contrarian",
    ],
    EventType.REGIME_CHANGE: [
        "momentum_detector", "probability_estimator", "adversarial",
    ],
    EventType.EDGE_DETECTED: [
        "adversarial", "odds_arbitrage",
    ],
}


class EventBus:
    """
    Central event bus for the swarm.

    Agents register themselves, and the bus notifies them when
    relevant events occur. Uses asyncio Events for non-blocking wake.

    Usage:
        bus = EventBus()
        bus.register_agent("probability_estimator", agent_instance)

        # When a price spike is detected (e.g., in world model):
        bus.emit(SwarmEvent(
            event_type=EventType.PRICE_SPIKE,
            market_id="abc123",
            severity=0.8,
            data={"old_price": 0.45, "new_price": 0.52, "change_pct": 15.6}
        ))

        # Agents check for pending events:
        events = bus.get_pending_events("probability_estimator")
    """

    def __init__(self, cooldown_seconds: float = 30):
        self._event_queue: list[SwarmEvent] = []
        self._agent_refs: dict[str, object] = {}  # agent_name -> agent instance
        self._agent_wake_events: dict[str, asyncio.Event] = {}
        self._pending: dict[str, list[SwarmEvent]] = defaultdict(list)
        self._cooldown = cooldown_seconds
        self._last_emit_per_market: dict[str, dict[str, float]] = defaultdict(dict)
        self._event_count: int = 0
        self._trigger_count: int = 0

    def register_agent(self, agent_name: str, agent_instance=None):
        """Register an agent to receive events."""
        self._agent_refs[agent_name] = agent_instance
        self._agent_wake_events[agent_name] = asyncio.Event()

    def emit(self, event: SwarmEvent):
        """
        Emit an event to the bus. Relevant agents will be notified.

        Includes cooldown to prevent event storms — the same event type
        for the same market won't fire twice within cooldown_seconds.
        """
        # Cooldown check
        last = self._last_emit_per_market.get(event.market_id, {}).get(event.event_type.value, 0)
        if time.time() - last < self._cooldown:
            return

        self._last_emit_per_market[event.market_id][event.event_type.value] = time.time()
        self._event_count += 1

        # Queue for archival
        self._event_queue.append(event)
        self._event_queue = self._event_queue[-200:]  # Keep last 200

        # Route to relevant agents
        target_agents = EVENT_AGENT_MAP.get(event.event_type, [])
        for agent_name in target_agents:
            if agent_name in self._agent_refs or agent_name in self._agent_wake_events:
                self._pending[agent_name].append(event)
                self._trigger_count += 1
                # Wake the agent if it's sleeping
                wake = self._agent_wake_events.get(agent_name)
                if wake:
                    wake.set()

        if event.severity > 0.6:
            print(
                f"  [EVENT] {event.event_type.value} on {event.data.get('question', event.market_id[:8])[:40]} "
                f"(severity: {event.severity:.0%})",
                flush=True,
            )

    def get_pending_events(self, agent_name: str,
                            max_events: int = 5) -> list[SwarmEvent]:
        """Get and clear pending events for an agent."""
        events = self._pending.get(agent_name, [])
        # Sort by severity, take top N
        events.sort(key=lambda e: e.severity, reverse=True)
        top = events[:max_events]
        self._pending[agent_name] = events[max_events:]

        # Mark as handled
        for e in top:
            e.handled_by.append(agent_name)

        return top

    def get_priority_markets_from_events(self, agent_name: str) -> list[str]:
        """Get market IDs that have pending events for this agent."""
        events = self._pending.get(agent_name, [])
        seen = set()
        result = []
        for e in sorted(events, key=lambda x: x.severity, reverse=True):
            if e.market_id not in seen:
                seen.add(e.market_id)
                result.append(e.market_id)
        return result

    async def wait_for_event(self, agent_name: str, timeout: float = None) -> bool:
        """
        Async wait for an event relevant to this agent.
        Returns True if an event arrived, False if timed out.
        Used by agents to sleep efficiently instead of polling.
        """
        wake = self._agent_wake_events.get(agent_name)
        if not wake:
            wake = asyncio.Event()
            self._agent_wake_events[agent_name] = wake

        try:
            if timeout:
                await asyncio.wait_for(wake.wait(), timeout=timeout)
            else:
                await wake.wait()
            wake.clear()
            return True
        except asyncio.TimeoutError:
            wake.clear()
            return False

    def has_pending(self, agent_name: str) -> bool:
        """Check if agent has pending events without consuming them."""
        return len(self._pending.get(agent_name, [])) > 0

    def get_status(self) -> dict:
        """Status report for dashboard."""
        now = time.time()
        recent = [e for e in self._event_queue if now - e.timestamp < 300]
        type_counts = defaultdict(int)
        for e in recent:
            type_counts[e.event_type.value] += 1

        return {
            "total_events": self._event_count,
            "total_triggers": self._trigger_count,
            "recent_events_5min": len(recent),
            "event_types": dict(type_counts),
            "pending_per_agent": {
                agent: len(events) for agent, events in self._pending.items() if events
            },
        }


# ── Event detectors (hook into world model) ──────────

def create_price_spike_detector(event_bus: EventBus, threshold_pct: float = 5.0):
    """
    Returns a callback that detects price spikes.
    Hook this into WorldModel.update_market_price().
    """
    _price_cache: dict[str, float] = {}

    def detect(market_id: str, new_price: float, question: str = "", **kwargs):
        old_price = _price_cache.get(market_id)
        _price_cache[market_id] = new_price

        if old_price is None or old_price == 0:
            return

        change_pct = abs(new_price - old_price) / old_price * 100
        if change_pct >= threshold_pct:
            severity = min(1.0, change_pct / 20)  # 20% move = max severity
            event_bus.emit(SwarmEvent(
                event_type=EventType.PRICE_SPIKE,
                market_id=market_id,
                severity=severity,
                data={
                    "old_price": round(old_price, 4),
                    "new_price": round(new_price, 4),
                    "change_pct": round(change_pct, 1),
                    "question": question,
                },
            ))

    return detect


def create_news_detector(event_bus: EventBus, urgency_threshold: float = 0.7):
    """
    Returns a callback that fires on breaking news.
    Hook this into WorldModel.submit_news_impact().
    """
    def detect(impact):
        if impact.urgency >= urgency_threshold:
            for mid in impact.affected_markets:
                event_bus.emit(SwarmEvent(
                    event_type=EventType.BREAKING_NEWS,
                    market_id=mid,
                    severity=impact.urgency,
                    data={
                        "headline": impact.headline,
                        "urgency": impact.urgency,
                        "confidence": impact.confidence,
                        "question": mid[:8],
                    },
                ))

    return detect


def create_surprise_detector(event_bus: EventBus, shift_threshold: float = 0.10):
    """
    Returns a callback that fires on surprise belief shifts.
    Hook this into WorldModel.submit_belief() surprise detection.
    """
    def detect(market_id: str, old_consensus: float, new_belief: float,
               agent_name: str, question: str = ""):
        shift = abs(new_belief - old_consensus)
        if shift >= shift_threshold:
            event_bus.emit(SwarmEvent(
                event_type=EventType.SURPRISE_BELIEF,
                market_id=market_id,
                severity=min(1.0, shift * 3),
                data={
                    "old_consensus": round(old_consensus, 3),
                    "new_belief": round(new_belief, 3),
                    "shift": round(shift, 3),
                    "agent": agent_name,
                    "question": question,
                },
            ))

    return detect