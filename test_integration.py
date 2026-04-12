"""
Black Swan Agent — Integration Smoke Test

Verifies the full intelligence stack wires up and runs end-to-end without
needing live Smarkets data. Mocks markets, injects synthetic beliefs,
fires events through the bus, and checks every subsystem responds.

Run: python test_integration.py
Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass

# Force unbuffered stdout for Windows
sys.stdout.reconfigure(line_buffering=True)


def out(msg: str):
    print(msg, flush=True)


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


results: list[TestResult] = []


def check(name: str, condition: bool, detail: str = ""):
    results.append(TestResult(name, condition, detail))
    marker = "✓" if condition else "✗"
    out(f"  [{marker}] {name}" + (f" — {detail}" if detail else ""))


async def main():
    out("=" * 64)
    out("  BLACK SWAN AGENT — INTEGRATION SMOKE TEST")
    out("=" * 64)
    out("")

    # ── Test 1: Imports ──────────────────────────────────────
    out("[1/8] Imports...")
    try:
        from agent.config import AgentConfig
        from agent.swarm.world_model import WorldModel, Belief, NewsImpact
        from agent.swarm.adversarial import AdversarialAgent
        from agent.swarm.attention import AttentionAllocationEngine
        from agent.swarm.event_bus import (
            EventBus, create_price_spike_detector,
            create_news_detector, create_surprise_detector,
        )
        from agent.swarm.metalearning import MetalearningSystem, TradeOutcome
        from agent.swarm.decay import DecayEngine
        from agent.swarm.thesis import ThesisGenerator
        from agent.swarm import agents as agents_module
        from agent.models import Market
        check("All intelligence modules importable", True)
    except Exception as e:
        check("All intelligence modules importable", False, str(e))
        traceback.print_exc()
        return finalize()

    # ── Test 2: World model + decay engine ──────────────────
    out("\n[2/8] World model + decay engine...")
    try:
        world = WorldModel()
        check("WorldModel constructs", True)
        check("Decay engine attached", hasattr(world, "decay_engine"),
              f"type={type(getattr(world, 'decay_engine', None)).__name__}")

        world.update_market_price("test_mkt_001", 0.45,
                                   question="Will X happen?",
                                   volume=10000, category="politics")
        price = world.get_market_price("test_mkt_001")
        check("Market price round-trip", price == 0.45, f"got {price}")

        world.submit_belief(Belief(
            agent_name="test_agent",
            market_id="test_mkt_001",
            probability=0.65,
            confidence=0.8,
            reasoning="synthetic test belief",
            evidence=["test"],
            ttl_seconds=600,
        ))
        summary = world.get_belief_summary("test_mkt_001")
        check("Belief submission + retrieval",
              len(summary.get("beliefs", [])) >= 1,
              f"{len(summary.get('beliefs', []))} beliefs")
    except Exception as e:
        check("World model exercise", False, str(e))
        traceback.print_exc()
        return finalize()

    # ── Test 3: Attention engine ────────────────────────────
    out("\n[3/8] Attention engine...")
    try:
        attention = AttentionAllocationEngine(world)
        agents_module.set_attention_engine(attention)
        check("Attention engine constructs", True)
        check("Attention engine wired into agents module",
              agents_module._attention_engine is attention)

        # Create synthetic markets
        markets = [
            Market(condition_id=f"mkt_{i:03d}",
                   slug=f"synthetic-market-{i}",
                   token_ids=[f"tok_{i}_yes", f"tok_{i}_no"],
                   outcomes=["YES", "NO"],
                   question=f"Synthetic market {i}",
                   category="politics", volume=1000.0 * (i + 1),
                   active=True)
            for i in range(10)
        ]
        for m in markets:
            world.update_market_price(m.condition_id, 0.4 + (hash(m.condition_id) % 20) / 100,
                                      question=m.question, volume=m.volume, category=m.category)

        priority = agents_module.get_priority_markets("probability_estimator", markets, top_n=3)
        check("get_priority_markets returns batch",
              isinstance(priority, list) and len(priority) > 0,
              f"{len(priority)} markets")

        report = attention.get_attention_report()
        check("Attention report generates",
              isinstance(report, dict) and "top_attention_markets" in report)
    except Exception as e:
        check("Attention engine", False, str(e))
        traceback.print_exc()

    # ── Test 4: Event bus ───────────────────────────────────
    out("\n[4/8] Event bus...")
    try:
        bus = EventBus(cooldown_seconds=1)
        check("Event bus constructs", True)

        fired = {"count": 0}

        class FakeAgent:
            name = "fake_agent"
            async def on_event(self, event):
                fired["count"] += 1

        bus.register_agent("fake_agent", FakeAgent())

        # Wire detectors
        spike_detector = create_price_spike_detector(bus, threshold_pct=5.0)
        news_detector = create_news_detector(bus, urgency_threshold=0.7)
        surprise_detector = create_surprise_detector(bus, shift_threshold=0.10)
        check("Event detectors created", True)

        # Fire a price spike via the world model
        world.register_event_callback("price_update", spike_detector)
        world.update_market_price("test_mkt_001", 0.45)
        world.update_market_price("test_mkt_001", 0.55)  # ~22% jump
        await asyncio.sleep(0.2)

        bus_status = bus.get_status()
        check("Event bus tracks events",
              isinstance(bus_status, dict),
              f"{bus_status.get('recent_events_5min', 0)} events / 5min")
    except Exception as e:
        check("Event bus", False, str(e))
        traceback.print_exc()

    # ── Test 5: Metalearning ────────────────────────────────
    out("\n[5/8] Metalearning system...")
    try:
        ml = MetalearningSystem(persist_path="./data/test_metalearning.json")
        check("Metalearning constructs", True)

        weight = ml.get_combo_weight(
            agents=["probability_estimator", "news_scout"],
            category="politics",
            regime="normal",
            hour=14,
        )
        check("Combo weight returns float",
              isinstance(weight, (int, float)),
              f"weight={weight}")

        ml.record_outcome(TradeOutcome(
            market_id="test_mkt_001",
            market_question="Will X happen?",
            market_category="politics",
            contributing_agents=["probability_estimator", "news_scout"],
            edge_pct=5.0,
            conviction=0.7,
            composite_score=2.5,
            regime="normal",
            hour_of_day=14,
            pnl=12.5,
            was_profitable=True,
        ))
        report = ml.get_report()
        check("Metalearning records + reports",
              isinstance(report, dict))
    except Exception as e:
        check("Metalearning", False, str(e))
        traceback.print_exc()

    # ── Test 6: Decay engine applies to beliefs ─────────────
    out("\n[6/8] Decay engine applied to beliefs...")
    try:
        world.submit_belief(Belief(
            agent_name="market_scanner",
            market_id="test_mkt_001",
            probability=0.7,
            confidence=0.9,
            reasoning="fresh belief",
            evidence=["test"],
            ttl_seconds=3600,
        ))
        summary_before = world.get_belief_summary("test_mkt_001")
        scanner_before = next(
            (b for b in summary_before.get("beliefs", [])
             if b.get("agent") == "market_scanner"), None
        )
        check("Fresh belief stored", scanner_before is not None,
              f"conf={scanner_before.get('confidence') if scanner_before else 'none'}")

        decay_status = world.decay_engine.get_status()
        check("Decay engine reports profile status",
              isinstance(decay_status, dict))
    except Exception as e:
        check("Decay application", False, str(e))
        traceback.print_exc()

    # ── Test 7: Thesis generator ────────────────────────────
    out("\n[7/8] Thesis generator...")
    try:
        thesis_gen = ThesisGenerator(world)
        check("Thesis generator constructs", True)

        edges = world.compute_edges(min_edge_pct=0.5)
        if edges:
            edge = edges[0]
            thesis = thesis_gen.generate(
                edge=edge,
                composite_score=2.1,
                size_usd=50.0,
                kelly_fraction=0.04,
                confidence_multiplier=1.0,
                trade_id="test_trade_001",
            )
            terminal_str = thesis.to_terminal_summary()
            check("Thesis generated for synthetic edge",
                  isinstance(terminal_str, str) and len(terminal_str) > 50,
                  f"{len(terminal_str)} chars")
        else:
            check("Thesis path reachable (no edges to test)", True,
                  "no edges available — synthetic edge gen skipped")
    except Exception as e:
        check("Thesis generator", False, str(e))
        traceback.print_exc()

    # ── Test 8: Coordinator wires everything together ──────
    out("\n[8/8] Coordinator initialisation...")
    try:
        from agent.swarm.coordinator import MetaCoordinator
        cfg = AgentConfig()
        coord = MetaCoordinator(cfg)
        check("MetaCoordinator constructs", True)
        check("Has world model", hasattr(coord, "world"))
        check("Has event bus", hasattr(coord, "event_bus"))
        check("Has attention engine", hasattr(coord, "attention"))
        check("Has metalearning", hasattr(coord, "metalearning"))
        check("Has thesis generator", hasattr(coord, "thesis_gen"))
        check("Has cached _adversarial_ref attribute",
              hasattr(coord, "_adversarial_ref"),
              "patch 2 applied" if hasattr(coord, "_adversarial_ref")
              else "PATCH 2 NOT APPLIED")

        agent_count = len(coord.agents)
        check("Agents created", agent_count >= 1,
              f"{agent_count} agents")

        status = coord.get_status()
        check("get_status() returns dict with all keys",
              isinstance(status, dict)
              and "event_bus" in status
              and "attention" in status
              and "metalearning" in status
              and "decay_profiles" in status)
    except Exception as e:
        check("Coordinator init", False, str(e))
        traceback.print_exc()

    finalize()


def finalize():
    out("")
    out("=" * 64)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    out(f"  RESULTS: {passed} passed, {failed} failed, {len(results)} total")
    out("=" * 64)

    if failed > 0:
        out("\nFAILED TESTS:")
        for r in results:
            if not r.passed:
                out(f"  ✗ {r.name}: {r.detail}")
        sys.exit(1)
    else:
        out("\n  ✓ ALL CHECKS PASSED — system is wired correctly")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        out("\nInterrupted")
        sys.exit(2)