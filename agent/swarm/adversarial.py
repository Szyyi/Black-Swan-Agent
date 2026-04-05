"""
Adversarial Agent — the philosophical core of Black Swan.

This agent's sole purpose is to BREAK the swarm's consensus.
For every high-conviction edge, it asks:
  "What would need to be true for us to be completely wrong?"

If it finds a plausible failure scenario, it submits a counter-belief
that forces the consensus to incorporate tail risk.

This is NOT a contrarian that bets against the crowd.
This is a red team that stress-tests the swarm's OWN reasoning.

Key differences from the existing contrarian agent:
- Contrarian looks for crowd bias in MARKET prices
- Adversarial looks for groupthink in the SWARM's beliefs
- Contrarian submits alternative estimates
- Adversarial submits specific failure scenarios with kill conditions
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import anthropic
import structlog

from agent.models import Market
from agent.swarm.world_model import Belief, WorldModel

logger = structlog.get_logger()


def safe_parse_json(text: str) -> list | dict | None:
    """Robust JSON parsing from LLM output."""
    import re
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in [r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


@dataclass
class FailureScenario:
    """A specific way the swarm could be catastrophically wrong."""
    market_id: str
    scenario: str
    plausibility: float          # 0-1, how realistic is this scenario
    impact_if_true: float        # 0-1, how wrong would we be
    kill_condition: str          # Observable event that would confirm this scenario
    counter_probability: float   # What probability should be if scenario is real
    invalidates_agents: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PreMortem:
    """A structured pre-mortem analysis of a potential trade."""
    market_id: str
    market_question: str
    swarm_consensus: float
    market_price: float
    failure_scenarios: list[FailureScenario]
    adjusted_probability: float  # Consensus adjusted for tail risk
    risk_rating: str             # "safe", "proceed_with_caution", "dangerous", "abort"
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class AdversarialAgent:
    """
    The swarm's internal red team.

    Runs on two triggers:
    1. SCHEDULED: Every cycle, reviews the top edges and blind spots
    2. EVENT-DRIVEN: When any edge exceeds high conviction, immediate review

    Output: counter-beliefs that encode tail risk, plus pre-mortem reports
    that the coordinator can use to gate trade execution.
    """

    name = "adversarial"
    interval_seconds = 240  # Every 4 minutes

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514"):
        self.world = world
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model
        self._running = False
        self._last_run: float = 0
        self._run_count: int = 0
        self._error_count: int = 0
        self._pre_mortems: dict[str, PreMortem] = {}  # market_id -> latest pre-mortem
        self._failure_log: list[FailureScenario] = []
        self._consensus_overrides: int = 0  # How many times we changed a decision

    async def run_cycle(self, markets: list[Market]):
        """Main adversarial cycle: attack the swarm's strongest convictions."""

        # === Phase 1: Attack blind spots (unanimous agreement) ===
        blind_spots = self.world.check_blind_spots()
        for spot in blind_spots[:3]:
            await self._attack_blind_spot(spot)

        # === Phase 2: Stress-test top edges ===
        edges = self.world.compute_edges(min_edge_pct=3.0)
        high_conviction = [e for e in edges if e.conviction > 50]
        for edge in high_conviction[:3]:
            await self._stress_test_edge(edge)

        # === Phase 3: Check for correlated failure modes ===
        if len(high_conviction) >= 2:
            await self._check_correlated_failures(high_conviction[:5])

        # === Phase 4: Detect regime blindness ===
        await self._detect_regime_blindness(markets)

    async def _attack_blind_spot(self, spot: dict):
        """
        When ALL agents agree strongly against the market,
        that's the most dangerous moment — ask WHY the market disagrees.
        """
        market_id = spot["market_id"]
        question = spot.get("question", "")
        consensus = spot["consensus"]
        market_price = spot["market_price"]
        edge = spot["edge"]

        beliefs = self.world.get_belief_summary(market_id)
        agent_views = "\n".join(
            f"- {b['agent']}: {b['probability']:.0%} (conf: {b['confidence']:.0%}) — {b['reasoning'][:80]}"
            for b in beliefs.get("beliefs", [])
        )

        prompt = f"""You are the adversarial red team for an AI trading swarm.
Your job is to find reasons the swarm could be CATASTROPHICALLY WRONG.

SITUATION: All agents unanimously agree on a position that disagrees with the market.
This is the most dangerous configuration — it means either:
(a) The swarm has found a genuine edge, OR
(b) The swarm has a shared blind spot that the market sees but the agents don't.

MARKET: {question}
MARKET PRICE: {market_price:.0%}
SWARM CONSENSUS: {consensus:.0%} (ALL agents agree)
EDGE: {edge:.1f}%

AGENT REASONING:
{agent_views}

YOUR TASK — Devil's Advocate Analysis:
1. WHY might the market be right and ALL our agents wrong?
   - What information could market participants have that our agents don't?
   - What assumptions are ALL our agents sharing that might be wrong?
   - What recent event could have changed things that our news hasn't captured?

2. CONSTRUCT a specific scenario where the swarm loses badly.
   - Be concrete: name specific events, dates, actors.
   - How plausible is this scenario? (0.0 = impossible, 1.0 = certain)

3. Define a KILL CONDITION — an observable event that would confirm the failure scenario.
   - This should be something we can check for.

4. What ADJUSTED probability accounts for this tail risk?

Respond with JSON only:
{{
  "failure_scenario": "<specific concrete scenario where we're wrong>",
  "plausibility": <float 0.05-0.8>,
  "impact_if_true": <float 0.5-1.0>,
  "kill_condition": "<observable event that confirms failure>",
  "adjusted_probability": <float 0.02-0.98>,
  "shared_assumption": "<the assumption all agents share that might be wrong>",
  "missing_information": "<what the market might know that we don't>",
  "risk_rating": "safe" | "proceed_with_caution" | "dangerous" | "abort"
}}

IMPORTANT: You are NOT trying to be contrarian for its own sake.
You are trying to find GENUINE risks the swarm is missing.
If the swarm really does have a valid edge, say so — rate it "safe" with high plausibility of being right."""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=800,
                temperature=0.9,  # High creativity for adversarial thinking
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, dict):
                return

            plausibility = float(result.get("plausibility", 0.1))
            adjusted = float(result.get("adjusted_probability", consensus))
            risk_rating = result.get("risk_rating", "proceed_with_caution")

            scenario = FailureScenario(
                market_id=market_id,
                scenario=result.get("failure_scenario", ""),
                plausibility=plausibility,
                impact_if_true=float(result.get("impact_if_true", 0.5)),
                kill_condition=result.get("kill_condition", ""),
                counter_probability=adjusted,
            )
            self._failure_log.append(scenario)
            self._failure_log = self._failure_log[-50:]

            # Store pre-mortem
            self._pre_mortems[market_id] = PreMortem(
                market_id=market_id,
                market_question=question,
                swarm_consensus=consensus,
                market_price=market_price,
                failure_scenarios=[scenario],
                adjusted_probability=adjusted,
                risk_rating=risk_rating,
                reasoning=result.get("shared_assumption", ""),
            )

            # === Submit counter-belief if scenario is plausible ===
            if plausibility > 0.15 and abs(adjusted - consensus) > 0.03:
                # Weight the counter-belief by plausibility
                # A 30% plausible failure scenario with major impact
                # should meaningfully shift consensus
                counter_conf = min(0.7, plausibility * float(result.get("impact_if_true", 0.5)))

                self.world.submit_belief(Belief(
                    agent_name=self.name,
                    market_id=market_id,
                    probability=adjusted,
                    confidence=counter_conf,
                    reasoning=(
                        f"ADVERSARIAL [{risk_rating}]: {result.get('failure_scenario', '')[:80]} "
                        f"| Kill condition: {result.get('kill_condition', '')[:50]}"
                    ),
                    evidence=[
                        result.get("shared_assumption", ""),
                        result.get("missing_information", ""),
                        f"plausibility={plausibility:.0%}",
                    ],
                    ttl_seconds=900,  # 15 min — adversarial views need frequent refresh
                ))
                self._consensus_overrides += 1

                print(
                    f"  [ADVERSARIAL] Challenging blind spot on '{question[:40]}'\n"
                    f"    Scenario: {result.get('failure_scenario', '')[:70]}\n"
                    f"    Risk: {risk_rating} | Plausibility: {plausibility:.0%}\n"
                    f"    Adjusting {consensus:.0%} -> {adjusted:.0%}\n",
                    flush=True,
                )

        except Exception as e:
            logger.debug("adversarial_blind_spot_error", error=str(e))

    async def _stress_test_edge(self, edge):
        """For each high-conviction edge, run a pre-mortem."""
        market_id = edge.market_id
        question = edge.market_question
        beliefs = self.world.get_belief_summary(market_id)

        agent_views = "\n".join(
            f"- {b['agent']}: {b['probability']:.0%} — {b['reasoning'][:60]}"
            for b in beliefs.get("beliefs", [])
        )

        prompt = f"""You are conducting a PRE-MORTEM on a trade our AI swarm is about to execute.

PROPOSED TRADE: {edge.direction} on "{question}"
EDGE: {edge.edge_pct:.1f}%
CONVICTION: {edge.conviction:.0f}/100
MARKET PRICE: {edge.market_price:.0%}
SWARM'S FAIR VALUE: {edge.fair_value:.0%}

AGENT CONTRIBUTIONS:
{agent_views}

PRE-MORTEM EXERCISE:
Imagine it's 2 weeks from now and this trade LOST MONEY. What went wrong?

Think about:
1. What SPECIFIC event caused the loss?
2. Which agent's reasoning was flawed, and why?
3. Was there a data gap — information we should have had but didn't?
4. Was there a timing issue — right thesis but wrong timing?
5. Was there a structural issue — e.g., market manipulation, low liquidity?

Rate the overall risk and suggest whether to proceed.

Respond with JSON only:
{{
  "most_likely_failure": "<what specifically goes wrong>",
  "vulnerable_agent": "<which agent's logic is weakest here>",
  "data_gap": "<information we're missing>",
  "timing_risk": "<timing-related concerns>",
  "structural_risk": "<market structure concerns>",
  "risk_rating": "safe" | "proceed_with_caution" | "dangerous" | "abort",
  "suggested_size_adjustment": <float 0.0-1.5 where 1.0=no change>,
  "reasoning": "<1-2 sentence summary>"
}}"""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=600,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, dict):
                return

            risk_rating = result.get("risk_rating", "proceed_with_caution")
            size_adj = float(result.get("suggested_size_adjustment", 1.0))

            self._pre_mortems[market_id] = PreMortem(
                market_id=market_id,
                market_question=question,
                swarm_consensus=edge.fair_value,
                market_price=edge.market_price,
                failure_scenarios=[FailureScenario(
                    market_id=market_id,
                    scenario=result.get("most_likely_failure", ""),
                    plausibility=0.3,  # Default for stress tests
                    impact_if_true=0.7,
                    kill_condition=result.get("data_gap", ""),
                    counter_probability=edge.market_price,
                    invalidates_agents=[result.get("vulnerable_agent", "")],
                )],
                adjusted_probability=edge.fair_value,
                risk_rating=risk_rating,
                reasoning=result.get("reasoning", ""),
            )

            # If dangerous or abort, submit a dampening belief
            if risk_rating in ("dangerous", "abort"):
                dampen_prob = (edge.fair_value + edge.market_price) / 2  # Pull toward market
                self.world.submit_belief(Belief(
                    agent_name=self.name,
                    market_id=market_id,
                    probability=dampen_prob,
                    confidence=0.5 if risk_rating == "dangerous" else 0.7,
                    reasoning=(
                        f"PRE-MORTEM [{risk_rating}]: {result.get('most_likely_failure', '')[:70]} "
                        f"| Vulnerable: {result.get('vulnerable_agent', 'unknown')}"
                    ),
                    evidence=[
                        result.get("data_gap", ""),
                        result.get("timing_risk", ""),
                        result.get("structural_risk", ""),
                    ],
                    ttl_seconds=600,
                ))
                self._consensus_overrides += 1

                print(
                    f"  [ADVERSARIAL] Pre-mortem WARNING on '{question[:40]}'\n"
                    f"    Risk: {risk_rating} | Failure: {result.get('most_likely_failure', '')[:60]}\n"
                    f"    Weakest link: {result.get('vulnerable_agent', '?')}\n",
                    flush=True,
                )

        except Exception as e:
            logger.debug("adversarial_stress_test_error", error=str(e))

    async def _check_correlated_failures(self, edges: list):
        """
        The most dangerous scenario: multiple positions that ALL fail together.
        Check if our top edges share a common failure mode.
        """
        if len(edges) < 2:
            return

        edge_descriptions = "\n".join(
            f"- {e.direction} '{e.market_question[:50]}' (edge {e.edge_pct:.1f}%, conv {e.conviction:.0f})"
            for e in edges
        )

        prompt = f"""You are a portfolio risk analyst for an AI trading swarm.

We are about to take MULTIPLE positions simultaneously:
{edge_descriptions}

CRITICAL QUESTION: Is there a single event or scenario that would cause ALL of these positions to lose simultaneously?

Think about:
1. Common underlying factors (e.g., all depend on same political outcome)
2. Market-wide events (e.g., exchange outage, liquidity crisis)
3. Correlated information sources (e.g., all based on same news that turns out wrong)
4. Timing correlations (e.g., all expire around same event)

Respond with JSON only:
{{
  "correlated_failure_exists": true | false,
  "common_factor": "<what links these positions>",
  "failure_scenario": "<specific event that kills all positions>",
  "correlation_strength": <float 0-1>,
  "recommendation": "<what to do about it>"
}}"""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=400,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, dict):
                return

            if result.get("correlated_failure_exists") and float(result.get("correlation_strength", 0)) > 0.5:
                print(
                    f"  [ADVERSARIAL] CORRELATED RISK DETECTED\n"
                    f"    Common factor: {result.get('common_factor', '')[:60]}\n"
                    f"    Scenario: {result.get('failure_scenario', '')[:60]}\n"
                    f"    Correlation: {float(result.get('correlation_strength', 0)):.0%}\n"
                    f"    Action: {result.get('recommendation', '')[:60]}\n",
                    flush=True,
                )

        except Exception as e:
            logger.debug("adversarial_correlated_error", error=str(e))

    async def _detect_regime_blindness(self, markets: list[Market]):
        """
        Check if the swarm is applying the wrong mental model.
        E.g., using stable-market logic in a volatile regime.
        """
        regime_mismatches = 0
        for market in markets[:20]:
            if not market.active:
                continue
            mid = market.condition_id
            beliefs = self.world._get_valid_beliefs(mid)
            if len(beliefs) < 2:
                continue

            regime = self.world._regimes.get(mid)
            if not regime:
                continue

            # Check if beliefs have low spread (stable-market logic)
            # but regime is volatile
            probs = [b.probability for b in beliefs]
            if len(probs) >= 2:
                from statistics import stdev
                spread = stdev(probs)
                if regime.regime == "volatile" and spread < 0.05:
                    regime_mismatches += 1
                    # Agents are too confident in a volatile market
                    self.world.submit_belief(Belief(
                        agent_name=self.name,
                        market_id=mid,
                        probability=self.world.get_market_price(mid) or 0.5,
                        confidence=0.3,
                        reasoning=(
                            f"REGIME WARNING: Market is volatile but all agents "
                            f"agree within {spread:.0%} spread — likely overconfident"
                        ),
                        ttl_seconds=300,
                    ))

        if regime_mismatches > 0:
            print(f"  [ADVERSARIAL] {regime_mismatches} markets with regime blindness\n", flush=True)

    # ── Lifecycle (matches SwarmAgent interface) ───────

    async def start(self, markets: list[Market]):
        self._running = True
        import random
        await asyncio.sleep(random.uniform(1, 10))
        while self._running:
            if time.time() - self._last_run >= self.interval_seconds:
                try:
                    print(f"    [{self.name}] starting cycle...", flush=True)
                    await asyncio.wait_for(self.run_cycle(markets), timeout=60)
                    self._run_count += 1
                    self._last_run = time.time()
                    print(f"    [{self.name}] cycle done (run #{self._run_count})", flush=True)
                except asyncio.TimeoutError:
                    self._error_count += 1
                    print(f"    [{self.name}] TIMED OUT after 60s", flush=True)
                except Exception as e:
                    self._error_count += 1
                    print(f"    [{self.name}] ERROR: {e}", flush=True)
            await asyncio.sleep(1)

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "runs": self._run_count,
            "errors": self._error_count,
            "last_run_ago": round(time.time() - self._last_run, 1) if self._last_run else None,
            "pre_mortems": len(self._pre_mortems),
            "consensus_overrides": self._consensus_overrides,
            "failure_scenarios_logged": len(self._failure_log),
        }

    def get_pre_mortem(self, market_id: str) -> PreMortem | None:
        """Called by coordinator before executing a trade."""
        return self._pre_mortems.get(market_id)

    def get_risk_rating(self, market_id: str) -> str:
        """Quick check: should we proceed with this trade?"""
        pm = self._pre_mortems.get(market_id)
        if pm is None:
            return "unreviewed"
        return pm.risk_rating