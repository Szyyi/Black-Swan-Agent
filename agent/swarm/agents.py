"""
Swarm agents v2: sophisticated multi-agent intelligence system.

Upgrades over v1:
- Structured chain-of-thought prompting with Fermi decomposition
- 5 analytical perspectives instead of 3
- Robust JSON parsing with fallback extraction
- Per-agent calibration tracking (learns which agents are best at what)
- Momentum detection agent (new)
- Smarter news scout with tiered urgency and multi-source cross-referencing
- Contrarian with structured debiasing framework
- Social signals agent that infers sentiment from news tone
- Better error recovery — agents never crash the swarm
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean, stdev

import anthropic
import structlog

from agent.data.feeds import NewsIngester
from agent.data.smarkets_client import SmarketsClient
from agent.models import Market
from agent.swarm.world_model import (
    Belief, Correlation, NewsImpact, TimingSignal, WorldModel,
)

logger = structlog.get_logger()


# ── Utilities ──────────────────────────────────────────

def safe_parse_json(text: str) -> list | dict | None:
    """
    Robustly parse JSON from LLM output.
    Handles markdown fences, preamble text, and common formatting issues.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Remove markdown fences
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object in the text
    for pattern in [
        r"(\[[\s\S]*\])",   # Find [...] anywhere
        r"(\{[\s\S]*\})",   # Find {...} anywhere
    ]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


class CalibrationTracker:
    """
    Tracks an agent's prediction accuracy over time.
    Used to weight agents dynamically — accurate agents get more influence.
    """

    def __init__(self):
        self.predictions: list[dict] = []  # {market_id, predicted, actual, timestamp}
        self.domain_accuracy: dict[str, list[float]] = defaultdict(list)

    def record_prediction(self, market_id: str, predicted: float, domain: str = "general"):
        self.predictions.append({
            "market_id": market_id,
            "predicted": predicted,
            "domain": domain,
            "timestamp": time.time(),
            "actual": None,  # Filled in when market resolves
        })

    def record_resolution(self, market_id: str, actual: float):
        for pred in reversed(self.predictions):
            if pred["market_id"] == market_id and pred["actual"] is None:
                pred["actual"] = actual
                error = abs(pred["predicted"] - actual)
                accuracy = 1.0 - error
                self.domain_accuracy[pred["domain"]].append(accuracy)
                break

    def get_accuracy(self, domain: str = "general") -> float:
        scores = self.domain_accuracy.get(domain, [])
        if len(scores) < 3:
            return 0.5  # Default until we have enough data
        return mean(scores[-20:])  # Rolling 20 predictions

    def get_confidence_weight(self, domain: str = "general") -> float:
        """Returns a multiplier 0.5-1.5 based on track record."""
        acc = self.get_accuracy(domain)
        return 0.5 + acc  # 0.5 accuracy -> 1.0 weight, 1.0 accuracy -> 1.5 weight


# ── Base Agent ─────────────────────────────────────────

class SwarmAgent(ABC):
    """Base class for all swarm agents."""

    name: str
    interval_seconds: float = 60

    def __init__(self, world: WorldModel):
        self.world = world
        self._running = False
        self._last_run: float = 0
        self._run_count: int = 0
        self._error_count: int = 0
        self.calibration = CalibrationTracker()

    @abstractmethod
    async def run_cycle(self, markets: list[Market]):
        ...

    async def start(self, markets: list[Market]):
        self._running = True
        import random
        import sys
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
        }


# ══════════════════════════════════════════════════════
#  NEWS SCOUT — sophisticated news analysis
# ══════════════════════════════════════════════════════

class NewsScoutAgent(SwarmAgent):
    """
    Scans news feeds and performs multi-layer impact analysis:
    1. Relevance scoring — which markets does this news affect?
    2. Impact magnitude — how much should probabilities shift?
    3. Urgency classification — how fast is the market likely to react?
    4. Cross-reference — does this confirm or contradict existing beliefs?
    """

    name = "news_scout"
    interval_seconds = 120

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 news_sources: list[str] | None = None):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model
        self.news = NewsIngester(sources=news_sources or [])
        self._seen_headlines: set[str] = set()
        self._headline_history: list[dict] = []  # Track narrative arcs

    async def run_cycle(self, markets: list[Market]):
        articles = await self.news.get_latest_news()
        new_articles = [a for a in articles if a["title"] not in self._seen_headlines]
        if not new_articles:
            return

        for a in new_articles[:10]:
            self._seen_headlines.add(a["title"])
            self._headline_history.append({"title": a["title"], "ts": time.time()})

        # Keep last 200 headlines for narrative tracking
        self._headline_history = self._headline_history[-200:]

        market_list = "\n".join(
            f"- ID:{m.condition_id[:8]} | {m.question} | "
            f"Price: {self.world.get_market_price(m.condition_id) or '?'} | "
            f"Category: {m.category}"
            for m in markets[:40] if m.active
        )

        news_block = "\n".join(
            f"[{i+1}] {a['title']}\n    {a['summary'][:200]}"
            for i, a in enumerate(new_articles[:8])
        )

        # Include recent headline history for narrative context
        recent_headlines = "\n".join(
            f"- {h['title']}" for h in self._headline_history[-15:-len(new_articles)]
        ) if len(self._headline_history) > len(new_articles) else ""

        prompt = f"""You are an expert prediction market news analyst working for a quantitative trading desk.

TASK: Analyse these news items and determine their impact on active prediction markets.

BREAKING / RECENT NEWS:
{news_block}

RECENT HEADLINE CONTEXT (for narrative tracking):
{recent_headlines if recent_headlines else "No prior context yet."}

ACTIVE PREDICTION MARKETS:
{market_list}

ANALYSIS FRAMEWORK:
For each news item that affects a market, assess:
1. RELEVANCE: Is this directly about the market topic, tangentially related, or background context?
2. DIRECTION: Does this make YES more or less likely? By how much?
3. URGENCY: Will the market react in minutes (breaking), hours (developing), or days (background)?
4. NARRATIVE: Does this reinforce an existing trend or represent a reversal?
5. CONFIDENCE: How certain are you of this assessment?

Respond with a JSON array only, no other text:
[{{
  "headline": "<headline text>",
  "affected_market_ids": ["<8-char-id>"],
  "impact": {{"<market_id>": <float -0.25 to +0.25>}},
  "urgency": <float 0-1 where 1=breaking>,
  "confidence": <float 0-1>,
  "reasoning": "<2 sentences: what changed and why it matters>",
  "narrative": "reinforcing" | "reversal" | "new_information"
}}]

RULES:
- Be conservative: most news has NO meaningful impact on most markets.
- Impact > 0.10 requires strong, direct evidence.
- If no news affects any market, respond with []
- NEVER fabricate market IDs. Only use IDs from the list above."""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, list):
                result = []

            for imp in result:
                if not imp.get("affected_market_ids"):
                    continue

                self.world.submit_news_impact(NewsImpact(
                    headline=imp.get("headline", ""),
                    source="news_scout",
                    affected_markets=imp["affected_market_ids"],
                    impact_direction=imp.get("impact", {}),
                    urgency=float(imp.get("urgency", 0.5)),
                    confidence=float(imp.get("confidence", 0.5)),
                    detected_by=self.name,
                ))

                # Submit beliefs for affected markets
                for mid, shift in imp.get("impact", {}).items():
                    current = self.world.get_market_price(mid)
                    if current is not None:
                        new_prob = max(0.01, min(0.99, current + shift))
                        # Narrative reversals get higher confidence
                        conf_mult = 1.2 if imp.get("narrative") == "reversal" else 1.0
                        self.world.submit_belief(Belief(
                            agent_name=self.name,
                            market_id=mid,
                            probability=new_prob,
                            confidence=float(imp.get("confidence", 0.5)) * 0.8 * conf_mult,
                            reasoning=f"News: {imp.get('headline', '')[:60]} ({imp.get('narrative', 'new')})",
                            evidence=[imp.get("headline", ""), imp.get("reasoning", "")],
                            ttl_seconds=1800 if imp.get("urgency", 0.5) > 0.7 else 3600,
                        ))

            logger.info("news_scout_cycle", new_articles=len(new_articles), impacts=len(result))

        except Exception as e:
            logger.error("news_scout_error", error=str(e))


# ══════════════════════════════════════════════════════
#  MARKET SCANNER — price anomaly detection
# ══════════════════════════════════════════════════════

class MarketScannerAgent(SwarmAgent):
    """
    Monitors all markets for:
    - Sudden price moves (momentum signals)
    - Price stagnation at round numbers (anchoring)
    - Volume spikes (smart money moving)
    - Markets approaching expiry (time decay)
    """

    name = "market_scanner"
    interval_seconds = 30

    def __init__(self, world: WorldModel, smarkets_client: SmarketsClient):
        super().__init__(world)
        self.smarkets = smarkets_client
        self._price_history: dict[str, list[tuple[float, float]]] = {}
        self._volume_history: dict[str, list[tuple[float, float]]] = {}

    async def run_cycle(self, markets: list[Market]):
        scanned = 0
        anomalies = 0

        for market in markets:
            if not market.active or len(market.token_ids) < 1:
                continue

            mid = market.condition_id
            try:
                price = await self.smarkets.get_price(market.token_ids[0], mid)
                if price is None:
                    continue

                scanned += 1
                self.world.update_market_price(mid, price, question=market.question, volume=market.volume,
                    category=market.category)

                # Track price history
                if mid not in self._price_history:
                    self._price_history[mid] = []
                self._price_history[mid].append((time.time(), price))

                # Keep last 2 hours
                cutoff = time.time() - 7200
                self._price_history[mid] = [
                    (ts, p) for ts, p in self._price_history[mid] if ts > cutoff
                ]

                history = self._price_history[mid]

                # ── Momentum detection ──
                if len(history) >= 5:
                    recent = [p for _, p in history[-5:]]
                    older = [p for _, p in history[:-5]] if len(history) > 5 else [recent[0]]
                    avg_recent = mean(recent)
                    avg_older = mean(older)
                    move = avg_recent - avg_older
                    abs_move = abs(move)

                    if abs_move > 0.03:
                        anomalies += 1
                        urgency = min(1.0, abs_move * 8)
                        action = "enter" if abs_move > 0.08 else "scale_up"

                        self.world.submit_timing_signal(TimingSignal(
                            market_id=mid,
                            action=action,
                            urgency=urgency,
                            reasoning=f"Price {'up' if move > 0 else 'down'} {abs_move:.1%} — momentum detected",
                            expires_at=time.time() + 300,
                            detected_by=self.name,
                        ))

                # ── Round number anchoring detection ──
                if len(history) >= 10:
                    prices = [p for _, p in history[-10:]]
                    price_stdev = stdev(prices) if len(prices) >= 2 else 0
                    # If price is stuck near a round number with low volatility
                    nearest_round = round(price * 10) / 10  # Nearest 10%
                    dist_to_round = abs(price - nearest_round)
                    if price_stdev < 0.01 and dist_to_round < 0.02:
                        self.world.submit_timing_signal(TimingSignal(
                            market_id=mid,
                            action="enter",
                            urgency=0.4,
                            reasoning=f"Price anchored at {nearest_round:.0%} — potential mispricing",
                            expires_at=time.time() + 600,
                            detected_by=self.name,
                        ))

            except Exception as e:
                logger.debug("scanner_error", market=mid[:8], error=str(e))

        if scanned > 0:
            logger.debug("market_scanner_cycle", scanned=scanned, anomalies=anomalies)


# ══════════════════════════════════════════════════════
#  PROBABILITY ESTIMATOR — the core forecasting engine
# ══════════════════════════════════════════════════════

class ProbabilityEstimatorAgent(SwarmAgent):
    """
    Sophisticated ensemble forecaster using 5 analytical perspectives.
    Each perspective uses chain-of-thought reasoning and Fermi decomposition.
    Results are combined with confidence-weighted averaging and agreement bonuses.
    """

    name = "probability_estimator"
    interval_seconds = 300

    PERSPECTIVES = [
        {
            "name": "base_rate_analyst",
            "instruction": """Think like a base rate statistician.
Step 1: What is the reference class for this type of event?
Step 2: What is the historical base rate? (How often do events like this resolve YES?)
Step 3: What specific evidence shifts the probability from the base rate?
Step 4: Apply Laplace's rule — with limited data, regress toward 50%.""",
            "temperature": 0.5,
        },
        {
            "name": "evidence_weigher",
            "instruction": """Think like an investigative analyst.
Step 1: List the 3 strongest pieces of evidence FOR this outcome.
Step 2: List the 3 strongest pieces of evidence AGAINST.
Step 3: Assign a weight (1-10) to each piece of evidence based on reliability.
Step 4: Calculate a weighted probability.""",
            "temperature": 0.6,
        },
        {
            "name": "contrarian_thinker",
            "instruction": """Think like a professional contrarian.
Step 1: What does the crowd believe? (The market price tells you.)
Step 2: Why might the crowd be wrong? Consider anchoring, recency bias, bandwagon effects.
Step 3: What scenario would cause the biggest surprise?
Step 4: Adjust your estimate to account for crowd biases.""",
            "temperature": 0.8,
        },
        {
            "name": "scenario_planner",
            "instruction": """Think like a scenario planner.
Step 1: Define 3 scenarios — bull case, base case, bear case for YES.
Step 2: Assign a probability to each scenario occurring.
Step 3: For each scenario, what is the probability of YES?
Step 4: Calculate the weighted average: P(YES) = sum(P(scenario) * P(YES|scenario)).""",
            "temperature": 0.7,
        },
        {
            "name": "time_decay_analyst",
            "instruction": """Think like a time-value analyst.
Step 1: When does this market resolve? How much time remains?
Step 2: What would need to happen for YES to win? Is there enough time?
Step 3: What is the current trajectory? If trends continue, what happens?
Step 4: Factor in that unlikely events become less likely as time runs out.""",
            "temperature": 0.6,
        },
    ]

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 ensemble_size: int = 3):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model
        self.ensemble_size = min(ensemble_size, len(self.PERSPECTIVES))

    async def run_cycle(self, markets: list[Market]):
        # Prioritize markets with news impacts, timing signals, or high volume
        scored_markets = []
        for m in markets:
            if not m.active:
                continue
            score = max(m.volume / 1000, 0.1)
            news = self.world.get_news_for_market(m.condition_id)
            if news:
                score += len(news) * 10
            signals = self.world.get_timing_signals(m.condition_id)
            if signals:
                score += sum(s.urgency for s in signals) * 5
            scored_markets.append((score, m))

        scored_markets.sort(key=lambda x: x[0], reverse=True)
        batch = [m for _, m in scored_markets[:3]]

        for market in batch:
            await self._estimate_market(market)

    async def _estimate_market(self, market: Market):
        price = self.world.get_market_price(market.condition_id)
        if price is None:
            return

        # Gather context from other agents
        news = self.world.get_news_for_market(market.condition_id)
        news_context = ""
        if news:
            news_context = "RELEVANT NEWS:\n" + "\n".join(
                f"- {n.headline} (urgency: {n.urgency:.0%}, confidence: {n.confidence:.0%})"
                for n in news[:5]
            )

        correlations = self.world.get_correlated_markets(market.condition_id)
        corr_context = ""
        if correlations:
            corr_context = "CORRELATED MARKETS:\n" + "\n".join(
                f"- {c.description} ({c.correlation_type}, strength: {c.strength:.0%})"
                for c in correlations[:3]
            )

        # Select perspectives (rotate through them across runs)
        perspective_indices = [
            (self._run_count * self.ensemble_size + i) % len(self.PERSPECTIVES)
            for i in range(self.ensemble_size)
        ]

        tasks = [
            self._single_estimate(market, price, self.PERSPECTIVES[idx], news_context, corr_context)
            for idx in perspective_indices
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        estimates = [r for r in results if isinstance(r, dict) and "probability" in r]
        if not estimates:
            return

        probs = [float(e["probability"]) for e in estimates]
        confs = [float(e.get("confidence", 0.5)) for e in estimates]

        # Confidence-weighted ensemble
        total_w = sum(confs)
        if total_w == 0:
            return

        ensemble_prob = sum(p * c for p, c in zip(probs, confs)) / total_w
        ensemble_conf = mean(confs)

        # Agreement bonus / penalty
        if len(probs) >= 2:
            spread = stdev(probs)
            if spread < 0.05:  # Strong agreement
                ensemble_conf = min(1.0, ensemble_conf * 1.4)
            elif spread < 0.10:  # Moderate agreement
                ensemble_conf = min(1.0, ensemble_conf * 1.15)
            elif spread > 0.20:  # Major disagreement — reduce confidence
                ensemble_conf *= 0.6

        # Track for calibration
        self.calibration.record_prediction(market.condition_id, ensemble_prob, market.category)

        perspectives_used = [self.PERSPECTIVES[i]["name"] for i in perspective_indices]

        self.world.submit_belief(Belief(
            agent_name=self.name,
            market_id=market.condition_id,
            probability=ensemble_prob,
            confidence=ensemble_conf,
            reasoning=(
                f"Ensemble ({', '.join(perspectives_used)}): "
                f"estimates=[{', '.join(f'{p:.0%}' for p in probs)}] "
                f"spread={stdev(probs) if len(probs) >= 2 else 0:.3f}"
            ),
            evidence=[e.get("reasoning", "") for e in estimates],
            ttl_seconds=600,
        ))

        logger.info(
            "probability_estimate",
            market=market.question[:50],
            estimates=[round(p, 3) for p in probs],
            ensemble=round(ensemble_prob, 3),
            market_price=price,
            edge=round((ensemble_prob - price) * 100, 1),
            confidence=round(ensemble_conf, 2),
        )

    async def _single_estimate(self, market: Market, price: float,
                                perspective: dict,
                                news_context: str, corr_context: str) -> dict:
        prompt = f"""You are a world-class calibrated superforecaster. Your track record shows
you are in the top 2% of forecasters globally.

QUESTION: {market.question}
CURRENT MARKET PRICE: {price:.2f} ({price:.0%} implied probability)
CATEGORY: {market.category}
END DATE: {market.end_date or 'Unknown'}

{news_context}
{corr_context}

YOUR ANALYTICAL PERSPECTIVE: {perspective['name']}
{perspective['instruction']}

CALIBRATION REMINDERS:
- If you think something is 90% likely, it should happen 9 out of 10 times.
- The market price reflects the wisdom of many traders. Respect it as a prior.
- Extraordinary claims (>15% deviation from market) require extraordinary evidence.
- When uncertain, stay closer to the market price.
- Think about what information you might be missing.

Show your reasoning step by step, then provide your final answer.

End your response with EXACTLY this JSON on its own line (no markdown):
{{"probability": <float 0.02-0.98>, "confidence": <float 0.1-0.9>, "reasoning": "<your key insight in 1-2 sentences>"}}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=perspective.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Extract JSON from end of response (after chain-of-thought)
            result = safe_parse_json(text)
            if isinstance(result, dict) and "probability" in result:
                return result

            # Try to find JSON at the end of the text
            lines = text.split("\n")
            for line in reversed(lines):
                parsed = safe_parse_json(line.strip())
                if isinstance(parsed, dict) and "probability" in parsed:
                    return parsed

            return {}
        except Exception as e:
            logger.debug("estimate_error", perspective=perspective["name"], error=str(e))
            return {}


# ══════════════════════════════════════════════════════
#  CORRELATION DETECTIVE — finds hidden market links
# ══════════════════════════════════════════════════════

class CorrelationDetectiveAgent(SwarmAgent):
    """
    Finds cross-market correlations using causal reasoning.
    Goes beyond simple keyword matching — looks for:
    - Causal chains (A causes B which affects C)
    - Shared underlying factors
    - Conditional dependencies
    - Temporal sequences (if A happens first, B becomes likely)
    """

    name = "correlation_detective"
    interval_seconds = 600

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514"):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model
        self._known_correlations: set[tuple[str, str]] = set()

    async def run_cycle(self, markets: list[Market]):
        active = [m for m in markets if m.active]
        active.sort(key=lambda m: m.volume, reverse=True)
        batch = active[:25]

        if len(batch) < 3:
            return

        market_descriptions = "\n".join(
            f"[{m.condition_id[:8]}] {m.question} (category: {m.category}, "
            f"price: {self.world.get_market_price(m.condition_id) or '?'})"
            for m in batch
        )

        prompt = f"""You are a causal reasoning expert specializing in prediction markets.

MARKETS:
{market_descriptions}

TASK: Find pairs of markets where the outcome of one LOGICALLY affects the probability of another.

Think through these types of connections:
1. CAUSAL: X directly causes Y (e.g., "PM resigns" → "snap election called")
2. SHARED FACTOR: Both depend on the same underlying variable (e.g., both affected by inflation)
3. CONDITIONAL: If X happens, Y becomes much more/less likely
4. OPPOSING: X and Y are mutually exclusive or inversely related
5. TEMPORAL: X typically precedes Y in a causal chain

For each correlation, explain the MECHANISM — why does one affect the other?

Respond with a JSON array only:
[{{
  "market_a_id": "<8-char-id from list above>",
  "market_b_id": "<8-char-id from list above>",
  "type": "positive" | "negative" | "conditional",
  "strength": <float 0.3-1.0>,
  "mechanism": "<1-2 sentences explaining the causal link>",
  "description": "<short label for this correlation>"
}}]

Only include correlations with strength >= 0.4 and clear causal mechanisms. Max 6 pairs.
If no meaningful correlations exist, respond with []."""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, list):
                result = []

            new_correlations = 0
            for corr in result:
                a_id = self._resolve_id(corr.get("market_a_id", ""), batch)
                b_id = self._resolve_id(corr.get("market_b_id", ""), batch)
                if not a_id or not b_id or a_id == b_id:
                    continue

                pair_key = tuple(sorted([a_id, b_id]))
                if pair_key in self._known_correlations:
                    continue
                self._known_correlations.add(pair_key)
                new_correlations += 1

                self.world.submit_correlation(Correlation(
                    market_a_id=a_id,
                    market_b_id=b_id,
                    market_a_question=self._get_question(a_id, batch),
                    market_b_question=self._get_question(b_id, batch),
                    correlation_type=corr.get("type", "positive"),
                    strength=float(corr.get("strength", 0.5)),
                    description=corr.get("description", corr.get("mechanism", "")),
                    detected_by=self.name,
                ))

            logger.info("correlation_detective_cycle", found=len(result), new=new_correlations)

        except Exception as e:
            logger.error("correlation_error", error=str(e))

    def _resolve_id(self, short_id: str, markets: list[Market]) -> str | None:
        if not short_id:
            return None
        for m in markets:
            if m.condition_id.startswith(short_id):
                return m.condition_id
        return None

    def _get_question(self, market_id: str, markets: list[Market]) -> str:
        for m in markets:
            if m.condition_id == market_id:
                return m.question
        return ""


# ══════════════════════════════════════════════════════
#  CONTRARIAN — structured debiasing
# ══════════════════════════════════════════════════════

class ContrarianAgent(SwarmAgent):
    """
    Hunts for crowd errors using a structured debiasing framework.
    Analyses each market through multiple cognitive bias lenses
    and flags specific mispricings with reasoning.
    """

    name = "contrarian"
    interval_seconds = 900

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514"):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model

    async def run_cycle(self, markets: list[Market]):
        candidates = [m for m in markets if m.active]
        candidates.sort(key=lambda m: m.volume, reverse=True)
        batch = candidates[:20]

        if not batch:
            return

        market_block = "\n".join(
            f"- ID:{m.condition_id[:8]} | {m.question} | "
            f"Price: {self.world.get_market_price(m.condition_id) or '?'} | "
            f"Vol: {m.volume:,.0f} | Cat: {m.category}"
            for m in batch
        )

        prompt = f"""You are a cognitive bias expert and prediction market analyst.
Your job is to find markets where the crowd is SYSTEMATICALLY wrong.

BIAS CHECKLIST — check each market against these:
1. ANCHORING: Is the price stuck near a psychologically significant number?
2. RECENCY BIAS: Has recent news moved the price too far, ignoring base rates?
3. BANDWAGON EFFECT: Has the price moved because people are following the herd?
4. AVAILABILITY BIAS: Is a dramatic recent event making people overestimate probability?
5. NEGLECT OF BASE RATES: Are people ignoring how rarely/commonly this type of event occurs?
6. CONJUNCTION FALLACY: Does YES require multiple independent things to all happen?
7. STATUS QUO BIAS: Are people assuming things won't change when change is likely?
8. OPTIMISM/PESSIMISM BIAS: Is the crowd systematically too hopeful or too fearful?

MARKETS:
{market_block}

TASK: Find 1-4 markets where you believe a specific bias is causing meaningful mispricing (5%+ error).
For each, explain WHICH bias, WHY you think the crowd is wrong, and what the correct price should be.

Respond with JSON array only:
[{{
  "market_id": "<8-char-id>",
  "question": "<market question>",
  "current_price": <float>,
  "your_estimate": <float 0.02-0.98>,
  "bias_detected": "<specific bias name>",
  "confidence": <float 0.3-0.9>,
  "reasoning": "<2-3 sentences: what bias is at play and why the true probability differs>"
}}]

BE HIGHLY SELECTIVE. Most markets are approximately correctly priced.
Only flag genuine mispricings where you have strong reasoning.
If nothing is clearly mispriced, respond with []."""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=1500,
                temperature=0.8,  # Higher temp for creative contrarian thinking
                messages=[{"role": "user", "content": prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, list):
                result = []

            submitted = 0
            for finding in result:
                # Match to a market
                market_id_hint = finding.get("market_id", "")
                matched = None
                for m in batch:
                    if m.condition_id.startswith(market_id_hint):
                        matched = m
                        break
                if not matched:
                    # Try fuzzy match on question
                    q = finding.get("question", "").lower()
                    for m in batch:
                        if q and (q in m.question.lower() or m.question.lower() in q):
                            matched = m
                            break
                if not matched:
                    continue

                estimate = float(finding.get("your_estimate", 0.5))
                confidence = float(finding.get("confidence", 0.5))
                current = self.world.get_market_price(matched.condition_id)

                # Only submit if the edge is meaningful
                if current and abs(estimate - current) < 0.03:
                    continue

                self.world.submit_belief(Belief(
                    agent_name=self.name,
                    market_id=matched.condition_id,
                    probability=estimate,
                    confidence=confidence * 0.85,  # Slight discount for contrarian views
                    reasoning=(
                        f"CONTRARIAN [{finding.get('bias_detected', 'bias')}]: "
                        f"{finding.get('reasoning', '')}"
                    ),
                    evidence=[finding.get("bias_detected", ""), finding.get("reasoning", "")],
                    ttl_seconds=1800,
                ))
                submitted += 1

                self.calibration.record_prediction(matched.condition_id, estimate, matched.category)

            logger.info("contrarian_cycle", analysed=len(batch), findings=len(result), submitted=submitted)

        except Exception as e:
            logger.error("contrarian_error", error=str(e))


# ══════════════════════════════════════════════════════
#  MOMENTUM AGENT — detects and trades price trends
# ══════════════════════════════════════════════════════

class MomentumAgent(SwarmAgent):
    """
    Detects markets where prices are trending and haven't yet reached
    equilibrium. Uses the world model's price history to find:
    - Consistent directional moves over multiple scanner cycles
    - Acceleration (moves getting bigger)
    - Mean reversion opportunities after overextension
    """

    name = "momentum_detector"
    interval_seconds = 180

    def __init__(self, world: WorldModel):
        super().__init__(world)
        self._trend_scores: dict[str, list[float]] = {}

    async def run_cycle(self, markets: list[Market]):
        trends_found = 0

        for market in markets:
            if not market.active:
                continue

            mid = market.condition_id
            price = self.world.get_market_price(mid)
            if price is None:
                continue

            # Track price over time
            if mid not in self._trend_scores:
                self._trend_scores[mid] = []
            self._trend_scores[mid].append(price)

            # Keep last 20 observations (~10 min at 30s scanner interval)
            self._trend_scores[mid] = self._trend_scores[mid][-20:]
            history = self._trend_scores[mid]

            if len(history) < 5:
                continue

            # Calculate trend strength
            # Linear regression slope approximation
            n = len(history)
            x_mean = (n - 1) / 2
            y_mean = mean(history)
            numerator = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                continue

            slope = numerator / denominator
            abs_slope = abs(slope)

            # Is the trend accelerating? (second derivative)
            if len(history) >= 10:
                first_half = history[:len(history)//2]
                second_half = history[len(history)//2:]
                first_move = mean(second_half) - mean(first_half) if first_half else 0
                # Acceleration is implicit in slope strength

            # Strong trend: submit a belief in the direction of the trend
            if abs_slope > 0.002:  # Meaningful trend
                trends_found += 1
                trend_direction = 1 if slope > 0 else -1
                projected = price + (slope * 5)  # Project 5 observations ahead
                projected = max(0.02, min(0.98, projected))

                # Trend-following belief
                self.world.submit_belief(Belief(
                    agent_name=self.name,
                    market_id=mid,
                    probability=projected,
                    confidence=min(0.7, abs_slope * 100),
                    reasoning=(
                        f"Momentum {'up' if slope > 0 else 'down'}: "
                        f"slope={slope:.4f}, projecting {projected:.0%}"
                    ),
                    evidence=[f"trend_slope={slope:.4f}"],
                    ttl_seconds=300,  # Short TTL — momentum is fleeting
                ))

                # If trend is very strong, also submit timing signal
                if abs_slope > 0.005:
                    self.world.submit_timing_signal(TimingSignal(
                        market_id=mid,
                        action="enter",
                        urgency=min(1.0, abs_slope * 150),
                        reasoning=f"Strong momentum: slope={slope:.4f}",
                        expires_at=time.time() + 180,
                        detected_by=self.name,
                    ))

        if trends_found > 0:
            logger.info("momentum_cycle", trends_found=trends_found)


# ══════════════════════════════════════════════════════
#  EDGE STACKER — combines micro-edges across markets
# ══════════════════════════════════════════════════════

class EdgeStackerAgent(SwarmAgent):
    """
    The portfolio-level intelligence agent.
    Finds implied edges from correlations and stacks them.
    Also detects when multiple independent signals point the same way.
    """

    name = "edge_stacker"
    interval_seconds = 120

    def __init__(self, world: WorldModel):
        super().__init__(world)

    async def run_cycle(self, markets: list[Market]):
        edges = self.world.compute_edges(min_edge_pct=2.0)
        implied_count = 0

        for edge in edges[:10]:
            correlations = self.world.get_correlated_markets(edge.market_id)
            for corr in correlations:
                other_id = (
                    corr.market_b_id if corr.market_a_id == edge.market_id
                    else corr.market_a_id
                )
                other_price = self.world.get_market_price(other_id)
                if other_price is None:
                    continue

                # Calculate implied edge
                if corr.correlation_type == "positive":
                    shift = (edge.edge_pct / 100) * corr.strength
                    implied_prob = other_price + shift
                elif corr.correlation_type == "negative":
                    shift = -(edge.edge_pct / 100) * corr.strength
                    implied_prob = other_price + shift
                else:
                    continue

                implied_prob = max(0.02, min(0.98, implied_prob))
                implied_edge = abs(implied_prob - other_price) * 100

                if implied_edge >= 2.0:
                    implied_count += 1
                    # Confidence decays through the correlation chain
                    implied_conf = edge.confidence * corr.strength * 0.6

                    self.world.submit_belief(Belief(
                        agent_name=self.name,
                        market_id=other_id,
                        probability=implied_prob,
                        confidence=implied_conf,
                        reasoning=(
                            f"Implied from '{edge.market_question[:35]}' "
                            f"via {corr.correlation_type} correlation "
                            f"(strength {corr.strength:.0%})"
                        ),
                        evidence=[f"Source edge: {edge.edge_pct:.1f}%"],
                        ttl_seconds=300,
                    ))

        if implied_count > 0:
            logger.info("edge_stacker_cycle", primary_edges=len(edges), implied=implied_count)


# ══════════════════════════════════════════════════════
#  SOCIAL SIGNALS — sentiment inference from news tone
# ══════════════════════════════════════════════════════

class SocialSignalsAgent(SwarmAgent):
    """
    Analyses the overall sentiment landscape around active markets.
    Without direct social media API access, it uses:
    - News article tone analysis
    - Volume of coverage as a proxy for attention
    - Narrative momentum (is the story growing or fading?)
    """

    name = "social_signals"
    interval_seconds = 300

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514"):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model

    async def run_cycle(self, markets: list[Market]):
        # Get recent news impacts as a proxy for social attention
        recent_news = self.world.get_recent_news(max_age_sec=3600)
        if not recent_news:
            logger.debug("social_signals_cycle", status="no_recent_news")
            return

        # Find markets with the most news attention
        market_attention: dict[str, int] = defaultdict(int)
        for news in recent_news:
            for mid in news.affected_markets:
                market_attention[mid] += 1

        if not market_attention:
            return

        # Analyse the top-attention markets
        top_markets = sorted(market_attention.items(), key=lambda x: x[1], reverse=True)[:5]

        for mid, attention_count in top_markets:
            market_news = self.world.get_news_for_market(mid)
            if not market_news:
                continue

            current_price = self.world.get_market_price(mid)
            if current_price is None:
                continue

            # High attention + directional news = potential overreaction
            impacts = [n.impact_direction.get(mid, 0) for n in market_news if mid in n.impact_direction]
            if not impacts:
                continue

            avg_impact = mean(impacts)
            attention_signal = min(1.0, attention_count / 5)

            # If lots of news all pointing one direction, market may overreact
            if len(impacts) >= 2 and abs(avg_impact) > 0.05:
                # Contrarian signal: heavy coverage often means the move is nearly priced in
                self.world.submit_belief(Belief(
                    agent_name=self.name,
                    market_id=mid,
                    probability=current_price,  # Stay near market price — the info is priced in
                    confidence=0.4 * attention_signal,
                    reasoning=(
                        f"High attention ({attention_count} articles) with "
                        f"avg impact {avg_impact:+.1%} — likely already priced in"
                    ),
                    evidence=[f"attention={attention_count}", f"avg_impact={avg_impact:.3f}"],
                    ttl_seconds=600,
                ))

        logger.debug("social_signals_cycle", markets_analysed=len(top_markets))


# ══════════════════════════════════════════════════════
#  SPORTS INTELLIGENCE — real match data for sports markets
# ══════════════════════════════════════════════════════

class SportsIntelligenceAgent(SwarmAgent):
    """
    Fetches real sports data (team form, standings, recent results)
    and feeds it into probability estimation for sports markets.
    Uses TheSportsDB (free) and football-data.org (free key).
    """

    name = "sports_intelligence"
    interval_seconds = 300  # Every 5 min

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 football_data_key: str = ""):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model

        from agent.data.sports_api import SportsDataClient
        self.sports = SportsDataClient(football_data_key=football_data_key)

    async def run_cycle(self, markets: list[Market]):
        # Find sports markets
        sports_markets = [
            m for m in markets
            if m.active and m.category in (
                "football", "soccer", "basketball", "tennis",
                "american_football", "cricket", "sport",
            )
        ]

        if not sports_markets:
            return

        analysed = 0
        for market in sports_markets[:6]:
            try:
                await self._analyse_sports_market(market)
                analysed += 1
                await asyncio.sleep(1)  # Rate limit respect
            except Exception as e:
                logger.debug("sports_intel_error", market=market.question[:30], error=str(e))

        if analysed > 0:
            logger.info("sports_intelligence_cycle", analysed=analysed)

    async def _analyse_sports_market(self, market: Market):
        """Extract team names from market question and fetch data."""
        question = market.question

        # Try to extract team names using LLM
        extract_prompt = f"""Extract the two team/competitor names from this betting market question.
If it's not a team vs team market, respond with {{}}.

Question: {question}

Respond with JSON only:
{{"home": "<team name>", "away": "<team name>", "sport": "<sport>", "competition": "<league/competition>"}}"""

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=200,
                messages=[{"role": "user", "content": extract_prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, dict) or "home" not in result:
                return

            home = result.get("home", "")
            away = result.get("away", "")
            sport = result.get("sport", "")
            competition = result.get("competition", "")

            if not home or not away:
                return

            # Fetch sports data
            intel = await self.sports.get_match_intelligence(home, away, competition)

            # Build analysis context
            context_parts = [f"Match: {home} vs {away}"]
            if intel.get("home_form"):
                context_parts.append(f"{home} form (last 5): {intel['home_form']}")
            if intel.get("away_form"):
                context_parts.append(f"{away} form (last 5): {intel['away_form']}")
            if intel.get("home_position"):
                context_parts.append(f"{home} league position: {intel['home_position']}")
            if intel.get("away_position"):
                context_parts.append(f"{away} league position: {intel['away_position']}")
            if intel.get("home_recent"):
                recent = ", ".join(
                    f"{r['score']} vs {r['vs']}" for r in intel["home_recent"][:3]
                )
                context_parts.append(f"{home} recent: {recent}")
            if intel.get("away_recent"):
                recent = ", ".join(
                    f"{r['score']} vs {r['vs']}" for r in intel["away_recent"][:3]
                )
                context_parts.append(f"{away} recent: {recent}")

            sports_context = "\n".join(context_parts)
            price = self.world.get_market_price(market.condition_id)
            if price is None:
                return

            # Use LLM to estimate probability with real data
            analysis_prompt = f"""You are an expert sports analyst and calibrated forecaster.

MARKET: {question}
CURRENT PRICE: {price:.2f} ({price:.0%} implied probability)

REAL MATCH DATA:
{sports_context}

Based on the real form data, league positions, and recent results above,
estimate the probability of the favoured outcome (YES).

Consider:
- Home advantage (typically 5-10% boost in football)
- Recent form trajectory (improving vs declining)
- League position gap
- Historical patterns for this type of matchup

Respond with JSON only:
{{"probability": <float 0.05-0.95>, "confidence": <float 0.3-0.85>, "reasoning": "<key factors>"}}"""

            response = await self.client.messages.create(
                model=self.model, max_tokens=400,
                messages=[{"role": "user", "content": analysis_prompt}],
            )
            result = safe_parse_json(response.content[0].text)
            if not isinstance(result, dict) or "probability" not in result:
                return

            prob = float(result["probability"])
            conf = float(result.get("confidence", 0.5))

            self.world.submit_belief(Belief(
                agent_name=self.name,
                market_id=market.condition_id,
                probability=prob,
                confidence=conf,
                reasoning=f"Sports data: {result.get('reasoning', '')}",
                evidence=[sports_context],
                ttl_seconds=900,  # 15 min — sports data stays relevant
                domain=market.category,
            ))

            logger.info(
                "sports_analysis",
                match=f"{home} vs {away}",
                estimate=round(prob, 3),
                market_price=price,
                edge=round((prob - price) * 100, 1),
            )

        except Exception as e:
            logger.debug("sports_analysis_error", error=str(e))


# ══════════════════════════════════════════════════════
#  ODDS ARBITRAGE — cross-bookmaker edge detection
# ══════════════════════════════════════════════════════

class OddsArbitrageAgent(SwarmAgent):
    """
    Compares Smarkets odds against other bookmakers to find:
    - Value bets (Smarkets price is better than market consensus)
    - Pure arbitrage (different bookmakers disagree enough for guaranteed profit)
    - Market consensus (what the "true" probability is across all bookmakers)

    Uses The Odds API (free tier: 500 requests/month).
    """

    name = "odds_arbitrage"
    interval_seconds = 600  # Every 10 min (preserve API quota)

    def __init__(self, world: WorldModel, odds_api_key: str = ""):
        super().__init__(world)
        self.odds_api_key = odds_api_key

        from agent.data.odds_api import OddsComparisonClient, SPORT_KEYS
        self.odds = OddsComparisonClient(api_key=odds_api_key)
        self.sport_keys = SPORT_KEYS

    async def run_cycle(self, markets: list[Market]):
        if not self.odds_api_key:
            logger.debug("odds_arbitrage_skipped", reason="no API key")
            return

        # Scan major sports for value
        sports_to_scan = ["soccer_epl", "basketball_nba"]
        total_value = 0
        total_arbs = 0

        for sport_key in sports_to_scan:
            try:
                # Find value bets
                value_bets = await self.odds.find_value_bets(sport_key, min_edge_pct=3.0)
                for vb in value_bets[:5]:
                    total_value += 1
                    # Try to match to a Smarkets market
                    matched = self._match_to_market(vb, markets)
                    if matched:
                        self.world.submit_belief(Belief(
                            agent_name=self.name,
                            market_id=matched.condition_id,
                            probability=vb["market_prob"],
                            confidence=min(0.85, 0.5 + vb["edge_pct"] / 20),
                            reasoning=(
                                f"Odds consensus: {vb['num_bookmakers']} bookmakers "
                                f"avg {vb['market_prob']:.0%} vs Smarkets "
                                f"({vb['edge_pct']:.1f}% edge)"
                            ),
                            evidence=[
                                f"best_price={vb['best_price']}",
                                f"worst_price={vb['worst_price']}",
                                f"avg_price={vb['avg_price']}",
                            ],
                            ttl_seconds=600,
                        ))

                # Find arbitrage
                arbs = await self.odds.find_arbitrage(sport_key)
                total_arbs += len(arbs)
                for arb in arbs:
                    logger.info(
                        "arbitrage_found",
                        event=arb["event"],
                        profit=f"{arb['profit_pct']:.2f}%",
                        legs=arb["legs"],
                    )

                # Get market consensus and update world model
                consensus = await self.odds.get_market_consensus(sport_key)
                for c in consensus:
                    matched = self._match_to_market_by_teams(
                        c.get("home_team", ""), c.get("away_team", ""), markets
                    )
                    if matched:
                        # Use consensus probability as a belief
                        home_prob = c["fair_probabilities"].get(c.get("home_team", ""), 0)
                        if home_prob > 0:
                            self.world.submit_belief(Belief(
                                agent_name=self.name,
                                market_id=matched.condition_id,
                                probability=home_prob,
                                confidence=min(0.8, 0.4 + c["num_bookmakers"] * 0.03),
                                reasoning=(
                                    f"Market consensus from {c['num_bookmakers']} bookmakers "
                                    f"(overround: {c['overround']:.1f}%)"
                                ),
                                evidence=[str(c["fair_probabilities"])],
                                ttl_seconds=600,
                            ))

            except Exception as e:
                logger.debug("odds_scan_error", sport=sport_key, error=str(e))

        remaining = self.odds.requests_remaining
        logger.info(
            "odds_arbitrage_cycle",
            value_bets=total_value,
            arbitrage=total_arbs,
            api_remaining=remaining,
        )

    def _match_to_market(self, value_bet: dict, markets: list[Market]) -> Market | None:
        event = value_bet.get("event", "").lower()
        for m in markets:
            q = m.question.lower()
            if any(word in q for word in event.split() if len(word) > 3):
                return m
        return None

    def _match_to_market_by_teams(self, home: str, away: str,
                                   markets: list[Market]) -> Market | None:
        home_l = home.lower()
        away_l = away.lower()
        for m in markets:
            q = m.question.lower()
            if home_l[:6] in q and away_l[:6] in q:
                return m
        return None


# ══════════════════════════════════════════════════════
#  WEB RESEARCH — searches for real-time market info
# ══════════════════════════════════════════════════════

class WebResearchAgent(SwarmAgent):
    """
    Researches specific markets by asking Claude to reason about them
    with its built-in knowledge. In future, this could use web search
    APIs to find real-time information.

    For now, it provides deep analytical research on high-priority markets
    that other agents have flagged as interesting.
    """

    name = "web_researcher"
    interval_seconds = 600  # Every 10 min

    def __init__(self, world: WorldModel, anthropic_key: str,
                 model: str = "claude-sonnet-4-20250514"):
        super().__init__(world)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key, timeout=30)
        self.model = model
        self._researched: set[str] = set()

    async def run_cycle(self, markets: list[Market]):
        # Research markets that have edges or conflicts
        edges = self.world.compute_edges(min_edge_pct=2.0)
        conflicts = self.world.get_conflicts()

        priority_ids = set()
        for edge in edges[:5]:
            priority_ids.add(edge.market_id)
        for conflict in conflicts[:3]:
            priority_ids.add(conflict.get("market_id", ""))

        # Don't re-research recently analysed markets
        to_research = [
            mid for mid in priority_ids
            if mid and mid not in self._researched
        ]

        if not to_research:
            # Research highest-volume markets we haven't looked at
            unresearched = [
                m for m in markets
                if m.active and m.condition_id not in self._researched
            ]
            unresearched.sort(key=lambda m: m.volume, reverse=True)
            to_research = [m.condition_id for m in unresearched[:3]]

        researched_count = 0
        for mid in to_research[:3]:
            question = self.world._market_questions.get(mid, "")
            price = self.world.get_market_price(mid)
            if not question or price is None:
                continue

            try:
                await self._deep_research(mid, question, price)
                self._researched.add(mid)
                researched_count += 1
            except Exception as e:
                logger.debug("research_error", market=mid[:8], error=str(e))

        # Clear old researched set periodically
        if len(self._researched) > 100:
            self._researched = set(list(self._researched)[-50:])

        if researched_count > 0:
            logger.info("web_research_cycle", researched=researched_count)

    async def _deep_research(self, market_id: str, question: str, price: float):
        """Conduct deep analysis on a single market."""

        # Gather all existing intelligence
        news = self.world.get_news_for_market(market_id)
        correlations = self.world.get_correlated_markets(market_id)
        existing_beliefs = self.world.get_belief_summary(market_id)
        regime = existing_beliefs.get("regime", "unknown")

        context_parts = []
        if news:
            context_parts.append("NEWS:\n" + "\n".join(
                f"- {n.headline} (impact: {n.impact_direction.get(market_id, 0):+.2f})"
                for n in news[:5]
            ))
        if correlations:
            context_parts.append("CORRELATIONS:\n" + "\n".join(
                f"- {c.description} ({c.correlation_type}, {c.strength:.0%})"
                for c in correlations[:3]
            ))
        if existing_beliefs.get("beliefs"):
            context_parts.append("EXISTING AGENT BELIEFS:\n" + "\n".join(
                f"- {b['agent']}: {b['probability']:.0%} (conf: {b['confidence']:.0%}) — {b['reasoning'][:60]}"
                for b in existing_beliefs["beliefs"]
            ))

        context = "\n\n".join(context_parts) if context_parts else "No prior intelligence."

        prompt = f"""You are a senior research analyst conducting deep analysis on a prediction market.

MARKET QUESTION: {question}
CURRENT PRICE: {price:.2f} ({price:.0%} implied probability)
MARKET REGIME: {regime}

EXISTING INTELLIGENCE FROM OTHER ANALYSTS:
{context}

DEEP RESEARCH TASK:
1. What are the key factors that will determine this outcome?
2. What information might the market be missing or underweighting?
3. Are there any upcoming catalysts or deadlines that matter?
4. What is your probability estimate after thorough analysis?
5. How does your estimate compare to the current market price?

Think through this carefully and systematically.

End with JSON on its own line:
{{"probability": <float 0.02-0.98>, "confidence": <float 0.3-0.9>, "key_insight": "<the single most important factor>", "reasoning": "<2-3 sentences>"}}"""

        response = await self.client.messages.create(
            model=self.model, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        result = safe_parse_json(response.content[0].text)
        if not isinstance(result, dict) or "probability" not in result:
            # Try extracting from end of response
            lines = response.content[0].text.strip().split("\n")
            for line in reversed(lines):
                result = safe_parse_json(line.strip())
                if isinstance(result, dict) and "probability" in result:
                    break
            else:
                return

        prob = float(result["probability"])
        conf = float(result.get("confidence", 0.5))

        self.world.submit_belief(Belief(
            agent_name=self.name,
            market_id=market_id,
            probability=prob,
            confidence=conf * 1.1,  # Slight boost for deep research
            reasoning=f"Deep research: {result.get('key_insight', result.get('reasoning', ''))}",
            evidence=[result.get("reasoning", ""), result.get("key_insight", "")],
            ttl_seconds=1800,  # Research beliefs last 30 min
        ))

        logger.info(
            "deep_research",
            market=question[:45],
            estimate=round(prob, 3),
            market_price=price,
            edge=round((prob - price) * 100, 1),
            insight=result.get("key_insight", "")[:60],
        )