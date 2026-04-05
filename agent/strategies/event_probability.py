"""Event probability: ensemble LLM probability estimation to find mispriced markets."""

from __future__ import annotations

import asyncio
import json
import time
from statistics import mean, stdev

import anthropic
import structlog

from agent.config import EventProbabilityConfig
from agent.data.feeds import PolymarketClient
from agent.models import Market, MarketOutcome, Signal, Side, StrategyType
from agent.strategies.base import BaseStrategy

logger = structlog.get_logger()

PROBABILITY_PROMPT_TEMPLATE = """You are an expert superforecaster trained in calibrated probability
estimation. You must estimate the probability of this event occurring.

QUESTION: {question}

CONTEXT:
- Current market price: ${market_price:.2f} (implying {implied_prob:.0%} probability)
- Market volume: ${volume:,.0f}
- Market end date: {end_date}
- Category: {category}

APPROACH (Perspective #{perspective_num}):
{perspective_instruction}

Think step by step:
1. What is the base rate for this type of event?
2. What specific evidence shifts the probability up or down?
3. Are there any known biases to correct for?
4. How should you weight the current market price as information?

Respond ONLY with a JSON object (no markdown):
{{
  "probability": <float 0.01-0.99>,
  "confidence": <float 0.0-1.0>,
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "reasoning": "<2-3 sentences>"
}}"""

PERSPECTIVES = [
    "Think like a base-rate analyst. Focus on historical frequencies and reference classes. What similar events have happened before, and what fraction resolved YES?",
    "Think like a news analyst. Focus on the latest available information, recent trends, and momentum. What is the current trajectory suggesting?",
    "Think like a contrarian. Consider what the market might be getting wrong. What do most people overlook? Where might the crowd be systematically biased?",
]


class EventProbabilityStrategy(BaseStrategy):
    """
    Uses multiple LLM 'perspectives' to generate independent probability
    estimates, then combines them via ensemble averaging.

    The ensemble approach reduces individual model biases and produces
    more calibrated estimates than a single prompt.
    """

    strategy_type = StrategyType.EVENT_PROBABILITY

    def __init__(
        self,
        config: EventProbabilityConfig,
        poly_client: PolymarketClient,
        anthropic_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        super().__init__(enabled=config.enabled, weight=config.weight)
        self.config = config
        self.poly = poly_client
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        self._last_check: float = 0
        self._estimates: dict[str, dict] = {}  # market_id -> last estimate

    async def initialize(self):
        logger.info(
            "event_probability_initialized",
            ensemble_size=self.config.ensemble_models,
        )

    async def evaluate(self, markets: list[Market]) -> list[Signal]:
        if not self._active:
            return []

        if time.time() - self._last_check < self.config.update_interval_sec:
            return []
        self._last_check = time.time()

        signals: list[Signal] = []

        # Select markets worth analyzing: active, decent volume, not crypto-short-term
        candidates = [
            m for m in markets
            if m.active
            and m.volume > 5000
            and m.liquidity > 2000
            and not self._is_short_term_crypto(m)
        ]
        candidates.sort(key=lambda m: m.volume, reverse=True)
        batch = candidates[:5]  # Limit API calls

        for market in batch:
            try:
                signal = await self._ensemble_estimate(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("ensemble_error", market=market.slug, error=str(e))

        return signals

    async def _ensemble_estimate(self, market: Market) -> Signal | None:
        """Run ensemble of LLM perspectives and combine results."""
        if len(market.token_ids) < 1:
            return None

        yes_token = market.token_ids[0]
        yes_price = await self.poly.get_price(yes_token)
        if yes_price is None:
            return None

        # Run perspectives concurrently
        num_perspectives = min(self.config.ensemble_models, len(PERSPECTIVES))
        tasks = [
            self._get_perspective(market, yes_price, i)
            for i in range(num_perspectives)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid estimates
        estimates = []
        confidences = []
        all_factors = []

        for r in results:
            if isinstance(r, dict) and "probability" in r:
                estimates.append(r["probability"])
                confidences.append(r.get("confidence", 0.5))
                all_factors.extend(r.get("key_factors", []))

        if len(estimates) < 2:
            return None

        # Ensemble: confidence-weighted average
        total_weight = sum(confidences)
        if total_weight == 0:
            ensemble_prob = mean(estimates)
        else:
            ensemble_prob = sum(e * c for e, c in zip(estimates, confidences)) / total_weight

        ensemble_confidence = mean(confidences)

        # Measure agreement (lower stdev = higher agreement = more confident)
        if len(estimates) >= 2:
            estimate_stdev = stdev(estimates)
            agreement_bonus = max(0, 0.2 - estimate_stdev)  # Bonus for tight agreement
            ensemble_confidence = min(1.0, ensemble_confidence + agreement_bonus)

        # Calculate edge
        edge_yes = ensemble_prob - yes_price
        edge_no = (1 - ensemble_prob) - (1 - yes_price)

        logger.info(
            "ensemble_result",
            market=market.question[:60],
            estimates=estimates,
            ensemble=round(ensemble_prob, 3),
            market_price=yes_price,
            edge_yes=round(edge_yes * 100, 1),
        )

        # Store estimate
        self._estimates[market.condition_id] = {
            "ensemble_prob": ensemble_prob,
            "estimates": estimates,
            "market_price": yes_price,
            "timestamp": time.time(),
        }

        # Generate signal if edge is sufficient
        if edge_yes * 100 >= self.config.min_edge_pct:
            return Signal(
                strategy=self.strategy_type,
                market_id=market.condition_id,
                token_id=market.token_ids[0],
                side=Side.BUY,
                outcome=MarketOutcome.YES,
                confidence=ensemble_confidence,
                edge_pct=edge_yes * 100,
                fair_value=ensemble_prob,
                market_price=yes_price,
                suggested_size_usd=self.config.max_position_usd,
                reasoning=f"Ensemble ({len(estimates)} models): {ensemble_prob:.1%} vs market {yes_price:.1%}",
                metadata={
                    "estimates": estimates,
                    "ensemble_prob": ensemble_prob,
                    "key_factors": list(set(all_factors))[:5],
                },
            )

        if len(market.token_ids) >= 2 and edge_no * 100 >= self.config.min_edge_pct:
            return Signal(
                strategy=self.strategy_type,
                market_id=market.condition_id,
                token_id=market.token_ids[1],
                side=Side.BUY,
                outcome=MarketOutcome.NO,
                confidence=ensemble_confidence,
                edge_pct=edge_no * 100,
                fair_value=1 - ensemble_prob,
                market_price=1 - yes_price,
                suggested_size_usd=self.config.max_position_usd,
                reasoning=f"Ensemble ({len(estimates)} models): YES overpriced at {yes_price:.1%} vs fair {ensemble_prob:.1%}",
                metadata={"estimates": estimates, "ensemble_prob": ensemble_prob},
            )

        return None

    async def _get_perspective(
        self, market: Market, market_price: float, perspective_idx: int
    ) -> dict:
        """Get a single perspective's probability estimate."""
        perspective = PERSPECTIVES[perspective_idx % len(PERSPECTIVES)]

        prompt = PROBABILITY_PROMPT_TEMPLATE.format(
            question=market.question,
            market_price=market_price,
            implied_prob=market_price,
            volume=market.volume,
            end_date=market.end_date or "Unknown",
            category=market.category or "General",
            perspective_num=perspective_idx + 1,
            perspective_instruction=perspective,
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.7 + (perspective_idx * 0.1),  # Vary temperature for diversity
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except Exception as e:
            logger.warning("perspective_error", idx=perspective_idx, error=str(e))
            return {}

    def _is_short_term_crypto(self, market: Market) -> bool:
        q = market.question.lower()
        crypto = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
        short = ["5 min", "15 min", "1 hour", "higher in", "lower in"]
        return any(k in q for k in crypto) and any(k in q for k in short)

    async def shutdown(self):
        pass
