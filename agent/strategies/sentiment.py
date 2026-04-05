"""Sentiment strategy: LLM-powered news analysis to find mispriced event markets."""

from __future__ import annotations

import time

import anthropic
import structlog

from agent.config import SentimentConfig
from agent.data.feeds import NewsIngester, PolymarketClient
from agent.models import Market, MarketOutcome, Signal, Side, StrategyType
from agent.strategies.base import BaseStrategy

logger = structlog.get_logger()

ANALYSIS_PROMPT = """You are a prediction market analyst. Given the following news headlines
and a prediction market question, estimate the probability the outcome will be YES.

MARKET QUESTION: {question}
CURRENT MARKET PRICE (YES): {market_price} (implying {implied_prob:.0%} probability)

RECENT NEWS:
{news_block}

Instructions:
1. Analyze how the news affects this specific market question
2. Consider base rates and prior probabilities
3. Be calibrated - avoid overconfidence
4. Factor in the current market price as a Bayesian prior

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief 1-2 sentence explanation>",
  "relevant_news": "<which headline(s) matter most>"
}}"""


class SentimentStrategy(BaseStrategy):
    """
    Uses Claude to analyze breaking news and assess whether prediction
    markets have mispriced events. Targets event markets (politics,
    economics, tech milestones) rather than crypto price markets.
    """

    strategy_type = StrategyType.SENTIMENT

    def __init__(
        self,
        config: SentimentConfig,
        poly_client: PolymarketClient,
        anthropic_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        super().__init__(enabled=config.enabled, weight=config.weight)
        self.config = config
        self.poly = poly_client
        self.model = model
        self.news = NewsIngester(sources=config.news_sources)
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        self._last_check: float = 0
        self._analyzed_markets: dict[str, float] = {}  # market_id -> last_analyzed_ts

    async def initialize(self):
        logger.info("sentiment_strategy_initialized", sources=len(self.config.news_sources))

    async def evaluate(self, markets: list[Market]) -> list[Signal]:
        if not self._active:
            return []

        # Respect check interval
        if time.time() - self._last_check < self.config.recheck_interval_sec:
            return []
        self._last_check = time.time()

        # Fetch latest news
        articles = await self.news.get_latest_news()
        if not articles:
            return []

        # Build news block for prompt
        news_block = "\n".join(
            f"- [{a['published']}] {a['title']}: {a['summary'][:200]}"
            for a in articles[:15]
        )

        signals: list[Signal] = []

        # Filter to event markets (not crypto price markets)
        event_markets = [
            m for m in markets
            if m.active
            and m.liquidity >= 1000
            and not self._is_crypto_market(m)
            and self._should_analyze(m)
        ]

        # Analyze top markets by volume (limit API calls)
        event_markets.sort(key=lambda m: m.volume, reverse=True)
        batch = event_markets[:10]

        for market in batch:
            try:
                signal = await self._analyze_market(market, news_block)
                if signal:
                    signals.append(signal)
                    self._analyzed_markets[market.condition_id] = time.time()
            except Exception as e:
                logger.error("sentiment_analysis_error", market=market.slug, error=str(e))

        return signals

    async def _analyze_market(self, market: Market, news_block: str) -> Signal | None:
        """Use LLM to analyze a market against current news."""
        if len(market.token_ids) < 1:
            return None

        yes_token = market.token_ids[0]
        yes_price = await self.poly.get_price(yes_token)
        if yes_price is None:
            return None

        prompt = ANALYSIS_PROMPT.format(
            question=market.question,
            market_price=f"${yes_price:.2f}",
            implied_prob=yes_price,
            news_block=news_block,
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            text = response.content[0].text.strip()
            # Clean potential markdown fences
            text = text.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(text)

            probability = float(analysis["probability"])
            confidence = float(analysis["confidence"])
            reasoning = analysis.get("reasoning", "")

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("llm_parse_error", error=str(e), market=market.slug)
            return None
        except anthropic.APIError as e:
            logger.error("anthropic_api_error", error=str(e))
            return None

        # Check if confidence meets threshold
        if confidence < self.config.confidence_threshold:
            return None

        # Calculate edge
        edge_yes = probability - yes_price
        edge_no = (1 - probability) - (1 - yes_price)

        # Trade YES if underpriced
        if edge_yes * 100 >= self.config.min_edge_pct:
            return Signal(
                strategy=self.strategy_type,
                market_id=market.condition_id,
                token_id=market.token_ids[0],
                side=Side.BUY,
                outcome=MarketOutcome.YES,
                confidence=confidence,
                edge_pct=edge_yes * 100,
                fair_value=probability,
                market_price=yes_price,
                suggested_size_usd=self.config.max_position_usd,
                reasoning=reasoning,
                metadata={"llm_probability": probability},
            )

        # Trade NO if YES is overpriced
        if len(market.token_ids) >= 2 and edge_no * 100 >= self.config.min_edge_pct:
            no_price = 1 - yes_price  # Approximate
            return Signal(
                strategy=self.strategy_type,
                market_id=market.condition_id,
                token_id=market.token_ids[1],
                side=Side.BUY,
                outcome=MarketOutcome.NO,
                confidence=confidence,
                edge_pct=edge_no * 100,
                fair_value=1 - probability,
                market_price=no_price,
                suggested_size_usd=self.config.max_position_usd,
                reasoning=reasoning,
                metadata={"llm_probability": probability},
            )

        return None

    def _is_crypto_market(self, market: Market) -> bool:
        q = market.question.lower()
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto"]
        time_keywords = ["5 min", "15 min", "1 hour", "higher", "lower"]
        return any(k in q for k in crypto_keywords) and any(k in q for k in time_keywords)

    def _should_analyze(self, market: Market) -> bool:
        """Check if enough time has passed since last analysis."""
        last = self._analyzed_markets.get(market.condition_id, 0)
        return time.time() - last > self.config.recheck_interval_sec

    async def shutdown(self):
        await self.news.close()
