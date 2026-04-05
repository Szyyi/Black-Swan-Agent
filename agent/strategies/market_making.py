"""Market making: provide liquidity with dynamic spread management."""

from __future__ import annotations

import time

import structlog

from agent.config import MarketMakingConfig
from agent.data.feeds import PolymarketClient
from agent.models import Market, MarketOutcome, Signal, Side, StrategyType
from agent.strategies.base import BaseStrategy

logger = structlog.get_logger()


class MarketMakingStrategy(BaseStrategy):
    """
    Places two-sided quotes (bid and ask) around the mid price to
    capture the spread. Manages inventory risk by adjusting quotes
    when position becomes skewed.

    Key mechanics:
    - Quote around mid price with configurable spread
    - Widen spread during high volatility
    - Skew quotes to reduce inventory when one side accumulates
    - Only operate on markets with sufficient existing liquidity
    """

    strategy_type = StrategyType.MARKET_MAKING

    def __init__(self, config: MarketMakingConfig, poly_client: PolymarketClient):
        super().__init__(enabled=config.enabled, weight=config.weight)
        self.config = config
        self.poly = poly_client

        # Inventory tracking per market
        self._inventory: dict[str, float] = {}  # market_id -> net position (+ = long YES)
        self._last_quotes: dict[str, float] = {}  # market_id -> last quote time
        self._price_history: dict[str, list[tuple[float, float]]] = {}

    async def initialize(self):
        logger.info("market_making_initialized", target_spread=self.config.target_spread_bps)

    async def evaluate(self, markets: list[Market]) -> list[Signal]:
        if not self._active:
            return []

        signals: list[Signal] = []

        # Select eligible markets
        eligible = [
            m for m in markets
            if m.active
            and m.liquidity >= self.config.min_liquidity
            and len(m.token_ids) >= 2
        ]

        for market in eligible:
            # Check quote refresh interval
            last_quote = self._last_quotes.get(market.condition_id, 0)
            if time.time() - last_quote < self.config.quote_refresh_sec:
                continue

            try:
                new_signals = await self._generate_quotes(market)
                signals.extend(new_signals)
                self._last_quotes[market.condition_id] = time.time()
            except Exception as e:
                logger.error("mm_quote_error", market=market.slug, error=str(e))

        return signals

    async def _generate_quotes(self, market: Market) -> list[Signal]:
        """Generate bid and ask quotes for a market."""
        yes_token = market.token_ids[0]
        book = await self.poly.get_order_book(yes_token)

        if book.mid_price is None:
            return []

        mid = book.mid_price

        # Track price for volatility calculation
        self._update_price_history(market.condition_id, mid)

        # Calculate dynamic spread based on volatility
        volatility = self._estimate_volatility(market.condition_id)
        spread_multiplier = 1.0 + (volatility * 5)  # Widen spread in volatile markets
        half_spread = (self.config.target_spread_bps / 10_000) * spread_multiplier / 2

        # Inventory skew: shift quotes to reduce position
        inventory = self._inventory.get(market.condition_id, 0.0)
        max_inv = self.config.max_inventory_usd
        skew = 0.0

        if max_inv > 0:
            inventory_ratio = inventory / max_inv
            if abs(inventory_ratio) > self.config.rebalance_threshold:
                # Skew to shed inventory: if long, lower ask to sell; if short, raise bid to buy
                skew = -inventory_ratio * half_spread * 0.5

        bid_price = max(0.01, mid - half_spread + skew)
        ask_price = min(0.99, mid + half_spread + skew)

        # Size based on remaining capacity
        remaining_capacity = max(0, self.config.max_inventory_usd - abs(inventory))
        quote_size = min(remaining_capacity * 0.5, self.config.max_inventory_usd * 0.2)

        if quote_size < 5:  # Minimum viable quote
            return []

        signals = []

        # Bid (buy YES)
        signals.append(Signal(
            strategy=self.strategy_type,
            market_id=market.condition_id,
            token_id=yes_token,
            side=Side.BUY,
            outcome=MarketOutcome.YES,
            confidence=0.5,  # Neutral - we're market making, not predicting
            edge_pct=half_spread * 100,
            fair_value=mid,
            market_price=bid_price,
            suggested_size_usd=quote_size,
            reasoning=f"MM bid: {bid_price:.3f} (mid={mid:.3f}, spread={half_spread*2*10000:.0f}bps)",
            metadata={
                "quote_type": "bid",
                "volatility": volatility,
                "inventory": inventory,
                "skew": skew,
            },
        ))

        # Ask (sell YES / buy NO)
        if len(market.token_ids) >= 2:
            signals.append(Signal(
                strategy=self.strategy_type,
                market_id=market.condition_id,
                token_id=market.token_ids[1],  # NO token
                side=Side.BUY,
                outcome=MarketOutcome.NO,
                confidence=0.5,
                edge_pct=half_spread * 100,
                fair_value=1 - mid,
                market_price=1 - ask_price,
                suggested_size_usd=quote_size,
                reasoning=f"MM ask: {ask_price:.3f} (mid={mid:.3f})",
                metadata={
                    "quote_type": "ask",
                    "volatility": volatility,
                    "inventory": inventory,
                },
            ))

        return signals

    def _update_price_history(self, market_id: str, price: float):
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        self._price_history[market_id].append((time.time(), price))
        # Keep last 30 minutes
        cutoff = time.time() - 1800
        self._price_history[market_id] = [
            (ts, p) for ts, p in self._price_history[market_id] if ts > cutoff
        ]

    def _estimate_volatility(self, market_id: str) -> float:
        """Estimate recent price volatility (standard deviation of returns)."""
        history = self._price_history.get(market_id, [])
        if len(history) < 5:
            return 0.01  # Default low volatility

        prices = [p for _, p in history]
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if not returns:
            return 0.01

        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def update_inventory(self, market_id: str, delta: float):
        """Update inventory after a fill. Positive = bought YES, negative = sold."""
        self._inventory[market_id] = self._inventory.get(market_id, 0) + delta

    async def shutdown(self):
        pass
