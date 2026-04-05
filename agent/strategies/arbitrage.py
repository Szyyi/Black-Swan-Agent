"""Latency arbitrage: exploit price lag between CEX spot and Polymarket contracts."""

from __future__ import annotations

import time

import structlog

from agent.config import ArbitrageConfig
from agent.data.feeds import ExternalPriceFeed, PolymarketClient
from agent.models import Market, MarketOutcome, Signal, Side, StrategyType
from agent.strategies.base import BaseStrategy

logger = structlog.get_logger()

# Maps Polymarket slug patterns to CEX symbols
CRYPTO_MARKET_MAP = {
    "btc": {"binance": "BTCUSDT", "coinbase": "BTC-USD"},
    "eth": {"binance": "ETHUSDT", "coinbase": "ETH-USD"},
    "sol": {"binance": "SOLUSDT", "coinbase": "SOL-USD"},
}


class ArbitrageStrategy(BaseStrategy):
    """
    Detects when Polymarket crypto up/down contract prices lag behind
    confirmed CEX spot momentum. Trades into the direction confirmed
    by spot price movement before the prediction market adjusts.

    Key edge: Polymarket's CLOB reprices slower than centralized exchanges,
    especially on 5-min and 15-min contracts.
    """

    strategy_type = StrategyType.ARBITRAGE

    def __init__(self, config: ArbitrageConfig, poly_client: PolymarketClient):
        super().__init__(enabled=config.enabled, weight=config.weight)
        self.config = config
        self.poly = poly_client
        self.price_feed = ExternalPriceFeed()

        # Track recent spot prices to determine momentum
        self._spot_history: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(ts, price)]
        self._last_poly_prices: dict[str, float] = {}  # token_id -> price

    async def initialize(self):
        logger.info("arbitrage_strategy_initialized", exchanges=self.config.exchanges)

    async def evaluate(self, markets: list[Market]) -> list[Signal]:
        """
        For each tracked crypto market:
        1. Get latest CEX spot price
        2. Compute spot momentum (direction + magnitude)
        3. Get Polymarket YES/NO prices
        4. If spot clearly moved in one direction but Polymarket hasn't repriced,
           buy the outcome that matches the spot direction
        """
        if not self._active:
            return []

        signals: list[Signal] = []

        # Filter to crypto up/down markets
        crypto_markets = [m for m in markets if self._is_crypto_market(m)]

        for market in crypto_markets:
            try:
                signal = await self._check_arbitrage(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("arb_eval_error", market=market.slug, error=str(e))

        return signals

    async def _check_arbitrage(self, market: Market) -> Signal | None:
        """Check a single market for arbitrage opportunity."""
        # Determine which crypto this market tracks
        asset = self._extract_asset(market)
        if not asset or asset not in CRYPTO_MARKET_MAP:
            return None

        # Get spot prices from multiple exchanges
        spot_prices: list[float] = []
        symbols = CRYPTO_MARKET_MAP[asset]

        if "binance" in self.config.exchanges and "binance" in symbols:
            tick = await self.price_feed.get_binance_price(symbols["binance"])
            if tick:
                spot_prices.append(tick.price)
                self._update_history(asset, tick.price)

        if "coinbase" in self.config.exchanges and "coinbase" in symbols:
            tick = await self.price_feed.get_coinbase_price(symbols["coinbase"])
            if tick:
                spot_prices.append(tick.price)

        if not spot_prices:
            return None

        # Compute consensus spot direction
        current_spot = sum(spot_prices) / len(spot_prices)
        momentum = self._compute_momentum(asset, current_spot)

        if abs(momentum) < 0.001:  # Less than 0.1% move - not significant
            return None

        # Get Polymarket prices
        if len(market.token_ids) < 2:
            return None

        yes_token, no_token = market.token_ids[0], market.token_ids[1]
        yes_book = await self.poly.get_order_book(yes_token)
        no_book = await self.poly.get_order_book(no_token)

        yes_price = yes_book.best_ask or 0
        no_price = no_book.best_ask or 0

        if yes_price == 0 or no_price == 0:
            return None

        # Determine edge
        # If spot is going UP but YES (price will be higher) is still cheap
        if momentum > 0:
            fair_value = min(0.95, 0.5 + (momentum * 10))  # Scale momentum to probability
            edge = fair_value - yes_price
            if edge * 10_000 > self.config.min_spread_bps:
                return Signal(
                    strategy=self.strategy_type,
                    market_id=market.condition_id,
                    token_id=yes_token,
                    side=Side.BUY,
                    outcome=MarketOutcome.YES,
                    confidence=min(0.95, abs(momentum) * 20),
                    edge_pct=edge * 100,
                    fair_value=fair_value,
                    market_price=yes_price,
                    suggested_size_usd=self.config.max_position_usd,
                    reasoning=f"Spot {asset.upper()} momentum +{momentum:.4f}, YES underpriced",
                    metadata={"asset": asset, "spot": current_spot, "momentum": momentum},
                )
        else:
            fair_value = min(0.95, 0.5 + (abs(momentum) * 10))
            edge = fair_value - no_price
            if edge * 10_000 > self.config.min_spread_bps:
                return Signal(
                    strategy=self.strategy_type,
                    market_id=market.condition_id,
                    token_id=no_token,
                    side=Side.BUY,
                    outcome=MarketOutcome.NO,
                    confidence=min(0.95, abs(momentum) * 20),
                    edge_pct=edge * 100,
                    fair_value=fair_value,
                    market_price=no_price,
                    suggested_size_usd=self.config.max_position_usd,
                    reasoning=f"Spot {asset.upper()} momentum {momentum:.4f}, NO underpriced",
                    metadata={"asset": asset, "spot": current_spot, "momentum": momentum},
                )

        return None

    def _compute_momentum(self, asset: str, current: float) -> float:
        """Compute price momentum as fractional change over recent window."""
        history = self._spot_history.get(asset, [])
        if len(history) < 2:
            return 0.0

        # Use price from ~30 seconds ago as baseline
        cutoff = time.time() - 30
        baseline_prices = [p for ts, p in history if ts < cutoff]
        if not baseline_prices:
            return 0.0

        baseline = baseline_prices[-1]
        return (current - baseline) / baseline

    def _update_history(self, asset: str, price: float):
        """Maintain rolling price history."""
        if asset not in self._spot_history:
            self._spot_history[asset] = []

        self._spot_history[asset].append((time.time(), price))

        # Keep last 5 minutes
        cutoff = time.time() - 300
        self._spot_history[asset] = [
            (ts, p) for ts, p in self._spot_history[asset] if ts > cutoff
        ]

    def _is_crypto_market(self, market: Market) -> bool:
        """Check if market is a crypto up/down prediction."""
        q = market.question.lower()
        keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
        time_keywords = ["5 min", "15 min", "1 hour", "higher", "lower", "up", "down"]
        return any(k in q for k in keywords) and any(k in q for k in time_keywords)

    def _extract_asset(self, market: Market) -> str | None:
        """Extract the crypto asset from market question."""
        q = market.question.lower()
        if "bitcoin" in q or "btc" in q:
            return "btc"
        if "ethereum" in q or "eth" in q:
            return "eth"
        if "solana" in q or "sol" in q:
            return "sol"
        return None

    async def shutdown(self):
        await self.price_feed.close()
