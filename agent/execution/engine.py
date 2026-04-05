"""Execution engine: order routing, paper trading, and live CLOB interaction."""

from __future__ import annotations

import time

import aiohttp
import structlog

from agent.config import AgentConfig
from agent.models import Order, OrderStatus, OrderType, Signal

logger = structlog.get_logger()


class ExecutionEngine:
    """
    Routes orders to either paper trading or live Polymarket CLOB.
    Handles order lifecycle: submit -> track -> fill/cancel.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.mode = config.mode  # "paper" or "live"
        self._open_orders: dict[str, Order] = {}
        self._filled_orders: list[Order] = []
        self._session: aiohttp.ClientSession | None = None

        # For paper trading: simulate fills
        self._paper_fill_rate: float = 0.85  # 85% of paper orders fill

    async def initialize(self):
        if self.mode == "live":
            logger.info("execution_engine_live", clob=self.config.polymarket_clob_url)
            # In live mode, initialize py-clob-client here
            # self.clob_client = ClobClient(...)
        else:
            logger.info("execution_engine_paper")

    async def submit_order(self, order: Order) -> Order:
        """Submit an order for execution."""
        if self.mode == "paper":
            return await self._paper_submit(order)
        else:
            return await self._live_submit(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id in self._open_orders:
            order = self._open_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()
            logger.info("order_cancelled", order_id=order_id)
            return True
        return False

    async def cancel_all(self, market_id: str | None = None):
        """Cancel all open orders, optionally filtered by market."""
        to_cancel = [
            oid for oid, o in self._open_orders.items()
            if market_id is None or o.market_id == market_id
        ]
        for oid in to_cancel:
            await self.cancel_order(oid)

    # ── Paper Trading ──────────────────────────────────

    async def _paper_submit(self, order: Order) -> Order:
        """Simulate order execution in paper mode."""
        import random

        # Simulate latency
        order.status = OrderStatus.OPEN
        order.updated_at = time.time()

        # Simulate fill probability
        if random.random() < self._paper_fill_rate:
            # Simulate some slippage
            slippage = random.uniform(-0.005, 0.005)
            fill_price = max(0.01, min(0.99, order.price + slippage))

            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.filled_price = fill_price
            order.updated_at = time.time()
            self._filled_orders.append(order)

            logger.info(
                "paper_fill",
                order_id=order.id,
                strategy=order.strategy.value,
                side=order.side.value,
                price=fill_price,
                size=order.size,
            )
        else:
            order.status = OrderStatus.OPEN
            self._open_orders[order.id] = order
            logger.info("paper_open", order_id=order.id, price=order.price)

        return order

    # ── Live Trading ───────────────────────────────────

    async def _live_submit(self, order: Order) -> Order:
        """
        Submit order to Polymarket CLOB.
        Uses py-clob-client for authentication and order signing.
        """
        # IMPORTANT: This is the integration point for live trading.
        # You must:
        # 1. Install py-clob-client
        # 2. Initialize with your API key and wallet
        # 3. Sign orders with your private key

        # Placeholder - implement with py-clob-client:
        #
        # from py_clob_client.client import ClobClient
        # from py_clob_client.order_builder.constants import BUY, SELL
        #
        # signed_order = self.clob_client.create_and_sign_order(
        #     OrderArgs(
        #         token_id=order.token_id,
        #         price=order.price,
        #         size=order.size / order.price,  # Convert USDC to shares
        #         side=BUY if order.side == Side.BUY else SELL,
        #     )
        # )
        # response = self.clob_client.post_order(signed_order)
        # order.exchange_order_id = response.get("orderID")

        logger.warning(
            "live_trading_not_implemented",
            message="Implement live trading with py-clob-client. See comments in code.",
        )
        order.status = OrderStatus.REJECTED
        order.updated_at = time.time()
        return order

    # ── Order Tracking ─────────────────────────────────

    def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        orders = list(self._open_orders.values())
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        return orders

    def get_fill_history(self, limit: int = 100) -> list[Order]:
        return self._filled_orders[-limit:]

    @property
    def total_orders_submitted(self) -> int:
        return len(self._filled_orders) + len(self._open_orders)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


def signal_to_order(signal: Signal, sized_amount: float) -> Order:
    """Convert a risk-approved signal into an executable order."""
    return Order(
        signal_id=signal.id,
        strategy=signal.strategy,
        market_id=signal.market_id,
        token_id=signal.token_id,
        side=signal.side,
        price=signal.market_price,  # Use current market price as limit
        size=sized_amount,
        order_type=OrderType.LIMIT,
    )
