"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from agent.models import Market, Signal, StrategyType

logger = structlog.get_logger()


class BaseStrategy(ABC):
    """
    All strategies inherit from this base class.
    Each strategy independently generates signals that the
    orchestrator then routes through risk management.
    """

    strategy_type: StrategyType

    def __init__(self, enabled: bool = True, weight: float = 0.25):
        self.enabled = enabled
        self.weight = weight
        self._signal_count = 0
        self._active = True

    @abstractmethod
    async def initialize(self):
        """One-time setup: load data, connect to feeds, etc."""
        ...

    @abstractmethod
    async def evaluate(self, markets: list[Market]) -> list[Signal]:
        """
        Evaluate current market conditions and return trading signals.
        Called periodically by the orchestrator.
        """
        ...

    @abstractmethod
    async def shutdown(self):
        """Clean up resources."""
        ...

    def pause(self):
        self._active = False
        logger.info("strategy_paused", strategy=self.strategy_type.value)

    def resume(self):
        self._active = True
        logger.info("strategy_resumed", strategy=self.strategy_type.value)
