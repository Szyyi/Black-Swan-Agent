"""Orchestrator: main event loop coordinating strategies, risk, and execution."""

from __future__ import annotations

import asyncio
import time

import structlog

from agent.config import AgentConfig, load_config
from agent.data.feeds import PolymarketClient
from agent.execution.engine import ExecutionEngine, signal_to_order
from agent.models import Market, OrderStatus, Signal, StrategyType
from agent.risk.manager import RiskManager
from agent.strategies import (
    ArbitrageStrategy,
    BaseStrategy,
    EventProbabilityStrategy,
    MarketMakingStrategy,
    SentimentStrategy,
)

logger = structlog.get_logger()


class Orchestrator:
    """
    Central coordinator that:
    1. Fetches market data on each tick
    2. Dispatches markets to each strategy for evaluation
    3. Collects signals and routes through risk management
    4. Sizes approved signals and sends to execution
    5. Tracks fills and updates portfolio state
    6. Logs everything for monitoring and analysis
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.running = False

        # Core components
        self.poly = PolymarketClient(
            gamma_url=config.polymarket_gamma_url,
            clob_url=config.polymarket_clob_url,
            ws_url=config.polymarket_ws_url,
        )
        self.risk = RiskManager(config.risk, total_capital=config.risk.max_total_exposure_usd)
        self.executor = ExecutionEngine(config)

        # Strategies
        self.strategies: list[BaseStrategy] = []
        self._init_strategies()

        # Market cache
        self._markets: list[Market] = []
        self._last_market_refresh: float = 0
        self._market_refresh_interval: float = 60  # Refresh market list every 60s

        # Performance tracking
        self._tick_count: int = 0
        self._signals_generated: int = 0
        self._orders_placed: int = 0
        self._start_time: float = 0

    def _init_strategies(self):
        """Initialize all enabled strategies."""
        cfg = self.config

        if cfg.arbitrage.enabled:
            self.strategies.append(
                ArbitrageStrategy(cfg.arbitrage, self.poly)
            )

        if cfg.sentiment.enabled:
            self.strategies.append(
                SentimentStrategy(
                    cfg.sentiment, self.poly,
                    anthropic_key=cfg.anthropic_api_key,
                    model=cfg.anthropic_model,
                )
            )

        if cfg.event_probability.enabled:
            self.strategies.append(
                EventProbabilityStrategy(
                    cfg.event_probability, self.poly,
                    anthropic_key=cfg.anthropic_api_key,
                    model=cfg.anthropic_model,
                )
            )

        if cfg.market_making.enabled:
            self.strategies.append(
                MarketMakingStrategy(cfg.market_making, self.poly)
            )

        logger.info(
            "strategies_loaded",
            count=len(self.strategies),
            types=[s.strategy_type.value for s in self.strategies],
        )

    # ── Main Loop ──────────────────────────────────────

    async def run(self):
        """Main trading loop."""
        self.running = True
        self._start_time = time.time()

        logger.info(
            "orchestrator_starting",
            mode=self.config.mode,
            strategies=len(self.strategies),
        )

        # Initialize all components
        await self.executor.initialize()
        for strategy in self.strategies:
            await strategy.initialize()

        try:
            while self.running:
                await self._tick()
                await asyncio.sleep(self._get_tick_interval())
        except KeyboardInterrupt:
            logger.info("shutdown_requested")
        except Exception as e:
            logger.critical("orchestrator_crash", error=str(e))
            raise
        finally:
            await self.shutdown()

    async def _tick(self):
        """Single iteration of the main loop."""
        self._tick_count += 1
        tick_start = time.time()

        # 1. Refresh market list periodically
        if time.time() - self._last_market_refresh > self._market_refresh_interval:
            await self._refresh_markets()

        if not self._markets:
            logger.warning("no_markets_available")
            return

        # 2. Run all strategies concurrently
        all_signals: list[Signal] = []
        strategy_tasks = [
            self._run_strategy(strategy) for strategy in self.strategies
            if strategy.enabled and strategy._active
        ]
        results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_signals.extend(result)
            elif isinstance(result, Exception):
                logger.error("strategy_error", error=str(result))

        self._signals_generated += len(all_signals)

        # 3. Sort signals by edge (best opportunities first)
        all_signals.sort(key=lambda s: s.edge_pct, reverse=True)

        # 4. Process signals through risk management and execute
        for signal in all_signals:
            await self._process_signal(signal)

        # 5. Log tick summary
        tick_duration = time.time() - tick_start
        if self._tick_count % 10 == 0:  # Log every 10th tick
            snapshot = self.risk.get_snapshot()
            logger.info(
                "tick_summary",
                tick=self._tick_count,
                duration_ms=round(tick_duration * 1000),
                signals=len(all_signals),
                positions=len(snapshot.positions),
                exposure=round(snapshot.total_exposure, 2),
                pnl=round(snapshot.total_pnl, 2),
            )

    async def _run_strategy(self, strategy: BaseStrategy) -> list[Signal]:
        """Run a single strategy's evaluation."""
        try:
            return await strategy.evaluate(self._markets)
        except Exception as e:
            logger.error(
                "strategy_evaluation_error",
                strategy=strategy.strategy_type.value,
                error=str(e),
            )
            return []

    async def _process_signal(self, signal: Signal):
        """Route a signal through risk management and execution."""
        # Risk check
        approved, reason = self.risk.check_signal(signal)
        if not approved:
            logger.debug(
                "signal_rejected",
                strategy=signal.strategy.value,
                reason=reason,
                market=signal.market_id[:8],
            )
            return

        # Size the order
        strategy = next(
            (s for s in self.strategies if s.strategy_type == signal.strategy), None
        )
        weight = strategy.weight if strategy else 0.25
        sized_amount = self.risk.size_order(signal, weight)

        if sized_amount <= 0:
            return

        # Create and submit order
        order = signal_to_order(signal, sized_amount)
        executed_order = await self.executor.submit_order(order)

        if executed_order.status == OrderStatus.FILLED:
            self.risk.on_fill(executed_order)
            self._orders_placed += 1

            logger.info(
                "order_filled",
                strategy=signal.strategy.value,
                side=signal.side.value,
                outcome=signal.outcome.value,
                price=executed_order.filled_price,
                size=executed_order.filled_size,
                edge=round(signal.edge_pct, 1),
                reasoning=signal.reasoning[:80],
            )

    # ── Market Refresh ─────────────────────────────────

    async def _refresh_markets(self):
        """Fetch current active markets from Polymarket."""
        try:
            markets = await self.poly.get_markets(active=True, limit=200)
            self._markets = [m for m in markets if m.active]
            self._last_market_refresh = time.time()
            logger.info("markets_refreshed", count=len(self._markets))
        except Exception as e:
            logger.error("market_refresh_error", error=str(e))

    # ── Tick Interval ──────────────────────────────────

    def _get_tick_interval(self) -> float:
        """
        Dynamic tick interval based on which strategies are active.
        Arbitrage needs fast ticks; sentiment can be slower.
        """
        has_arb = any(
            s.strategy_type == StrategyType.ARBITRAGE and s.enabled
            for s in self.strategies
        )
        has_mm = any(
            s.strategy_type == StrategyType.MARKET_MAKING and s.enabled
            for s in self.strategies
        )

        if has_arb:
            return 1.0   # 1 second for arbitrage
        elif has_mm:
            return 5.0   # 5 seconds for market making
        else:
            return 15.0  # 15 seconds for slower strategies

    # ── Control ────────────────────────────────────────

    def stop(self):
        """Gracefully stop the orchestrator."""
        self.running = False

    async def shutdown(self):
        """Clean up all resources."""
        logger.info("orchestrator_shutting_down")

        # Cancel all open orders
        await self.executor.cancel_all()

        # Shutdown strategies
        for strategy in self.strategies:
            await strategy.shutdown()

        # Close connections
        await self.poly.close()
        await self.executor.close()

        # Final report
        uptime = time.time() - self._start_time if self._start_time else 0
        snapshot = self.risk.get_snapshot()

        logger.info(
            "final_report",
            uptime_hours=round(uptime / 3600, 2),
            ticks=self._tick_count,
            signals=self._signals_generated,
            orders=self._orders_placed,
            total_trades=snapshot.total_trades,
            total_pnl=round(snapshot.total_pnl, 2),
            max_drawdown=round(snapshot.max_drawdown, 2),
            win_rate=snapshot.win_rate,
        )

    # ── Status ─────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current agent status for monitoring."""
        snapshot = self.risk.get_snapshot()
        return {
            "running": self.running,
            "mode": self.config.mode,
            "uptime_sec": time.time() - self._start_time if self._start_time else 0,
            "tick_count": self._tick_count,
            "signals_generated": self._signals_generated,
            "orders_placed": self._orders_placed,
            "active_strategies": [
                s.strategy_type.value for s in self.strategies if s.enabled
            ],
            "portfolio": {
                "total_capital": snapshot.total_capital,
                "available": snapshot.available_capital,
                "exposure": snapshot.total_exposure,
                "positions": len(snapshot.positions),
                "daily_pnl": snapshot.daily_pnl,
                "total_pnl": snapshot.total_pnl,
                "max_drawdown": snapshot.max_drawdown,
                "win_rate": snapshot.win_rate,
                "total_trades": snapshot.total_trades,
            },
        }
