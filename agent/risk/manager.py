"""Risk management: position limits, drawdown control, kill switches."""

from __future__ import annotations

import time
from collections import defaultdict

import structlog

from agent.config import RiskConfig
from agent.models import (
    Order, Position, PortfolioSnapshot, Signal, StrategyType, TradeRecord,
)

logger = structlog.get_logger()


class RiskManager:
    """
    Central risk manager. Every signal must pass through here before
    becoming an order. Enforces:
    - Total exposure limits
    - Per-market concentration limits
    - Daily loss limits / drawdown
    - Kill switch on catastrophic loss
    - Cooldown periods after large losses
    - Strategy-level weight enforcement
    """

    def __init__(self, config: RiskConfig, total_capital: float):
        self.config = config
        self.total_capital = total_capital
        self.available_capital = total_capital

        # Tracking state
        self.positions: dict[str, Position] = {}  # market_id -> Position
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.peak_capital: float = total_capital
        self.max_drawdown: float = 0.0
        self.trade_history: list[TradeRecord] = []
        self.daily_trade_count: int = 0

        # Kill switch & cooldown
        self._killed: bool = False
        self._cooldown_until: float = 0.0

        # Per-strategy exposure tracking
        self._strategy_exposure: dict[StrategyType, float] = defaultdict(float)

        logger.info(
            "risk_manager_initialized",
            total_capital=total_capital,
            max_exposure=config.max_total_exposure_usd,
            kill_switch=config.kill_switch_loss_usd,
        )

    # ── Signal Validation ──────────────────────────────

    def check_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Validate a signal against all risk rules.
        Returns (approved, reason).
        """
        # Kill switch
        if self._killed:
            return False, "KILL_SWITCH_ACTIVE"

        # Cooldown
        if time.time() < self._cooldown_until:
            remaining = int(self._cooldown_until - time.time())
            return False, f"COOLDOWN_ACTIVE ({remaining}s remaining)"

        # Daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            return False, f"DAILY_LOSS_LIMIT (${self.daily_pnl:.2f})"

        # Drawdown limit
        current_capital = self.total_capital + self.total_pnl
        if self.peak_capital > 0:
            drawdown_pct = ((self.peak_capital - current_capital) / self.peak_capital) * 100
            if drawdown_pct >= self.config.max_drawdown_pct:
                return False, f"MAX_DRAWDOWN ({drawdown_pct:.1f}%)"

        # Total exposure limit
        total_exposure = sum(p.market_value for p in self.positions.values())
        if total_exposure + signal.suggested_size_usd > self.config.max_total_exposure_usd:
            return False, f"TOTAL_EXPOSURE_LIMIT (${total_exposure:.2f})"

        # Per-market concentration
        if signal.market_id in self.positions:
            existing = self.positions[signal.market_id].market_value
            combined = existing + signal.suggested_size_usd
            max_allowed = self.total_capital * (self.config.max_single_market_pct / 100)
            if combined > max_allowed:
                return False, f"MARKET_CONCENTRATION ({combined/self.total_capital*100:.1f}%)"

        # Available capital
        if signal.suggested_size_usd > self.available_capital:
            return False, f"INSUFFICIENT_CAPITAL (need ${signal.suggested_size_usd:.2f}, have ${self.available_capital:.2f})"

        return True, "APPROVED"

    def size_order(self, signal: Signal, strategy_weight: float) -> float:
        """
        Calculate the actual order size based on signal strength,
        strategy allocation, and risk constraints.
        """
        # Strategy allocation
        max_strategy_capital = self.total_capital * strategy_weight
        strategy_used = self._strategy_exposure.get(signal.strategy, 0.0)
        strategy_remaining = max(0, max_strategy_capital - strategy_used)

        # Scale by confidence
        confidence_scalar = min(1.0, signal.confidence)
        base_size = min(signal.suggested_size_usd, strategy_remaining)
        sized = base_size * confidence_scalar

        # Apply per-market max
        max_per_market = self.total_capital * (self.config.max_single_market_pct / 100)
        existing_exposure = 0.0
        if signal.market_id in self.positions:
            existing_exposure = self.positions[signal.market_id].market_value
        sized = min(sized, max_per_market - existing_exposure)

        # Don't exceed available capital
        sized = min(sized, self.available_capital)

        # Minimum viable order (avoid dust)
        if sized < 1.0:
            return 0.0

        return round(sized, 2)

    # ── Position Updates ───────────────────────────────

    def on_fill(self, order: Order, market_question: str = ""):
        """Update state when an order is filled."""
        key = order.market_id

        if key in self.positions:
            pos = self.positions[key]
            # Update average entry price
            total_cost = (pos.size * pos.avg_entry_price) + (order.filled_size * order.filled_price)
            pos.size += order.filled_size
            if pos.size > 0:
                pos.avg_entry_price = total_cost / pos.size
        else:
            self.positions[key] = Position(
                market_id=order.market_id,
                token_id=order.token_id,
                outcome=order.side.value,
                strategy=order.strategy,
                size=order.filled_size,
                avg_entry_price=order.filled_price,
                current_price=order.filled_price,
            )

        # Update capital tracking
        cost = order.filled_size * order.filled_price
        self.available_capital -= cost
        self._strategy_exposure[order.strategy] += cost
        self.daily_trade_count += 1

        # Record trade
        self.trade_history.append(TradeRecord(
            order_id=order.id,
            signal_id=order.signal_id,
            strategy=order.strategy,
            market_id=order.market_id,
            market_question=market_question,
            side=order.side,
            outcome=order.side.value,
            price=order.filled_price,
            size=order.filled_size,
        ))

        logger.info(
            "position_updated",
            market=order.market_id[:8],
            strategy=order.strategy.value,
            size=order.filled_size,
            price=order.filled_price,
        )

    def on_settlement(self, market_id: str, payout_per_share: float):
        """Handle market resolution / settlement."""
        if market_id not in self.positions:
            return

        pos = self.positions[market_id]
        payout = pos.size * payout_per_share
        cost = pos.cost_basis
        pnl = payout - cost

        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.available_capital += payout

        # Update peak & drawdown
        current = self.total_capital + self.total_pnl
        self.peak_capital = max(self.peak_capital, current)
        drawdown = ((self.peak_capital - current) / self.peak_capital) * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Check kill switch
        if self.total_pnl <= -self.config.kill_switch_loss_usd:
            self._killed = True
            logger.critical("KILL_SWITCH_TRIGGERED", total_pnl=self.total_pnl)

        # Check if cooldown needed
        if pnl < 0 and abs(pnl) > self.config.max_daily_loss_usd * 0.5:
            self._cooldown_until = time.time() + self.config.cooldown_after_loss_sec
            logger.warning("cooldown_activated", duration=self.config.cooldown_after_loss_sec)

        # Remove position
        self._strategy_exposure[pos.strategy] -= pos.cost_basis
        del self.positions[market_id]

        logger.info(
            "position_settled",
            market=market_id[:8],
            pnl=round(pnl, 2),
            total_pnl=round(self.total_pnl, 2),
        )

    def update_prices(self, market_id: str, current_price: float):
        """Update mark-to-market prices."""
        if market_id in self.positions:
            pos = self.positions[market_id]
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.avg_entry_price) * pos.size

    # ── Portfolio Snapshot ─────────────────────────────

    def get_snapshot(self) -> PortfolioSnapshot:
        total_exposure = sum(p.market_value for p in self.positions.values())
        win_count = sum(1 for t in self.trade_history if t.pnl > 0)
        total = len(self.trade_history)

        return PortfolioSnapshot(
            total_capital=self.total_capital,
            available_capital=self.available_capital,
            total_exposure=total_exposure,
            positions=list(self.positions.values()),
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            max_drawdown=self.max_drawdown,
            win_rate=(win_count / total * 100) if total > 0 else None,
            total_trades=total,
        )

    def reset_daily(self):
        """Reset daily counters (call at midnight UTC)."""
        logger.info("daily_reset", daily_pnl=self.daily_pnl, trades=self.daily_trade_count)
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
