"""Main entry point for the Polymarket Multi-Strategy Trading Agent."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

import structlog


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    import logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Multi-Strategy AI Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agent.main --mode paper              # Paper trading (default)
  python -m agent.main --mode live                # Live trading (real money)
  python -m agent.main --swarm                    # Multi-agent swarm mode
  python -m agent.main --swarm --mode paper       # Swarm + paper trading
  python -m agent.main --config my_config.yaml    # Custom config file
  python -m agent.main --status                   # Print status and exit
        """,
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default=None,
        help="Trading mode (overrides config file)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit",
    )
    parser.add_argument(
        "--swarm", action="store_true",
        help="Use multi-agent swarm mode (recommended)",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    from agent.config import load_config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run: cp config.example.yaml config.yaml", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.mode:
        config.mode = args.mode
    if args.log_level:
        config.log_level = args.log_level

    setup_logging(config.log_level)
    log = structlog.get_logger()

    # Safety confirmation for live mode
    if config.mode == "live":
        print("\n" + "=" * 60)
        print("⚠️  LIVE TRADING MODE")
        print("=" * 60)
        print("This will execute REAL trades with REAL money.")
        print("Ensure you have:")
        print("  - Tested extensively in paper mode")
        print("  - Set appropriate risk limits in config.yaml")
        print("  - Verified jurisdictional compliance")
        print("  - Funded your wallet with USDC on Polygon")
        print("=" * 60)
        confirm = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
        if confirm != "I UNDERSTAND THE RISKS":
            print("Aborted.")
            sys.exit(0)

    # Create orchestrator
    if args.swarm:
        from agent.swarm.coordinator import MetaCoordinator
        orchestrator = MetaCoordinator(config)
        log.info("swarm_mode_enabled", agents="12 specialized agents + adversarial + intelligence layer")
    else:
        from agent.orchestrator import Orchestrator
        orchestrator = Orchestrator(config)

    if args.status:
        print(orchestrator.get_status())
        sys.exit(0)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(signum, frame):
        log.info("signal_received", signal=signum)
        orchestrator.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Banner
    # Banner
    if args.swarm:
        log.info(
            "swarm_starting",
            mode=config.mode,
            risk_limits={
                "max_exposure": config.risk.max_total_exposure_usd,
                "daily_loss_limit": config.risk.max_daily_loss_usd,
                "kill_switch": config.risk.kill_switch_loss_usd,
            },
        )
    else:
        log.info(
            "agent_starting",
            mode=config.mode,
            strategies={
                "arbitrage": config.arbitrage.enabled,
                "sentiment": config.sentiment.enabled,
                "market_making": config.market_making.enabled,
                "event_probability": config.event_probability.enabled,
            },
            risk_limits={
                "max_exposure": config.risk.max_total_exposure_usd,
                "daily_loss_limit": config.risk.max_daily_loss_usd,
                "kill_switch": config.risk.kill_switch_loss_usd,
            },
        )

    # Run
    try:
        loop.run_until_complete(orchestrator.run())
    except Exception as e:
        log.critical("fatal_error", error=str(e))
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
