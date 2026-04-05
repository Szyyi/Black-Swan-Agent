"""Configuration loader with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        for match in pattern.finditer(value):
            env_val = os.environ.get(match.group(1), "")
            value = value.replace(match.group(0), env_val)
        return value
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


class StrategyConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.25
    max_position_usd: float = 500.0

class ArbitrageConfig(StrategyConfig):
    min_spread_bps: float = 150.0
    exchanges: list[str] = ["binance"]
    markets: list[str] = ["btc_5min"]
    max_latency_ms: float = 500.0

class SentimentConfig(StrategyConfig):
    min_edge_pct: float = 5.0
    recheck_interval_sec: int = 300
    news_sources: list[str] = ["reuters_rss"]
    confidence_threshold: float = 0.7

class MarketMakingConfig(StrategyConfig):
    target_spread_bps: float = 200.0
    max_inventory_usd: float = 1000.0
    rebalance_threshold: float = 0.6
    min_liquidity: float = 5000.0
    quote_refresh_sec: int = 10

class EventProbabilityConfig(StrategyConfig):
    min_edge_pct: float = 8.0
    ensemble_models: int = 3
    update_interval_sec: int = 600

class RiskConfig(BaseModel):
    max_total_exposure_usd: float = 5000.0
    max_single_market_pct: float = 15.0
    max_daily_loss_usd: float = 500.0
    max_drawdown_pct: float = 10.0
    correlation_limit: float = 0.7
    kill_switch_loss_usd: float = 1000.0
    cooldown_after_loss_sec: int = 3600


class AgentConfig(BaseModel):
    mode: str = "paper"
    platform: str = "smarkets"

    # Smarkets
    smarkets_username: str = ""
    smarkets_password: str = ""
    smarkets_api_url: str = "https://api.smarkets.com/v3"

    # External data APIs
    football_data_api_key: str = ""
    odds_api_key: str = ""

    # Strategy configs
    arbitrage: ArbitrageConfig = ArbitrageConfig()
    sentiment: SentimentConfig = SentimentConfig()
    market_making: MarketMakingConfig = MarketMakingConfig()
    event_probability: EventProbabilityConfig = EventProbabilityConfig()
    risk: RiskConfig = RiskConfig()

    # Polymarket (unused for Smarkets)
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_ws_url: str = ""
    rpc_url: str = ""
    private_key: str = ""
    wallet_address: str = ""

    # AI
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Storage
    db_path: str = "./data/trades.db"

    # Monitoring
    log_level: str = "INFO"
    metrics_port: int = 9090


def load_config(path: str = "config.yaml") -> AgentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    raw = _expand_env_vars(raw)

    strategies = raw.get("strategies", {})
    wallet = raw.get("wallet", {})
    polymarket = raw.get("polymarket", {})
    anthropic = raw.get("anthropic", {})
    risk = raw.get("risk", {})
    storage = raw.get("storage", {})
    monitoring = raw.get("monitoring", {})
    smarkets = raw.get("smarkets", {})
    data_apis = raw.get("data_apis", {})

    return AgentConfig(
        mode=raw.get("mode", "paper"),
        platform=raw.get("platform", "smarkets"),
        smarkets_username=smarkets.get("username", ""),
        smarkets_password=smarkets.get("password", ""),
        smarkets_api_url=smarkets.get("api_url", "https://api.smarkets.com/v3"),
        football_data_api_key=data_apis.get("football_data_key", ""),
        odds_api_key=data_apis.get("odds_api_key", ""),
        arbitrage=ArbitrageConfig(**strategies.get("arbitrage", {})),
        sentiment=SentimentConfig(**strategies.get("sentiment", {})),
        market_making=MarketMakingConfig(**strategies.get("market_making", {})),
        event_probability=EventProbabilityConfig(**strategies.get("event_probability", {})),
        risk=RiskConfig(**risk),
        polymarket_clob_url=polymarket.get("clob_url", "https://clob.polymarket.com"),
        polymarket_gamma_url=polymarket.get("gamma_url", ""),
        polymarket_ws_url=polymarket.get("ws_url", ""),
        rpc_url=polymarket.get("rpc_url", ""),
        private_key=wallet.get("private_key", ""),
        wallet_address=wallet.get("address", ""),
        anthropic_api_key=anthropic.get("api_key", ""),
        anthropic_model=anthropic.get("model", "claude-sonnet-4-20250514"),
        db_path=storage.get("db_path", "./data/trades.db"),
        log_level=monitoring.get("log_level", "INFO"),
        metrics_port=monitoring.get("metrics_port", 9090),
    )