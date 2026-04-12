"""
Odds API Price Feed — quality-first external price source.

Registers Odds API events as first-class markets in the world model using
synthetic canonical IDs (no fragile fuzzy matching to Smarkets). Tracks API
quota persistently to avoid blowing the 500/month free tier mid-bake.

Canonical market ID format:
    oddsapi:{sport_key}:{home_slug}_v_{away_slug}:h2h_{home|away}

Example:
    oddsapi:soccer_epl:liverpool_v_arsenal:h2h_home

Each Odds API event produces TWO markets in the world model:
one for the home team winning, one for the away team winning.
Both register with category="oddsapi:{sport_key}" so the recorder/analyser
can later compute per-source calibration.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import structlog

from agent.swarm.world_model import WorldModel

logger = structlog.get_logger()


def slugify(name: str) -> str:
    """Normalise a team name to a stable slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def make_market_ids(sport_key: str, home: str, away: str) -> tuple[str, str]:
    """Return (home_market_id, away_market_id) for an Odds API event."""
    h = slugify(home)
    a = slugify(away)
    base = f"oddsapi:{sport_key}:{h}_v_{a}"
    return f"{base}:h2h_home", f"{base}:h2h_away"


class OddsPriceFeed:
    """
    Polls The Odds API on a fixed cycle and pushes consensus prices into the
    world model. Designed to be the PRIMARY price source during paper bakes
    when Smarkets contracts endpoint is unreliable.
    """

    def __init__(
        self,
        world: WorldModel,
        api_key: str,
        sport_keys: list[str] | None = None,
        poll_interval: float = 90.0,
        quota_path: str = "./data/odds_api_quota.json",
        monthly_quota: int = 500,
    ):
        self.world = world
        self.api_key = api_key
        self.sport_keys = sport_keys or ["soccer_epl", "basketball_nba"]
        self.poll_interval = poll_interval
        self.quota_path = Path(quota_path)
        self.monthly_quota = monthly_quota
        self._running = False
        self._last_poll: dict[str, float] = {}
        self._known_events: dict[str, dict] = {}  # market_id -> {home, away, sport, event_id}
        self._stats = {
            "polls": 0,
            "events_seen": 0,
            "markets_registered": 0,
            "price_updates": 0,
            "api_errors": 0,
        }
        from agent.data.odds_api import OddsComparisonClient
        self.client = OddsComparisonClient(api_key=api_key)

    # ── Quota tracking ─────────────────────────────────

    def _load_quota(self) -> dict:
        if not self.quota_path.exists():
            return {"month": time.strftime("%Y-%m"), "used": 0}
        try:
            data = json.loads(self.quota_path.read_text())
            current_month = time.strftime("%Y-%m")
            if data.get("month") != current_month:
                return {"month": current_month, "used": 0}
            return data
        except Exception:
            return {"month": time.strftime("%Y-%m"), "used": 0}

    def _save_quota(self, used_increment: int = 1):
        data = self._load_quota()
        data["used"] = data.get("used", 0) + used_increment
        try:
            self.quota_path.parent.mkdir(parents=True, exist_ok=True)
            self.quota_path.write_text(json.dumps(data))
        except Exception as e:
            logger.debug("quota_save_error", error=str(e))

    def quota_remaining(self) -> int:
        data = self._load_quota()
        return max(0, self.monthly_quota - data.get("used", 0))

    # ── Main poll loop ─────────────────────────────────

    async def start(self):
        self._running = True
        # Stagger startup so we don't collide with agents starting up
        await asyncio.sleep(5)
        logger.info("odds_price_feed_starting", sports=self.sport_keys,
                    quota_remaining=self.quota_remaining())

        while self._running:
            try:
                await asyncio.wait_for(self._poll_cycle(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning("odds_feed_cycle_timeout")
                self._stats["api_errors"] += 1
            except Exception as e:
                logger.warning("odds_feed_cycle_error", error=str(e))
                self._stats["api_errors"] += 1
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._running = False

    async def _poll_cycle(self):
        # Hard quota guard — never poll if we have <50 requests left this month
        remaining = self.quota_remaining()
        if remaining < 50:
            logger.warning("odds_feed_quota_low", remaining=remaining)
            return

        self._stats["polls"] += 1

        for sport_key in self.sport_keys:
            try:
                consensus_list = await self.client.get_market_consensus(sport_key)
                self._save_quota(1)
                if not consensus_list:
                    continue
                self._stats["events_seen"] += len(consensus_list)
                for c in consensus_list:
                    self._ingest_event(sport_key, c)
                await asyncio.sleep(0.5)
            except Exception as e:
                self._stats["api_errors"] += 1
                logger.debug("odds_feed_sport_error", sport=sport_key, error=str(e))

        logger.info("odds_feed_cycle", **self._stats,
                    quota_remaining=self.quota_remaining())

    def _ingest_event(self, sport_key: str, consensus: dict):
        """
        Convert one Odds API consensus dict into two world model markets
        (home wins, away wins) with real consensus probabilities.
        """
        home = consensus.get("home_team", "")
        away = consensus.get("away_team", "")
        if not home or not away:
            return

        fair_probs = consensus.get("fair_probabilities", {})
        if not fair_probs:
            return

        home_prob = float(fair_probs.get(home, 0))
        away_prob = float(fair_probs.get(away, 0))
        if home_prob <= 0 or away_prob <= 0:
            return

        num_books = consensus.get("num_bookmakers", 0)
        # Volume proxy: more bookmakers = more liquid event. Scale to roughly
        # match Smarkets volume scale so the coordinator's volume gate behaves.
        volume_proxy = max(250.0, num_books * 100.0)

        home_id, away_id = make_market_ids(sport_key, home, away)

        home_question = f"{home} vs {away}: {home} to win"
        away_question = f"{home} vs {away}: {away} to win"
        category = f"oddsapi:{sport_key}"

        # Track known events for the OddsArbitrageAgent to consume
        if home_id not in self._known_events:
            self._stats["markets_registered"] += 2
        self._known_events[home_id] = {
            "home": home, "away": away, "sport": sport_key,
            "side": "home", "num_books": num_books,
        }
        self._known_events[away_id] = {
            "home": home, "away": away, "sport": sport_key,
            "side": "away", "num_books": num_books,
        }

        # Push prices into world model
        self.world.update_market_price(
            home_id, home_prob,
            question=home_question, volume=volume_proxy, category=category,
        )
        self.world.update_market_price(
            away_id, away_prob,
            question=away_question, volume=volume_proxy, category=category,
        )
        self._stats["price_updates"] += 2

    # ── Public API for agents ──────────────────────────

    def get_known_market_ids(self) -> list[str]:
        """Used by OddsArbitrageAgent to operate on feed-sourced markets."""
        return list(self._known_events.keys())

    def get_event_meta(self, market_id: str) -> dict | None:
        return self._known_events.get(market_id)

    def get_status(self) -> dict:
        return {
            **self._stats,
            "quota_remaining": self.quota_remaining(),
            "known_markets": len(self._known_events),
            "running": self._running,
        }