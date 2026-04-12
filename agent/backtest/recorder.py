"""
Black Swan Agent — Session Recorder

Passive recording layer that captures everything the swarm does for later
analysis. Designed to be completely non-blocking and crash-safe:

- All writes batched and flushed every 5 seconds in a background task
- Never blocks the decision path
- Fails silently on disk errors (recording is never critical-path)
- JSONL format: append-only, crash-resistant, trivially parseable
- Session-keyed so multiple runs can be analysed independently
- Five streams: markets, beliefs, edges, news, trades

Output: ./data/recordings/{session_id}/{stream}.jsonl

Usage in coordinator:

    from agent.backtest.recorder import SessionRecorder
    self.recorder = SessionRecorder()
    await self.recorder.start()

    # Then sprinkle these calls into the decision cycle / world model:
    self.recorder.record_market_snapshot(market_id, price, volume, category)
    self.recorder.record_belief(belief)
    self.recorder.record_edge(edge, composite_score)
    self.recorder.record_news(news_impact)
    self.recorder.record_trade(trade_dict)

    # On shutdown:
    await self.recorder.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def _out(msg: str):
    """Windows-safe print."""
    print(msg, flush=True)


@dataclass
class RecorderStats:
    markets_recorded: int = 0
    beliefs_recorded: int = 0
    edges_recorded: int = 0
    news_recorded: int = 0
    trades_recorded: int = 0
    flushes: int = 0
    write_errors: int = 0
    bytes_written: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "markets": self.markets_recorded,
            "beliefs": self.beliefs_recorded,
            "edges": self.edges_recorded,
            "news": self.news_recorded,
            "trades": self.trades_recorded,
            "flushes": self.flushes,
            "errors": self.write_errors,
            "bytes": self.bytes_written,
            "uptime_sec": round(time.time() - self.started_at, 1),
        }


class SessionRecorder:
    """
    Passive recorder that buffers events and flushes to disk in batches.

    Five separate JSONL streams per session, written to:
        ./data/recordings/{session_id}/

    The session_id is a date-prefixed UUID for natural sorting.
    """

    def __init__(
        self,
        base_dir: str = "./data/recordings",
        flush_interval_seconds: float = 5.0,
        max_buffer_size: int = 10000,
        market_snapshot_interval_seconds: float = 30.0,
    ):
        date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        self.session_id = f"{date_prefix}_{short_uuid}"

        self.base_dir = Path(base_dir) / self.session_id
        self.flush_interval = flush_interval_seconds
        self.max_buffer_size = max_buffer_size
        self.market_snapshot_interval = market_snapshot_interval_seconds

        # In-memory buffers — flushed in batches
        self._buffers: dict[str, list[dict]] = {
            "markets": [],
            "beliefs": [],
            "edges": [],
            "news": [],
            "trades": [],
        }

        # Track when we last snapshotted each market to throttle volume
        self._last_market_snapshot: dict[str, float] = {}

        # Throttle duplicate beliefs (same agent + market + ~same prob within 60s)
        self._recent_belief_hashes: dict[str, float] = {}

        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self.stats = RecorderStats()

    # ── Lifecycle ───────────────────────────────────────

    async def start(self):
        """Initialise the session directory and start the flush loop."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            _out(f"  [RECORDER] Failed to create directory {self.base_dir}: {e}")
            return

        # Write a session manifest
        manifest = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "started_at_ts": time.time(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
        try:
            (self.base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            _out(f"  [RECORDER] Manifest write failed: {e}")

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        _out(f"  [RECORDER] Session started: {self.session_id}")
        _out(f"  [RECORDER] Recording to: {self.base_dir}")

    async def stop(self):
        """Flush remaining buffers and stop the background task."""
        self._running = False
        if self._flush_task:
            try:
                await asyncio.wait_for(self._flush_task, timeout=10)
            except asyncio.TimeoutError:
                self._flush_task.cancel()

        # Final flush
        await self._flush_all()

        # Write session summary
        try:
            summary = {
                "session_id": self.session_id,
                "ended_at": datetime.now().isoformat(),
                "stats": self.stats.to_dict(),
            }
            (self.base_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        _out(f"  [RECORDER] Session stopped: {self.stats.to_dict()}")

    # ── Recording API (called from coordinator/world model) ─────

    def record_market_snapshot(
        self,
        market_id: str,
        price: float,
        volume: float = 0,
        category: str = "",
        question: str = "",
    ):
        """Snapshot a market price. Throttled to ~30s per market by default."""
        if not self._running:
            return
        try:
            now = time.time()
            last = self._last_market_snapshot.get(market_id, 0)
            if now - last < self.market_snapshot_interval:
                return
            self._last_market_snapshot[market_id] = now

            self._buffers["markets"].append({
                "ts": now,
                "session_id": self.session_id,
                "market_id": market_id,
                "price": round(float(price), 4),
                "volume": float(volume),
                "category": category,
                "question": question[:200] if question else "",
            })
            self.stats.markets_recorded += 1
            self._check_buffer_overflow("markets")
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_market_error", error=str(e))

    def record_belief(self, belief: Any):
        """Record an agent belief submission."""
        if not self._running:
            return
        try:
            # Deduplicate near-identical beliefs within a 60s window
            agent = getattr(belief, "agent_name", "unknown")
            mid = getattr(belief, "market_id", "")
            prob = getattr(belief, "probability", 0)
            bucket = round(float(prob), 2)
            dedup_key = f"{agent}:{mid}:{bucket}"

            now = time.time()
            last = self._recent_belief_hashes.get(dedup_key, 0)
            if now - last < 60:
                return
            self._recent_belief_hashes[dedup_key] = now

            # Periodic cleanup of dedup map
            if len(self._recent_belief_hashes) > 5000:
                cutoff = now - 300
                self._recent_belief_hashes = {
                    k: v for k, v in self._recent_belief_hashes.items() if v > cutoff
                }

            self._buffers["beliefs"].append({
                "ts": now,
                "session_id": self.session_id,
                "agent": agent,
                "market_id": mid,
                "probability": round(float(prob), 4),
                "confidence": round(float(getattr(belief, "confidence", 0)), 4),
                "reasoning": str(getattr(belief, "reasoning", ""))[:300],
                "ttl_seconds": int(getattr(belief, "ttl_seconds", 0)),
                "domain": str(getattr(belief, "domain", "")),
            })
            self.stats.beliefs_recorded += 1
            self._check_buffer_overflow("beliefs")
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_belief_error", error=str(e))

    def record_edge(self, edge: Any, composite_score: float = 0.0):
        """Record a computed edge with its composite score."""
        if not self._running:
            return
        try:
            self._buffers["edges"].append({
                "ts": time.time(),
                "session_id": self.session_id,
                "market_id": getattr(edge, "market_id", ""),
                "market_question": str(getattr(edge, "market_question", ""))[:200],
                "direction": str(getattr(edge, "direction", "")),
                "fair_value": round(float(getattr(edge, "fair_value", 0)), 4),
                "market_price": round(float(getattr(edge, "market_price", 0)), 4),
                "edge_pct": round(float(getattr(edge, "edge_pct", 0)), 3),
                "confidence": round(float(getattr(edge, "confidence", 0)), 4),
                "conviction": round(float(getattr(edge, "conviction", 0)), 4),
                "regime": str(getattr(edge, "regime", "")),
                "contributing_agents": list(getattr(edge, "contributing_agents", [])),
                "composite_score": round(float(composite_score), 3),
            })
            self.stats.edges_recorded += 1
            self._check_buffer_overflow("edges")
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_edge_error", error=str(e))

    def record_news(self, news_impact: Any):
        """Record a news impact submission."""
        if not self._running:
            return
        try:
            self._buffers["news"].append({
                "ts": time.time(),
                "session_id": self.session_id,
                "headline": str(getattr(news_impact, "headline", ""))[:300],
                "source": str(getattr(news_impact, "source", "")),
                "detected_by": str(getattr(news_impact, "detected_by", "")),
                "affected_markets": list(getattr(news_impact, "affected_markets", [])),
                "impact_direction": dict(getattr(news_impact, "impact_direction", {})),
                "urgency": round(float(getattr(news_impact, "urgency", 0)), 3),
                "confidence": round(float(getattr(news_impact, "confidence", 0)), 3),
            })
            self.stats.news_recorded += 1
            self._check_buffer_overflow("news")
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_news_error", error=str(e))

    def record_trade(
        self,
        trade_id: str,
        market_id: str,
        market_question: str,
        side: str,
        size_usd: float,
        entry_price: float,
        edge_pct: float,
        composite_score: float,
        contributing_agents: list[str],
        regime: str = "",
        adversarial_rating: str = "",
        kelly_fraction: float = 0.0,
        confidence_multiplier: float = 1.0,
    ):
        """Record a paper trade with full decision context."""
        if not self._running:
            return
        try:
            self._buffers["trades"].append({
                "ts": time.time(),
                "session_id": self.session_id,
                "trade_id": str(trade_id),
                "market_id": market_id,
                "market_question": market_question[:200],
                "side": side,
                "size_usd": round(float(size_usd), 2),
                "entry_price": round(float(entry_price), 4),
                "edge_pct": round(float(edge_pct), 3),
                "composite_score": round(float(composite_score), 3),
                "contributing_agents": list(contributing_agents),
                "regime": regime,
                "adversarial_rating": adversarial_rating,
                "kelly_fraction": round(float(kelly_fraction), 4),
                "confidence_multiplier": round(float(confidence_multiplier), 3),
            })
            self.stats.trades_recorded += 1
            self._check_buffer_overflow("trades")
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_trade_error", error=str(e))

    # ── Internal: flushing ──────────────────────────────

    def _check_buffer_overflow(self, stream: str):
        """If a buffer is too big, schedule an immediate flush."""
        if len(self._buffers[stream]) > self.max_buffer_size:
            asyncio.create_task(self._flush_stream(stream))

    async def _flush_loop(self):
        """Background task — flush all buffers every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("flush_loop_error", error=str(e))

    async def _flush_all(self):
        """Flush every stream's buffer to disk."""
        async with self._lock:
            for stream in list(self._buffers.keys()):
                await self._flush_stream_locked(stream)
            self.stats.flushes += 1

    async def _flush_stream(self, stream: str):
        """Flush a single stream (used by overflow trigger)."""
        async with self._lock:
            await self._flush_stream_locked(stream)

    async def _flush_stream_locked(self, stream: str):
        """Internal flush — caller must hold the lock."""
        buffer = self._buffers[stream]
        if not buffer:
            return

        # Swap the buffer atomically
        to_write = buffer
        self._buffers[stream] = []

        try:
            path = self.base_dir / f"{stream}.jsonl"
            # Synchronous file write — fast enough for batched JSONL
            # and avoids dependency on aiofiles
            lines = "\n".join(json.dumps(item, default=str) for item in to_write) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(lines)
            self.stats.bytes_written += len(lines.encode("utf-8"))
        except Exception as e:
            self.stats.write_errors += 1
            logger.debug("recorder_flush_error", stream=stream, error=str(e))
            # On failure, return items to buffer so we don't lose them
            self._buffers[stream] = to_write + self._buffers[stream]

    def get_status(self) -> dict:
        return {
            "session_id": self.session_id,
            "running": self._running,
            "buffer_sizes": {k: len(v) for k, v in self._buffers.items()},
            "stats": self.stats.to_dict(),
        }