"""
World Model v3: Next-generation belief engine with novel features.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from threading import RLock
 
from agent.swarm.decay import DecayEngine

import structlog

logger = structlog.get_logger()


@dataclass
class Belief:
    agent_name: str
    market_id: str
    probability: float
    confidence: float
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 600
    domain: str = "general"


@dataclass
class Edge:
    market_id: str
    market_question: str
    market_price: float
    fair_value: float
    edge_pct: float
    confidence: float
    contributing_agents: list[str]
    direction: str
    reasoning: str
    quality_score: float = 0.0
    conviction: float = 0.0
    surprise_factor: float = 0.0
    entropy: float = 0.0
    regime: str = "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Correlation:
    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    correlation_type: str
    strength: float
    description: str
    detected_by: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class NewsImpact:
    headline: str
    source: str
    affected_markets: list[str]
    impact_direction: dict[str, float]
    urgency: float
    confidence: float
    detected_by: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingSignal:
    market_id: str
    action: str
    urgency: float
    reasoning: str
    expires_at: float
    detected_by: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MarketRegime:
    market_id: str
    regime: str
    confidence: float
    volatility: float
    trend_strength: float
    updated_at: float = field(default_factory=time.time)


@dataclass
class SurpriseEvent:
    market_id: str
    old_consensus: float
    new_belief: float
    shift_magnitude: float
    agent: str
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class AgentTrustScorer:
    def __init__(self):
        self._predictions: dict[str, list[dict]] = defaultdict(list)
        self._domain_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._ensemble_scores: dict[str, list[float]] = defaultdict(list)

    def record_prediction(self, agent, market_id, predicted, domain="general"):
        self._predictions[agent].append({"market_id": market_id, "predicted": predicted, "domain": domain, "timestamp": time.time(), "resolved": False})

    def record_resolution(self, market_id, actual_outcome):
        for agent, preds in self._predictions.items():
            for pred in preds:
                if pred["market_id"] == market_id and not pred["resolved"]:
                    pred["resolved"] = True
                    error = abs(pred["predicted"] - actual_outcome)
                    accuracy = max(0, 1.0 - error * 2)
                    self._domain_scores[agent][pred["domain"]].append(accuracy)
                    self._domain_scores[agent][pred["domain"]] = self._domain_scores[agent][pred["domain"]][-50:]

    def record_ensemble_result(self, agents, accuracy):
        key = "+".join(sorted(agents))
        self._ensemble_scores[key].append(accuracy)
        self._ensemble_scores[key] = self._ensemble_scores[key][-30:]

    def get_trust_weight(self, agent, domain="general"):
        scores = self._domain_scores.get(agent, {}).get(domain, [])
        if len(scores) < 5:
            scores = self._domain_scores.get(agent, {}).get("general", [])
        if len(scores) < 5:
            return 1.0
        return 0.4 + mean(scores[-20:]) * 1.2

    def get_rankings(self):
        rankings = []
        for agent, domains in self._domain_scores.items():
            all_s = []
            for s in domains.values():
                all_s.extend(s)
            if all_s:
                rankings.append({"agent": agent, "avg_accuracy": round(mean(all_s), 3), "predictions": len(all_s), "domains": {d: round(mean(s), 3) for d, s in domains.items() if s}})
        rankings.sort(key=lambda x: x["avg_accuracy"], reverse=True)
        return rankings


class CalibrationTracker:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self._bins: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def record(self, predicted_confidence, was_correct):
        bin_idx = min(self.num_bins - 1, int(predicted_confidence * self.num_bins))
        self._bins[bin_idx].append((predicted_confidence, 1.0 if was_correct else 0.0))
        self._bins[bin_idx] = self._bins[bin_idx][-50:]

    def get_calibration_adjustment(self, raw_confidence):
        bin_idx = min(self.num_bins - 1, int(raw_confidence * self.num_bins))
        data = self._bins.get(bin_idx, [])
        if len(data) < 5:
            return raw_confidence
        actual_rate = mean(o for _, o in data)
        expected_rate = mean(c for c, _ in data)
        if expected_rate == 0:
            return raw_confidence
        ratio = actual_rate / expected_rate
        return max(0.1, min(0.95, raw_confidence * ratio))

    def get_calibration_curve(self):
        curve = {}
        for i in range(self.num_bins):
            data = self._bins.get(i, [])
            if data:
                curve[f"{i*10}-{(i+1)*10}%"] = {"expected": round(mean(c for c, _ in data), 3), "actual": round(mean(o for _, o in data), 3), "samples": len(data)}
        return curve


class WorldModel:
    def __init__(self):
        self._lock = RLock()
        self._beliefs: dict[str, list[Belief]] = defaultdict(list)
        self._edges: list[Edge] = []
        self._correlations: list[Correlation] = []
        self._news_impacts: list[NewsImpact] = []
        self._timing_signals: list[TimingSignal] = []
        self._market_prices: dict[str, float] = {}
        self._market_questions: dict[str, str] = {}
        self._market_volumes: dict[str, float] = {}
        self._market_categories: dict[str, str] = {}
        self._price_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._regimes: dict[str, MarketRegime] = {}
        self.trust = AgentTrustScorer()
        self.calibration = CalibrationTracker()
        self._surprises: list[SurpriseEvent] = []
        self._belief_history: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        self._entropy: dict[str, float] = {}
        self._conviction: dict[str, float] = {}
        self._agent_last_active: dict[str, float] = {}
        self._conflicts: list[dict] = []
        self.decay_engine = DecayEngine()
        self._event_callbacks: dict[str, list] = defaultdict(list)
        from collections import deque
        self._thought_buffer: deque = deque(maxlen=200)

    def set_recorder(self, recorder):
        """Optional: attach a SessionRecorder for passive event capture."""
        self._recorder = recorder

    def _record_belief(self, belief):
        rec = getattr(self, "_recorder", None)
        if rec is not None:
            rec.record_belief(belief)

    def _record_news(self, news):
        rec = getattr(self, "_recorder", None)
        if rec is not None:
            rec.record_news(news)

    def _record_market(self, market_id, price, volume=0, category="", question=""):
        rec = getattr(self, "_recorder", None)
        if rec is not None:
            rec.record_market_snapshot(market_id, price, volume, category, question)


    def register_event_callback(self, event_type: str, callback):
        """Register callback for event types: 'price_update', 'news_impact', 'surprise', 'belief_update'"""
        self._event_callbacks[event_type].append(callback)


    def submit_belief(self, belief: Belief):
        with self._lock:
            old_consensus, old_conf, _ = self._get_consensus_unlocked(belief.market_id)
            self._beliefs[belief.market_id] = [b for b in self._beliefs[belief.market_id] if b.agent_name != belief.agent_name]
            self._beliefs[belief.market_id].append(belief)
            self._agent_last_active[belief.agent_name] = time.time()
            domain = self._market_categories.get(belief.market_id, "general")
            self.trust.record_prediction(belief.agent_name, belief.market_id, belief.probability, domain)
            self._record_belief(belief)
            

            # ── Narrate agent thinking ──
            # ── Narrate agent thinking ──
            market_q = self._market_questions.get(belief.market_id, belief.market_id[:8])[:45]
            price = self._market_prices.get(belief.market_id, 0.5)
            edge = (belief.probability - price) * 100
            direction = "YES" if edge > 0 else "NO"
            import sys
            print(f"  [{belief.agent_name}] {market_q}", flush=True)
            print(f"    thinks {belief.probability:.0%} (market: {price:.0%}, edge: {edge:+.1f}%) conf={belief.confidence:.0%}", flush=True)
            print(f"    reason: {belief.reasoning[:80]}", flush=True)

            # Push to ring buffer for dashboard
            self._thought_buffer.append({
                "ts": time.time(),
                "agent": belief.agent_name,
                "market_id": belief.market_id,
                "question": market_q,
                "probability": float(belief.probability),
                "market_price": float(price),
                "edge": float(edge),
                "confidence": float(belief.confidence),
                "reasoning": belief.reasoning[:120],
                "surprise": False,
            })

            # Surprise detection with narration
            if old_consensus is not None:
                shift = abs(belief.probability - old_consensus)
                if shift > 0.10:
                    self._surprises.append(SurpriseEvent(market_id=belief.market_id, old_consensus=old_consensus, new_belief=belief.probability, shift_magnitude=shift, agent=belief.agent_name, reasoning=belief.reasoning))
                    self._surprises = self._surprises[-50:]
                    print(f"    *** SURPRISE: shifted consensus by {shift:.0%} ***", flush=True)
                    if self._thought_buffer:
                        self._thought_buffer[-1]["surprise"] = True

            new_consensus, new_conf, contributors = self._get_consensus_unlocked(belief.market_id)
            if new_consensus is not None:
                self._belief_history[belief.market_id].append((time.time(), new_consensus, new_conf))
                self._belief_history[belief.market_id] = self._belief_history[belief.market_id][-100:]

                # Show consensus update
                if len(contributors) > 1:
                    print(f"    >> consensus now {new_consensus:.0%} from {len(contributors)} agents: {', '.join(contributors)}", flush=True)

            self._update_entropy(belief.market_id)

            # Belief propagation with narration
            corrs = [c for c in self._correlations if c.market_a_id == belief.market_id or c.market_b_id == belief.market_id]
            if corrs and old_consensus is not None:
                for corr in corrs:
                    other_id = corr.market_b_id if corr.market_a_id == belief.market_id else corr.market_a_id
                    other_price = self._market_prices.get(other_id)
                    if other_price is None:
                        continue
                    bshift = belief.probability - old_consensus
                    if abs(bshift) < 0.02:
                        continue
                    if corr.correlation_type == "positive":
                        implied = bshift * corr.strength * 0.5
                    elif corr.correlation_type == "negative":
                        implied = -bshift * corr.strength * 0.5
                    else:
                        continue
                    implied_prob = max(0.02, min(0.98, other_price + implied))
                    other_q = self._market_questions.get(other_id, other_id[:8])[:40]
                    print(f"    >> propagating {implied:+.1%} to '{other_q}' ({corr.correlation_type} link)", flush=True)
                    self._beliefs[other_id] = [b for b in self._beliefs.get(other_id, []) if b.agent_name != "belief_propagation"]
                    self._beliefs[other_id].append(Belief(agent_name="belief_propagation", market_id=other_id, probability=implied_prob, confidence=belief.confidence * corr.strength * 0.3, reasoning=f"Propagated from {belief.market_id[:8]}", ttl_seconds=300))

            print("", flush=True)

    def _propagate_belief(self, belief: Belief):
        corrs = [c for c in self._correlations if c.market_a_id == belief.market_id or c.market_b_id == belief.market_id]
        if not corrs:
            return
        old_consensus, _, _ = self._get_consensus_unlocked(belief.market_id)
        if old_consensus is None:
            return
        for corr in corrs:
            other_id = corr.market_b_id if corr.market_a_id == belief.market_id else corr.market_a_id
            other_price = self._market_prices.get(other_id)
            if other_price is None:
                continue
            shift = belief.probability - old_consensus
            if abs(shift) < 0.02:
                continue
            if corr.correlation_type == "positive":
                implied_shift = shift * corr.strength * 0.5
            elif corr.correlation_type == "negative":
                implied_shift = -shift * corr.strength * 0.5
            else:
                continue
            implied_prob = max(0.02, min(0.98, other_price + implied_shift))
            self._beliefs[other_id] = [b for b in self._beliefs.get(other_id, []) if b.agent_name != "belief_propagation"]
            self._beliefs[other_id].append(Belief(agent_name="belief_propagation", market_id=other_id, probability=implied_prob, confidence=belief.confidence * corr.strength * 0.3, reasoning=f"Propagated from {belief.market_id[:8]} via {corr.correlation_type} link", ttl_seconds=300))

    def get_consensus(self, market_id):
        with self._lock:
            return self._get_consensus_unlocked(market_id)

    def _get_consensus_unlocked(self, market_id):
        beliefs = self._get_valid_beliefs(market_id)
        if not beliefs:
            return None, 0.0, []
        now = time.time()
        domain = self._market_categories.get(market_id, "general")
        weighted_sum = 0.0
        total_weight = 0.0
        agents = []
        for b in beliefs:
            age = now - b.timestamp
            regime = self._regimes.get(market_id)
            regime_str = regime.regime if regime else "unknown"
            freshness = self.decay_engine.compute_freshness(
                b.agent_name, age, regime_str
            )
            trust = self.trust.get_trust_weight(b.agent_name, domain)
            weight = b.confidence * trust * freshness
            if weight > 0.01:
                weighted_sum += b.probability * weight
                total_weight += weight
                agents.append(b.agent_name)
        if total_weight == 0:
            return None, 0.0, []
        consensus = weighted_sum / total_weight
        if len(beliefs) >= 2:
            probs = [b.probability for b in beliefs]
            disagreement = stdev(probs) if len(probs) >= 2 else 0
            agreement_factor = max(0.3, 1.0 - disagreement * 2.5)
            if disagreement > 0.15:
                self._record_conflict(market_id, beliefs, disagreement)
        else:
            agreement_factor = 0.6
        avg_confidence = (total_weight / len(beliefs)) * agreement_factor
        calibrated_conf = self.calibration.get_calibration_adjustment(min(1.0, avg_confidence))
        return consensus, calibrated_conf, list(set(agents))

    def _get_valid_beliefs(self, market_id):
        now = time.time()
        regime = self._regimes.get(market_id)
        regime_str = regime.regime if regime else "unknown"
        valid = []
        for b in self._beliefs.get(market_id, []):
            age = now - b.timestamp
            if self.decay_engine.is_belief_alive(b.agent_name, age, regime_str):
                valid.append(b)
        return valid

    def _record_conflict(self, market_id, beliefs, disagreement):
        self._conflicts = [c for c in self._conflicts if c.get("market_id") != market_id]
        self._conflicts.append({"market_id": market_id, "question": self._market_questions.get(market_id, ""), "disagreement": round(disagreement, 3), "beliefs": [{"agent": b.agent_name, "prob": round(b.probability, 3)} for b in beliefs], "timestamp": time.time()})
        self._conflicts = self._conflicts[-20:]
        # Narrate the conflict
        q = self._market_questions.get(market_id, "")[:40]
        print(f"  !! CONFLICT on '{q}' — agents disagree by {disagreement:.0%}:", flush=True)
        for b in beliefs:
            print(f"     {b.agent_name}: {b.probability:.0%}", flush=True)
        print("", flush=True)

    def _update_entropy(self, market_id):
        beliefs = self._get_valid_beliefs(market_id)
        if len(beliefs) < 2:
            self._entropy[market_id] = 1.0
            return
        probs = [b.probability for b in beliefs]
        bins = [0] * 10
        for p in probs:
            bins[min(9, int(p * 10))] += 1
        total = sum(bins)
        if total == 0:
            self._entropy[market_id] = 1.0
            return
        dist = [b / total for b in bins if b > 0]
        entropy = -sum(p * math.log2(p) for p in dist if p > 0)
        self._entropy[market_id] = round(entropy / math.log2(10), 3)

    def get_entropy(self, market_id):
        return self._entropy.get(market_id, 1.0)

    def get_highest_entropy_markets(self, n=5):
        return sorted(self._entropy.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_recent_surprises(self, max_age_sec=1800):
        cutoff = time.time() - max_age_sec
        return [s for s in self._surprises if s.timestamp > cutoff]

    def get_surprise_score(self, market_id):
        recent = [s for s in self._surprises if s.market_id == market_id and time.time() - s.timestamp < 3600]
        if not recent:
            return 0.0
        return min(1.0, sum(s.shift_magnitude for s in recent) / len(recent))

    def get_belief_momentum(self, market_id):
        history = self._belief_history.get(market_id, [])
        if len(history) < 3:
            return 0.0
        consensuses = [c for _, c, _ in history[-10:]]
        n = len(consensuses)
        if n < 3:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = mean(consensuses)
        numer = sum((i - x_mean) * (consensuses[i] - y_mean) for i in range(n))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        return round(numer / denom, 4) if denom != 0 else 0.0

    def is_belief_oscillating(self, market_id):
        history = self._belief_history.get(market_id, [])
        if len(history) < 5:
            return False
        recent = [c for _, c, _ in history[-10:]]
        if len(recent) < 5:
            return False
        changes = sum(1 for i in range(1, len(recent) - 1) if (recent[i] > recent[i-1] and recent[i] > recent[i+1]) or (recent[i] < recent[i-1] and recent[i] < recent[i+1]))
        return changes >= 3

    def check_blind_spots(self):
        blind_spots = []
        with self._lock:
            for market_id in self._market_prices:
                beliefs = self._get_valid_beliefs(market_id)
                if len(beliefs) < 3:
                    continue
                probs = [b.probability for b in beliefs]
                confs = [b.confidence for b in beliefs]
                if len(probs) >= 3:
                    spread = stdev(probs)
                    avg_conf = mean(confs)
                    avg_prob = mean(probs)
                    if spread < 0.05 and avg_conf > 0.6:
                        price = self._market_prices.get(market_id, 0.5)
                        edge = abs(avg_prob - price)
                        if edge > 0.05:
                            blind_spots.append({"market_id": market_id, "question": self._market_questions.get(market_id, ""), "consensus": round(avg_prob, 3), "market_price": price, "edge": round(edge * 100, 1), "warning": "All agents agree strongly against market", "num_agents": len(beliefs)})
        return blind_spots

    def compute_conviction(self, market_id):
        beliefs = self._get_valid_beliefs(market_id)
        if not beliefs:
            return 0.0
        price = self._market_prices.get(market_id)
        if price is None:
            return 0.0
        consensus, confidence, agents = self._get_consensus_unlocked(market_id)
        if consensus is None:
            return 0.0
        edge = abs(consensus - price) * 100
        edge_score = min(25, edge * 2.5)
        if len(beliefs) >= 2:
            agreement = max(0, 1 - stdev([b.probability for b in beliefs]) * 5)
        else:
            agreement = 0.3
        agreement_score = agreement * 20
        ages = [time.time() - b.timestamp for b in beliefs]
        freshness = max(0, 1 - mean(ages) / 600)
        freshness_score = freshness * 15
        domain = self._market_categories.get(market_id, "general")
        trusts = [self.trust.get_trust_weight(b.agent_name, domain) for b in beliefs]
        trust_score = min(15, (mean(trusts) - 0.4) * 12.5) if trusts else 0
        entropy = self._entropy.get(market_id, 1.0)
        entropy_score = (1 - entropy) * 10
        surprise = self.get_surprise_score(market_id)
        surprise_score = (1 - surprise) * 10
        momentum = self.get_belief_momentum(market_id)
        if consensus > price:
            momentum_score = min(5, max(0, momentum * 500))
        else:
            momentum_score = min(5, max(0, -momentum * 500))
        total = edge_score + agreement_score + freshness_score + trust_score + entropy_score + surprise_score + momentum_score
        self._conviction[market_id] = round(total, 1)
        return round(total, 1)

    def compute_edges(self, min_edge_pct=3.0):
        edges = []
        with self._lock:
            for market_id, price in self._market_prices.items():
                consensus, confidence, agents = self._get_consensus_unlocked(market_id)
                if consensus is None or confidence < 0.25:
                    continue
                edge_yes = (consensus - price) * 100
                edge_no = ((1 - consensus) - (1 - price)) * 100
                regime = self._regimes.get(market_id)
                regime_str = regime.regime if regime else "unknown"
                conviction = self.compute_conviction(market_id)
                entropy = self._entropy.get(market_id, 1.0)
                surprise = self.get_surprise_score(market_id)
                for edge_val, direction in [(edge_yes, "BUY_YES"), (edge_no, "BUY_NO")]:
                    if edge_val >= min_edge_pct:
                        quality = self._compute_edge_quality(edge_val, confidence, agents, market_id, regime)
                        edges.append(Edge(market_id=market_id, market_question=self._market_questions.get(market_id, ""), market_price=price, fair_value=consensus, edge_pct=edge_val, confidence=confidence, contributing_agents=agents, direction=direction, reasoning=f"Consensus {consensus:.1%} vs market {price:.1%}", quality_score=quality, conviction=conviction, surprise_factor=surprise, entropy=entropy, regime=regime_str))
                        break
        edges.sort(key=lambda e: e.conviction, reverse=True)
        self._edges = edges
        return edges

    def _compute_edge_quality(self, edge_pct, confidence, agents, market_id, regime):
        quality = edge_pct * confidence
        quality *= min(2.0, 0.5 + len(agents) * 0.25)
        if regime:
            if regime.regime == "stable":
                quality *= 1.2
            elif regime.regime == "volatile":
                quality *= 0.7
        volume = self._market_volumes.get(market_id, 0)
        if volume < 100:
            quality *= 0.5
        elif volume < 500:
            quality *= 0.8
        return round(quality, 2)

    def update_regime(self, market_id):
        history = self._price_history.get(market_id, [])
        if len(history) < 8:
            return
        prices = [p for _, p in history[-20:]]
        n = len(prices)
        vol = stdev(prices) if n >= 2 else 0
        x_mean = (n - 1) / 2
        y_mean = mean(prices)
        numer = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom != 0 else 0
        if vol < 0.015 and abs(slope) < 0.001:
            regime, conf = "stable", 0.8
        elif abs(slope) > 0.003:
            regime = "trending_up" if slope > 0 else "trending_down"
            conf = min(0.9, abs(slope) * 100)
        elif vol > 0.04:
            regime, conf = "volatile", min(0.9, vol * 10)
        else:
            regime, conf = "unknown", 0.3
        self._regimes[market_id] = MarketRegime(market_id=market_id, regime=regime, confidence=conf, volatility=vol, trend_strength=slope)

    def submit_correlation(self, corr):
        with self._lock:
            self._correlations = [c for c in self._correlations if not (c.market_a_id == corr.market_a_id and c.market_b_id == corr.market_b_id and c.detected_by == corr.detected_by)]
            self._correlations.append(corr)

    def get_correlated_markets(self, market_id):
        with self._lock:
            return [c for c in self._correlations if c.market_a_id == market_id or c.market_b_id == market_id]

    def submit_news_impact(self, impact):
        with self._lock:
            self._news_impacts.append(impact)
            self._news_impacts = self._news_impacts[-100:]
            for cb in self._event_callbacks.get("news_impact", []):
                try:
                    cb(impact)
                except Exception:
                    pass
        self._record_news(impact)

    def get_recent_news(self, max_age_sec=3600):
        cutoff = time.time() - max_age_sec
        with self._lock:
            return [n for n in self._news_impacts if n.timestamp > cutoff]

    def get_news_for_market(self, market_id):
        with self._lock:
            return [n for n in self._news_impacts if market_id in n.affected_markets]

    def submit_timing_signal(self, signal):
        with self._lock:
            self._timing_signals.append(signal)
            for cb in self._event_callbacks.get("timing_signal", []):
                try:
                    cb(signal)
                except Exception:
                    pass
    def get_timing_signals(self, market_id):
        now = time.time()
        with self._lock:
            return [s for s in self._timing_signals if s.market_id == market_id and s.expires_at > now]

    def update_market_price(self, market_id, price, question="", volume=0, category=""):
        with self._lock:
            self._market_prices[market_id] = price
            if question:
                self._market_questions[market_id] = question
            if volume:
                self._market_volumes[market_id] = volume
            if category:
                self._market_categories[market_id] = category
            self._price_history[market_id].append((time.time(), price))
            cutoff = time.time() - 7200
            self._price_history[market_id] = [(ts, p) for ts, p in self._price_history[market_id] if ts > cutoff]
        self.update_regime(market_id)
        for cb in self._event_callbacks.get("price_update", []):
            try:
                cb(market_id, price, question=question)
            except Exception:
                pass
        self._record_market(market_id, price, volume=volume, category=category, question=question)

    def get_market_price(self, market_id):
        with self._lock:
            return self._market_prices.get(market_id)

    def get_status(self):
        with self._lock:
            total_beliefs = sum(len(bs) for bs in self._beliefs.values())
            now = time.time()
            active_signals = len([s for s in self._timing_signals if s.expires_at > now])
            regime_counts = defaultdict(int)
            for r in self._regimes.values():
                regime_counts[r.regime] += 1
            blind_spots = self.check_blind_spots()
            surprises = self.get_recent_surprises(1800)
            high_ent = self.get_highest_entropy_markets(3)
            return {
                "markets_tracked": len(self._market_prices),
                "total_beliefs": total_beliefs,
                "active_edges": len(self._edges),
                "correlations": len(self._correlations),
                "recent_news": len([n for n in self._news_impacts if now - n.timestamp < 3600]),
                "active_timing_signals": active_signals,
                "conflicts": len(self._conflicts),
                "regimes": dict(regime_counts),
                "blind_spots": len(blind_spots),
                "surprises": len(surprises),
                "high_entropy": [(self._market_questions.get(m, m[:8]), round(e, 2)) for m, e in high_ent],
                "agent_trust_rankings": self.trust.get_rankings()[:5],
                "agent_activity": {name: round(now - ts, 1) for name, ts in self._agent_last_active.items()},
            }

    def get_belief_summary(self, market_id):
        beliefs = self._get_valid_beliefs(market_id)
        consensus, confidence, agents = self.get_consensus(market_id)
        market_price = self._market_prices.get(market_id)
        regime = self._regimes.get(market_id)
        return {
            "market_id": market_id,
            "question": self._market_questions.get(market_id, ""),
            "market_price": market_price,
            "consensus_probability": consensus,
            "consensus_confidence": confidence,
            "edge_pct": ((consensus - market_price) * 100) if consensus and market_price else None,
            "regime": regime.regime if regime else "unknown",
            "conviction": self._conviction.get(market_id, 0),
            "entropy": self._entropy.get(market_id, 1.0),
            "surprise_score": self.get_surprise_score(market_id),
            "belief_momentum": self.get_belief_momentum(market_id),
            "is_oscillating": self.is_belief_oscillating(market_id),
            "beliefs": [{"agent": b.agent_name, "probability": b.probability, "confidence": b.confidence, "reasoning": b.reasoning, "age_seconds": round(time.time() - b.timestamp, 1), "trust_weight": self.trust.get_trust_weight(b.agent_name)} for b in beliefs],
        }

    def get_conflicts(self):
        return self._conflicts