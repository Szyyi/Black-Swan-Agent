<p align="center">
  <h1 align="center">Black Swan Agent</h1>
  <img width="1920" height="1080" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/dfaf8644-c8f7-4e5c-87a9-47c789df58ca" />
  <p align="center"><strong>Multi-Agent AI Trading Swarm for Prediction Markets</strong></p>
  <p align="center">
    12 specialised AI agents · Bayesian belief engine · Adversarial red team · Event-driven intelligence · Metalearning · Cross-bookmaker arbitrage
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/agents-12-green?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-Claude%20Sonnet%204-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/architecture-event--driven-red?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square" />
</p>

---

## What is Black Swan Agent?

Black Swan Agent is an autonomous trading system that deploys a swarm of 12 specialised AI agents to analyse prediction markets, detect mispricings, and execute trades. Unlike single-model trading bots, the swarm architecture produces diverse analytical perspectives that are synthesised through a novel Bayesian world model — creating emergent intelligence that no individual agent could achieve alone.

The system is named after its **Black Swan philosophy**: the most dangerous moment in any trading system is when *everyone agrees*. Unanimous confidence is often a sign of shared bias, not of truth. Black Swan Agent treats consensus as a hypothesis to be attacked, not a conclusion to be trusted. A dedicated **Adversarial Agent** acts as an internal red team, stress-testing every high-conviction trade through pre-mortems and failure scenarios before capital is committed.

### Key Differentiators

- **Adversarial red team** — a dedicated agent whose only job is to break the swarm's consensus and find catastrophic failure modes before they happen
- **Event-driven architecture** — agents react instantly to price spikes, breaking news, and surprise belief shifts instead of waiting on fixed timers
- **Multi-armed bandit attention allocation** — UCB1 algorithm dynamically directs agent cycles to the markets where they'll produce the most value
- **Metalearning over agent combinations** — learns which agent ensembles work best per market category, regime, and time of day
- **Type-specific information decay** — breaking news decays in 30 minutes, sports form in 6 hours, structural analysis in 24 hours; regime-adaptive
- **Belief propagation network** — when one market shifts, correlated markets update automatically through causal links
- **Natural language trade theses** — every trade generates a complete human-readable explanation: which agents contributed, what evidence they used, what the adversarial pre-mortem found, and what the kill conditions are
- **Cross-bookmaker arbitrage** — compares odds across 50+ bookmakers to find mathematically guaranteed edges
- **Real sports data integration** — team form, league standings, and head-to-head records feed directly into probability estimation
- **Conviction scoring** — 7-factor composite metric that goes beyond simple edge detection to measure how *sure* the swarm is
- **Self-calibrating confidence** — tracks prediction accuracy at each confidence level and adjusts future outputs accordingly

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    META COORDINATOR                          │
│   Composite scoring · Kelly sizing · Risk-gated execution    │
└────────────────────────┬────────────────────────────────────┘
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  12 AGENTS   │  │ INTELLIGENCE │  │     RISK     │
│              │  │    LAYER     │  │   MANAGER    │
│ news_scout   │  │              │  │              │
│ prob_est     │  │  Event Bus   │  │ Kelly size   │
│ contrarian   │◄─┤  Attention   │──│ Kill switch  │
│ adversarial  │  │  Engine      │  │ Drawdown     │
│ correlation  │  │  Metalearn   │  │ Correlation  │
│ momentum     │  │  Decay       │  │ guard        │
│ sports_intel │  │  Thesis Gen  │  │ Cool-off     │
│ odds_arb     │  └──────┬───────┘  └──────────────┘
│ web_research │         │
│ social_sig   │         ▼
│ edge_stack   │  ┌──────────────┐
│ mkt_scanner  │──┤ WORLD MODEL  │
└──────────────┘  │              │
                  │ Bayesian     │
                  │ consensus    │
                  │ Belief prop  │
                  │ Entropy      │
                  │ Surprise     │
                  │ Conviction   │
                  │ Calibration  │
                  │ Black Swan   │
                  │ sentinel     │
                  └──────────────┘
```

The architecture is built around a **shared world model** as the swarm's collective brain. Agents do not communicate directly with each other — they submit beliefs, news impacts, correlations, and timing signals to the world model, which fuses them into a coherent picture using Bayesian consensus. The **intelligence layer** sits between the agents and the world model, providing event-driven scheduling, attention allocation, performance feedback, and trade narration.

---

## The Agent Swarm

| Agent | Default Interval | Function |
|-------|------------------|----------|
| **Probability Estimator** | 5 min | Ensemble forecasting with 5 analytical perspectives and chain-of-thought reasoning |
| **News Scout** | 2 min | RSS ingestion with LLM-powered impact analysis and narrative arc tracking |
| **Adversarial** | 4 min | Red team — stress-tests consensus, runs pre-mortems, finds catastrophic failure modes |
| **Correlation Detective** | 10 min | Discovers causal links between markets using structured reasoning |
| **Contrarian** | 15 min | Hunts for crowd errors using an 8-bias cognitive debiasing framework |
| **Sports Intelligence** | 5 min | Fetches real team form, standings, and H2H data from sports APIs |
| **Odds Arbitrage** | 10 min | Cross-bookmaker comparison across 50+ bookmakers for value and arbitrage |
| **Web Researcher** | 10 min | Deep analytical research on high-priority markets flagged by other agents |
| **Momentum Detector** | 3 min | Linear regression trend detection with projected price trajectories |
| **Edge Stacker** | 2 min | Finds implied edges through correlation chains across markets |
| **Social Signals** | 5 min | Infers market sentiment from news coverage volume and tone |
| **Market Scanner** | 30 sec | Price anomaly detection, round-number anchoring, volume spikes |

> **Note on intervals:** The values above are *base intervals*. The Attention Allocation Engine dynamically shortens intervals for agents producing high-value signals and lengthens them for agents producing noise. The Event Bus can also wake any agent immediately when a triggering event occurs, so an agent with a 5-minute base interval may actually run every 30 seconds during a breaking news event.

---

## The Intelligence Layer

This is what separates Black Swan Agent from a simple ensemble of LLM calls. Six interlocking systems sit between the agents and the world model, transforming raw analyses into emergent collective intelligence.

### 1. Adversarial Agent — The Internal Red Team

Most trading systems try to *find* edges. The adversarial agent tries to *destroy* them. For every high-conviction trade the swarm wants to make, it runs a structured attack:

- **Blind spot attack** — when all agents agree strongly against the market, asks "what does the market know that our agents don't?" and constructs a specific failure scenario
- **Pre-mortem analysis** — imagines the trade has already lost, identifies which agent's reasoning was the weakest link, what data was missing, what timing went wrong
- **Correlated failure detection** — checks if multiple positions share a common failure mode (the most dangerous portfolio configuration)
- **Regime blindness detection** — flags when agents are applying stable-market logic in volatile conditions

If the adversarial agent finds a plausible failure scenario, it submits a counter-belief weighted by `plausibility × impact_if_true`. This automatically incorporates tail risk into the consensus calculation. Trade execution is gated by an adversarial multiplier: `dangerous` ratings reduce the composite score by 60%, `abort` ratings block the trade entirely.

The adversarial agent is the philosophical core of Black Swan: the swarm must prove its consensus survives a structured attack before capital is committed.

### 2. Attention Allocation Engine — Multi-Armed Bandit Scheduling

Instead of every agent scanning every market on a fixed timer, the Attention Allocation Engine uses **UCB1 (Upper Confidence Bound)** to dynamically direct agent cycles to the markets that need them most. Each market is scored on six factors:

- **Entropy** — agent disagreement (high entropy = needs more analysis)
- **Surprise** — recent shocks (recent belief shifts demand investigation)
- **Urgency** — time to expiry (approaching deadlines get priority)
- **Staleness** — time since last analysis (exploration vs exploitation)
- **Edge magnitude** — bigger edges deserve more scrutiny
- **Volatility** — volatile markets change fast and need frequent updates

Each agent has a **personalised weighting** over these factors. Sports Intelligence prioritises staleness and urgency. Momentum Detector prioritises volatility and surprise. The Adversarial Agent prioritises entropy and edge magnitude — exactly where consensus is most likely to be wrong.

The engine also adjusts agent cycle intervals based on recent value: agents producing useful signals run faster, agents producing noise slow down to conserve API credits. This is essentially a contextual bandit over agent allocation, optimising for value-per-API-call.

### 3. Event Bus — Event-Driven Triggers

Fixed timers are wasteful. The Event Bus lets the system react instantly to critical events instead of waiting for the next scheduled cycle. When the world model detects a triggering event, the bus routes it to the relevant agents through `asyncio.Event` wake signals.

Triggering events:
- **Price spike** (>5% move) → wakes momentum, probability estimator, adversarial, odds arbitrage
- **Breaking news** (urgency > 0.7) → wakes news scout, probability estimator, web researcher, social signals
- **Surprise belief shift** (>10% from consensus) → wakes probability estimator, contrarian, adversarial
- **New correlation discovered** → wakes edge stacker, adversarial
- **Market approaching expiry** (<1 hour) → wakes probability estimator, odds arbitrage
- **Adversarial alert** (dangerous risk rating) → wakes web researcher, contrarian
- **Regime change** → wakes momentum detector, adversarial

The bus includes per-event-per-market cooldowns to prevent event storms, and severity scoring so agents handle the most urgent events first. This dramatically improves response time: instead of a 5-minute lag between breaking news and analysis, agents react within seconds.

### 4. Metalearning System — Learning Which Combinations Work

Tracks performance across multiple dimensions:
- **Agent combinations** — which ensembles produce the best trades (e.g., `probability_estimator + sports_intelligence` on football)
- **Market categories** — which agents excel at which domains (politics, sports, entertainment)
- **Market regimes** — which combinations work in stable vs volatile conditions
- **Time of day** — performance patterns across 4-hour buckets

Outputs feed back into the composite scoring formula as a `meta_weight` multiplier. Proven combinations get amplified (up to 1.3x), failed combinations get dampened (down to 0.7x). After 10+ trades on a combo, the system can also flag "should_skip" verdicts that prevent wasting API credits on consistently losing configurations.

The system persists to disk so learning survives restarts. This is essentially a contextual bandit over agent ensembles — the swarm gets measurably smarter over time, not just statically clever.

### 5. Information Decay Curves — Type-Specific Half-Lives

Different information types should age at different rates. A 2-hour-old breaking news headline is essentially worthless. A 2-hour-old structural political analysis is still completely valid. The Decay Engine assigns each agent a specific **exponential decay half-life**:

| Agent | Half-life | Reasoning |
|-------|-----------|-----------|
| Market Scanner | 5 min | Price snapshots stale almost immediately |
| Momentum Detector | 10 min | Momentum signals are fleeting |
| News Scout | 30 min | Breaking news priced in fast |
| Odds Arbitrage | 30 min | Bookmakers update constantly |
| Edge Stacker | 15 min | Derived signals decay fast |
| Social Signals | 40 min | Sentiment shifts moderately |
| Probability Estimator | 1 hour | Solid analytical baseline |
| Adversarial | 90 min | Risk warnings stay relevant |
| Contrarian | 2 hours | Bias corrections are medium-term |
| Web Researcher | 4 hours | Deep research stays valid |
| Sports Intelligence | 6 hours | Form data valid until next match |
| Correlation Detective | 8 hours | Structural insights last long |

Beliefs don't simply expire — they **fade** smoothly. Their weight in consensus calculations decreases on an exponential curve. This is then **regime-adjusted**: in volatile markets, all decay rates accelerate by 30-50%; in stable markets, they slow by 30%; in trending markets, momentum signals last twice as long while contrarian views decay 40% faster.

This replaces the original flat-TTL system entirely and dramatically improves signal quality.

### 6. Natural Language Trade Thesis

Every trade generates a complete human-readable thesis covering:

- **Action** — direction, entry price, fair value, edge, size, Kelly fraction
- **Agent intelligence** — which agents contributed, their probabilities, confidences, and reasoning
- **Key evidence** — the specific facts driving the trade
- **News context** — relevant breaking news with urgency scores
- **Correlated markets** — related positions and correlation strengths
- **Risk assessment** — adversarial risk rating, pre-mortem failure scenario, kill conditions, market regime
- **Scoring breakdown** — composite score and confidence multiplier

Every thesis is stored as structured data for the dashboard, with both compact terminal summaries and full text reports. This makes the system completely **interpretable**: you can audit every trade decision and understand exactly why it happened, which agents drove it, and what could go wrong.

---

## Novel Features (World Model Layer)

### Belief Propagation Network
When an agent updates its belief about a market, the shift ripples through all correlated markets proportionally. If "Party A wins election" shifts +10%, and "Policy X enacted" is positively correlated at 0.7 strength, a dampened +3.5% implied shift automatically propagates. This creates second-order intelligence that no individual agent produces.

### Information Entropy Tracker
Measures Shannon entropy across agent beliefs per market. High-entropy markets — where agents disagree most — are automatically prioritised by the Attention Allocation Engine because that's where the most alpha potential lives. The system directs its attention where uncertainty is highest.

### Conviction Meter
A 0-100 composite score combining seven independent factors:
1. Edge magnitude (0-25 pts)
2. Agent agreement (0-20 pts)
3. Belief freshness (0-15 pts)
4. Agent trust level (0-15 pts)
5. Low entropy bonus (0-10 pts)
6. Surprise history (0-10 pts)
7. Belief momentum alignment (0-5 pts)

Edges are ranked by conviction rather than raw size, ensuring the system prioritises trades where the swarm has genuine multi-dimensional confidence.

### Black Swan Sentinel
When all agents agree strongly against the market price, the system flags a "blind spot" rather than treating it as a high-conviction trade. Unanimous agreement with high confidence is often a sign of shared bias — the very scenario that produces black swan losses. The sentinel forces additional scrutiny and triggers the Adversarial Agent to specifically attack the consensus.

### Self-Calibrating Confidence
Tracks a calibration curve: do 70% confidence predictions actually resolve correctly 70% of the time? Over time, the system learns its own biases and adjusts future confidence outputs. If the swarm is consistently overconfident at the 80% level, future 80% confidences get scaled down automatically.

### Temporal Belief Dynamics
Tracks how consensus evolves over time per market. Detects:
- **Belief momentum** — consensus trending directionally (strengthening conviction)
- **Oscillation** — consensus flip-flopping (weakening conviction, signal of confusion)
- **Surprise events** — sudden shifts that contradict existing consensus (potential opportunities)

---

## The Composite Scoring Formula

Every potential trade is scored through a multi-factor formula that aggregates the entire intelligence stack into a single number. Trades only execute when the composite score exceeds a threshold:

```
composite = edge_pct
          × confidence
          × urgency           (from timing signals)
          × freshness         (from decay-weighted belief age)
          × agreement_bonus   (more agents = more confidence)
          × (1 - corr_penalty)(reduce if correlated to existing positions)
          × confidence_mult   (recent win/loss streak adjustment)
          × meta_weight       (metalearning: proven combos amplified)
          × adversarial_mult  (adversarial gate: risky trades dampened)
```

Each factor encodes a different layer of the system's intelligence. The metalearning weight makes the system improve over time. The adversarial multiplier acts as a hard gate that can completely block trades flagged as `abort`. The whole formula is auditable through the trade thesis output.

---

## Quick Start

### Prerequisites
- Python 3.12+
- API keys (all free tiers available):
  - [Anthropic](https://console.anthropic.com) — Claude API for LLM reasoning
  - [Smarkets](https://smarkets.com) — Prediction market account
  - [The Odds API](https://the-odds-api.com) — Cross-bookmaker odds (500 free req/month)
  - [football-data.org](https://www.football-data.org/client/register) — Football statistics

### Installation

```bash
git clone https://github.com/Szyyi/black-swan-agent.git
cd black-swan-agent
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

### Configuration

Edit `config.yaml` with your API keys:

```yaml
mode: paper  # 'paper' for simulation, 'live' for real trading
platform: smarkets

smarkets:
  username: "your_email@example.com"
  password: "your_password"

anthropic:
  api_key: "sk-ant-..."
  model: "claude-sonnet-4-20250514"

data_apis:
  football_data_key: "your_key_here"
  odds_api_key: "your_key_here"
```

### Run

```bash
python -m agent.main --swarm --mode paper
```

The system will:
1. Login to Smarkets and load active markets
2. Wire up the event bus, attention engine, metalearning, and decay engine
3. Launch all 12 agents on staggered schedules
4. Begin analysing markets, building beliefs, and detecting edges
5. The adversarial agent stress-tests every high-conviction trade
6. Print status updates every 20 seconds with event counts and attention highlights
7. Execute paper trades with full natural language theses when conviction thresholds are met

---

## How It Works

```
1. DATA COLLECTION
   Smarkets markets · RSS news · Sports APIs · 50+ bookmaker odds
                              │
2. EVENT-DRIVEN ANALYSIS      ▼
   Agents react to events (price spikes, news, surprises) instantly
   Attention engine routes cycles to highest-value markets
   Each agent analyses from its unique perspective
   Beliefs submitted to shared world model
                              │
3. BAYESIAN CONSENSUS         ▼
   Beliefs weighted by: confidence × trust × type-specific decay
   Propagated through correlation network
   Calibrated against historical accuracy
                              │
4. ADVERSARIAL STRESS-TEST    ▼
   Red team attacks high-conviction edges
   Pre-mortems identify failure scenarios
   Counter-beliefs encode tail risk
                              │
5. EDGE DETECTION             ▼
   Consensus vs market price = edge
   Scored by 7-factor conviction meter
   Filtered by Black Swan Sentinel
                              │
6. COMPOSITE SCORING          ▼
   9-factor composite formula including metalearning weight
   and adversarial gate
                              │
7. RISK-MANAGED EXECUTION     ▼
   Kelly criterion sizing · Position limits
   Correlation guards · Drawdown protection
   Adaptive confidence multiplier
                              │
8. THESIS GENERATION          ▼
   Every trade gets a full natural language explanation
   Stored for audit, learning, and dashboard display
                              │
9. METALEARNING FEEDBACK      ▼
   Trade outcomes feed back into combo weights
   System gets measurably smarter over time
```

---

## Roadmap

### Phase 1 — Foundation ✅
- [x] 11-agent swarm architecture
- [x] Bayesian world model with belief propagation
- [x] Cross-bookmaker odds comparison
- [x] Real sports data integration
- [x] Paper trading execution
- [x] Risk management with Kelly sizing

### Phase 2 — Intelligence Enhancement ✅
- [x] Adversarial agent that stress-tests consensus
- [x] Attention allocation engine (multi-armed bandit for agent scheduling)
- [x] Event-driven architecture (react to news instantly, not on fixed intervals)
- [x] Metalearning system (learn which agent combinations work best per market type)
- [x] Information decay curves (type-specific, regime-adaptive belief aging)
- [x] Natural language trade thesis generation

### Phase 3 — Multi-Platform
- [ ] Web dashboard with real-time visualisation of agents, beliefs, edges, and theses
- [ ] Live Smarkets order execution via streaming API
- [ ] Stock broker integration (Trading 212 / Interactive Brokers)
- [ ] Cross-platform arbitrage (prediction markets vs stock markets)
- [ ] Defence contractor correlation trading (geopolitical events → defence stocks)
- [ ] Synthetic market creation from correlated positions

### Phase 4 — Production
- [ ] Backtesting framework with historical replay
- [ ] Reinforcement learning position sizing
- [ ] Web search integration for real-time research
- [ ] Twitter/X and Reddit sentiment monitoring
- [ ] Cloud deployment with auto-restart and monitoring
- [ ] Regime-adaptive strategy selection (mean-reversion vs momentum vs market-making)

---

## Project Structure

```
black-swan-agent/
├── config.example.yaml          # Template configuration
├── requirements.txt             # Python dependencies
├── agent/
│   ├── main.py                  # Entry point
│   ├── config.py                # Configuration loader
│   ├── models.py                # Data models (Market, Order, Signal)
│   ├── data/
│   │   ├── feeds.py             # RSS news ingester
│   │   ├── smarkets_client.py   # Smarkets exchange API
│   │   ├── sports_api.py        # Sports data (TheSportsDB + football-data.org)
│   │   └── odds_api.py          # Cross-bookmaker odds comparison
│   ├── swarm/
│   │   ├── agents.py            # All 12 agent implementations
│   │   ├── coordinator.py       # Meta coordinator and decision engine
│   │   ├── world_model.py       # Bayesian world model
│   │   ├── adversarial.py       # Adversarial red team agent
│   │   ├── attention.py         # Multi-armed bandit attention allocation
│   │   ├── event_bus.py         # Event-driven trigger system
│   │   ├── metalearning.py      # Agent combination performance tracking
│   │   ├── decay.py             # Type-specific information decay curves
│   │   └── thesis.py            # Natural language trade thesis generation
│   ├── execution/
│   │   └── engine.py            # Paper and live trade execution
│   └── risk/
│       └── manager.py           # Position sizing and risk controls
```

---

## Contributing

Black Swan Agent is in active development. Contributions are welcome in:

- **New agent implementations** — domain-specific agents for crypto, commodities, or geopolitics
- **Data source integrations** — social media APIs, alternative data feeds, web search
- **Backtesting infrastructure** — historical data collection and replay engines
- **Dashboard development** — web UI for real-time swarm monitoring
- **Stock broker adapters** — Trading 212, Interactive Brokers, IG integrations

---

## Disclaimer

This software is for educational and research purposes. Trading prediction markets involves financial risk. Paper trading mode is enabled by default. The author accepts no responsibility for financial losses incurred through use of this software. Always understand the risks before trading with real capital.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---