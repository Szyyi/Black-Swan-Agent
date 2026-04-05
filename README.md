<p align="center">
  <h1 align="center">🦢 Black Swan Agent</h1>
  <p align="center"><strong>Multi-Agent AI Trading Swarm for Prediction Markets</strong></p>
  <p align="center">
    11 specialised AI agents · Bayesian belief engine · Cross-bookmaker arbitrage · Real-time sports intelligence
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/agents-11-green?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-Claude%20Sonnet%204-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square" />
</p>

---

## What is Black Swan Agent?

Black Swan Agent is an autonomous trading system that deploys a swarm of 11 specialised AI agents to analyse prediction markets, detect mispricings, and execute trades. Unlike single-model trading bots, the swarm architecture produces diverse analytical perspectives that are synthesised through a novel Bayesian world model — creating emergent intelligence that no individual agent could achieve alone.

The system is named after its **Black Swan Sentinel** — a novel feature that detects when all agents agree too strongly, flagging potential blind spots where unanimous confidence may signal shared bias rather than truth.

### Key Differentiators

- **Multi-agent ensemble reasoning** — 5 analytical perspectives (base rate, evidence weighing, contrarian, scenario planning, time decay) produce calibrated probability estimates
- **Belief propagation network** — when one market shifts, correlated markets update automatically through causal links
- **Cross-bookmaker arbitrage** — compares odds across 50+ bookmakers to find mathematically guaranteed edges
- **Real sports data integration** — team form, league standings, and head-to-head records feed directly into probability estimation
- **Conviction scoring** — 7-factor composite metric that goes beyond simple edge detection to measure how *sure* the swarm is
- **Self-calibrating confidence** — tracks prediction accuracy at each confidence level and adjusts future outputs accordingly

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    META COORDINATOR                          │
│  Adaptive decision intervals · Kelly sizing · Risk guards   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  11 AGENTS  │  │ WORLD MODEL │  │    RISK     │
│             │  │             │  │   MANAGER   │
│ news_scout  │──│ Bayesian    │──│ Kelly size  │
│ prob_est    │  │ consensus   │  │ Kill switch │
│ contrarian  │  │ Belief prop │  │ Drawdown    │
│ correlation │  │ Entropy     │  │ Correlation │
│ momentum    │  │ Surprise    │  │ guard       │
│ sports_intel│  │ Conviction  │  │ Cool-off    │
│ odds_arb    │  │ Calibration │  └─────────────┘
│ web_research│  │ Black Swan  │
│ social_sig  │  │ sentinel    │
│ edge_stack  │  └─────────────┘
│ mkt_scanner │
└─────────────┘
```

---

## The Agent Swarm

| Agent | Interval | Function |
|-------|----------|----------|
| **Probability Estimator** | 5 min | Ensemble forecasting with 5 analytical perspectives and chain-of-thought reasoning |
| **News Scout** | 2 min | RSS ingestion with LLM-powered impact analysis and narrative arc tracking |
| **Correlation Detective** | 10 min | Discovers causal links between markets using structured reasoning |
| **Contrarian** | 15 min | Hunts for crowd errors using an 8-bias cognitive debiasing framework |
| **Sports Intelligence** | 5 min | Fetches real team form, standings, and H2H data from sports APIs |
| **Odds Arbitrage** | 10 min | Cross-bookmaker comparison across 50+ bookmakers for value and arbitrage |
| **Web Researcher** | 10 min | Deep analytical research on high-priority markets flagged by other agents |
| **Momentum Detector** | 3 min | Linear regression trend detection with projected price trajectories |
| **Edge Stacker** | 2 min | Finds implied edges through correlation chains across markets |
| **Social Signals** | 5 min | Infers market sentiment from news coverage volume and tone |
| **Market Scanner** | 30 sec | Price anomaly detection, round-number anchoring, volume spikes |

---

## Novel Features

### Belief Propagation Network
When an agent updates its belief about a market, the shift ripples through all correlated markets proportionally. If "Party A wins election" shifts +10%, and "Policy X enacted" is positively correlated at 0.7 strength, a dampened +3.5% implied shift automatically propagates. This creates second-order intelligence that no individual agent produces.

### Information Entropy Tracker
Measures Shannon entropy across agent beliefs per market. High-entropy markets — where agents disagree most — are automatically prioritised for analysis because that's where the most alpha potential lives. The system directs its attention where uncertainty is highest.

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
When all agents agree strongly against the market price, the system flags a "blind spot" rather than treating it as a high-conviction trade. Unanimous agreement with high confidence is often a sign of shared bias — the very scenario that produces black swan losses. The sentinel forces additional scrutiny on these positions.

### Self-Calibrating Confidence
Tracks a calibration curve: do 70% confidence predictions actually resolve correctly 70% of the time? Over time, the system learns its own biases and adjusts future confidence outputs. If the swarm is consistently overconfident at the 80% level, future 80% confidences get scaled down automatically.

### Temporal Belief Dynamics
Tracks how consensus evolves over time per market. Detects:
- **Belief momentum** — consensus trending directionally (strengthening conviction)
- **Oscillation** — consensus flip-flopping (weakening conviction, signal of confusion)
- **Surprise events** — sudden shifts that contradict existing consensus (potential opportunities)

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
git clone https://github.com/yourusername/black-swan-agent.git
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
2. Launch all 11 agents on staggered schedules
3. Begin analysing markets, building beliefs, and detecting edges
4. Print status updates every 20 seconds
5. Execute paper trades when conviction thresholds are met

---

## How It Works

```
1. DATA COLLECTION
   Smarkets markets · RSS news · Sports APIs · 50+ bookmaker odds
                              │
2. MULTI-AGENT ANALYSIS       ▼
   Each agent analyses from its unique perspective
   and submits beliefs to the shared world model
                              │
3. BAYESIAN CONSENSUS         ▼
   Beliefs weighted by: confidence × trust × freshness
   Propagated through correlation network
   Calibrated against historical accuracy
                              │
4. EDGE DETECTION             ▼
   Consensus vs market price = edge
   Scored by 7-factor conviction meter
   Filtered by Black Swan Sentinel
                              │
5. RISK-MANAGED EXECUTION     ▼
   Kelly criterion sizing · Position limits
   Correlation guards · Drawdown protection
   Adaptive confidence multiplier
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

### Phase 2 — Intelligence Enhancement
- [ ] Adversarial agent that stress-tests consensus
- [ ] Attention allocation engine (multi-armed bandit for agent scheduling)
- [ ] Event-driven architecture (react to news instantly, not on fixed intervals)
- [ ] Regime-adaptive strategy selection (different strategies for trending vs stable markets)
- [ ] Metalearning system (learn which agent combinations work best per market type)

### Phase 3 — Multi-Platform
- [ ] Live Smarkets order execution via streaming API
- [ ] Stock broker integration (Interactive Brokers / IG)
- [ ] Cross-platform arbitrage (prediction markets vs stock markets)
- [ ] Synthetic market creation from correlated positions

### Phase 4 — Production
- [ ] Web dashboard with real-time visualisation
- [ ] Backtesting framework with historical replay
- [ ] Reinforcement learning position sizing
- [ ] Natural language trade thesis generation
- [ ] Cloud deployment with auto-restart and monitoring

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
│   │   ├── agents.py            # All 11 agent implementations
│   │   ├── coordinator.py       # Meta coordinator and decision engine
│   │   └── world_model.py       # Bayesian world model
│   ├── execution/
│   │   └── engine.py            # Paper and live trade execution
│   └── risk/
│       └── manager.py           # Position sizing and risk controls
```

---

## Contributing

Black Swan Agent is in active development. Contributions are welcome in:

- **New agent implementations** — domain-specific agents for crypto, commodities, or geopolitics
- **Data source integrations** — social media APIs, alternative data feeds
- **Backtesting infrastructure** — historical data collection and replay engines
- **Visualisation** — web dashboard for real-time swarm monitoring

---

## Disclaimer

This software is for educational and research purposes. Trading prediction markets involves financial risk. Paper trading mode is enabled by default. The authors accept no responsibility for financial losses incurred through use of this software. Always understand the risks before trading with real capital.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---
