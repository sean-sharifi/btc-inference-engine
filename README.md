# BTC Options + Onchain Inference Engine

**Production-Grade Quantitative Research System**  
*Status: ✅ 100% Complete - Ready for Deployment*

---

## Overview

A sophisticated quantitative research platform combining **Deribit options analytics** and **Glassnode onchain data** to:
- Detect market regimes via Switching Kalman Filter
- Quantify dealer hedging pressure fields  
- Build mechanistic onchain indices
- Generate distributional forecasts (24h, 7d)
- Identify options ↔ onchain divergences

**37 Production Modules** | **7,500+ Lines** | **Research-Grade Quality**

---

## Quick Start

```bash
# Installation
cd btc-options-onchain-engine
pip install -e .

# Configuration
cp .env.example .env
# Edit .env with your Deribit and Glassnode API keys

# Initialize
btc-engine init-db

# Run Pipeline
btc-engine ingest-deribit
btc-engine ingest-glassnode --days 90
btc-engine build-features
btc-engine train-model
btc-engine forecast
btc-engine dashboard  # Launch at http://localhost:8501
```

---

## Features

### ✅ Data Pipeline
- **Deribit Connector**: IV surface, Greeks, OI, funding rates
- **Glassnode Connector**: 14+ top-tier onchain metrics
- **Robust Infrastructure**: Caching, retry logic, rate limiting

### ✅ Feature Engineering (Research-Grade)

**Options Surface Factorization**:
- 11 factors: Level, Skew, Curvature, Term Structure, Wing Asymmetry, PCA 1-5, Surface Shock
- RBF interpolation for smooth surfaces
- Handles sparse/missing data

**Risk-Neutral Distribution**:
- Breeden-Litzenberger approximation
- Left/right tail masses
- Jump risk proxies

**Hedging Pressure Fields**:
- Dealer gamma/vanna/charm aggregation
- Stabilization vs acceleration indices
- Spot-level pressure maps

**Onchain Mechanical Indices**:
- **Supply Elasticity**: LTH, reserves, dormancy
- **Forced Flow**: Inflow spikes, STH selling
- **Liquidity Impulse**: Stablecoin flows

**Divergence Detection**:
- Options ↔ Onchain comparison
- 4 classifications: Hedge Demand, Underpriced Risk, Breakout, Balanced
- Top 5 contributing signals

### ✅ Modeling

**Switching Kalman Filter**:
- 3 regimes: Risk On, Compression, Distress
- 6D latent state inference
- EM-style training

**Distributional Forecasting**:
- LightGBM quantile regression
- 5 quantiles (5%, 25%, 50%, 75%, 95%)
- Tail risk metrics (expected shortfall, vol-of-vol)

**Evaluation Framework**:
- CRPS (Continuous Ranked Probability Score)
- Pinball loss per quantile
- Hit rate calibration
- Walk-forward validation

### ✅ Applications

**Streamlit Dashboard** (6 tabs):
1. Regimes - Probabilities chart + latent states
2. Options Surface - IV factors over time
3. Hedging Pressure - Stabilization/acceleration maps
4. Onchain Mechanics - Index trends
5. Divergence - Real-time scoreboard
6. Forecasts - 24h/7d quantile predictions

**Newsletter Generator**:
- Weekly risk reports (Markdown)
- Automated regime summaries
- Forecast tables
- Risk memos

---

## CLI Commands

```bash
# Database
btc-engine init-db                 # Initialize schema
btc-engine init-db --reset         # Reset (WARNING: deletes data)

# Data Ingestion
btc-engine ingest-deribit          # Fetch options data
btc-engine ingest-glassnode        # Fetch onchain data (incremental)
btc-engine ingest-glassnode --days 30  # Fetch last 30 days

# Feature Engineering
btc-engine build-features          # Calculate all features
btc-engine build-features --timestamp "2024-01-15 12:00:00"

# Modeling
btc-engine train-model             # Train Switching Kalman Filter
btc-engine train-model --days 180 # Use 180 days of data
btc-engine forecast                # Generate forecasts
btc-engine evaluate                # Run backtesting
btc-engine evaluate --days 90      # 90-day backtest

# Output
btc-engine newsletter              # Generate weekly report
btc-engine newsletter --week "2024-01-15"

# Visualization
btc-engine dashboard               # Launch dashboard
btc-engine dashboard --port 8080

# System
btc-engine status                  # Show data freshness & health
```

---

## Architecture

```
Data Sources (Deribit + Glassnode)
    ↓
Ingestion Layer (Caching, Retry, Rate Limiting)
    ↓
DuckDB (14 tables: raw data, features, models, evaluations)
    ↓
Feature Engineering (20+ sophisticated features)
    ↓
State-Space Model (Regime inference + latent states)
    ↓
Distributional Forecasting (Quantile regression)
    ↓
Applications (Dashboard + Newsletter)
```

---

## Project Structure

```
btc-options-onchain-engine/
├── configs/              # YAML configuration
│   ├── data_sources.yaml
│   ├── features.yaml
│   ├── model.yaml
│   └── pipeline.yaml
├── src/btc_engine/
│   ├── connectors/      # API clients
│   ├── database/        # DuckDB schema & client
│   ├── features/        # 5 feature modules
│   ├── models/          # State-space + forecasting
│   ├── dashboard/       # Streamlit app
│   ├── newsletter/      # Report generator
│   └── cli.py          # Command-line interface
├── tests/               # Pytest suite
├── docs/                # Documentation
│   ├── methodology.md   # Technical details
│   ├── cli_reference.md # Command reference
│   ├── data_sources.md  # API documentation
│   └── VERIFICATION.md  # Component verification
├── data/                # Database (gitignored)
├── outputs/             # Generated reports
└── artifacts/           # Model artifacts
```

---

## Configuration

All parameters configurable via YAML:

- **data_sources.yaml**: API endpoints, rate limits, cache TTLs
- **features.yaml**: Feature calculation parameters (moneyness ranges, smoothing windows)
- **model.yaml**: Hyperparameters (n_regimes, state dimensions, quantiles)
- **pipeline.yaml**: Scheduling, checkpointing, parallelism

---

## Requirements

- **Python**: 3.11+
- **Database**: DuckDB (included)
- **API Keys**:
  - Deribit (public endpoints, key optional but recommended)
  - Glassnode (top-tier subscription required for all metrics)
- **Disk**: ~500MB for data

---

## Documentation

- [Methodology](docs/methodology.md) - Formulas and theory
- [CLI Reference](docs/cli_reference.md) - Command guide
- [Data Sources](docs/data_sources.md) - API specifications
- [Verification Report](docs/VERIFICATION.md) - Component validation

---

## Development

```bash
# Testing
pytest tests/ -v --cov=btc_engine

# Code Quality
black src/ tests/         # Format
ruff check src/ tests/    # Lint

# Convenience
make install    # Install dependencies
make test       # Run tests
make format     # Format code
make dashboard  # Launch dashboard
```

---

## What Makes This Special

1. **Research-Grade Features**: Not basic metrics - sophisticated quant techniques (Breeden-Litzenberger, gamma exposure fields, mechanistic onchain composites)

2. **Unique Divergence Detection**: First-principles comparison of options pricing vs onchain fundamentals

3. **Production Architecture**: Modular, tested, documented, configurable

4. **Complete Pipeline**: From raw data to final reports, all automated

5. **Interpretable**: Every model decision explained, no black boxes

---

## Status

✅ **100% Complete** - All 13 development phases delivered:
- Data ingestion ✓
- Feature engineering ✓  
- State-space modeling ✓
- Distributional forecasting ✓
- Evaluation framework ✓
- Dashboard ✓
- CLI ✓
- Documentation ✓

**Ready for production deployment and live trading research.**

---

## License

Proprietary - Quantitative Research System

---

**Built**: January 2026  
**Tech Stack**: Python 3.11, DuckDB, LightGBM, Streamlit, Plotly  
**Quality**: Production-grade with comprehensive testing
