# BTC Options + Onchain Engine - Methodology

## Overview

This document explains the methodology behind each component of the BTC Options + Onchain Inference Engine.

## 1. Options Surface Factorization

### Approach

We decompose the implied volatility surface into interpretable factors and orthogonal PCA components.

**Handcrafted Factors**:

1. **Level**: Average ATM volatility across expiries, represents base volatility level
2. **Skew**: Put-call IV difference, captures downside demand (negative skew = crash hedging)
3. **Curvature**: Smile intensity, measures convexity/kurtosis expectations
4. **Term Structure**: Front-back vol spread, indicates volatility term premium
5. **Wing Asymmetry**: Asymmetry between far OTM puts and calls
6. **Surface Shock**: Time derivative of surface, captures rapid regime shifts

**PCA Factors**: Capture remaining orthogonal variance not explained by handcrafted factors.

### Implementation

- Log-moneyness standardization for strike comparability
- RBF interpolation for smooth surface with controlled smoothing
- Robust to missing data and sparse surfaces

## 2. Risk-Neutral Distribution

### Breeden-Litzenberger Approximation

The risk-neutral density is approximated via:

```
RN_density(K) ≈ e^(rT) * d²C/dK²
```

Where C is the call option price and K is the strike.

**Metrics Derived**:

- **Tail Mass**: Probability mass in far OTM regions  
- **Jump Risk**: Far-wing IV steepening rate
- **Skew Elasticity**: Sensitivity of skew to spot moves (requires time series)

### Limitations

- Requires sufficient strike density
- Numerical differentiation introduces noise
- We use robust finite differences with forward/backward/central schemes

## 3. Hedging Pressure Field

### Theory

Market makers (dealers) delta-hedge their sold options by buying/selling underlying. Their aggregate gamma position creates reflexive flows:

- **Positive Dealer Gamma** (client short gamma): Dealers buy dips, sell rallies → stabilizing
- **Negative Dealer Gamma** (client long gamma): Dealers sell dips, buy rallies → destabilizing

### Calculation

1. Aggregate dealer gamma = -Σ(client_gamma × OI) across all strikes/expiries
2. Map to spot grid using Gaussian kernel around strike
3. Calculate hedge flow per 1% move = -gamma × spot × 0.01

**Indices**:

- **Stabilization Index**: Average gamma above spot (positive = mean-reverting pressure)
- **Acceleration Index**: Average negative gamma (positive = trend-amplifying pressure)
- **Key Levels**: Zero-crossings, max exposure strikes

## 4. Onchain Mechanical Indices

### Supply Elasticity Index

Measures how responsive BTC supply is to price changes. Low elasticity → supply constrained.

**Components**:
- LTH supply increasing → holders not selling → low elasticity
- Exchange reserves increasing → more supply available → high elasticity  
- Dormancy high → old coins inactive → low elasticity

**Formula**: Weighted combination of z-scored components, exponentially smoothed.

### Forced Flow Index

Detects non-discretionary selling pressure (distressed/forced sellers, not patient profit-takers).

**Signals**:
- Exchange inflow spikes (>2σ) → coins moving to exchanges to sell
- STH supply decreasing → short-term holders capitulating
- Netflow acceleration → sudden increase in outflows

**Formula**: Weighted spike detection + z-scored accelerations.

### Liquidity Impulse Index

Tracks stablecoin liquidity available to absorb BTC selling or fuel buying.

**Components**:
- Stablecoin supply changes → more stables = more dry powder
- Stablecoin exchange inflows → liquidity arriving at exchanges

**Formula**: Z-scored changes, exponentially smoothed.

## 5. Divergence Detection

### Concept

Options pricing can diverge from onchain fundamentals in two ways:

1. **Options overpricing tail risk** relative to onchain pressure → hedging demand, false fear
2. **Options underpricing tail risk** relative to onchain pressure → complacency, underpriced crash

### Algorithm

1. **Options Tail Signal**: Aggregate left tail mass, negative skew, jump risk, curvature
2. **Onchain Pressure Signal**: Aggregate forced flow, low elasticity, negative liquidity

3. **Divergence Score** = Tail Signal - Pressure Signal

**Classifications**:
- Score > +1.5: "Hedge Demand / False Fear"
- Score < -1.5: "Underpriced Crash Risk"  
- High calls + low elasticity: "Supply Inelastic Breakout Risk"
- Otherwise: "Balanced / Consistent"

4. **Explainability**: Rank top 5 contributing signals by |magnitude|

## 6. State-Space Model (Planned)

### Switching Kalman Filter

**Latent State Vector** (6-dim):
- Risk appetite
- Leverage stress
- Dealer stabilization regime
- Tail demand asymmetry
- Inventory imbalance
- Liquidity regime

**Observations** (~15-dim):
- Options factors, pressure indices, onchain indices, divergence

**Model**:
- 3 discrete regimes (Risk-On, Compression, Distress)
- Linear-Gaussian emissions per regime
- EM algorithm for MaximumLikelihood estimation
- Forward pass for filtered state estimates

**Outputs**:
- Regime probabilities P(regime_t | data_{1:t})
- Latent state estimates
- Regime transition dynamics

## 7. Distributional Forecasting (Planned)

### Regime-Conditioned Quantile Regression

For horizons h ∈ {24h, 7d}:

1. Features: Current state vector, regime history, macro proxies
2. **Separate quantile regressors** per regime (LightGBM quantile loss)
3. Walk-forward validation with strict temporal splits

**Outputs**:
- Quantiles: Q_0.05, Q_0.25, Q_0.50, Q_0.75, Q_0.95
- Tail metrics: Expected shortfall, tail probability mass
- Vol-of-vol: Std of vol distribution

### Evaluation

- **CRPS**: Continuous Ranked Probability Score (proper scoring rule)
- **Pinball Loss**: Per-quantile calibration
- **Hit Rates**: Empirical coverage vs theoretical
- **Calibration Plots**: Reliability diagrams

## References

- Breeden & Litzenberger (1978) - Prices of State-Contingent Claims Implicit in Option Prices
- SqueezeMetrics GEX methodology
- Glassnode onchain metrics documentation
- Hamilton (1989) - State-space models with regime switching

---

*Last updated: 2024-01-21*
