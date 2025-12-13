# Stock Volatility Prediction System - Technical Overview

## Table of Contents
1. [Models Architecture](#1-models-architecture)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Data Pipeline & Feature Engineering](#3-data-pipeline--feature-engineering)

---

## 1. Models Architecture

### 1.1 Overview

The system employs a **multi-layer ensemble architecture** combining traditional gradient boosting models with deep learning approaches:

```
                    ┌─────────────────────────────────────┐
                    │       Final Prediction Output       │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Signal Optimizer (15 Fixes)    │
                    │   - Position sizing adjustments     │
                    │   - Stop-loss management            │
                    │   - Kelly Criterion                 │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │       Hybrid Ensemble Predictor     │
                    │    (Weighted combination w1, w2)    │
                    └───────┬─────────────────┬───────────┘
                            │                 │
          ┌─────────────────▼─────┐   ┌───────▼─────────────────┐
          │   Old Model (w1)      │   │   Hybrid Model (w2)     │
          │  EnhancedEnsemble     │   │  HybridLSTMCNN          │
          │  - LightGBM           │   │  - Multi-scale CNN      │
          │  - XGBoost            │   │  - Bidirectional LSTM   │
          │  - LSTM               │   │  - Profit-maximizing    │
          │  - TCN                │   │    loss function        │
          │  - Transformer        │   │                         │
          └───────────────────────┘   └─────────────────────────┘
```

### 1.2 Model Components

#### 1.2.1 Base Models (Gradient Boosting)

**LightGBM** (`src/models/base_models.py`)
- Leaf-wise tree growth strategy
- Histogram-based algorithm for efficiency
- Optimized parameters for volatility prediction:
  ```python
  {
      'learning_rate': 0.02,
      'num_leaves': 31,
      'max_depth': 6,
      'min_data_in_leaf': 100,
      'feature_fraction': 0.6,
      'bagging_fraction': 0.7,
      'lambda_l1': 0.5,  # L1 regularization
      'lambda_l2': 0.5   # L2 regularization
  }
  ```

**XGBoost** (`src/models/base_models.py`)
- Level-wise tree growth
- Newton boosting with second-order gradients
- Regularized parameters:
  ```python
  {
      'learning_rate': 0.02,
      'max_depth': 5,
      'min_child_weight': 10,
      'subsample': 0.7,
      'colsample_bytree': 0.6,
      'gamma': 0.2,
      'reg_alpha': 0.5,  # L1 regularization
      'reg_lambda': 2.0  # L2 regularization
  }
  ```

#### 1.2.2 Ensemble Model (`src/models/ensemble_model.py`)

Combines multiple base models using **adaptive weighting**:

```
Ensemble_Prediction = Σ(wi × Predictioni)

where wi = (1/MAEi) / Σ(1/MAEj)  (inverse MAE weighting)
```

Features:
- **Uncertainty Quantification**: Uses ensemble disagreement as confidence measure
- **Adaptive Weights**: Weights updated based on recent validation performance
- **Feature Importance Aggregation**: Weighted average across all models

#### 1.2.3 Hybrid Ensemble (`src/models/hybrid_ensemble.py`)

Combines traditional ML with deep learning:

```python
Final_Prediction = w_old × Old_Model_Pred + w_hybrid × Hybrid_LSTM_CNN_Pred
```

**Weight Determination**:
- Weights auto-determined from validation directional accuracy
- Better performing model gets higher weight
- Default: 50/50 split if no validation data

#### 1.2.4 US/International Signal Optimizer (`src/models/us_intl_optimizer.py`)

Post-prediction optimization layer implementing **15 trading fixes**:

| Fix # | Description | Implementation |
|-------|-------------|----------------|
| 1-2 | SELL confidence thresholds | Asset-class specific (75-80%) |
| 3 | SELL position reduction | 50% position size reduction |
| 4 | Asset-specific stop-losses | 6-12% based on asset class |
| 5 | BUY position boosts | 30% boost for US stocks |
| 6-8 | Blocklists | NG=F, LUMN, stablecoins |
| 9 | Win-rate sizing | Position multiplier 0.1x-2.0x |
| 10 | Base allocations | 4-10% per asset class |
| 11 | Kelly Criterion | Optimal position sizing |
| 12-13 | BUY/SELL signal rules | Min confidence, multipliers |
| 14 | High-profit detection | Pattern-based boost |
| 15 | Profit-taking levels | 50%@15%, 75%@25%, 100%@40% |

### 1.3 Market-Specific Models

| Market | Model | Special Features |
|--------|-------|------------------|
| US/International | `USIntlModelOptimizer` | 15 optimization fixes |
| China (HK/SS/SZ) | `ChinaPredictor` | Sector-specific routing, A-share handling |

---

## 2. Mathematical Foundations

### 2.1 Volatility Estimators

The system implements **four advanced volatility estimators**, each with different properties:

#### 2.1.1 Parkinson Volatility (High-Low Based)

**Formula:**
```
σ_parkinson = √[(1 / 4ln(2)) × ln(H/L)²]
```

**Rolling Implementation:**
```
σ_parkinson(n) = √[(1 / 4ln(2)) × mean(ln(Hi/Li)², i=1..n)]
```

- **Efficiency**: ~5x more efficient than close-to-close
- **Limitation**: Assumes no drift, continuous trading
- **Use Case**: Best for intraday volatility estimation

#### 2.1.2 Garman-Klass Volatility (OHLC Based)

**Formula:**
```
σ_GK = √[0.5 × ln(H/L)² - (2ln(2) - 1) × ln(C/O)²]
```

**Rolling Implementation:**
```
σ_GK(n) = √[mean(0.5×ln(Hi/Li)² - (2ln(2)-1)×ln(Ci/Oi)², i=1..n)]
```

- **Efficiency**: ~8x more efficient than close-to-close
- **Advantage**: Uses all OHLC information
- **Limitation**: Assumes no overnight jumps

#### 2.1.3 Rogers-Satchell Volatility (Drift-Adjusted)

**Formula:**
```
σ_RS = √[ln(H/C) × ln(H/O) + ln(L/C) × ln(L/O)]
```

**Rolling Implementation:**
```
σ_RS(n) = √[mean(ln(Hi/Ci)×ln(Hi/Oi) + ln(Li/Ci)×ln(Li/Oi), i=1..n)]
```

- **Key Feature**: Handles price drift (trending markets)
- **Use Case**: Preferred for trending assets

#### 2.1.4 Yang-Zhang Volatility (Most Complete)

**Formula:**
```
σ_YZ = √[σ²_overnight + k × σ²_open-close + (1-k) × σ²_RS]

where:
  σ²_overnight = Var(ln(Open_t / Close_{t-1}))
  σ²_open-close = Var(ln(Close_t / Open_t))
  k = 0.34 / (1.34 + (n+1)/(n-1))
```

- **Most Accurate**: Combines overnight gaps + intraday movement
- **Components**: Overnight volatility + Open-to-Close + Rogers-Satchell
- **Use Case**: Production-grade volatility estimation

### 2.2 Position Sizing Mathematics

#### 2.2.1 Kelly Criterion

**Formula:**
```
f* = (p × b - q) / b

where:
  f* = optimal fraction of capital to risk
  p  = probability of winning (historical win rate)
  q  = probability of losing (1 - p)
  b  = win/loss ratio (avg_win / avg_loss)
```

**Implementation:**
```python
def calculate_kelly_fraction(win_rate, avg_win=0.05, avg_loss=0.03):
    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss
    kelly = (p * b - q) / b
    return max(0.0, min(kelly, 0.25))  # Capped at 25% (quarter-Kelly)
```

**Safety Cap**: Limited to 25% (quarter-Kelly) for risk management.

#### 2.2.2 Win-Rate Position Multipliers

| Win Rate Range | Position Multiplier |
|----------------|---------------------|
| 80-100% | 2.0x |
| 70-80% | 1.5x |
| 60-70% | 1.2x |
| 50-60% | 1.0x (baseline) |
| 40-50% | 0.7x |
| 30-40% | 0.5x |
| 20-30% | 0.3x |
| 0-20% | 0.1x (near-skip) |

### 2.3 Ensemble Weighting Mathematics

#### 2.3.1 Inverse-MAE Weighting

```
wi = (1/MAEi) / Σj(1/MAEj)
```

This ensures:
- Lower error → Higher weight
- Weights sum to 1.0
- Automatic normalization

#### 2.3.2 Directional Accuracy Weighting (Hybrid Ensemble)

```
w_old = Accuracy_old / (Accuracy_old + Accuracy_hybrid)
w_hybrid = Accuracy_hybrid / (Accuracy_old + Accuracy_hybrid)
```

**Rationale**: Directional accuracy more important for trading than MSE.

### 2.4 Technical Indicators Mathematics

#### 2.4.1 RSI (Relative Strength Index)

```
RSI = 100 - (100 / (1 + RS))

where RS = Average Gain / Average Loss over n periods
```

#### 2.4.2 MACD (Moving Average Convergence Divergence)

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

#### 2.4.3 Bollinger Bands

```
Middle Band = SMA(20)
Upper Band = SMA(20) + 2 × σ(20)
Lower Band = SMA(20) - 2 × σ(20)
%B = (Price - Lower) / (Upper - Lower)
```

### 2.5 Gradient Boosting Loss Functions

#### 2.5.1 MAE Loss (LightGBM)

```
L = (1/n) × Σ|yi - ŷi|
```

#### 2.5.2 MSE Loss (XGBoost)

```
L = (1/n) × Σ(yi - ŷi)²
```

#### 2.5.3 Regularized Objective

```
Obj = Σ L(yi, ŷi) + Σ Ω(fk)

where Ω(f) = γT + (1/2)λ||w||²

  T = number of leaves
  w = leaf weights
  γ = complexity penalty
  λ = L2 regularization
```

---

## 3. Data Pipeline & Feature Engineering

### 3.1 Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION                             │
├──────────────────────────────────────────────────────────────────┤
│  Yahoo Finance API → OHLCV data for all tickers                  │
│  - US: AAPL, MSFT, GOOGL, etc.                                   │
│  - China: 0700.HK, 600519.SS, 000858.SZ                          │
│  - Crypto: BTC-USD, ETH-USD                                      │
│  - Commodities: GC=F, CL=F                                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Technical (60+) │  │ Volatility (30) │  │ Price Features  │   │
│  │ - RSI, MACD     │  │ - Parkinson     │  │ - Returns       │   │
│  │ - SMA, EMA      │  │ - Garman-Klass  │  │ - Gap           │   │
│  │ - ADX, Aroon    │  │ - Yang-Zhang    │  │ - Momentum      │   │
│  │ - ATR, BB       │  │ - Regimes       │  │ - H/L Ratios    │   │
│  │ - OBV, MFI      │  │ - Spikes        │  │                 │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                  │
│                   Total: 90+ Engineered Features                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TARGET CREATION                               │
├──────────────────────────────────────────────────────────────────┤
│  target_volatility = (High - Low) / Close  [shifted -1 day]      │
│                                                                  │
│  Options:                                                        │
│  - next_day_volatility: Intraday range proxy                     │
│  - next_day_range: Absolute High-Low range                       │
│  - next_day_return: Close-to-close return                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  DATA SPLITTING (Time-Series)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┬────────────┬────────────┐                    │
│  │    TRAIN       │    VAL     │    TEST    │                    │
│  │     70%        │    15%     │    15%     │                    │
│  │  Historical    │  Recent    │  Latest    │                    │
│  └────────────────┴────────────┴────────────┘                    │
│                                                                  │
│  CRITICAL: Data sorted by date, NO look-ahead bias               │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Feature Categories

#### 3.2.1 Momentum Features (11 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `rsi_14` | 100 - 100/(1+RS) | Overbought/oversold detection |
| `rsi_21` | Same, 21-period | Longer-term momentum |
| `macd` | EMA(12) - EMA(26) | Trend momentum |
| `macd_signal` | EMA(9) of MACD | Signal line crossovers |
| `macd_diff` | MACD - Signal | Momentum histogram |
| `stoch_k` | Stochastic %K | Price position in range |
| `stoch_d` | Stochastic %D | Smoothed stochastic |
| `roc_10` | (P - P_10) / P_10 | 10-day rate of change |
| `roc_20` | (P - P_20) / P_20 | 20-day rate of change |
| `williams_r` | Williams %R | Momentum oscillator |
| `ultimate_osc` | Ultimate Oscillator | Multi-timeframe momentum |

#### 3.2.2 Trend Features (12+ features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `sma_5/10/20/50/200` | Simple Moving Average | Trend direction |
| `ema_12/26/50` | Exponential MA | Faster trend response |
| `adx` | Average Directional Index | Trend strength |
| `adx_pos/neg` | +DI / -DI | Directional movement |
| `aroon_up/down` | Aroon Indicator | Trend timing |
| `sma_cross_*` | Boolean crossovers | Golden/death cross signals |

#### 3.2.3 Volatility Features (30 features)

| Feature | Estimator | Windows |
|---------|-----------|---------|
| `parkinson_vol_*` | Parkinson | 5, 10, 20, 60 days |
| `gk_vol_*` | Garman-Klass | 5, 10, 20, 60 days |
| `rs_vol_*` | Rogers-Satchell | 5, 10, 20, 60 days |
| `yz_vol_*` | Yang-Zhang | 5, 10, 20, 60 days |
| `vol_ratio_*` | Short/Long ratio | Regime detection |
| `vol_percentile_*` | Historical rank | Relative volatility |
| `regime_low/med/high` | One-hot encoded | Regime classification |
| `vol_spike` | >2x recent avg | Spike detection |
| `vol_change_*` | Percentage change | Volatility momentum |
| `vol_acceleration` | Change of change | Second derivative |

#### 3.2.4 Volume Features (8 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `obv` | On Balance Volume | Cumulative volume flow |
| `volume_ma_20/60` | Volume SMA | Volume trend |
| `volume_roc` | Volume rate of change | Volume momentum |
| `mfi` | Money Flow Index | Volume-weighted RSI |
| `ad` | Accumulation/Distribution | Money flow |
| `cmf` | Chaikin Money Flow | Buying/selling pressure |
| `volume_ratio` | Vol / Vol_MA | Relative volume |

#### 3.2.5 Price Features (12+ features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `returns_1d/5d/20d` | Percentage change | Multi-horizon returns |
| `intraday_range` | (H-L)/C | Daily volatility proxy |
| `gap` | (Open - Prev_Close)/Prev_Close | Overnight gap |
| `price_momentum` | Return - Prev_Return | Momentum acceleration |
| `high_low_ratio` | H/L | Daily range ratio |
| `close_high_ratio` | C/H | Close position |
| `close_low_ratio` | C/L | Close position |
| `price_vs_sma*` | (C - SMA)/SMA | Distance from trend |

### 3.3 Adaptive Windows

The system uses **adaptive window sizes** based on available data history:

| Data Length | SMA Windows | Volatility Windows | Regime Window |
|-------------|-------------|-------------------|---------------|
| ≥252 days | 5, 10, 20, 50, 200 | 5, 10, 20, 60 | 60 |
| 100-252 days | 5, 10, 20, 50, 100 | 5, 10, 20, 60 | 60 |
| 50-100 days | 5, 10, 20, 30, 40 | 5, 10, 20, 30 | 30 |
| <50 days | 3, 5, 10, 15, 20 | 3, 5, 10, 15 | 15 |

**Rationale**: Recently listed stocks have limited history; adaptive windows prevent NaN values while maintaining statistical significance.

### 3.4 Data Quality Handling

1. **Missing Data**:
   - Critical features (RSI, MACD, ATR): Drop rows if NaN
   - Long-window features: Forward-fill then backward-fill

2. **Outlier Handling**:
   - Volatility capped at reasonable bounds
   - Extreme returns flagged but not removed

3. **Time-Series Integrity**:
   - Data always sorted by date
   - No future information leakage
   - Train/Val/Test split respects temporal order

### 3.5 Supported Markets

| Region | Exchange | Ticker Format | Examples |
|--------|----------|---------------|----------|
| USA | NYSE/NASDAQ | SYMBOL | AAPL, MSFT |
| Hong Kong | HKEX | 0000.HK | 0700.HK, 9988.HK |
| Shanghai | SSE | 000000.SS | 600519.SS |
| Shenzhen | SZSE | 000000.SZ | 000858.SZ |
| Crypto | - | SYMBOL-USD | BTC-USD, ETH-USD |
| Commodities | - | SYMBOL=F | GC=F, CL=F |
| Forex | - | PAIR=X | EURUSD=X |

---

## Summary

This stock volatility prediction system combines:

1. **Multi-Model Ensemble**: LightGBM + XGBoost + LSTM + CNN with adaptive weighting
2. **Advanced Volatility Math**: Four estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
3. **Rich Feature Engineering**: 90+ features across momentum, trend, volatility, volume, and price categories
4. **Signal Optimization**: 15 trading fixes including Kelly Criterion position sizing
5. **Global Market Support**: 14 markets with market-specific handling

The architecture prioritizes:
- **Accuracy**: Multiple volatility estimators for robust measurement
- **Risk Management**: Kelly Criterion + asset-class specific stop-losses
- **Adaptability**: Adaptive windows + ensemble weighting
- **Production Readiness**: Comprehensive error handling and logging

---

*Document Version: 1.0*
*Last Updated: December 2025*
