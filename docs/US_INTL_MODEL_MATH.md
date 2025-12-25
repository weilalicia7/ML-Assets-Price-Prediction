# US/International Model - Mathematical Calculations

**Last Updated:** 2025-12-23

---

## CRITICAL ISSUES (Investigation 2025-12-23)

### Problem Summary
US stock predictions are currently **unreliable** with the following observed metrics:
- **R² = -0.0001** (worse than random prediction)
- **MAE = 0.078** (7.8% error on ~1% daily moves)
- **Profit Score = 23.3** with Multiplier = 0.735x (too conservative)

### Root Causes Identified

#### 1. Feature Engineering Mismatch
**Location:** `webapp.py` lines 353-399

```python
# Problem: Silent failure when macro features unavailable
try:
    macro_eng = SelectiveMacroFeatureEngineer()
    df = macro_eng.add_all_features(df)
except Exception as e:
    logger.warning(f"[FEATURES] Failed to add US macro features: {e}")
    # CONTINUES WITHOUT MACRO FEATURES - causes train/test mismatch!
```

**Impact:** Model trained on 100+ features but predicts with 30-50 features.

#### 2. Missing Features Filled with Zero
**Location:** `webapp.py` lines 2254-2267

```python
# Problem: Missing features replaced with 0
for col in missing_features:
    data_features[col] = np.nan
X_latest = X_latest.fillna(0)  # VIX=0 is WRONG (should be ~15-20)
```

**Impact:** Features like VIX (normally 10-30) become 0, distorting predictions.

#### 3. Conservative Profit Regime
**Location:** `src/features/profit_maximizing_regime.py` lines 309-393

```python
# Problem: All 4 conditions must be met for high score
momentum_score = min(abs(momentum_5d) * 10, 30)        # Needs high momentum
volume_score = 25 if volume_ratio > 1.5 else partial   # Needs 1.5x volume
rsi_score = 25 if 40 < rsi < 70 else partial           # Needs RSI sweet spot
volatility_score = 20 if volatility < 0.30 else partial # Needs low vol

# Result: Most stocks score 20-30 out of 100
# Multiplier = 0.5 + (score * 0.0083) = 0.66-0.75x (too low!)
```

#### 4. Too Many Features (Overfitting)
- **US Model:** 100-114 features
- **China Model:** 10-30 features (works better)
- More features = worse generalization on limited training data

### Recommended Fixes

| Priority | Fix | Location |
|----------|-----|----------|
| P0 | Don't fill missing features with 0 - use historical means | webapp.py:2258 |
| P0 | Require macro features or raise error | webapp.py:353-358 |
| P1 | Reduce feature count to 30-40 core features | webapp.py:1150 |
| P1 | Relax profit score requirements (don't need all 4 conditions) | profit_maximizing_regime.py:309-365 |
| P2 | Use validation-weighted ensemble like China model | ensemble_model.py |
| P2 | Add model MAE to SNR formula for confidence | webapp.py:2389 |

---

## Overview

The US/International Model handles all non-China assets:
- **US Stocks** (AAPL, MSFT, TSLA, etc.)
- **International Stocks** (European, Asian ex-China, etc.)
- **Forex** (EURUSD, GBPUSD, USDJPY, etc.)
- **Commodities** (Gold, Silver, Oil, etc.)
- **Cryptocurrency** (BTC-USD, ETH-USD, etc.)

---

## Model Architecture

### Hybrid Ensemble Predictor

The US/Intl model uses `HybridEnsemblePredictor` which combines two sub-models:

```
HybridEnsemblePredictor
    |
    +-- EnhancedEnsemblePredictor (Old Model)
    |       |-- LightGBM
    |       |-- XGBoost
    |       +-- LSTM
    |
    +-- HybridLSTMCNNPredictor (New Model)
            |-- Multi-scale CNN branches (kernel sizes: 3, 5, 7)
            +-- Bidirectional LSTM
```

**File Location:** `src/models/hybrid_ensemble.py`

---

## Step 1: Return Prediction

### Weighted Ensemble Prediction

```python
# Final prediction is weighted average of both models
ensemble_pred = old_weight * old_model_pred + hybrid_weight * hybrid_model_pred
```

### Weight Determination

Weights are auto-determined from validation performance:

```python
# Directional accuracy determines weights
old_accuracy = (sign(old_pred) == sign(actual)).mean()
hybrid_accuracy = (sign(hybrid_pred) == sign(actual)).mean()

# Higher accuracy = higher weight
old_weight = old_accuracy / (old_accuracy + hybrid_accuracy)
hybrid_weight = hybrid_accuracy / (old_accuracy + hybrid_accuracy)
```

| Default Weights | Value |
|-----------------|-------|
| Old Model | 50% |
| Hybrid Model | 50% |

---

## Step 2: Confidence Calculation

### Signal-to-Noise Ratio (SNR) Method

**Formula:**
```
SNR = |predicted_return| / historical_volatility

confidence = 0.3 + 0.65 * sigmoid(1.5 * (SNR - threshold))

where sigmoid(x) = 1 / (1 + e^(-x))
```

**Code Location:** `webapp.py:1960-1968`

### Asset-Class Specific SNR Thresholds

Different asset classes have different volatility characteristics:

| Asset Class | SNR Threshold | Rationale |
|-------------|---------------|-----------|
| **Stocks** | 0.5 | Standard volatility |
| **Crypto** | 0.8 | Higher threshold - inherently volatile |
| **Forex** | 0.3 | Lower threshold - small moves are significant |
| **Commodities** | 0.5 | Standard with seasonal patterns |

### Confidence Output Range

| SNR Value | Stock Confidence | Crypto Confidence | Forex Confidence |
|-----------|------------------|-------------------|------------------|
| 0.3 | ~55% | ~42% | ~62% |
| 0.5 | ~62% | ~47% | ~73% |
| 0.8 | ~73% | ~62% | ~85% |
| 1.0 | ~78% | ~70% | ~88% |
| 2.0 | ~91% | ~88% | ~94% |

### Visual Representation

```
Confidence
   0.95 ─────────────────────────────────────────────────
         │                                          ╱
         │        Forex (0.3)    Stock (0.5)    ╱ Crypto (0.8)
   0.80 ─┼──────────╱───────────────────────╱
         │       ╱                       ╱
         │    ╱                       ╱
   0.60 ─┼─╱───────────────────────╱
         │                      ╱
         │                   ╱
   0.40 ─┼────────────────╱
         │
   0.30 ─────────────────────────────────────────────────
         0        0.5       1.0       1.5       2.0       SNR
```

---

## Step 3: Volatility Estimation

### Yang-Zhang Volatility

The model uses Yang-Zhang volatility estimator which combines:
- Overnight volatility (close-to-open)
- Intraday volatility (open-to-close)
- High-low range volatility

**Code Location:** `webapp.py:1902-1913`

```python
# Priority order for volatility:
if 'yz_vol_20' in data_features:
    volatility = data_features['yz_vol_20'].iloc[-1]
elif 'hist_vol_20' in data_features:
    volatility = data_features['hist_vol_20'].iloc[-1]
else:
    volatility = data['returns_1d'].tail(20).std()
```

---

## Step 4: Profit Score (Ranking)

### Enhanced Profit Score Formula

Signals are ranked using a risk-adjusted profit score:

```python
def profit_score(asset):
    expected_return = asset['expected_return']
    confidence = asset['confidence']
    volatility = asset['volatility']

    # Base expected profit
    expected_profit = |expected_return| * confidence

    # Phase 1: Volatility scaling
    vol_scaling = calculate_volatility_scaling(returns, benchmark_vol=0.15)

    # Phase 4: Asset-specific multiplier
    asset_mult = asset_specific_multiplier(asset_class)

    # Phase 6: Expected Shortfall (CVaR) penalty
    ES_95 = calculate_expected_shortfall(volatility, alpha=0.95)
    es_factor = 1 / (1 + ES_95 * 5)

    # Phase 5: Confidence boost
    confidence_boost = 1.2 if confidence > 0.70 else 0.9 if confidence < 0.55 else 1.0

    # Final score
    score = expected_profit * vol_scaling * asset_mult * es_factor * confidence_boost

    return score
```

**Code Location:** `webapp.py:3918-3990`

### Component Breakdown

#### 1. Base Expected Profit
```
expected_profit = |predicted_return| × confidence
```

#### 2. Volatility Scaling (Phase 1)
```
vol_scaling = benchmark_vol / current_vol
           = 0.15 / current_volatility

# High volatility → lower scaling → smaller position
# Low volatility → higher scaling → larger position
```

#### 3. Asset-Specific Multiplier (Phase 4)

| Asset Class | Multiplier | Rationale |
|-------------|------------|-----------|
| Stock | 1.0 | Baseline |
| Crypto | 0.8 | Higher risk |
| Forex | 1.1 | Lower volatility |
| Commodity | 0.95 | Moderate risk |

#### 4. Expected Shortfall Factor (Phase 6)
```
ES_95 = volatility * Z_0.05 * adjustment
      ≈ volatility * 1.645 * factor

es_factor = 1 / (1 + ES_95 * 5)

# High ES (tail risk) → lower factor → penalized score
```

#### 5. Confidence Boost (Phase 5)

| Confidence Level | Boost |
|------------------|-------|
| > 70% | 1.20 (20% bonus) |
| 55-70% | 1.00 (neutral) |
| < 55% | 0.90 (10% penalty) |

---

## Step 5: Signal Generation

### BUY Signal Criteria
```python
if direction == 1 and confidence >= 0.65:
    signal = 'BUY'
    should_trade = True
```

### SELL Signal Criteria
```python
if direction == -1 and confidence >= 0.65:
    signal = 'SELL'
    should_trade = True
```

### HOLD Criteria
```python
if abs(direction) < 1 or confidence < 0.65:
    signal = 'HOLD'
    should_trade = False
```

---

## Full Pipeline Flow

```
User Request: /api/top-picks?regime=Stock
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  1. Data Fetching (Yahoo Finance)                       │
│     - 500 days historical data                          │
│     - OHLCV + adjusted close                           │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  2. Feature Engineering (US/Intl specific)              │
│     - VIX correlation                                   │
│     - SPY beta                                         │
│     - DXY (USD Index) correlation                      │
│     - GLD (Gold) correlation                           │
│     - Technical indicators (RSI, MACD, Bollinger)      │
│     - Yang-Zhang volatility                            │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  3. Model Prediction                                    │
│     HybridEnsemblePredictor                             │
│     ├─ EnhancedEnsemble (LGBM + XGB + LSTM)            │
│     └─ HybridLSTMCNN                                   │
│                                                         │
│     Output: predicted_return (5-day forward)           │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  4. Confidence Calculation                              │
│     SNR = |predicted_return| / volatility              │
│     confidence = 0.3 + 0.65 * sigmoid(1.5*(SNR-0.5))   │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  5. Signal Generation                                   │
│     direction = sign(predicted_return)                 │
│     signal = BUY if direction>0 & confidence≥65%       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  6. Profit Score Ranking                                │
│     score = expected_profit × vol_scaling ×             │
│             asset_mult × es_factor × conf_boost        │
│                                                         │
│     Sort by score descending → Top 10                  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
              Top 10 BUY/SELL Signals
```

---

## Example Calculation

### Input: AAPL Stock

```
predicted_return = +2.5% (0.025)
historical_volatility = 3% (0.03)
```

### Step 1: SNR Calculation
```
SNR = |0.025| / 0.03 = 0.833
```

### Step 2: Confidence (Stock threshold = 0.5)
```
confidence = 0.3 + 0.65 * (1 / (1 + e^(-1.5 * (0.833 - 0.5))))
           = 0.3 + 0.65 * (1 / (1 + e^(-0.5)))
           = 0.3 + 0.65 * 0.622
           = 0.704 (70.4%)
```

### Step 3: Profit Score
```
expected_profit = 0.025 × 0.704 = 0.0176

vol_scaling = 0.15 / 0.03 = 5.0 → capped at 2.0
asset_mult = 1.0 (stock)
es_factor = 1 / (1 + 0.03 × 1.645 × 5) ≈ 0.80
conf_boost = 1.20 (>70%)

score = 0.0176 × 2.0 × 1.0 × 0.80 × 1.20 = 0.0338
```

### Step 4: Signal
```
direction = +1 (positive return)
confidence = 70.4% (≥65%)
→ Signal = BUY
```

---

## Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| generate_prediction() | webapp.py | 1688-2200 |
| Confidence calculation | webapp.py | 1921-1970 |
| SNR thresholds | webapp.py | 1943-1951 |
| profit_score() | webapp.py | 3918-3990 |
| HybridEnsemblePredictor | src/models/hybrid_ensemble.py | 29-254 |
| EnhancedEnsemblePredictor | src/models/enhanced_ensemble.py | - |
| HybridLSTMCNNPredictor | src/models/hybrid_lstm_cnn.py | - |

---

## Comparison: US/Intl vs China Model

| Aspect | US/Intl Model | China Model |
|--------|---------------|-------------|
| **Ensemble** | HybridEnsemble (LGBM+XGB+LSTM+CNN) | 70% Tree + 30% Neural |
| **Features** | 100-114 features (too many) | 10-30 features (optimized) |
| **Macro Features** | VIX, SPY, DXY, GLD | CSI300, CNY, HSI |
| **Phase 1-6** | Partially applied | Fully applied |
| **DeepSeek** | Not used | 40% weight for sentiment |
| **Sector Routing** | No | Yes (optimized by sector) |
| **SNR Threshold** | 0.5 (stocks), varies | 0.5 |
| **Missing Features** | Filled with 0 (wrong) | Required or error |
| **Weight Determination** | Fixed 50/50 | Validation-based |
| **Feature Selection** | All available | Top importance only |

### Why China Model Works Better

1. **Fewer Features:** 10-30 core features vs 100+ prevents overfitting
2. **Validation-Weighted:** Models weighted by test performance, not fixed
3. **Domain-Specific:** CSI300 is right macro for China (VIX wrong for China)
4. **Error Handling:** Missing features cause errors, not silent 0-fill
5. **FIXING3 Relaxed Thresholds:** EV thresholds reduced 50-60%

### US Model Should Adopt

```python
# From China Model - Reduced Feature Set (10-30 core)
CORE_FEATURES = [
    'returns_1d', 'returns_5d',        # Returns (2)
    'rsi_14', 'macd',                  # Momentum (2)
    'parkinson_vol_20', 'volatility',  # Volatility (2)
    'vix_return', 'spy_return',        # US Macro (2)
    'volume_ratio',                    # Volume (1)
    'regime'                           # Regime (1)
]
# Total: 10 features vs 100+ current
```

```python
# From China Model - Validation-Based Weighting
old_accuracy = directional_accuracy(old_pred, actual)
hybrid_accuracy = directional_accuracy(hybrid_pred, actual)

old_weight = old_accuracy / (old_accuracy + hybrid_accuracy)
new_weight = hybrid_accuracy / (old_accuracy + hybrid_accuracy)
# NOT fixed 50/50
```

---

*Document updated: 2025-12-23*
