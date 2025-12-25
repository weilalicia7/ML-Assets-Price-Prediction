# China/DeepSeek Model - Mathematical Logic & Code Reference

**Last Updated:** 2025-12-23

---

## FIXING3 Updates (Latest Optimizations)

### EV Thresholds Relaxed (50-60% reduction)

| Regime | Old EV | New EV (FIXING3) | Reduction |
|--------|--------|------------------|-----------|
| BULL | 0.75 | 0.30 | 60% |
| BEAR | 1.25 | 0.60 | 52% |
| HIGH_VOL | 2.50 | 1.00 | 60% |
| NEUTRAL | 1.00 | 0.50 | 50% |

### Stop-Loss Widened

| Regime | Old Stop | New Stop (FIXING3) | Trailing |
|--------|----------|-------------------|----------|
| BULL | 8% | 15% | 15% |
| BEAR | 5% | 8% | 8% |
| HIGH_VOL | 4% | 6% | 5% |
| NEUTRAL | 6% | 10% | 10% |

### Quality Filters Relaxed

```python
# FIXING3 - More permissive filters
FIXING3_FILTERS = {
    'min_eps': -0.1,        # Was 0.0 (allow slight losses)
    'max_debt_equity': 3.0,  # Was 2.0
    'min_market_cap': 500_000_000,  # Was $1B
}

# Momentum Override (bypass quality filters)
if return_5d >= 0.10:   # 10% in 5 days
    BYPASS_QUALITY_FILTER = True
if return_10d >= 0.15:  # 15% in 10 days
    BYPASS_QUALITY_FILTER = True
```

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Lag-Free Transition Detection](#lag-free-transition-detection)
4. [Adaptive Regime Detection](#adaptive-regime-detection)
5. [Profit Maximization Layer](#profit-maximization-layer)
6. [Dynamic Position Sizing](#dynamic-position-sizing)
7. [DeepSeek API Integration](#deepseek-api-integration)
8. [China Macro Features](#china-macro-features)
9. [Mathematical Formulas](#mathematical-formulas)
10. [Code File Reference](#code-file-reference)

---

## Overview

The China/DeepSeek model is a specialized prediction system for Chinese markets (Hong Kong, Shanghai, Shenzhen) that combines:

- **DeepSeek API** for policy/social sentiment analysis (40% weight)
- **China ML Model** for technical prediction (60% weight)
- **Lag-Free Transition Detection** for early regime changes (10-30 days faster)
- **Adaptive Profit Maximization** with regime-specific optimization

---

## Architecture

```
+---------------------------------------------------------------------+
|                    CHINA/DEEPSEEK MODEL PIPELINE                     |
+---------------------------------------------------------------------+
|                                                                      |
|  +------------------+    +------------------+    +---------------+   |
|  |  Yahoo Finance   |    |   DeepSeek API   |    |  China Macro  |   |
|  |   (OHLCV Data)   |    |   (Sentiment)    |    |   Features    |   |
|  +--------+---------+    +--------+---------+    +-------+-------+   |
|           |                       |                       |          |
|           v                       v                       v          |
|  +-------------------------------------------------------------+    |
|  |              LAG-FREE TRANSITION DETECTOR                    |    |
|  |  * Higher Lows/Lower Highs (25%)                            |    |
|  |  * RSI Divergence (25%)                                     |    |
|  |  * Volume Patterns (20%)                                    |    |
|  |  * SMA Breakout/Breakdown (20%)                             |    |
|  |  * Momentum Acceleration (10%)                              |    |
|  +-------------------------------------------------------------+    |
|                              |                                       |
|                              v                                       |
|  +-------------------------------------------------------------+    |
|  |              ADAPTIVE REGIME DETECTOR                        |    |
|  |  * BULL: 95% allocation, 6 positions, 8% stop-loss          |    |
|  |  * BEAR: 70% allocation, 4 positions, 5% stop-loss          |    |
|  |  * HIGH_VOL: 50% allocation, 3 positions, 4% stop-loss      |    |
|  |  * NEUTRAL: 85% allocation, 5 positions, 6% stop-loss       |    |
|  +-------------------------------------------------------------+    |
|                              |                                       |
|                              v                                       |
|  +-------------------------------------------------------------+    |
|  |              PROFIT MAXIMIZATION LAYER                       |    |
|  |  * BULL: Momentum + High-Beta + Volume (EV >= 0.5)          |    |
|  |  * BEAR: Quality + Oversold (EV >= 1.0)                     |    |
|  |  * HIGH_VOL: Low Vol + Mean Reversion (EV >= 2.0)           |    |
|  +-------------------------------------------------------------+    |
|                              |                                       |
|                              v                                       |
|  +-------------------------------------------------------------+    |
|  |              DYNAMIC POSITION SIZER                          |    |
|  |  Position = Base x Quality x EV x Regime x Transition        |    |
|  +-------------------------------------------------------------+    |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## Lag-Free Transition Detection

**File:** `china_model/src/china_lag_free_transition.py`

### Purpose
Detects bear-to-bull and bull-to-bear transitions **10-30 days faster** than traditional indicators.

### Bull Signal Weights

| Signal | Weight | Detection Method |
|--------|--------|------------------|
| Higher Lows | 25% | `Low(t) > Low(t-5) > Low(t-10)` |
| Bullish Divergence | 25% | Price down but RSI up by >3 points |
| Volume Expansion | 20% | Up-day volume > Down-day volume x 1.3 |
| SMA Breakout | 20% | Price crosses above SMA(20) |
| Momentum Acceleration | 10% | Return(5d) > 2% AND Return(5d) > Return(10d) |

### Bear Signal Weights

| Signal | Weight | Detection Method |
|--------|--------|------------------|
| Lower Highs | 25% | `High(t) < High(t-5) < High(t-10)` |
| Bearish Divergence | 25% | Price up but RSI down by >3 points |
| Volume on Down Days | 20% | Down-day volume > Up-day volume x 1.3 |
| SMA Breakdown | 20% | Price crosses below SMA(20) |
| Momentum Deceleration | 10% | Return(5d) < -2% AND Return(5d) < Return(10d) |

### Transition Thresholds

| Transition | Min Signals | Confidence | Confirmation Days | Blend Speed |
|------------|-------------|------------|-------------------|-------------|
| Bear to Bull | 3 | 60% | 3 days | 30%/day |
| Bull to Bear | 2 | 50% | 2 days | 50%/day |

### Mathematical Formulas

**RSI Calculation:**
```
delta = Price(t) - Price(t-1)
Gain = SMA(max(delta, 0), 14)
Loss = SMA(max(-delta, 0), 14)
RS = Gain / Loss
RSI = 100 - (100 / (1 + RS))
```

**Higher Lows Detection:**
```python
lows_5d = data['Low'].rolling(5).min()
higher_lows = (lows_5d[-1] > lows_5d[-6] > lows_5d[-11])
```

**Bullish Divergence:**
```python
price_declining = price_now < price_10d_ago
rsi_rising = rsi_now > rsi_10d_ago + 3
divergence = price_declining AND rsi_rising
```

**Confidence Calculation:**
```python
confidence = sum(signal_weights for detected_signals)
# Range: 0.0 to 1.0
```

**Blend Factor:**
```python
blend_factor = min(days_in_transition x blend_speed x confidence, 1.0)
# Bear-to-Bull: blend_speed = 0.3
# Bull-to-Bear: blend_speed = 0.5
```

**Allocation Adjustment:**
```python
# Confirmed Bull Transition
allocation_adjustment = +0.15  # +15%

# Early Bull Transition (confidence >= 0.5)
allocation_adjustment = +0.10 x blend_factor

# Early Bull Signal (confidence < 0.5)
allocation_adjustment = +0.05 x blend_factor

# Confirmed Bear Transition
allocation_adjustment = -0.20  # -20%

# Early Bear Transition (confidence >= 0.5)
allocation_adjustment = -0.15 x blend_factor

# Early Bear Signal (confidence < 0.5)
allocation_adjustment = -0.10 x blend_factor
```

---

## Adaptive Regime Detection

**File:** `china_model/src/china_adaptive_profit_maximizer.py`

### Regime Parameters (FIXING3 Updated)

| Parameter | BULL | BEAR | HIGH_VOL | NEUTRAL |
|-----------|------|------|----------|---------|
| Total Allocation | 95% | 70% | 50% | 85% |
| Max Positions | 6 | 4 | 3 | 5 |
| Position Cap | 30% | 20% | 15% | 25% |
| **Stop Loss (FIXING3)** | **15%** | **8%** | **6%** | **10%** |
| **Trailing Stop (FIXING3)** | **15%** | **8%** | **5%** | **10%** |
| Take Profit Mult | 2.5x | 1.5x | 1.2x | 2.0x |
| **Min EV Threshold (FIXING3)** | **0.30** | **0.60** | **1.00** | **0.50** |
| Exit Days (No Profit) | 10 | 5 | 3 | 7 |
| Position Size Mult | 1.2 | 0.7 | 0.5 | 1.0 |
| RSI Buy Boost | 1.2 | 0.8 | 0.6 | 1.0 |

**Note:** FIXING3 changes shown in **bold**. EV thresholds reduced 50-60%, stops widened for fewer whipsaws.

### Base Regime Classification

```python
def classify_base_regime(volatility_20d, hsi_return_20d, hsi_return_5d):
    # High volatility override
    if volatility_20d > 0.03:  # > 3% daily vol
        return 'HIGH_VOL'

    # Adaptive threshold based on volatility
    threshold = 0.05 if volatility_20d < 0.02 else 0.07

    if hsi_return_20d > threshold:
        if hsi_return_5d > threshold x 0.4:
            return 'BULL'
        return 'NEUTRAL'  # Bull but consolidating
    elif hsi_return_20d < -threshold:
        if hsi_return_5d < -threshold x 0.4:
            return 'BEAR'
        return 'NEUTRAL'  # Bear but rallying

    return 'NEUTRAL'
```

### Parameter Blending During Transitions

```python
# Blend speed by transition type
BLENDING_SPEEDS = {
    'bear_to_bull': 0.3,    # 30% per day (patient)
    'bull_to_bear': 0.5,    # 50% per day (faster defense)
    'neutral_shifts': 0.4,  # 40% per day
}

# Blended parameter calculation
blend_factor = min(days_in_transition x blend_speed, 1.0)
blend_factor *= transition_confidence

# For each numeric parameter:
blended_value = start_value + (target_value - start_value) x blend_factor
```

---

## Profit Maximization Layer

**File:** `china_model/src/china_adaptive_profit_maximizer.py`

### BULL Market Strategy
**Focus:** Maximize upside capture

```python
# 1. Score momentum
momentum_score = momentum + rsi_score

# 2. Score beta exposure (prefer higher beta)
beta_score = min(beta / 1.5, 1.5)  # Target beta 1.5

# 3. Score volume expansion
volume_score = 1.0 if volume_ratio >= 1.5 else volume_ratio / 1.5

# 4. Apply EV filter (looser: 0.5)
filtered = [s for s in signals if s.ev >= 0.5]

# 5. Sort by composite score
score = momentum_score x volume_score x EV
```

### BEAR Market Strategy
**Focus:** Preserve capital, selective opportunities

```python
# 1. Score quality
quality_score = (profitability + low_debt + earnings_stability) / 3

# 2. Score oversold conditions
if rsi < 30:
    oversold_score = 1.5
elif rsi < 40:
    oversold_score = 1.2
else:
    oversold_score = 1.0

# 3. Require volume confirmation
filtered = [s for s in signals if volume_ratio >= 1.0]

# 4. Apply strict EV filter (1.0)
filtered = [s for s in filtered if s.ev >= 1.0]

# 5. Sort by composite score
score = EV x quality_score x oversold_score
```

### HIGH_VOL Market Strategy
**Focus:** Risk-off, very selective

```python
# 1. Score low volatility (prefer lower vol stocks)
low_vol_score = 1.0 / (1.0 + stock_volatility x 20)

# 2. Score mean reversion
if rsi < 35 AND distance_from_sma < -5%:
    mean_reversion_score = 1.5
elif rsi < 45 AND distance_from_sma < -3%:
    mean_reversion_score = 1.2
else:
    mean_reversion_score = 1.0

# 3. Apply very strict EV filter (2.0)
filtered = [s for s in signals if s.ev >= 2.0]

# 4. Additional safety filters
filtered = [s for s in filtered if beta < 1.2]
filtered = [s for s in filtered if volume_ratio > 0.8]

# 5. Sort by composite score
score = low_vol_score x mean_reversion_score x EV
```

---

## Dynamic Position Sizing

**File:** `china_model/src/china_adaptive_profit_maximizer.py`

### Position Size Formula

```python
# Base allocation per position
base_size = total_allocation / num_positions

# Quality multiplier (0 to 2)
quality_mult = confidence x 2

# EV multiplier (capped at 2x)
ev_mult = min(EV / 5.0, 2.0)

# Transition confidence multiplier
if transition_confidence > 0.6:
    transition_mult = 1 + (transition_confidence x 0.5)
else:
    transition_mult = 1.0

# Final position size
final_size = base_size x quality_mult x ev_mult x regime_mult x transition_mult

# Apply position cap
final_size = min(final_size, position_cap)
```

### Example Calculation

```
Regime: BULL
Total Allocation: 95%
Num Positions: 4
Position Cap: 30%
Confidence: 0.7
EV: 2.5
Transition Confidence: 0.8

base_size = 0.95 / 4 = 0.2375 (23.75%)
quality_mult = 0.7 x 2 = 1.4
ev_mult = min(2.5 / 5.0, 2.0) = 0.5
regime_mult = 1.2 (BULL)
transition_mult = 1 + (0.8 x 0.5) = 1.4

final_size = 0.2375 x 1.4 x 0.5 x 1.2 x 1.4 = 0.2787 (27.87%)
final_size = min(0.2787, 0.30) = 0.2787 (27.87%)
```

---

## DeepSeek API Integration

**File:** `webapp.py` (lines 3556-3788)

### API Configuration

```python
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
```

### Sentiment Analysis Prompt

```
Analyze {stock_name} ({ticker}) in the {sector} sector.
Current context: China's economic environment

Based on:
1. Government policy toward this sector
2. Social media sentiment in China
3. Retail investor behavior patterns
4. Five-Year Plan alignment

Respond ONLY with:
DIRECTION: [-1, 0, 1] (sell/hold/buy)
CONFIDENCE: [0.0-1.0]
POLICY_SENTIMENT: [-1.0 to 1.0]
SOCIAL_SENTIMENT: [-1.0 to 1.0]
RETAIL_SENTIMENT: [0-100]
POLICY_ALIGNMENT: [0-10]
REASON: (brief explanation)
```

### Weighted Ensemble

```python
# DeepSeek vs ML Model weights
DEEPSEEK_WEIGHT = 0.40  # 40%
ML_MODEL_WEIGHT = 0.60  # 60%

# Combined prediction
final_direction = (deepseek_direction x 0.4) + (ml_direction x 0.6)
final_confidence = (deepseek_confidence x 0.4) + (ml_confidence x 0.6)
```

### Response Parsing

```python
def parse_deepseek_response(response):
    result = {
        'direction': 0,
        'confidence': 0.5,
        'policy_sentiment': 0.0,
        'social_sentiment': 0.0,
        'retail_sentiment': 50.0,
        'policy_alignment': 5.0,
    }

    for line in response.split('\n'):
        if line.startswith('DIRECTION:'):
            result['direction'] = parse_int(line, -1, 1)
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = parse_float(line, 0.0, 1.0)
        # ... parse other fields

    return result
```

---

## China Macro Features

**File:** `china_model/src/china_macro_features.py`

### Data Sources

| Feature | Ticker | Description |
|---------|--------|-------------|
| CSI300 | 000300.SS | CSI 300 Index (China large cap) |
| SSEC | 000001.SS | Shanghai Composite Index |
| HSI | ^HSI | Hang Seng Index (Hong Kong) |
| CNY | CNY=X | Chinese Yuan to USD |
| GLD | GC=F | Gold Futures (safe haven) |

### Feature Engineering

```python
# 1. Distance from Moving Average (5-day)
for index in ['CSI300', 'SSEC', 'HSI']:
    ma_5d = index.rolling(5).mean()
    dist_ma_5d = (index - ma_5d) / ma_5d

# 2. Distance from Moving Average (20-day)
ma_20d = CSI300.rolling(20).mean()
CSI300_dist_ma_20d = (CSI300 - ma_20d) / ma_20d

# 3. Momentum (5-day and 20-day)
CSI300_momentum_5d = CSI300.pct_change(5)
CSI300_momentum_20d = CSI300.pct_change(20)

# 4. Beta to CSI300 (60-day rolling)
stock_returns = stock_close.pct_change()
csi300_returns = CSI300.pct_change()

covariance = stock_returns.rolling(60).cov(csi300_returns)
variance = csi300_returns.rolling(60).var()
beta_csi300_60d = (covariance / variance).clip(-1, 3)

# 5. Currency Momentum
CNY_momentum_5d = CNY.pct_change(5)

# 6. Cross-Market Ratio
HSI_SSEC_ratio = HSI / SSEC
```

### Feature Summary (10 Features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| CSI300_dist_ma_5d | (CSI300 - SMA5) / SMA5 | Short-term market momentum |
| SSEC_dist_ma_5d | (SSEC - SMA5) / SMA5 | A-share momentum |
| HSI_dist_ma_5d | (HSI - SMA5) / SMA5 | HK market sentiment |
| CNY | Raw exchange rate | Currency strength |
| CSI300_dist_ma_20d | (CSI300 - SMA20) / SMA20 | Medium-term trend |
| CSI300_momentum_5d | pct_change(5) | Short-term momentum |
| CSI300_momentum_20d | pct_change(20) | Medium-term momentum |
| beta_csi300_60d | Cov/Var (clipped to [-1,3]) | Market-relative volatility |
| CNY_momentum_5d | pct_change(5) | Currency trend |
| HSI_SSEC_ratio | HSI / SSEC | Cross-market divergence |

---

## Mathematical Formulas

### Technical Indicators

**RSI (Relative Strength Index):**
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
Average Gain = SMA(gains, 14)
Average Loss = SMA(losses, 14)
```

**MACD:**
```
MACD Line = EMA(Close, 12) - EMA(Close, 26)
Signal Line = EMA(MACD Line, 9)
Histogram = MACD Line - Signal Line
```

**Stochastic Oscillator:**
```
%K = 100 x (Close - Low_14) / (High_14 - Low_14)
%D = SMA(%K, 3)
```

**Beta Calculation:**
```
Beta = Cov(R_stock, R_market) / Var(R_market)
Rolling window: 60 days
Regularization: Clamp to [-1, 3]
```

### Expected Value (EV)

```
EV = (Win_Rate x Avg_Win) - (Loss_Rate x Avg_Loss)

# With position sizing adjustment
Adjusted_EV = EV x regime_multiplier x confidence
```

### Position Sizing (Kelly Criterion Inspired)

```python
# Simplified Kelly fraction
kelly_fraction = (win_rate x avg_win - loss_rate x avg_loss) / avg_win

# Capped Kelly (more conservative)
position_size = min(kelly_fraction x 0.25, position_cap)
```

### Risk-Adjusted Score

```
Risk_Adjusted_Score = (|Expected_Return| x Confidence) / Volatility
```

### Blend Factor for Transitions

```
Blend_Factor = min(Days_In_Transition x Blend_Speed x Transition_Confidence, 1.0)

# Blended parameter
Param_Blended = Param_Start + (Param_Target - Param_Start) x Blend_Factor
```

---

## Code File Reference

| Component | File Path |
|-----------|-----------|
| Lag-Free Transition Detector | `china_model/src/china_lag_free_transition.py` |
| Adaptive Profit Maximizer | `china_model/src/china_adaptive_profit_maximizer.py` |
| China Macro Features | `china_model/src/china_macro_features.py` |
| DeepSeek Integration | `webapp.py` (lines 3556-3788) |
| China Predictor | `china_model/src/china_predictor.py` |
| Sector Router | `china_model/src/china_sector_router.py` |
| Stock Screener | `china_model/src/china_stock_screener.py` |

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `ChinaLagFreeTransitionDetector` | china_lag_free_transition.py | Early warning detection |
| `ChinaTransitionOutput` | china_lag_free_transition.py | Transition result container |
| `ChinaAdaptiveRegimeDetector` | china_adaptive_profit_maximizer.py | Regime classification |
| `ChinaProfitMaximizationLayer` | china_adaptive_profit_maximizer.py | Signal optimization |
| `ChinaDynamicPositionSizer` | china_adaptive_profit_maximizer.py | Position sizing |
| `ChinaAdaptiveProfitMaximizer` | china_adaptive_profit_maximizer.py | Main orchestrator |
| `ChinaMacroFeatureEngineer` | china_macro_features.py | Feature engineering |
| `DeepSeekChinaAnalyzer` | webapp.py | API integration |

---

---

## Composite Quality Scoring (FIXING3)

**File:** `china_model/src/china_adaptive_profit_maximizer.py` lines 182-258

### Quality Score Components (0-1.0 total)

| Component | Max Score | Criteria |
|-----------|-----------|----------|
| EPS Score | 0.25 | >0.1: 0.25, 0-0.1: 0.15, -0.1-0: 0.05, <-0.1: 0 |
| Debt/Equity | 0.25 | <0.5: 0.25, <1.0: 0.20, <2.0: 0.15, <3.0: 0.10, >=3.0: 0.05 |
| Market Cap | 0.25 | >$5B: 0.25, >$1B: 0.20, >$500M: 0.15, >$100M: 0.10, else: 0.05 |
| Profitability | 0.25 | >5%: 0.25, 2-5%: 0.15, 0-2%: 0.10, <=0%: 0 |

### Sector-Specific Quality Rules

```python
SECTOR_RULES = {
    'Technology':  {'max_de': 2.5, 'min_eps': -0.1, 'min_mcap': 300_000_000},
    'Financials':  {'max_de': 15.0, 'min_eps': 0.0, 'min_mcap': 1_000_000_000},
    'Energy':      {'max_de': 2.0, 'min_eps': -0.05, 'min_mcap': 500_000_000},
    'Consumer':    {'max_de': 1.5, 'min_eps': 0.0, 'min_mcap': 500_000_000},
    'Healthcare':  {'max_de': 2.0, 'min_eps': -0.1, 'min_mcap': 300_000_000},
    'Default':     {'max_de': 3.0, 'min_eps': -0.1, 'min_mcap': 500_000_000},
}
```

### Regime-Weighted Entry Scoring

```python
# Different regimes weight momentum vs quality differently
def calculate_entry_score(momentum_score, quality_score, regime):
    weights = {
        'BULL':     (0.70, 0.30),  # 70% momentum, 30% quality
        'NEUTRAL':  (0.50, 0.50),  # Balanced
        'BEAR':     (0.30, 0.70),  # 30% momentum, 70% quality
        'HIGH_VOL': (0.20, 0.80),  # 20% momentum, 80% quality
    }
    m_weight, q_weight = weights[regime]
    return m_weight * momentum_score + q_weight * quality_score
```

---

## Sharpe-Adjusted EV Calculation

**Formula:**
```python
def calculate_adjusted_ev(base_ev, sharpe_ratio, volatility):
    # Sharpe adjustment
    if sharpe_ratio > 0:
        ev_adjusted = base_ev * math.sqrt(min(sharpe_ratio, 3.0))
    else:
        ev_adjusted = base_ev * 0.5  # Penalize negative Sharpe

    # Volatility penalty
    if volatility:
        vol_penalty = 1.0 / (1.0 + volatility * 5)
        ev_adjusted *= vol_penalty

    return ev_adjusted

# Example:
# base_ev=1.0, sharpe=1.5, vol=2%
# ev_adjusted = 1.0 * sqrt(1.5) = 1.22
# with vol: 1.22 * (1/(1+0.02*5)) = 1.22 * 0.91 = 1.11
```

---

## Dynamic Stop-Loss Formula

```python
def calculate_dynamic_stop(base_stop, volatility_percentile, regime):
    # Volatility adjustment: tighter stops in high vol
    vol_adjustment = 1.2 - (volatility_percentile * 0.4)
    # At 25% percentile: 1.2 - 0.1 = 1.1 (wider)
    # At 100% percentile: 1.2 - 0.4 = 0.8 (tighter)

    dynamic_stop = base_stop * vol_adjustment
    return max(0.02, min(dynamic_stop, 0.15))  # 2% to 15% bounds

# FIXING3 Base Stops by Regime:
# BULL: 15%, NEUTRAL: 10%, BEAR: 8%, HIGH_VOL: 6%
```

---

## Summary

The China/DeepSeek model provides a comprehensive trading system for Chinese markets with:

1. **40% DeepSeek API** for policy/sentiment analysis
2. **60% ML Model** for technical predictions (70% tree, 30% neural)
3. **4 Market Regimes** with adaptive parameters
4. **Lag-Free Detection** catching transitions 10-30 days early
5. **Dynamic Position Sizing** based on regime and signal quality
6. **10 China Macro Features** replacing US-centric indicators (CSI300, CNY, HSI)
7. **FIXING3 Optimizations:** Relaxed EV thresholds (50-60% reduction), wider stops, momentum override

### Why This Model Outperforms US Model

| Factor | China Model | US Model |
|--------|-------------|----------|
| Features | 10-30 core | 100-114 (overfitting) |
| Ensemble Weight | Validation-based | Fixed 50/50 |
| Missing Data | Error/reject | Fill with 0 (wrong) |
| Quality Filters | Sector-specific | Generic |
| EV Thresholds | Relaxed (FIXING3) | Conservative |
| Stops | Wider (15% bull) | Tighter (8%) |

The system is designed for maximum profit extraction while managing risk through regime-appropriate position sizing and stop-loss levels.

---

*Document updated: 2025-12-23*
