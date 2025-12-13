# Confidence Calculation & Profit Ranking Algorithms

## Overview

This document explains the two key algorithms used in the ML Trading Platform for generating meaningful trading signals:

1. **Confidence Calculation** - How we measure prediction reliability
2. **Profit Ranking** - How we rank and prioritize trading opportunities

---

## 1. Confidence Calculation

### Problem Statement

The ML model outputs a predicted volatility/price movement value (`predicted_return`), but we need a **confidence score** that represents how reliable this prediction is for decision-making.

### Previous Approach (Flawed)

```python
# Old: Simple linear scaling (problematic)
confidence = min(abs(predicted_return) / (hist_vol * 2), 0.95)
```

**Issues:**
- Linear scaling doesn't properly bound values
- No theoretical basis for the formula
- Extreme predictions could produce unrealistic confidence values

### Current Approach: Signal-to-Noise Ratio (SNR)

We use a **sigmoid-based transformation** of the Signal-to-Noise Ratio:

```python
# Signal-to-Noise Ratio: how many "standard deviations" is the prediction
snr = abs(predicted_return) / historical_volatility

# Sigmoid transformation to bound confidence between 0.3 and 0.95
confidence = 0.3 + 0.65 * (1 / (1 + exp(-1.5 * (snr - 0.5))))
```

### How It Works

| SNR Value | Meaning | Confidence |
|-----------|---------|------------|
| 0.0 | No signal (prediction = 0) | ~50% |
| 0.5 | Weak signal (half a std dev) | ~62% |
| 1.0 | Moderate signal (1 std dev) | ~73% |
| 2.0 | Strong signal (2 std devs) | ~88% |
| 3.0+ | Very strong signal | ~95% (capped) |

### Mathematical Interpretation

- **SNR = |predicted_return| / historical_volatility**
  - Measures how large the predicted move is relative to typical market noise
  - Higher SNR = prediction stands out more from random fluctuations

- **Sigmoid Transformation**
  - Converts unbounded SNR to bounded confidence range [0.3, 0.95]
  - S-shaped curve: gradual increase in middle, plateaus at extremes
  - Prevents overconfidence even for extreme predictions

### Visual Representation

```
Confidence
   0.95 ─────────────────────────────────────────────────
         │                                          ╱
         │                                       ╱
   0.80 ─┼───────────────────────────────────╱
         │                                 ╱
         │                              ╱
   0.60 ─┼─────────────────────────╱
         │                      ╱
         │                   ╱
   0.40 ─┼────────────────╱
         │
   0.30 ─────────────────────────────────────────────────
         0        0.5       1.0       1.5       2.0       SNR
```

### Why This Matters

1. **Bounded Output**: Confidence always stays between 30% and 95%
2. **Theoretically Grounded**: Based on signal detection theory
3. **Risk-Aware**: Low volatility assets need smaller predictions for same confidence
4. **Prevents Overconfidence**: Even extreme predictions cap at 95%

---

## 2. Profit Ranking Algorithm

### Problem Statement

Given multiple assets with predictions, how do we rank them to maximize expected profit while managing risk?

### Previous Approach (Flawed)

```python
# Old: Simple confidence-weighted return (ignores risk)
score = abs(expected_return) * (confidence + 0.3)
```

**Issues:**
- Doesn't account for volatility risk
- High-volatility assets with large predictions could dominate
- No risk-adjusted measure

### Current Approach: Sharpe-Like Risk-Adjusted Score

```python
def profit_score(asset):
    expected_return = asset['expected_return']  # Model's predicted move
    confidence = asset['confidence']             # Direction confidence
    volatility = asset['volatility']             # Current volatility estimate

    # Expected profit = |expected_move| * direction_confidence
    expected_profit = abs(expected_return) * confidence

    # Risk factor: higher volatility = more uncertainty
    # Capped at 3x to avoid extreme outliers
    risk_factor = 1 + min(volatility * 10, 2.0)

    # Sharpe-like ratio: profit / risk
    score = expected_profit / (risk_factor + 0.01)

    # Bonus for very high confidence signals (>70%)
    if confidence > 0.70:
        score *= 1.2

    return score
```

### Components Explained

#### 1. Expected Profit
```
expected_profit = |predicted_return| × confidence
```
- Combines the magnitude of expected move with our confidence in direction
- Higher is better for both factors

#### 2. Risk Factor
```
risk_factor = 1 + min(volatility × 10, 2.0)
```
- Converts volatility to a penalty multiplier
- Range: 1.0 (low vol) to 3.0 (high vol)
- Examples:
  - 2% volatility → risk_factor = 1.2
  - 10% volatility → risk_factor = 2.0
  - 20%+ volatility → risk_factor = 3.0 (capped)

#### 3. Sharpe-Like Score
```
score = expected_profit / risk_factor
```
- Divides expected profit by risk
- Favors consistent, moderate-volatility predictions over wild swings

#### 4. High-Confidence Bonus
```
if confidence > 0.70:
    score *= 1.2  # 20% bonus
```
- Rewards signals where we're very confident in direction
- Helps high-quality signals rise to the top

### Ranking Examples

| Asset | Expected Return | Confidence | Volatility | Score Calculation | Final Score |
|-------|-----------------|------------|------------|-------------------|-------------|
| AAPL | +2.5% | 75% | 3% | (0.025×0.75)/(1.3) × 1.2 | 0.0173 |
| BTC | +8.0% | 60% | 15% | (0.08×0.60)/(2.5) | 0.0192 |
| MSFT | +1.8% | 82% | 2% | (0.018×0.82)/(1.2) × 1.2 | 0.0148 |
| NVDA | +4.0% | 68% | 5% | (0.04×0.68)/(1.5) | 0.0181 |

**Ranking Result:** BTC > NVDA > AAPL > MSFT

Note: Despite BTC having lower confidence, its large expected return with moderate risk makes it rank highest. AAPL ranks above MSFT due to its higher expected return offsetting slightly lower confidence.

### Why This Approach is Better

1. **Risk-Adjusted**: High-volatility assets are penalized, preventing them from dominating
2. **Balanced**: Considers magnitude, confidence, AND risk together
3. **Sharpe-Like**: Similar to Sharpe ratio used in professional finance
4. **Practical**: Capped risk factor prevents extreme edge cases
5. **Rewards Quality**: High-confidence bonus promotes reliable signals

---

## Implementation Location

| Algorithm | File | Lines |
|-----------|------|-------|
| Confidence Calculation | `webapp.py` | 1936-1949 |
| Profit Ranking | `webapp.py` | 2756-2784 |

---

## Integration with Trading Strategy

These algorithms integrate with the hybrid trading strategy:

```
User requests prediction
        ↓
ML Model generates predicted_return
        ↓
Confidence calculated using SNR sigmoid
        ↓
Signal determined (BUY if direction=+1 & confidence≥65%)
        ↓
Assets ranked by profit_score
        ↓
Top 10 displayed in UI (sorted by risk-adjusted score)
```

---

## Future Improvements

1. ~~**Asset-Class Specific SNR**: Different SNR thresholds for crypto vs stocks~~ **IMPLEMENTED (2025-11-28)**
2. **Dynamic Risk Factor**: Adjust based on market regime (bull/bear/sideways)
3. **Drawdown Penalty**: Factor in maximum drawdown potential
4. **Correlation Adjustment**: Penalize correlated positions in portfolio ranking

---

## Implemented: Asset-Class Specific SNR Thresholds

**Date Implemented:** 2025-11-28
**Location:** `webapp.py`, lines 1936-1985

### Overview

Different asset classes have fundamentally different volatility characteristics. A 2% move in Bitcoin is noise, but a 2% move in EUR/USD is a major event. The SNR threshold now adjusts accordingly.

### SNR Thresholds by Asset Class

| Asset Class | SNR Threshold | Rationale |
|-------------|---------------|-----------|
| **Stocks** | 0.5 | Standard threshold, moderate volatility |
| **China Stocks** | 0.5 | Same as regular stocks |
| **Cryptocurrency** | 0.8 | Higher threshold - crypto is inherently volatile |
| **Forex** | 0.3 | Lower threshold - forex is smooth, small moves are significant |
| **Commodities** | 0.5 | Standard threshold with seasonal patterns |

### How It Works

The sigmoid curve shifts based on asset class:

```python
# For stocks (threshold=0.5):
#   SNR=0.5 → ~50% confidence
#   SNR=1.0 → ~73% confidence
#   SNR=2.0 → ~88% confidence

# For crypto (threshold=0.8):
#   SNR=0.8 → ~50% confidence  (requires larger signal)
#   SNR=1.6 → ~73% confidence
#   SNR=2.4 → ~88% confidence

# For forex (threshold=0.3):
#   SNR=0.3 → ~50% confidence  (smaller signal = confident)
#   SNR=0.6 → ~73% confidence
#   SNR=0.9 → ~88% confidence
```

### Implementation Code

```python
# Asset-class specific SNR thresholds
SNR_THRESHOLDS = {
    'stock': 0.5,       # Standard threshold for stocks
    'china_stock': 0.5, # Same as regular stocks
    'crypto': 0.8,      # Higher threshold - crypto is inherently volatile
    'forex': 0.3,       # Lower threshold - forex is smoother
    'commodity': 0.5,   # Standard threshold with seasonal adjustment
}

# Sigmoid transformation with asset-specific threshold
direction_confidence = 0.3 + 0.65 * (1 / (1 + exp(-1.5 * (snr - snr_threshold))))
```

### Benefits

1. **More Accurate Confidence**: Confidence levels now reflect asset-specific volatility
2. **Better Signal Calibration**: Forex signals don't need to be as strong as crypto signals
3. **Prevents Overconfidence**: Crypto predictions require stronger signals for high confidence
4. **Improved Rankings**: Top picks now correctly prioritize based on asset type

---

## References

- Signal-to-Noise Ratio: Standard concept in signal processing
- Sharpe Ratio: William Sharpe, 1966
- Sigmoid Function: Logistic function for probability bounding

---

*Document created: 2025-11-28*
*Last updated: 2025-11-28*
