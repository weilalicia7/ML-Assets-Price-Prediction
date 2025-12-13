# Top-10 Picks Hybrid Ranking System

## Overview

The Top-10 window uses a **smart hybrid ranking** algorithm that balances **risk-adjusted returns** with **growth potential**. This approach ensures users see the best opportunities across the risk spectrum - from safe conservative picks to high-conviction growth plays.

## The Problem with Simple Ranking

### Pure Confidence Sorting (Naive Approach)
```
Sort by: confidence DESC
```
**Issue**: A 90% confident prediction of 0.1% return ranks higher than 60% confident prediction of 5% return. Misses high-return opportunities.

### Pure Risk-Adjusted Sorting (Over-Conservative)
```
Risk-Adjusted Score = (Return × Confidence) / Volatility
```
**Issue**: Over-penalizes volatile assets. NVDA with 3% expected return and 40% volatility gets crushed by JNJ with 0.5% return and 12% volatility. Misses growth opportunities.

### Pure Return Sorting (Reckless)
```
Sort by: expected_return DESC
```
**Issue**: Ignores confidence and risk. A speculative 10% return prediction with 15% confidence ranks #1. Too risky.

## The Hybrid Solution

### Core Formula

```python
def hybrid_score(asset):
    confidence = asset['confidence']           # ML model confidence (0-1)
    risk_adjusted = asset['risk_adjusted_score']  # (return × conf) / volatility
    expected_return = abs(asset['expected_return'])
    growth_potential = expected_return * confidence

    # Dynamic weights based on confidence
    risk_weight = 0.4 + (confidence * 0.3)    # Range: 0.4 to 0.7
    growth_weight = 1.0 - risk_weight          # Range: 0.3 to 0.6

    return (risk_weight * risk_adjusted) + (growth_weight * growth_potential * 10)
```

### Mathematical Breakdown

#### Step 1: Calculate Base Scores

| Score | Formula | What It Measures |
|-------|---------|------------------|
| **Risk-Adjusted** | `(return × confidence) / volatility` | Return per unit of risk |
| **Growth Potential** | `return × confidence` | Raw profit potential |

#### Step 2: Dynamic Weight Calculation

The key insight is that **confidence should influence how we weight risk vs. growth**:

```
risk_weight = 0.4 + (confidence × 0.3)
```

| Confidence | Risk Weight | Growth Weight | Interpretation |
|------------|-------------|---------------|----------------|
| 100% | 0.70 | 0.30 | High conviction → trust risk-adjusted |
| 80% | 0.64 | 0.36 | Strong signal → lean towards safety |
| 50% | 0.55 | 0.45 | Moderate → balanced approach |
| 20% | 0.46 | 0.54 | Low confidence → need high return to justify |
| 0% | 0.40 | 0.60 | Speculative → only worth it for big returns |

**Why this makes sense:**
- **High confidence** = we trust the prediction → prioritize risk-adjusted (safer)
- **Low confidence** = prediction is uncertain → only worth taking if return is high

#### Step 3: Combine with Scaling

```
Final Score = (risk_weight × risk_adjusted) + (growth_weight × growth_potential × 10)
```

The `× 10` multiplier on growth_potential normalizes the scales:
- Risk-adjusted scores are typically 0.01-0.50
- Growth potential (return × confidence) is typically 0.001-0.05
- Multiplying by 10 brings them to comparable ranges

## Real-World Examples

### Example 1: Conservative Safe Bet (JNJ)
```
Expected Return: 0.8%
Confidence: 85%
Volatility: 15%

risk_adjusted = (0.008 × 0.85) / 0.15 = 0.045
growth_potential = 0.008 × 0.85 = 0.0068

risk_weight = 0.4 + (0.85 × 0.3) = 0.655
growth_weight = 0.345

Final Score = (0.655 × 0.045) + (0.345 × 0.0068 × 10)
           = 0.0295 + 0.0235
           = 0.053
```

### Example 2: High-Growth Opportunity (NVDA)
```
Expected Return: 3.2%
Confidence: 70%
Volatility: 38%

risk_adjusted = (0.032 × 0.70) / 0.38 = 0.059
growth_potential = 0.032 × 0.70 = 0.0224

risk_weight = 0.4 + (0.70 × 0.3) = 0.61
growth_weight = 0.39

Final Score = (0.61 × 0.059) + (0.39 × 0.0224 × 10)
           = 0.036 + 0.087
           = 0.123  ← Higher than JNJ!
```

### Example 3: Speculative Play (BTC-USD)
```
Expected Return: 2.0%
Confidence: 13%
Volatility: 45%

risk_adjusted = (0.02 × 0.13) / 0.45 = 0.0058
growth_potential = 0.02 × 0.13 = 0.0026

risk_weight = 0.4 + (0.13 × 0.3) = 0.439
growth_weight = 0.561

Final Score = (0.439 × 0.0058) + (0.561 × 0.0026 × 10)
           = 0.0025 + 0.0146
           = 0.017  ← Ranks lower due to low confidence
```

### Ranking Result
```
1. NVDA     Score: 0.123  (High growth, decent confidence)
2. JNJ      Score: 0.053  (Safe bet, high confidence)
3. BTC-USD  Score: 0.017  (Speculative, low confidence)
```

## Visual Representation

```
                    HIGH CONFIDENCE
                          ↑
                          |
         CONSERVATIVE     |     GROWTH LEADERS
         (Safe Bets)      |     (Best Picks)
         JNJ, GLD         |     NVDA, GOOGL
         Score: Medium    |     Score: HIGH
                          |
    ←─────────────────────┼─────────────────────→
    LOW RETURN            |              HIGH RETURN
                          |
         AVOID            |     SPECULATIVE
         (Not Worth It)   |     (High Risk-Reward)
         Low conf/return  |     BTC, Meme Stocks
         Score: LOW       |     Score: Medium-Low
                          |
                          ↓
                    LOW CONFIDENCE
```

## Key Properties

### 1. Confidence Matters More for Safe Assets
High-confidence low-volatility assets get boosted by risk-adjusted component.

### 2. Return Matters More for Risky Assets
Low-confidence high-return assets still appear if the potential payoff justifies the risk.

### 3. No Hard Cutoffs
Unlike threshold-based systems, every asset gets a fair score. Nothing is arbitrarily excluded.

### 4. Self-Balancing
The formula naturally creates a diverse top-10 with:
- ~3-4 conservative picks (high confidence, low vol)
- ~4-5 balanced picks (moderate confidence and return)
- ~1-2 aggressive picks (high return potential)

## Implementation Details

### Location
`webapp.py` - `/api/top-picks` endpoint (lines 2642-2669)

### Dependencies
Each prediction must include:
- `confidence`: ML direction confidence (0-1)
- `expected_return`: Predicted % move
- `volatility`: Historical volatility
- `risk_adjusted_score`: Pre-calculated (return × conf) / vol

### API Response
```json
{
  "status": "success",
  "regime": "all",
  "top_buys": [
    {
      "ticker": "NVDA",
      "signal": "BUY 70%",
      "confidence": 0.70,
      "expected_return": 0.032,
      "volatility": 0.38,
      "risk_adjusted_score": 0.059
    }
  ],
  "top_sells": [...],
  "total_analyzed": 20
}
```

## Comparison with Other Approaches

| Method | Pros | Cons | Our Solution |
|--------|------|------|--------------|
| **Pure Confidence** | Simple | Ignores return magnitude | Dynamic weighting |
| **Pure Risk-Adjusted** | Safe | Too conservative | Blended with growth |
| **Pure Return** | Aggressive | Ignores confidence | Confidence-weighted |
| **Sharpe Ratio** | Industry standard | Fixed formula | Adaptive weights |
| **Hybrid (Ours)** | Balanced | More complex | Best of all worlds |

## Future Enhancements

### Market Regime Awareness
```python
if get_vix() < 15:  # Bull market
    risk_weight_base = 0.3  # Favor growth
else:  # Bear/volatile market
    risk_weight_base = 0.5  # Favor safety
```

### User Risk Profile
```python
risk_profiles = {
    'conservative': lambda: 0.6,  # Higher risk weight
    'balanced': lambda: 0.4,       # Default
    'aggressive': lambda: 0.2      # Lower risk weight
}
```

### Sector Momentum Bonus
```python
if asset['sector'] in hot_sectors:
    growth_potential *= 1.2  # Boost trending sectors
```

---

*Last Updated: November 2024*
*Algorithm Version: 2.0 (Hybrid Scoring)*
