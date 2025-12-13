# Short-List Assets: ML Analysis for Limited Data Stocks

## Overview

This document explains how the ML Stock Trading Platform handles **newly-listed stocks** and **assets with limited trading history**. Traditional ML models require substantial historical data (250+ days) to train effectively. For stocks with less data, we use a **tiered analysis system** that gracefully degrades to simpler, more robust methods.

## The Problem

When a stock is newly listed or has limited trading history:

1. **Insufficient Training Data**: ML models like LSTM, XGBoost, and LightGBM need hundreds of data points for reliable training
2. **Feature Engineering Loss**: Adding technical indicators (RSI, MACD, moving averages) drops ~30-40% of rows due to lookback periods
3. **Overfitting Risk**: Small datasets lead to models that memorize noise rather than learn patterns
4. **Validation Failure**: Train/validation splits become too small, causing errors like `__len__() should return >= 0`

### Real Example: 2590.HK
- Raw data: 100 trading days
- After feature engineering: 67 days
- After train/val split: ~31 training, ~22 validation samples
- Result: LSTM model fails due to insufficient batch sizes

## Solution: Tiered Analysis System

### Data Tier Classification

| Tier | Days Available | Analysis Type | ML Available | Confidence |
|------|----------------|---------------|--------------|------------|
| **Insufficient** | < 30 days | Basic info only | No | None |
| **Minimal** | 30-99 days | Basic technical | No | Low |
| **Light** | 100-149 days | Basic technical* | No | Low-Medium |
| **Moderate** | 150-249 days | Light ML | Yes | Medium |
| **Full** | 250+ days | Full ML ensemble | Yes | High |

*Note: 100-149 days uses basic technical analysis because feature engineering reduces usable data below ML thresholds.

### Implementation

```python
def get_data_tier(days_available):
    """Determine analysis tier based on available trading days."""
    if days_available < 30:
        return {'tier': 'insufficient', 'ml_available': False}
    elif days_available < 100:
        return {'tier': 'minimal', 'ml_available': False}
    elif days_available < 250:
        return {'tier': 'light', 'ml_available': True}  # But we override to 150
    else:
        return {'tier': 'full', 'ml_available': True}
```

**Practical Threshold**: We use **150 days** as the minimum for ML because:
- 150 raw days → ~100 days after features → ~60 train / ~40 val samples
- This is the minimum viable size for gradient boosting models

## Basic Technical Analysis

For stocks with 5-149 days of data, we use `basic_technical_analysis()`:

### Features Used (5 total)
1. **5-Day Momentum**: `(price_today - price_5days_ago) / price_5days_ago`
2. **10-Day Momentum**: Same calculation over 10 days
3. **5-Day SMA Trend**: Price position relative to 5-day moving average
4. **Volume Trend**: Recent volume vs average volume
5. **Price Volatility**: Standard deviation of recent returns

### Signal Generation

```python
def basic_technical_analysis(data, ticker):
    """Simple technical analysis for limited data stocks."""

    # Calculate basic indicators
    momentum_5d = (close[-1] - close[-6]) / close[-6] * 100
    momentum_10d = (close[-1] - close[-11]) / close[-11] * 100
    sma_5 = close[-5:].mean()

    # Generate signal
    if momentum_5d > 3 and close[-1] > sma_5:
        signal = 'BULLISH'
        confidence = min(0.6, 0.3 + momentum_5d / 20)
    elif momentum_5d < -3 and close[-1] < sma_5:
        signal = 'BEARISH'
        confidence = min(0.6, 0.3 + abs(momentum_5d) / 20)
    else:
        signal = 'NEUTRAL'
        confidence = 0.3

    return {
        'signal': signal,
        'confidence': confidence,
        'momentum_5d': momentum_5d,
        'reason': f"Based on {len(data)} days of price action"
    }
```

### Confidence Scaling

| Momentum | Signal | Max Confidence |
|----------|--------|----------------|
| > 5% | Strong BUY | 55% |
| 3-5% | BUY | 45% |
| -3% to 3% | HOLD | 30% |
| -5% to -3% | SELL | 45% |
| < -5% | Strong SELL | 55% |

Confidence is capped at 60% for basic analysis to reflect uncertainty.

## China Stocks: DeepSeek Integration

For China-listed stocks (`.HK`, `.SS`, `.SZ`), we combine basic technical analysis with **DeepSeek API** sentiment analysis:

### Weighting Strategy

| Data Availability | DeepSeek Weight | Technical Weight |
|-------------------|-----------------|------------------|
| < 5 days | 100% | 0% |
| 5-149 days | 50% | 50% |
| 150+ days | 40% | 60% (full ML) |

### Combined Signal

```python
if days_available < 150:
    # Limited data: equal weight DeepSeek + Basic Technical
    combined_direction = 0.5 * deepseek_direction + 0.5 * technical_direction
    confidence = 0.5 * deepseek_confidence + 0.5 * technical_confidence
else:
    # Full data: favor ML model
    combined_direction = 0.4 * deepseek_direction + 0.6 * ml_direction
    confidence = 0.4 * deepseek_confidence + 0.6 * ml_confidence
```

## API Response Format

### Basic Analysis Response

```json
{
  "status": "success",
  "ticker": "2590.HK",
  "company": {
    "name": "2590.HK",
    "type": "Stock"
  },
  "prediction": {
    "direction": 1,
    "direction_confidence": 0.3,
    "expected_return": 0.073,
    "volatility": 0.04,
    "regime": "BULLISH"
  },
  "signal": {
    "action": "BUY",
    "confidence": 0.3,
    "reason": "Basic technical analysis (100 days): Positive momentum with upward trend"
  },
  "model_info": {
    "type": "BasicTechnicalAnalysis",
    "features_used": 5,
    "data_tier": "light",
    "days_available": 100
  }
}
```

### Key Differences from Full ML Response

| Field | Basic Analysis | Full ML |
|-------|----------------|---------|
| `model_info.type` | "BasicTechnicalAnalysis" | "HybridEnsemble" |
| `model_info.features_used` | 5 | 90-106 |
| `prediction.volatility_percentile` | 50 (default) | Calculated |
| `signal.confidence` | Max 60% | Up to 95% |

## Best Practices for Short-List Assets

### For Users

1. **Lower Position Sizes**: Reduce position size by 50% for basic analysis signals
2. **Wider Stop Losses**: Use 5-7% stops instead of 3-5% due to higher uncertainty
3. **Monitor Closely**: Re-analyze weekly as more data accumulates
4. **Confirm with News**: Cross-reference signals with company announcements

### For Developers

1. **Check `model_info.type`**: Display appropriate confidence warnings in UI
2. **Show Data Tier**: Let users know analysis quality level
3. **Log Transitions**: When a stock crosses 150-day threshold, trigger full ML retraining

## Transition to Full ML

When a stock accumulates 150+ days of data:

1. **Automatic Detection**: Next prediction request detects sufficient data
2. **Full Feature Engineering**: 90+ features calculated
3. **Model Training**: HybridEnsemble (LightGBM + XGBoost + CatBoost + LSTM)
4. **Higher Confidence**: Signals can now reach 95% confidence

```
Day 100: BasicTechnicalAnalysis (5 features, max 60% confidence)
Day 150: HybridEnsemble (90+ features, up to 95% confidence)
```

## File References

- **Tier Classification**: `webapp.py:get_data_tier()` (lines 1591-1635)
- **Basic Analysis**: `webapp.py:basic_technical_analysis()` (lines 1638-1699)
- **Predict Endpoint**: `webapp.py:predict()` (lines 2568-2705)
- **China Pick Generation**: `webapp.py:_generate_china_pick_for_ticker()` (lines 3116-3250)

## Summary

The tiered analysis system ensures that:

1. **No Crashes**: Limited data stocks don't cause ML training failures
2. **Graceful Degradation**: Users get useful signals even with minimal data
3. **Appropriate Confidence**: Lower confidence reflects higher uncertainty
4. **Automatic Upgrade**: Stocks transition to full ML when data permits

This approach follows the "short time list machine learning" recommendations:
- Use simpler models with fewer features for limited data
- Focus on relative ranking rather than precise predictions
- Apply strong regularization (via confidence caps)
- Prefer interpretable signals over complex ensembles
