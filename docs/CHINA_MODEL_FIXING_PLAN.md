# China Model Fixing Plan

Based on the "china model fixing suggestion.pdf" analysis, this document outlines the implementation plan for fixing identified issues in the China Stock Model.

---

## Phase Overview

| Phase | Priority | Issues Addressed | Estimated Complexity |
|-------|----------|------------------|---------------------|
| **Phase 1** | Critical | Data leakage, Confidence standardization | High |
| **Phase 2** | High | Beta calculation, Walk-forward validation | Medium |
| **Phase 3** | Medium | Model uncertainty, Dynamic ensemble weights | Medium |
| **Phase 4** | Medium | Regime detection, Macro feature fallbacks | Low-Medium |
| **Phase 5** | Low | CI distribution, News labeling, Position sizing | Low |

---

## Phase 1: Critical Fixes (Data Integrity)

### 1.1 Fix Data Leakage in Feature Engineering
**Issue:** Current feature calculation may use future data, causing unrealistically optimistic backtesting results.

**Current Problem:**
```python
# Using entire dataset for feature engineering
df['sma_20'] = df['Close'].rolling(20).mean()  # Uses all data at once
```

**Solution:**
```python
def calculate_features_no_leakage(df, date_index):
    """Only use data up to current date for feature calculation"""
    df_slice = df.loc[:date_index].copy()
    # Calculate features only on historical data
```

**Files to Modify:**
- `webapp.py` - `basic_technical_analysis()` function
- `china_model/src/china_macro_features.py` - `add_all_features()` method

**Validation:**
- Ensure all rolling calculations use `min_periods` parameter
- Verify no future data leaks into training set

---

### 1.2 Standardize Confidence Calculation
**Issue:** Basic analysis uses fixed 0.3 confidence; ML uses dynamic SNR-based (0.3-0.95). This inconsistency confuses users.

**Current Problem:**
```python
# Basic analysis: always low confidence
confidence = 0.3

# ML analysis: dynamic calculation
snr = signal_strength / historical_volatility
confidence = 0.3 + 0.65 * sigmoid(snr)
```

**Solution:** Create unified `calculate_standardized_confidence()` function:
```python
def calculate_standardized_confidence(predicted_return, historical_data,
                                       analysis_type='basic', asset_class='stock'):
    """
    Standardized confidence for both basic and ML analysis
    - Uses SNR (Signal-to-Noise Ratio) for both
    - Adjusts for data quality in basic analysis
    - Returns 0.3-0.95 range
    """
    # Calculate historical volatility
    historical_vol = calculate_rolling_volatility(historical_data, 20)

    # Calculate SNR
    snr = abs(predicted_return) / (historical_vol + 1e-10)

    # Base confidence via sigmoid
    base_confidence = 0.3 + 0.65 * (1 / (1 + math.exp(-1.5 * (snr - 0.5))))

    # Adjust for data quality (basic analysis penalty)
    if analysis_type == 'basic':
        data_quality_factor = min(len(historical_data) / 100, 1.0)
        final_confidence = 0.3 + (base_confidence - 0.3) * data_quality_factor
    else:
        final_confidence = base_confidence

    return min(max(final_confidence, 0.3), 0.95)
```

**Files to Modify:**
- `webapp.py` - Add new function, update `basic_technical_analysis()` and `predict()` endpoints

---

## Phase 2: Calculation Stability

### 2.1 Improve Beta Calculation
**Issue:** 20-day rolling window is too noisy, causes extreme beta values.

**Current Problem:**
```python
# 20-day rolling beta - very unstable
rolling_cov = stock_returns.rolling(20).cov(csi300_returns)
rolling_var = csi300_returns.rolling(20).var()
df['beta_csi300_20d'] = rolling_cov / rolling_var
```

**Solution:**
```python
def calculate_stable_beta(stock_returns, market_returns, window=60, min_periods=30):
    """
    Calculate stable beta with:
    - Longer window (60 days instead of 20)
    - Regularization to prevent extreme values (-1 to 3)
    - Correlation-weighted averaging
    """
    # Use expanding window for more stability
    for i in range(min_periods, len(stock_returns)):
        beta = covariance / variance
        # Regularize extreme betas
        beta = max(min(beta, 3.0), -1.0)

    return final_beta
```

**Files to Modify:**
- `china_model/src/china_macro_features.py` - Update `add_all_features()` beta calculation

---

### 2.2 Add Walk-Forward Validation Framework
**Issue:** No proper time-series validation, may overfit to historical data.

**Solution:** Implement walk-forward validation:
```python
def walk_forward_validation(df, model, features, target, train_size=0.7, step_size=21):
    """
    Proper time-series validation:
    - Train on historical data only
    - Test on next 21 days (1 month)
    - Roll forward and repeat
    - Calculate Sharpe ratio for each fold
    """
    for i in range(n_train, n_total - step_size, step_size):
        train_data = df.iloc[:i]
        test_data = df.iloc[i:i+step_size]
        # Train, predict, evaluate
```

**Files to Add:**
- `china_model/src/validation.py` - New validation framework

---

## Phase 3: Model Improvements

### 3.1 Add Model Uncertainty Estimation
**Issue:** Model doesn't know what it doesn't know. No prediction intervals for ML outputs.

**Solution:** Bootstrap ensemble for uncertainty:
```python
class UncertaintyAwarePredictor:
    def __init__(self, base_model, n_models=5):
        self.models = []  # Train multiple models on bootstrap samples

    def predict_with_uncertainty(self, X):
        predictions = [model.predict(X) for model in self.models]
        return {
            'prediction': np.mean(predictions),
            'lower_80': np.percentile(predictions, 10),
            'upper_80': np.percentile(predictions, 90),
            'uncertainty': np.std(predictions) / np.abs(np.mean(predictions))
        }
```

**Files to Modify:**
- `webapp.py` - Integrate uncertainty into prediction response

---

### 3.2 Dynamic Ensemble Weighting
**Issue:** Fixed weights for LightGBM/XGBoost/CatBoost/LSTM ensemble.

**Current Problem:**
```python
# Fixed equal weights
ensemble_prediction = 0.25 * lgb + 0.25 * xgb + 0.25 * cat + 0.25 * lstm
```

**Solution:**
```python
class DynamicEnsemble:
    def calculate_model_weights(self, X_val, y_val):
        """Weight models based on recent performance"""
        for model in self.models:
            accuracy = calculate_accuracy(model, X_val, y_val)
            sharpe = calculate_sharpe(model, X_val, y_val)
            score = 0.4 * accuracy + 0.4 * sharpe + 0.2 * (1/(1+mse))

        # Softmax to get weights
        weights = softmax(scores)
        return weights

    def update_weights(self, X_recent, y_recent):
        """Smooth weight updates: 70% old + 30% new"""
        new_weights = self.calculate_model_weights(X_recent, y_recent)
        self.weights = 0.7 * self.weights + 0.3 * new_weights
```

**Files to Modify:**
- `china_model/src/hybrid_ensemble.py` (if exists) or `webapp.py`

---

## Phase 4: Robustness Improvements

### 4.1 Enhanced Regime Detection
**Issue:** Simple 3-regime (low/medium/high volatility) misses complex market states.

**Current Problem:**
```python
if volatility < 0.02:
    regime = 'LOW_VOL'
elif volatility < 0.04:
    regime = 'MEDIUM_VOL'
else:
    regime = 'HIGH_VOL'
```

**Solution:** Multi-dimensional regime detection:
```python
def detect_market_regime_advanced(returns, volatility_lookback=63):
    """
    6 regimes based on volatility + trend:
    - trending_low_vol
    - ranging_low_vol
    - trending_medium_vol
    - ranging_medium_vol
    - high_vol_clustered
    - high_vol_random
    """
    volatility = calculate_volatility(returns)
    trend_strength = calculate_trend_strength(returns)  # R-squared of linear fit
    volatility_clustering = calculate_autocorr(returns**2)  # ARCH effect

    # Classify based on multiple dimensions
```

**Files to Modify:**
- `webapp.py` - Update regime detection logic

---

### 4.2 China Macro Feature Fallbacks
**Issue:** No fallback when CSI300/HSI data is missing.

**Current Problem:**
```python
# Assumes macro data always available
macro_aligned = self.macro_data.reindex(df.index).ffill()
```

**Solution:**
```python
def add_china_macro_features_with_fallback(df, macro_data, ticker):
    """Add macro features with graceful fallback"""
    try:
        if 'CSI300' not in macro_data.columns or macro_data['CSI300'].isna().all():
            # Fallback: use stock's own momentum as proxy
            df['CSI300_proxy'] = df['Close'].pct_change(20)
            print(f"[WARNING] CSI300 unavailable, using stock momentum as proxy")

        # Similar fallbacks for HSI, CNY
    except Exception as e:
        # Set neutral values
        df['CSI300_dist_ma_5d'] = 0
        df['beta_csi300_60d'] = 1.0  # Market beta
```

**Files to Modify:**
- `china_model/src/china_macro_features.py`

---

## Phase 5: Minor Improvements

### 5.1 Confidence Interval Distribution
**Issue:** Using z=1.28 assumes normal distribution; financial returns are fat-tailed.

**Solution:** Use Student's t-distribution or empirical quantiles:
```python
def calculate_confidence_interval(returns, expected_return, confidence=0.80):
    """Use empirical quantiles for fat-tailed distributions"""
    # Calculate empirical quantiles from historical returns
    lower_quantile = np.percentile(returns, (1 - confidence) / 2 * 100)
    upper_quantile = np.percentile(returns, (1 + confidence) / 2 * 100)

    return {
        'lower': expected_return + lower_quantile,
        'upper': expected_return + upper_quantile
    }
```

---

### 5.2 News Generation Labeling
**Issue:** Generated news could mislead users into thinking it's real.

**Solution:** Add clear labels:
```python
def generate_news_feed(ticker, company_info, current_price, prediction):
    news_items = []
    for item in generated_items:
        item['source'] = 'ML Analysis (Simulated)'  # Clear label
        item['is_simulated'] = True
    return news_items
```

**Frontend Change:**
- Add visual indicator (different color/icon) for simulated news

---

### 5.3 Position Sizing Integration
**Issue:** Confidence scores are display-only, not linked to position sizing.

**Solution:** Add Kelly-based position sizing:
```python
def calculate_position_size(confidence, expected_return, volatility, portfolio_value):
    """
    Kelly Criterion with half-Kelly for safety:
    f* = (p * b - q) / b
    where p = win probability, b = win/loss ratio, q = 1-p
    """
    win_prob = 0.5 + (confidence - 0.5) * 0.5  # Scale confidence to win prob
    win_loss_ratio = abs(expected_return) / volatility

    kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    half_kelly = kelly_fraction / 2  # Conservative

    position_size = portfolio_value * max(0, min(half_kelly, 0.25))  # Max 25%
    return position_size
```

---

## Implementation Timeline

### Week 1: Phase 1 (Critical)
- [ ] Fix data leakage in feature engineering
- [ ] Implement standardized confidence calculation
- [ ] Test with 2590.HK and other limited-data stocks

### Week 2: Phase 2 (Stability)
- [ ] Update beta calculation to 60-day window
- [ ] Add walk-forward validation framework
- [ ] Run validation on China stock universe

### Week 3: Phase 3 (Model)
- [ ] Implement uncertainty estimation
- [ ] Add dynamic ensemble weighting
- [ ] Update API responses with uncertainty info

### Week 4: Phase 4 (Robustness)
- [ ] Enhance regime detection
- [ ] Add macro feature fallbacks
- [ ] Test edge cases

### Week 5: Phase 5 (Polish)
- [ ] Update CI calculation
- [ ] Add news labeling
- [ ] Integrate position sizing
- [ ] Final testing and documentation

---

## Testing Checklist

### Unit Tests
- [ ] `calculate_standardized_confidence()` returns 0.3-0.95 range
- [ ] `calculate_stable_beta()` handles missing data
- [ ] `walk_forward_validation()` preserves time order
- [ ] `UncertaintyAwarePredictor` produces valid intervals

### Integration Tests
- [ ] Basic analysis for 2590.HK (100 days data)
- [ ] Full ML analysis for 0700.HK (1000+ days data)
- [ ] Macro feature fallback when CSI300 unavailable
- [ ] Regime detection across different market conditions

### Performance Tests
- [ ] Walk-forward validation Sharpe ratio > 0.5
- [ ] Direction accuracy > 55%
- [ ] No data leakage confirmed via purged cross-validation

---

## Files Summary

| File | Changes |
|------|---------|
| `webapp.py` | Confidence calculation, regime detection, position sizing |
| `china_model/src/china_macro_features.py` | Beta calculation, fallbacks |
| `china_model/src/validation.py` | New file for walk-forward validation |
| `china_model/src/uncertainty.py` | New file for uncertainty estimation |
| `templates/index.html` | News labeling UI changes |
| `static/js/app.js` | Display uncertainty info |

---

## Excluded Items (Future Consideration)

The following items from the PDF are excluded from this phase:

1. **Survivorship Bias** (#15) - Requires historical delisted stock data, complex to implement
2. **Currency Conversion** (#16) - HKEX uses HKD, SSE/SZSE uses CNY - requires exchange rate integration

These can be addressed in a future enhancement phase.

---

*Document created: 2025-11-29*
*Based on: china model fixing suggestion.pdf*
