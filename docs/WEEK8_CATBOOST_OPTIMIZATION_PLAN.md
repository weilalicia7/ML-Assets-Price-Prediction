# Week 8+: CatBoost-Only Architecture Optimization Plan

## Executive Summary

**Baseline Performance (Week 6):**
- Pass rate: 16.7% (5/30 stocks)
- Architecture: Market-specific CatBoost only
- Features: 15 base features
- Market-specific profit gates: HK 0.4%, SS 0.3%, SZ 0.5%

**Week 7 Ensemble Result:**
- Pass rate: 6.7% (2/30 stocks) - FAILED
- Architecture: CatBoost 70% + LSTM 30%
- Conclusion: LSTM added noise, not signal → Revert to CatBoost-only

**Goal:** Achieve 30%+ pass rate (9+/30 stocks) using CatBoost-only optimizations

---

## Optimization Strategy Roadmap

### Phase 1: Hyperparameter Optimization (Week 8)
**Target: +3-5pp improvement → 20-22% pass rate**

#### 1.1 Grid Search on Market-Specific Hyperparameters
Current settings (from market_specific_predictor.py:59-112):
```python
# Hong Kong
iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=default

# Shanghai
iterations=200, learning_rate=0.06, depth=5, l2_leaf_reg=default

# Shenzhen
iterations=150, learning_rate=0.08, depth=4, l2_leaf_reg=5
```

**Optimization approach:**
- Test grid for each market independently
- Focus on: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `subsample`
- Use Bayesian optimization (10-20 trials per market)
- Metric: Average return per prediction (must exceed profit gate)

**Why this works:**
- Markets have different dynamics (HK stable, SZ volatile)
- Current hyperparameters are educated guesses, not optimized
- CatBoost is sensitive to these parameters

#### 1.2 Class Weight Adjustment
**Problem:** Binary classification with imbalanced outcomes
- Current: No class weighting
- Proposal: Add `class_weights='Balanced'` or custom weights based on profit gates

**Implementation:**
```python
# In MarketSpecificPredictor._create_market_model()
return CatBoostClassifier(
    # ... existing params
    class_weights={0: 1.0, 1: 1.2},  # Emphasize up moves slightly
    # OR
    auto_class_weights='Balanced'
)
```

---

### Phase 2: Feature Engineering Enhancements (Week 9-10)
**Target: +4-6pp improvement → 24-28% pass rate**

#### 2.1 Market Microstructure Features
**Add 8 new features:**
1. `intraday_range` = (High - Low) / Open
2. `close_position` = (Close - Low) / (High - Low + 1e-8)
3. `upper_shadow` = (High - max(Open, Close)) / (High - Low + 1e-8)
4. `lower_shadow` = (min(Open, Close) - Low) / (High - Low + 1e-8)
5. `gap_open` = (Open - Close.shift(1)) / Close.shift(1)
6. `gap_close` = (Close - Open) / Open
7. `volume_price_trend` = Volume * (Close - Close.shift(1)) / Close.shift(1)
8. `accumulation_distribution` = ((Close - Low) - (High - Close)) / (High - Low + 1e-8) * Volume

**Why these help:**
- Capture intraday price action (candlestick patterns)
- Volume-price relationships (institutional flow)
- Gap behavior (market sentiment shifts)

#### 2.2 Multi-Timeframe Features
**Add 6 features:**
1. `returns_10d` = Close.pct_change(10)
2. `returns_20d` = Close.pct_change(20)
3. `sma_cross_5_20` = (sma_5 > sma_20).astype(int)
4. `momentum_change` = momentum_10.diff()
5. `volatility_regime` = (volatility_14 > volatility_14.rolling(50).mean()).astype(int)
6. `trend_strength` = abs(Close - sma_50) / (atr_14 + 1e-8)

**Why these help:**
- Capture longer-term trends (10d, 20d momentum)
- Regime detection (volatility, trend)
- Technical signals (moving average crossovers)

#### 2.3 Market-Specific Features
**For each market, add context:**
1. **Hong Kong:** International flow indicators
   - `hsi_correlation` (if index data available)
   - `usd_hkd_change` (currency impact)

2. **Shanghai:** Policy-sensitive features
   - `sse_sector_momentum`
   - `month_of_year` (policy cycles)

3. **Shenzhen:** Volatility regime features
   - `high_volatility_flag` (>80th percentile)
   - `extreme_move_flag` (>2 std daily return)

---

### Phase 3: Advanced Techniques (Week 11-12)
**Target: +2-4pp improvement → 26-32% pass rate**

#### 3.1 Feature Selection with SHAP
**Goal:** Reduce 15 → 10-12 most important features per market

**Process:**
1. Train baseline CatBoost on all features
2. Calculate SHAP values for each market
3. Rank features by mean(|SHAP value|)
4. Remove bottom 20-30% features
5. Retrain and measure improvement

**Why this works:**
- Reduces overfitting from weak features
- Market-specific feature importance
- Focuses model capacity on signal

**Implementation:**
```python
import shap

# After training
explainer = shap.TreeExplainer(self.model)
shap_values = explainer.shap_values(X_train)

# Get feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(feature_importance)[-12:]  # Top 12
```

#### 3.2 Ensemble of Market Models
**Not LSTM, but:**
- Train 3 CatBoost models per market with different seeds
- Average predictions (reduces variance)
- OR: Use best-of-3 based on validation Sharpe

**Implementation:**
```python
class EnsembleCatBoost:
    def __init__(self, market_type, n_models=3):
        self.models = [
            MarketSpecificPredictor(market_type, random_seed=42+i)
            for i in range(n_models)
        ]

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        return np.mean(probas, axis=0)  # Average probabilities
```

#### 3.3 Dynamic Confidence Thresholds
**Current:** Fixed threshold (0.55 across all markets)
**Proposed:** Market-specific, adaptive thresholds

**Method 1: Percentile-based**
```python
# Set threshold at 70th percentile of training probabilities
threshold = np.percentile(train_probas[:, 1], 70)
```

**Method 2: Profit-optimized**
```python
# Find threshold that maximizes avg return > profit_gate on validation
thresholds = np.arange(0.50, 0.75, 0.01)
for thresh in thresholds:
    preds = (probas[:, 1] >= thresh)
    avg_return = returns[preds].mean()
    if avg_return > profit_gate:
        optimal_threshold = thresh
        break
```

---

### Phase 4: Training Procedure Improvements (Week 13)
**Target: +1-2pp improvement → 28-34% pass rate**

#### 4.1 Walk-Forward Validation
**Current:** Single 80/20 split
**Proposed:** Rolling window retraining

**Implementation:**
```python
# Instead of single train/val split
# Use 6 rolling windows:
# Train on: 2023-01 to 2024-06 → Validate on 2024-07 to 2024-12
# Train on: 2023-02 to 2024-07 → Validate on 2024-08 to 2025-01
# ...
# Final model: ensemble of 6 models
```

**Why this works:**
- Captures regime changes over time
- Reduces overfitting to single time period
- More robust to market shifts

#### 4.2 Early Stopping with Custom Metric
**Current:** Uses CatBoost default metric
**Proposed:** Custom metric = Average return on high-confidence predictions

**Implementation:**
```python
class ProfitMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True  # Higher is better

    def evaluate(self, approxes, target, weight):
        # Calculate profit-focused metric
        probas = sigmoid(approxes[0])
        high_conf = (probas >= 0.55)

        if high_conf.sum() == 0:
            return 0.0, 1.0

        # Predict next-day return
        # Return average return for high-confidence predictions
        avg_return = target[high_conf].mean()
        return avg_return, 1.0

# Use in training
self.model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    eval_metric=ProfitMetric(),
    use_best_model=True
)
```

---

## Implementation Priority

### Week 8: Quick Wins (Days 1-7)
1. **Day 1-2:** Hyperparameter grid search (Phase 1.1)
2. **Day 3-4:** Class weight optimization (Phase 1.2)
3. **Day 5-6:** Test and measure improvements
4. **Day 7:** If passing 20% threshold, move to Phase 2

### Week 9-10: Feature Engineering (Days 8-21)
1. **Days 8-11:** Add microstructure features (Phase 2.1)
2. **Days 12-15:** Add multi-timeframe features (Phase 2.2)
3. **Days 16-18:** Add market-specific features (Phase 2.3)
4. **Days 19-21:** Feature selection with SHAP (Phase 3.1)

### Week 11-12: Advanced Techniques (Days 22-35)
1. **Days 22-25:** Ensemble of CatBoost models (Phase 3.2)
2. **Days 26-30:** Dynamic confidence thresholds (Phase 3.3)
3. **Days 31-35:** Walk-forward validation (Phase 4.1)

### Week 13: Final Optimization (Days 36-42)
1. **Days 36-38:** Custom profit metric (Phase 4.2)
2. **Days 39-41:** Final testing and tuning
3. **Day 42:** Measure final pass rate vs 30% target

---

## Expected Outcomes

| Phase | Improvement | Cumulative Pass Rate | Key Changes |
|-------|-------------|---------------------|-------------|
| Baseline (Week 6) | - | 16.7% | Market-specific CatBoost |
| Phase 1 (Week 8) | +3-5pp | 20-22% | Hyperparameters + class weights |
| Phase 2 (Week 9-10) | +4-6pp | 24-28% | +14 new features, SHAP selection |
| Phase 3 (Week 11-12) | +2-4pp | 26-32% | Ensemble + dynamic thresholds |
| Phase 4 (Week 13) | +1-2pp | 28-34% | Walk-forward + profit metric |

**Target: 30%+ pass rate = 9+/30 stocks passing**

---

## Risk Mitigation

### If improvements plateau:
1. **Revisit data quality:** Check for data issues (missing values, outliers)
2. **Alternative algorithms:** Try XGBoost or LightGBM instead of CatBoost
3. **Target variable engineering:** Instead of binary (up/down), try:
   - Multi-class: Strong up / Up / Flat / Down / Strong down
   - Regression: Predict actual return, then threshold
4. **Market selection:** Focus on best-performing markets only (e.g., Shenzhen had 30% in Week 6)

### If specific markets underperform:
- **Hong Kong (0% in Week 7):** Needs urgent attention
  - Lower profit gate to 0.3% temporarily
  - Add HK-specific features (HKEX index correlation)
  - Increase confidence threshold (more selective)

- **Shanghai (10% stable):** Maintain current approach
  - Already performing reasonably
  - Minor tweaks only

- **Shenzhen (was 30%, now 10%):** Investigate volatility regime
  - High vol → need different features
  - Consider separate "high vol" and "normal vol" models

---

## Success Criteria

### Minimum Viable Progress:
- **Week 8 end:** ≥20% pass rate (6/30 stocks)
- **Week 10 end:** ≥25% pass rate (7-8/30 stocks)
- **Week 13 end:** ≥30% pass rate (9/30 stocks) ✅ TARGET

### Stretch Goal:
- **Week 13 end:** ≥35% pass rate (10-11/30 stocks)

### Key Metrics to Track:
1. **Pass rate by market** (HK, SS, SZ separately)
2. **Average return per prediction** (must exceed profit gates)
3. **Sharpe ratio** (risk-adjusted returns)
4. **Number of trades** (too few = model too conservative)
5. **Max drawdown** (risk management)

---

## Conclusion

The ensemble approach (CatBoost + LSTM) failed because LSTM added noise. The path forward is clear:

1. **Stick with CatBoost** (proven to work at 16.7%)
2. **Optimize hyperparameters** (market-specific tuning)
3. **Engineer better features** (microstructure, multi-timeframe, market-specific)
4. **Apply advanced techniques** (SHAP selection, ensembles, dynamic thresholds)
5. **Improve training procedure** (walk-forward, custom metrics)

Expected timeline: **6 weeks to 30%+ pass rate**

Next step: Start with Phase 1.1 (Hyperparameter Grid Search) on Monday.
