# Phase 5 Dynamic Weighting - Calculation Reference

This document contains only the core calculation formulas and algorithms used in Phase 5.

---

## 1. Dynamic Ensemble Weighting

### Sharpe Ratio Calculation
```python
# Exponentially weighted returns
weights = np.array([decay_factor ** i for i in range(len(returns)-1, -1, -1)])
weights = weights / weights.sum()

weighted_mean = np.average(returns, weights=weights)
weighted_std = np.sqrt(np.average((returns - weighted_mean) ** 2, weights=weights))

# Annualized Sharpe (assuming 252 trading days)
sharpe = (weighted_mean * 252) / (weighted_std * np.sqrt(252)) if weighted_std > 0 else 0
```

### Win Rate Calculation
```python
wins = sum(1 for r in returns if r > 0)
win_rate = wins / len(returns)
```

### Calmar Ratio Calculation
```python
# Calculate maximum drawdown
cumulative = np.cumprod(1 + returns)
running_max = np.maximum.accumulate(cumulative)
drawdowns = (cumulative - running_max) / running_max
max_drawdown = drawdowns.min()

# Calmar = annualized return / |max drawdown|
calmar = weighted_mean / abs(max_drawdown) if max_drawdown < -0.001 else 0
```

### Consistency Score Calculation
```python
# Percentage of positive rolling 5-day periods
rolling_returns = pd.Series(returns).rolling(5).apply(
    lambda x: np.prod(1 + x) - 1
).dropna()

consistency = np.mean([1 if r > 0 else 0 for r in rolling_returns])
```

### Composite Score for Weight Calculation
```python
# Metric weights
metric_weights = {
    'sharpe': 0.40,
    'win_rate': 0.30,
    'calmar': 0.20,
    'consistency': 0.10
}

composite_score = (
    sharpe * 0.40 +
    win_rate * 0.30 +
    calmar * 0.20 +
    consistency * 0.10
)
```

### Weight Normalization with Bounds
```python
min_weight = 0.05  # 5%
max_weight = 0.35  # 35%

# Apply bounds
bounded_weights = {
    ac: max(min_weight, min(max_weight, score))
    for ac, score in raw_weights.items()
}

# Normalize to sum = 1.0
total = sum(bounded_weights.values())
final_weights = {ac: w / total for ac, w in bounded_weights.items()}
```

---

## 2. Confidence-Calibrated Position Sizing

### Kelly Criterion Formula
```python
# Kelly formula: f* = p - (1-p)/b
# where p = win rate, b = win/loss ratio

kelly_optimal = win_rate - (1 - win_rate) / win_loss_ratio

# Apply fractional Kelly (Quarter-Kelly recommended)
kelly_fraction = 0.25
base_position = kelly_optimal * kelly_fraction
```

### Confidence Adjustment
```python
# Adjust position by signal strength and confidence
adjusted_position = base_position * signal_strength * confidence
```

### Diversification Penalty
```python
# Sector correlation penalty
sector_correlations = {
    'Technology': {'Technology': 1.0, 'Consumer': 0.6, 'Healthcare': 0.4, ...},
    ...
}

# Calculate penalty based on existing portfolio
penalty = 0.0
for holding_ticker, holding_weight in portfolio.items():
    correlation = get_correlation(new_ticker, holding_ticker)
    penalty += correlation * holding_weight * 0.5

# Apply penalty (reduces position for correlated assets)
final_position = adjusted_position * (1 - min(penalty, 0.5))
```

### Position Bounds
```python
min_position = 0.02      # 2% minimum
max_position = 0.15      # 15% maximum single position
max_total_exposure = 0.30  # 30% max portfolio exposure

# Enforce bounds
position = max(min_position, min(max_position, calculated_position))

# Check total exposure
remaining_capacity = max_total_exposure - current_exposure
position = min(position, remaining_capacity)
```

---

## 3. Bayesian Signal Combination

### Beta-Bernoulli Conjugate Prior
```python
# Prior parameters (uniform prior)
prior_alpha = 1.0
prior_beta = 1.0

# Update rule after observing success/failure
if signal_correct:
    posterior_alpha = prior_alpha + 1
else:
    posterior_beta = prior_beta + 1

# Expected reliability (posterior mean)
expected_reliability = posterior_alpha / (posterior_alpha + posterior_beta)
```

### Exponential Decay for Historical Updates
```python
decay_factor = 0.99

# Decay old observations
posterior_alpha = prior_alpha + (posterior_alpha - prior_alpha) * decay_factor
posterior_beta = prior_beta + (posterior_beta - prior_beta) * decay_factor

# Add new observation
if correct:
    posterior_alpha += 1
else:
    posterior_beta += 1
```

### Weighted Signal Combination
```python
# Get reliability for each signal
reliabilities = {
    name: alpha / (alpha + beta)
    for name, (alpha, beta) in signal_priors.items()
}

# Normalize weights
total_reliability = sum(reliabilities.values())
weights = {name: r / total_reliability for name, r in reliabilities.items()}

# Combine signals
combined_signal = sum(
    signal_value * weights[signal_name]
    for signal_name, signal_value in signals.items()
)
```

### Confidence Calculation
```python
# Confidence based on sample size and agreement
total_observations = sum(alpha + beta - 2 for alpha, beta in priors.values())
sample_confidence = min(1.0, total_observations / 100)

# Agreement confidence
signal_values = list(signals.values())
positive = sum(1 for s in signal_values if s > 0)
negative = sum(1 for s in signal_values if s < 0)
agreement = max(positive, negative) / len(signal_values)

# Combined confidence
confidence = 0.5 * sample_confidence + 0.5 * agreement
```

---

## 4. Multi-Timeframe Ensemble

### Default Timeframe Weights
```python
timeframe_weights = {
    '1h': 0.15,   # 15% - short-term momentum
    '4h': 0.25,   # 25% - intraday trends
    '1d': 0.35,   # 35% - primary signal (highest weight)
    '1w': 0.25    # 25% - long-term trend confirmation
}
```

### Volatility Regime Adjustments
```python
volatility_adjustments = {
    'low': {'1h': 0.10, '4h': 0.20, '1d': 0.40, '1w': 0.30},
    'normal': {'1h': 0.15, '4h': 0.25, '1d': 0.35, '1w': 0.25},
    'high': {'1h': 0.20, '4h': 0.30, '1d': 0.30, '1w': 0.20},
    'crisis': {'1h': 0.05, '4h': 0.15, '1d': 0.40, '1w': 0.40}
}
```

### Volatility Regime Detection
```python
# Calculate realized volatility
returns = close.pct_change().dropna()
recent_vol = returns.iloc[-20:].std() * np.sqrt(252)
historical_vol = returns.std() * np.sqrt(252)

vol_ratio = recent_vol / historical_vol

# Classify regime
if vol_ratio < 0.7:
    regime = 'low'
elif vol_ratio < 1.3:
    regime = 'normal'
elif vol_ratio < 2.0:
    regime = 'high'
else:
    regime = 'crisis'
```

### Timeframe Signal Generation
```python
# Momentum signal
momentum_signal = np.tanh(momentum_10d * 10) * 0.30

# MA crossover signal
ma_crossover = (1 if ma_5 > ma_10 else -1) * 0.20

# Price vs MA signal
ma_signal = np.tanh(price_vs_ma20 * 5) * 0.25

# RSI mean reversion signal
if rsi > 70:
    rsi_signal = -((rsi - 70) / 30) * 0.25  # Overbought
elif rsi < 30:
    rsi_signal = ((30 - rsi) / 30) * 0.25   # Oversold
else:
    rsi_signal = 0

# Combine
direction = np.clip(momentum_signal + ma_crossover + ma_signal + rsi_signal, -1, 1)
```

### Agreement Score Calculation
```python
# Count signal directions across timeframes
directions = [signal.direction for signal in timeframe_signals.values()]

positive_count = sum(1 for d in directions if d > 0)
negative_count = sum(1 for d in directions if d < 0)

max_agreement = max(positive_count, negative_count)
agreement_score = max_agreement / len(directions)
```

### Weight Adjustment Based on Agreement
```python
if agreement_score >= 0.6:  # High agreement threshold
    # Boost weights of agreeing timeframes by 20%
    dominant_direction = 1 if positive_count >= negative_count else -1

    for tf, signal in signals.items():
        if signal.direction * dominant_direction > 0:
            weight_adjustments[tf] = 1.2  # 20% boost
        else:
            weight_adjustments[tf] = 0.8  # 20% reduction
else:
    # Low agreement - reduce all weights
    weight_adjustments = {tf: 0.9 for tf in timeframes}
```

### Combined Signal Calculation
```python
# Apply adjusted weights
combined_direction = sum(
    signal.direction * adjusted_weights[tf]
    for tf, signal in timeframe_signals.items()
)

combined_confidence = sum(
    signal.confidence * adjusted_weights[tf]
    for tf, signal in timeframe_signals.items()
)

# Adjust confidence based on agreement
combined_confidence *= (0.5 + 0.5 * agreement_score)
```

---

## 5. Position Multipliers

### Regime-Based Position Multipliers
```python
regime_multipliers = {
    'low': 1.2,      # Low volatility: increase position
    'normal': 1.0,   # Normal: standard position
    'high': 0.6,     # High volatility: reduce position
    'crisis': 0.2    # Crisis: minimal position
}

final_position = base_position * regime_multipliers[current_regime]
```

### Trading Action Thresholds
```python
# Determine action from signal
if confidence < 0.5:
    action = 'NO_TRADE'
elif direction > 0.3:
    action = 'STRONG_BUY' if confidence > 0.7 else 'BUY'
elif direction < -0.3:
    action = 'STRONG_SELL' if confidence > 0.7 else 'SELL'
else:
    action = 'HOLD'
```

---

## Summary of Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback Period | 63 days | ~3 months of trading data |
| Min Weight | 5% | Minimum asset class weight |
| Max Weight | 35% | Maximum asset class weight |
| Kelly Fraction | 0.25 | Quarter-Kelly for safety |
| Min Position | 2% | Minimum single position |
| Max Position | 15% | Maximum single position |
| Max Exposure | 30% | Maximum total portfolio exposure |
| Decay Factor | 0.95-0.99 | Exponential decay for history |
| Agreement Threshold | 0.6 | MTF agreement threshold |
| Confidence Threshold | 0.5 | Minimum to generate trade |

---

*Document Version: 5.0*
*Phase: Phase 5 - Dynamic Weighting*
