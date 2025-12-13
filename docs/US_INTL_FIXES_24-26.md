# US/Intl Model Fixes 24-26 (December 2025)

## Overview

These fixes implement profit maximization optimizations for the US/International model:
- Fix 24: Dynamic Kelly Fraction - Adaptive position sizing
- Fix 25: Position Concentration - Top-heavy allocation
- Fix 26: Dynamic Profit Targets - ATR-based targets

**Important**: These fixes apply ONLY to US/Intl model. China/DeepSeek model is unchanged.

---

## Fix 24: Adaptive Kelly Fraction

**Problem**: Fixed quarter-Kelly (25% cap) position sizing ignores market conditions, account size, and recent performance.

**Solution**: Dynamically adjust Kelly fraction based on multiple factors.

### Regime Multipliers
| Regime | Multiplier | Rationale |
|--------|------------|-----------|
| Low Volatility | 1.00 | Full Kelly in calm markets |
| Strong Uptrend | 0.90 | Slightly reduced (momentum risk) |
| Normal Volatility | 0.75 | Three-quarter Kelly |
| Strong Downtrend | 0.60 | More conservative |
| High Volatility | 0.50 | Half-Kelly |
| Crisis | 0.25 | Quarter-Kelly (maximum caution) |

### Account Size Multipliers
| Account Size | Multiplier | Rationale |
|--------------|------------|-----------|
| < $10,000 | 1.50 | Aggressive growth for small accounts |
| $10k-$50k | 1.20 | Moderate growth |
| $50k-$100k | 1.00 | Standard position sizing |
| $100k-$500k | 0.80 | Conservative for larger accounts |
| > $500k | 0.60 | Very conservative (capital preservation) |

### Momentum Multipliers (Recent Performance)
| Recent Win Rate | Multiplier | Rationale |
|-----------------|------------|-----------|
| > 70% (Hot Streak) | 1.20 | Capitalize on momentum |
| 55-70% (Winning) | 1.10 | Slight increase |
| 45-55% (Neutral) | 1.00 | No adjustment |
| 35-45% (Losing) | 0.80 | Reduce exposure |
| < 35% (Cold Streak) | 0.60 | Maximum reduction |

### Usage Example
```python
from src.models.us_intl_optimizer import AdaptiveKellyOptimizer

kelly = AdaptiveKellyOptimizer()

adaptive_kelly, components = kelly.calculate_adaptive_kelly(
    win_rate=0.60,           # 60% historical win rate
    avg_win=0.10,            # 10% average win
    avg_loss=0.05,           # 5% average loss
    market_regime='normal_volatility',
    volatility=0.015,
    account_size=50000,
    recent_win_rate=0.65     # Recent performance
)

print(f"Position size: {adaptive_kelly:.1%}")
# Output: Position size: 44.0%
```

---

## Fix 25: Position Concentration Optimizer

**Problem**: Equal allocation across signals wastes capital on low-conviction trades.

**Solution**: Concentrate capital in top signals using exponential weighting (2^(-i)).

### Weight Distribution Formula
```
weight_i = 2^(-i) / sum(2^(-j) for j in 0..n-1)
```

### Allocation Examples
| # Positions | Position 1 | Position 2 | Position 3 | Top 3 Share |
|-------------|------------|------------|------------|-------------|
| 3 | 57.1% | 28.6% | 14.3% | 100% |
| 5 | 51.6% | 25.8% | 12.9% | 90.3% |
| 8 | 50.2% | 25.1% | 12.5% | 87.8% |
| 10 | 50.0% | 25.0% | 12.5% | 87.6% |

### Composite Signal Scoring
Signals are ranked by composite score:
```
composite = confidence * quality_score * win_rate * trend_alignment
```

### Usage Example
```python
from src.models.us_intl_optimizer import PositionConcentrationOptimizer

concentrator = PositionConcentrationOptimizer()

signals = [
    {'ticker': 'AAPL', 'confidence': 0.85, 'quality_score': 0.75, 'win_rate': 0.65, 'trend_alignment': 0.80},
    {'ticker': 'MSFT', 'confidence': 0.78, 'quality_score': 0.70, 'win_rate': 0.60, 'trend_alignment': 0.75},
    {'ticker': 'GOOGL', 'confidence': 0.72, 'quality_score': 0.65, 'win_rate': 0.55, 'trend_alignment': 0.70},
]

allocations, stats = concentrator.optimize_allocations(signals, total_capital=100000)

# allocations:
# {'AAPL': $57,143, 'MSFT': $28,571, 'GOOGL': $14,286}
```

---

## Fix 26: Dynamic Profit Targets

**Problem**: Fixed profit-taking levels (15%/25%/40%) ignore current volatility and trend strength.

**Solution**: ATR-based dynamic targets that scale with market conditions.

### Base Targets by Asset Class
| Asset Class | Target 1 | Target 2 | Target 3 |
|-------------|----------|----------|----------|
| Stock | 8% | 15% | 25% |
| ETF | 6% | 12% | 20% |
| Cryptocurrency | 12% | 25% | 50% |
| Forex | 4% | 8% | 15% |

### Volatility Scaling
Targets scale with the ratio of current volatility to baseline (1.5% daily):
```
vol_multiplier = current_volatility / 0.015
adjusted_target = base_target * vol_multiplier
```

| Daily Volatility | Multiplier | Example (Stock T1) |
|------------------|------------|-------------------|
| 0.8% (Low) | 0.53x | 4.3% |
| 1.5% (Baseline) | 1.00x | 8.0% |
| 3.0% (High) | 2.00x | 16.0% |

### Trend Strength Adjustment
| Trend Strength | Multiplier | Rationale |
|----------------|------------|-----------|
| Weak (< 0.3) | 0.75x | Take profits earlier |
| Normal | 1.00x | Standard targets |
| Strong (> 0.7) | 1.50x | Let winners run |

### Partial Exit Strategy
| Target | Close % | Running Total |
|--------|---------|---------------|
| Target 1 | 50% | 50% closed |
| Target 2 | 25% | 75% closed |
| Target 3 | 25% | 100% closed |

### Trailing Stop
- Enabled by default (50% of Target 1)
- Example: If T1 = 8%, trailing stop = 4%

### Usage Example
```python
from src.models.us_intl_optimizer import DynamicProfitTargets

targets = DynamicProfitTargets()

result = targets.calculate_targets(
    asset_class='stock',
    volatility=0.015,      # 1.5% daily volatility
    trend_strength=0.6,    # Medium-strong trend
    momentum=0.05          # Slight positive momentum
)

print(f"Target 1: {result['partial_take_1']:.1%}")
print(f"Target 2: {result['partial_take_2']:.1%}")
print(f"Full Exit: {result['full_exit']:.1%}")
print(f"Trailing Stop: {result['trailing_stop_pct']:.1%}")

# Get action recommendation for current profit
action = targets.get_recommended_action(0.10, result)
print(f"Action: {action['action']}")  # CLOSE (hit 8% target -> close 50%)
```

---

## Integration with USIntlModelOptimizer

All three fixes are integrated into the main optimizer:

```python
from src.models.us_intl_optimizer import USIntlModelOptimizer

optimizer = USIntlModelOptimizer(
    # Fix 24: Adaptive Kelly
    enable_adaptive_kelly=True,
    account_size=100000,

    # Fix 25: Position Concentration
    enable_position_concentration=True,

    # Fix 26: Dynamic Profit Targets
    enable_dynamic_profit_targets=True,
)

# Access components directly
optimizer.adaptive_kelly_optimizer.calculate_adaptive_kelly(...)
optimizer.position_concentrator.optimize_allocations(...)
optimizer.dynamic_profit_targets.calculate_targets(...)

# Check configuration
config = optimizer.get_configuration_summary()
print(config['fix_24'])  # Adaptive Kelly status
print(config['fix_25'])  # Position Concentration status
print(config['fix_26'])  # Dynamic Profit Targets status
```

---

## Summary

| Fix | Problem | Solution | Impact |
|-----|---------|----------|--------|
| Fix 24 | Fixed 25% Kelly cap | Regime/account/momentum adaptive | Up to 2x position sizing in optimal conditions |
| Fix 25 | Equal allocation | Exponential weighting | Top 3 get ~88% of capital |
| Fix 26 | Fixed profit targets | ATR-based scaling | 0.5x-3x target adjustment |

All fixes in `src/models/us_intl_optimizer.py`.
China/DeepSeek model remains unchanged.
