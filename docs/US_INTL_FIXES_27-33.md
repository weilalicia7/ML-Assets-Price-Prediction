# US/Intl Model Fixes 27-33 (December 2025)

## Overview

These fixes implement US market-specific optimizations for the US/International model:
- Fix 27: US Market Regime Classifier - VIX-based regime detection
- Fix 28: Sector Momentum Integration - Relative strength vs sector ETFs
- Fix 29: Earnings Season Optimizer - Pre-earnings drift exploitation
- Fix 30: FOMC & Economic Calendar - FOMC week position reduction
- Fix 31: Options Expiration Optimizer - Gamma hedging impact handling
- Fix 32: Market Internals Integration - Breadth indicators
- Fix 33: US-Specific Risk Models - Sector concentration limits

**Important**: These fixes apply ONLY to US/Intl model. China/DeepSeek model is unchanged.

---

## Fix 27: US Market Regime Classifier

**Problem**: Generic trend classification ignores US-specific market patterns like FOMC weeks, earnings seasons, sector rotations, and options expiration.

**Solution**: Classify into US-specific regimes and adjust ensemble weights accordingly.

### Regimes
| Regime | Description | CatBoost | LSTM |
|--------|-------------|----------|------|
| bull_momentum | Strong uptrend with momentum | 35% | 65% |
| bull_consolidation | Uptrend but consolidating | 60% | 40% |
| bear_momentum | Strong downtrend | 30% | 70% |
| bear_rally | Counter-trend rally | 75% | 25% |
| fomc_week | FOMC meeting week | 90% | 10% |
| earnings_season | Major earnings season | 80% | 20% |
| sector_rotation | Sector rotation in progress | 70% | 30% |
| opex_week | Options expiration week | 65% | 35% |

### VIX Thresholds
| Level | VIX Range | Impact |
|-------|-----------|--------|
| Low | < 15 | Full momentum following |
| Normal | 15-20 | Standard weights |
| Elevated | 20-25 | Slight conservative shift |
| High | 25-30 | +10% CatBoost |
| Extreme | > 40 | +15% CatBoost |

### Position Multipliers by Regime
| Regime | BUY | SELL |
|--------|-----|------|
| bull_momentum | 1.30x | 0.60x |
| bear_momentum | 0.60x | 1.20x |
| fomc_week | 0.50x | 0.50x |
| earnings_season | 0.80x | 0.60x |
| opex_week | 0.70x | 0.60x |

### Usage Example
```python
from src.models.us_intl_optimizer import USMarketRegimeClassifier

classifier = USMarketRegimeClassifier()

regime, weights = classifier.classify_regime(
    spy_returns_20d=0.05,
    spy_returns_5d=0.02,
    vix_level=18.0,
    is_fomc_week=False,
    is_earnings_season=False,
    is_opex_week=False
)

print(f"Regime: {regime}")  # Output: bull_momentum
print(f"Weights: CatBoost={weights['catboost']:.0%}, LSTM={weights['lstm']:.0%}")
```

---

## Fix 28: Sector Momentum Integration

**Problem**: US markets are heavily sector-driven. Ignoring sector momentum misses key alpha opportunities.

**Solution**: Add sector ETF momentum, relative strength, and rotation detection.

### Sector ETF Mapping
| ETF | Sector | Leader Boost |
|-----|--------|--------------|
| XLK | Technology | 1.30x |
| XLV | Healthcare | 1.20x |
| XLI | Industrials | 1.10x |
| XLY | Consumer Discretionary | 1.10x |
| XLF | Financials | 1.00x |
| XLE | Energy | 0.90x |
| XLU | Utilities | 0.90x |
| XLRE | Real Estate | 0.90x |

### Relative Strength Calculation
```
RS Ratio = ticker_returns / sector_returns
> 1.0 = outperforming sector
< 1.0 = underperforming sector
```

### Position Adjustments
| Condition | BUY Multiplier | SELL Multiplier |
|-----------|----------------|-----------------|
| RS > 1.2 (Strong outperformer) | 1.2x * sector_boost | 0.7x |
| RS > 1.0 (Outperformer) | 1.1x * sector_boost | 1.0x |
| RS < 0.8 (Underperformer) | 0.8x | 1.15x |

### Usage Example
```python
from src.models.us_intl_optimizer import SectorMomentumAnalyzer

analyzer = SectorMomentumAnalyzer()

# Get sector for a stock
sector = analyzer.get_stock_sector('AAPL')  # Returns 'XLK'

# Calculate momentum score
score, details = analyzer.get_sector_momentum_score(
    ticker='AAPL',
    ticker_returns_20d=0.10,  # 10% return
    sector_returns_20d=0.05   # 5% sector return
)

# Get position adjustment
adjustment, reason = analyzer.get_position_adjustment(
    ticker='AAPL',
    signal_type='BUY',
    ticker_returns_20d=0.10,
    sector_returns_20d=0.05
)
```

---

## Fix 29: Earnings Season Optimizer

**Problem**: Earnings announcements cause significant volatility and predictable patterns (pre-earnings drift, PEAD).

**Solution**: Adjust signals based on proximity to earnings dates.

### Earnings Season Months
- January (Q4 reports)
- April (Q1 reports)
- July (Q2 reports)
- October (Q3 reports)

### Position Adjustments
| Period | Days | BUY | SELL |
|--------|------|-----|------|
| Earnings Week | -2 to +2 | 0.70x | **BLOCKED** |
| Pre-Earnings | 3 to 10 | 1.20x (+5% conf) | 0.70x |
| Post-Earnings | -10 to -2 | Normal | Normal |

### Major Earnings Tickers
AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, BAC, JNJ, UNH, XOM, WMT, PG, HD, MA, V, DIS

### Usage Example
```python
from src.models.us_intl_optimizer import EarningsSeasonOptimizer

optimizer = EarningsSeasonOptimizer()

# Check if earnings season
is_season = optimizer.is_earnings_season(date(2025, 1, 15))  # True

# Optimize signal for earnings
result = optimizer.optimize_for_earnings(
    ticker='AAPL',
    signal_type='BUY',
    confidence=0.75,
    days_to_earnings=5,
    is_earnings_season=True
)
print(f"Multiplier: {result['position_multiplier']}")  # 1.20
```

---

## Fix 30: FOMC & Economic Calendar

**Problem**: FOMC meetings and major economic reports cause significant market volatility and predictable patterns.

**Solution**: Adjust signals based on proximity to FOMC and economic events.

### Sector Sensitivity
| Type | Sectors |
|------|---------|
| Rate-Sensitive | XLF, XLU, XLRE, TLT |
| Growth-Sensitive | XLK, XLY, XLC |

### Position Adjustments
| Period | All Positions | Rate-Sensitive |
|--------|---------------|----------------|
| FOMC Week (-1 to +2 days) | 0.50x | 0.30x |
| Pre-FOMC (3-7 days) | BUY: 1.15x, SELL: 0.80x | Normal |

### Rate Expectation Impact
| Expectation | Growth Stocks |
|-------------|---------------|
| Hike | BUY: 0.80x, SELL: 1.10x |
| Cut | BUY: 1.15x |
| Hold | Normal |

### Usage Example
```python
from src.models.us_intl_optimizer import FOMCOptimizer

optimizer = FOMCOptimizer()

result = optimizer.adjust_for_fomc(
    ticker='XLF',
    signal_type='BUY',
    days_to_fomc=1,
    rate_expectation='hike'
)
print(f"Multiplier: {result['position_multiplier']}")  # 0.30
```

---

## Fix 31: Options Expiration Optimizer

**Problem**: Monthly options expiration (3rd Friday) causes significant gamma hedging flows and volatility patterns.

**Solution**: Adjust signals around OpEx dates.

### OpEx Effects
- **Gamma Pinning**: Prices gravitate toward max pain
- **Volatility Crush**: IV drops after expiration
- **Gamma Hedging**: Large flows from market makers

### High Gamma Stocks
SPY, QQQ, IWM, AAPL, TSLA, NVDA, AMD, AMZN, META, GOOGL, MSFT, NFLX, COIN, GME, AMC

### Position Adjustments
| Period | Days to OpEx | Regular | High Gamma |
|--------|--------------|---------|------------|
| OpEx Window | -2 to +2 | 0.60x | 0.40x |
| Approaching | 3 to 5 | 0.85x | 0.85x |
| OpEx Friday | 0 | **Avoid Entry** | **Avoid Entry** |

### Usage Example
```python
from src.models.us_intl_optimizer import OpExOptimizer

optimizer = OpExOptimizer()

# Check days to OpEx
days = optimizer.get_days_to_opex()

# Check if OpEx week
is_opex = optimizer.is_opex_week()

# Get adjustment
result = optimizer.adjust_for_opex(
    ticker='TSLA',
    signal_type='BUY',
    days_to_opex=1,
    is_high_gamma_stock=True
)
```

---

## Fix 32: Market Internals Integration

**Problem**: Ignoring market breadth misses important signals about overall market health.

**Solution**: Integrate advance-decline, new highs/lows, TRIN, McClellan Oscillator.

### Market Breadth Indicators
| Indicator | Bullish | Bearish |
|-----------|---------|---------|
| AD Ratio | > 1.5 | < 0.67 |
| NHNL Ratio | > 2.0 | < 0.5 |
| TRIN | < 0.8 | > 1.2 |
| McClellan | > 50 | < -50 |

### Market Health Score
Combined score from -1 (very bearish) to +1 (very bullish)

### Position Adjustments
| Health Score | BUY | SELL |
|--------------|-----|------|
| > 0.5 | 1.20x | 0.70x |
| > 0.2 | 1.10x | 0.85x |
| < -0.2 | 0.85x | 1.05x |
| < -0.5 | 0.70x | 1.15x |

### Usage Example
```python
from src.models.us_intl_optimizer import USMarketInternals

internals = USMarketInternals()

# Calculate indicators
ad_ratio = internals.calculate_ad_ratio(advances=2500, declines=1000)
nhnl = internals.calculate_nhnl_ratio(new_highs=150, new_lows=30)
trin = internals.calculate_trin(2000, 1500, 5e9, 3e9)
mcclellan = internals.calculate_mcclellan_oscillator(100, 50)

# Get health score
health, desc = internals.get_market_health_score(ad_ratio, nhnl, trin, mcclellan)

# Get position adjustment
mult, reason = internals.get_position_adjustment('BUY', health)
```

---

## Fix 33: US-Specific Risk Models

**Problem**: Generic risk models ignore US-specific factors like sector concentration, factor exposure, and style drift.

**Solution**: Implement US-specific risk constraints and optimization.

### Sector Concentration
- Maximum 35% in any single sector
- Automatic allocation reduction when limit approached

### Factor Exposure Limits
| Factor | Min | Max |
|--------|-----|-----|
| Momentum | -0.5 | 1.5 |
| Value | -0.5 | 0.5 |
| Size | -0.5 | 0.5 |
| Volatility | -0.5 | 0.3 |
| Quality | 0.0 | 1.0 |

### Risk-Adjusted Allocation
- **Volatility Factor**: Higher vol stocks get smaller allocation
- **Market Health Factor**: Weak markets reduce allocation

### Usage Example
```python
from src.models.us_intl_optimizer import USRiskModel, SectorMomentumAnalyzer

risk_model = USRiskModel()
sector_mapper = SectorMomentumAnalyzer()

# Check sector concentration
allowed, max_alloc, reason = risk_model.check_sector_concentration(
    ticker='NVDA',
    proposed_allocation=0.15,
    current_allocations={'AAPL': 0.15, 'MSFT': 0.15},
    sector_mapper=sector_mapper
)

# Calculate portfolio risk
risk_score, components = risk_model.calculate_portfolio_risk_score(
    allocations={'AAPL': 0.20, 'MSFT': 0.15, 'JPM': 0.10},
    volatilities={'AAPL': 0.30, 'MSFT': 0.28, 'JPM': 0.25}
)

# Get risk-adjusted allocation
adjusted, reason = risk_model.get_risk_adjusted_allocation(
    ticker='AAPL',
    base_allocation=0.15,
    volatility=0.30,
    market_health=0.5
)
```

---

## Integration with USIntlModelOptimizer

All seven fixes are integrated into the main optimizer:

```python
from src.models.us_intl_optimizer import USIntlModelOptimizer

optimizer = USIntlModelOptimizer(
    # Fix 27: US Market Regime Classifier
    enable_us_regime_classifier=True,

    # Fix 28: Sector Momentum Integration
    enable_sector_momentum=True,

    # Fix 29: Earnings Season Optimizer
    enable_earnings_optimizer=True,

    # Fix 30: FOMC & Economic Calendar
    enable_fomc_optimizer=True,

    # Fix 31: Options Expiration Optimizer
    enable_opex_optimizer=True,

    # Fix 32: Market Internals Integration
    enable_market_internals=True,

    # Fix 33: US-Specific Risk Models
    enable_us_risk_model=True,
)

# Access components directly
optimizer.us_regime_classifier.classify_regime(...)
optimizer.sector_momentum_analyzer.get_sector_momentum_score(...)
optimizer.earnings_optimizer.optimize_for_earnings(...)
optimizer.fomc_optimizer.adjust_for_fomc(...)
optimizer.opex_optimizer.adjust_for_opex(...)
optimizer.market_internals.get_market_health_score(...)
optimizer.us_risk_model.check_sector_concentration(...)

# Check configuration
config = optimizer.get_configuration_summary()
for fix in ['fix_27', 'fix_28', 'fix_29', 'fix_30', 'fix_31', 'fix_32', 'fix_33']:
    print(f"{fix}: {config[fix]['enabled']}")
```

---

## Summary

| Fix | Problem | Solution | Impact |
|-----|---------|----------|--------|
| Fix 27 | Generic regime detection | US-specific regime + VIX | Event-aware ensemble weights |
| Fix 28 | Ignoring sector momentum | Relative strength vs sector ETFs | Up to 1.56x boost for leaders |
| Fix 29 | Earnings volatility | Pre-earnings drift exploitation | 1.2x BUY, block SELL |
| Fix 30 | FOMC uncertainty | Position reduction during FOMC | 0.3-0.5x during FOMC week |
| Fix 31 | OpEx gamma effects | Gamma hedging awareness | 0.4-0.6x during OpEx |
| Fix 32 | Missing market breadth | AD, NHNL, TRIN, McClellan | Health-based adjustments |
| Fix 33 | Sector concentration risk | 35% sector limit | Risk-adjusted allocations |

All fixes in `src/models/us_intl_optimizer.py`.
China/DeepSeek model remains unchanged.
