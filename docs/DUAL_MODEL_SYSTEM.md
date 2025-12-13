# Dual Model System Architecture

## Overview

The trading platform uses a dual model architecture to ensure complete isolation between China and US/International markets. Each market has its own:
- Prediction models
- Feature engineering pipelines
- Phase 1-6 calculation parameters
- Stress test scenarios
- Risk constraints

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL MODEL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │    CHINA MODEL      │       │   US/INTL MODEL     │          │
│  │                     │       │                     │          │
│  │  • HK (.HK)         │       │  • NASDAQ/NYSE      │          │
│  │  • Shanghai (.SS)   │       │  • ADRs (BABA, JD)  │          │
│  │  • Shenzhen (.SZ)   │       │  • Crypto           │          │
│  │                     │       │  • Commodities      │          │
│  └──────────┬──────────┘       └──────────┬──────────┘          │
│             │                              │                     │
│             ▼                              ▼                     │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │  China Calculator   │       │  US/Intl Calculator │          │
│  │  (Phase 1-6)        │       │  (Phase 1-6)        │          │
│  └─────────────────────┘       └─────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Market Detection

Tickers are automatically routed to the correct model based on their suffix:

| Suffix | Market | Model |
|--------|--------|-------|
| `.HK` | Hong Kong Stock Exchange | China |
| `.SS` | Shanghai Stock Exchange | China |
| `.SZ` | Shenzhen Stock Exchange | China |
| (none) | US Exchanges (NASDAQ, NYSE) | US/Intl |
| `-USD` | Crypto | US/Intl |
| `=F` | Commodities/Futures | US/Intl |

```python
from app.ml.dual_model_loader import detect_market_type, MarketType

detect_market_type("0700.HK")    # → MarketType.CHINA
detect_market_type("9988.HK")    # → MarketType.CHINA
detect_market_type("600519.SS")  # → MarketType.CHINA
detect_market_type("300750.SZ")  # → MarketType.CHINA
detect_market_type("AAPL")       # → MarketType.US_INTL
detect_market_type("BABA")       # → MarketType.US_INTL (ADR)
detect_market_type("BTC-USD")    # → MarketType.US_INTL
```

---

## Market-Specific Parameters

### China Model Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk Aversion | 1.5 - 3.0 | Higher risk aversion for China volatility |
| Max Position | 20% | More conservative position limits |
| Turnover Limit | 8% | Lower turnover for T+1 settlement (A-shares) |

### US/Intl Model Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk Aversion | 1.0 - 2.5 | Standard risk aversion |
| Max Position | 25% | Standard position limits |
| Turnover Limit | 10% | Standard turnover for T+2 settlement |

---

## Stress Test Scenarios

### China-Specific Scenarios

| Scenario | Volatility Multiplier | Correlation Boost | Description |
|----------|----------------------|-------------------|-------------|
| `2015_china_crash` | 3.5x | +0.50 | 2015 stock market crash |
| `2020_covid` | 2.5x | +0.30 | COVID-19 pandemic |
| `hk_political` | 2.0x | +0.25 | Hong Kong political events |
| `cny_devaluation` | 2.2x | +0.30 | CNY currency devaluation |
| `normal` | 1.0x | +0.00 | Normal market conditions |

### US/Intl Scenarios

| Scenario | Volatility Multiplier | Correlation Boost | Description |
|----------|----------------------|-------------------|-------------|
| `2008_lehman` | 3.0x | +0.40 | 2008 financial crisis |
| `2020_covid` | 2.5x | +0.30 | COVID-19 pandemic |
| `inflation_shock` | 2.0x | +0.20 | Inflation/rate shock |
| `normal` | 1.0x | +0.00 | Normal market conditions |

---

## Regime Multipliers

Risk budgets are adjusted based on market regime:

### China Regime Multipliers

| Regime | Multiplier | Effect |
|--------|------------|--------|
| Crisis | 0.40 | 60% reduction in risk budget |
| High Volatility | 0.65 | 35% reduction |
| Normal | 1.00 | No adjustment |
| Low Volatility | 1.10 | 10% increase |

### US/Intl Regime Multipliers

| Regime | Multiplier | Effect |
|--------|------------|--------|
| Crisis | 0.50 | 50% reduction in risk budget |
| High Volatility | 0.75 | 25% reduction |
| Normal | 1.00 | No adjustment |
| Low Volatility | 1.15 | 15% increase |

---

## Usage

### Web Platform (Predictor)

```python
from app.ml.predictor import MLPredictionService

# Automatically uses dual model system
service = MLPredictionService(use_dual_models=True)

# China stock - uses China model
prediction = await service.predict_ticker("0700.HK")
# prediction['model_info']['market_type'] == 'china'

# US stock - uses US/Intl model
prediction = await service.predict_ticker("AAPL")
# prediction['model_info']['market_type'] == 'us_intl'
```

### Direct Calculator Access

```python
from app.ml.dual_model_loader import (
    get_dual_model_manager,
    MarketSpecificCalculator,
    MarketType
)

manager = get_dual_model_manager()

# Get calculator for specific ticker
calculator = manager.get_calculator("0700.HK")  # China calculator
calculator = manager.get_calculator("AAPL")     # US/Intl calculator

# Or create directly
china_calc = MarketSpecificCalculator(MarketType.CHINA)
us_calc = MarketSpecificCalculator(MarketType.US_INTL)

# Use Phase 6 calculations with market-specific params
risk_budget = china_calc.get_regime_risk_budget(
    base_risk=0.02,
    regime='high_vol'
)
# Returns 0.02 * 0.65 = 0.013 for China

stress_results = china_calc.stress_test(weights, covariance_matrix)
# Uses China-specific scenarios (2015_china_crash, etc.)
```

### China Portfolio Constructor

```python
from portfolio_constructor import PortfolioConstructor, RiskAnalyzer

# Constructor automatically uses Phase 1-6 calculations
constructor = PortfolioConstructor(model_factory=factory)

# Build portfolio with Phase 6 risk parity
portfolio = constructor.construct_portfolio(symbols, total_capital=100000)

# Risk analysis with China-specific stress tests
analyzer = RiskAnalyzer(constructor)
report = analyzer.generate_risk_report()

# Includes:
# - VaR (95%, 99%)
# - Expected Shortfall (Phase 6)
# - China stress tests (2015 crash, HK political, etc.)
```

---

## Phase 1-6 Integration

Both models integrate all Phase 1-6 calculations:

### Phase 1: Core Features & Trading
- `calculate_volatility_scaling()` - Vol-adjusted position sizing
- `correlation_aware_sizing()` - Sector concentration limits
- `risk_parity_allocation()` - Equal risk contribution weights
- `calculate_turnover()` - Turnover tracking

### Phase 2: Asset Class Ensembles
- `safe_data_extraction()` - Flexible input handling (Series, 2D arrays)
- `momentum_signal()` - Trend following
- `volatility_signal()` - Vol-based signals
- `mean_reversion_signal()` - Mean reversion
- `combine_signals()` - Signal aggregation

### Phase 5: Dynamic Weighting & Bayesian
- `decay_weighted_sharpe()` - Recency-weighted Sharpe
- `calmar_ratio()` - Return/drawdown ratio
- `composite_performance_score()` - Multi-metric scoring
- `BayesianSignalUpdater` - Adaptive signal reliability with confidence intervals

### Phase 6: Portfolio Optimization
- `risk_contribution_analysis()` - Risk decomposition
- `expected_shortfall()` - CVaR calculation
- `_calculate_parametric_es()` - Parametric ES helper
- `regime_aware_risk_budget()` - Regime-adjusted risk
- `portfolio_optimization()` - Mean-variance optimization
- `stress_test_portfolio()` - Monte Carlo stress testing

---

## File Structure

```
trading-platform/backend/app/ml/
├── dual_model_loader.py     # Dual model architecture
├── predictor.py             # Updated to use dual models
├── model_loader.py          # Original single model (backward compat)
└── signal_generator.py      # Signal generation

china_model/src/
├── portfolio_constructor.py # Phase 1-6 integrated
├── china_predictor.py       # China-specific predictor
└── model_factory.py         # Model training

src/utils/
└── calculation_utils.py     # Phase 1-6 calculations
```

---

## Key Benefits

1. **No Contamination**: China and US models never share state
2. **Market-Specific Optimization**: Parameters tuned for each market
3. **Appropriate Stress Tests**: Relevant scenarios for each region
4. **Regulatory Compliance**: Different settlement rules (T+1 vs T+2)
5. **Risk Management**: Market-appropriate risk constraints

---

## Testing

Run integration tests:

```bash
python -c "
from app.ml.dual_model_loader import detect_market_type, MarketType

# Test market detection
assert detect_market_type('0700.HK') == MarketType.CHINA
assert detect_market_type('AAPL') == MarketType.US_INTL
print('All tests passed!')
"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-11 | Initial dual model system |
| 1.1 | 2024-11 | Added Phase 1-6 integration |
| 1.2 | 2024-11 | Added Bayesian confidence intervals, parametric ES |
