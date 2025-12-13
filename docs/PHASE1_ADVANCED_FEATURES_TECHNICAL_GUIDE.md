# Phase 1 Advanced Features - Technical Documentation

## Overview

This document provides comprehensive technical documentation for all **20+ advanced features** implemented in Phase 1 of the stock prediction model. These features are designed to create an institutional-grade trading system with robust risk management, stress protection, and intelligent signal processing.

**Implementation Location:** `src/trading/production_advanced.py`

**Test Suites:**
- `tests/test_advanced_features.py` - All 20 features unit tests
- `tests/test_stress_scenario_protection.py` - Stress scenario integration tests
- `tests/test_walk_forward_validation.py` - Out-of-sample validation
- `tests/test_drawdown_analysis.py` - Drawdown source analysis

---

## Feature Categories

| Category | Features | Purpose |
|----------|----------|---------|
| AI/ML | 1-2 | Adaptive learning and model selection |
| Risk Management | 3-4 | Position sizing and volatility adjustment |
| Portfolio Management | 5-6 | Allocation and rebalancing |
| Alpha Generation | 7-8 | Signal enhancement |
| Execution | 9-10 | Order routing and algorithms |
| Monitoring | 11-12 | Performance tracking |
| Predictive | 13-14 | Forward-looking risk metrics |
| Production | 15 | Configuration and contingency |
| Stress Protection | 16-20 | Crisis management and circuit breakers |

---

## Core Features (1-15)

### Feature 1: AdaptiveFeatureSelector (AI/ML)

**Purpose:** Dynamically select the most predictive features for each stock based on mutual information and correlation analysis.

**Class:** `AdaptiveFeatureSelector`

**Key Methods:**
```python
def select_features(
    self,
    stock_data: pd.DataFrame,
    target_returns: pd.Series,
    ticker: str = None
) -> List[str]
```

**Algorithm:**
1. Calculate absolute correlation between each feature and target returns
2. Rank features by correlation strength
3. Select top N features above minimum importance threshold
4. Cache results per ticker for consistency

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | 30 | Maximum features to select |
| `min_importance` | 0.01 | Minimum correlation threshold |

**Usage Example:**
```python
selector = AdaptiveFeatureSelector(n_features=30)
features = selector.select_features(stock_data, target_returns, ticker="0700.HK")
# Returns: ['momentum_5d', 'rsi_14', 'macd_signal', ...]
```

---

### Feature 2: MetaLearner (AI/ML)

**Purpose:** Learn which ML model works best for different market regimes and automatically select the optimal model.

**Class:** `MetaLearner`

**Regime-Model Mapping:**
| Regime | Recommended Model | Rationale |
|--------|-------------------|-----------|
| High Volatility | LSTM | Better at capturing non-linear patterns |
| Low Volatility | LightGBM | Tree-based models excel in calm markets |
| Trending | XGBoost | Strong at momentum capture |
| Ranging | CatBoost | Handles noise well |
| Crisis | Conservative | Capital preservation priority |
| Recovery | Momentum | Captures rebound opportunities |

**Key Methods:**
```python
def select_best_model(self, current_regime: str, stock_characteristics: Dict = None) -> str
def record_model_performance(self, regime: str, model: str, return_pct: float)
```

**Learning Process:**
- Records performance of each model by regime
- Requires minimum 5 samples before overriding defaults
- Keeps rolling window of last 100 trades

---

### Feature 3: CorrelationAwarePositionSizer (Risk)

**Purpose:** Adjust position sizes based on portfolio correlation to reduce concentration risk.

**Class:** `CorrelationAwarePositionSizer`

**Sector Mapping:** Pre-defined sector classification for HK/China stocks:
- Technology: 0700.HK, 9988.HK, 3690.HK, 1810.HK
- Financials: 2318.HK, 0939.HK, 1398.HK
- Healthcare: 2269.HK, 1177.HK, 2319.HK
- Real Estate: 0960.HK, 1109.HK

**Position Adjustment Logic:**
```python
def correlation_adjusted_sizing(
    self,
    new_trade: Dict,
    existing_positions: List[Dict],
    base_size: float
) -> float
```

**Penalty Structure:**
| Condition | Penalty |
|-----------|---------|
| 1 same-sector position | 15% reduction |
| 2+ same-sector positions | 30% reduction |
| Sector exposure > 25% | 50% reduction |
| Minimum floor | 70% of base |

---

### Feature 4: VolatilityScaler (Risk)

**Purpose:** Dynamically adjust positions based on changing market volatility.

**Class:** `VolatilityScaler`

**Volatility Regimes:**
| Annualized Vol | Regime | Position Multiplier |
|----------------|--------|---------------------|
| < 15% | LOW | 100% |
| 15-30% | MEDIUM | 75% |
| > 30% | HIGH | 50% |

**Key Methods:**
```python
def calculate_realized_volatility(self, returns_series: pd.Series) -> float
def get_volatility_adjustment(self, returns_series: pd.Series, benchmark_vol: float = 0.15) -> float
def quick_volatility_scaling(self, position_size: float, recent_volatility: float) -> float
```

**Quick Implementation (5-minute formula):**
```python
vol_ratio = recent_volatility / benchmark_vol
adjustment = 1.0 / (1.0 + max(vol_ratio - 1.0, 0) * 0.5)
```

---

### Feature 5: RiskParityAllocator (Portfolio)

**Purpose:** Allocate capital based on risk contribution rather than equal weighting.

**Class:** `RiskParityAllocator`

**Algorithm:**
1. Calculate risk contribution for each position (volatility * correlation factor)
2. Inverse weight by risk - lower risk gets higher allocation
3. Normalize allocations to sum to 1.0

**Key Method:**
```python
def risk_parity_allocation(
    self,
    predictions: Dict[str, Dict],
    risk_budgets: Dict[str, float] = None
) -> Dict[str, float]
```

**Example:**
```python
allocator = RiskParityAllocator()
predictions = {
    "0700.HK": {"return": 0.05, "volatility": 0.20},
    "9988.HK": {"return": 0.03, "volatility": 0.25}
}
allocations = allocator.risk_parity_allocation(predictions)
# Lower vol stock gets higher allocation
```

---

### Feature 6: SmartRebalancer (Portfolio)

**Purpose:** Only rebalance when economically justified (improvement > transaction costs).

**Class:** `SmartRebalancer`

**Decision Logic:**
```python
def should_rebalance(
    self,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    expected_returns: Dict[str, float] = None
) -> Tuple[bool, str]
```

**Thresholds:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_turnover_threshold` | 5% | Minimum turnover to consider |
| `transaction_cost` | 0.1% | Assumed transaction cost |

**Rebalance Conditions:**
1. Turnover must exceed minimum threshold
2. Expected improvement must exceed round-trip costs
3. Returns reason string for logging

---

### Feature 7: CrossAssetSignalEnhancer (Alpha)

**Purpose:** Enhance signal confidence using signals from related assets (sector ETFs, competitors).

**Class:** `CrossAssetSignalEnhancer`

**Related Asset Mapping:**
```python
'0700.HK': {'sector_etf': '2800.HK', 'competitors': ['9988.HK', '3690.HK']}
'9988.HK': {'sector_etf': '2800.HK', 'competitors': ['0700.HK', '9999.HK']}
```

**Enhancement Logic:**
| Condition | Confidence Boost |
|-----------|------------------|
| Sector ETF confirms direction | +10% |
| >60% competitor alignment | +5% |
| Maximum confidence cap | 95% |

---

### Feature 8: RegimeSignalWeighter (Alpha)

**Purpose:** Adjust signal strength based on historical performance in current regime.

**Class:** `RegimeSignalWeighter`

**Regime Weights:**
| Regime | Bull Weight | Bear Weight |
|--------|-------------|-------------|
| Bull Market | 1.0 | 0.7 |
| Bear Market | 0.6 | 1.0 |
| High Volatility | 0.7 | 0.8 |
| Ranging | 0.8 | 0.8 |

**Key Method:**
```python
def regime_aware_signal_weighting(self, signal: float, current_regime: str) -> float
```

---

### Feature 9: SmartOrderRouter (Execution)

**Purpose:** Optimize order execution based on market conditions.

**Class:** `SmartOrderRouter`

**Urgency Classification:**
| Volume/Spread Condition | Urgency | Algorithm |
|------------------------|---------|-----------|
| High volume, tight spread | HIGH | MARKET |
| Medium conditions | MEDIUM | TWAP |
| Low volume, wide spread | LOW | VWAP |

**Output:**
```python
{
    'urgency': 'HIGH',
    'algorithm': 'MARKET',
    'reason': 'Tight spread and high volume'
}
```

---

### Feature 10: ExecutionAlgorithms (Execution)

**Purpose:** Implement VWAP and TWAP execution strategies for large orders.

**Class:** `ExecutionAlgorithms`

**VWAP Strategy:**
- Distributes order across time weighted by historical volume
- Minimizes market impact
- Uses 5 default time slices

**TWAP Strategy:**
- Distributes order evenly across time
- Specified number of slices over duration
- Simple and predictable execution

```python
vwap_plan = algo.vwap_execution_strategy(order, historical_volume)
twap_plan = algo.twap_execution_strategy(order, duration_minutes=60, num_slices=5)
```

---

### Feature 11: PerformanceAttribution (Monitoring)

**Purpose:** Analyze P&L by stock, regime, signal type, and time period.

**Class:** `PerformanceAttribution`

**Attribution Dimensions:**
- By ticker
- By regime
- By signal type
- By time period

**Output:**
```python
{
    'by_ticker': {'0700.HK': {'pnl': 500, 'trades': 10, 'win_rate': 0.6}},
    'by_regime': {'bull': {'pnl': 300, 'trades': 5}},
    'total_pnl': 500,
    'total_trades': 10
}
```

---

### Feature 12: StrategyDriftDetector (Monitoring)

**Purpose:** Detect when strategy behavior deviates significantly from baseline.

**Class:** `StrategyDriftDetector`

**Monitored Metrics:**
| Metric | Drift Threshold |
|--------|-----------------|
| Win Rate | 10% deviation |
| Avg Holding Period | 50% deviation |
| Signals per Day | 50% deviation |

**Key Methods:**
```python
def set_baseline(self, baseline_metrics: Dict)
def check_for_drift(self, recent_trades: List[Dict]) -> Tuple[bool, Dict]
```

---

### Feature 13: DrawdownPredictor (Predictive)

**Purpose:** Forecast potential drawdown risk using stress scenarios.

**Class:** `DrawdownPredictor`

**Stress Scenarios:**
| Scenario | Market Drop | Probability |
|----------|-------------|-------------|
| Normal Correction | -5% | 30% |
| Sector Rotation | -8% | 20% |
| Market Crash | -15% | 10% |
| Black Swan | -25% | 5% |

**Output:**
```python
{
    'scenarios': [...],
    'expected_drawdown': 0.08,
    'worst_case_drawdown': 0.25,
    'risk_level': 'MEDIUM'
}
```

---

### Feature 14: LiquidityRiskMonitor (Predictive)

**Purpose:** Assess liquidity risk for current positions.

**Class:** `LiquidityRiskMonitor`

**Risk Classification:**
| Position/Volume Ratio | Risk Level |
|----------------------|------------|
| < 1% | LOW |
| 1-5% | MEDIUM |
| > 5% | HIGH |

**Days to Liquidate Calculation:**
```python
days_to_liquidate = position_value / (avg_volume * 0.1)  # 10% participation rate
```

---

### Feature 15: ContingencyManager (Production)

**Purpose:** Define and execute contingency plans for various risk scenarios.

**Class:** `ContingencyManager`

**Contingency Triggers:**

| Scenario | Trigger | Actions |
|----------|---------|---------|
| Low Win Rate | < 40% | Reduce position sizes, increase confidence threshold |
| High Drawdown | > 10% | Halt new trades, close positions |
| System Drift | Detected | Pause trading, review parameters |

---

## Stress Protection Features (16-20)

### Feature 16: StressScenarioProtection

**Purpose:** Monitor market stress indicators and activate protective measures.

**Class:** `StressScenarioProtection`

**VIX-Based Stress Levels:**

| VIX Level | Stress Level | Position Multiplier | Max Daily Trades |
|-----------|--------------|---------------------|------------------|
| 0-20 | NORMAL | 100% | 8 |
| 20-25 | ALERT | 75% | 5 |
| 25-30 | WARNING | 50% | 3 |
| 30-35 | DANGER | 25% | 1 |
| 35+ | HALT | 0% | 0 |

**Key Methods:**
```python
def update_vix(self, current_vix: float) -> Dict
def get_position_limits(self) -> Dict
```

---

### Feature 17: FlashCrashDetector

**Purpose:** Detect flash crashes using price drop and volume spike indicators.

**Class:** `FlashCrashDetector`

**Detection Criteria:**
| Indicator | Threshold |
|-----------|-----------|
| Price Drop | > 8% |
| Volume Spike | > 5x average |

**Response Actions:**
```python
{
    'flash_crash': True,
    'price_drop': -0.12,
    'volume_ratio': 10.0,
    'response': {
        'action': 'EMERGENCY_HALT',
        'close_positions': True,
        'block_new_trades': True
    }
}
```

---

### Feature 18: BlackSwanPreparer

**Purpose:** Detect and respond to black swan events (extreme market moves).

**Class:** `BlackSwanPreparer`

**Black Swan Criteria:**
| Condition | Threshold |
|-----------|-----------|
| Market Return | < -15% |
| VIX Level | > 40 |
| Both must be true | AND condition |

**Exposure Score Calculation:**
```python
def calculate_exposure_score(self, positions: Dict) -> float:
    # Returns 0.0 (no exposure) to 1.0 (maximum exposure)
    # Based on total position value and concentration
```

**Defense Protocol:**
1. Halt all new positions
2. Queue emergency liquidation
3. Activate circuit breaker
4. Alert risk management

---

### Feature 19: EmergencyLiquidation

**Purpose:** Create and execute emergency liquidation plans prioritized by liquidity.

**Class:** `EmergencyLiquidation`

**Liquidation Priority:**
1. Highest liquidity positions first (can exit quickly)
2. Proportional to position value
3. Market orders for urgency='HIGH'

**Key Methods:**
```python
def create_liquidation_plan(
    self,
    positions: Dict,
    volumes: Dict,
    urgency: str = 'MEDIUM'
) -> Dict

def execute_emergency_close(self, position: Dict) -> Dict
```

**Plan Output:**
```python
{
    'orders': [
        {'ticker': '0700.HK', 'quantity': 100, 'order_type': 'MARKET', 'priority': 1},
        {'ticker': '9988.HK', 'quantity': 200, 'order_type': 'MARKET', 'priority': 2}
    ],
    'total_value': 80000,
    'estimated_time': '5 minutes'
}
```

---

### Feature 20: StressHardenedTradingSystem

**Purpose:** Integrated stress-hardened trading system combining all protection features.

**Class:** `StressHardenedTradingSystem`

**Components:**
- StressScenarioProtection
- FlashCrashDetector
- BlackSwanPreparer
- EmergencyLiquidation

**Stress Position Sizing Table:**

| VIX Range | Base Position | High Correlation | Drawdown > 5% | Drawdown > 8% |
|-----------|---------------|------------------|---------------|---------------|
| 0-20 | 12% | 8% | 6% | 4% |
| 20-25 | 8% | 6% | 4% | 2% |
| 25-30 | 4% | 3% | 2% | 1% |
| 30-35 | 2% | 1% | 1% | 0% |
| 35+ | 0% | 0% | 0% | 0% |

**Key Methods:**
```python
def update_market_conditions(self, vix: float, market_return: float) -> Dict
def check_trade_allowed(self) -> Tuple[bool, str]
def get_adjusted_position_size(self, base_size: float, portfolio_drawdown: float) -> float
def get_status_report(self) -> Dict
```

**Status Report:**
```python
{
    'system_status': 'NORMAL',  # NORMAL, REDUCED, HALTED
    'stress_level': 'WARNING',
    'position_multiplier': 0.50,
    'circuit_breaker_active': False,
    'flash_crash_detected': False,
    'black_swan_detected': False,
    'recommended_actions': [...]
}
```

---

## Production Configuration

### ProductionConfig Dataclass

**Default Settings:**
```python
@dataclass
class ProductionConfig:
    min_win_rate: float = 0.40
    min_trades_for_ban: int = 8
    confidence_threshold: float = 0.50
    max_position_size: float = 0.12
    oos_position_discount: float = 0.70
```

### Deployment Week Settings

| Week | Max Daily Trades | Max Position | Confidence |
|------|------------------|--------------|------------|
| 1 | 3 | 8% | 60% |
| 2 | 5 | 10% | 55% |
| 3+ | 8 | 12% | 50% |

### Expected Performance Metrics

| Metric | Expected Range |
|--------|----------------|
| Win Rate | 60-65% |
| Trades/Period | 40-60 |
| Max Drawdown | 8-12% |
| Sharpe Ratio | 1.5-2.2 |
| Profit Factor | 1.6-2.5 |
| Recovery Time | 14-30 days |

### 30-Day Validation Criteria

| Metric | Minimum Requirement |
|--------|---------------------|
| Win Rate | 55% |
| Max Drawdown | 15% |
| Sharpe Ratio | 1.2 |
| Auto-Ban Effectiveness | 80% |
| System Uptime | 95% |

---

## Test Results Summary

### All Features Test (20/20 PASS)
```
[1] AdaptiveFeatureSelector......PASS
[2] MetaLearner................PASS
[3] CorrelationAwarePositionSizer.PASS
[4] VolatilityScaler...........PASS
[5] RiskParityAllocator........PASS
[6] SmartRebalancer............PASS
[7] CrossAssetSignalEnhancer...PASS
[8] RegimeSignalWeighter.......PASS
[9] SmartOrderRouter...........PASS
[10] ExecutionAlgorithms.......PASS
[11] PerformanceAttribution....PASS
[12] StrategyDriftDetector.....PASS
[13] DrawdownPredictor.........PASS
[14] LiquidityRiskMonitor......PASS
[15] ContingencyManager........PASS
[16] StressScenarioProtection..PASS
[17] FlashCrashDetector........PASS
[18] BlackSwanPreparer.........PASS
[19] EmergencyLiquidation......PASS
[20] StressHardenedTradingSystem.PASS
```

### Stress Scenario Tests (9/9 PASS)
```
[PASS] test_stress_indicators_monitoring
[PASS] test_flash_crash_detection
[PASS] test_black_swan_exposure_calculation
[PASS] test_stress_aware_position_sizing
[PASS] test_emergency_liquidation_prioritization
[PASS] test_stress_position_limits_table
[PASS] test_circuit_breaker_activation
[PASS] test_complete_stress_scenario
[PASS] test_flash_crash_emergency_response
```

---

## Usage Example

```python
from src.trading.production_advanced import *

# Initialize stress-hardened system
system = StressHardenedTradingSystem()

# Update market conditions
status = system.update_market_conditions(vix=25, market_return=-0.02)
print(f"System: {status['system_status']}, Stress: {status['stress_level']}")

# Check if trading allowed
can_trade, reason = system.check_trade_allowed()
if not can_trade:
    print(f"Trading blocked: {reason}")

# Get adjusted position size
base_size = 0.10  # 10% position
adjusted = system.get_adjusted_position_size(base_size, portfolio_drawdown=0.03)
print(f"Adjusted position: {adjusted:.2%}")

# Get full status report
report = system.get_status_report()
```

---

## Architecture Diagram

```
                    +---------------------------+
                    |   StressHardenedTradingSystem   |
                    +---------------------------+
                              |
         +--------------------+--------------------+
         |                    |                    |
+----------------+  +------------------+  +-------------------+
| StressScenario |  | FlashCrash       |  | BlackSwan         |
| Protection     |  | Detector         |  | Preparer          |
+----------------+  +------------------+  +-------------------+
         |                    |                    |
         +--------------------+--------------------+
                              |
                    +-------------------+
                    | Emergency         |
                    | Liquidation       |
                    +-------------------+
```

---

## File References

| File | Line | Description |
|------|------|-------------|
| `production_advanced.py` | 1-106 | Production configuration |
| `production_advanced.py` | 107-181 | Feature 1: AdaptiveFeatureSelector |
| `production_advanced.py` | 183-265 | Feature 2: MetaLearner |
| `production_advanced.py` | 268-393 | Feature 3: CorrelationAwarePositionSizer |
| `production_advanced.py` | 396-485 | Feature 4: VolatilityScaler |
| `production_advanced.py` | 488-561 | Feature 5: RiskParityAllocator |
| `production_advanced.py` | 564-653 | Feature 6: SmartRebalancer |
| `production_advanced.py` | 656-755 | Feature 7: CrossAssetSignalEnhancer |
| `production_advanced.py` | 758-850 | Feature 8: RegimeSignalWeighter |
| `production_advanced.py` | 853-920 | Feature 9: SmartOrderRouter |
| `production_advanced.py` | 923-1000 | Feature 10: ExecutionAlgorithms |
| `production_advanced.py` | 1003-1080 | Feature 11: PerformanceAttribution |
| `production_advanced.py` | 1083-1150 | Feature 12: StrategyDriftDetector |
| `production_advanced.py` | 1153-1230 | Feature 13: DrawdownPredictor |
| `production_advanced.py` | 1233-1300 | Feature 14: LiquidityRiskMonitor |
| `production_advanced.py` | 1303-1380 | Feature 15: ContingencyManager |
| `production_advanced.py` | 1383-1500 | Feature 16-20: Stress Protection |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-11 | Initial 15 core features |
| 1.1 | 2024-11 | Added 5 stress protection features (16-20) |
| 1.2 | 2024-11 | Full test coverage (29/29 tests passing) |

---

## References

- `phase1 fixing on C model_extra 4.pdf` - Core feature specifications
- `phase1 fixing on C model 5.pdf` - Stress protection specifications
- `phase1 fixing test on C model.pdf` - Test suite specifications
