# Phase 6 Integration Guide

## Overview

Phase 6 consists of three modules that work together to provide complete portfolio optimization:

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 6 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  phase6_improvements.py      phase6_final_improvements.py   │
│  (Basic Optimization)        (Advanced Optimization)         │
│         │                            │                       │
│         └──────────┬─────────────────┘                       │
│                    ▼                                         │
│           phase6_integration.py                              │
│           (Unified System)                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Module 1: Basic Optimization (`phase6_improvements.py`)

**Purpose**: Core risk and execution management

| Class | Function |
|-------|----------|
| `LiquidityAdjustedRisk` | Adjusts risk metrics based on asset liquidity |
| `ExpectedShortfallCalculator` | Calculates CVaR/ES for tail risk |
| `AdaptiveExecutionSelector` | Chooses optimal execution strategy |

```python
from portfolio.phase6_improvements import (
    LiquidityAdjustedRisk,
    ExpectedShortfallCalculator,
    AdaptiveExecutionSelector
)
```

---

## Module 2: Advanced Optimization (`phase6_final_improvements.py`)

**Purpose**: Sophisticated portfolio analytics and stress testing

| Class | Function |
|-------|----------|
| `PortfolioOptimizationMonitor` | Tracks optimization performance, triggers recalibration |
| `MultiTimeframeOptimizer` | Blends daily/weekly/monthly optimizations |
| `AdaptiveConstraintManager` | Adjusts constraints based on market regime |
| `CrossPortfolioCorrelationManager` | Manages strategy correlations |
| `PortfolioStressTester` | Tests against crisis scenarios (2008, COVID, etc.) |

```python
from portfolio.phase6_final_improvements import (
    PortfolioOptimizationMonitor,
    MultiTimeframeOptimizer,
    AdaptiveConstraintManager,
    CrossPortfolioCorrelationManager,
    PortfolioStressTester
)
```

---

## Module 3: Integration (`phase6_integration.py`)

**Purpose**: Unifies all components into production-ready system

| Class | Function |
|-------|----------|
| `LiquidityAwareRebalancer` | Rebalances considering liquidity constraints |
| `MultiObjectiveOptimizer` | Balances return, risk, and other objectives |
| `ComplianceManager` | Ensures regulatory compliance |

```python
from portfolio.phase6_integration import (
    LiquidityAwareRebalancer,
    MultiObjectiveOptimizer,
    ComplianceManager
)
```

---

## Data Flow

```
Input Data
    │
    ▼
┌─────────────────────────┐
│ LiquidityAdjustedRisk   │ ──► Risk metrics adjusted for liquidity
│ ExpectedShortfallCalculator │ ──► Tail risk (CVaR)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ AdaptiveConstraintManager │ ──► Regime-based constraints
│ MultiTimeframeOptimizer   │ ──► Blended timeframe weights
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ CrossPortfolioCorrelationManager │ ──► Strategy allocation
│ PortfolioStressTester            │ ──► Stress test validation
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ MultiObjectiveOptimizer │ ──► Final optimized weights
│ ComplianceManager       │ ──► Compliance check
│ LiquidityAwareRebalancer│ ──► Execution plan
└─────────────────────────┘
    │
    ▼
Output: Optimized Portfolio
```

---

## Quick Start Example

```python
import numpy as np
from portfolio.phase6_improvements import LiquidityAdjustedRisk
from portfolio.phase6_final_improvements import (
    AdaptiveConstraintManager,
    PortfolioStressTester,
    PortfolioOptimizationMonitor
)
from portfolio.phase6_integration import MultiObjectiveOptimizer

# 1. Initialize components
risk_calc = LiquidityAdjustedRisk()
constraint_mgr = AdaptiveConstraintManager()
stress_tester = PortfolioStressTester()
monitor = PortfolioOptimizationMonitor()

# 2. Get regime-aware constraints
constraints = constraint_mgr.get_regime_aware_constraints(
    regime='normal',
    portfolio_size=10
)

# 3. Stress test portfolio
weights = np.array([0.1] * 10)
covariance = np.eye(10) * 0.04
stress_result = stress_tester.stress_test_portfolio(
    weights=weights,
    covariance_matrix=covariance,
    scenario='2008_lehman'
)

# 4. Track performance
monitor.track_optimization_performance(
    predicted_return=0.08,
    actual_return=0.075,
    predicted_risk=0.15,
    actual_risk=0.16
)
```

---

## Market Regimes

| Regime | Max Position | Max Turnover | Description |
|--------|-------------|--------------|-------------|
| `crisis` | 10% | 15% | Defensive, tight limits |
| `high_vol` | 15% | 20% | Cautious positioning |
| `normal` | 25% | 30% | Standard operation |
| `low_vol` | 30% | 40% | Aggressive allowed |

---

## Stress Scenarios

| Scenario | Volatility Multiplier | Correlation Boost |
|----------|----------------------|-------------------|
| `2008_lehman` | 3.0x | +0.4 |
| `2020_covid` | 2.5x | +0.3 |
| `inflation_shock` | 2.0x | +0.2 |

---

## File Locations

```
src/portfolio/
├── phase6_improvements.py       # Basic optimization
├── phase6_final_improvements.py # Advanced optimization
└── phase6_integration.py        # Unified system
```

---

## Test Coverage

All 30 Phase 6 tests pass:
- `TestPortfolioOptimizationMonitor` (6 tests)
- `TestMultiTimeframeOptimizer` (4 tests)
- `TestAdaptiveConstraintManager` (6 tests)
- `TestCrossPortfolioCorrelationManager` (4 tests)
- `TestPortfolioStressTester` (4 tests)
- `TestIntegrationScenariosAdvanced` (2 tests)
- `TestEdgeCasesAdvanced` (4 tests)

Run tests:
```bash
pytest tests/test_phase6_advanced_improvements.py -v
```
