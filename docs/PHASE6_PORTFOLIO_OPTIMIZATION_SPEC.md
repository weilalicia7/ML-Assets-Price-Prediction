# Phase 6: Portfolio Optimization - Detailed Specification

## Overview

Phase 6 focuses on portfolio-level optimization to maximize risk-adjusted returns while managing constraints. Building on Phase 5's dynamic weighting foundation, Phase 6 adds sophisticated portfolio construction, rebalancing, and compliance capabilities.

**Expected Improvement: +2-4% profit rate**
**Total Expected After Phase 6: +21-36% from baseline**

---

## 1. Cross-Portfolio Risk Budgeting

### 1.1 Purpose
Allocate risk budget across the entire portfolio rather than treating each position independently. This ensures optimal diversification and prevents risk concentration.

### 1.2 Components

#### 1.2.1 RiskBudgetAllocator
```python
class RiskBudgetAllocator:
    """
    Allocates total portfolio risk budget across positions.

    Methods:
    - allocate_risk_budget(positions, total_risk_budget, method='risk_parity')
    - calculate_marginal_risk_contribution(position, portfolio)
    - optimize_risk_allocation(target_return, max_risk)
    """
```

**Risk Allocation Methods:**
- **Risk Parity**: Equal risk contribution from each position
- **Inverse Volatility**: Weight inversely proportional to volatility
- **Maximum Diversification**: Maximize diversification ratio
- **Minimum Variance**: Minimize total portfolio variance

#### 1.2.2 MarginalRiskCalculator
```python
class MarginalRiskCalculator:
    """
    Calculates how each position contributes to total portfolio risk.

    Methods:
    - calculate_marginal_var(position, portfolio_returns)
    - calculate_component_var(positions, covariance_matrix)
    - calculate_risk_contribution_percent(position, portfolio)
    """
```

**Formulas:**
```
Marginal VaR = ∂VaR/∂w_i = (Σw)_i / σ_p * VaR_p
Component VaR = w_i * Marginal VaR_i
Risk Contribution % = Component VaR_i / Total VaR
```

#### 1.2.3 CorrelationAwareAllocator
```python
class CorrelationAwareAllocator:
    """
    Adjusts allocations based on correlation structure.

    Methods:
    - calculate_diversification_benefit(positions, correlations)
    - find_uncorrelated_opportunities(universe, current_portfolio)
    - adjust_for_correlation_regime(allocations, regime)
    """
```

**Correlation Regimes:**
| Regime | Correlation Threshold | Allocation Adjustment |
|--------|----------------------|----------------------|
| Low Correlation | < 0.3 | Standard allocation |
| Moderate | 0.3 - 0.6 | Reduce concentrated positions |
| High Correlation | > 0.6 | Aggressive diversification |
| Crisis | > 0.8 | Defensive positioning |

### 1.3 Configuration
```python
RISK_BUDGET_CONFIG = {
    'total_portfolio_var_limit': 0.15,      # 15% max portfolio VaR
    'max_single_position_risk': 0.25,       # 25% of total risk budget
    'min_diversification_ratio': 1.2,       # Minimum diversification benefit
    'correlation_lookback_days': 60,        # Days for correlation calculation
    'risk_parity_tolerance': 0.05,          # 5% deviation from equal risk
    'reallocation_threshold': 0.10,         # 10% drift triggers reallocation
}
```

### 1.4 Expected Impact: +0.5-1% profit rate

---

## 2. Liquidity-Aware Rebalancing

### 2.1 Purpose
Minimize market impact and slippage during portfolio rebalancing by considering liquidity constraints and optimal execution strategies.

### 2.2 Components

#### 2.2.1 LiquidityAwareRebalancer
```python
class LiquidityAwareRebalancer:
    """
    Rebalances portfolio while minimizing market impact.

    Methods:
    - calculate_rebalance_trades(current, target, liquidity_data)
    - estimate_market_impact(trade_size, avg_volume, volatility)
    - optimize_execution_schedule(trades, urgency, market_conditions)
    - should_rebalance(current, target, threshold, costs)
    """
```

**Market Impact Model (Square-Root Model):**
```
Impact = σ * sign(Q) * sqrt(|Q| / V) * k
Where:
  σ = daily volatility
  Q = order quantity
  V = average daily volume
  k = impact coefficient (typically 0.1-0.5)
```

#### 2.2.2 RebalanceTrigger
```python
class RebalanceTrigger:
    """
    Determines when to trigger portfolio rebalancing.

    Methods:
    - check_drift_threshold(current, target, threshold)
    - check_time_based(last_rebalance, frequency)
    - check_event_based(market_events, regime_changes)
    - combined_trigger(current, target, config)
    """
```

**Trigger Conditions:**
| Trigger Type | Condition | Priority |
|--------------|-----------|----------|
| Drift-Based | Position drift > 5% | High |
| Time-Based | Weekly/Monthly | Medium |
| Regime Change | Market regime shift | High |
| Risk Breach | Risk limit exceeded | Critical |
| Opportunity | New signal strength > 0.8 | Medium |

#### 2.2.3 ExecutionOptimizer
```python
class ExecutionOptimizer:
    """
    Optimizes trade execution to minimize costs.

    Methods:
    - calculate_optimal_trade_size(position, volume, urgency)
    - schedule_trades_twap(total_quantity, duration, intervals)
    - schedule_trades_vwap(total_quantity, volume_profile)
    - adaptive_execution(trades, real_time_data)
    """
```

**Execution Strategies:**
- **TWAP** (Time-Weighted Average Price): Equal slices over time
- **VWAP** (Volume-Weighted Average Price): Match volume profile
- **Implementation Shortfall**: Minimize deviation from decision price
- **Adaptive**: Adjust based on real-time market conditions

#### 2.2.4 SlippageTracker
```python
class SlippageTracker:
    """
    Tracks and analyzes execution slippage.

    Methods:
    - record_execution(expected_price, actual_price, quantity)
    - calculate_slippage_stats(lookback_period)
    - update_impact_model(recent_executions)
    - get_slippage_forecast(trade_size, asset)
    """
```

### 2.3 Configuration
```python
REBALANCING_CONFIG = {
    'drift_threshold': 0.05,                # 5% position drift
    'min_rebalance_interval_hours': 24,     # Minimum 24h between rebalances
    'max_daily_turnover': 0.20,             # 20% max daily turnover
    'market_impact_coefficient': 0.3,       # Impact model k parameter
    'max_single_trade_volume_pct': 0.05,    # 5% of daily volume per trade
    'execution_urgency_levels': {
        'low': 0.01,                        # Can wait days
        'medium': 0.1,                      # Within hours
        'high': 0.5,                        # Within minutes
        'critical': 1.0                     # Immediate
    }
}
```

### 2.4 Expected Impact: +0.5-1% profit rate

---

## 3. Tax-Efficient Optimization

### 3.1 Purpose
Maximize after-tax returns through tax-loss harvesting, capital gains management, and hold period optimization.

### 3.2 Components

#### 3.2.1 TaxLotManager
```python
class TaxLotManager:
    """
    Tracks tax lots for each position.

    Methods:
    - add_lot(ticker, quantity, cost_basis, purchase_date)
    - get_lots(ticker, sort_by='fifo')
    - calculate_unrealized_gains(ticker, current_price)
    - get_short_term_lots(ticker)
    - get_long_term_lots(ticker)
    """
```

**Tax Lot Data Structure:**
```python
@dataclass
class TaxLot:
    ticker: str
    quantity: float
    cost_basis: float           # Per share
    purchase_date: datetime
    is_long_term: bool          # > 1 year holding
    unrealized_gain: float
    wash_sale_disallowed: float # Disallowed loss amount
```

#### 3.2.2 TaxLossHarvester
```python
class TaxLossHarvester:
    """
    Identifies and executes tax-loss harvesting opportunities.

    Methods:
    - find_harvesting_opportunities(portfolio, min_loss_threshold)
    - check_wash_sale_risk(ticker, recent_trades)
    - calculate_tax_benefit(loss_amount, tax_rate)
    - execute_harvest(ticker, lots, replacement_ticker)
    - track_wash_sale_window(ticker)
    """
```

**Wash Sale Rules:**
- 30-day window before and after sale
- Cannot repurchase substantially identical security
- Track related securities (ETFs containing the stock)

**Harvesting Thresholds:**
| Loss Size | Action | Replacement Strategy |
|-----------|--------|---------------------|
| < 1% | Monitor only | N/A |
| 1-3% | Harvest if no wash sale risk | Similar sector ETF |
| 3-5% | Harvest with priority | Correlated asset |
| > 5% | Immediate harvest | Any suitable replacement |

#### 3.2.3 CapitalGainsOptimizer
```python
class CapitalGainsOptimizer:
    """
    Optimizes capital gains recognition.

    Methods:
    - select_lots_for_sale(ticker, quantity, strategy)
    - estimate_tax_liability(gains, tax_rates)
    - optimize_gain_timing(portfolio, tax_situation)
    - balance_short_long_term(target_ratio)
    """
```

**Lot Selection Strategies:**
- **FIFO** (First In, First Out): Default method
- **LIFO** (Last In, First Out): Minimize short-term gains
- **Highest Cost**: Minimize realized gains
- **Specific Identification**: Optimize based on tax situation
- **Tax-Optimized**: Balance ST/LT gains with losses

#### 3.2.4 HoldPeriodOptimizer
```python
class HoldPeriodOptimizer:
    """
    Optimizes decisions based on tax holding periods.

    Methods:
    - days_to_long_term(lot)
    - should_defer_sale(lot, urgency, expected_tax_savings)
    - calculate_breakeven_return(lot, tax_rates)
    - recommend_action(lot, signal_strength, tax_impact)
    """
```

**Hold Period Decision Matrix:**
| Days to Long-Term | Signal Strength | Recommendation |
|-------------------|-----------------|----------------|
| < 30 | Weak Sell | Hold for LT |
| < 30 | Strong Sell | Sell (accept ST) |
| 30-90 | Weak Sell | Hold for LT |
| 30-90 | Strong Sell | Evaluate breakeven |
| > 90 | Any Sell | Sell immediately |

### 3.3 Configuration
```python
TAX_CONFIG = {
    'short_term_rate': 0.35,               # 35% short-term capital gains
    'long_term_rate': 0.15,                # 15% long-term capital gains
    'loss_harvesting_threshold': 0.03,     # 3% minimum loss to harvest
    'wash_sale_window_days': 30,           # 30 days before/after
    'min_tax_benefit': 100,                # $100 minimum benefit
    'long_term_holding_days': 365,         # 1 year for long-term
    'defer_sale_threshold_days': 30,       # Defer if < 30 days to LT
    'replacement_correlation_min': 0.7,    # Min correlation for replacement
}
```

### 3.4 Expected Impact: +1-2% net returns

---

## 4. Multi-Objective Optimization

### 4.1 Purpose
Balance multiple competing objectives (returns, risk, costs, taxes) to find Pareto-optimal portfolio allocations.

### 4.2 Components

#### 4.2.1 MultiObjectiveOptimizer
```python
class MultiObjectiveOptimizer:
    """
    Optimizes portfolio across multiple objectives.

    Methods:
    - optimize(objectives, constraints, method='pareto')
    - calculate_pareto_frontier(portfolios)
    - select_portfolio(frontier, preference_weights)
    - sensitivity_analysis(optimal_portfolio, parameters)
    """
```

**Objectives:**
```python
OPTIMIZATION_OBJECTIVES = {
    'maximize_return': {
        'weight': 0.35,
        'function': 'expected_return',
        'direction': 'maximize'
    },
    'minimize_risk': {
        'weight': 0.30,
        'function': 'portfolio_volatility',
        'direction': 'minimize'
    },
    'minimize_costs': {
        'weight': 0.15,
        'function': 'transaction_costs + tax_impact',
        'direction': 'minimize'
    },
    'maximize_sharpe': {
        'weight': 0.20,
        'function': 'sharpe_ratio',
        'direction': 'maximize'
    }
}
```

#### 4.2.2 ConstraintManager
```python
class ConstraintManager:
    """
    Manages portfolio constraints.

    Methods:
    - add_constraint(name, type, value)
    - check_constraints(portfolio)
    - get_binding_constraints(portfolio)
    - relax_constraints(priority_order)
    """
```

**Constraint Types:**
| Constraint | Type | Default Value |
|------------|------|---------------|
| Max Position Size | Upper Bound | 20% |
| Min Position Size | Lower Bound | 2% |
| Max Sector Exposure | Upper Bound | 40% |
| Max Total Leverage | Upper Bound | 1.0 (no leverage) |
| Min Cash Buffer | Lower Bound | 5% |
| Max Correlation | Upper Bound | 0.7 |
| Max Turnover | Upper Bound | 50% monthly |

#### 4.2.3 ObjectiveWeightOptimizer
```python
class ObjectiveWeightOptimizer:
    """
    Dynamically adjusts objective weights based on market conditions.

    Methods:
    - get_regime_adjusted_weights(current_regime)
    - update_weights_from_performance(recent_results)
    - calculate_optimal_weights(historical_data)
    """
```

**Regime-Based Weight Adjustments:**
| Regime | Return Weight | Risk Weight | Cost Weight |
|--------|--------------|-------------|-------------|
| Bull | 0.40 | 0.25 | 0.15 |
| Normal | 0.35 | 0.30 | 0.15 |
| Bear | 0.25 | 0.40 | 0.15 |
| Crisis | 0.15 | 0.50 | 0.15 |

#### 4.2.4 UtilityFunction
```python
class UtilityFunction:
    """
    Combines objectives into single utility score.

    Methods:
    - calculate_utility(portfolio, weights)
    - calculate_certainty_equivalent(expected_return, risk, risk_aversion)
    - get_risk_aversion_parameter(investor_profile)
    """
```

**Utility Function:**
```
U(w) = E[R_p] - (λ/2) * σ²_p - TC(w) - Tax(w)

Where:
  E[R_p] = Expected portfolio return
  σ²_p = Portfolio variance
  λ = Risk aversion parameter (typically 2-4)
  TC(w) = Transaction costs
  Tax(w) = Tax impact
```

### 4.3 Optimization Methods

#### 4.3.1 Mean-Variance Optimization
```python
def mean_variance_optimize(expected_returns, covariance, risk_aversion):
    """Classic Markowitz optimization."""
    # w* = (1/λ) * Σ^(-1) * μ
    pass
```

#### 4.3.2 Black-Litterman Model
```python
def black_litterman_optimize(market_weights, views, confidence):
    """Incorporate views into market equilibrium."""
    # Combines market equilibrium with investor views
    pass
```

#### 4.3.3 Risk Parity
```python
def risk_parity_optimize(covariance, risk_budget=None):
    """Equal risk contribution from each asset."""
    # σ_i * w_i * (Σw)_i / σ_p = 1/n for all i
    pass
```

#### 4.3.4 Hierarchical Risk Parity (HRP)
```python
def hrp_optimize(returns):
    """Machine learning based allocation."""
    # Uses clustering and recursive bisection
    pass
```

### 4.4 Configuration
```python
OPTIMIZATION_CONFIG = {
    'risk_aversion': 2.5,                  # Lambda parameter
    'optimization_method': 'black_litterman',
    'reoptimize_frequency_days': 5,        # Re-optimize every 5 days
    'min_weight_change': 0.02,             # 2% minimum change to act
    'max_iterations': 1000,                # Optimizer iterations
    'convergence_tolerance': 1e-6,
    'use_robust_covariance': True,         # Shrinkage estimator
    'covariance_shrinkage': 0.2,           # Ledoit-Wolf shrinkage
}
```

### 4.5 Expected Impact: +0.5-1% profit rate

---

## 5. Regulatory Compliance Manager

### 5.1 Purpose
Ensure portfolio adheres to regulatory requirements, position limits, and internal policies.

### 5.2 Components

#### 5.2.1 PositionLimitManager
```python
class PositionLimitManager:
    """
    Manages position limits and concentration.

    Methods:
    - check_position_limits(portfolio)
    - get_remaining_capacity(ticker)
    - calculate_concentration_risk(portfolio)
    - enforce_limits(proposed_trade)
    """
```

**Position Limits:**
| Limit Type | Threshold | Action |
|------------|-----------|--------|
| Single Position | 20% of portfolio | Block trade |
| Sector Concentration | 40% of portfolio | Warning + reduce |
| Market Cap Category | 50% of portfolio | Advisory |
| Geographic Region | 60% of portfolio | Advisory |
| Liquidity Tier | Based on tier | Sliding scale |

#### 5.2.2 ConcentrationMonitor
```python
class ConcentrationMonitor:
    """
    Monitors portfolio concentration metrics.

    Methods:
    - calculate_herfindahl_index(weights)
    - calculate_effective_n(weights)
    - check_concentration_limits(portfolio)
    - suggest_diversification(portfolio)
    """
```

**Concentration Metrics:**
```
Herfindahl-Hirschman Index (HHI) = Σ(w_i)²
Effective N = 1 / HHI
Diversification Ratio = (Σ w_i * σ_i) / σ_p
```

**HHI Thresholds:**
| HHI Range | Concentration Level | Action |
|-----------|---------------------|--------|
| < 0.10 | Well Diversified | None |
| 0.10 - 0.18 | Moderate | Monitor |
| 0.18 - 0.25 | High | Warning |
| > 0.25 | Very High | Reduce |

#### 5.2.3 ComplianceChecker
```python
class ComplianceChecker:
    """
    Checks compliance with various rules.

    Methods:
    - pre_trade_check(proposed_trade, portfolio)
    - post_trade_validation(executed_trade, portfolio)
    - generate_compliance_report(portfolio, period)
    - check_restricted_list(ticker)
    """
```

**Pre-Trade Checks:**
1. Position limit check
2. Sector concentration check
3. Restricted list check
4. Liquidity adequacy check
5. Risk budget availability
6. Wash sale window check

#### 5.2.4 AuditTrailManager
```python
class AuditTrailManager:
    """
    Maintains audit trail for compliance.

    Methods:
    - log_decision(decision_type, details, rationale)
    - log_trade(trade_details, compliance_checks)
    - generate_audit_report(start_date, end_date)
    - export_for_regulatory(format, period)
    """
```

**Audit Log Entry:**
```python
@dataclass
class AuditEntry:
    timestamp: datetime
    action_type: str           # 'trade', 'rebalance', 'harvest'
    ticker: str
    quantity: float
    rationale: str             # Signal strength, regime, etc.
    compliance_checks: Dict    # All checks performed
    approved_by: str           # System or manual
    risk_metrics: Dict         # Portfolio risk at time of action
```

### 5.3 Configuration
```python
COMPLIANCE_CONFIG = {
    'max_single_position_pct': 0.20,       # 20% max single position
    'max_sector_concentration_pct': 0.40,  # 40% max sector
    'max_hhi': 0.25,                        # Maximum HHI allowed
    'min_effective_n': 5,                   # Minimum effective positions
    'restricted_list': [],                  # Tickers not allowed
    'min_liquidity_score': 0.3,            # Minimum liquidity for trading
    'audit_retention_days': 365 * 7,       # 7 years retention
    'compliance_check_frequency': 'pre_trade',
}
```

### 5.4 Expected Impact: Risk reduction, regulatory safety

---

## 6. Phase 6 Integration System

### 6.1 Unified Architecture

```python
class Phase6PortfolioOptimizationSystem:
    """
    Integrates all Phase 6 components into unified system.

    Components:
    - risk_budget_allocator: RiskBudgetAllocator
    - liquidity_rebalancer: LiquidityAwareRebalancer
    - tax_optimizer: TaxOptimizer
    - multi_objective_optimizer: MultiObjectiveOptimizer
    - compliance_manager: ComplianceManager

    Methods:
    - optimize_portfolio(current_portfolio, signals, market_data)
    - execute_rebalance(current, target, constraints)
    - get_portfolio_status()
    - generate_reports()
    """
```

### 6.2 Decision Flow

```
                    ┌─────────────────────────┐
                    │   Portfolio Signals     │
                    │   (from Phase 5)        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Compliance Pre-Check   │
                    │  (restricted list,      │
                    │   position limits)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Multi-Objective        │
                    │  Optimization           │
                    │  (return, risk, cost)   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Risk Budget Check      │
                    │  (marginal risk,        │
                    │   correlation)          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Tax Optimization       │
                    │  (lot selection,        │
                    │   harvest check)        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Liquidity-Aware        │
                    │  Execution Planning     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Final Compliance       │
                    │  Validation             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Execute & Log          │
                    └─────────────────────────┘
```

### 6.3 Integration with Phase 5

```python
def integrate_phase6_with_phase5(phase5_system, phase6_system):
    """
    Connect Phase 6 portfolio optimization with Phase 5 signals.

    Phase 5 provides:
    - Trading signals with confidence
    - Position sizing recommendations
    - Risk regime detection
    - Correlation regime detection

    Phase 6 adds:
    - Portfolio-level optimization
    - Cross-asset risk budgeting
    - Tax-aware execution
    - Compliance validation
    """
    pass
```

### 6.4 Factory Function

```python
def create_phase6_system(
    config: Optional[Dict] = None,
    phase5_system: Optional[Phase5System] = None
) -> Phase6PortfolioOptimizationSystem:
    """
    Create fully configured Phase 6 system.

    Args:
        config: Override default configuration
        phase5_system: Connect to existing Phase 5 system

    Returns:
        Configured Phase 6 system ready for use
    """
    pass
```

---

## 7. Testing Requirements

### 7.1 Unit Tests (per component)

| Component | Test Count | Coverage Target |
|-----------|------------|-----------------|
| RiskBudgetAllocator | 15 | 95% |
| LiquidityAwareRebalancer | 20 | 95% |
| TaxLotManager | 15 | 95% |
| TaxLossHarvester | 12 | 90% |
| MultiObjectiveOptimizer | 18 | 90% |
| ComplianceManager | 15 | 95% |
| Phase6Integration | 20 | 90% |
| **Total** | **115** | **93%** |

### 7.2 Integration Tests

1. End-to-end portfolio optimization cycle
2. Phase 5 + Phase 6 integration
3. Stress testing under various regimes
4. Tax harvesting with wash sale detection
5. Rebalancing with liquidity constraints

### 7.3 Performance Tests

- Optimization should complete in < 1 second for 50 positions
- Rebalancing calculation in < 500ms
- Compliance checks in < 100ms

---

## 8. Success Criteria

### 8.1 Performance Targets

```python
PHASE6_SUCCESS_CRITERIA = {
    'profit_rate_improvement': '+2-4%',
    'max_drawdown': '< 15%',
    'sharpe_ratio': '> 1.5',
    'tax_efficiency_gain': '+1-2% net',
    'rebalancing_cost_reduction': '30%',
    'compliance_violations': '0',
    'execution_slippage': '< 20 bps average',
}
```

### 8.2 Final System Performance (After Phase 6)

```python
FINAL_SYSTEM_TARGETS = {
    'total_profit_rate': '65-70%',
    'sharpe_ratio': '> 1.5',
    'max_drawdown': '< 15%',
    'win_rate': '> 55%',
    'profitable_assets_ratio': '> 70%',
}
```

---

## 9. Implementation Order

1. **Week 1**: Risk Budgeting + Compliance Manager (foundation)
2. **Week 2**: Liquidity-Aware Rebalancing
3. **Week 3**: Tax Optimization
4. **Week 4**: Multi-Objective Optimization
5. **Week 5**: Integration + Testing

---

## 10. File Structure

```
src/portfolio/
├── __init__.py                     # Module exports
├── risk_budgeting.py               # Component 1
├── liquidity_rebalancer.py         # Component 2
├── tax_optimizer.py                # Component 3
├── multi_objective_optimizer.py    # Component 4
├── compliance_manager.py           # Component 5
├── phase6_integration.py           # Unified system
└── utils.py                        # Shared utilities

tests/
├── test_phase6_risk_budgeting.py
├── test_phase6_rebalancing.py
├── test_phase6_tax_optimizer.py
├── test_phase6_multi_objective.py
├── test_phase6_compliance.py
└── test_phase6_integration.py
```

---

## Appendix A: Mathematical Formulas

### A.1 Risk Parity Optimization
```
Minimize: Σ_i Σ_j (RC_i - RC_j)²

Where:
RC_i = w_i * (Σw)_i / σ_p  (Risk Contribution of asset i)

Subject to:
Σw_i = 1
w_i >= 0
```

### A.2 Black-Litterman Expected Returns
```
E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]

Where:
π = equilibrium expected returns
P = view matrix
Q = view expected returns
Ω = view uncertainty
τ = scaling factor
```

### A.3 Market Impact Model
```
Cost = 0.5 × spread + η × σ × (Q/V)^δ

Where:
spread = bid-ask spread
η = permanent impact coefficient
σ = volatility
Q = order size
V = daily volume
δ = market impact exponent (typically 0.5)
```

### A.4 Tax-Adjusted Return
```
R_after_tax = R_gross - τ_ST × G_ST - τ_LT × G_LT + τ × L_harvested

Where:
τ_ST = short-term tax rate
τ_LT = long-term tax rate
G_ST = short-term gains
G_LT = long-term gains
L_harvested = harvested losses
```

---

*Document Version: 1.0*
*Phase 6 Status: SPECIFICATION COMPLETE*
*Ready for Implementation*
