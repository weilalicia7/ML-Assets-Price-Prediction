# Phase 6 Portfolio Optimization - Calculation Code Reference

This document contains all core calculation code from Phase 6 components.

---

## 1. Risk Budgeting Calculations

### 1.1 Portfolio Variance & Volatility

```python
def calculate_portfolio_variance(
    self,
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio variance.
    Formula: σ²_p = w' Σ w
    """
    return float(weights.T @ covariance_matrix @ weights)

def calculate_portfolio_volatility(
    self,
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(self.calculate_portfolio_variance(weights, covariance_matrix))
```

### 1.2 Marginal VaR Calculation

```python
def calculate_marginal_var(
    self,
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    portfolio_value: float = 1.0
) -> np.ndarray:
    """
    Calculate marginal VaR for each position.

    Formula: Marginal VaR = ∂VaR/∂w_i = (Σw)_i / σ_p * VaR_p
    """
    portfolio_vol = self.calculate_portfolio_volatility(weights, covariance_matrix)

    if portfolio_vol == 0:
        return np.zeros(len(weights))

    # Marginal contribution to volatility
    marginal_vol = (covariance_matrix @ weights) / portfolio_vol

    # Convert to VaR (z_score for 95% confidence = 1.645)
    marginal_var = marginal_vol * self.z_score * portfolio_value

    return marginal_var
```

### 1.3 Component VaR

```python
def calculate_component_var(
    self,
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    portfolio_value: float = 1.0
) -> np.ndarray:
    """
    Calculate component VaR for each position.

    Formula: Component VaR = w_i * Marginal VaR_i
    """
    marginal_var = self.calculate_marginal_var(weights, covariance_matrix, portfolio_value)
    return weights * marginal_var
```

### 1.4 Risk Contribution Percentage

```python
def calculate_risk_contribution_percent(
    self,
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate each position's percentage contribution to total risk.

    Formula: RC_i% = Component VaR_i / Total VaR
    """
    component_var = self.calculate_component_var(weights, covariance_matrix)
    total_var = component_var.sum()

    if total_var == 0:
        return np.ones(len(weights)) / len(weights)

    return component_var / total_var
```

### 1.5 Average Correlation

```python
def calculate_average_correlation(
    self,
    correlation_matrix: np.ndarray
) -> float:
    """
    Calculate average pairwise correlation.
    Excludes diagonal (self-correlation).
    """
    n = correlation_matrix.shape[0]
    if n <= 1:
        return 0.0

    # Get upper triangle excluding diagonal
    upper_tri = np.triu(correlation_matrix, k=1)
    num_pairs = (n * (n - 1)) / 2

    return upper_tri.sum() / num_pairs if num_pairs > 0 else 0.0
```

### 1.6 Diversification Ratio

```python
def calculate_diversification_ratio(
    self,
    weights: np.ndarray,
    volatilities: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculate diversification ratio.

    Formula: DR = (Σ w_i * σ_i) / σ_p

    Higher ratio indicates better diversification (always >= 1.0).
    """
    weighted_vol_sum = np.sum(weights * volatilities)
    portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)

    if portfolio_vol == 0:
        return 1.0

    return weighted_vol_sum / portfolio_vol
```

### 1.7 Risk Parity Allocation

```python
def _risk_parity_allocation(
    self,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk parity weights.
    Each position contributes equally to total portfolio risk.

    Objective: Minimize Σ_i Σ_j (RC_i - RC_j)²
    """
    n = covariance_matrix.shape[0]

    def risk_contribution_error(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()

        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
        if portfolio_vol == 0:
            return 0

        # Marginal risk contribution
        mrc = (covariance_matrix @ weights) / portfolio_vol
        # Component risk contribution
        crc = weights * mrc
        # Target: equal contribution
        target = portfolio_vol / n

        # Minimize squared deviation from equal
        return np.sum((crc - target) ** 2)

    # Initial guess: inverse volatility
    vol = np.sqrt(np.diag(covariance_matrix))
    vol = np.where(vol > 0, vol, 1e-6)
    x0 = (1 / vol) / (1 / vol).sum()

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.5) for _ in range(n)]

    result = minimize(
        risk_contribution_error,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    weights = result.x if result.success else x0
    return weights / weights.sum()
```

### 1.8 Inverse Volatility Allocation

```python
def _inverse_volatility_allocation(
    self,
    volatilities: np.ndarray
) -> np.ndarray:
    """
    Weight inversely proportional to volatility.

    Formula: w_i = (1/σ_i) / Σ(1/σ_j)
    """
    vol = np.where(volatilities > 0, volatilities, 1e-6)
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()
```

### 1.9 Minimum Variance Allocation

```python
def _minimum_variance_allocation(
    self,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Minimize portfolio variance.

    Objective: min w' Σ w
    Subject to: Σw_i = 1, w_i >= 0
    """
    n = covariance_matrix.shape[0]

    def portfolio_variance(weights):
        weights = np.array(weights)
        return weights.T @ covariance_matrix @ weights

    x0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.5) for _ in range(n)]

    result = minimize(
        portfolio_variance,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    weights = result.x if result.success else x0
    return weights / weights.sum()
```

---

## 2. Liquidity & Rebalancing Calculations

### 2.1 Slippage Forecast (Square-Root Market Impact Model)

```python
def get_slippage_forecast(
    self,
    ticker: str,
    trade_size: float,
    daily_volume: float,
    volatility: float
) -> float:
    """
    Forecast expected slippage for a trade.

    Square-Root Impact Model:
    Impact = k * σ * sqrt(Q/V)

    Where:
        k = impact coefficient (typically 0.1-0.5)
        σ = daily volatility
        Q = order quantity
        V = daily volume
    """
    volume_ratio = trade_size / daily_volume if daily_volume > 0 else 0.1

    # Get asset-specific parameters or defaults
    params = self.impact_model_params.get(ticker, {'k': 0.3, 'avg_slippage': 10})

    # Square-root market impact model
    impact = params['k'] * volatility * np.sqrt(volume_ratio)

    # Add historical average slippage
    historical_component = params.get('avg_slippage', 10) / 10000

    return impact + historical_component
```

### 2.2 Execution Cost Estimation

```python
def estimate_execution_cost(
    self,
    trade_size: float,
    daily_volume: float,
    volatility: float,
    spread: float,
    strategy: ExecutionStrategy
) -> Dict[str, float]:
    """
    Estimate total execution cost.

    Total Cost = Spread Cost + Market Impact + Timing Risk
    """
    volume_ratio = trade_size / daily_volume if daily_volume > 0 else 0.1

    # Spread cost (always pay half spread)
    spread_cost = spread / 2

    # Market impact (square-root model)
    k = self.config.market_impact_coefficient  # typically 0.3
    market_impact = k * volatility * np.sqrt(volume_ratio)

    # Strategy-specific adjustments
    strategy_multiplier = {
        ExecutionStrategy.IMMEDIATE: 1.5,   # Higher impact
        ExecutionStrategy.TWAP: 0.8,        # Reduced impact
        ExecutionStrategy.VWAP: 0.7,        # Lower impact
        ExecutionStrategy.ADAPTIVE: 0.75    # Optimized
    }.get(strategy, 1.0)

    adjusted_impact = market_impact * strategy_multiplier

    # Timing risk (for slower strategies)
    timing_risk = 0.0
    if strategy in [ExecutionStrategy.TWAP, ExecutionStrategy.VWAP]:
        timing_risk = volatility * 0.1  # 10% of daily vol

    total_cost = spread_cost + adjusted_impact + timing_risk

    return {
        'spread_cost': spread_cost,
        'market_impact': adjusted_impact,
        'timing_risk': timing_risk,
        'total_cost': total_cost,
        'total_cost_bps': total_cost * 10000
    }
```

### 2.3 Optimal Trade Size Calculation

```python
def calculate_optimal_trade_size(
    self,
    total_quantity: float,
    daily_volume: float,
    volatility: float,
    urgency: UrgencyLevel
) -> Tuple[float, int]:
    """
    Calculate optimal trade size and number of slices.

    Based on max volume participation rate adjusted by urgency.
    """
    max_volume_pct = self.config.max_single_trade_volume_pct  # e.g., 0.05 (5%)
    patience = self.urgency_params[urgency]['patience']

    # Adjust max participation based on urgency
    adjusted_max = max_volume_pct * (1 + (1 - patience))

    # Calculate max size per slice
    max_slice = daily_volume * adjusted_max

    if total_quantity <= max_slice:
        return total_quantity, 1

    # Calculate number of slices needed
    num_slices = int(np.ceil(total_quantity / max_slice))

    # Limit slices based on urgency
    max_slices = int(10 * patience) + 1
    num_slices = min(num_slices, max_slices)

    optimal_slice = total_quantity / num_slices

    return optimal_slice, num_slices
```

### 2.4 Drift Threshold Check

```python
def check_drift_threshold(
    self,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float]
) -> Tuple[bool, float]:
    """
    Check if position drift exceeds threshold.

    Drift = |current_weight - target_weight|
    """
    max_drift = 0.0

    all_tickers = set(current_weights.keys()) | set(target_weights.keys())

    for ticker in all_tickers:
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        drift = abs(current - target)
        max_drift = max(max_drift, drift)

    should_trigger = max_drift > self.config.drift_threshold  # e.g., 0.05 (5%)

    return should_trigger, max_drift
```

---

## 3. Tax Optimization Calculations

### 3.1 Unrealized Gains Calculation

```python
def calculate_unrealized_gains(
    self,
    ticker: str,
    current_price: float
) -> Dict[str, float]:
    """
    Calculate unrealized gains for a ticker.

    Gain = (Current Price - Cost Basis) * Quantity
    """
    lots = self.lots.get(ticker, [])

    short_term_gain = 0.0
    long_term_gain = 0.0
    total_gain = 0.0

    for lot in lots:
        lot.update_unrealized(current_price)
        total_gain += lot.unrealized_gain

        if lot.is_long_term:  # Held > 365 days
            long_term_gain += lot.unrealized_gain
        else:
            short_term_gain += lot.unrealized_gain

    return {
        'total_gain': total_gain,
        'short_term_gain': short_term_gain,
        'long_term_gain': long_term_gain,
        'total_quantity': sum(l.quantity for l in lots),
        'num_lots': len(lots)
    }
```

### 3.2 Tax Liability Calculation

```python
def sell_lots(
    self,
    ticker: str,
    quantity: float,
    current_price: float,
    method: LotSelectionMethod = LotSelectionMethod.FIFO
) -> TaxImpact:
    """
    Sell shares and calculate tax impact.

    Tax = ST_Gain * ST_Rate + LT_Gain * LT_Rate
    """
    lots = self.get_lots(ticker, method)
    remaining = quantity

    short_term_gain = 0.0
    long_term_gain = 0.0
    lots_used = []

    for lot in lots:
        if remaining <= 0:
            break

        sell_qty = min(lot.quantity, remaining)
        gain = (current_price - lot.cost_basis) * sell_qty

        if lot.is_long_term:
            long_term_gain += gain
        else:
            short_term_gain += gain

        lot.quantity -= sell_qty
        remaining -= sell_qty
        lots_used.append(lot.lot_id)

    # Calculate tax liability
    st_tax = short_term_gain * self.config.short_term_rate if short_term_gain > 0 else 0
    lt_tax = long_term_gain * self.config.long_term_rate if long_term_gain > 0 else 0
    total_tax = st_tax + lt_tax

    total_gain = short_term_gain + long_term_gain
    effective_rate = total_tax / total_gain if total_gain > 0 else 0

    return TaxImpact(
        ticker=ticker,
        quantity=quantity - remaining,
        realized_gain=total_gain,
        short_term_gain=short_term_gain,
        long_term_gain=long_term_gain,
        tax_liability=total_tax,
        effective_rate=effective_rate,
        lots_used=lots_used
    )
```

### 3.3 Tax Benefit Calculation

```python
def calculate_harvest_benefit(
    self,
    opportunities: List[HarvestOpportunity],
    ytd_gains: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate benefit of harvesting against YTD gains.

    Offset Priority:
    1. Short-term gains (highest tax rate)
    2. Long-term gains
    3. $3,000 ordinary income deduction
    """
    total_harvestable_loss = sum(
        abs(o.total_loss) for o in opportunities
        if o.action == HarvestAction.HARVEST
    )

    # Can offset short-term first, then long-term, then $3k ordinary
    st_offset = min(total_harvestable_loss, ytd_gains.get('short_term_gains', 0))
    remaining = total_harvestable_loss - st_offset

    lt_offset = min(remaining, ytd_gains.get('long_term_gains', 0))
    remaining = remaining - lt_offset

    ordinary_offset = min(remaining, 3000)  # $3k limit

    st_tax_savings = st_offset * self.config.short_term_rate
    lt_tax_savings = lt_offset * self.config.long_term_rate
    ordinary_savings = ordinary_offset * self.config.short_term_rate

    return {
        'total_harvestable_loss': total_harvestable_loss,
        'st_gains_offset': st_offset,
        'lt_gains_offset': lt_offset,
        'ordinary_income_offset': ordinary_offset,
        'carryforward': max(0, remaining - ordinary_offset),
        'total_tax_savings': st_tax_savings + lt_tax_savings + ordinary_savings
    }
```

### 3.4 Breakeven Return for Deferral

```python
def _calculate_breakeven_return(
    self,
    lot: TaxLot,
    days_to_lt: int
) -> float:
    """
    Calculate breakeven return for deferring sale.

    Returns the return needed to justify waiting for long-term treatment.

    Breakeven = Tax_Savings_Rate / Annualized_Factor
    """
    if lot.unrealized_gain <= 0:
        return 0.0  # No tax benefit for losses

    # Tax savings from long-term vs short-term
    tax_savings_rate = self.config.short_term_rate - self.config.long_term_rate

    # Convert to annualized return
    days_per_year = 365
    annualized_factor = days_per_year / max(days_to_lt, 1)

    # Breakeven: the return that equals tax savings
    breakeven = tax_savings_rate / annualized_factor

    return breakeven
```

---

## 4. Multi-Objective Optimization Calculations

### 4.1 Utility Function

```python
def calculate_utility(
    self,
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    current_weights: Optional[np.ndarray] = None,
    expected_gains: Optional[np.ndarray] = None
) -> float:
    """
    Calculate portfolio utility.

    Formula: U(w) = E[R_p] - (λ/2) * σ²_p - TC(w) - Tax(w)

    Where:
        E[R_p] = Expected portfolio return
        λ = Risk aversion parameter
        σ²_p = Portfolio variance
        TC(w) = Transaction costs
        Tax(w) = Tax impact
    """
    # Expected return
    exp_return = np.dot(weights, expected_returns)

    # Risk penalty
    variance = weights.T @ covariance_matrix @ weights
    risk_penalty = (self.risk_aversion / 2) * variance

    # Transaction cost
    transaction_cost = 0.0
    if current_weights is not None:
        turnover = np.sum(np.abs(weights - current_weights))
        transaction_cost = turnover * self.transaction_cost_rate

    # Tax impact
    tax_cost = 0.0
    if expected_gains is not None:
        tax_cost = np.sum(np.maximum(expected_gains, 0)) * self.tax_rate

    utility = exp_return - risk_penalty - transaction_cost - tax_cost

    return float(utility)
```

### 4.2 Certainty Equivalent

```python
def calculate_certainty_equivalent(
    self,
    expected_return: float,
    volatility: float
) -> float:
    """
    Calculate certainty equivalent return.

    Formula: CE = E[R] - (λ/2) * σ²

    The risk-free return that provides same utility as the risky portfolio.
    """
    return expected_return - (self.risk_aversion / 2) * (volatility ** 2)
```

### 4.3 Mean-Variance Optimization

```python
def _mean_variance_optimize(
    self,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    current_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Mean-variance optimization (Markowitz).

    Maximize: U = E[R] - (λ/2) * σ² - Turnover_Penalty
    Subject to: Σw_i = 1, 0 <= w_i <= 0.20
    """
    n = len(expected_returns)

    def objective(w):
        ret = np.dot(w, expected_returns)
        var = w.T @ covariance_matrix @ w
        utility = ret - (self.config.risk_aversion / 2) * var

        # Add turnover penalty if current weights exist
        if current_weights is not None:
            turnover = np.sum(np.abs(w - current_weights))
            utility -= turnover * 0.001

        return -utility  # Minimize negative utility

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.20) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': self.config.max_iterations}
    )

    return result.x, {'iterations': result.nit, 'converged': result.success}
```

### 4.4 Maximum Sharpe Ratio Optimization

```python
def _max_sharpe_optimize(
    self,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02
) -> Tuple[np.ndarray, Dict]:
    """
    Maximize Sharpe ratio.

    Sharpe = (E[R_p] - R_f) / σ_p
    """
    n = len(expected_returns)

    def neg_sharpe(w):
        ret = np.dot(w, expected_returns)
        vol = np.sqrt(w.T @ covariance_matrix @ w)
        if vol == 0:
            return 0
        return -(ret - risk_free_rate) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.20) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': self.config.max_iterations}
    )

    return result.x, {'iterations': result.nit, 'converged': result.success}
```

### 4.5 Maximum Diversification Optimization

```python
def _max_diversification_optimize(
    self,
    covariance_matrix: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """
    Maximize diversification ratio.

    DR = (Σ w_i * σ_i) / σ_p
    """
    n = covariance_matrix.shape[0]
    vol = np.sqrt(np.diag(covariance_matrix))
    vol = np.where(vol > 0, vol, 1e-6)

    def neg_diversification_ratio(w):
        weighted_vol = np.sum(w * vol)
        port_vol = np.sqrt(w.T @ covariance_matrix @ w)
        if port_vol == 0:
            return 0
        return -weighted_vol / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.50) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        neg_diversification_ratio,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': self.config.max_iterations}
    )

    return result.x, {'iterations': result.nit, 'converged': result.success}
```

### 4.6 Covariance Shrinkage (Ledoit-Wolf)

```python
def _shrink_covariance(
    self,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply Ledoit-Wolf shrinkage to covariance matrix.

    Shrunk = (1 - δ) * Sample + δ * Target

    Where:
        δ = shrinkage intensity (e.g., 0.2)
        Target = scaled identity matrix
    """
    n = covariance_matrix.shape[0]

    # Target: scaled identity matrix
    trace = np.trace(covariance_matrix)
    target = np.eye(n) * (trace / n)

    # Shrink
    shrinkage = self.config.covariance_shrinkage  # e.g., 0.2
    shrunk = (1 - shrinkage) * covariance_matrix + shrinkage * target

    return shrunk
```

### 4.7 Hierarchical Risk Parity (HRP)

```python
def _hrp_optimize(
    self,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """
    Hierarchical Risk Parity (HRP) optimization.

    Steps:
    1. Calculate correlation distance matrix
    2. Hierarchical clustering
    3. Allocate within clusters using inverse volatility
    """
    n = len(expected_returns)

    # Calculate correlation matrix
    vol = np.sqrt(np.diag(covariance_matrix))
    vol = np.where(vol > 0, vol, 1e-6)
    corr_matrix = covariance_matrix / np.outer(vol, vol)

    # Ensure valid correlation matrix
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1, 1)

    # Distance matrix: d = sqrt(0.5 * (1 - ρ))
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

    # Hierarchical clustering
    try:
        condensed_dist = squareform(dist_matrix)
        link = linkage(condensed_dist, method='single')
        clusters = fcluster(link, t=0.5, criterion='distance')
    except Exception:
        return np.ones(n) / n, {'iterations': 0, 'converged': False}

    # Allocate within clusters using inverse volatility
    weights = np.zeros(n)
    unique_clusters = np.unique(clusters)

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_vol = vol[cluster_mask]
        inv_vol = 1 / cluster_vol
        cluster_weights = inv_vol / inv_vol.sum()

        # Allocate cluster budget (equal budget per cluster)
        cluster_budget = 1.0 / len(unique_clusters)
        weights[cluster_mask] = cluster_weights * cluster_budget

    # Normalize
    weights = weights / weights.sum()

    return weights, {'iterations': 1, 'converged': True}
```

---

## 5. Compliance Calculations

### 5.1 Herfindahl-Hirschman Index (HHI)

```python
def calculate_herfindahl_index(
    self,
    weights: Dict[str, float]
) -> float:
    """
    Calculate Herfindahl-Hirschman Index.

    Formula: HHI = Σ(w_i)²

    Range: 1/n (equal weight) to 1 (single position)
    """
    weight_values = list(weights.values())
    if not weight_values:
        return 0.0
    return sum(w ** 2 for w in weight_values)
```

### 5.2 Effective Number of Positions

```python
def calculate_effective_n(
    self,
    weights: Dict[str, float]
) -> float:
    """
    Calculate effective number of positions.

    Formula: Effective N = 1 / HHI

    Represents the equivalent number of equal-weighted positions.
    """
    hhi = self.calculate_herfindahl_index(weights)
    return 1 / hhi if hhi > 0 else len(weights)
```

### 5.3 Concentration Risk Metrics

```python
def calculate_concentration_risk(
    self,
    weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate comprehensive concentration risk metrics.
    """
    weight_values = list(weights.values())

    if not weight_values:
        return {
            'hhi': 0.0,
            'effective_n': 0,
            'max_position': 0.0,
            'top_3_concentration': 0.0,
            'top_5_concentration': 0.0
        }

    # HHI (Herfindahl-Hirschman Index)
    hhi = sum(w ** 2 for w in weight_values)

    # Effective N
    effective_n = 1 / hhi if hhi > 0 else len(weight_values)

    # Top concentrations
    sorted_weights = sorted(weight_values, reverse=True)
    top_3 = sum(sorted_weights[:3])
    top_5 = sum(sorted_weights[:5])

    return {
        'hhi': hhi,
        'effective_n': effective_n,
        'max_position': max(weight_values),
        'top_3_concentration': top_3,
        'top_5_concentration': top_5
    }
```

---

## 6. Integration Calculations

### 6.1 Blend Allocations

```python
def _blend_allocations(
    self,
    weights1: Dict[str, float],
    weights2: Dict[str, float],
    blend_factor: float = 0.5
) -> Dict[str, float]:
    """
    Blend two allocation strategies.

    Formula: w_blended = α * w1 + (1-α) * w2
    """
    all_tickers = set(weights1.keys()) | set(weights2.keys())
    blended = {}

    for ticker in all_tickers:
        w1 = weights1.get(ticker, 0.0)
        w2 = weights2.get(ticker, 0.0)
        blended[ticker] = blend_factor * w1 + (1 - blend_factor) * w2

    # Normalize to sum to 1
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended
```

### 6.2 Signal-Based Weight Adjustment

```python
def _apply_signal_adjustments(
    self,
    weights: Dict[str, float],
    signals: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply Phase 5 signal adjustments to weights.

    Adjustment = 1 + (signal * scale)
    Where scale = 0.2 (±20% based on signal)
    """
    adjusted = {}

    for ticker, weight in weights.items():
        signal = signals.get(ticker, 0.0)
        # Adjust weight based on signal strength
        # Strong positive signal: increase weight
        # Strong negative signal: decrease weight
        adjustment = 1.0 + (signal * 0.2)  # +/- 20% based on signal
        adjusted[ticker] = weight * max(0.5, min(1.5, adjustment))

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted
```

### 6.3 Expected Benefit Calculation

```python
def _calculate_expected_benefit(
    self,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    expected_returns: Dict[str, float],
    target_risk: float
) -> float:
    """
    Calculate expected benefit of rebalancing.

    Benefit = Target Return - Current Return
    """
    current_return = sum(
        current_weights.get(t, 0.0) * r
        for t, r in expected_returns.items()
    )
    target_return = sum(
        target_weights.get(t, 0.0) * r
        for t, r in expected_returns.items()
    )

    return target_return - current_return
```

---

## Summary of Key Formulas

| Calculation | Formula |
|-------------|---------|
| Portfolio Variance | σ²_p = w'Σw |
| Portfolio Volatility | σ_p = √(w'Σw) |
| Marginal VaR | MVaR_i = (Σw)_i / σ_p × z × V |
| Component VaR | CVaR_i = w_i × MVaR_i |
| Risk Contribution | RC_i = CVaR_i / ΣCVaR |
| Diversification Ratio | DR = (Σw_i×σ_i) / σ_p |
| Slippage (Impact) | Impact = k × σ × √(Q/V) |
| Utility | U = E[R] - (λ/2)σ² - TC - Tax |
| Certainty Equivalent | CE = E[R] - (λ/2)σ² |
| Sharpe Ratio | SR = (E[R] - R_f) / σ |
| HHI | HHI = Σ(w_i)² |
| Effective N | N_eff = 1 / HHI |
| Tax Impact | Tax = ST_Gain×ST_Rate + LT_Gain×LT_Rate |

---

*Document Version: 1.0*
*Phase 6 Calculation Code Reference*
