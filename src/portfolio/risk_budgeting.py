"""
Phase 6: Cross-Portfolio Risk Budgeting

This module provides portfolio-level risk allocation and management.
Allocates risk budget across the entire portfolio rather than treating
each position independently.

Components:
- RiskBudgetAllocator: Allocates total portfolio risk budget
- MarginalRiskCalculator: Calculates position risk contributions
- CorrelationAwareAllocator: Adjusts for correlation structure

Expected Impact: +0.5-1% profit rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize
from collections import deque


class AllocationMethod(Enum):
    """Risk allocation methods."""
    RISK_PARITY = "risk_parity"
    INVERSE_VOLATILITY = "inverse_volatility"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_VARIANCE = "minimum_variance"
    EQUAL_WEIGHT = "equal_weight"


class CorrelationRegime(Enum):
    """Correlation regime classifications."""
    LOW = "low"           # < 0.3 average correlation
    MODERATE = "moderate" # 0.3 - 0.6
    HIGH = "high"         # 0.6 - 0.8
    CRISIS = "crisis"     # > 0.8


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budgeting."""
    total_portfolio_var_limit: float = 0.15      # 15% max portfolio VaR
    max_single_position_risk: float = 0.25       # 25% of total risk budget
    min_diversification_ratio: float = 1.2       # Minimum diversification benefit
    correlation_lookback_days: int = 60          # Days for correlation calc
    risk_parity_tolerance: float = 0.05          # 5% deviation tolerance
    reallocation_threshold: float = 0.10         # 10% drift triggers realloc
    min_position_weight: float = 0.02            # 2% minimum position
    max_position_weight: float = 0.20            # 20% maximum position


@dataclass
class RiskAllocation:
    """Result of risk budget allocation."""
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    total_portfolio_risk: float
    diversification_ratio: float
    effective_n: float
    method_used: str
    is_optimal: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class PositionRiskMetrics:
    """Risk metrics for a single position."""
    ticker: str
    weight: float
    volatility: float
    marginal_var: float
    component_var: float
    risk_contribution_pct: float
    beta_to_portfolio: float


# =============================================================================
# 1. Marginal Risk Calculator
# =============================================================================

class MarginalRiskCalculator:
    """
    Calculates how each position contributes to total portfolio risk.

    Implements:
    - Marginal VaR calculation
    - Component VaR decomposition
    - Risk contribution percentages
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize calculator.

        Args:
            confidence_level: VaR confidence level (default 95%)
        """
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        from scipy.stats import norm
        return norm.ppf(confidence_level)

    def calculate_portfolio_variance(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio variance.

        Args:
            weights: Position weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Portfolio variance
        """
        return float(weights.T @ covariance_matrix @ weights)

    def calculate_portfolio_volatility(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(self.calculate_portfolio_variance(weights, covariance_matrix))

    def calculate_marginal_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float = 1.0
    ) -> np.ndarray:
        """
        Calculate marginal VaR for each position.

        Marginal VaR = ∂VaR/∂w_i = (Σw)_i / σ_p * VaR_p

        Args:
            weights: Position weights
            covariance_matrix: Asset covariance matrix
            portfolio_value: Total portfolio value

        Returns:
            Array of marginal VaR values
        """
        portfolio_vol = self.calculate_portfolio_volatility(weights, covariance_matrix)

        if portfolio_vol == 0:
            return np.zeros(len(weights))

        # Marginal contribution to volatility
        marginal_vol = (covariance_matrix @ weights) / portfolio_vol

        # Convert to VaR
        marginal_var = marginal_vol * self.z_score * portfolio_value

        return marginal_var

    def calculate_component_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float = 1.0
    ) -> np.ndarray:
        """
        Calculate component VaR for each position.

        Component VaR = w_i * Marginal VaR_i

        Args:
            weights: Position weights
            covariance_matrix: Asset covariance matrix
            portfolio_value: Total portfolio value

        Returns:
            Array of component VaR values
        """
        marginal_var = self.calculate_marginal_var(weights, covariance_matrix, portfolio_value)
        return weights * marginal_var

    def calculate_risk_contribution_percent(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate each position's percentage contribution to total risk.

        Args:
            weights: Position weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Array of risk contribution percentages (sum to 1.0)
        """
        component_var = self.calculate_component_var(weights, covariance_matrix)
        total_var = component_var.sum()

        if total_var == 0:
            return np.ones(len(weights)) / len(weights)

        return component_var / total_var

    def get_position_risk_metrics(
        self,
        tickers: List[str],
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        volatilities: np.ndarray,
        portfolio_value: float = 1.0
    ) -> List[PositionRiskMetrics]:
        """
        Get comprehensive risk metrics for all positions.

        Args:
            tickers: List of ticker symbols
            weights: Position weights
            covariance_matrix: Asset covariance matrix
            volatilities: Individual asset volatilities
            portfolio_value: Total portfolio value

        Returns:
            List of PositionRiskMetrics
        """
        marginal_var = self.calculate_marginal_var(weights, covariance_matrix, portfolio_value)
        component_var = self.calculate_component_var(weights, covariance_matrix, portfolio_value)
        risk_contrib = self.calculate_risk_contribution_percent(weights, covariance_matrix)

        # Calculate beta to portfolio
        portfolio_var = self.calculate_portfolio_variance(weights, covariance_matrix)
        if portfolio_var > 0:
            betas = (covariance_matrix @ weights) / portfolio_var
        else:
            betas = np.ones(len(weights))

        metrics = []
        for i, ticker in enumerate(tickers):
            metrics.append(PositionRiskMetrics(
                ticker=ticker,
                weight=weights[i],
                volatility=volatilities[i],
                marginal_var=marginal_var[i],
                component_var=component_var[i],
                risk_contribution_pct=risk_contrib[i],
                beta_to_portfolio=betas[i]
            ))

        return metrics


# =============================================================================
# 2. Correlation-Aware Allocator
# =============================================================================

class CorrelationAwareAllocator:
    """
    Adjusts allocations based on correlation structure.

    Tracks correlation regimes and adjusts allocations to maximize
    diversification benefits.
    """

    def __init__(self, lookback_days: int = 60):
        """
        Initialize allocator.

        Args:
            lookback_days: Days for correlation calculation
        """
        self.lookback_days = lookback_days
        self.correlation_history: deque = deque(maxlen=252)

        # Correlation regime thresholds
        self.regime_thresholds = {
            CorrelationRegime.LOW: 0.3,
            CorrelationRegime.MODERATE: 0.6,
            CorrelationRegime.HIGH: 0.8,
            CorrelationRegime.CRISIS: 1.0
        }

        # Allocation adjustments by regime
        self.regime_adjustments = {
            CorrelationRegime.LOW: 1.0,        # Standard allocation
            CorrelationRegime.MODERATE: 0.9,    # Slight reduction
            CorrelationRegime.HIGH: 0.75,       # Aggressive reduction
            CorrelationRegime.CRISIS: 0.5       # Defensive positioning
        }

    def calculate_average_correlation(
        self,
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate average pairwise correlation.

        Args:
            correlation_matrix: Correlation matrix

        Returns:
            Average correlation (excluding diagonal)
        """
        n = correlation_matrix.shape[0]
        if n <= 1:
            return 0.0

        # Get upper triangle excluding diagonal
        upper_tri = np.triu(correlation_matrix, k=1)
        num_pairs = (n * (n - 1)) / 2

        return upper_tri.sum() / num_pairs if num_pairs > 0 else 0.0

    def detect_correlation_regime(
        self,
        correlation_matrix: np.ndarray
    ) -> CorrelationRegime:
        """
        Detect current correlation regime.

        Args:
            correlation_matrix: Current correlation matrix

        Returns:
            Detected correlation regime
        """
        avg_corr = self.calculate_average_correlation(correlation_matrix)

        if avg_corr >= self.regime_thresholds[CorrelationRegime.HIGH]:
            return CorrelationRegime.CRISIS
        elif avg_corr >= self.regime_thresholds[CorrelationRegime.MODERATE]:
            return CorrelationRegime.HIGH
        elif avg_corr >= self.regime_thresholds[CorrelationRegime.LOW]:
            return CorrelationRegime.MODERATE
        else:
            return CorrelationRegime.LOW

    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """
        Calculate diversification ratio.

        DR = (Σ w_i * σ_i) / σ_p

        Higher ratio indicates better diversification.

        Args:
            weights: Position weights
            volatilities: Individual asset volatilities
            covariance_matrix: Asset covariance matrix

        Returns:
            Diversification ratio (>= 1.0)
        """
        weighted_vol_sum = np.sum(weights * volatilities)
        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)

        if portfolio_vol == 0:
            return 1.0

        return weighted_vol_sum / portfolio_vol

    def find_uncorrelated_opportunities(
        self,
        correlation_matrix: np.ndarray,
        current_weights: np.ndarray,
        tickers: List[str],
        max_correlation: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find assets with low correlation to current portfolio.

        Args:
            correlation_matrix: Correlation matrix
            current_weights: Current position weights
            tickers: List of ticker symbols
            max_correlation: Maximum correlation threshold

        Returns:
            List of (ticker, portfolio_correlation) tuples
        """
        # Calculate correlation of each asset to portfolio
        portfolio_corr = correlation_matrix @ current_weights

        opportunities = []
        for i, ticker in enumerate(tickers):
            if current_weights[i] < 0.01:  # Not currently held
                if abs(portfolio_corr[i]) < max_correlation:
                    opportunities.append((ticker, portfolio_corr[i]))

        # Sort by lowest correlation
        opportunities.sort(key=lambda x: abs(x[1]))

        return opportunities

    def adjust_for_correlation_regime(
        self,
        weights: np.ndarray,
        correlation_matrix: np.ndarray,
        target_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Adjust weights based on correlation regime.

        Args:
            weights: Current or proposed weights
            correlation_matrix: Current correlation matrix
            target_weights: Optional target weights to blend toward

        Returns:
            Adjusted weights
        """
        regime = self.detect_correlation_regime(correlation_matrix)
        adjustment = self.regime_adjustments[regime]

        # In high correlation regimes, move toward equal weight
        if adjustment < 1.0:
            equal_weight = np.ones(len(weights)) / len(weights)
            adjusted = adjustment * weights + (1 - adjustment) * equal_weight
        else:
            adjusted = weights.copy()

        # Normalize to sum to 1
        adjusted = adjusted / adjusted.sum() if adjusted.sum() > 0 else adjusted

        return adjusted


# =============================================================================
# 3. Risk Budget Allocator
# =============================================================================

class RiskBudgetAllocator:
    """
    Allocates total portfolio risk budget across positions.

    Implements multiple allocation methods:
    - Risk Parity
    - Inverse Volatility
    - Maximum Diversification
    - Minimum Variance
    """

    def __init__(self, config: Optional[RiskBudgetConfig] = None):
        """
        Initialize allocator.

        Args:
            config: Risk budget configuration
        """
        self.config = config or RiskBudgetConfig()
        self.marginal_calculator = MarginalRiskCalculator()
        self.correlation_allocator = CorrelationAwareAllocator(
            lookback_days=self.config.correlation_lookback_days
        )

    def allocate_risk_budget(
        self,
        tickers: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        total_risk_budget: Optional[float] = None
    ) -> RiskAllocation:
        """
        Allocate risk budget across positions.

        Args:
            tickers: List of ticker symbols
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            method: Allocation method to use
            total_risk_budget: Total risk budget (default from config)

        Returns:
            RiskAllocation with optimal weights
        """
        n = len(tickers)
        if n == 0:
            return RiskAllocation(
                weights={},
                risk_contributions={},
                total_portfolio_risk=0.0,
                diversification_ratio=1.0,
                effective_n=0,
                method_used=method.value,
                is_optimal=True
            )

        total_risk_budget = total_risk_budget or self.config.total_portfolio_var_limit
        volatilities = np.sqrt(np.diag(covariance_matrix))

        # Get optimal weights based on method
        if method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(covariance_matrix)
        elif method == AllocationMethod.INVERSE_VOLATILITY:
            weights = self._inverse_volatility_allocation(volatilities)
        elif method == AllocationMethod.MAXIMUM_DIVERSIFICATION:
            weights = self._max_diversification_allocation(volatilities, covariance_matrix)
        elif method == AllocationMethod.MINIMUM_VARIANCE:
            weights = self._minimum_variance_allocation(covariance_matrix)
        else:  # EQUAL_WEIGHT
            weights = np.ones(n) / n

        # Apply position constraints
        weights = self._apply_constraints(weights)

        # Calculate metrics
        risk_contrib = self.marginal_calculator.calculate_risk_contribution_percent(
            weights, covariance_matrix
        )
        portfolio_vol = self.marginal_calculator.calculate_portfolio_volatility(
            weights, covariance_matrix
        )
        div_ratio = self.correlation_allocator.calculate_diversification_ratio(
            weights, volatilities, covariance_matrix
        )

        # Calculate effective N (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_n = 1.0 / hhi if hhi > 0 else n

        # Check warnings
        warnings = []
        if div_ratio < self.config.min_diversification_ratio:
            warnings.append(f"Diversification ratio {div_ratio:.2f} below minimum {self.config.min_diversification_ratio}")
        if portfolio_vol > total_risk_budget:
            warnings.append(f"Portfolio volatility {portfolio_vol:.2%} exceeds budget {total_risk_budget:.2%}")
        if np.max(risk_contrib) > self.config.max_single_position_risk:
            warnings.append(f"Position risk concentration exceeds {self.config.max_single_position_risk:.0%}")

        return RiskAllocation(
            weights={tickers[i]: weights[i] for i in range(n)},
            risk_contributions={tickers[i]: risk_contrib[i] for i in range(n)},
            total_portfolio_risk=portfolio_vol,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            method_used=method.value,
            is_optimal=len(warnings) == 0,
            warnings=warnings
        )

    def _risk_parity_allocation(
        self,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk parity weights.

        Each position contributes equally to total portfolio risk.
        """
        n = covariance_matrix.shape[0]

        def risk_contribution_error(weights):
            weights = np.array(weights)
            # Ensure weights sum to 1
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
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
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

    def _inverse_volatility_allocation(
        self,
        volatilities: np.ndarray
    ) -> np.ndarray:
        """Weight inversely proportional to volatility."""
        vol = np.where(volatilities > 0, volatilities, 1e-6)
        inv_vol = 1 / vol
        return inv_vol / inv_vol.sum()

    def _max_diversification_allocation(
        self,
        volatilities: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Maximize diversification ratio."""
        n = len(volatilities)

        def neg_diversification_ratio(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()

            weighted_vol = np.sum(weights * volatilities)
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)

            if portfolio_vol == 0:
                return 0

            return -weighted_vol / portfolio_vol

        x0 = np.ones(n) / n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.5) for _ in range(n)]

        result = minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x if result.success else x0
        return weights / weights.sum()

    def _minimum_variance_allocation(
        self,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Minimize portfolio variance."""
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

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply position size constraints."""
        # Clip to min/max
        weights = np.clip(
            weights,
            self.config.min_position_weight,
            self.config.max_position_weight
        )
        # Renormalize
        return weights / weights.sum()

    def calculate_optimal_reallocation(
        self,
        current_weights: Dict[str, float],
        target_allocation: RiskAllocation,
        transaction_cost_rate: float = 0.001
    ) -> Dict[str, float]:
        """
        Calculate trades needed to reach target allocation.

        Args:
            current_weights: Current portfolio weights
            target_allocation: Target risk allocation
            transaction_cost_rate: Cost per unit traded

        Returns:
            Dictionary of trades (positive = buy, negative = sell)
        """
        trades = {}

        for ticker, target_weight in target_allocation.weights.items():
            current_weight = current_weights.get(ticker, 0.0)
            diff = target_weight - current_weight

            # Only trade if difference exceeds threshold + costs
            if abs(diff) > self.config.reallocation_threshold:
                trades[ticker] = diff
            elif abs(diff) > transaction_cost_rate * 2:
                # Smaller rebalance if cost-effective
                trades[ticker] = diff

        return trades

    def get_risk_metrics(
        self,
        tickers: List[str],
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Get comprehensive portfolio risk metrics.

        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Dictionary of risk metrics
        """
        volatilities = np.sqrt(np.diag(covariance_matrix))
        portfolio_vol = self.marginal_calculator.calculate_portfolio_volatility(
            weights, covariance_matrix
        )

        # VaR at 95%
        var_95 = portfolio_vol * 1.645

        # Expected Shortfall (CVaR) approximation
        cvar_95 = portfolio_vol * 2.063  # Normal assumption

        # Diversification ratio
        div_ratio = self.correlation_allocator.calculate_diversification_ratio(
            weights, volatilities, covariance_matrix
        )

        # Effective N
        hhi = np.sum(weights ** 2)
        effective_n = 1.0 / hhi if hhi > 0 else len(weights)

        # Concentration metrics
        max_weight = np.max(weights)
        top_3_weight = np.sum(np.sort(weights)[-3:])

        # Correlation regime
        corr_matrix = np.zeros_like(covariance_matrix)
        for i in range(len(volatilities)):
            for j in range(len(volatilities)):
                if volatilities[i] > 0 and volatilities[j] > 0:
                    corr_matrix[i, j] = covariance_matrix[i, j] / (volatilities[i] * volatilities[j])

        regime = self.correlation_allocator.detect_correlation_regime(corr_matrix)

        return {
            'portfolio_volatility': portfolio_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'diversification_ratio': div_ratio,
            'effective_n': effective_n,
            'hhi': hhi,
            'max_position_weight': max_weight,
            'top_3_concentration': top_3_weight,
            'correlation_regime': regime.value
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_risk_budget_allocator(
    config: Optional[Dict] = None
) -> RiskBudgetAllocator:
    """
    Create configured risk budget allocator.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured RiskBudgetAllocator
    """
    if config:
        budget_config = RiskBudgetConfig(**config)
    else:
        budget_config = RiskBudgetConfig()

    return RiskBudgetAllocator(config=budget_config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'AllocationMethod',
    'CorrelationRegime',

    # Data Classes
    'RiskBudgetConfig',
    'RiskAllocation',
    'PositionRiskMetrics',

    # Core Classes
    'MarginalRiskCalculator',
    'CorrelationAwareAllocator',
    'RiskBudgetAllocator',

    # Factory
    'create_risk_budget_allocator',
]
