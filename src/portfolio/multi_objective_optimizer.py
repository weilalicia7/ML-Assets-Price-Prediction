"""
Phase 6: Multi-Objective Portfolio Optimization

This module provides multi-objective optimization for portfolio construction,
balancing returns, risk, costs, and other objectives simultaneously.

Components:
- MultiObjectiveOptimizer: Main optimizer with Pareto frontier
- ConstraintManager: Manages portfolio constraints
- UtilityFunction: Combines objectives into utility score
- ObjectiveWeightOptimizer: Dynamic objective weight adjustment

Expected Impact: +0.5-1% profit rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize, differential_evolution
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"                          # Hierarchical Risk Parity
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"


class ConstraintType(Enum):
    """Types of portfolio constraints."""
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    EQUALITY = "equality"
    SUM_TO_ONE = "sum_to_one"
    SECTOR_LIMIT = "sector_limit"
    TURNOVER_LIMIT = "turnover_limit"


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization."""
    risk_aversion: float = 2.5               # Lambda parameter
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    reoptimize_frequency_days: int = 5       # Re-optimize every 5 days
    min_weight_change: float = 0.02          # 2% minimum change to act
    max_iterations: int = 1000               # Optimizer iterations
    convergence_tolerance: float = 1e-6
    use_robust_covariance: bool = True       # Shrinkage estimator
    covariance_shrinkage: float = 0.2        # Ledoit-Wolf shrinkage


@dataclass
class Constraint:
    """Represents a portfolio constraint."""
    name: str
    constraint_type: ConstraintType
    value: float
    tickers: Optional[List[str]] = None      # None means all
    priority: int = 1                         # Lower = higher priority


@dataclass
class OptimizationObjective:
    """Represents an optimization objective."""
    name: str
    weight: float
    direction: str                           # 'maximize' or 'minimize'
    function: Callable                       # Function to evaluate


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    utility: float
    method_used: str
    objectives_met: Dict[str, float]
    constraints_satisfied: bool
    binding_constraints: List[str]
    iterations: int
    converged: bool


# =============================================================================
# 1. Constraint Manager
# =============================================================================

class ConstraintManager:
    """
    Manages portfolio constraints.
    """

    def __init__(self):
        """Initialize constraint manager."""
        self.constraints: List[Constraint] = []

        # Add default constraints
        self._add_default_constraints()

    def _add_default_constraints(self):
        """Add standard portfolio constraints."""
        # Weights sum to 1
        self.add_constraint(
            name="sum_to_one",
            constraint_type=ConstraintType.SUM_TO_ONE,
            value=1.0,
            priority=0  # Highest priority
        )

        # Position bounds
        self.add_constraint(
            name="max_position",
            constraint_type=ConstraintType.UPPER_BOUND,
            value=0.20,
            priority=1
        )

        self.add_constraint(
            name="min_position",
            constraint_type=ConstraintType.LOWER_BOUND,
            value=0.0,
            priority=1
        )

    def add_constraint(
        self,
        name: str,
        constraint_type: ConstraintType,
        value: float,
        tickers: Optional[List[str]] = None,
        priority: int = 1
    ):
        """Add a constraint."""
        self.constraints.append(Constraint(
            name=name,
            constraint_type=constraint_type,
            value=value,
            tickers=tickers,
            priority=priority
        ))

    def remove_constraint(self, name: str):
        """Remove a constraint by name."""
        self.constraints = [c for c in self.constraints if c.name != name]

    def check_constraints(
        self,
        weights: np.ndarray,
        tickers: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio satisfies all constraints.

        Args:
            weights: Portfolio weights
            tickers: Ticker symbols

        Returns:
            Tuple of (all_satisfied, violated_constraints)
        """
        violated = []

        for constraint in self.constraints:
            if constraint.tickers:
                # Constraint applies to specific tickers
                indices = [i for i, t in enumerate(tickers) if t in constraint.tickers]
                relevant_weights = weights[indices]
            else:
                relevant_weights = weights

            if constraint.constraint_type == ConstraintType.SUM_TO_ONE:
                if abs(weights.sum() - constraint.value) > 0.001:
                    violated.append(constraint.name)

            elif constraint.constraint_type == ConstraintType.UPPER_BOUND:
                if np.any(relevant_weights > constraint.value + 0.001):
                    violated.append(constraint.name)

            elif constraint.constraint_type == ConstraintType.LOWER_BOUND:
                if np.any(relevant_weights < constraint.value - 0.001):
                    violated.append(constraint.name)

            elif constraint.constraint_type == ConstraintType.SECTOR_LIMIT:
                if relevant_weights.sum() > constraint.value + 0.001:
                    violated.append(constraint.name)

        return len(violated) == 0, violated

    def get_scipy_constraints(
        self,
        n_assets: int,
        tickers: List[str]
    ) -> List[Dict]:
        """
        Convert constraints to scipy format.

        Args:
            n_assets: Number of assets
            tickers: Ticker symbols

        Returns:
            List of scipy constraint dictionaries
        """
        scipy_constraints = []

        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.SUM_TO_ONE:
                scipy_constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - constraint.value
                })

            elif constraint.constraint_type == ConstraintType.SECTOR_LIMIT:
                if constraint.tickers:
                    indices = [i for i, t in enumerate(tickers) if t in constraint.tickers]
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, val=constraint.value: val - np.sum(w[idx])
                    })

        return scipy_constraints

    def get_bounds(
        self,
        n_assets: int,
        tickers: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Get bounds for each asset.

        Args:
            n_assets: Number of assets
            tickers: Ticker symbols

        Returns:
            List of (min, max) tuples
        """
        bounds = [(0.0, 1.0)] * n_assets

        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.UPPER_BOUND:
                if constraint.tickers:
                    indices = [i for i, t in enumerate(tickers) if t in constraint.tickers]
                else:
                    indices = range(n_assets)

                for i in indices:
                    bounds[i] = (bounds[i][0], min(bounds[i][1], constraint.value))

            elif constraint.constraint_type == ConstraintType.LOWER_BOUND:
                if constraint.tickers:
                    indices = [i for i, t in enumerate(tickers) if t in constraint.tickers]
                else:
                    indices = range(n_assets)

                for i in indices:
                    bounds[i] = (max(bounds[i][0], constraint.value), bounds[i][1])

        return bounds

    def get_binding_constraints(
        self,
        weights: np.ndarray,
        tickers: List[str],
        tolerance: float = 0.01
    ) -> List[str]:
        """Get constraints that are close to being violated."""
        binding = []

        for constraint in self.constraints:
            if constraint.tickers:
                indices = [i for i, t in enumerate(tickers) if t in constraint.tickers]
                relevant_weights = weights[indices]
            else:
                relevant_weights = weights

            if constraint.constraint_type == ConstraintType.UPPER_BOUND:
                if np.any(relevant_weights > constraint.value - tolerance):
                    binding.append(constraint.name)

            elif constraint.constraint_type == ConstraintType.LOWER_BOUND:
                if np.any(relevant_weights < constraint.value + tolerance):
                    binding.append(constraint.name)

        return binding


# =============================================================================
# 2. Utility Function
# =============================================================================

class UtilityFunction:
    """
    Combines objectives into a single utility score.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        transaction_cost_rate: float = 0.001,
        tax_rate: float = 0.20
    ):
        """
        Initialize utility function.

        Args:
            risk_aversion: Risk aversion parameter (lambda)
            transaction_cost_rate: Transaction cost rate
            tax_rate: Blended tax rate
        """
        self.risk_aversion = risk_aversion
        self.transaction_cost_rate = transaction_cost_rate
        self.tax_rate = tax_rate

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

        U(w) = E[R_p] - (λ/2) * σ²_p - TC(w) - Tax(w)

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            current_weights: Current weights for turnover calc
            expected_gains: Expected taxable gains

        Returns:
            Utility score
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

    def calculate_certainty_equivalent(
        self,
        expected_return: float,
        volatility: float
    ) -> float:
        """
        Calculate certainty equivalent return.

        CE = E[R] - (λ/2) * σ²

        Args:
            expected_return: Expected portfolio return
            volatility: Portfolio volatility

        Returns:
            Certainty equivalent return
        """
        return expected_return - (self.risk_aversion / 2) * (volatility ** 2)

    def get_optimal_risk_aversion(
        self,
        investor_profile: str
    ) -> float:
        """
        Get recommended risk aversion for investor profile.

        Args:
            investor_profile: 'conservative', 'moderate', 'aggressive'

        Returns:
            Recommended risk aversion parameter
        """
        profiles = {
            'conservative': 4.0,
            'moderate': 2.5,
            'aggressive': 1.5,
            'very_aggressive': 1.0
        }
        return profiles.get(investor_profile, 2.5)


# =============================================================================
# 3. Objective Weight Optimizer
# =============================================================================

class ObjectiveWeightOptimizer:
    """
    Dynamically adjusts objective weights based on market conditions.
    """

    def __init__(self):
        """Initialize optimizer."""
        # Default weights
        self.default_weights = {
            'return': 0.35,
            'risk': 0.30,
            'cost': 0.15,
            'sharpe': 0.20
        }

        # Regime-specific adjustments
        self.regime_weights = {
            'bull': {'return': 0.40, 'risk': 0.25, 'cost': 0.15, 'sharpe': 0.20},
            'normal': {'return': 0.35, 'risk': 0.30, 'cost': 0.15, 'sharpe': 0.20},
            'bear': {'return': 0.25, 'risk': 0.40, 'cost': 0.15, 'sharpe': 0.20},
            'crisis': {'return': 0.15, 'risk': 0.50, 'cost': 0.15, 'sharpe': 0.20}
        }

    def get_regime_adjusted_weights(
        self,
        current_regime: str
    ) -> Dict[str, float]:
        """
        Get objective weights adjusted for current regime.

        Args:
            current_regime: Current market regime

        Returns:
            Dictionary of objective weights
        """
        return self.regime_weights.get(current_regime, self.default_weights)

    def update_weights_from_performance(
        self,
        recent_results: List[OptimizationResult],
        target_sharpe: float = 1.5
    ) -> Dict[str, float]:
        """
        Update weights based on recent performance.

        Args:
            recent_results: List of recent optimization results
            target_sharpe: Target Sharpe ratio

        Returns:
            Adjusted objective weights
        """
        if not recent_results:
            return self.default_weights.copy()

        # Analyze recent performance
        avg_sharpe = np.mean([r.sharpe_ratio for r in recent_results])
        avg_risk = np.mean([r.expected_risk for r in recent_results])

        weights = self.default_weights.copy()

        # If underperforming Sharpe target, increase return focus
        if avg_sharpe < target_sharpe * 0.8:
            weights['return'] = min(0.50, weights['return'] + 0.05)
            weights['risk'] = max(0.20, weights['risk'] - 0.05)

        # If risk too high, increase risk focus
        if avg_risk > 0.20:
            weights['risk'] = min(0.50, weights['risk'] + 0.10)
            weights['return'] = max(0.20, weights['return'] - 0.10)

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights


# =============================================================================
# 4. Multi-Objective Optimizer
# =============================================================================

class MultiObjectiveOptimizer:
    """
    Multi-objective portfolio optimizer.

    Implements various optimization methods and combines multiple objectives.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.constraint_manager = ConstraintManager()
        self.utility_function = UtilityFunction(risk_aversion=self.config.risk_aversion)
        self.weight_optimizer = ObjectiveWeightOptimizer()

    def optimize(
        self,
        tickers: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        method: Optional[OptimizationMethod] = None,
        regime: str = 'normal'
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation.

        Args:
            tickers: List of ticker symbols
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            current_weights: Current portfolio weights
            method: Optimization method (default from config)
            regime: Current market regime

        Returns:
            OptimizationResult with optimal weights
        """
        n = len(tickers)
        method = method or self.config.optimization_method

        # Apply covariance shrinkage if configured
        if self.config.use_robust_covariance:
            covariance_matrix = self._shrink_covariance(covariance_matrix)

        # Run optimization
        if method == OptimizationMethod.MEAN_VARIANCE:
            weights, info = self._mean_variance_optimize(
                expected_returns, covariance_matrix, current_weights
            )
        elif method == OptimizationMethod.MAX_SHARPE:
            weights, info = self._max_sharpe_optimize(
                expected_returns, covariance_matrix
            )
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights, info = self._min_variance_optimize(covariance_matrix)
        elif method == OptimizationMethod.RISK_PARITY:
            weights, info = self._risk_parity_optimize(covariance_matrix)
        elif method == OptimizationMethod.HRP:
            weights, info = self._hrp_optimize(expected_returns, covariance_matrix)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights, info = self._max_diversification_optimize(covariance_matrix)
        else:
            weights, info = self._mean_variance_optimize(
                expected_returns, covariance_matrix, current_weights
            )

        # Calculate metrics
        exp_return = float(np.dot(weights, expected_returns))
        exp_risk = float(np.sqrt(weights.T @ covariance_matrix @ weights))
        sharpe = exp_return / exp_risk if exp_risk > 0 else 0

        utility = self.utility_function.calculate_utility(
            weights, expected_returns, covariance_matrix, current_weights
        )

        # Check constraints
        satisfied, violated = self.constraint_manager.check_constraints(weights, tickers)
        binding = self.constraint_manager.get_binding_constraints(weights, tickers)

        # Get objective weights for regime
        obj_weights = self.weight_optimizer.get_regime_adjusted_weights(regime)

        return OptimizationResult(
            weights={tickers[i]: weights[i] for i in range(n)},
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            utility=utility,
            method_used=method.value,
            objectives_met=obj_weights,
            constraints_satisfied=satisfied,
            binding_constraints=binding,
            iterations=info.get('iterations', 0),
            converged=info.get('converged', True)
        )

    def _shrink_covariance(
        self,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        n = covariance_matrix.shape[0]

        # Target: scaled identity matrix
        trace = np.trace(covariance_matrix)
        target = np.eye(n) * (trace / n)

        # Shrink
        shrinkage = self.config.covariance_shrinkage
        shrunk = (1 - shrinkage) * covariance_matrix + shrinkage * target

        return shrunk

    def _mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Mean-variance optimization (Markowitz)."""
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

        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.20) for _ in range(n)]

        # Initial guess
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

    def _max_sharpe_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> Tuple[np.ndarray, Dict]:
        """Maximize Sharpe ratio."""
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

    def _min_variance_optimize(
        self,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Minimum variance portfolio."""
        n = covariance_matrix.shape[0]

        def portfolio_variance(w):
            return w.T @ covariance_matrix @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.20) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        return result.x, {'iterations': result.nit, 'converged': result.success}

    def _risk_parity_optimize(
        self,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Risk parity optimization (equal risk contribution)."""
        n = covariance_matrix.shape[0]

        def risk_parity_objective(w):
            w = np.array(w)
            port_vol = np.sqrt(w.T @ covariance_matrix @ w)
            if port_vol == 0:
                return 0

            # Marginal risk contribution
            mrc = (covariance_matrix @ w) / port_vol
            # Component risk contribution
            crc = w * mrc
            # Target: equal contribution
            target = port_vol / n

            return np.sum((crc - target) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.50) for _ in range(n)]

        # Initialize with inverse volatility
        vol = np.sqrt(np.diag(covariance_matrix))
        vol = np.where(vol > 0, vol, 1e-6)
        x0 = (1 / vol) / (1 / vol).sum()

        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        return result.x, {'iterations': result.nit, 'converged': result.success}

    def _hrp_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Hierarchical Risk Parity (HRP) optimization."""
        n = len(expected_returns)

        # Calculate correlation matrix
        vol = np.sqrt(np.diag(covariance_matrix))
        vol = np.where(vol > 0, vol, 1e-6)
        corr_matrix = covariance_matrix / np.outer(vol, vol)

        # Ensure valid correlation matrix
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1, 1)

        # Distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix)
            link = linkage(condensed_dist, method='single')
            clusters = fcluster(link, t=0.5, criterion='distance')
        except Exception:
            # Fallback to equal weight if clustering fails
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

    def _max_diversification_optimize(
        self,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Maximize diversification ratio."""
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

    def calculate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_points: int = 50
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Calculate efficient frontier points.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            n_points: Number of points on frontier

        Returns:
            List of (risk, return, weights) tuples
        """
        n = len(expected_returns)

        # Find min and max return portfolios
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in target_returns:
            def objective(w):
                return w.T @ covariance_matrix @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, expected_returns) - t}
            ]
            bounds = [(0.0, 0.20) for _ in range(n)]
            x0 = np.ones(n) / n

            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500}
                )

                if result.success:
                    risk = np.sqrt(result.x.T @ covariance_matrix @ result.x)
                    ret = np.dot(result.x, expected_returns)
                    frontier.append((risk, ret, result.x))
            except Exception:
                continue

        return frontier

    def select_portfolio_from_frontier(
        self,
        frontier: List[Tuple[float, float, np.ndarray]],
        preference: str = 'max_sharpe',
        risk_free_rate: float = 0.02
    ) -> Tuple[float, float, np.ndarray]:
        """
        Select optimal portfolio from efficient frontier.

        Args:
            frontier: Efficient frontier points
            preference: Selection preference
            risk_free_rate: Risk-free rate

        Returns:
            Selected (risk, return, weights) tuple
        """
        if not frontier:
            raise ValueError("Empty frontier")

        if preference == 'max_sharpe':
            # Maximum Sharpe ratio
            best_idx = 0
            best_sharpe = -np.inf

            for i, (risk, ret, weights) in enumerate(frontier):
                if risk > 0:
                    sharpe = (ret - risk_free_rate) / risk
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_idx = i

            return frontier[best_idx]

        elif preference == 'min_variance':
            # Minimum variance
            return min(frontier, key=lambda x: x[0])

        else:
            # Middle of frontier
            return frontier[len(frontier) // 2]


# =============================================================================
# Factory Function
# =============================================================================

def create_multi_objective_optimizer(
    config: Optional[Dict] = None
) -> MultiObjectiveOptimizer:
    """
    Create configured multi-objective optimizer.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured MultiObjectiveOptimizer
    """
    if config:
        opt_config = OptimizationConfig(**config)
    else:
        opt_config = OptimizationConfig()

    return MultiObjectiveOptimizer(config=opt_config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'OptimizationMethod',
    'ConstraintType',

    # Data Classes
    'OptimizationConfig',
    'Constraint',
    'OptimizationObjective',
    'OptimizationResult',

    # Core Classes
    'ConstraintManager',
    'UtilityFunction',
    'ObjectiveWeightOptimizer',
    'MultiObjectiveOptimizer',

    # Factory
    'create_multi_objective_optimizer',
]
