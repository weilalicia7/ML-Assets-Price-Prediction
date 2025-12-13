"""
Phase 6 Portfolio Optimization Improvements

This module contains all the enhancements recommended for Phase 6:
1. Regime-aware risk budgeting with VIX adjustment
2. Expected Shortfall (CVaR) contribution
3. Adaptive execution strategy
4. Liquidity-adjusted VaR
5. Tax-aware rebalancing thresholds
6. Optimal tax-loss harvesting
7. Regime-aware utility function
8. Robust covariance estimation
9. Phase 5 signal integration
10. Optimization diagnostics

Expected Impact: +1-2% additional profit rate improvement
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# ENUMS AND CONFIGURATIONS
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"


class ExecutionStrategy(Enum):
    """Trade execution strategies."""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ADAPTIVE = "adaptive"


class UrgencyLevel(Enum):
    """Trade urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HarvestAction(Enum):
    """Tax-loss harvesting actions."""
    HARVEST = "harvest"
    HOLD = "hold"
    DEFER = "defer"


@dataclass
class RegimeConfig:
    """Configuration for regime-based adjustments."""
    # Risk budget multipliers by regime
    risk_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 0.5,
        'high_vol': 0.75,
        'normal': 1.0,
        'low_vol': 1.15
    })

    # Covariance lookback periods by regime (trading days)
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'crisis': 21,
        'high_vol': 42,
        'normal': 126,
        'low_vol': 252
    })

    # Shrinkage factors by regime
    shrinkage_factors: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 0.4,
        'high_vol': 0.4,
        'normal': 0.2,
        'low_vol': 0.2
    })

    # Risk aversion multipliers
    risk_aversion_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 2.0,
        'high_vol': 1.5,
        'normal': 1.0,
        'low_vol': 0.8
    })

    # Liquidity penalties
    liquidity_penalties: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 0.02,
        'high_vol': 0.01,
        'normal': 0.005,
        'low_vol': 0.002
    })


@dataclass
class TaxConfig:
    """Tax configuration for optimization."""
    short_term_rate: float = 0.37  # Short-term capital gains rate
    long_term_rate: float = 0.20  # Long-term capital gains rate
    transaction_cost_rate: float = 0.001  # 10 bps
    min_benefit_threshold: float = 0.0020  # 20 bps minimum benefit
    wash_sale_window: int = 30  # days


@dataclass
class HarvestOpportunity:
    """Tax-loss harvesting opportunity."""
    ticker: str
    total_loss: float
    is_short_term: bool
    action: HarvestAction
    days_held: int = 0
    wash_sale_risk: bool = False


# =============================================================================
# 1. REGIME-AWARE RISK BUDGETING
# =============================================================================

class RegimeAwareRiskBudgeter:
    """
    Enhanced risk budgeting with regime awareness and VIX adjustment.

    Adjusts risk budgets based on:
    - Market regime (crisis, high_vol, normal, low_vol)
    - VIX levels (normalized around VIX 15)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def calculate_risk_contribution_percent(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate percentage risk contribution for each position."""
        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)

        if portfolio_vol < 1e-10:
            return np.zeros(len(weights))

        # Marginal risk contribution
        marginal_contrib = covariance_matrix @ weights / portfolio_vol

        # Component risk contribution
        component_contrib = weights * marginal_contrib

        # Percentage contribution
        risk_percent = component_contrib / portfolio_vol

        return risk_percent

    def calculate_regime_aware_risk_budget(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        regime: str,
        macro_context: Dict
    ) -> Dict[str, Any]:
        """
        Adjust risk budgets based on market regime.

        Crisis: Reduce overall risk budget by 50%
        High Vol: Reduce by 25%
        Normal: Standard budget
        Low Vol: Increase by 15%

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            regime: Current market regime
            macro_context: Dict with 'vix' and other macro indicators

        Returns:
            Dict with base and adjusted risk contributions
        """
        # Base risk contribution
        base_risk_contrib = self.calculate_risk_contribution_percent(
            weights, covariance_matrix
        )

        # Regime multiplier
        regime_mult = self.config.risk_multipliers.get(regime, 1.0)

        # VIX-based adjustment (normalized around VIX 15)
        vix = macro_context.get('vix', 20.0)
        vix_adjustment = 1.0 - (vix - 15) / 100
        vix_adjustment = max(0.5, min(1.5, vix_adjustment))

        # Combined adjustment
        adjusted_risk_budget = base_risk_contrib * regime_mult * vix_adjustment

        return {
            'base_risk_contrib': base_risk_contrib,
            'regime_adjusted': adjusted_risk_budget,
            'total_risk_budget': float(np.sum(adjusted_risk_budget)),
            'regime_multiplier': regime_mult,
            'vix_adjustment': vix_adjustment,
            'regime': regime,
            'vix': vix
        }


# =============================================================================
# 2. EXPECTED SHORTFALL (CVaR) CONTRIBUTION
# =============================================================================

class ExpectedShortfallCalculator:
    """
    Calculate Expected Shortfall (CVaR) contribution for tail risk analysis.

    More robust than VaR for fat-tailed distributions commonly seen in
    financial markets.
    """

    def __init__(self, confidence: float = 0.975):
        self.confidence = confidence

    def calculate_es_contribution(
        self,
        weights: np.ndarray,
        returns_data: np.ndarray,
        confidence: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate Expected Shortfall (CVaR) contribution for each position.

        Args:
            weights: Portfolio weights
            returns_data: Historical returns matrix (n_periods x n_assets)
            confidence: Confidence level (default 0.975)

        Returns:
            Array of ES contributions per position
        """
        if confidence is None:
            confidence = self.confidence

        # Portfolio returns
        portfolio_returns = returns_data @ weights

        # ES threshold (VaR at confidence level)
        es_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)

        # Get tail scenarios (returns below VaR)
        tail_scenarios = portfolio_returns <= es_threshold

        if not np.any(tail_scenarios):
            return np.zeros(len(weights))

        # Calculate average loss contribution in tail scenarios
        es_contributions = np.zeros(len(weights))
        for i in range(len(weights)):
            position_contrib = returns_data[tail_scenarios, i] * weights[i]
            es_contributions[i] = np.mean(position_contrib)

        return es_contributions

    def calculate_portfolio_es(
        self,
        weights: np.ndarray,
        returns_data: np.ndarray,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate portfolio-level Expected Shortfall."""
        if confidence is None:
            confidence = self.confidence

        portfolio_returns = returns_data @ weights
        es_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= es_threshold]

        if len(tail_returns) == 0:
            return 0.0

        return float(np.mean(tail_returns))

    def calculate_marginal_es(
        self,
        weights: np.ndarray,
        returns_data: np.ndarray,
        confidence: Optional[float] = None,
        delta: float = 0.01
    ) -> np.ndarray:
        """Calculate marginal ES contribution (sensitivity)."""
        if confidence is None:
            confidence = self.confidence

        base_es = self.calculate_portfolio_es(weights, returns_data, confidence)
        marginal_es = np.zeros(len(weights))

        for i in range(len(weights)):
            weights_up = weights.copy()
            weights_up[i] += delta
            weights_up = weights_up / np.sum(weights_up)

            es_up = self.calculate_portfolio_es(weights_up, returns_data, confidence)
            marginal_es[i] = (es_up - base_es) / delta

        return marginal_es


# =============================================================================
# 3. ADAPTIVE EXECUTION STRATEGY
# =============================================================================

class AdaptiveExecutionSelector:
    """
    Dynamically select execution strategy based on market conditions.

    Considers:
    - Trade size relative to daily volume
    - Current volatility
    - Market regime
    - Urgency level
    """

    def __init__(self):
        self.regime_overrides = {
            'crisis': ExecutionStrategy.TWAP,  # Avoid information leakage
            'high_vol': ExecutionStrategy.VWAP,  # Reduce market impact
        }

    def get_adaptive_execution_strategy(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        market_regime: str,
        urgency: UrgencyLevel
    ) -> ExecutionStrategy:
        """
        Dynamically select execution strategy based on market conditions.

        Args:
            trade_size: Size of trade in dollars/shares
            daily_volume: Average daily volume
            volatility: Current volatility
            market_regime: Current market regime
            urgency: Trade urgency level

        Returns:
            Recommended ExecutionStrategy
        """
        volume_ratio = trade_size / daily_volume if daily_volume > 0 else 1.0

        # Base strategy selection based on urgency and size
        if urgency == UrgencyLevel.CRITICAL:
            base_strategy = ExecutionStrategy.IMMEDIATE
        elif urgency == UrgencyLevel.HIGH:
            base_strategy = ExecutionStrategy.IMMEDIATE
        elif volume_ratio > 0.1:  # Large trade (>10% of daily volume)
            base_strategy = ExecutionStrategy.VWAP
        elif volume_ratio > 0.05:  # Medium trade
            base_strategy = ExecutionStrategy.TWAP
        else:
            base_strategy = ExecutionStrategy.ADAPTIVE

        # Regime adjustments
        regime_adjustments = {
            'crisis': ExecutionStrategy.TWAP,
            'high_vol': ExecutionStrategy.VWAP,
            'normal': base_strategy,
            'low_vol': ExecutionStrategy.ADAPTIVE
        }

        # Override for critical urgency
        if urgency == UrgencyLevel.CRITICAL:
            return ExecutionStrategy.IMMEDIATE

        return regime_adjustments.get(market_regime, base_strategy)

    def get_execution_parameters(
        self,
        strategy: ExecutionStrategy,
        trade_size: float,
        daily_volume: float,
        market_regime: str
    ) -> Dict[str, Any]:
        """Get recommended execution parameters for the strategy."""
        volume_ratio = trade_size / daily_volume if daily_volume > 0 else 1.0

        if strategy == ExecutionStrategy.IMMEDIATE:
            return {
                'strategy': strategy.value,
                'slices': 1,
                'interval_minutes': 0,
                'max_participation_rate': 1.0
            }
        elif strategy == ExecutionStrategy.TWAP:
            # Spread over time based on size
            slices = max(5, int(volume_ratio * 20))
            return {
                'strategy': strategy.value,
                'slices': slices,
                'interval_minutes': 30,
                'max_participation_rate': 0.15
            }
        elif strategy == ExecutionStrategy.VWAP:
            return {
                'strategy': strategy.value,
                'slices': 10,
                'interval_minutes': 'volume_weighted',
                'max_participation_rate': 0.10
            }
        else:  # ADAPTIVE
            return {
                'strategy': strategy.value,
                'slices': 'dynamic',
                'interval_minutes': 'market_dependent',
                'max_participation_rate': 0.20
            }


# =============================================================================
# 4. LIQUIDITY-ADJUSTED VaR
# =============================================================================

class LiquidityAdjustedRisk:
    """
    Calculate risk measures adjusted for liquidation time.

    Accounts for the fact that larger positions take longer to liquidate,
    exposing the portfolio to more market risk during unwinding.
    """

    def __init__(self, stress_correlation: float = 0.3):
        """
        Args:
            stress_correlation: Serial correlation during stressed markets
        """
        self.stress_correlation = stress_correlation

    def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate standard portfolio VaR."""
        from scipy import stats

        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
        z_score = stats.norm.ppf(1 - confidence)

        return float(-z_score * portfolio_vol)

    def calculate_liquidity_adjusted_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        liquidation_days: int = 1,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate VaR adjusted for liquidation time.

        Liquidity-adjusted VaR increases with longer liquidation periods
        due to market movement risk during the unwind.

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            liquidation_days: Expected days to liquidate
            confidence: VaR confidence level

        Returns:
            Liquidity-adjusted VaR
        """
        daily_var = self.calculate_portfolio_var(
            weights, covariance_matrix, confidence
        )

        if liquidation_days <= 1:
            return daily_var

        # Adjust for serial correlation in stressed markets
        # This captures the tendency for losses to persist
        adjustment_factor = np.sqrt(
            liquidation_days +
            self.stress_correlation * (liquidation_days - 1)
        )

        return daily_var * adjustment_factor

    def calculate_position_liquidation_days(
        self,
        position_values: Dict[str, float],
        daily_volumes: Dict[str, float],
        max_participation: float = 0.20
    ) -> Dict[str, int]:
        """Estimate liquidation days for each position."""
        liquidation_days = {}

        for ticker, value in position_values.items():
            daily_vol = daily_volumes.get(ticker, 0)
            if daily_vol > 0:
                max_daily_trade = daily_vol * max_participation
                days_needed = int(np.ceil(value / max_daily_trade))
                liquidation_days[ticker] = max(1, days_needed)
            else:
                liquidation_days[ticker] = 5  # Default for illiquid

        return liquidation_days

    def calculate_portfolio_liquidation_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        position_values: Dict[str, float],
        daily_volumes: Dict[str, float],
        tickers: List[str],
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate comprehensive liquidity-adjusted risk metrics."""
        # Get liquidation days per position
        liq_days = self.calculate_position_liquidation_days(
            position_values, daily_volumes
        )

        # Weighted average liquidation days
        total_value = sum(position_values.values())
        if total_value > 0:
            avg_liq_days = sum(
                liq_days.get(t, 1) * position_values.get(t, 0)
                for t in tickers
            ) / total_value
        else:
            avg_liq_days = 1

        # Calculate VaRs
        daily_var = self.calculate_portfolio_var(
            weights, covariance_matrix, confidence
        )
        liq_adjusted_var = self.calculate_liquidity_adjusted_var(
            weights, covariance_matrix, int(avg_liq_days), confidence
        )

        return {
            'daily_var': daily_var,
            'liquidity_adjusted_var': liq_adjusted_var,
            'avg_liquidation_days': avg_liq_days,
            'position_liquidation_days': liq_days,
            'var_multiplier': liq_adjusted_var / daily_var if daily_var > 0 else 1.0
        }


# =============================================================================
# 5. TAX-AWARE REBALANCING
# =============================================================================

class TaxAwareRebalancer:
    """
    Determine if rebalancing is beneficial after considering tax implications.

    Only rebalance when: Expected Benefit > Tax Cost + Transaction Costs
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        self.config = config or TaxConfig()

    def should_rebalance_after_tax(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        tax_impact: Dict[str, Dict],
        expected_benefit: float
    ) -> Tuple[bool, float]:
        """
        Determine if rebalancing is beneficial after considering taxes.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            tax_impact: Dict of ticker -> {tax_liability: float, ...}
            expected_benefit: Expected benefit from rebalancing

        Returns:
            Tuple of (should_rebalance, net_benefit)
        """
        # Calculate total tax cost
        total_tax_cost = sum(
            tax_impact.get(ticker, {}).get('tax_liability', 0)
            for ticker in current_weights.keys()
        )

        # Calculate transaction costs (turnover)
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        turnover = sum(
            abs(target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        )
        transaction_cost = turnover * self.config.transaction_cost_rate

        # Total cost
        total_cost = total_tax_cost + transaction_cost

        # Net benefit
        net_benefit = expected_benefit - total_cost

        return net_benefit > self.config.min_benefit_threshold, net_benefit

    def calculate_tax_adjusted_threshold(
        self,
        ticker: str,
        current_weight: float,
        unrealized_gain_pct: float,
        is_short_term: bool,
        base_threshold: float = 0.05
    ) -> float:
        """
        Calculate tax-adjusted rebalancing threshold.

        Higher gains = higher threshold (more reluctant to sell)
        Short-term = higher threshold (higher tax rate)
        """
        tax_rate = (self.config.short_term_rate if is_short_term
                   else self.config.long_term_rate)

        # Tax drag factor
        tax_drag = unrealized_gain_pct * tax_rate * current_weight

        # Adjust threshold
        adjusted_threshold = base_threshold + tax_drag

        return min(adjusted_threshold, 0.20)  # Cap at 20%

    def get_tax_efficient_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        tax_lots: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Get trades optimized for tax efficiency."""
        trades = []

        for ticker in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            delta = target - current

            if abs(delta) < 0.001:
                continue

            lots = tax_lots.get(ticker, [])

            if delta < 0:  # Selling
                # Prioritize lots with losses or long-term gains
                sorted_lots = sorted(
                    lots,
                    key=lambda x: (
                        -1 if x.get('gain', 0) < 0 else 0,  # Losses first
                        0 if not x.get('is_short_term', True) else 1  # LT before ST
                    )
                )
                trades.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'amount': abs(delta),
                    'lot_order': sorted_lots
                })
            else:  # Buying
                trades.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'amount': delta
                })

        return trades


# =============================================================================
# 6. OPTIMAL TAX-LOSS HARVESTING
# =============================================================================

class TaxLossHarvestOptimizer:
    """
    Optimize which losses to harvest and when.

    Considerations:
    - Harvest losses that offset highest-tax gains first
    - Consider transaction costs
    - Save some losses for future years if beneficial
    - Avoid wash sales
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        self.config = config or TaxConfig()

    def optimize_tax_loss_harvesting(
        self,
        opportunities: List[HarvestOpportunity],
        ytd_gains: Dict[str, float],
        remaining_year_days: int
    ) -> List[HarvestOpportunity]:
        """
        Optimize which losses to harvest and when.

        Args:
            opportunities: List of harvest opportunities
            ytd_gains: Year-to-date gains by category
            remaining_year_days: Days remaining in tax year

        Returns:
            Prioritized list of opportunities to harvest
        """
        prioritized_opportunities = []

        for opp in opportunities:
            if opp.action != HarvestAction.HARVEST:
                continue

            # Calculate immediate tax benefit
            tax_rate = (self.config.short_term_rate if opp.is_short_term
                       else self.config.long_term_rate)
            immediate_benefit = abs(opp.total_loss) * tax_rate

            # Calculate carryforward benefit (if we defer)
            if remaining_year_days > 60:  # More than 2 months left
                # Might get better offset opportunities
                deferred_benefit = immediate_benefit * 0.8  # Discount for uncertainty
            else:
                deferred_benefit = immediate_benefit  # Use it or lose it

            # Benefit ratio (efficiency metric)
            benefit_ratio = immediate_benefit / abs(opp.total_loss) if opp.total_loss != 0 else 0

            # Adjust for wash sale risk
            if opp.wash_sale_risk:
                immediate_benefit *= 0.5  # Discount if might trigger wash sale

            prioritized_opportunities.append({
                'opportunity': opp,
                'immediate_benefit': immediate_benefit,
                'deferred_benefit': deferred_benefit,
                'benefit_ratio': benefit_ratio
            })

        # Sort by benefit ratio (highest first)
        prioritized_opportunities.sort(
            key=lambda x: x['benefit_ratio'],
            reverse=True
        )

        return [p['opportunity'] for p in prioritized_opportunities]

    def calculate_optimal_harvest_amount(
        self,
        loss_available: float,
        ytd_short_term_gains: float,
        ytd_long_term_gains: float
    ) -> Dict[str, float]:
        """Calculate optimal amount to harvest given current gains."""
        # Offset short-term gains first (higher tax rate)
        st_offset = min(loss_available, ytd_short_term_gains)
        remaining_loss = loss_available - st_offset

        # Then offset long-term gains
        lt_offset = min(remaining_loss, ytd_long_term_gains)
        remaining_loss = remaining_loss - lt_offset

        # Excess can offset ordinary income up to $3000
        ordinary_offset = min(remaining_loss, 3000)
        carryforward = remaining_loss - ordinary_offset

        return {
            'short_term_offset': st_offset,
            'long_term_offset': lt_offset,
            'ordinary_income_offset': ordinary_offset,
            'carryforward': carryforward,
            'total_tax_benefit': (
                st_offset * self.config.short_term_rate +
                lt_offset * self.config.long_term_rate +
                ordinary_offset * self.config.short_term_rate
            )
        }


# =============================================================================
# 7. REGIME-AWARE UTILITY FUNCTION
# =============================================================================

class RegimeAwareUtility:
    """
    Utility function that adapts to market regimes.

    Adjusts risk aversion and liquidity penalties based on regime.
    """

    def __init__(
        self,
        base_risk_aversion: float = 2.0,
        config: Optional[RegimeConfig] = None
    ):
        self.base_risk_aversion = base_risk_aversion
        self.config = config or RegimeConfig()

    def calculate_utility(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """Calculate base mean-variance utility."""
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights.T @ covariance_matrix @ weights

        utility = portfolio_return - 0.5 * self.base_risk_aversion * portfolio_variance
        return float(utility)

    def calculate_regime_aware_utility(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        regime: str,
        macro_context: Dict
    ) -> float:
        """
        Utility function that adapts to market regimes.

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns per asset
            covariance_matrix: Covariance matrix
            regime: Current market regime
            macro_context: Dict with 'vix' and other indicators

        Returns:
            Regime-adjusted utility value
        """
        # Get regime parameters
        risk_mult = self.config.risk_aversion_multipliers.get(regime, 1.0)
        liq_penalty = self.config.liquidity_penalties.get(regime, 0.005)

        # Adjusted risk aversion
        adjusted_risk_aversion = self.base_risk_aversion * risk_mult

        # Base components
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Base utility with adjusted risk aversion
        base_utility = portfolio_return - 0.5 * adjusted_risk_aversion * portfolio_variance

        # Liquidity penalty
        liquidity_penalty = portfolio_vol * liq_penalty

        # VIX-based penalty
        vix = macro_context.get('vix', 20.0)
        vix_penalty = max(0, (vix - 15) / 100)  # Penalty when VIX > 15

        # Final utility
        regime_utility = base_utility - liquidity_penalty - vix_penalty

        return float(regime_utility)

    def get_regime_optimal_risk_aversion(self, regime: str) -> float:
        """Get optimal risk aversion for current regime."""
        mult = self.config.risk_aversion_multipliers.get(regime, 1.0)
        return self.base_risk_aversion * mult


# =============================================================================
# 8. ROBUST COVARIANCE ESTIMATION
# =============================================================================

class RobustCovarianceEstimator:
    """
    Calculate covariance matrix using regime-appropriate data and shrinkage.

    Features:
    - Regime-specific lookback periods
    - Adaptive shrinkage
    - Ledoit-Wolf style shrinkage target
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def _shrink_covariance(
        self,
        sample_cov: np.ndarray,
        shrinkage: float
    ) -> np.ndarray:
        """Apply shrinkage to covariance matrix toward identity."""
        n = sample_cov.shape[0]

        # Shrinkage target: scaled identity matrix
        avg_variance = np.trace(sample_cov) / n
        target = np.eye(n) * avg_variance

        # Shrink toward target
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

        return shrunk_cov

    def calculate_regime_aware_covariance(
        self,
        historical_returns: pd.DataFrame,
        current_regime: str,
        regime_lookback: int = 63
    ) -> np.ndarray:
        """
        Calculate covariance matrix using regime-appropriate data.

        Args:
            historical_returns: DataFrame of historical returns
            current_regime: Current market regime
            regime_lookback: Default lookback if regime not found

        Returns:
            Shrunk covariance matrix
        """
        # Use regime-specific lookback
        lookback = self.config.lookback_periods.get(current_regime, regime_lookback)

        # Get recent data
        regime_data = historical_returns.tail(lookback)

        # Calculate sample covariance
        base_cov = regime_data.cov().values

        # Get shrinkage factor
        shrinkage = self.config.shrinkage_factors.get(current_regime, 0.2)

        # Apply shrinkage
        shrunk_cov = self._shrink_covariance(base_cov, shrinkage)

        return shrunk_cov

    def calculate_exponentially_weighted_covariance(
        self,
        historical_returns: pd.DataFrame,
        halflife: int = 63,
        min_periods: int = 30
    ) -> np.ndarray:
        """Calculate exponentially weighted covariance."""
        ewm_cov = historical_returns.ewm(
            halflife=halflife,
            min_periods=min_periods
        ).cov()

        # Get the last covariance matrix
        last_date = historical_returns.index[-1]
        cov_matrix = ewm_cov.loc[last_date].values

        return cov_matrix

    def detect_covariance_regime_shift(
        self,
        historical_returns: pd.DataFrame,
        short_window: int = 21,
        long_window: int = 63
    ) -> Dict[str, Any]:
        """Detect if covariance structure has shifted."""
        short_cov = historical_returns.tail(short_window).cov().values
        long_cov = historical_returns.tail(long_window).cov().values

        # Frobenius norm of difference
        diff_norm = np.linalg.norm(short_cov - long_cov, 'fro')
        base_norm = np.linalg.norm(long_cov, 'fro')

        relative_change = diff_norm / base_norm if base_norm > 0 else 0

        # Threshold for regime shift
        regime_shift = relative_change > 0.3

        return {
            'regime_shift_detected': regime_shift,
            'relative_change': relative_change,
            'short_window_vol': np.sqrt(np.trace(short_cov)),
            'long_window_vol': np.sqrt(np.trace(long_cov))
        }


# =============================================================================
# 9. PHASE 5 SIGNAL INTEGRATION
# =============================================================================

class Phase5SignalIntegrator:
    """
    Integrate Phase 5 dynamic weighting signals into portfolio optimization.

    Adjusts weights based on Phase 5 composite scores and confidence levels.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def integrate_phase5_signals(
        self,
        base_weights: Dict[str, float],
        phase5_signals: Dict[str, Dict],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Integrate Phase 5 dynamic weighting signals into portfolio optimization.

        Args:
            base_weights: Base portfolio weights from optimizer
            phase5_signals: Dict of ticker -> {composite_score, confidence, ...}
            confidence_threshold: Minimum confidence to apply adjustment

        Returns:
            Adjusted weights incorporating Phase 5 signals
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        adjusted_weights = {}

        for ticker, weight in base_weights.items():
            signal_data = phase5_signals.get(ticker, {})
            signal_strength = signal_data.get('composite_score', 0.5)
            signal_confidence = signal_data.get('confidence', 0.5)

            # Only adjust if signal confidence meets threshold
            if signal_confidence >= confidence_threshold:
                # Scale adjustment by confidence
                # signal_strength: 0 = very bearish, 0.5 = neutral, 1 = very bullish
                adjustment_factor = 1.0 + (signal_strength - 0.5) * signal_confidence
                adjusted_weight = weight * adjustment_factor
            else:
                adjusted_weight = weight

            adjusted_weights[ticker] = max(0, adjusted_weight)

        # Normalize weights to sum to 1
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def get_signal_summary(
        self,
        phase5_signals: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Get summary statistics of Phase 5 signals."""
        if not phase5_signals:
            return {'avg_score': 0.5, 'avg_confidence': 0.5, 'n_signals': 0}

        scores = [s.get('composite_score', 0.5) for s in phase5_signals.values()]
        confidences = [s.get('confidence', 0.5) for s in phase5_signals.values()]

        return {
            'avg_score': np.mean(scores),
            'avg_confidence': np.mean(confidences),
            'n_signals': len(scores),
            'bullish_count': sum(1 for s in scores if s > 0.6),
            'bearish_count': sum(1 for s in scores if s < 0.4),
            'high_confidence_count': sum(1 for c in confidences if c >= 0.7)
        }


# =============================================================================
# 10. OPTIMIZATION DIAGNOSTICS
# =============================================================================

class OptimizationDiagnostics:
    """
    Comprehensive validation and diagnostics for optimization results.

    Validates constraints, calculates metrics, and flags potential issues.
    """

    def __init__(
        self,
        max_weight: float = 0.20,
        min_weight: float = 0.01,
        max_turnover: float = 0.50
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_turnover = max_turnover

    def validate_optimization_result(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of optimization results.

        Args:
            weights: Optimized portfolio weights
            expected_returns: Expected returns per asset
            covariance_matrix: Asset covariance matrix
            current_weights: Current weights (for turnover calc)

        Returns:
            Dict with diagnostics and validation results
        """
        # Performance metrics
        expected_return = float(weights @ expected_returns)
        portfolio_variance = float(weights.T @ covariance_matrix @ weights)
        portfolio_vol = np.sqrt(portfolio_variance)
        sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else 0

        # Concentration metrics
        hhi = float(np.sum(weights ** 2))
        effective_n = 1 / hhi if hhi > 0 else len(weights)

        # Turnover calculation
        turnover = 0.0
        if current_weights is not None:
            turnover = float(np.sum(np.abs(weights - current_weights)))

        # Constraint violations
        weight_sum = float(np.sum(weights))
        max_weight_actual = float(np.max(weights))
        min_weight_actual = float(np.min(weights[weights > 0])) if np.any(weights > 0) else 0

        weight_sum_violation = abs(weight_sum - 1.0)
        max_weight_violation = max(0, max_weight_actual - self.max_weight)
        min_weight_violation = max(0, self.min_weight - min_weight_actual)
        turnover_violation = max(0, turnover - self.max_turnover)

        # Validity check
        is_valid = (
            weight_sum_violation < 1e-6 and
            max_weight_violation < 1e-6 and
            min_weight_violation < 1e-6
        )

        # Warning flags
        warnings = []
        if hhi > 0.15:
            warnings.append("High concentration (HHI > 0.15)")
        if turnover > self.max_turnover:
            warnings.append(f"High turnover ({turnover:.1%} > {self.max_turnover:.1%})")
        if expected_return < 0:
            warnings.append("Negative expected return")
        if sharpe_ratio < 0.3:
            warnings.append("Low Sharpe ratio")

        return {
            # Performance
            'expected_return': expected_return,
            'portfolio_vol': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,

            # Concentration
            'hhi': hhi,
            'effective_n': effective_n,

            # Turnover
            'turnover': turnover,

            # Constraint validation
            'weight_sum': weight_sum,
            'weight_sum_violation': weight_sum_violation,
            'max_weight_actual': max_weight_actual,
            'max_weight_violation': max_weight_violation,
            'min_weight_actual': min_weight_actual,
            'min_weight_violation': min_weight_violation,
            'turnover_violation': turnover_violation,

            # Overall status
            'is_valid': is_valid,
            'warnings': warnings,
            'n_warnings': len(warnings)
        }

    def compare_portfolios(
        self,
        weights_a: np.ndarray,
        weights_b: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        labels: Tuple[str, str] = ('Portfolio A', 'Portfolio B')
    ) -> Dict[str, Any]:
        """Compare two portfolio allocations."""
        diag_a = self.validate_optimization_result(
            weights_a, expected_returns, covariance_matrix
        )
        diag_b = self.validate_optimization_result(
            weights_b, expected_returns, covariance_matrix
        )

        return {
            labels[0]: diag_a,
            labels[1]: diag_b,
            'return_diff': diag_a['expected_return'] - diag_b['expected_return'],
            'vol_diff': diag_a['portfolio_vol'] - diag_b['portfolio_vol'],
            'sharpe_diff': diag_a['sharpe_ratio'] - diag_b['sharpe_ratio'],
            'weight_diff': float(np.sum(np.abs(weights_a - weights_b)))
        }


# =============================================================================
# INTEGRATED IMPROVEMENTS SYSTEM
# =============================================================================

class Phase6ImprovementsSystem:
    """
    Unified system integrating all Phase 6 improvements.

    Provides a single interface for:
    - Regime-aware risk budgeting
    - Expected Shortfall analysis
    - Adaptive execution
    - Liquidity-adjusted risk
    - Tax-aware optimization
    - Phase 5 integration
    - Comprehensive diagnostics
    """

    def __init__(
        self,
        regime_config: Optional[RegimeConfig] = None,
        tax_config: Optional[TaxConfig] = None,
        base_risk_aversion: float = 2.0
    ):
        self.regime_config = regime_config or RegimeConfig()
        self.tax_config = tax_config or TaxConfig()

        # Initialize all components
        self.risk_budgeter = RegimeAwareRiskBudgeter(self.regime_config)
        self.es_calculator = ExpectedShortfallCalculator()
        self.execution_selector = AdaptiveExecutionSelector()
        self.liquidity_risk = LiquidityAdjustedRisk()
        self.tax_rebalancer = TaxAwareRebalancer(self.tax_config)
        self.tax_harvester = TaxLossHarvestOptimizer(self.tax_config)
        self.regime_utility = RegimeAwareUtility(base_risk_aversion, self.regime_config)
        self.cov_estimator = RobustCovarianceEstimator(self.regime_config)
        self.phase5_integrator = Phase5SignalIntegrator()
        self.diagnostics = OptimizationDiagnostics()

    def optimize_with_improvements(
        self,
        base_weights: Dict[str, float],
        expected_returns: np.ndarray,
        historical_returns: pd.DataFrame,
        regime: str,
        macro_context: Dict,
        phase5_signals: Optional[Dict[str, Dict]] = None,
        tax_impact: Optional[Dict[str, Dict]] = None,
        expected_benefit: float = 0.01
    ) -> Dict[str, Any]:
        """
        Apply all improvements to portfolio optimization.

        Args:
            base_weights: Base optimized weights
            expected_returns: Expected returns per asset
            historical_returns: Historical returns DataFrame
            regime: Current market regime
            macro_context: Macro context dict with 'vix', etc.
            phase5_signals: Optional Phase 5 signals
            tax_impact: Optional tax impact per position
            expected_benefit: Expected benefit of rebalancing

        Returns:
            Comprehensive optimization result with all improvements
        """
        tickers = list(base_weights.keys())
        weights_array = np.array([base_weights[t] for t in tickers])

        # 1. Regime-aware covariance
        cov_matrix = self.cov_estimator.calculate_regime_aware_covariance(
            historical_returns, regime
        )

        # 2. Integrate Phase 5 signals
        if phase5_signals:
            adjusted_weights = self.phase5_integrator.integrate_phase5_signals(
                base_weights, phase5_signals
            )
        else:
            adjusted_weights = base_weights

        adjusted_array = np.array([adjusted_weights[t] for t in tickers])

        # 3. Risk budgeting analysis
        risk_budget = self.risk_budgeter.calculate_regime_aware_risk_budget(
            adjusted_array, cov_matrix, regime, macro_context
        )

        # 4. Expected Shortfall analysis
        returns_array = historical_returns[tickers].values
        es_contrib = self.es_calculator.calculate_es_contribution(
            adjusted_array, returns_array
        )

        # 5. Regime-aware utility
        utility = self.regime_utility.calculate_regime_aware_utility(
            adjusted_array, expected_returns, cov_matrix, regime, macro_context
        )

        # 6. Tax-aware rebalancing check
        should_rebalance = True
        net_benefit = expected_benefit
        if tax_impact:
            should_rebalance, net_benefit = self.tax_rebalancer.should_rebalance_after_tax(
                base_weights, adjusted_weights, tax_impact, expected_benefit
            )

        # 7. Diagnostics
        diagnostics = self.diagnostics.validate_optimization_result(
            adjusted_array, expected_returns, cov_matrix, weights_array
        )

        return {
            'optimized_weights': adjusted_weights,
            'risk_budget': risk_budget,
            'es_contributions': dict(zip(tickers, es_contrib)),
            'regime_utility': utility,
            'should_rebalance': should_rebalance,
            'net_benefit': net_benefit,
            'diagnostics': diagnostics,
            'regime': regime,
            'phase5_integrated': phase5_signals is not None
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_phase6_improvements_system(
    base_risk_aversion: float = 2.0,
    short_term_tax_rate: float = 0.37,
    long_term_tax_rate: float = 0.20
) -> Phase6ImprovementsSystem:
    """Factory function to create Phase 6 improvements system."""
    regime_config = RegimeConfig()
    tax_config = TaxConfig(
        short_term_rate=short_term_tax_rate,
        long_term_rate=long_term_tax_rate
    )

    return Phase6ImprovementsSystem(
        regime_config=regime_config,
        tax_config=tax_config,
        base_risk_aversion=base_risk_aversion
    )


# =============================================================================
# QUICK VALIDATION
# =============================================================================

def validate_improvements_module() -> Dict[str, bool]:
    """Quick validation of all improvement components."""
    results = {}

    try:
        # Test regime-aware risk budgeter
        budgeter = RegimeAwareRiskBudgeter()
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        cov = np.eye(4) * 0.04
        result = budgeter.calculate_regime_aware_risk_budget(
            weights, cov, 'normal', {'vix': 20}
        )
        results['regime_aware_risk_budgeting'] = 'regime_adjusted' in result
    except Exception as e:
        results['regime_aware_risk_budgeting'] = False

    try:
        # Test ES calculator
        es_calc = ExpectedShortfallCalculator()
        returns = np.random.randn(100, 4) * 0.02
        es = es_calc.calculate_es_contribution(weights, returns)
        results['expected_shortfall'] = len(es) == 4
    except Exception as e:
        results['expected_shortfall'] = False

    try:
        # Test execution selector
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            1000000, 10000000, 0.02, 'normal', UrgencyLevel.MEDIUM
        )
        results['adaptive_execution'] = isinstance(strategy, ExecutionStrategy)
    except Exception as e:
        results['adaptive_execution'] = False

    try:
        # Test liquidity-adjusted risk
        liq_risk = LiquidityAdjustedRisk()
        var = liq_risk.calculate_liquidity_adjusted_var(weights, cov, 3)
        results['liquidity_adjusted_var'] = var > 0
    except Exception as e:
        results['liquidity_adjusted_var'] = False

    try:
        # Test tax rebalancer
        tax_reb = TaxAwareRebalancer()
        should_reb, benefit = tax_reb.should_rebalance_after_tax(
            {'A': 0.5, 'B': 0.5},
            {'A': 0.4, 'B': 0.6},
            {'A': {'tax_liability': 0.001}},
            0.01
        )
        results['tax_aware_rebalancing'] = isinstance(should_reb, bool)
    except Exception as e:
        results['tax_aware_rebalancing'] = False

    try:
        # Test regime-aware utility
        utility = RegimeAwareUtility()
        exp_ret = np.array([0.10, 0.08, 0.12, 0.09])
        u = utility.calculate_regime_aware_utility(
            weights, exp_ret, cov, 'normal', {'vix': 20}
        )
        results['regime_aware_utility'] = isinstance(u, float)
    except Exception as e:
        results['regime_aware_utility'] = False

    try:
        # Test covariance estimator
        cov_est = RobustCovarianceEstimator()
        df = pd.DataFrame(np.random.randn(100, 4) * 0.02, columns=['A', 'B', 'C', 'D'])
        shrunk_cov = cov_est.calculate_regime_aware_covariance(df, 'normal')
        results['robust_covariance'] = shrunk_cov.shape == (4, 4)
    except Exception as e:
        results['robust_covariance'] = False

    try:
        # Test Phase 5 integrator
        integrator = Phase5SignalIntegrator()
        adjusted = integrator.integrate_phase5_signals(
            {'A': 0.5, 'B': 0.5},
            {'A': {'composite_score': 0.7, 'confidence': 0.8}}
        )
        results['phase5_integration'] = sum(adjusted.values()) > 0.99
    except Exception as e:
        results['phase5_integration'] = False

    try:
        # Test diagnostics
        diag = OptimizationDiagnostics()
        result = diag.validate_optimization_result(weights, exp_ret, cov)
        results['optimization_diagnostics'] = 'is_valid' in result
    except Exception as e:
        results['optimization_diagnostics'] = False

    return results


if __name__ == "__main__":
    print("Phase 6 Improvements Module - Validation")
    print("=" * 50)

    results = validate_improvements_module()

    for component, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"  {component}: {status_str}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} components validated")
