"""
Phase 6 Improvements Test Suite - Aligned with Implementation

This test suite tests the actual implementation in phase6_improvements.py:
- RegimeAwareRiskBudgeter
- ExpectedShortfallCalculator
- AdaptiveExecutionSelector
- LiquidityAdjustedRisk
- TaxAwareRebalancer
- TaxLossHarvestOptimizer
- RegimeAwareUtility
- RobustCovarianceEstimator
- Phase5SignalIntegrator
- OptimizationDiagnostics
- Phase6ImprovementsSystem
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio.phase6_improvements import (
    # Enums
    MarketRegime,
    ExecutionStrategy,
    UrgencyLevel,
    HarvestAction,

    # Configurations
    RegimeConfig,
    TaxConfig,
    HarvestOpportunity,

    # Core Classes
    RegimeAwareRiskBudgeter,
    ExpectedShortfallCalculator,
    AdaptiveExecutionSelector,
    LiquidityAdjustedRisk,
    TaxAwareRebalancer,
    TaxLossHarvestOptimizer,
    RegimeAwareUtility,
    RobustCovarianceEstimator,
    Phase5SignalIntegrator,
    OptimizationDiagnostics,

    # Main System
    Phase6ImprovementsSystem,

    # Factory
    create_phase6_improvements_system,

    # Validation
    validate_improvements_module,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_weights():
    """Standard 4-asset portfolio weights."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def sample_covariance():
    """Sample covariance matrix (4x4)."""
    # Create a positive definite covariance matrix
    cov = np.array([
        [0.04, 0.01, 0.005, 0.002],
        [0.01, 0.03, 0.008, 0.003],
        [0.005, 0.008, 0.025, 0.004],
        [0.002, 0.003, 0.004, 0.02]
    ])
    return cov


@pytest.fixture
def sample_returns():
    """Sample historical returns (100 days x 4 assets)."""
    np.random.seed(42)
    return np.random.randn(100, 4) * 0.02


@pytest.fixture
def sample_expected_returns():
    """Expected returns for 4 assets."""
    return np.array([0.10, 0.08, 0.12, 0.09])


@pytest.fixture
def sample_historical_df():
    """Historical returns as DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    data = np.random.randn(252, 4) * 0.02
    return pd.DataFrame(data, index=dates, columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN'])


@pytest.fixture
def macro_context():
    """Sample macro context."""
    return {
        'vix': 20.0,
        'interest_rate': 0.05,
        'inflation': 0.03
    }


@pytest.fixture
def regime_config():
    """Sample regime configuration."""
    return RegimeConfig()


@pytest.fixture
def tax_config():
    """Sample tax configuration."""
    return TaxConfig(
        short_term_rate=0.37,
        long_term_rate=0.20,
        transaction_cost_rate=0.001
    )


# =============================================================================
# 1. REGIME-AWARE RISK BUDGETER TESTS
# =============================================================================

class TestRegimeAwareRiskBudgeter:
    """Test suite for RegimeAwareRiskBudgeter."""

    def test_initialization(self, regime_config):
        """Test initialization with config."""
        budgeter = RegimeAwareRiskBudgeter(regime_config)
        assert budgeter.config is not None
        assert budgeter.config.risk_multipliers['crisis'] == 0.5

    def test_initialization_default(self):
        """Test initialization with default config."""
        budgeter = RegimeAwareRiskBudgeter()
        assert budgeter.config is not None

    def test_calculate_risk_contribution_percent(self, sample_weights, sample_covariance):
        """Test risk contribution calculation."""
        budgeter = RegimeAwareRiskBudgeter()
        risk_contrib = budgeter.calculate_risk_contribution_percent(
            sample_weights, sample_covariance
        )

        assert len(risk_contrib) == 4
        assert all(isinstance(x, (float, np.floating)) for x in risk_contrib)

    def test_calculate_regime_aware_risk_budget_normal(self, sample_weights, sample_covariance, macro_context):
        """Test regime-aware risk budget in normal regime."""
        budgeter = RegimeAwareRiskBudgeter()
        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_covariance, 'normal', macro_context
        )

        assert 'base_risk_contrib' in result
        assert 'regime_adjusted' in result
        assert 'total_risk_budget' in result
        assert 'regime_multiplier' in result
        assert 'vix_adjustment' in result
        assert result['regime'] == 'normal'
        assert result['regime_multiplier'] == 1.0

    def test_calculate_regime_aware_risk_budget_crisis(self, sample_weights, sample_covariance, macro_context):
        """Test regime-aware risk budget in crisis regime."""
        budgeter = RegimeAwareRiskBudgeter()
        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_covariance, 'crisis', macro_context
        )

        assert result['regime_multiplier'] == 0.5
        assert result['regime'] == 'crisis'

    def test_calculate_regime_aware_risk_budget_high_vix(self, sample_weights, sample_covariance):
        """Test risk budget with high VIX."""
        budgeter = RegimeAwareRiskBudgeter()
        high_vix_context = {'vix': 40.0}

        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_covariance, 'normal', high_vix_context
        )

        # VIX adjustment should be lower (more conservative) with high VIX
        assert result['vix_adjustment'] < 1.0

    def test_calculate_regime_aware_risk_budget_low_vix(self, sample_weights, sample_covariance):
        """Test risk budget with low VIX."""
        budgeter = RegimeAwareRiskBudgeter()
        low_vix_context = {'vix': 10.0}

        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_covariance, 'low_vol', low_vix_context
        )

        # VIX adjustment should be higher (more aggressive) with low VIX
        assert result['vix_adjustment'] >= 1.0


# =============================================================================
# 2. EXPECTED SHORTFALL CALCULATOR TESTS
# =============================================================================

class TestExpectedShortfallCalculator:
    """Test suite for ExpectedShortfallCalculator."""

    def test_initialization(self):
        """Test initialization with default confidence."""
        es_calc = ExpectedShortfallCalculator()
        assert es_calc.confidence == 0.975

    def test_initialization_custom_confidence(self):
        """Test initialization with custom confidence."""
        es_calc = ExpectedShortfallCalculator(confidence=0.99)
        assert es_calc.confidence == 0.99

    def test_calculate_es_contribution(self, sample_weights, sample_returns):
        """Test ES contribution calculation."""
        es_calc = ExpectedShortfallCalculator()
        es_contrib = es_calc.calculate_es_contribution(
            sample_weights, sample_returns
        )

        assert len(es_contrib) == 4
        assert all(isinstance(x, (float, np.floating)) for x in es_contrib)

    def test_calculate_portfolio_es(self, sample_weights, sample_returns):
        """Test portfolio-level ES calculation."""
        es_calc = ExpectedShortfallCalculator()
        portfolio_es = es_calc.calculate_portfolio_es(
            sample_weights, sample_returns
        )

        assert isinstance(portfolio_es, float)
        # ES should be negative (it's a loss measure)
        assert portfolio_es <= 0

    def test_calculate_marginal_es(self, sample_weights, sample_returns):
        """Test marginal ES calculation."""
        es_calc = ExpectedShortfallCalculator()
        marginal_es = es_calc.calculate_marginal_es(
            sample_weights, sample_returns
        )

        assert len(marginal_es) == 4

    def test_es_with_different_confidence(self, sample_weights, sample_returns):
        """Test ES with different confidence levels."""
        es_calc = ExpectedShortfallCalculator()

        es_95 = es_calc.calculate_portfolio_es(sample_weights, sample_returns, confidence=0.95)
        es_99 = es_calc.calculate_portfolio_es(sample_weights, sample_returns, confidence=0.99)

        # Higher confidence should give more extreme (more negative) ES
        assert es_99 <= es_95


# =============================================================================
# 3. ADAPTIVE EXECUTION SELECTOR TESTS
# =============================================================================

class TestAdaptiveExecutionSelector:
    """Test suite for AdaptiveExecutionSelector."""

    def test_initialization(self):
        """Test initialization."""
        selector = AdaptiveExecutionSelector()
        assert selector.regime_overrides is not None

    def test_get_strategy_critical_urgency(self):
        """Test that CRITICAL urgency always returns IMMEDIATE."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=1_000_000,
            daily_volume=10_000_000,
            volatility=0.02,
            market_regime='crisis',
            urgency=UrgencyLevel.CRITICAL
        )

        assert strategy == ExecutionStrategy.IMMEDIATE

    def test_get_strategy_high_urgency(self):
        """Test HIGH urgency behavior."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=100_000,
            daily_volume=10_000_000,
            volatility=0.02,
            market_regime='normal',
            urgency=UrgencyLevel.HIGH
        )

        assert strategy == ExecutionStrategy.IMMEDIATE

    def test_get_strategy_large_trade(self):
        """Test strategy for large trades (>10% of volume)."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=2_000_000,
            daily_volume=10_000_000,
            volatility=0.02,
            market_regime='normal',
            urgency=UrgencyLevel.LOW
        )

        # Large trades should use VWAP or similar
        assert strategy in [ExecutionStrategy.VWAP, ExecutionStrategy.ADAPTIVE]

    def test_get_strategy_crisis_regime(self):
        """Test strategy during crisis regime."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=100_000,
            daily_volume=10_000_000,
            volatility=0.05,
            market_regime='crisis',
            urgency=UrgencyLevel.MEDIUM
        )

        assert strategy == ExecutionStrategy.TWAP

    def test_get_execution_parameters_immediate(self):
        """Test execution parameters for IMMEDIATE strategy."""
        selector = AdaptiveExecutionSelector()
        params = selector.get_execution_parameters(
            ExecutionStrategy.IMMEDIATE,
            trade_size=100_000,
            daily_volume=10_000_000,
            market_regime='normal'
        )

        assert params['strategy'] == 'immediate'
        assert params['slices'] == 1

    def test_get_execution_parameters_twap(self):
        """Test execution parameters for TWAP strategy."""
        selector = AdaptiveExecutionSelector()
        params = selector.get_execution_parameters(
            ExecutionStrategy.TWAP,
            trade_size=1_000_000,
            daily_volume=10_000_000,
            market_regime='normal'
        )

        assert params['strategy'] == 'twap'
        assert params['slices'] >= 5


# =============================================================================
# 4. LIQUIDITY-ADJUSTED RISK TESTS
# =============================================================================

class TestLiquidityAdjustedRisk:
    """Test suite for LiquidityAdjustedRisk."""

    def test_initialization(self):
        """Test initialization."""
        liq_risk = LiquidityAdjustedRisk()
        assert liq_risk.stress_correlation == 0.3

    def test_initialization_custom_correlation(self):
        """Test initialization with custom correlation."""
        liq_risk = LiquidityAdjustedRisk(stress_correlation=0.5)
        assert liq_risk.stress_correlation == 0.5

    def test_calculate_portfolio_var(self, sample_weights, sample_covariance):
        """Test standard VaR calculation."""
        liq_risk = LiquidityAdjustedRisk()
        var = liq_risk.calculate_portfolio_var(
            sample_weights, sample_covariance, confidence=0.95
        )

        assert var > 0
        assert isinstance(var, float)

    def test_calculate_liquidity_adjusted_var_single_day(self, sample_weights, sample_covariance):
        """Test liquidity-adjusted VaR with 1-day liquidation."""
        liq_risk = LiquidityAdjustedRisk()

        daily_var = liq_risk.calculate_portfolio_var(sample_weights, sample_covariance)
        liq_var = liq_risk.calculate_liquidity_adjusted_var(
            sample_weights, sample_covariance, liquidation_days=1
        )

        # Should be equal for 1-day liquidation
        assert abs(daily_var - liq_var) < 1e-10

    def test_calculate_liquidity_adjusted_var_multi_day(self, sample_weights, sample_covariance):
        """Test liquidity-adjusted VaR with multi-day liquidation."""
        liq_risk = LiquidityAdjustedRisk()

        daily_var = liq_risk.calculate_portfolio_var(sample_weights, sample_covariance)
        liq_var = liq_risk.calculate_liquidity_adjusted_var(
            sample_weights, sample_covariance, liquidation_days=5
        )

        # Multi-day should have higher VaR
        assert liq_var > daily_var

    def test_calculate_position_liquidation_days(self):
        """Test liquidation days calculation."""
        liq_risk = LiquidityAdjustedRisk()

        position_values = {'AAPL': 1_000_000, 'GOOGL': 500_000}
        daily_volumes = {'AAPL': 10_000_000, 'GOOGL': 1_000_000}

        liq_days = liq_risk.calculate_position_liquidation_days(
            position_values, daily_volumes, max_participation=0.20
        )

        assert 'AAPL' in liq_days
        assert 'GOOGL' in liq_days
        assert liq_days['AAPL'] >= 1

    def test_calculate_portfolio_liquidation_var(self, sample_weights, sample_covariance):
        """Test comprehensive liquidity-adjusted risk metrics."""
        liq_risk = LiquidityAdjustedRisk()

        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        position_values = {'AAPL': 250_000, 'GOOGL': 250_000, 'MSFT': 250_000, 'AMZN': 250_000}
        daily_volumes = {'AAPL': 5_000_000, 'GOOGL': 3_000_000, 'MSFT': 4_000_000, 'AMZN': 2_000_000}

        result = liq_risk.calculate_portfolio_liquidation_var(
            sample_weights, sample_covariance, position_values, daily_volumes, tickers
        )

        assert 'daily_var' in result
        assert 'liquidity_adjusted_var' in result
        assert 'avg_liquidation_days' in result
        assert 'position_liquidation_days' in result
        assert 'var_multiplier' in result


# =============================================================================
# 5. TAX-AWARE REBALANCER TESTS
# =============================================================================

class TestTaxAwareRebalancer:
    """Test suite for TaxAwareRebalancer."""

    def test_initialization(self, tax_config):
        """Test initialization with config."""
        rebalancer = TaxAwareRebalancer(tax_config)
        assert rebalancer.config.short_term_rate == 0.37

    def test_should_rebalance_after_tax_beneficial(self):
        """Test rebalancing decision when beneficial."""
        rebalancer = TaxAwareRebalancer()

        current = {'AAPL': 0.5, 'GOOGL': 0.5}
        target = {'AAPL': 0.6, 'GOOGL': 0.4}
        tax_impact = {'AAPL': {'tax_liability': 0.001}, 'GOOGL': {'tax_liability': 0.001}}
        expected_benefit = 0.05  # 5% expected benefit

        should_rebalance, net_benefit = rebalancer.should_rebalance_after_tax(
            current, target, tax_impact, expected_benefit
        )

        assert should_rebalance is True
        assert net_benefit > 0

    def test_should_rebalance_after_tax_not_beneficial(self):
        """Test rebalancing decision when not beneficial."""
        rebalancer = TaxAwareRebalancer()

        current = {'AAPL': 0.5, 'GOOGL': 0.5}
        target = {'AAPL': 0.51, 'GOOGL': 0.49}
        tax_impact = {'AAPL': {'tax_liability': 0.01}, 'GOOGL': {'tax_liability': 0.01}}
        expected_benefit = 0.001  # Small expected benefit

        should_rebalance, net_benefit = rebalancer.should_rebalance_after_tax(
            current, target, tax_impact, expected_benefit
        )

        # High tax costs should prevent rebalancing
        assert should_rebalance is False

    def test_calculate_tax_adjusted_threshold(self):
        """Test tax-adjusted threshold calculation."""
        rebalancer = TaxAwareRebalancer()

        threshold_short = rebalancer.calculate_tax_adjusted_threshold(
            ticker='AAPL',
            current_weight=0.20,
            unrealized_gain_pct=0.50,
            is_short_term=True
        )

        threshold_long = rebalancer.calculate_tax_adjusted_threshold(
            ticker='AAPL',
            current_weight=0.20,
            unrealized_gain_pct=0.50,
            is_short_term=False
        )

        # Short-term should have higher threshold
        assert threshold_short > threshold_long

    def test_get_tax_efficient_trades(self):
        """Test tax-efficient trade generation."""
        rebalancer = TaxAwareRebalancer()

        current = {'AAPL': 0.5, 'GOOGL': 0.5}
        target = {'AAPL': 0.4, 'GOOGL': 0.6}
        tax_lots = {
            'AAPL': [
                {'gain': -100, 'is_short_term': True},
                {'gain': 200, 'is_short_term': False}
            ]
        }

        trades = rebalancer.get_tax_efficient_trades(current, target, tax_lots)

        assert len(trades) == 2  # One sell, one buy


# =============================================================================
# 6. TAX LOSS HARVEST OPTIMIZER TESTS
# =============================================================================

class TestTaxLossHarvestOptimizer:
    """Test suite for TaxLossHarvestOptimizer."""

    def test_initialization(self, tax_config):
        """Test initialization with config."""
        harvester = TaxLossHarvestOptimizer(tax_config)
        assert harvester.config is not None

    def test_optimize_tax_loss_harvesting(self):
        """Test tax loss harvesting optimization."""
        harvester = TaxLossHarvestOptimizer()

        opportunities = [
            HarvestOpportunity(
                ticker='AAPL',
                total_loss=-1000,
                is_short_term=True,
                action=HarvestAction.HARVEST
            ),
            HarvestOpportunity(
                ticker='GOOGL',
                total_loss=-500,
                is_short_term=False,
                action=HarvestAction.HARVEST
            )
        ]

        ytd_gains = {'short_term': 2000, 'long_term': 1000}

        prioritized = harvester.optimize_tax_loss_harvesting(
            opportunities, ytd_gains, remaining_year_days=100
        )

        assert len(prioritized) == 2

    def test_calculate_optimal_harvest_amount(self):
        """Test optimal harvest amount calculation."""
        harvester = TaxLossHarvestOptimizer()

        result = harvester.calculate_optimal_harvest_amount(
            loss_available=10000,
            ytd_short_term_gains=5000,
            ytd_long_term_gains=3000
        )

        assert 'short_term_offset' in result
        assert 'long_term_offset' in result
        assert 'ordinary_income_offset' in result
        assert 'carryforward' in result
        assert 'total_tax_benefit' in result

        # Verify offsets
        assert result['short_term_offset'] == 5000
        assert result['long_term_offset'] == 3000


# =============================================================================
# 7. REGIME-AWARE UTILITY TESTS
# =============================================================================

class TestRegimeAwareUtility:
    """Test suite for RegimeAwareUtility."""

    def test_initialization(self, regime_config):
        """Test initialization with config."""
        utility = RegimeAwareUtility(base_risk_aversion=3.0, config=regime_config)
        assert utility.base_risk_aversion == 3.0

    def test_calculate_utility(self, sample_weights, sample_expected_returns, sample_covariance):
        """Test base utility calculation."""
        utility = RegimeAwareUtility()

        u = utility.calculate_utility(
            sample_weights, sample_expected_returns, sample_covariance
        )

        assert isinstance(u, float)

    def test_calculate_regime_aware_utility_normal(self, sample_weights, sample_expected_returns, sample_covariance, macro_context):
        """Test regime-aware utility in normal regime."""
        utility = RegimeAwareUtility()

        u = utility.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_covariance,
            regime='normal', macro_context=macro_context
        )

        assert isinstance(u, float)

    def test_calculate_regime_aware_utility_crisis(self, sample_weights, sample_expected_returns, sample_covariance, macro_context):
        """Test regime-aware utility in crisis regime."""
        utility = RegimeAwareUtility()

        u_normal = utility.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_covariance,
            regime='normal', macro_context=macro_context
        )

        u_crisis = utility.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_covariance,
            regime='crisis', macro_context=macro_context
        )

        # Crisis utility should be lower due to higher risk aversion
        assert u_crisis < u_normal

    def test_get_regime_optimal_risk_aversion(self):
        """Test regime-specific risk aversion."""
        utility = RegimeAwareUtility(base_risk_aversion=2.0)

        ra_normal = utility.get_regime_optimal_risk_aversion('normal')
        ra_crisis = utility.get_regime_optimal_risk_aversion('crisis')

        assert ra_crisis > ra_normal


# =============================================================================
# 8. ROBUST COVARIANCE ESTIMATOR TESTS
# =============================================================================

class TestRobustCovarianceEstimator:
    """Test suite for RobustCovarianceEstimator."""

    def test_initialization(self, regime_config):
        """Test initialization with config."""
        estimator = RobustCovarianceEstimator(regime_config)
        assert estimator.config is not None

    def test_calculate_regime_aware_covariance(self, sample_historical_df):
        """Test regime-aware covariance calculation."""
        estimator = RobustCovarianceEstimator()

        cov = estimator.calculate_regime_aware_covariance(
            sample_historical_df, current_regime='normal'
        )

        assert cov.shape == (4, 4)
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        assert all(eigvals >= -1e-10)

    def test_calculate_regime_aware_covariance_crisis(self, sample_historical_df):
        """Test crisis regime covariance (shorter lookback)."""
        estimator = RobustCovarianceEstimator()

        cov = estimator.calculate_regime_aware_covariance(
            sample_historical_df, current_regime='crisis'
        )

        assert cov.shape == (4, 4)

    def test_detect_covariance_regime_shift(self, sample_historical_df):
        """Test regime shift detection."""
        estimator = RobustCovarianceEstimator()

        result = estimator.detect_covariance_regime_shift(sample_historical_df)

        assert 'regime_shift_detected' in result
        assert 'relative_change' in result
        assert 'short_window_vol' in result
        assert 'long_window_vol' in result
        assert result['regime_shift_detected'] in [True, False]


# =============================================================================
# 9. PHASE 5 SIGNAL INTEGRATOR TESTS
# =============================================================================

class TestPhase5SignalIntegrator:
    """Test suite for Phase5SignalIntegrator."""

    def test_initialization(self):
        """Test initialization."""
        integrator = Phase5SignalIntegrator()
        assert integrator.confidence_threshold == 0.6

    def test_initialization_custom_threshold(self):
        """Test initialization with custom threshold."""
        integrator = Phase5SignalIntegrator(confidence_threshold=0.8)
        assert integrator.confidence_threshold == 0.8

    def test_integrate_phase5_signals_high_confidence(self):
        """Test signal integration with high confidence signals."""
        integrator = Phase5SignalIntegrator()

        base_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        phase5_signals = {
            'AAPL': {'composite_score': 0.8, 'confidence': 0.9},  # Bullish
            'GOOGL': {'composite_score': 0.3, 'confidence': 0.85}  # Bearish
        }

        adjusted = integrator.integrate_phase5_signals(base_weights, phase5_signals)

        # AAPL should have higher weight (bullish), GOOGL lower
        assert adjusted['AAPL'] > adjusted['GOOGL']
        # Weights should sum to ~1
        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_integrate_phase5_signals_low_confidence(self):
        """Test signal integration with low confidence (should not adjust)."""
        integrator = Phase5SignalIntegrator(confidence_threshold=0.7)

        base_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        phase5_signals = {
            'AAPL': {'composite_score': 0.9, 'confidence': 0.5},  # Low confidence
            'GOOGL': {'composite_score': 0.1, 'confidence': 0.5}
        }

        adjusted = integrator.integrate_phase5_signals(base_weights, phase5_signals)

        # Weights should remain equal due to low confidence
        assert abs(adjusted['AAPL'] - adjusted['GOOGL']) < 0.01

    def test_get_signal_summary(self):
        """Test signal summary calculation."""
        integrator = Phase5SignalIntegrator()

        phase5_signals = {
            'AAPL': {'composite_score': 0.8, 'confidence': 0.9},
            'GOOGL': {'composite_score': 0.3, 'confidence': 0.7},
            'MSFT': {'composite_score': 0.6, 'confidence': 0.8}
        }

        summary = integrator.get_signal_summary(phase5_signals)

        assert 'avg_score' in summary
        assert 'avg_confidence' in summary
        assert 'n_signals' in summary
        assert 'bullish_count' in summary
        assert 'bearish_count' in summary
        assert summary['n_signals'] == 3


# =============================================================================
# 10. OPTIMIZATION DIAGNOSTICS TESTS
# =============================================================================

class TestOptimizationDiagnostics:
    """Test suite for OptimizationDiagnostics."""

    def test_initialization(self):
        """Test initialization with defaults."""
        diag = OptimizationDiagnostics()
        assert diag.max_weight == 0.20
        assert diag.min_weight == 0.01
        assert diag.max_turnover == 0.50

    def test_initialization_custom(self):
        """Test initialization with custom values."""
        diag = OptimizationDiagnostics(max_weight=0.30, min_weight=0.02)
        assert diag.max_weight == 0.30
        assert diag.min_weight == 0.02

    def test_validate_optimization_result(self, sample_weights, sample_expected_returns, sample_covariance):
        """Test optimization validation."""
        diag = OptimizationDiagnostics()

        result = diag.validate_optimization_result(
            sample_weights, sample_expected_returns, sample_covariance
        )

        assert 'expected_return' in result
        assert 'portfolio_vol' in result
        assert 'sharpe_ratio' in result
        assert 'hhi' in result
        assert 'effective_n' in result
        assert 'is_valid' in result
        assert 'warnings' in result

    def test_validate_with_turnover(self, sample_weights, sample_expected_returns, sample_covariance):
        """Test validation with turnover calculation."""
        diag = OptimizationDiagnostics()

        current_weights = np.array([0.30, 0.30, 0.20, 0.20])

        result = diag.validate_optimization_result(
            sample_weights, sample_expected_returns, sample_covariance,
            current_weights=current_weights
        )

        assert result['turnover'] > 0

    def test_compare_portfolios(self, sample_expected_returns, sample_covariance):
        """Test portfolio comparison."""
        diag = OptimizationDiagnostics()

        weights_a = np.array([0.40, 0.30, 0.20, 0.10])
        weights_b = np.array([0.25, 0.25, 0.25, 0.25])

        comparison = diag.compare_portfolios(
            weights_a, weights_b,
            sample_expected_returns, sample_covariance,
            labels=('Concentrated', 'Equal Weight')
        )

        assert 'Concentrated' in comparison
        assert 'Equal Weight' in comparison
        assert 'return_diff' in comparison
        assert 'vol_diff' in comparison
        assert 'sharpe_diff' in comparison


# =============================================================================
# PHASE 6 IMPROVEMENTS SYSTEM TESTS
# =============================================================================

class TestPhase6ImprovementsSystem:
    """Test suite for integrated Phase6ImprovementsSystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = Phase6ImprovementsSystem()

        assert system.risk_budgeter is not None
        assert system.es_calculator is not None
        assert system.execution_selector is not None
        assert system.liquidity_risk is not None
        assert system.tax_rebalancer is not None
        assert system.tax_harvester is not None
        assert system.regime_utility is not None
        assert system.cov_estimator is not None
        assert system.phase5_integrator is not None
        assert system.diagnostics is not None

    def test_factory_function(self):
        """Test factory function."""
        system = create_phase6_improvements_system(
            base_risk_aversion=3.0,
            short_term_tax_rate=0.35,
            long_term_tax_rate=0.15
        )

        assert system is not None
        assert system.regime_utility.base_risk_aversion == 3.0

    def test_optimize_with_improvements(self, sample_historical_df, sample_expected_returns, macro_context):
        """Test full optimization with improvements."""
        system = Phase6ImprovementsSystem()

        base_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
        phase5_signals = {
            'AAPL': {'composite_score': 0.7, 'confidence': 0.8},
            'GOOGL': {'composite_score': 0.5, 'confidence': 0.7},
            'MSFT': {'composite_score': 0.6, 'confidence': 0.75},
            'AMZN': {'composite_score': 0.4, 'confidence': 0.65}
        }

        result = system.optimize_with_improvements(
            base_weights=base_weights,
            expected_returns=sample_expected_returns,
            historical_returns=sample_historical_df,
            regime='normal',
            macro_context=macro_context,
            phase5_signals=phase5_signals
        )

        assert 'optimized_weights' in result
        assert 'risk_budget' in result
        assert 'es_contributions' in result
        assert 'regime_utility' in result
        assert 'should_rebalance' in result
        assert 'diagnostics' in result


# =============================================================================
# MODULE VALIDATION TESTS
# =============================================================================

class TestModuleValidation:
    """Test module-level validation functions."""

    def test_validate_improvements_module(self):
        """Test module validation function."""
        results = validate_improvements_module()

        assert 'regime_aware_risk_budgeting' in results
        assert 'expected_shortfall' in results
        assert 'adaptive_execution' in results
        assert 'liquidity_adjusted_var' in results
        assert 'tax_aware_rebalancing' in results
        assert 'regime_aware_utility' in results
        assert 'robust_covariance' in results
        assert 'phase5_integration' in results
        assert 'optimization_diagnostics' in results

        # All should pass
        passed = sum(1 for v in results.values() if v == True)
        total = len(results)
        assert passed == total, f"Only {passed}/{total} validations passed"


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_market_regime_enum(self):
        """Test MarketRegime enum."""
        assert MarketRegime.CRISIS.value == 'crisis'
        assert MarketRegime.HIGH_VOL.value == 'high_vol'
        assert MarketRegime.NORMAL.value == 'normal'
        assert MarketRegime.LOW_VOL.value == 'low_vol'

    def test_execution_strategy_enum(self):
        """Test ExecutionStrategy enum."""
        assert ExecutionStrategy.IMMEDIATE.value == 'immediate'
        assert ExecutionStrategy.TWAP.value == 'twap'
        assert ExecutionStrategy.VWAP.value == 'vwap'
        assert ExecutionStrategy.ADAPTIVE.value == 'adaptive'

    def test_urgency_level_enum(self):
        """Test UrgencyLevel enum."""
        assert UrgencyLevel.LOW.value == 'low'
        assert UrgencyLevel.MEDIUM.value == 'medium'
        assert UrgencyLevel.HIGH.value == 'high'
        assert UrgencyLevel.CRITICAL.value == 'critical'

    def test_harvest_action_enum(self):
        """Test HarvestAction enum."""
        assert HarvestAction.HARVEST.value == 'harvest'
        assert HarvestAction.HOLD.value == 'hold'
        assert HarvestAction.DEFER.value == 'defer'


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfigurations:
    """Test configuration dataclasses."""

    def test_regime_config_defaults(self):
        """Test RegimeConfig default values."""
        config = RegimeConfig()

        assert config.risk_multipliers['crisis'] == 0.5
        assert config.risk_multipliers['normal'] == 1.0
        assert config.lookback_periods['crisis'] == 21
        assert config.shrinkage_factors['crisis'] == 0.4

    def test_tax_config_defaults(self):
        """Test TaxConfig default values."""
        config = TaxConfig()

        assert config.short_term_rate == 0.37
        assert config.long_term_rate == 0.20
        assert config.transaction_cost_rate == 0.001

    def test_harvest_opportunity_creation(self):
        """Test HarvestOpportunity creation."""
        opp = HarvestOpportunity(
            ticker='AAPL',
            total_loss=-1000,
            is_short_term=True,
            action=HarvestAction.HARVEST,
            days_held=30
        )

        assert opp.ticker == 'AAPL'
        assert opp.total_loss == -1000
        assert opp.is_short_term is True
        assert opp.action == HarvestAction.HARVEST


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_weights(self):
        """Test with zero weights."""
        budgeter = RegimeAwareRiskBudgeter()
        weights = np.array([0.0, 0.0, 0.0, 0.0])
        cov = np.eye(4) * 0.04

        risk_contrib = budgeter.calculate_risk_contribution_percent(weights, cov)
        assert all(r == 0 for r in risk_contrib)

    def test_single_asset(self):
        """Test with single asset portfolio."""
        budgeter = RegimeAwareRiskBudgeter()
        weights = np.array([1.0])
        cov = np.array([[0.04]])

        result = budgeter.calculate_regime_aware_risk_budget(
            weights, cov, 'normal', {'vix': 15}
        )
        assert result['total_risk_budget'] > 0

    def test_empty_phase5_signals(self):
        """Test with empty Phase 5 signals."""
        integrator = Phase5SignalIntegrator()

        base_weights = {'AAPL': 0.5, 'GOOGL': 0.5}

        adjusted = integrator.integrate_phase5_signals(base_weights, {})

        assert adjusted['AAPL'] == 0.5
        assert adjusted['GOOGL'] == 0.5

    def test_extreme_vix_values(self):
        """Test with extreme VIX values."""
        budgeter = RegimeAwareRiskBudgeter()
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        cov = np.eye(4) * 0.04

        # Very low VIX
        result_low = budgeter.calculate_regime_aware_risk_budget(
            weights, cov, 'normal', {'vix': 5}
        )
        assert result_low['vix_adjustment'] >= 1.0

        # Very high VIX
        result_high = budgeter.calculate_regime_aware_risk_budget(
            weights, cov, 'normal', {'vix': 80}
        )
        assert result_high['vix_adjustment'] <= 1.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
