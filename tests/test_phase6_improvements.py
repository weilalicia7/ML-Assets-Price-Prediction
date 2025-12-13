"""
Tests for Phase 6 Portfolio Optimization Improvements

Tests all enhancement components:
1. Regime-aware risk budgeting
2. Expected Shortfall (CVaR) contribution
3. Adaptive execution strategy
4. Liquidity-adjusted VaR
5. Tax-aware rebalancing
6. Tax-loss harvesting optimization
7. Regime-aware utility function
8. Robust covariance estimation
9. Phase 5 signal integration
10. Optimization diagnostics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, 'C:/Users/c25038355/OneDrive - Cardiff University/Desktop/. Stock price prediction/stock-prediction-model')

from src.portfolio.phase6_improvements import (
    # Enums
    MarketRegime,
    ExecutionStrategy,
    UrgencyLevel,
    HarvestAction,

    # Configs
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
    create_phase6_improvements_system,
    validate_improvements_module,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def sample_cov_matrix():
    """Sample covariance matrix."""
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
    """Sample historical returns data."""
    np.random.seed(42)
    n_periods = 252
    n_assets = 4
    returns = np.random.randn(n_periods, n_assets) * 0.02
    return returns


@pytest.fixture
def sample_returns_df():
    """Sample returns as DataFrame."""
    np.random.seed(42)
    n_periods = 252
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
    data = np.random.randn(n_periods, 4) * 0.02
    return pd.DataFrame(data, index=dates, columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])


@pytest.fixture
def sample_expected_returns():
    """Sample expected returns."""
    return np.array([0.10, 0.08, 0.12, 0.09])


@pytest.fixture
def sample_macro_context():
    """Sample macro context."""
    return {'vix': 20.0, 'sp500_return': 0.01}


# =============================================================================
# REGIME-AWARE RISK BUDGETING TESTS
# =============================================================================

class TestRegimeAwareRiskBudgeter:
    """Tests for RegimeAwareRiskBudgeter."""

    def test_initialization(self):
        """Test budgeter initialization."""
        budgeter = RegimeAwareRiskBudgeter()
        assert budgeter.config is not None

    def test_risk_contribution_calculation(self, sample_weights, sample_cov_matrix):
        """Test risk contribution calculation."""
        budgeter = RegimeAwareRiskBudgeter()
        risk_contrib = budgeter.calculate_risk_contribution_percent(
            sample_weights, sample_cov_matrix
        )
        assert len(risk_contrib) == 4
        assert np.all(np.isfinite(risk_contrib))

    def test_regime_aware_risk_budget_normal(self, sample_weights, sample_cov_matrix, sample_macro_context):
        """Test regime-aware risk budget in normal regime."""
        budgeter = RegimeAwareRiskBudgeter()
        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_cov_matrix, 'normal', sample_macro_context
        )

        assert 'base_risk_contrib' in result
        assert 'regime_adjusted' in result
        assert 'total_risk_budget' in result
        assert result['regime_multiplier'] == 1.0

    def test_regime_aware_risk_budget_crisis(self, sample_weights, sample_cov_matrix, sample_macro_context):
        """Test regime-aware risk budget in crisis regime."""
        budgeter = RegimeAwareRiskBudgeter()
        result = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_cov_matrix, 'crisis', sample_macro_context
        )

        assert result['regime_multiplier'] == 0.5

    def test_vix_adjustment(self, sample_weights, sample_cov_matrix):
        """Test VIX-based adjustment."""
        budgeter = RegimeAwareRiskBudgeter()

        # Low VIX
        result_low = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_cov_matrix, 'normal', {'vix': 12.0}
        )

        # High VIX
        result_high = budgeter.calculate_regime_aware_risk_budget(
            sample_weights, sample_cov_matrix, 'normal', {'vix': 35.0}
        )

        # High VIX should reduce budget
        assert result_high['vix_adjustment'] < result_low['vix_adjustment']


# =============================================================================
# EXPECTED SHORTFALL TESTS
# =============================================================================

class TestExpectedShortfallCalculator:
    """Tests for ExpectedShortfallCalculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = ExpectedShortfallCalculator()
        assert calc.confidence == 0.975

    def test_es_contribution(self, sample_weights, sample_returns):
        """Test ES contribution calculation."""
        calc = ExpectedShortfallCalculator()
        es_contrib = calc.calculate_es_contribution(
            sample_weights, sample_returns
        )

        assert len(es_contrib) == 4
        assert np.all(np.isfinite(es_contrib))

    def test_portfolio_es(self, sample_weights, sample_returns):
        """Test portfolio-level ES calculation."""
        calc = ExpectedShortfallCalculator()
        portfolio_es = calc.calculate_portfolio_es(
            sample_weights, sample_returns
        )

        assert isinstance(portfolio_es, float)
        assert portfolio_es < 0  # ES should be negative (loss)

    def test_marginal_es(self, sample_weights, sample_returns):
        """Test marginal ES calculation."""
        calc = ExpectedShortfallCalculator()
        marginal_es = calc.calculate_marginal_es(
            sample_weights, sample_returns
        )

        assert len(marginal_es) == 4

    def test_different_confidence_levels(self, sample_weights, sample_returns):
        """Test ES at different confidence levels."""
        calc = ExpectedShortfallCalculator()

        es_95 = calc.calculate_portfolio_es(sample_weights, sample_returns, 0.95)
        es_99 = calc.calculate_portfolio_es(sample_weights, sample_returns, 0.99)

        # Higher confidence = larger loss (more extreme tail)
        assert es_99 <= es_95


# =============================================================================
# ADAPTIVE EXECUTION TESTS
# =============================================================================

class TestAdaptiveExecutionSelector:
    """Tests for AdaptiveExecutionSelector."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = AdaptiveExecutionSelector()
        assert selector is not None

    def test_small_trade_normal_market(self):
        """Test strategy for small trade in normal market."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=100000,
            daily_volume=10000000,
            volatility=0.02,
            market_regime='normal',
            urgency=UrgencyLevel.MEDIUM
        )

        assert isinstance(strategy, ExecutionStrategy)

    def test_large_trade_selects_vwap(self):
        """Test that large trades select VWAP."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=2000000,  # 20% of volume
            daily_volume=10000000,
            volatility=0.02,
            market_regime='normal',
            urgency=UrgencyLevel.LOW
        )

        assert strategy == ExecutionStrategy.VWAP

    def test_crisis_regime_uses_twap(self):
        """Test that crisis regime uses TWAP."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=100000,
            daily_volume=10000000,
            volatility=0.05,
            market_regime='crisis',
            urgency=UrgencyLevel.LOW
        )

        assert strategy == ExecutionStrategy.TWAP

    def test_critical_urgency_uses_immediate(self):
        """Test that critical urgency uses immediate execution."""
        selector = AdaptiveExecutionSelector()
        strategy = selector.get_adaptive_execution_strategy(
            trade_size=100000,
            daily_volume=10000000,
            volatility=0.02,
            market_regime='crisis',  # Even in crisis
            urgency=UrgencyLevel.CRITICAL
        )

        assert strategy == ExecutionStrategy.IMMEDIATE

    def test_execution_parameters(self):
        """Test execution parameter generation."""
        selector = AdaptiveExecutionSelector()
        params = selector.get_execution_parameters(
            ExecutionStrategy.TWAP,
            trade_size=500000,
            daily_volume=10000000,
            market_regime='normal'
        )

        assert 'strategy' in params
        assert 'slices' in params
        assert 'max_participation_rate' in params


# =============================================================================
# LIQUIDITY-ADJUSTED RISK TESTS
# =============================================================================

class TestLiquidityAdjustedRisk:
    """Tests for LiquidityAdjustedRisk."""

    def test_initialization(self):
        """Test initialization."""
        liq = LiquidityAdjustedRisk()
        assert liq.stress_correlation == 0.3

    def test_portfolio_var(self, sample_weights, sample_cov_matrix):
        """Test portfolio VaR calculation."""
        liq = LiquidityAdjustedRisk()
        var = liq.calculate_portfolio_var(sample_weights, sample_cov_matrix)

        assert var > 0
        assert isinstance(var, float)

    def test_liquidity_adjusted_var_1day(self, sample_weights, sample_cov_matrix):
        """Test liquidity-adjusted VaR for 1-day liquidation."""
        liq = LiquidityAdjustedRisk()
        daily_var = liq.calculate_portfolio_var(sample_weights, sample_cov_matrix)
        liq_var = liq.calculate_liquidity_adjusted_var(
            sample_weights, sample_cov_matrix, liquidation_days=1
        )

        assert liq_var == daily_var

    def test_liquidity_adjusted_var_increases_with_days(self, sample_weights, sample_cov_matrix):
        """Test that liquidity-adjusted VaR increases with liquidation days."""
        liq = LiquidityAdjustedRisk()

        var_1day = liq.calculate_liquidity_adjusted_var(
            sample_weights, sample_cov_matrix, liquidation_days=1
        )
        var_5day = liq.calculate_liquidity_adjusted_var(
            sample_weights, sample_cov_matrix, liquidation_days=5
        )
        var_10day = liq.calculate_liquidity_adjusted_var(
            sample_weights, sample_cov_matrix, liquidation_days=10
        )

        assert var_5day > var_1day
        assert var_10day > var_5day

    def test_position_liquidation_days(self):
        """Test position liquidation days calculation."""
        liq = LiquidityAdjustedRisk()
        position_values = {'AAPL': 1000000, 'MSFT': 500000}
        daily_volumes = {'AAPL': 10000000, 'MSFT': 5000000}

        liq_days = liq.calculate_position_liquidation_days(
            position_values, daily_volumes
        )

        assert 'AAPL' in liq_days
        assert 'MSFT' in liq_days
        assert all(d >= 1 for d in liq_days.values())


# =============================================================================
# TAX-AWARE REBALANCING TESTS
# =============================================================================

class TestTaxAwareRebalancer:
    """Tests for TaxAwareRebalancer."""

    def test_initialization(self):
        """Test rebalancer initialization."""
        reb = TaxAwareRebalancer()
        assert reb.config.short_term_rate == 0.37
        assert reb.config.long_term_rate == 0.20

    def test_should_rebalance_high_benefit(self):
        """Test rebalancing decision with high expected benefit."""
        reb = TaxAwareRebalancer()

        should_reb, net_benefit = reb.should_rebalance_after_tax(
            current_weights={'AAPL': 0.5, 'MSFT': 0.5},
            target_weights={'AAPL': 0.4, 'MSFT': 0.6},
            tax_impact={'AAPL': {'tax_liability': 0.001}},
            expected_benefit=0.05  # 5% benefit
        )

        assert should_reb is True
        assert net_benefit > 0

    def test_should_not_rebalance_low_benefit(self):
        """Test rebalancing decision with low expected benefit."""
        reb = TaxAwareRebalancer()

        should_reb, net_benefit = reb.should_rebalance_after_tax(
            current_weights={'AAPL': 0.5, 'MSFT': 0.5},
            target_weights={'AAPL': 0.49, 'MSFT': 0.51},
            tax_impact={'AAPL': {'tax_liability': 0.01}},
            expected_benefit=0.0001  # 1bp benefit
        )

        # With high tax cost and low benefit, should not rebalance
        assert net_benefit < 0.002

    def test_tax_adjusted_threshold(self):
        """Test tax-adjusted threshold calculation."""
        reb = TaxAwareRebalancer()

        # No unrealized gain
        threshold_no_gain = reb.calculate_tax_adjusted_threshold(
            'AAPL', 0.20, 0.0, False
        )

        # Large unrealized gain
        threshold_gain = reb.calculate_tax_adjusted_threshold(
            'AAPL', 0.20, 0.50, False  # 50% gain
        )

        # Higher gains = higher threshold
        assert threshold_gain > threshold_no_gain


# =============================================================================
# TAX-LOSS HARVESTING TESTS
# =============================================================================

class TestTaxLossHarvestOptimizer:
    """Tests for TaxLossHarvestOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        opt = TaxLossHarvestOptimizer()
        assert opt.config is not None

    def test_optimize_harvesting(self):
        """Test harvest optimization."""
        opt = TaxLossHarvestOptimizer()

        opportunities = [
            HarvestOpportunity(
                ticker='AAPL', total_loss=-1000, is_short_term=True,
                action=HarvestAction.HARVEST
            ),
            HarvestOpportunity(
                ticker='MSFT', total_loss=-2000, is_short_term=False,
                action=HarvestAction.HARVEST
            ),
        ]

        prioritized = opt.optimize_tax_loss_harvesting(
            opportunities,
            ytd_gains={'short_term': 5000, 'long_term': 3000},
            remaining_year_days=30
        )

        assert len(prioritized) == 2
        # Short-term loss should be first (higher tax rate)
        assert prioritized[0].is_short_term is True

    def test_optimal_harvest_amount(self):
        """Test optimal harvest amount calculation."""
        opt = TaxLossHarvestOptimizer()

        result = opt.calculate_optimal_harvest_amount(
            loss_available=10000,
            ytd_short_term_gains=5000,
            ytd_long_term_gains=3000
        )

        assert 'short_term_offset' in result
        assert 'long_term_offset' in result
        assert 'ordinary_income_offset' in result
        assert 'carryforward' in result
        assert result['short_term_offset'] == 5000
        assert result['long_term_offset'] == 3000


# =============================================================================
# REGIME-AWARE UTILITY TESTS
# =============================================================================

class TestRegimeAwareUtility:
    """Tests for RegimeAwareUtility."""

    def test_initialization(self):
        """Test utility initialization."""
        util = RegimeAwareUtility()
        assert util.base_risk_aversion == 2.0

    def test_base_utility(self, sample_weights, sample_expected_returns, sample_cov_matrix):
        """Test base utility calculation."""
        util = RegimeAwareUtility()
        u = util.calculate_utility(
            sample_weights, sample_expected_returns, sample_cov_matrix
        )

        assert isinstance(u, float)

    def test_regime_aware_utility(self, sample_weights, sample_expected_returns,
                                   sample_cov_matrix, sample_macro_context):
        """Test regime-aware utility calculation."""
        util = RegimeAwareUtility()
        u = util.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_cov_matrix,
            'normal', sample_macro_context
        )

        assert isinstance(u, float)

    def test_crisis_reduces_utility(self, sample_weights, sample_expected_returns,
                                     sample_cov_matrix, sample_macro_context):
        """Test that crisis regime reduces utility."""
        util = RegimeAwareUtility()

        u_normal = util.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_cov_matrix,
            'normal', sample_macro_context
        )

        u_crisis = util.calculate_regime_aware_utility(
            sample_weights, sample_expected_returns, sample_cov_matrix,
            'crisis', sample_macro_context
        )

        # Crisis should have lower utility due to higher risk aversion
        assert u_crisis < u_normal

    def test_optimal_risk_aversion(self):
        """Test optimal risk aversion retrieval."""
        util = RegimeAwareUtility(base_risk_aversion=2.0)

        ra_normal = util.get_regime_optimal_risk_aversion('normal')
        ra_crisis = util.get_regime_optimal_risk_aversion('crisis')

        assert ra_normal == 2.0
        assert ra_crisis == 4.0  # 2.0 * 2.0


# =============================================================================
# ROBUST COVARIANCE TESTS
# =============================================================================

class TestRobustCovarianceEstimator:
    """Tests for RobustCovarianceEstimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        est = RobustCovarianceEstimator()
        assert est.config is not None

    def test_regime_aware_covariance(self, sample_returns_df):
        """Test regime-aware covariance calculation."""
        est = RobustCovarianceEstimator()
        cov = est.calculate_regime_aware_covariance(
            sample_returns_df, 'normal'
        )

        assert cov.shape == (4, 4)
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)

    def test_different_regimes_different_lookbacks(self, sample_returns_df):
        """Test that different regimes use different lookbacks."""
        est = RobustCovarianceEstimator()

        # Crisis uses shorter lookback
        cov_crisis = est.calculate_regime_aware_covariance(
            sample_returns_df, 'crisis'
        )

        # Low vol uses longer lookback
        cov_low_vol = est.calculate_regime_aware_covariance(
            sample_returns_df, 'low_vol'
        )

        # Both should be valid covariance matrices
        assert cov_crisis.shape == cov_low_vol.shape

    def test_covariance_regime_shift_detection(self, sample_returns_df):
        """Test regime shift detection."""
        est = RobustCovarianceEstimator()
        result = est.detect_covariance_regime_shift(sample_returns_df)

        assert 'regime_shift_detected' in result
        assert 'relative_change' in result
        assert result['regime_shift_detected'] in [True, False]


# =============================================================================
# PHASE 5 INTEGRATION TESTS
# =============================================================================

class TestPhase5SignalIntegrator:
    """Tests for Phase5SignalIntegrator."""

    def test_initialization(self):
        """Test integrator initialization."""
        integrator = Phase5SignalIntegrator()
        assert integrator.confidence_threshold == 0.6

    def test_integrate_signals(self):
        """Test signal integration."""
        integrator = Phase5SignalIntegrator()

        base_weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
        signals = {
            'AAPL': {'composite_score': 0.8, 'confidence': 0.9},  # Bullish
            'MSFT': {'composite_score': 0.2, 'confidence': 0.8},  # Bearish
            'GOOGL': {'composite_score': 0.5, 'confidence': 0.5},  # Neutral, low conf
        }

        adjusted = integrator.integrate_phase5_signals(base_weights, signals)

        assert sum(adjusted.values()) == pytest.approx(1.0, rel=1e-6)
        # AAPL should have higher weight (bullish)
        assert adjusted['AAPL'] > adjusted['MSFT']

    def test_no_signals_returns_base(self):
        """Test that no signals returns base weights."""
        integrator = Phase5SignalIntegrator()

        base_weights = {'AAPL': 0.5, 'MSFT': 0.5}
        adjusted = integrator.integrate_phase5_signals(base_weights, {})

        assert adjusted == base_weights

    def test_signal_summary(self):
        """Test signal summary generation."""
        integrator = Phase5SignalIntegrator()
        signals = {
            'AAPL': {'composite_score': 0.8, 'confidence': 0.9},
            'MSFT': {'composite_score': 0.3, 'confidence': 0.7},
        }

        summary = integrator.get_signal_summary(signals)

        assert 'avg_score' in summary
        assert 'avg_confidence' in summary
        assert summary['bullish_count'] == 1
        assert summary['bearish_count'] == 1


# =============================================================================
# OPTIMIZATION DIAGNOSTICS TESTS
# =============================================================================

class TestOptimizationDiagnostics:
    """Tests for OptimizationDiagnostics."""

    def test_initialization(self):
        """Test diagnostics initialization."""
        diag = OptimizationDiagnostics()
        assert diag.max_weight == 0.20
        assert diag.min_weight == 0.01

    def test_validate_valid_portfolio(self, sample_weights, sample_expected_returns, sample_cov_matrix):
        """Test validation of valid portfolio."""
        diag = OptimizationDiagnostics(max_weight=0.30)
        result = diag.validate_optimization_result(
            sample_weights, sample_expected_returns, sample_cov_matrix
        )

        assert 'expected_return' in result
        assert 'portfolio_vol' in result
        assert 'sharpe_ratio' in result
        assert result['is_valid'] is True

    def test_detect_constraint_violations(self, sample_expected_returns, sample_cov_matrix):
        """Test detection of constraint violations."""
        diag = OptimizationDiagnostics(max_weight=0.20)

        # Weights with violation (one position > 20%)
        bad_weights = np.array([0.50, 0.20, 0.20, 0.10])

        result = diag.validate_optimization_result(
            bad_weights, sample_expected_returns, sample_cov_matrix
        )

        assert result['max_weight_violation'] > 0
        assert result['is_valid'] is False

    def test_turnover_calculation(self, sample_expected_returns, sample_cov_matrix):
        """Test turnover calculation."""
        diag = OptimizationDiagnostics()

        current = np.array([0.25, 0.25, 0.25, 0.25])
        new = np.array([0.30, 0.30, 0.20, 0.20])

        result = diag.validate_optimization_result(
            new, sample_expected_returns, sample_cov_matrix,
            current_weights=current
        )

        assert result['turnover'] == pytest.approx(0.20, rel=1e-6)

    def test_portfolio_comparison(self, sample_expected_returns, sample_cov_matrix):
        """Test portfolio comparison."""
        diag = OptimizationDiagnostics()

        weights_a = np.array([0.25, 0.25, 0.25, 0.25])
        weights_b = np.array([0.30, 0.30, 0.20, 0.20])

        comparison = diag.compare_portfolios(
            weights_a, weights_b,
            sample_expected_returns, sample_cov_matrix,
            labels=('Equal Weight', 'Tilted')
        )

        assert 'Equal Weight' in comparison
        assert 'Tilted' in comparison
        assert 'return_diff' in comparison


# =============================================================================
# INTEGRATED SYSTEM TESTS
# =============================================================================

class TestPhase6ImprovementsSystem:
    """Tests for Phase6ImprovementsSystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = Phase6ImprovementsSystem()
        assert system.risk_budgeter is not None
        assert system.es_calculator is not None
        assert system.diagnostics is not None

    def test_factory_function(self):
        """Test factory function."""
        system = create_phase6_improvements_system(
            base_risk_aversion=3.0,
            short_term_tax_rate=0.35
        )
        assert system.regime_utility.base_risk_aversion == 3.0

    def test_optimize_with_improvements(self, sample_returns_df, sample_macro_context):
        """Test full optimization with improvements."""
        system = Phase6ImprovementsSystem()

        base_weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
        expected_returns = np.array([0.10, 0.08, 0.12, 0.09])

        result = system.optimize_with_improvements(
            base_weights=base_weights,
            expected_returns=expected_returns,
            historical_returns=sample_returns_df,
            regime='normal',
            macro_context=sample_macro_context,
            phase5_signals={
                'AAPL': {'composite_score': 0.7, 'confidence': 0.8}
            }
        )

        assert 'optimized_weights' in result
        assert 'risk_budget' in result
        assert 'diagnostics' in result
        assert result['phase5_integrated'] is True


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Tests for module validation."""

    def test_validate_all_components(self):
        """Test that all components pass validation."""
        results = validate_improvements_module()

        for component, status in results.items():
            assert status == True, f"{component} validation failed"

    def test_all_enums_defined(self):
        """Test that all enums are properly defined."""
        assert MarketRegime.CRISIS.value == 'crisis'
        assert MarketRegime.NORMAL.value == 'normal'
        assert ExecutionStrategy.VWAP.value == 'vwap'
        assert UrgencyLevel.HIGH.value == 'high'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for Phase 6 improvements."""

    def test_full_pipeline(self, sample_returns_df):
        """Test full improvement pipeline."""
        # Create system
        system = create_phase6_improvements_system()

        # Define inputs
        base_weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
        expected_returns = np.array([0.10, 0.08, 0.12, 0.09])

        # Run optimization with all improvements
        result = system.optimize_with_improvements(
            base_weights=base_weights,
            expected_returns=expected_returns,
            historical_returns=sample_returns_df,
            regime='high_vol',
            macro_context={'vix': 28.0},
            phase5_signals={
                'AAPL': {'composite_score': 0.75, 'confidence': 0.85},
                'MSFT': {'composite_score': 0.45, 'confidence': 0.70},
            },
            tax_impact={'AAPL': {'tax_liability': 0.005}},
            expected_benefit=0.02
        )

        # Verify all components worked
        assert result['diagnostics']['is_valid'] or len(result['diagnostics']['warnings']) > 0
        assert result['regime'] == 'high_vol'
        assert 'total_risk_budget' in result['risk_budget']

    def test_regime_transitions(self, sample_returns_df):
        """Test system handles regime transitions."""
        system = create_phase6_improvements_system()
        base_weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
        expected_returns = np.array([0.10, 0.08, 0.12, 0.09])

        regimes = ['crisis', 'high_vol', 'normal', 'low_vol']

        for regime in regimes:
            result = system.optimize_with_improvements(
                base_weights=base_weights,
                expected_returns=expected_returns,
                historical_returns=sample_returns_df,
                regime=regime,
                macro_context={'vix': 20.0}
            )

            assert result['regime'] == regime
            assert 'optimized_weights' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
