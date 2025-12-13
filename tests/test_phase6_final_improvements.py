"""
Phase 6 Final Improvements Test Suite

Tests for production-grade enhancements:
- PortfolioOptimizationMonitor
- MultiTimeframeOptimizer
- AdaptiveConstraintManager
- CrossPortfolioCorrelationManager
- PortfolioStressTester
- Phase6FinalImprovementsSystem
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio.phase6_final_improvements import (
    # Core Classes
    PortfolioOptimizationMonitor,
    MultiTimeframeOptimizer,
    AdaptiveConstraintManager,
    CrossPortfolioCorrelationManager,
    PortfolioStressTester,

    # Main System
    Phase6FinalImprovementsSystem,

    # Factories
    create_phase6_final_system,
    create_portfolio_monitor,
    create_stress_tester,

    # Checklist
    PHASE6_FINAL_PRODUCTION_CHECKLIST,

    # Validation
    validate_final_improvements,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_weights():
    """Standard portfolio weights."""
    return {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}


@pytest.fixture
def sample_covariance():
    """Sample 4x4 covariance matrix."""
    return np.array([
        [0.04, 0.01, 0.005, 0.002],
        [0.01, 0.03, 0.008, 0.003],
        [0.005, 0.008, 0.025, 0.004],
        [0.002, 0.003, 0.004, 0.02]
    ])


@pytest.fixture
def sample_expected_returns():
    """Expected returns for 4 assets."""
    return {'AAPL': 0.12, 'GOOGL': 0.10, 'MSFT': 0.11, 'AMZN': 0.09}


@pytest.fixture
def macro_context():
    """Sample macro context."""
    return {'vix': 20.0, 'interest_rate': 0.05}


@pytest.fixture
def asset_betas():
    """Sample asset betas for stress testing."""
    return {
        'AAPL': {'equity_shock': 1.2, 'volatility_spike': 0.8},
        'GOOGL': {'equity_shock': 1.1, 'volatility_spike': 0.9},
        'MSFT': {'equity_shock': 1.0, 'volatility_spike': 0.7},
        'AMZN': {'equity_shock': 1.3, 'volatility_spike': 1.0}
    }


# =============================================================================
# 1. PORTFOLIO OPTIMIZATION MONITOR TESTS
# =============================================================================

class TestPortfolioOptimizationMonitor:
    """Test suite for PortfolioOptimizationMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PortfolioOptimizationMonitor()
        assert monitor.performance_history == []
        assert monitor.regime_performance == {}

    def test_calculate_realized_return(self, sample_weights):
        """Test realized return calculation."""
        monitor = PortfolioOptimizationMonitor()
        actual_returns = {'AAPL': 0.05, 'GOOGL': 0.03, 'MSFT': 0.04, 'AMZN': 0.02}
        opt_result = {'optimized_weights': sample_weights}

        realized = monitor.calculate_realized_return(opt_result, actual_returns)

        expected = 0.25 * 0.05 + 0.25 * 0.03 + 0.25 * 0.04 + 0.25 * 0.02
        assert abs(realized - expected) < 1e-10

    def test_track_optimization_performance(self, sample_weights):
        """Test performance tracking."""
        monitor = PortfolioOptimizationMonitor()

        opt_result = {
            'optimized_weights': sample_weights,
            'expected_return': 0.03,
            'sharpe_ratio': 1.5,
            'turnover': 0.10
        }
        actual_returns = {'AAPL': 0.04, 'GOOGL': 0.02, 'MSFT': 0.03, 'AMZN': 0.01}

        result = monitor.track_optimization_performance(
            opt_result, actual_returns, 'normal', 0.001
        )

        assert 'current_performance' in result
        assert 'regime_stats' in result
        assert 'needs_recalibration' in result
        assert len(monitor.performance_history) == 1
        assert 'normal' in monitor.regime_performance

    def test_should_recalibrate_no_data(self):
        """Test recalibration check with no data."""
        monitor = PortfolioOptimizationMonitor()
        stats = {'avg_forecast_error': 0.0, 'forecast_error_std': 0.0}

        should_recal = monitor.should_recalibrate(stats)
        assert isinstance(should_recal, bool)

    def test_should_recalibrate_high_error(self):
        """Test recalibration triggered by high error."""
        monitor = PortfolioOptimizationMonitor()
        stats = {'avg_forecast_error': 0.05, 'forecast_error_std': 0.02}

        should_recal = monitor.should_recalibrate(stats)
        assert should_recal is True

    def test_get_default_parameters(self):
        """Test default parameter retrieval."""
        monitor = PortfolioOptimizationMonitor()

        params = monitor.get_default_parameters('crisis')
        assert 'risk_aversion_multiplier' in params
        assert params['risk_aversion_multiplier'] == 2.0

        params_normal = monitor.get_default_parameters('normal')
        assert params_normal['risk_aversion_multiplier'] == 1.0

    def test_get_adaptive_parameters_insufficient_data(self):
        """Test adaptive parameters with insufficient data."""
        monitor = PortfolioOptimizationMonitor()

        params = monitor.get_adaptive_parameters('normal')
        # Should return default parameters
        assert 'risk_aversion_multiplier' in params

    def test_get_performance_summary_no_data(self):
        """Test performance summary with no data."""
        monitor = PortfolioOptimizationMonitor()
        summary = monitor.get_performance_summary()
        assert summary['status'] == 'no_data'

    def test_get_performance_summary_with_data(self, sample_weights):
        """Test performance summary with data."""
        monitor = PortfolioOptimizationMonitor()

        # Add some data
        for _ in range(5):
            monitor.track_optimization_performance(
                {'optimized_weights': sample_weights, 'expected_return': 0.02},
                {'AAPL': 0.03, 'GOOGL': 0.02, 'MSFT': 0.01, 'AMZN': 0.02},
                'normal', 0.001
            )

        summary = monitor.get_performance_summary()
        assert 'total_observations' in summary
        assert summary['total_observations'] == 5


# =============================================================================
# 2. MULTI-TIMEFRAME OPTIMIZER TESTS
# =============================================================================

class TestMultiTimeframeOptimizer:
    """Test suite for MultiTimeframeOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = MultiTimeframeOptimizer()
        assert 'tactical' in optimizer.timeframe_configs
        assert 'strategic' in optimizer.timeframe_configs
        assert 'structural' in optimizer.timeframe_configs

    def test_calculate_blending_weights_normal(self):
        """Test blending weights for normal regime."""
        optimizer = MultiTimeframeOptimizer()
        weights = optimizer.calculate_blending_weights('normal')

        assert 'tactical' in weights
        assert 'strategic' in weights
        assert 'structural' in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_calculate_blending_weights_crisis(self):
        """Test blending weights for crisis regime."""
        optimizer = MultiTimeframeOptimizer()
        weights = optimizer.calculate_blending_weights('crisis')

        # In crisis, strategic should have higher weight
        assert weights['strategic'] >= weights['tactical']

    def test_optimize_for_timeframe(self, sample_weights, sample_expected_returns, sample_covariance, macro_context):
        """Test single timeframe optimization."""
        optimizer = MultiTimeframeOptimizer()
        tickers = list(sample_weights.keys())
        config = optimizer.timeframe_configs['tactical']

        weights, diagnostics = optimizer.optimize_for_timeframe(
            sample_weights, sample_expected_returns, sample_covariance,
            tickers, config, 'normal', macro_context
        )

        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert 'expected_return' in diagnostics
        assert 'sharpe_ratio' in diagnostics

    def test_blend_timeframe_weights(self, sample_weights, macro_context):
        """Test weight blending across timeframes."""
        optimizer = MultiTimeframeOptimizer()

        timeframe_results = {
            'tactical': {'weights': {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.1}},
            'strategic': {'weights': {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}},
            'structural': {'weights': {'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.3, 'AMZN': 0.3}}
        }

        blended = optimizer.blend_timeframe_weights(timeframe_results, 'normal', macro_context)

        assert len(blended) == 4
        assert abs(sum(blended.values()) - 1.0) < 0.01

    def test_optimize_multi_timeframe(self, sample_weights, sample_covariance, macro_context):
        """Test full multi-timeframe optimization."""
        optimizer = MultiTimeframeOptimizer()

        expected_returns = {
            'tactical': {'AAPL': 0.15, 'GOOGL': 0.12, 'MSFT': 0.13, 'AMZN': 0.10},
            'strategic': {'AAPL': 0.10, 'GOOGL': 0.09, 'MSFT': 0.11, 'AMZN': 0.08},
            'structural': {'AAPL': 0.08, 'GOOGL': 0.07, 'MSFT': 0.09, 'AMZN': 0.06}
        }
        covariance_matrices = {
            'tactical': sample_covariance,
            'strategic': sample_covariance,
            'structural': sample_covariance
        }

        result = optimizer.optimize_multi_timeframe(
            sample_weights, expected_returns, covariance_matrices,
            'normal', macro_context
        )

        assert 'timeframe_results' in result
        assert 'blended_weights' in result
        assert 'blending_weights' in result


# =============================================================================
# 3. ADAPTIVE CONSTRAINT MANAGER TESTS
# =============================================================================

class TestAdaptiveConstraintManager:
    """Test suite for AdaptiveConstraintManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AdaptiveConstraintManager()
        assert 'max_single_position' in manager.base_constraints
        assert 'max_turnover' in manager.base_constraints

    def test_get_regime_aware_constraints_normal(self, macro_context):
        """Test constraints for normal regime."""
        manager = AdaptiveConstraintManager()

        constraints = manager.get_regime_aware_constraints(
            'normal', macro_context, 1_000_000, {'hhi': 0.10}
        )

        assert constraints['max_single_position'] == 0.20
        assert constraints['max_turnover'] == 0.15

    def test_get_regime_aware_constraints_crisis(self, macro_context):
        """Test constraints for crisis regime."""
        manager = AdaptiveConstraintManager()

        constraints = manager.get_regime_aware_constraints(
            'crisis', macro_context, 1_000_000, {'hhi': 0.10}
        )

        # Crisis should have tighter constraints
        assert constraints['max_single_position'] < 0.20
        assert constraints['max_turnover'] < 0.15

    def test_get_regime_aware_constraints_high_vix(self):
        """Test constraints with high VIX."""
        manager = AdaptiveConstraintManager()
        high_vix_context = {'vix': 40.0}

        constraints = manager.get_regime_aware_constraints(
            'normal', high_vix_context, 1_000_000, {'hhi': 0.10}
        )

        # High VIX should tighten constraints
        assert constraints['max_single_position'] < 0.20

    def test_get_regime_aware_constraints_large_portfolio(self, macro_context):
        """Test constraints for large portfolio."""
        manager = AdaptiveConstraintManager()

        constraints = manager.get_regime_aware_constraints(
            'normal', macro_context, 50_000_000, {'hhi': 0.10}
        )

        # Large portfolio should have tighter position limits
        assert constraints['max_single_position'] < 0.20

    def test_validate_constraints_valid(self, macro_context):
        """Test constraint validation for valid weights."""
        manager = AdaptiveConstraintManager()
        constraints = manager.get_regime_aware_constraints(
            'normal', macro_context, 1_000_000, {'hhi': 0.10}
        )

        # Create weights that are within the max constraint (0.20 for normal regime)
        valid_weights = {'AAPL': 0.20, 'GOOGL': 0.20, 'MSFT': 0.20, 'AMZN': 0.20, 'NVDA': 0.20}

        validation = manager.validate_constraints(valid_weights, constraints)

        assert validation['is_valid'] is True
        assert len(validation['violations']) == 0

    def test_validate_constraints_violation(self, macro_context):
        """Test constraint validation with violations."""
        manager = AdaptiveConstraintManager()
        constraints = manager.get_regime_aware_constraints(
            'normal', macro_context, 1_000_000, {'hhi': 0.10}
        )

        # Create weights with violation
        invalid_weights = {'AAPL': 0.50, 'GOOGL': 0.30, 'MSFT': 0.15, 'AMZN': 0.05}

        validation = manager.validate_constraints(invalid_weights, constraints)

        assert validation['is_valid'] is False
        assert len(validation['violations']) > 0

    def test_adjust_weights_to_constraints(self):
        """Test weight adjustment to satisfy constraints."""
        manager = AdaptiveConstraintManager()
        # Use constraints with enough positions to distribute
        constraints = {'max_single_position': 0.25, 'min_single_position': 0.05}

        # Weights with position too large
        weights = {'AAPL': 0.50, 'GOOGL': 0.25, 'MSFT': 0.15, 'AMZN': 0.10}

        adjusted = manager.adjust_weights_to_constraints(weights, constraints)

        # Max position should now be <= 0.25
        assert max(adjusted.values()) <= 0.25 + 0.01  # Small tolerance for normalization
        assert abs(sum(adjusted.values()) - 1.0) < 0.02


# =============================================================================
# 4. CROSS-PORTFOLIO CORRELATION MANAGER TESTS
# =============================================================================

class TestCrossPortfolioCorrelationManager:
    """Test suite for CrossPortfolioCorrelationManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CrossPortfolioCorrelationManager()
        assert manager.portfolio_correlations == {}
        assert manager.strategy_exposures == {}

    def test_covariance_to_correlation(self, sample_covariance):
        """Test covariance to correlation conversion."""
        manager = CrossPortfolioCorrelationManager()
        correlation = manager.covariance_to_correlation(sample_covariance)

        # Diagonal should be 1
        assert all(abs(correlation[i, i] - 1.0) < 1e-10 for i in range(4))
        # Should be symmetric
        assert np.allclose(correlation, correlation.T)
        # Values should be between -1 and 1
        assert np.all(correlation >= -1.0) and np.all(correlation <= 1.0)

    def test_calculate_strategy_correlation(self, sample_covariance):
        """Test strategy correlation calculation."""
        manager = CrossPortfolioCorrelationManager()

        portfolio_weights = {
            'momentum': {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.1},
            'value': {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': 0.3, 'AMZN': 0.4}
        }
        asset_universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

        correlations = manager.calculate_strategy_correlation(
            portfolio_weights, sample_covariance, asset_universe
        )

        assert 'momentum_value' in correlations
        assert -1.0 <= correlations['momentum_value'] <= 1.0

    def test_calculate_strategy_metrics(self, sample_covariance, sample_expected_returns):
        """Test strategy metrics calculation."""
        manager = CrossPortfolioCorrelationManager()

        portfolio_weights = {
            'momentum': {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.1},
            'value': {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': 0.3, 'AMZN': 0.4}
        }
        asset_universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

        metrics = manager.calculate_strategy_metrics(
            portfolio_weights, sample_expected_returns, sample_covariance, asset_universe
        )

        assert 'momentum' in metrics
        assert 'value' in metrics
        assert 'expected_return' in metrics['momentum']
        assert 'sharpe_ratio' in metrics['momentum']

    def test_optimize_strategy_allocation(self, sample_covariance):
        """Test strategy allocation optimization."""
        manager = CrossPortfolioCorrelationManager()

        strategy_returns = {'momentum': 0.12, 'value': 0.10, 'quality': 0.08}
        strategy_cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.008],
            [0.005, 0.008, 0.025]
        ])
        constraints = {
            'momentum': {'min_allocation': 0.1, 'max_allocation': 0.5},
            'value': {'min_allocation': 0.1, 'max_allocation': 0.5},
            'quality': {'min_allocation': 0.1, 'max_allocation': 0.5}
        }

        allocation = manager.optimize_strategy_allocation(
            strategy_returns, strategy_cov, constraints
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.01
        assert all(0.1 <= w <= 0.5 for w in allocation.values())

    def test_get_diversification_benefit(self, sample_covariance):
        """Test diversification benefit calculation."""
        manager = CrossPortfolioCorrelationManager()

        portfolio_weights = {
            'momentum': {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.1},
            'value': {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': 0.3, 'AMZN': 0.4}
        }
        asset_universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

        div_benefit = manager.get_diversification_benefit(
            portfolio_weights, sample_covariance, asset_universe
        )

        assert 'diversification_ratio' in div_benefit
        assert 'benefit' in div_benefit
        assert div_benefit['diversification_ratio'] >= 1.0  # Should be >= 1


# =============================================================================
# 5. PORTFOLIO STRESS TESTER TESTS
# =============================================================================

class TestPortfolioStressTester:
    """Test suite for PortfolioStressTester."""

    def test_initialization(self):
        """Test stress tester initialization."""
        tester = PortfolioStressTester()
        assert '2008_lehman' in tester.stress_scenarios
        assert '2020_covid' in tester.stress_scenarios

    def test_stress_test_portfolio_lehman(self, sample_weights, sample_covariance, asset_betas):
        """Test stress testing with 2008 Lehman scenario."""
        tester = PortfolioStressTester()

        result = tester.stress_test_portfolio(
            sample_weights, sample_covariance, asset_betas, '2008_lehman'
        )

        assert 'scenario_name' in result
        assert 'portfolio_return' in result
        assert 'scenario_var' in result
        assert result['portfolio_return'] < 0  # Should be negative in crisis

    def test_stress_test_portfolio_covid(self, sample_weights, sample_covariance):
        """Test stress testing with 2020 COVID scenario."""
        tester = PortfolioStressTester()

        # Use proper betas that respond to scenario factors
        covid_betas = {
            'AAPL': {'equity_shock': 1.2, 'volatility_spike': 0.1, 'liquidity_dry_up': 0.05},
            'GOOGL': {'equity_shock': 1.1, 'volatility_spike': 0.1, 'liquidity_dry_up': 0.05},
            'MSFT': {'equity_shock': 1.0, 'volatility_spike': 0.1, 'liquidity_dry_up': 0.05},
            'AMZN': {'equity_shock': 1.3, 'volatility_spike': 0.1, 'liquidity_dry_up': 0.05}
        }

        result = tester.stress_test_portfolio(
            sample_weights, sample_covariance, covid_betas, '2020_covid'
        )

        assert result['scenario_name'] == '2020_covid'
        # The portfolio return depends on betas and shocks - check it's calculated
        assert 'portfolio_return' in result

    def test_stress_test_unknown_scenario(self, sample_weights, sample_covariance, asset_betas):
        """Test stress testing with unknown scenario."""
        tester = PortfolioStressTester()

        result = tester.stress_test_portfolio(
            sample_weights, sample_covariance, asset_betas, 'unknown_scenario'
        )

        assert 'error' in result

    def test_adjust_covariance_for_scenario(self, sample_covariance):
        """Test covariance adjustment for stress scenario."""
        tester = PortfolioStressTester()

        adjusted = tester.adjust_covariance_for_scenario(sample_covariance, 0.5)

        # Adjusted correlations should be higher
        base_corr = tester.covariance_to_correlation(sample_covariance)
        adj_corr = tester.covariance_to_correlation(adjusted)

        # Off-diagonal elements should be higher (more correlated)
        assert adj_corr[0, 1] >= base_corr[0, 1]

    def test_run_all_stress_tests(self, sample_weights, sample_covariance, asset_betas):
        """Test running all stress scenarios."""
        tester = PortfolioStressTester()

        results = tester.run_all_stress_tests(
            sample_weights, sample_covariance, asset_betas
        )

        assert '2008_lehman' in results
        assert '2020_covid' in results
        assert 'summary' in results
        assert 'worst_case_return' in results['summary']

    def test_calculate_stress_adjusted_weights(self, sample_weights, sample_covariance, asset_betas):
        """Test stress-adjusted weight calculation."""
        tester = PortfolioStressTester()

        stress_results = tester.run_all_stress_tests(
            sample_weights, sample_covariance, asset_betas
        )

        adjusted = tester.calculate_stress_adjusted_weights(
            sample_weights, stress_results, max_stress_loss=-0.20
        )

        assert abs(sum(adjusted.values()) - 1.0) < 0.01

    def test_add_custom_scenario(self, sample_weights, sample_covariance, asset_betas):
        """Test adding custom stress scenario."""
        tester = PortfolioStressTester()

        tester.add_custom_scenario('custom_crisis', {
            'equity_shock': -0.25,
            'volatility_spike': 0.30
        })

        assert 'custom_crisis' in tester.stress_scenarios

        result = tester.stress_test_portfolio(
            sample_weights, sample_covariance, asset_betas, 'custom_crisis'
        )

        assert result['scenario_name'] == 'custom_crisis'


# =============================================================================
# 6. INTEGRATED SYSTEM TESTS
# =============================================================================

class TestPhase6FinalImprovementsSystem:
    """Test suite for integrated Phase6FinalImprovementsSystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = Phase6FinalImprovementsSystem()

        assert system.monitor is not None
        assert system.timeframe_optimizer is not None
        assert system.constraint_manager is not None
        assert system.correlation_manager is not None
        assert system.stress_tester is not None

    def test_factory_function(self):
        """Test factory function."""
        system = create_phase6_final_system()
        assert isinstance(system, Phase6FinalImprovementsSystem)

    def test_create_portfolio_monitor(self):
        """Test monitor factory."""
        monitor = create_portfolio_monitor()
        assert isinstance(monitor, PortfolioOptimizationMonitor)

    def test_create_stress_tester(self):
        """Test stress tester factory."""
        tester = create_stress_tester()
        assert isinstance(tester, PortfolioStressTester)

    def test_optimize_with_all_improvements(self, sample_weights, sample_covariance, macro_context, asset_betas):
        """Test full optimization with all improvements."""
        system = Phase6FinalImprovementsSystem()

        expected_returns = {
            'tactical': {'AAPL': 0.15, 'GOOGL': 0.12, 'MSFT': 0.13, 'AMZN': 0.10},
            'strategic': {'AAPL': 0.10, 'GOOGL': 0.09, 'MSFT': 0.11, 'AMZN': 0.08},
            'structural': {'AAPL': 0.08, 'GOOGL': 0.07, 'MSFT': 0.09, 'AMZN': 0.06}
        }
        covariance_matrices = {
            'tactical': sample_covariance,
            'strategic': sample_covariance,
            'structural': sample_covariance
        }

        result = system.optimize_with_all_improvements(
            sample_weights, expected_returns, covariance_matrices,
            'normal', macro_context, portfolio_size=1_000_000,
            asset_betas=asset_betas
        )

        assert 'optimized_weights' in result
        assert 'timeframe_results' in result
        assert 'constraints' in result
        assert 'stress_results' in result
        assert 'adaptive_parameters' in result

    def test_track_and_adapt(self, sample_weights):
        """Test performance tracking and adaptation."""
        system = Phase6FinalImprovementsSystem()

        opt_result = {
            'optimized_weights': sample_weights,
            'expected_return': 0.03
        }
        actual_returns = {'AAPL': 0.04, 'GOOGL': 0.02, 'MSFT': 0.03, 'AMZN': 0.01}

        result = system.track_and_adapt(
            opt_result, actual_returns, 'normal', 0.001
        )

        assert 'current_performance' in result
        assert 'regime_stats' in result
        assert 'needs_recalibration' in result


# =============================================================================
# CHECKLIST AND VALIDATION TESTS
# =============================================================================

class TestChecklistAndValidation:
    """Test checklist and validation functions."""

    def test_production_checklist_structure(self):
        """Test production checklist structure."""
        assert 'monitoring' in PHASE6_FINAL_PRODUCTION_CHECKLIST
        assert 'risk_management' in PHASE6_FINAL_PRODUCTION_CHECKLIST
        assert 'optimization' in PHASE6_FINAL_PRODUCTION_CHECKLIST
        assert 'operational' in PHASE6_FINAL_PRODUCTION_CHECKLIST

    def test_validate_final_improvements(self):
        """Test module validation function."""
        results = validate_final_improvements()

        assert 'portfolio_monitor' in results
        assert 'multi_timeframe_optimizer' in results
        assert 'adaptive_constraints' in results
        assert 'correlation_manager' in results
        assert 'stress_tester' in results
        assert 'integrated_system' in results

        # All should pass
        passed = sum(1 for v in results.values() if v == True)
        total = len(results)
        assert passed == total, f"Only {passed}/{total} validations passed"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_weights(self):
        """Test with empty weights."""
        manager = AdaptiveConstraintManager()
        constraints = manager.base_constraints

        validation = manager.validate_constraints({}, constraints)
        assert validation['is_valid'] is False

    def test_single_asset_portfolio(self, sample_covariance):
        """Test with single asset portfolio."""
        tester = PortfolioStressTester()

        single_weights = {'AAPL': 1.0}
        single_cov = np.array([[0.04]])
        single_betas = {'AAPL': {'equity_shock': 1.0}}

        result = tester.stress_test_portfolio(
            single_weights, single_cov, single_betas, '2008_lehman'
        )

        assert 'portfolio_return' in result

    def test_high_concentration(self, macro_context):
        """Test with highly concentrated portfolio."""
        manager = AdaptiveConstraintManager()

        constraints = manager.get_regime_aware_constraints(
            'normal', macro_context, 1_000_000, {'hhi': 0.30}
        )

        # High concentration should allow more rebalancing
        assert constraints['max_turnover'] >= 0.15

    def test_extreme_stress_scenario(self, sample_weights, sample_covariance):
        """Test with extreme custom stress scenario."""
        tester = PortfolioStressTester()

        tester.add_custom_scenario('extreme_crisis', {
            'equity_shock': -0.60,
            'volatility_spike': 1.0,
            'liquidity_dry_up': 0.9,
            'correlation_breakdown': 0.95
        })

        # Use proper betas that respond to equity shock
        extreme_betas = {
            'AAPL': {'equity_shock': 1.2},
            'GOOGL': {'equity_shock': 1.1},
            'MSFT': {'equity_shock': 1.0},
            'AMZN': {'equity_shock': 1.3}
        }

        result = tester.stress_test_portfolio(
            sample_weights, sample_covariance, extreme_betas, 'extreme_crisis'
        )

        # Portfolio should have significant negative return in extreme crisis
        assert result['portfolio_return'] < -0.50  # -60% shock * ~1.15 avg beta


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
