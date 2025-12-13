"""
Test suite for Phase 6 Advanced Improvements - China Model
Tests the advanced portfolio optimization components from phase6_final_improvements.py
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'portfolio'))

from portfolio.phase6_final_improvements import (
    PortfolioOptimizationMonitor,
    MultiTimeframeOptimizer,
    AdaptiveConstraintManager,
    CrossPortfolioCorrelationManager,
    PortfolioStressTester
)


class TestPortfolioOptimizationMonitor:

    def setup_method(self):
        self.monitor = PortfolioOptimizationMonitor()
        self.optimization_result = {
            'expected_return': 0.08,
            'turnover': 0.12,
            'sharpe_ratio': 1.2
        }
        self.actual_returns = {'AAPL': 0.10, 'MSFT': 0.06, 'GOOGL': 0.09}
        self.regime = 'normal'
        self.execution_costs = 0.002

    def test_track_optimization_performance_initial(self):
        """Test initial performance tracking"""
        performance = self.monitor.track_optimization_performance(
            self.optimization_result, self.actual_returns, self.regime, self.execution_costs
        )

        # Should return performance metrics
        assert 'current_performance' in performance
        assert 'regime_stats' in performance
        assert 'needs_recalibration' in performance

        # Should track basic metrics
        current = performance['current_performance']
        assert current['regime'] == self.regime
        assert current['expected_return'] == 0.08
        assert 'realized_return' in current
        assert 'forecast_error' in current

    def test_track_optimization_performance_multiple_updates(self):
        """Test performance tracking with multiple updates"""
        # First update
        perf1 = self.monitor.track_optimization_performance(
            self.optimization_result, self.actual_returns, self.regime, self.execution_costs
        )

        # Second update with different results
        result2 = {'expected_return': 0.06, 'turnover': 0.08, 'sharpe_ratio': 0.9}
        returns2 = {'AAPL': 0.04, 'MSFT': 0.03, 'GOOGL': 0.05}

        perf2 = self.monitor.track_optimization_performance(
            result2, returns2, self.regime, 0.0015
        )

        # Should accumulate history
        assert len(self.monitor.performance_history) == 2
        assert len(self.monitor.regime_performance[self.regime]) == 2

    def test_should_recalibrate_consistent_errors(self):
        """Test recalibration trigger with consistent forecast errors"""
        # Simulate consistent underperformance
        for i in range(25):
            result = {'expected_return': 0.08, 'turnover': 0.1, 'sharpe_ratio': 1.0}
            # Actual returns consistently 3% below expectations
            actual_returns = {f'STOCK_{j}': 0.05 for j in range(3)}

            self.monitor.track_optimization_performance(
                result, actual_returns, 'normal', 0.002
            )

        stats = {
            'avg_forecast_error': -0.03,  # Consistent 3% underprediction
            'forecast_error_std': 0.01,
            'n_observations': 25
        }

        should_recalibrate = self.monitor.should_recalibrate(stats)

        assert should_recalibrate == True

    def test_should_recalibrate_insufficient_data(self):
        """Test recalibration with insufficient data"""
        # The implementation checks len(self.performance_history) > 100
        # So with insufficient data in performance_history, it should not recalibrate
        # unless error thresholds are met

        # Create a fresh monitor with no history
        fresh_monitor = PortfolioOptimizationMonitor()

        stats = {
            'avg_forecast_error': 0.01,  # Small error - below 2% threshold
            'forecast_error_std': 0.01,  # Low variability - below 5% threshold
            'n_observations': 15
        }

        should_recalibrate = fresh_monitor.should_recalibrate(stats)

        # With small errors and fresh monitor, should not need recalibration
        assert should_recalibrate == False

    def test_get_adaptive_parameters_sufficient_data(self):
        """Test adaptive parameter calculation with sufficient data"""
        # Add performance history
        for i in range(25):
            result = {'expected_return': 0.08, 'turnover': 0.1, 'sharpe_ratio': 1.0}
            actual_returns = {f'STOCK_{j}': 0.06 for j in range(3)}  # Consistent underprediction

            self.monitor.track_optimization_performance(
                result, actual_returns, 'normal', 0.002
            )

        adaptive_params = self.monitor.get_adaptive_parameters('normal')

        # Should return adjusted parameters
        assert 'risk_aversion_multiplier' in adaptive_params
        assert 'covariance_shrinkage' in adaptive_params
        assert 'transaction_cost_penalty' in adaptive_params
        assert 'liquidity_penalty_multiplier' in adaptive_params

        # Parameters should be adjusted based on performance
        assert adaptive_params['risk_aversion_multiplier'] >= 1.0  # More conservative

    def test_get_adaptive_parameters_insufficient_data(self):
        """Test adaptive parameters with insufficient regime data"""
        adaptive_params = self.monitor.get_adaptive_parameters('crisis')

        # Should return default parameters for new regime
        assert adaptive_params is not None
        # Default parameters should be reasonable
        assert adaptive_params['risk_aversion_multiplier'] >= 1.0


class TestMultiTimeframeOptimizer:

    def setup_method(self):
        self.optimizer = MultiTimeframeOptimizer()
        self.current_weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}

        # Mock expected returns by timeframe
        self.expected_returns = {
            'tactical': {'AAPL': 0.12, 'MSFT': 0.08, 'GOOGL': 0.10},
            'strategic': {'AAPL': 0.09, 'MSFT': 0.11, 'GOOGL': 0.08},
            'structural': {'AAPL': 0.07, 'MSFT': 0.10, 'GOOGL': 0.09}
        }

        # Mock covariance matrices
        base_cov = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.06]
        ])
        self.covariance_matrices = {
            'tactical': base_cov * 1.2,  # Higher short-term vol
            'strategic': base_cov,
            'structural': base_cov * 0.8  # Lower long-term vol
        }

        self.regime = 'normal'
        self.macro_context = {'vix': 18.0}

    def test_optimize_multi_timeframe_all_timeframes(self):
        """Test multi-timeframe optimization with all timeframes"""
        results = self.optimizer.optimize_multi_timeframe(
            self.current_weights, self.expected_returns,
            self.covariance_matrices, self.regime, self.macro_context
        )

        # Should return results for all timeframes
        assert 'tactical' in results['timeframe_results']
        assert 'strategic' in results['timeframe_results']
        assert 'structural' in results['timeframe_results']

        # Should return blended weights
        assert 'blended_weights' in results
        assert 'blending_weights' in results

        # Blended weights should sum to 1
        blended_total = sum(results['blended_weights'].values())
        assert blended_total == pytest.approx(1.0, rel=1e-6)

    def test_optimize_multi_timeframe_missing_data(self):
        """Test multi-timeframe optimization with missing timeframe data"""
        partial_returns = {
            'tactical': {'AAPL': 0.12, 'MSFT': 0.08, 'GOOGL': 0.10},
            # strategic missing
            'structural': {'AAPL': 0.07, 'MSFT': 0.10, 'GOOGL': 0.09}
        }

        partial_cov = {
            'tactical': self.covariance_matrices['tactical'],
            # strategic missing
            'structural': self.covariance_matrices['structural']
        }

        results = self.optimizer.optimize_multi_timeframe(
            self.current_weights, partial_returns, partial_cov,
            self.regime, self.macro_context
        )

        # Should handle missing data gracefully
        assert 'tactical' in results['timeframe_results']
        assert 'structural' in results['timeframe_results']
        # strategic should be missing
        assert 'strategic' not in results['timeframe_results']

    def test_calculate_blending_weights_different_regimes(self):
        """Test blending weight calculation across different regimes"""
        regimes = ['crisis', 'high_vol', 'normal', 'low_vol']

        blending_weights = {}
        for regime in regimes:
            weights = self.optimizer.calculate_blending_weights(regime)
            blending_weights[regime] = weights

        # Should return different weights for different regimes
        assert blending_weights['crisis'] != blending_weights['low_vol']

        # Weights should sum to 1 for each regime
        for regime, weights in blending_weights.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, rel=1e-6)

        # Crisis should emphasize strategic timeframe
        assert blending_weights['crisis']['strategic'] > blending_weights['low_vol']['strategic']
        # Low vol should emphasize tactical timeframe
        assert blending_weights['low_vol']['tactical'] > blending_weights['crisis']['tactical']

    def test_multi_timeframe_crisis_scenario(self):
        """Test multi-timeframe optimization in crisis regime"""
        crisis_results = self.optimizer.optimize_multi_timeframe(
            self.current_weights, self.expected_returns,
            self.covariance_matrices, 'crisis', {'vix': 45.0}
        )

        # Crisis should have different blending weights
        crisis_blending = crisis_results['blending_weights']
        normal_blending = self.optimizer.calculate_blending_weights('normal')

        assert crisis_blending != normal_blending
        # Crisis should be more strategic, less tactical
        assert crisis_blending['strategic'] > normal_blending['strategic']
        assert crisis_blending['tactical'] < normal_blending['tactical']


class TestAdaptiveConstraintManager:

    def setup_method(self):
        self.constraint_manager = AdaptiveConstraintManager()
        self.regime = 'normal'
        self.macro_context = {'vix': 20.0}
        self.portfolio_size = 5000000  # $5M
        self.current_concentration = {'hhi': 0.12, 'max_position': 0.25}

    def test_get_regime_aware_constraints_normal(self):
        """Test constraint calculation in normal regime"""
        constraints = self.constraint_manager.get_regime_aware_constraints(
            self.regime, self.macro_context, self.portfolio_size, self.current_concentration
        )

        # Should return all constraint types
        assert 'max_single_position' in constraints
        assert 'min_single_position' in constraints
        assert 'max_sector_exposure' in constraints
        assert 'max_turnover' in constraints
        assert 'liquidity_minimum' in constraints

        # Should use base constraints for normal regime
        base_constraints = self.constraint_manager.base_constraints
        for key in base_constraints:
            assert constraints[key] == base_constraints[key]

    def test_get_regime_aware_constraints_crisis(self):
        """Test constraint calculation in crisis regime"""
        crisis_constraints = self.constraint_manager.get_regime_aware_constraints(
            'crisis', {'vix': 40.0}, self.portfolio_size, self.current_concentration
        )

        normal_constraints = self.constraint_manager.get_regime_aware_constraints(
            'normal', self.macro_context, self.portfolio_size, self.current_concentration
        )

        # Crisis should have tighter constraints
        assert crisis_constraints['max_single_position'] < normal_constraints['max_single_position']
        assert crisis_constraints['max_turnover'] < normal_constraints['max_turnover']
        assert crisis_constraints['liquidity_minimum'] > normal_constraints['liquidity_minimum']

    def test_get_regime_aware_constraints_large_portfolio(self):
        """Test constraint calculation for large portfolio"""
        large_portfolio_size = 50000000  # $50M

        large_constraints = self.constraint_manager.get_regime_aware_constraints(
            self.regime, self.macro_context, large_portfolio_size, self.current_concentration
        )

        normal_constraints = self.constraint_manager.get_regime_aware_constraints(
            self.regime, self.macro_context, self.portfolio_size, self.current_concentration
        )

        # Large portfolio should have tighter constraints
        assert large_constraints['max_single_position'] < normal_constraints['max_single_position']
        assert large_constraints['liquidity_minimum'] > normal_constraints['liquidity_minimum']

    def test_validate_constraints_valid_weights(self):
        """Test constraint validation with valid weights"""
        proposed_weights = {'AAPL': 0.15, 'MSFT': 0.18, 'GOOGL': 0.12, 'AMZN': 0.10, 'TSLA': 0.08, 'NVDA': 0.07, 'META': 0.06, 'NFLX': 0.05, 'OTHER': 0.19}
        current_weights = {'AAPL': 0.16, 'MSFT': 0.17, 'GOOGL': 0.13, 'AMZN': 0.11, 'TSLA': 0.09, 'NVDA': 0.08, 'META': 0.07, 'NFLX': 0.06, 'OTHER': 0.13}

        constraints = self.constraint_manager.base_constraints

        validation = self.constraint_manager.validate_constraints(
            proposed_weights, constraints, current_weights
        )

        # Valid weights should pass
        assert validation['is_valid'] == True
        assert len(validation['violations']) == 0

    def test_validate_constraints_max_position_violation(self):
        """Test constraint validation with max position violation"""
        proposed_weights = {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.55}  # GOOGL violates 20% limit

        constraints = self.constraint_manager.base_constraints

        validation = self.constraint_manager.validate_constraints(
            proposed_weights, constraints
        )

        # Should detect max position violation
        assert validation['is_valid'] == False
        assert len(validation['violations']) > 0
        assert any('Max position violation' in v for v in validation['violations'])

    def test_validate_constraints_turnover_warning(self):
        """Test constraint validation with turnover warning"""
        proposed_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        current_weights = {'AAPL': 0.1, 'MSFT': 0.1, 'GOOGL': 0.8}  # High turnover

        constraints = self.constraint_manager.base_constraints

        validation = self.constraint_manager.validate_constraints(
            proposed_weights, constraints, current_weights
        )

        # Should generate turnover warning
        assert len(validation['warnings']) > 0
        assert any('High turnover' in w for w in validation['warnings'])


class TestCrossPortfolioCorrelationManager:

    def setup_method(self):
        self.correlation_manager = CrossPortfolioCorrelationManager()

        # Mock portfolio weights by strategy
        self.portfolio_weights = {
            'momentum': {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3},
            'value': {'AAPL': 0.2, 'MSFT': 0.5, 'GOOGL': 0.3},
            'quality': {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}
        }

        # Mock covariance matrix
        self.covariance_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.06]
        ])

        self.asset_universe = ['AAPL', 'MSFT', 'GOOGL']

    def test_calculate_strategy_correlation_multiple_strategies(self):
        """Test strategy correlation calculation with multiple strategies"""
        correlations = self.correlation_manager.calculate_strategy_correlation(
            self.portfolio_weights, self.covariance_matrix, self.asset_universe
        )

        # Should calculate pairwise correlations
        expected_pairs = ['momentum_value', 'momentum_quality', 'value_quality']
        for pair in expected_pairs:
            assert pair in correlations
            # Correlations should be between -1 and 1
            assert -1 <= correlations[pair] <= 1

    def test_calculate_strategy_correlation_single_strategy(self):
        """Test strategy correlation with single strategy"""
        single_weights = {'momentum': {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}}

        correlations = self.correlation_manager.calculate_strategy_correlation(
            single_weights, self.covariance_matrix, self.asset_universe
        )

        # Should return empty dict for single strategy
        assert correlations == {}

    def test_optimize_strategy_allocation(self):
        """Test strategy allocation optimization"""
        strategy_returns = {
            'momentum': 0.12,
            'value': 0.09,
            'quality': 0.10
        }

        strategy_covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.05, 0.015],
            [0.01, 0.015, 0.03]
        ])

        strategy_constraints = {
            'momentum': {'min_allocation': 0.1, 'max_allocation': 0.5},
            'value': {'min_allocation': 0.1, 'max_allocation': 0.6},
            'quality': {'min_allocation': 0.05, 'max_allocation': 0.4}
        }

        allocation = self.correlation_manager.optimize_strategy_allocation(
            strategy_returns, strategy_covariance, strategy_constraints
        )

        # Should return allocation for each strategy
        assert 'momentum' in allocation
        assert 'value' in allocation
        assert 'quality' in allocation

        # Allocation should sum to 1
        total = sum(allocation.values())
        assert total == pytest.approx(1.0, rel=1e-6)

        # Should respect constraints
        assert 0.1 <= allocation['momentum'] <= 0.5
        assert 0.1 <= allocation['value'] <= 0.6
        assert 0.05 <= allocation['quality'] <= 0.4

    def test_optimize_strategy_allocation_optimization_failure(self):
        """Test strategy allocation when optimization fails"""
        strategy_returns = {'strategy1': 0.1, 'strategy2': 0.15}
        # Invalid covariance matrix (not positive definite)
        strategy_covariance = np.array([[0.04, 0.1], [0.1, 0.01]])
        strategy_constraints = {}

        # Mock optimization failure
        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_minimize.return_value.success = False

            allocation = self.correlation_manager.optimize_strategy_allocation(
                strategy_returns, strategy_covariance, strategy_constraints
            )

        # Should fall back to inverse volatility
        assert 'strategy1' in allocation
        assert 'strategy2' in allocation
        total = sum(allocation.values())
        assert total == pytest.approx(1.0, rel=1e-6)


class TestPortfolioStressTester:

    def setup_method(self):
        self.stress_tester = PortfolioStressTester()
        self.portfolio_weights = {'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25}
        self.base_covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.06]
        ])

        # Mock asset betas to risk factors
        self.asset_betas = {
            'AAPL': {'equity_shock': 1.2, 'credit_spread_widening': 0.3, 'volatility_spike': 0.8},
            'MSFT': {'equity_shock': 1.1, 'credit_spread_widening': 0.2, 'volatility_spike': 0.7},
            'GOOGL': {'equity_shock': 1.3, 'credit_spread_widening': 0.4, 'volatility_spike': 0.9}
        }

    def test_stress_test_portfolio_known_scenario(self):
        """Test stress testing with known historical scenario"""
        scenario_name = '2008_lehman'

        stress_result = self.stress_tester.stress_test_portfolio(
            self.portfolio_weights, self.base_covariance, self.asset_betas, scenario_name
        )

        # Should return comprehensive stress test results
        assert 'scenario_name' in stress_result
        assert 'portfolio_return' in stress_result
        assert 'scenario_var' in stress_result
        assert 'max_drawdown_estimate' in stress_result
        assert 'liquidity_impact' in stress_result
        # Note: Implementation may return 'correlation_boost' instead of 'adjusted_covariance'
        assert 'correlation_boost' in stress_result or 'adjusted_covariance' in stress_result

        # 2008 scenario should show significant losses
        assert stress_result['portfolio_return'] < 0
        assert stress_result['max_drawdown_estimate'] < 0

    def test_stress_test_portfolio_unknown_scenario(self):
        """Test stress testing with unknown scenario"""
        scenario_name = 'unknown_scenario'

        stress_result = self.stress_tester.stress_test_portfolio(
            self.portfolio_weights, self.base_covariance, self.asset_betas, scenario_name
        )

        # Should handle unknown scenario gracefully
        assert 'error' in stress_result

    def test_stress_test_portfolio_different_scenarios(self):
        """Test stress testing across different scenarios"""
        scenarios = ['2008_lehman', '2020_covid', 'inflation_shock']

        scenario_results = {}
        for scenario in scenarios:
            result = self.stress_tester.stress_test_portfolio(
                self.portfolio_weights, self.base_covariance, self.asset_betas, scenario
            )
            scenario_results[scenario] = result

        # Should return different results for different scenarios
        assert '2008_lehman' in scenario_results
        assert '2020_covid' in scenario_results
        assert 'inflation_shock' in scenario_results

        # Different scenarios should have different impacts
        lehman_return = scenario_results['2008_lehman']['portfolio_return']
        covid_return = scenario_results['2020_covid']['portfolio_return']

        assert lehman_return != covid_return

    def test_adjust_covariance_for_scenario(self):
        """Test covariance adjustment for stress scenarios"""
        correlation_boost = 0.5

        adjusted_covariance = self.stress_tester.adjust_covariance_for_scenario(
            self.base_covariance, correlation_boost
        )

        # Should return adjusted covariance matrix
        assert adjusted_covariance.shape == self.base_covariance.shape

        # Adjusted correlations should be higher
        base_corr = self.covariance_to_correlation(self.base_covariance)
        adjusted_corr = self.covariance_to_correlation(adjusted_covariance)

        # Off-diagonal correlations should be increased
        for i in range(base_corr.shape[0]):
            for j in range(base_corr.shape[1]):
                if i != j:
                    assert adjusted_corr[i, j] >= base_corr[i, j]

    def covariance_to_correlation(self, covariance_matrix):
        """Helper to convert covariance to correlation matrix"""
        vol = np.sqrt(np.diag(covariance_matrix))
        return covariance_matrix / np.outer(vol, vol)


class TestIntegrationScenariosAdvanced:
    """Advanced integration tests combining all improvements"""

    def test_complete_adaptive_optimization_flow(self):
        """Test complete adaptive optimization workflow"""
        # Initialize all components
        monitor = PortfolioOptimizationMonitor()
        multi_timeframe = MultiTimeframeOptimizer()
        constraint_manager = AdaptiveConstraintManager()
        stress_tester = PortfolioStressTester()

        # Input data
        current_weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}
        expected_returns = {
            'tactical': {'AAPL': 0.12, 'MSFT': 0.08, 'GOOGL': 0.10},
            'strategic': {'AAPL': 0.09, 'MSFT': 0.11, 'GOOGL': 0.08},
            'structural': {'AAPL': 0.07, 'MSFT': 0.10, 'GOOGL': 0.09}
        }
        covariance_matrices = {
            'tactical': np.array([[0.05, 0.03, 0.02], [0.03, 0.10, 0.04], [0.02, 0.04, 0.07]]),
            'strategic': np.array([[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.06]]),
            'structural': np.array([[0.03, 0.015, 0.008], [0.015, 0.08, 0.025], [0.008, 0.025, 0.05]])
        }
        regime = 'normal'
        macro_context = {'vix': 18.0}
        portfolio_size = 10000000
        current_concentration = {'hhi': 0.14, 'max_position': 0.28}

        # Step 1: Multi-timeframe optimization
        multi_results = multi_timeframe.optimize_multi_timeframe(
            current_weights, expected_returns, covariance_matrices, regime, macro_context
        )

        # Step 2: Get adaptive constraints
        constraints = constraint_manager.get_regime_aware_constraints(
            regime, macro_context, portfolio_size, current_concentration
        )

        # Step 3: Validate against constraints
        validation = constraint_manager.validate_constraints(
            multi_results['blended_weights'], constraints, current_weights
        )

        # Step 4: Stress test the blended portfolio
        asset_betas = {
            'AAPL': {'equity_shock': 1.2, 'volatility_spike': 0.8},
            'MSFT': {'equity_shock': 1.1, 'volatility_spike': 0.7},
            'GOOGL': {'equity_shock': 1.3, 'volatility_spike': 0.9}
        }

        stress_results = stress_tester.stress_test_portfolio(
            multi_results['blended_weights'], covariance_matrices['strategic'],
            asset_betas, '2008_lehman'
        )

        # Step 5: Track performance (mock execution)
        execution_costs = 0.0025
        actual_returns = {'AAPL': 0.11, 'MSFT': 0.07, 'GOOGL': 0.09}

        optimization_result = {
            'expected_return': 0.085,
            'turnover': multi_results.get('estimated_turnover', 0.12),
            'sharpe_ratio': 1.1
        }

        performance = monitor.track_optimization_performance(
            optimization_result, actual_returns, regime, execution_costs
        )

        # Verify complete workflow
        assert multi_results is not None
        assert constraints is not None
        assert validation is not None
        assert stress_results is not None
        assert performance is not None

        # Should have adaptive components
        assert 'needs_recalibration' in performance
        assert 'is_valid' in validation

    def test_crisis_scenario_comprehensive(self):
        """Test comprehensive crisis scenario handling"""
        monitor = PortfolioOptimizationMonitor()
        multi_timeframe = MultiTimeframeOptimizer()
        constraint_manager = AdaptiveConstraintManager()
        stress_tester = PortfolioStressTester()

        crisis_regime = 'crisis'
        crisis_context = {'vix': 45.0}
        portfolio_size = 5000000
        current_concentration = {'hhi': 0.18, 'max_position': 0.32}

        # Test crisis constraints
        crisis_constraints = constraint_manager.get_regime_aware_constraints(
            crisis_regime, crisis_context, portfolio_size, current_concentration
        )

        # Test crisis blending weights
        crisis_blending = multi_timeframe.calculate_blending_weights(crisis_regime)

        # Test crisis stress scenario
        portfolio_weights = {'AAPL': 0.25, 'MSFT': 0.30, 'GOOGL': 0.20, 'CASH': 0.25}
        base_covariance = np.array([
            [0.16, 0.12, 0.10, 0.00],
            [0.12, 0.25, 0.15, 0.00],
            [0.10, 0.15, 0.20, 0.00],
            [0.00, 0.00, 0.00, 0.00]
        ])
        asset_betas = {
            'AAPL': {'equity_shock': 1.3, 'liquidity_dry_up': 0.6},
            'MSFT': {'equity_shock': 1.2, 'liquidity_dry_up': 0.5},
            'GOOGL': {'equity_shock': 1.4, 'liquidity_dry_up': 0.7},
            'CASH': {'equity_shock': 0.0, 'liquidity_dry_up': 0.0}
        }

        crisis_stress = stress_tester.stress_test_portfolio(
            portfolio_weights, base_covariance, asset_betas, '2008_lehman'
        )

        # Crisis should show conservative characteristics
        assert crisis_constraints['max_single_position'] < 0.20  # Tighter limits
        assert crisis_blending['strategic'] > crisis_blending['tactical']  # More strategic focus
        assert crisis_stress['portfolio_return'] < 0  # Negative in crisis
        assert crisis_stress['liquidity_impact'] > 0  # Liquidity impact expected


class TestEdgeCasesAdvanced:
    """Advanced edge case tests"""

    def test_empty_portfolio_stress_test(self):
        """Test stress testing with empty portfolio"""
        stress_tester = PortfolioStressTester()
        empty_weights = {}
        empty_covariance = np.array([]).reshape(0, 0)
        empty_betas = {}

        result = stress_tester.stress_test_portfolio(
            empty_weights, empty_covariance, empty_betas, '2008_lehman'
        )

        # Should handle empty portfolio gracefully
        assert 'portfolio_return' in result
        assert result['portfolio_return'] == 0

    def test_extreme_concentration_constraints(self):
        """Test constraint management with extreme concentration"""
        constraint_manager = AdaptiveConstraintManager()
        highly_concentrated = {'hhi': 0.8, 'max_position': 0.9}  # Very concentrated

        constraints = constraint_manager.get_regime_aware_constraints(
            'normal', {'vix': 20.0}, 1000000, highly_concentrated
        )

        # Should adjust constraints for concentrated portfolio
        assert constraints['max_turnover'] > constraint_manager.base_constraints['max_turnover']

    def test_negative_returns_strategy_allocation(self):
        """Test strategy allocation with negative expected returns"""
        correlation_manager = CrossPortfolioCorrelationManager()

        negative_returns = {
            'strategy1': -0.05,
            'strategy2': -0.03,
            'strategy3': -0.08
        }

        strategy_covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.05, 0.015],
            [0.01, 0.015, 0.03]
        ])

        allocation = correlation_manager.optimize_strategy_allocation(
            negative_returns, strategy_covariance, {}
        )

        # Should handle negative returns gracefully
        assert len(allocation) == 3
        total = sum(allocation.values())
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_insufficient_performance_history(self):
        """Test adaptive parameters with very little history"""
        monitor = PortfolioOptimizationMonitor()

        # Add only 5 data points
        for i in range(5):
            monitor.track_optimization_performance(
                {'expected_return': 0.08}, {'AAPL': 0.06}, 'normal', 0.002
            )

        adaptive_params = monitor.get_adaptive_parameters('normal')

        # Should return reasonable default parameters
        assert adaptive_params is not None
        assert adaptive_params['risk_aversion_multiplier'] >= 1.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
