"""
Phase 5 Advanced Improvements - Final Test Suite

Adapted from 'phase5 impro on C model final test.pdf' to work with
the actual implementation in phase5_production.py.

Tests all advanced Phase 5 components:
- Performance Monitoring with alerts
- Adaptive Parameter Optimization
- Regime Detection Validation
- Correlation Regime Detection
- Portfolio Risk Budgeting
- ML Enhancement
- Integration Scenarios
- Edge Cases
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, 'src')

from ensemble.phase5_production import (
    Phase5PerformanceMonitor,
    AdaptiveParameterOptimizer,
    RegimeDetectionValidator,
    CorrelationRegimeDetector,
    PortfolioRiskBudget,
    MLEnhancedPhase5,
    ProductionPhase5System,
    create_production_phase5_system
)


class TestPhase5PerformanceMonitorAdvanced:
    """Advanced tests for performance monitoring."""

    def setup_method(self):
        self.monitor = Phase5PerformanceMonitor()
        self.sample_macro_context = {'vix': 25.0, 'risk_off': True}

    def test_macro_adjustment_effectiveness_tracking(self):
        """Test tracking of macro adjustment effectiveness."""
        # Test case where adjustment improves prediction
        original_score = 0.6
        adjusted_score = 0.8  # Macro boosted this
        actual_return = 0.05  # Positive return

        self.monitor.track_macro_adjustment_effectiveness(
            original_score, adjusted_score, actual_return
        )

        effectiveness_data = list(self.monitor.performance_metrics['macro_adjustment_effectiveness'])
        assert len(effectiveness_data) == 1
        assert effectiveness_data[0] == 1.0  # Should be effective

    def test_macro_adjustment_ineffective_tracking(self):
        """Test tracking when macro adjustments hurt performance."""
        original_score = 0.8
        adjusted_score = 0.5  # Macro reduced this
        actual_return = 0.10  # Strong positive return

        self.monitor.track_macro_adjustment_effectiveness(
            original_score, adjusted_score, actual_return
        )

        effectiveness_data = list(self.monitor.performance_metrics['macro_adjustment_effectiveness'])
        assert effectiveness_data[0] == -1.0  # Should be ineffective

    def test_regime_accuracy_tracking(self):
        """Test regime detection accuracy tracking."""
        predicted_regime = 'normal'
        actual_volatility = 0.18  # Normal volatility
        historical_volatility = 0.15

        self.monitor.track_regime_accuracy(
            predicted_regime, actual_volatility, historical_volatility
        )

        accuracy_data = list(self.monitor.performance_metrics['regime_detection_accuracy'])
        assert len(accuracy_data) == 1
        # Should be correct since actual_volatility/historical_volatility ~ 1.2 (normal range)

    def test_alert_triggering_for_ineffective_adjustments(self):
        """Test alert system for consistently ineffective adjustments."""
        # Set a low lookback window for testing
        self.monitor.lookback_window = 10

        # Simulate many ineffective adjustments
        for _ in range(15):
            self.monitor.performance_metrics['macro_adjustment_effectiveness'].append(-1.0)

        # This should trigger an alert
        original_score = 0.7
        adjusted_score = 0.4
        actual_return = 0.08

        # Track another ineffective adjustment
        self.monitor.track_macro_adjustment_effectiveness(
            original_score, adjusted_score, actual_return
        )

        # Check if alert was triggered
        assert len(self.monitor.alerts_history) > 0
        assert self.monitor.alerts_history[0].alert_type == "MACRO_ADJUSTMENT_INEFFECTIVE"

    def test_performance_summary_comprehensive(self):
        """Test comprehensive performance summary."""
        # Add various metrics
        for i in range(30):
            self.monitor.track_signal_accuracy(0.5, 0.01 if i % 2 == 0 else -0.005)
            self.monitor.track_kelly_sizing_performance(0.05, 0.01, i % 2 == 0)

        summary = self.monitor.get_performance_summary()

        assert 'overall_signal_accuracy' in summary
        assert summary['overall_signal_accuracy']['count'] == 30
        assert 'kelly_sizing_performance' in summary
        assert summary['kelly_sizing_performance']['count'] == 30


class TestAdaptiveParameterOptimizerAdvanced:
    """Advanced tests for adaptive parameter optimization."""

    def setup_method(self):
        self.optimizer = AdaptiveParameterOptimizer(
            optimization_lookback=30,
            min_samples=5,
            learning_rate=0.2
        )

    def test_optimize_macro_boost_factors_with_performance_data(self):
        """Test dynamic optimization of macro boost factors with actual performance."""
        # Record performance data
        # Equity performs well in risk-on, poorly in risk-off
        for i in range(10):
            self.optimizer.record_performance('equity', 'risk_on', 0.02 + np.random.randn() * 0.005)
            self.optimizer.record_performance('equity', 'risk_off', -0.01 + np.random.randn() * 0.005)

        # Bonds perform well in risk-off
        for i in range(10):
            self.optimizer.record_performance('bond', 'risk_on', 0.005 + np.random.randn() * 0.003)
            self.optimizer.record_performance('bond', 'risk_off', 0.015 + np.random.randn() * 0.003)

        optimized_boosts = self.optimizer.optimize_macro_boost_factors()

        # Equity should have higher risk-on boost
        assert 'equity' in optimized_boosts
        assert optimized_boosts['equity']['risk_on'] >= optimized_boosts['equity']['risk_off']

        # All values should be within bounds
        for asset_class, boosts in optimized_boosts.items():
            assert 0.5 <= boosts['risk_on'] <= 2.0
            assert 0.5 <= boosts['risk_off'] <= 2.0

    def test_optimize_kelly_fraction_high_performance(self):
        """Test Kelly fraction optimization with good performance."""
        np.random.seed(42)
        # Simulate strong recent returns (high Sharpe)
        strong_returns = list(np.random.normal(0.02, 0.01, 150))

        optimal_fraction = self.optimizer.optimize_kelly_fraction(strong_returns)

        # Should be more aggressive with good performance
        assert optimal_fraction >= 0.25  # At least standard quarter-Kelly
        assert optimal_fraction <= 0.4   # Within upper bound

    def test_optimize_kelly_fraction_poor_performance(self):
        """Test Kelly fraction optimization with poor performance."""
        np.random.seed(42)
        # Simulate poor recent returns (low Sharpe)
        poor_returns = list(np.random.normal(-0.005, 0.025, 150))

        optimal_fraction = self.optimizer.optimize_kelly_fraction(poor_returns)

        # Should be more conservative with poor performance
        assert optimal_fraction <= 0.25  # More conservative than standard
        assert optimal_fraction >= 0.1   # Within lower bound

    def test_optimize_kelly_fraction_insufficient_data(self):
        """Test Kelly fraction with insufficient data."""
        minimal_data = [0.01, -0.02, 0.015]  # Only 3 data points

        optimal_fraction = self.optimizer.optimize_kelly_fraction(minimal_data)

        # Should return current optimized value (default 0.25) with insufficient data
        assert optimal_fraction == 0.25


class TestRegimeDetectionValidatorAdvanced:
    """Advanced tests for regime detection validation."""

    def setup_method(self):
        self.validator = RegimeDetectionValidator()

    def test_cross_validate_regime_params_with_data(self):
        """Test cross-validation with actual data."""
        np.random.seed(42)
        # Create synthetic returns data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        returns = np.random.normal(0.001, 0.02, 500)
        df = pd.DataFrame({'returns': returns}, index=dates)

        result = self.validator.cross_validate_regime_params(df, 'returns')

        # Should return synthesized parameters
        assert 'thresholds' in result
        assert 'lookback' in result
        assert 'validation_accuracy' in result

    def test_cross_validate_handles_different_periods(self):
        """Test cross-validation handles different market periods."""
        np.random.seed(42)
        # Create longer data
        dates = pd.date_range('2018-01-01', periods=1000, freq='D')
        returns = np.random.normal(0.0005, 0.018, 1000)
        df = pd.DataFrame({'returns': returns}, index=dates)

        result = self.validator.cross_validate_regime_params(df, 'returns')

        assert 'period_results' in result
        # Should have results for multiple periods
        assert len(result.get('period_results', {})) >= 1


class TestCorrelationRegimeDetectorAdvanced:
    """Advanced tests for correlation regime detection."""

    def setup_method(self):
        self.detector = CorrelationRegimeDetector()

    def test_calculate_correlation_entropy_stable(self):
        """Test entropy calculation for stable correlation matrix."""
        # Create a stable correlation matrix (moderate correlations)
        stable_corr = np.array([
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0]
        ])

        entropy = self.detector.calculate_correlation_entropy(stable_corr)

        # Stable correlations should have positive entropy
        assert entropy > 0.0

    def test_calculate_correlation_entropy_unstable(self):
        """Test entropy calculation for unstable correlation matrix (crisis)."""
        # Create an unstable correlation matrix (all highly correlated)
        unstable_corr = np.array([
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.94],
            [0.92, 0.94, 1.0]
        ])

        entropy = self.detector.calculate_correlation_entropy(unstable_corr)

        # Should still return valid entropy
        assert entropy > 0.0

    def test_monitor_correlation_stability_builds_history(self):
        """Test that monitoring builds entropy history."""
        base_corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        # Monitor multiple times
        for _ in range(10):
            self.detector.monitor_correlation_stability(base_corr)

        assert len(self.detector.entropy_history) == 10
        # First 4 calls return early (need 5 entropy samples), so regime_history has 6 entries
        assert len(self.detector.regime_history) == 6

    def test_get_correlation_regime_multiplier_all_regimes(self):
        """Test position sizing multipliers for all correlation regimes."""
        assert self.detector.get_correlation_regime_multiplier('stable') == 1.0
        assert self.detector.get_correlation_regime_multiplier('correlation_normalization') == 1.1
        assert self.detector.get_correlation_regime_multiplier('correlation_breakdown') == 0.3
        assert self.detector.get_correlation_regime_multiplier('unknown') == 1.0

    def test_get_regime_statistics(self):
        """Test regime statistics retrieval."""
        # Build some history
        corr = np.eye(3)
        for _ in range(5):
            self.detector.monitor_correlation_stability(corr)

        stats = self.detector.get_regime_statistics()

        assert 'regime_counts' in stats
        assert 'current_entropy' in stats
        assert 'entropy_trend' in stats
        assert 'history_length' in stats


class TestPortfolioRiskBudgetAdvanced:
    """Advanced tests for portfolio risk budgeting."""

    def setup_method(self):
        self.risk_budget = PortfolioRiskBudget()
        self.sample_portfolio = {'AAPL': 0.1, 'MSFT': 0.08, 'GOOGL': 0.06}
        self.sample_macro_context = {'vix': 20.0}

    def test_calculate_risk_budget_basic(self):
        """Test basic risk budget calculation."""
        risk_scores = self.risk_budget.calculate_risk_budget(
            self.sample_portfolio, self.sample_macro_context
        )

        # Should return risk score for each position
        assert 'AAPL' in risk_scores
        assert 'MSFT' in risk_scores
        assert 'GOOGL' in risk_scores

        # All risk scores should be positive
        for score in risk_scores.values():
            assert score >= 0.0

    def test_calculate_risk_budget_with_high_vix(self):
        """Test risk budget with high VIX (crisis levels)."""
        high_risk_context = {'vix': 45.0}

        risk_scores = self.risk_budget.calculate_risk_budget(
            self.sample_portfolio, high_risk_context
        )

        # Total risk should be constrained
        total_risk = sum(risk_scores.values())
        assert total_risk <= self.risk_budget.total_risk_budget

    def test_get_macro_risk_multiplier_various_levels(self):
        """Test macro risk multiplier across VIX levels."""
        assert self.risk_budget._get_macro_risk_multiplier({'vix': 10.0}) == 1.2
        assert self.risk_budget._get_macro_risk_multiplier({'vix': 20.0}) == 1.0
        assert self.risk_budget._get_macro_risk_multiplier({'vix': 30.0}) == 0.7
        assert self.risk_budget._get_macro_risk_multiplier({'vix': 40.0}) == 0.4

    def test_risk_budget_normalization(self):
        """Test risk budget normalization when total risk exceeds budget."""
        # Large portfolio that would exceed budget
        large_portfolio = {f'STOCK_{i}': 0.15 for i in range(10)}

        risk_scores = self.risk_budget.calculate_risk_budget(
            large_portfolio, self.sample_macro_context
        )

        total_risk = sum(risk_scores.values())
        # Should be normalized to not exceed 1.0
        assert total_risk <= 1.0 + 1e-10  # Allow small floating point error

    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        self.risk_budget.calculate_risk_budget(
            self.sample_portfolio, self.sample_macro_context
        )

        remaining = self.risk_budget.get_remaining_budget()
        assert remaining >= 0
        assert remaining <= self.risk_budget.total_risk_budget

    def test_get_budget_summary(self):
        """Test budget summary."""
        self.risk_budget.calculate_risk_budget(
            self.sample_portfolio, self.sample_macro_context
        )

        summary = self.risk_budget.get_budget_summary()

        assert 'total_budget' in summary
        assert 'allocated' in summary
        assert 'remaining' in summary
        assert 'position_allocations' in summary


class TestMLEnhancedPhase5Advanced:
    """Advanced tests for ML enhancement."""

    def setup_method(self):
        self.ml_enhancer = MLEnhancedPhase5()
        self.sample_signals = {
            'macro_adjusted_score': 0.75,
            'regime_kelly_position': 0.08,
            'diversification_penalty': 0.15,
            'timeframe_agreement': 0.8,
            'correlation_regime': 'stable',
            'vix_level': 18.0,
            'market_trend': 0.05
        }

    def test_prepare_features_complete(self):
        """Test feature preparation with all signals."""
        features = self.ml_enhancer.prepare_features(self.sample_signals)

        assert len(features) == 7
        assert features[0] == 0.75  # macro_adjusted_score
        assert features[4] == 0    # 'stable' mapped to 0

    def test_prepare_features_missing_values(self):
        """Test feature preparation with missing values."""
        incomplete_signals = {'macro_adjusted_score': 0.6}

        features = self.ml_enhancer.prepare_features(incomplete_signals)

        # Should still produce 7 features with defaults
        assert len(features) == 7

    def test_add_training_sample(self):
        """Test adding training samples."""
        self.ml_enhancer.add_training_sample(self.sample_signals, was_profitable=True)
        self.ml_enhancer.add_training_sample(self.sample_signals, was_profitable=False)

        assert len(self.ml_enhancer.training_data) == 2
        assert self.ml_enhancer.training_data[0]['actual_profitability'] == 1
        assert self.ml_enhancer.training_data[1]['actual_profitability'] == 0

    def test_get_ml_enhanced_decision_not_trained(self):
        """Test ML decision when model not trained."""
        base_decision = {'position': 0.08, 'confidence': 0.75}

        result = self.ml_enhancer.get_ml_enhanced_decision(
            self.sample_signals, base_decision
        )

        # Should return base decision when not trained
        assert result == base_decision

    def test_get_feature_importance_not_trained(self):
        """Test feature importance when not trained."""
        importance = self.ml_enhancer.get_feature_importance()
        assert importance == {}


class TestIntegrationScenariosAdvanced:
    """Advanced integration tests for real-world scenarios."""

    def test_full_crisis_scenario(self):
        """Test complete crisis scenario with all advanced features."""
        # Crisis conditions
        crisis_macro = {
            'risk_off': True,
            'vix': 45.0,
            'market_regime': 'crisis',
            'spy_trend': -0.18
        }

        crisis_portfolio = {'SPY': 0.15, 'QQQ': 0.12, 'IWM': 0.08}

        # Initialize components
        monitor = Phase5PerformanceMonitor()
        optimizer = AdaptiveParameterOptimizer()
        correlation_detector = CorrelationRegimeDetector()
        risk_budget = PortfolioRiskBudget()

        # Test risk budgeting in crisis
        risk_scores = risk_budget.calculate_risk_budget(
            crisis_portfolio, crisis_macro
        )

        # Total risk should be very low in crisis (VIX 45 -> multiplier 0.4)
        total_risk = sum(risk_scores.values())
        assert total_risk < 0.5  # Very conservative in crisis

    def test_adaptive_learning_scenario(self):
        """Test adaptive learning over multiple periods."""
        optimizer = AdaptiveParameterOptimizer(min_samples=5, learning_rate=0.3)

        # Simulate learning over multiple periods
        kelly_fractions = []

        for period in range(5):
            # Simulate improving performance
            np.random.seed(period)
            returns = list(np.random.normal(0.01 + period * 0.002, 0.015, 50))

            # Optimize Kelly fraction based on performance
            optimal_kelly = optimizer.optimize_kelly_fraction(returns)
            kelly_fractions.append(optimal_kelly)

        # Kelly should adapt (values should be within valid bounds)
        for kf in kelly_fractions:
            assert 0.1 <= kf <= 0.4

    def test_production_system_full_cycle(self):
        """Test production system through full trading cycle."""
        system = create_production_phase5_system(enable_ml=False, enable_monitoring=True)

        # Generate multiple signals
        for i in range(5):
            signal = system.generate_production_signal(
                ticker=f'STOCK_{i}',
                data=pd.DataFrame({'close': [100 + i, 101 + i, 102 + i]}),
                raw_signals={
                    'direction': 0.5 if i % 2 == 0 else -0.3,
                    'confidence': 0.6 + i * 0.05,
                    'position_size': 0.05
                },
                macro_context={'vix': 20 + i}
            )

            # Record outcome
            actual_return = 0.02 if i % 2 == 0 else -0.01
            system.record_outcome(f'STOCK_{i}', signal, actual_return, 'equity')

        # Check system status
        status = system.get_system_status()
        assert status['total_decisions'] == 5
        assert 'win_rate' in status


class TestEdgeCasesAdvanced:
    """Test edge cases for advanced improvements."""

    def test_empty_portfolio_risk_budget(self):
        """Test risk budget with empty portfolio."""
        risk_budget = PortfolioRiskBudget()
        empty_portfolio = {}

        risk_scores = risk_budget.calculate_risk_budget(
            empty_portfolio, {'vix': 20.0}
        )

        assert risk_scores == {}

    def test_extreme_vix_values_risk_multiplier(self):
        """Test risk multiplier with extreme VIX values."""
        risk_budget = PortfolioRiskBudget()

        # Test very low VIX
        assert risk_budget._get_macro_risk_multiplier({'vix': 5.0}) == 1.2

        # Test very high VIX
        assert risk_budget._get_macro_risk_multiplier({'vix': 80.0}) == 0.4

        # Test missing VIX (should default to normal)
        mult = risk_budget._get_macro_risk_multiplier({})
        assert mult == 1.0  # Default VIX 20 -> normal multiplier

    def test_correlation_entropy_edge_cases(self):
        """Test correlation entropy with edge case matrices."""
        detector = CorrelationRegimeDetector()

        # Identity matrix
        identity = np.eye(5)
        entropy = detector.calculate_correlation_entropy(identity)
        assert entropy >= 0

        # Perfect correlation matrix
        ones = np.ones((3, 3))
        entropy = detector.calculate_correlation_entropy(ones)
        # May have tiny floating point errors near zero
        assert entropy >= -1e-9  # Should handle gracefully

    def test_optimizer_with_volatile_returns(self):
        """Test optimizer with highly volatile returns."""
        optimizer = AdaptiveParameterOptimizer()

        np.random.seed(42)
        # Very volatile returns
        volatile_returns = list(np.random.normal(0.0, 0.10, 100))

        kelly = optimizer.optimize_kelly_fraction(volatile_returns)

        # Should be conservative with high volatility
        assert kelly <= 0.25

    def test_monitor_with_rapid_changes(self):
        """Test monitor handles rapid performance changes."""
        monitor = Phase5PerformanceMonitor(lookback_window=5)

        # Rapid alternation between good and bad
        for i in range(20):
            effectiveness = 1.0 if i % 2 == 0 else -1.0
            monitor.track_macro_adjustment_effectiveness(
                0.5, 0.6 if i % 2 == 0 else 0.4, 0.01 * effectiveness
            )

        # Should have tracked all
        summary = monitor.get_performance_summary()
        assert summary['macro_adjustment_effectiveness']['count'] == 20


# =============================================================================
# QUICK VALIDATION
# =============================================================================

def run_quick_validation():
    """Run quick validation of all advanced features."""
    print("=" * 60)
    print("Phase 5 Advanced Final Tests - Quick Validation")
    print("=" * 60)

    # Test 1: Performance Monitor with alerts
    print("\n1. Testing Performance Monitor with Alerts...")
    monitor = Phase5PerformanceMonitor(lookback_window=10)
    # Add ineffective adjustments to trigger alert
    for _ in range(15):
        monitor.track_macro_adjustment_effectiveness(0.7, 0.4, 0.05)
    alert_count = len(monitor.alerts_history)
    print(f"   Alerts triggered: {alert_count}")
    assert alert_count > 0, "Should trigger alert for poor performance"
    print("   [OK]")

    # Test 2: Adaptive Optimizer learning
    print("\n2. Testing Adaptive Optimizer Learning...")
    optimizer = AdaptiveParameterOptimizer(min_samples=5)
    for i in range(15):
        optimizer.record_performance('equity', 'risk_on', 0.02)
        optimizer.record_performance('equity', 'risk_off', -0.01)
    boosts = optimizer.optimize_macro_boost_factors()
    print(f"   Equity risk_on boost: {boosts.get('equity', {}).get('risk_on', 'N/A'):.3f}")
    print(f"   Equity risk_off boost: {boosts.get('equity', {}).get('risk_off', 'N/A'):.3f}")
    assert boosts['equity']['risk_on'] > boosts['equity']['risk_off']
    print("   [OK]")

    # Test 3: Correlation Regime Detection
    print("\n3. Testing Correlation Regime Detection...")
    detector = CorrelationRegimeDetector()
    # Build history - need 5 entropy samples before regime detection starts
    # So with 10 calls, only 6 get regime history (calls 5-10)
    stable_corr = np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]])
    for _ in range(10):
        detector.monitor_correlation_stability(stable_corr)
    stats = detector.get_regime_statistics()
    print(f"   History length: {stats['history_length']}")
    print(f"   Current entropy: {stats['current_entropy']:.3f}")
    # First 4 calls return early (need 5 entropy samples), so regime_history has 6 entries
    assert stats['history_length'] == 6
    print("   [OK]")

    # Test 4: Risk Budget in Crisis
    print("\n4. Testing Risk Budget in Crisis...")
    risk_budget = PortfolioRiskBudget()
    portfolio = {'SPY': 0.2, 'QQQ': 0.15, 'IWM': 0.1}
    crisis_alloc = risk_budget.calculate_risk_budget(portfolio, {'vix': 45})
    normal_alloc = risk_budget.calculate_risk_budget(portfolio, {'vix': 20})
    crisis_total = sum(crisis_alloc.values())
    normal_total = sum(normal_alloc.values())
    print(f"   Crisis allocation: {crisis_total:.3f}")
    print(f"   Normal allocation: {normal_total:.3f}")
    assert crisis_total < normal_total, "Crisis should have lower allocation"
    print("   [OK]")

    # Test 5: Production System
    print("\n5. Testing Production System End-to-End...")
    system = create_production_phase5_system(enable_ml=False)
    signal = system.generate_production_signal(
        ticker='AAPL',
        data=pd.DataFrame({'close': [150, 151, 152]}),
        raw_signals={'direction': 0.6, 'confidence': 0.75, 'position_size': 0.08},
        macro_context={'vix': 18, 'spy_trend': 0.02}
    )
    system.record_outcome('AAPL', signal, 0.025, 'equity')
    status = system.get_system_status()
    print(f"   Decisions: {status['total_decisions']}")
    print(f"   Win rate: {status['win_rate']:.1%}")
    assert status['total_decisions'] == 1
    assert status['profitable_decisions'] == 1
    print("   [OK]")

    print("\n" + "=" * 60)
    print("Phase 5 Advanced Final Tests Validation PASSED")
    print("=" * 60)

    return True


if __name__ == '__main__':
    # Run quick validation first
    if run_quick_validation():
        print("\n\nRunning full test suite...\n")
        pytest.main([__file__, '-v', '--tb=short'])
