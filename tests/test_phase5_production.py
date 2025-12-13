"""
Tests for Phase 5 Production-Grade Improvements

Tests all 6 production improvements:
1. Real-Time Performance Monitoring
2. Adaptive Parameter Optimization
3. Cross-Validation for Regime Detection
4. Advanced Correlation Regime Detection
5. Portfolio-Level Risk Budgeting
6. ML Enhancement
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, 'src')

from ensemble.phase5_production import (
    Phase5PerformanceMonitor,
    PerformanceAlert,
    AdaptiveParameterOptimizer,
    RegimeDetectionValidator,
    CorrelationRegimeDetector,
    PortfolioRiskBudget,
    MLEnhancedPhase5,
    ProductionPhase5System,
    create_production_phase5_system
)


class TestPhase5PerformanceMonitor:
    """Test real-time performance monitoring."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = Phase5PerformanceMonitor(
            alert_threshold=0.4,
            lookback_window=10
        )

    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.alert_threshold == 0.4
        assert self.monitor.lookback_window == 10
        assert len(self.monitor.performance_metrics) == 6

    def test_track_macro_adjustment_effectiveness_good(self):
        """Test tracking effective macro adjustments."""
        # Simulate good adjustments (aligned with returns)
        for _ in range(15):
            self.monitor.track_macro_adjustment_effectiveness(
                original_score=0.5,
                adjusted_score=0.7,
                actual_return=0.02  # Positive return, higher adjusted score is correct
            )

        summary = self.monitor.get_performance_summary()
        assert summary['macro_adjustment_effectiveness']['count'] == 15
        assert len(self.monitor.alerts_history) == 0  # No alerts for good performance

    def test_track_macro_adjustment_effectiveness_triggers_alert(self):
        """Test that poor performance triggers alert."""
        # Simulate poor adjustments
        for _ in range(15):
            self.monitor.track_macro_adjustment_effectiveness(
                original_score=0.7,
                adjusted_score=0.5,  # Lowered score
                actual_return=0.02   # But positive return - adjustment was wrong
            )

        # Should trigger alert after lookback_window observations
        assert len(self.monitor.alerts_history) > 0
        assert self.monitor.alerts_history[0].alert_type == "MACRO_ADJUSTMENT_INEFFECTIVE"

    def test_track_regime_accuracy(self):
        """Test regime detection accuracy tracking."""
        # Accurate predictions
        for _ in range(10):
            self.monitor.track_regime_accuracy(
                predicted_regime='normal',
                actual_volatility=0.15,
                historical_volatility=0.15
            )

        summary = self.monitor.get_performance_summary()
        assert summary['regime_detection_accuracy']['mean'] == 1.0

    def test_track_kelly_sizing_performance(self):
        """Test Kelly sizing performance tracking."""
        # Profitable trade with good sizing
        self.monitor.track_kelly_sizing_performance(
            position_size=0.1,
            actual_return=0.05,
            was_profitable=True
        )

        # Losing trade
        self.monitor.track_kelly_sizing_performance(
            position_size=0.05,
            actual_return=-0.02,
            was_profitable=False
        )

        summary = self.monitor.get_performance_summary()
        assert summary['kelly_sizing_performance']['count'] == 2

    def test_performance_summary(self):
        """Test getting performance summary."""
        # Add various metrics
        self.monitor.track_signal_accuracy(0.5, 0.02)
        self.monitor.track_signal_accuracy(0.5, 0.01)
        self.monitor.track_signal_accuracy(-0.5, -0.01)

        summary = self.monitor.get_performance_summary()

        assert 'overall_signal_accuracy' in summary
        assert summary['overall_signal_accuracy']['count'] == 3
        assert summary['total_alerts'] >= 0


class TestAdaptiveParameterOptimizer:
    """Test adaptive parameter optimization."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AdaptiveParameterOptimizer(
            optimization_lookback=30,
            min_samples=5,
            learning_rate=0.2
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.optimization_lookback == 30
        assert self.optimizer.min_samples == 5
        assert self.optimizer.optimized_kelly_fraction == 0.25

    def test_record_performance(self):
        """Test recording performance data."""
        self.optimizer.record_performance('equity', 'risk_on', 0.02)
        self.optimizer.record_performance('equity', 'risk_off', -0.01)
        self.optimizer.record_performance('bond', 'risk_off', 0.01)

        assert 'equity' in self.optimizer.performance_by_asset_class
        assert len(self.optimizer.performance_by_asset_class['equity']['risk_on']) == 1
        assert len(self.optimizer.performance_by_regime['risk_on']) == 1

    def test_optimize_macro_boost_factors(self):
        """Test macro boost factor optimization."""
        # Record enough data
        for i in range(10):
            self.optimizer.record_performance('equity', 'risk_on', 0.01 + i * 0.001)
            self.optimizer.record_performance('equity', 'risk_off', -0.005)
            self.optimizer.record_performance('bond', 'risk_off', 0.008)

        optimized = self.optimizer.optimize_macro_boost_factors()

        assert 'equity' in optimized
        assert 'risk_on' in optimized['equity']
        assert 'risk_off' in optimized['equity']
        # Boosts should be bounded
        assert 0.5 <= optimized['equity']['risk_on'] <= 2.0
        assert 0.5 <= optimized['equity']['risk_off'] <= 2.0

    def test_optimize_kelly_fraction_high_sharpe(self):
        """Test Kelly optimization with high Sharpe."""
        # High Sharpe returns
        np.random.seed(42)
        good_returns = list(np.random.normal(0.02, 0.01, 100))  # High mean, low std

        kelly = self.optimizer.optimize_kelly_fraction(good_returns)

        # Should be more aggressive with good performance
        assert kelly >= 0.20
        assert kelly <= 0.40

    def test_optimize_kelly_fraction_low_sharpe(self):
        """Test Kelly optimization with low Sharpe."""
        # Low Sharpe returns
        np.random.seed(42)
        bad_returns = list(np.random.normal(0.001, 0.03, 100))  # Low mean, high std

        kelly = self.optimizer.optimize_kelly_fraction(bad_returns)

        # Should be more conservative
        assert kelly >= 0.10
        assert kelly <= 0.30

    def test_get_optimized_parameters(self):
        """Test getting all optimized parameters."""
        params = self.optimizer.get_optimized_parameters()

        assert 'boost_factors' in params
        assert 'kelly_fraction' in params
        assert 'samples_collected' in params


class TestRegimeDetectionValidator:
    """Test cross-validation for regime detection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = RegimeDetectionValidator()

    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.default_thresholds == {'low': 0.7, 'normal': 1.3, 'high': 2.0}

    def test_cross_validate_with_sufficient_data(self):
        """Test cross-validation with sufficient data."""
        # Generate synthetic returns
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        returns = np.random.normal(0.001, 0.02, 500)

        df = pd.DataFrame({'returns': returns}, index=dates)

        result = self.validator.cross_validate_regime_params(df, 'returns')

        assert 'thresholds' in result
        assert 'lookback' in result
        assert 'validation_accuracy' in result

    def test_cross_validate_insufficient_data(self):
        """Test cross-validation with insufficient data."""
        df = pd.DataFrame({'returns': [0.01] * 50})

        result = self.validator.cross_validate_regime_params(df, 'returns')

        assert 'error' in result

    def test_cross_validate_missing_column(self):
        """Test cross-validation with missing column."""
        df = pd.DataFrame({'price': [100, 101, 102]})

        result = self.validator.cross_validate_regime_params(df, 'returns')

        assert 'error' in result


class TestCorrelationRegimeDetector:
    """Test advanced correlation regime detection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = CorrelationRegimeDetector(
            entropy_threshold_high=0.15,
            entropy_threshold_low=-0.10,
            lookback_days=30
        )

    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.entropy_threshold_high == 0.15
        assert self.detector.lookback_days == 30

    def test_calculate_correlation_entropy(self):
        """Test correlation entropy calculation."""
        # Create a simple correlation matrix
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        entropy = self.detector.calculate_correlation_entropy(corr_matrix)

        assert entropy > 0
        assert isinstance(entropy, float)

    def test_monitor_correlation_stability_initial(self):
        """Test initial monitoring returns stable."""
        corr_matrix = np.eye(5)

        regime = self.detector.monitor_correlation_stability(corr_matrix)

        assert regime == 'stable'  # Not enough history

    def test_monitor_correlation_stability_with_history(self):
        """Test monitoring with accumulated history."""
        # Build up history with stable correlations
        base_corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        for _ in range(10):
            self.detector.monitor_correlation_stability(base_corr)

        regime = self.detector.monitor_correlation_stability(base_corr)

        assert regime in ['stable', 'correlation_breakdown', 'correlation_normalization']

    def test_get_correlation_regime_multiplier(self):
        """Test regime multipliers."""
        assert self.detector.get_correlation_regime_multiplier('stable') == 1.0
        assert self.detector.get_correlation_regime_multiplier('correlation_normalization') == 1.1
        assert self.detector.get_correlation_regime_multiplier('correlation_breakdown') == 0.3
        assert self.detector.get_correlation_regime_multiplier('unknown') == 1.0

    def test_get_regime_statistics(self):
        """Test getting regime statistics."""
        stats = self.detector.get_regime_statistics()

        assert 'regime_counts' in stats
        assert 'current_entropy' in stats


class TestPortfolioRiskBudget:
    """Test portfolio-level risk budgeting."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_budget = PortfolioRiskBudget(
            total_risk_budget=1.0,
            max_single_position_risk=0.25,
            min_position_risk=0.02
        )

    def test_initialization(self):
        """Test risk budget initialization."""
        assert self.risk_budget.total_risk_budget == 1.0
        assert self.risk_budget.max_single_position_risk == 0.25

    def test_calculate_risk_budget_empty_portfolio(self):
        """Test with empty portfolio."""
        result = self.risk_budget.calculate_risk_budget({}, {'vix': 20})

        assert result == {}

    def test_calculate_risk_budget_single_position(self):
        """Test with single position."""
        portfolio = {'AAPL': 0.2}
        macro_context = {'vix': 20}

        result = self.risk_budget.calculate_risk_budget(portfolio, macro_context)

        assert 'AAPL' in result
        assert 0.02 <= result['AAPL'] <= 0.25

    def test_calculate_risk_budget_multiple_positions(self):
        """Test with multiple positions."""
        portfolio = {'AAPL': 0.2, 'GOOGL': 0.15, 'MSFT': 0.1}
        macro_context = {'vix': 20}

        result = self.risk_budget.calculate_risk_budget(portfolio, macro_context)

        assert len(result) == 3
        # Total should not exceed budget
        assert sum(result.values()) <= self.risk_budget.total_risk_budget

    def test_macro_risk_multiplier_low_vix(self):
        """Test macro multiplier with low VIX."""
        mult = self.risk_budget._get_macro_risk_multiplier({'vix': 12})
        assert mult == 1.2  # Low fear = more risk

    def test_macro_risk_multiplier_high_vix(self):
        """Test macro multiplier with high VIX."""
        mult = self.risk_budget._get_macro_risk_multiplier({'vix': 40})
        assert mult == 0.4  # Crisis = minimal risk

    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        portfolio = {'AAPL': 0.3}
        self.risk_budget.calculate_risk_budget(portfolio, {'vix': 20})

        remaining = self.risk_budget.get_remaining_budget()

        assert remaining >= 0
        assert remaining <= self.risk_budget.total_risk_budget

    def test_get_budget_summary(self):
        """Test getting budget summary."""
        portfolio = {'AAPL': 0.2}
        self.risk_budget.calculate_risk_budget(portfolio, {'vix': 20})

        summary = self.risk_budget.get_budget_summary()

        assert 'total_budget' in summary
        assert 'allocated' in summary
        assert 'remaining' in summary
        assert 'position_allocations' in summary


class TestMLEnhancedPhase5:
    """Test ML enhancement."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ml_enhancer = MLEnhancedPhase5(
            ml_weight=0.3,
            confidence_threshold_high=0.7,
            confidence_threshold_low=0.3
        )

    def test_initialization(self):
        """Test ML enhancer initialization."""
        assert self.ml_enhancer.ml_weight == 0.3
        assert not self.ml_enhancer.is_trained
        assert len(self.ml_enhancer.feature_names) == 7

    def test_prepare_features(self):
        """Test feature preparation."""
        signals = {
            'macro_adjusted_score': 0.6,
            'regime_kelly_position': 0.05,
            'diversification_penalty': 0.1,
            'timeframe_agreement': 0.7,
            'correlation_regime': 'stable',
            'vix_level': 20,
            'market_trend': 0.02
        }

        features = self.ml_enhancer.prepare_features(signals)

        assert len(features) == 7
        assert features[4] == 0  # 'stable' mapped to 0

    def test_get_ml_enhanced_decision_not_trained(self):
        """Test ML decision when not trained."""
        base_decision = {'position': 0.05, 'confidence': 0.6}

        result = self.ml_enhancer.get_ml_enhanced_decision({}, base_decision)

        # Should return base decision when not trained
        assert result == base_decision

    def test_add_training_sample(self):
        """Test adding training samples."""
        signals = {'macro_adjusted_score': 0.6}

        self.ml_enhancer.add_training_sample(signals, was_profitable=True)
        self.ml_enhancer.add_training_sample(signals, was_profitable=False)

        assert len(self.ml_enhancer.training_data) == 2
        assert self.ml_enhancer.training_data[0]['actual_profitability'] == 1
        assert self.ml_enhancer.training_data[1]['actual_profitability'] == 0

    def test_get_feature_importance_not_trained(self):
        """Test feature importance when not trained."""
        importance = self.ml_enhancer.get_feature_importance()

        assert importance == {}


class TestProductionPhase5System:
    """Test integrated production system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.system = create_production_phase5_system(
            base_system=None,
            enable_ml=True,
            enable_monitoring=True
        )

    def test_initialization(self):
        """Test system initialization."""
        assert self.system.enable_ml
        assert self.system.enable_monitoring
        assert self.system.performance_monitor is not None
        assert self.system.parameter_optimizer is not None
        assert self.system.correlation_detector is not None
        assert self.system.risk_budget is not None
        assert self.system.ml_enhancer is not None

    def test_generate_production_signal(self):
        """Test generating production signal."""
        np.random.seed(42)
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        })

        raw_signals = {
            'composite_score': 0.6,
            'kelly_position': 0.05,
            'direction': 0.5,
            'confidence': 0.7,
            'position_size': 0.08
        }

        macro_context = {'vix': 20, 'spy_trend': 0.02}

        signal = self.system.generate_production_signal(
            ticker='AAPL',
            data=data,
            raw_signals=raw_signals,
            macro_context=macro_context
        )

        assert 'ticker' in signal
        assert 'direction' in signal
        assert 'confidence' in signal
        assert 'position' in signal
        assert 'production_grade' in signal
        assert signal['production_grade'] == True

    def test_record_outcome(self):
        """Test recording trade outcome."""
        # First generate a signal (which increments total_decisions)
        signal = self.system.generate_production_signal(
            ticker='AAPL',
            data=pd.DataFrame({'close': [100, 101, 102]}),
            raw_signals={'direction': 0.5, 'confidence': 0.7, 'position_size': 0.05},
            macro_context={'vix': 20}
        )

        # Then record the outcome
        self.system.record_outcome(
            ticker='AAPL',
            signal=signal,
            actual_return=0.02,
            asset_class='equity'
        )

        assert self.system.total_decisions == 1
        assert self.system.profitable_decisions == 1

    def test_get_system_status(self):
        """Test getting system status."""
        status = self.system.get_system_status()

        assert 'total_decisions' in status
        assert 'win_rate' in status
        assert 'ml_enabled' in status
        assert 'parameter_optimization' in status
        assert 'correlation_regime' in status
        assert 'risk_budget' in status


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_full_trading_cycle(self):
        """Test complete trading cycle."""
        system = create_production_phase5_system(enable_ml=False)

        # Generate signal
        signal = system.generate_production_signal(
            ticker='AAPL',
            data=pd.DataFrame({'close': [100, 101, 102]}),
            raw_signals={'direction': 0.5, 'confidence': 0.7, 'position_size': 0.05},
            macro_context={'vix': 20}
        )

        # Record outcome
        system.record_outcome('AAPL', signal, 0.02, 'equity')

        # Check status
        status = system.get_system_status()
        assert status['total_decisions'] == 1

    def test_parameter_optimization_cycle(self):
        """Test parameter optimization cycle."""
        system = create_production_phase5_system(enable_ml=False)

        # Record multiple outcomes
        for i in range(30):
            signal = {'direction': 0.5, 'position': 0.05, 'correlation_regime': 'stable'}
            return_val = 0.01 if i % 3 != 0 else -0.005
            system.record_outcome('AAPL', signal, return_val, 'equity')

        # Optimize parameters
        trade_results = [0.01 if i % 3 != 0 else -0.005 for i in range(30)]
        optimized = system.optimize_parameters(trade_results)

        assert 'boost_factors' in optimized
        assert 'kelly_fraction' in optimized

    def test_correlation_crisis_scenario(self):
        """Test behavior during correlation crisis."""
        system = create_production_phase5_system()

        # Simulate crisis correlation matrix
        crisis_corr = np.ones((5, 5)) * 0.95
        np.fill_diagonal(crisis_corr, 1.0)

        # Monitor multiple times to build history
        for _ in range(10):
            system.correlation_detector.monitor_correlation_stability(crisis_corr)

        regime = system.correlation_detector.monitor_correlation_stability(crisis_corr)
        multiplier = system.correlation_detector.get_correlation_regime_multiplier(regime)

        # In crisis, should have reduced multiplier
        assert multiplier <= 1.0


# =============================================================================
# QUICK VALIDATION
# =============================================================================

def run_quick_validation():
    """Run quick validation of all production components."""
    print("=" * 60)
    print("Phase 5 Production Improvements - Quick Validation")
    print("=" * 60)

    # Test 1: Performance Monitor
    print("\n1. Testing Performance Monitor...")
    monitor = Phase5PerformanceMonitor()
    for i in range(25):
        monitor.track_signal_accuracy(0.5, 0.01 if i % 2 == 0 else -0.005)
    summary = monitor.get_performance_summary()
    print(f"   Signal accuracy tracked: {summary['overall_signal_accuracy']['count']} samples")
    print(f"   Mean accuracy: {summary['overall_signal_accuracy']['mean']:.2f}")
    print("   [OK]")

    # Test 2: Adaptive Parameter Optimizer
    print("\n2. Testing Adaptive Parameter Optimizer...")
    optimizer = AdaptiveParameterOptimizer(min_samples=5)
    for i in range(20):
        optimizer.record_performance('equity', 'risk_on', 0.01 + np.random.randn() * 0.005)
    optimized = optimizer.optimize_macro_boost_factors()
    kelly = optimizer.optimize_kelly_fraction([0.01] * 50 + [-0.005] * 10)
    print(f"   Equity risk_on boost: {optimized.get('equity', {}).get('risk_on', 'N/A'):.3f}")
    print(f"   Optimized Kelly: {kelly:.3f}")
    print("   [OK]")

    # Test 3: Correlation Regime Detector
    print("\n3. Testing Correlation Regime Detector...")
    detector = CorrelationRegimeDetector()
    corr_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    entropy = detector.calculate_correlation_entropy(corr_matrix)
    for _ in range(10):
        detector.monitor_correlation_stability(corr_matrix)
    regime = detector.monitor_correlation_stability(corr_matrix)
    print(f"   Correlation entropy: {entropy:.3f}")
    print(f"   Current regime: {regime}")
    print("   [OK]")

    # Test 4: Portfolio Risk Budget
    print("\n4. Testing Portfolio Risk Budget...")
    risk_budget = PortfolioRiskBudget()
    portfolio = {'AAPL': 0.2, 'GOOGL': 0.15, 'MSFT': 0.1}
    allocation = risk_budget.calculate_risk_budget(portfolio, {'vix': 20})
    remaining = risk_budget.get_remaining_budget()
    print(f"   Allocated risk: {sum(allocation.values()):.3f}")
    print(f"   Remaining budget: {remaining:.3f}")
    print("   [OK]")

    # Test 5: Production System
    print("\n5. Testing Production System...")
    system = create_production_phase5_system(enable_ml=False)
    signal = system.generate_production_signal(
        ticker='AAPL',
        data=pd.DataFrame({'close': [100, 101, 102]}),
        raw_signals={'direction': 0.5, 'confidence': 0.7, 'position_size': 0.05},
        macro_context={'vix': 20}
    )
    system.record_outcome('AAPL', signal, 0.02, 'equity')
    status = system.get_system_status()
    print(f"   Signal generated: production_grade={signal.get('production_grade')}")
    print(f"   Win rate: {status['win_rate']:.1%}")
    print("   [OK]")

    print("\n" + "=" * 60)
    print("Phase 5 Production Improvements Validation PASSED")
    print("=" * 60)

    return True


if __name__ == '__main__':
    # Run quick validation first
    if run_quick_validation():
        print("\n\nRunning full test suite...\n")
        pytest.main([__file__, '-v'])
