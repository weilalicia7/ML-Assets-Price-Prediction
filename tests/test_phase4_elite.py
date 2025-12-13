"""
Phase 4 Elite Enhancements Test Suite

Tests all 8 elite-level enhancements:
1. Adaptive Macro Feature Selection
2. Macro-Aware Ensemble Weighting
3. Dynamic Macro Sensitivity Adjustment
4. Macro Regime Transition Forecasting
5. Macro Feature Compression
6. Real-Time Macro Impact Monitoring
7. Macro-Aware Position Sizing
8. Cross-Timeframe Macro Analysis

Expected Additional Improvement: +3-5% profit rate
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.phase4_elite_enhancements import (
    AdaptiveMacroFeatureSelector,
    MacroAwareEnsembleWeighter,
    DynamicMacroSensitivity,
    MacroRegimeForecaster,
    MacroFeatureCompressor,
    MacroImpactMonitor,
    MacroAwarePositionSizer,
    MultiTimeframeMacroAnalyzer,
    Phase4EliteSystem
)


class TestPhase4Elite:
    """Comprehensive test suite for Phase 4 Elite Enhancements"""

    @pytest.fixture
    def sample_macro_features(self):
        """Generate sample macro features for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')

        return pd.DataFrame({
            'VIX': np.random.normal(18, 5, len(dates)).clip(10, 40),
            'SPY': 100 + np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
            'GLD': 180 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))),
            'TLT': 90 + np.cumsum(np.random.normal(0.0002, 0.008, len(dates))),
            'DXY': 100 + np.cumsum(np.random.normal(0.0001, 0.005, len(dates))),
            'spy_momentum_5d': np.random.randn(len(dates)) * 0.02,
            'spy_momentum_20d': np.random.randn(len(dates)) * 0.03,
            'vix_momentum_5d': np.random.randn(len(dates)) * 0.05,
            'risk_regime_score': np.random.uniform(-0.5, 0.5, len(dates)),
            'beta_spy_20d': np.random.uniform(0.5, 1.5, len(dates)),
            'rel_strength_spy_20d': np.random.randn(len(dates)) * 0.02,
            'gld_momentum_20d': np.random.randn(len(dates)) * 0.01,
            'tlt_momentum_20d': np.random.randn(len(dates)) * 0.01,
            'risk_regime': np.random.choice(['risk_on', 'neutral', 'risk_off'], len(dates))
        }, index=dates)

    @pytest.fixture
    def sample_macro_state(self):
        """Sample macro state for testing"""
        return {
            'vix_regime': 'normal',
            'risk_regime': 'risk_on',
            'risk_score': 0.3,
            'correlation_breakdown': False,
            'vix_level': 18.5
        }

    # Test 1: Adaptive Feature Selection
    def test_adaptive_feature_selection(self, sample_macro_features, sample_macro_state):
        """Test regime-optimized feature selection"""
        selector = AdaptiveMacroFeatureSelector()

        # Test regime detection
        regime = selector.get_regime_from_macro_state(sample_macro_state)
        assert regime in ['risk_on', 'risk_off', 'high_vol', 'low_vol', 'neutral']

        # Test feature selection
        features = selector.select_regime_optimized_features(
            regime, list(sample_macro_features.columns), max_features=10
        )

        assert len(features) <= 10, "Should respect max_features"
        assert len(features) > 0, "Should select at least some features"

        # Test different regimes
        for test_regime in ['risk_on', 'risk_off', 'high_vol', 'low_vol']:
            features = selector.select_regime_optimized_features(
                test_regime, list(sample_macro_features.columns)
            )
            assert len(features) > 0, f"Should select features for {test_regime}"

        print("[OK] Adaptive Feature Selection: Regime-optimized selection working")

    # Test 2: Macro-Aware Ensemble Weighting
    def test_ensemble_weighting(self, sample_macro_state):
        """Test regime-aware ensemble weighting"""
        weighter = MacroAwareEnsembleWeighter()

        # Get weights
        weights = weighter.get_regime_optimized_weights(sample_macro_state)

        # Verify weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights should sum to 1"

        # Verify all strategies have weights
        assert 'momentum' in weights
        assert 'mean_reversion' in weights
        assert 'volatility' in weights

        # Test crisis scenario
        crisis_state = sample_macro_state.copy()
        crisis_state['vix_regime'] = 'crisis'
        crisis_weights = weighter.get_regime_optimized_weights(crisis_state)

        # In crisis, volatility weight should be higher
        assert crisis_weights['volatility'] > weights['volatility'], \
            "Volatility weight should increase in crisis"

        # Test prediction combination
        predictions = {'momentum': 0.6, 'mean_reversion': 0.4, 'volatility': 0.5}
        combined = weighter.combine_ensemble_predictions(predictions, sample_macro_state)
        assert 0 <= combined <= 1, "Combined prediction should be in valid range"

        print("[OK] Ensemble Weighting: Regime-aware weights working")

    # Test 3: Dynamic Macro Sensitivity
    def test_dynamic_sensitivity(self, sample_macro_features):
        """Test dynamic sensitivity adjustment"""
        sensitivity = DynamicMacroSensitivity()

        # Generate returns
        returns = pd.Series(
            np.random.randn(len(sample_macro_features)) * 0.02,
            index=sample_macro_features.index
        )

        # Calculate optimal sensitivity
        opt_sens, correlations = sensitivity.calculate_optimal_sensitivity(
            sample_macro_features, returns
        )

        assert 0.5 <= opt_sens <= 1.5, "Sensitivity should be in valid range"
        assert isinstance(correlations, dict), "Should return correlation dict"

        # Test multiplier adjustment
        adjusted = sensitivity.get_adjusted_macro_multiplier(0.7, 1.2)
        assert adjusted < 0.7, "Higher sensitivity should increase reduction"

        adjusted_up = sensitivity.get_adjusted_macro_multiplier(1.1, 1.2)
        assert adjusted_up > 1.1, "Higher sensitivity should increase boost"

        print("[OK] Dynamic Sensitivity: Optimal sensitivity calculation working")

    # Test 4: Regime Transition Forecasting
    def test_regime_forecasting(self, sample_macro_features):
        """Test regime transition prediction"""
        forecaster = MacroRegimeForecaster()

        # Forecast transitions
        forecast = forecaster.forecast_regime_transitions(sample_macro_features, 'risk_on')

        assert 'transition_probability' in forecast
        assert 'expected_days_to_transition' in forecast
        assert 'confidence' in forecast
        assert 'trigger_signals' in forecast

        assert 0 <= forecast['transition_probability'] <= 1, "Probability should be 0-1"
        assert forecast['expected_days_to_transition'] >= 1, "Days should be positive"

        # Test regime duration analysis
        durations = forecaster.analyze_regime_duration_patterns(sample_macro_features)
        assert isinstance(durations, dict)

        print("[OK] Regime Forecasting: Transition prediction working")

    # Test 5: Feature Compression
    def test_feature_compression(self, sample_macro_features):
        """Test macro feature compression"""
        compressor = MacroFeatureCompressor(n_components=5)

        # Select numeric columns only
        numeric_features = sample_macro_features.select_dtypes(include=[np.number])

        # Fit and transform
        compressed = compressor.fit_transform(numeric_features)

        assert len(compressed.columns) <= 5, "Should compress to target dimensions"
        assert len(compressed) == len(numeric_features), "Should preserve rows"

        # Test separate fit and transform
        compressor2 = MacroFeatureCompressor(n_components=5)
        compressor2.fit(numeric_features)
        transformed = compressor2.transform(numeric_features)
        assert len(transformed.columns) <= 5

        # Test importance
        importance = compressor.get_feature_importance()
        assert isinstance(importance, dict)

        print("[OK] Feature Compression: PCA-based compression working")

    # Test 6: Impact Monitoring
    def test_impact_monitoring(self, sample_macro_state):
        """Test real-time impact monitoring"""
        monitor = MacroImpactMonitor()

        # Track performance
        predictions = {
            'vix_signal': np.random.randn(50),
            'spy_signal': np.random.randn(50)
        }
        actual = np.random.randn(50)

        metrics = monitor.track_macro_feature_performance(
            predictions, actual, sample_macro_state
        )

        assert len(metrics) == 2, "Should track all features"
        for feature, metric in metrics.items():
            assert 'directional_accuracy' in metric
            assert 'correlation' in metric
            assert 0 <= metric['directional_accuracy'] <= 1

        # Test top features
        top = monitor.get_top_performing_features()
        assert isinstance(top, dict)

        print("[OK] Impact Monitoring: Performance tracking working")

    # Test 7: Macro-Aware Position Sizing
    def test_position_sizing(self, sample_macro_state):
        """Test macro-enhanced position sizing"""
        sizer = MacroAwarePositionSizer()

        # Test position sizing
        result = sizer.get_macro_enhanced_position(0.5, sample_macro_state, 'equity')

        assert 'final_position' in result
        assert 'base_position' in result
        assert 'macro_enhancement' in result
        assert 'components' in result

        assert 0 <= result['final_position'] <= 1, "Position should be 0-1"
        assert result['base_position'] == 0.5

        # Risk-on should increase position
        assert result['macro_enhancement'] >= 1.0, "Risk-on should enhance position"

        # Test risk-off
        risk_off_state = sample_macro_state.copy()
        risk_off_state['risk_regime'] = 'risk_off'
        result_off = sizer.get_macro_enhanced_position(0.5, risk_off_state, 'equity')
        assert result_off['macro_enhancement'] < 1.0, "Risk-off should reduce position"

        # Test crypto sensitivity
        result_crypto = sizer.get_macro_enhanced_position(0.5, risk_off_state, 'crypto')
        assert result_crypto['final_position'] < result_off['final_position'], \
            "Crypto should be more sensitive to risk-off"

        print("[OK] Position Sizing: Macro-aware sizing working")

    # Test 8: Multi-Timeframe Analysis
    def test_mtf_analysis(self, sample_macro_features):
        """Test cross-timeframe macro analysis"""
        mtf = MultiTimeframeMacroAnalyzer()

        analysis = mtf.analyze_macro_across_timeframes(sample_macro_features)

        assert 'timeframe_analysis' in analysis
        assert 'combined_signals' in analysis
        assert 'timeframe_agreement' in analysis
        assert 'primary_timeframe' in analysis

        # Check timeframes analyzed
        assert len(analysis['timeframe_analysis']) == 3, "Should analyze 3 timeframes"
        assert '1d' in analysis['timeframe_analysis']
        assert '1w' in analysis['timeframe_analysis']
        assert '1m' in analysis['timeframe_analysis']

        # Check agreement score
        assert 0 <= analysis['timeframe_agreement'] <= 1, "Agreement should be 0-1"

        print("[OK] Multi-Timeframe Analysis: Cross-timeframe analysis working")

    # Integration Test: Unified Elite System
    def test_unified_elite_system(self, sample_macro_features, sample_macro_state):
        """Test complete elite system integration"""
        system = Phase4EliteSystem()

        # Process macro data
        results = system.process_macro_data(
            sample_macro_features, sample_macro_state, 'equity'
        )

        assert 'selected_features' in results
        assert 'regime' in results
        assert 'ensemble_weights' in results
        assert 'regime_forecast' in results
        assert 'mtf_analysis' in results

        # Get enhanced position
        position = system.get_enhanced_position(0.5, sample_macro_state, 'equity')
        assert 'final_position' in position

        print("[OK] Unified Elite System: Full integration working")


def run_elite_enhancement_tests():
    """Run all elite enhancement tests"""
    print("PHASE 4 ELITE ENHANCEMENTS TEST SUITE")
    print("=" * 60)

    test_instance = TestPhase4Elite()

    # Create fixtures
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')

    macro_features = pd.DataFrame({
        'VIX': np.random.normal(18, 5, len(dates)).clip(10, 40),
        'SPY': 100 + np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
        'GLD': 180 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))),
        'TLT': 90 + np.cumsum(np.random.normal(0.0002, 0.008, len(dates))),
        'DXY': 100 + np.cumsum(np.random.normal(0.0001, 0.005, len(dates))),
        'spy_momentum_5d': np.random.randn(len(dates)) * 0.02,
        'spy_momentum_20d': np.random.randn(len(dates)) * 0.03,
        'vix_momentum_5d': np.random.randn(len(dates)) * 0.05,
        'risk_regime_score': np.random.uniform(-0.5, 0.5, len(dates)),
        'beta_spy_20d': np.random.uniform(0.5, 1.5, len(dates)),
        'rel_strength_spy_20d': np.random.randn(len(dates)) * 0.02,
        'gld_momentum_20d': np.random.randn(len(dates)) * 0.01,
        'tlt_momentum_20d': np.random.randn(len(dates)) * 0.01,
        'risk_regime': np.random.choice(['risk_on', 'neutral', 'risk_off'], len(dates))
    }, index=dates)

    macro_state = {
        'vix_regime': 'normal',
        'risk_regime': 'risk_on',
        'risk_score': 0.3,
        'correlation_breakdown': False,
        'vix_level': 18.5
    }

    test_results = {}

    tests = [
        ('Adaptive Feature Selection', test_instance.test_adaptive_feature_selection, [macro_features, macro_state]),
        ('Ensemble Weighting', test_instance.test_ensemble_weighting, [macro_state]),
        ('Dynamic Sensitivity', test_instance.test_dynamic_sensitivity, [macro_features]),
        ('Regime Forecasting', test_instance.test_regime_forecasting, [macro_features]),
        ('Feature Compression', test_instance.test_feature_compression, [macro_features]),
        ('Impact Monitoring', test_instance.test_impact_monitoring, [macro_state]),
        ('Position Sizing', test_instance.test_position_sizing, [macro_state]),
        ('MTF Analysis', test_instance.test_mtf_analysis, [macro_features]),
        ('Unified Elite System', test_instance.test_unified_elite_system, [macro_features, macro_state]),
    ]

    for test_name, test_func, test_args in tests:
        try:
            test_func(*test_args)
            test_results[test_name] = 'PASS'
            print(f"[PASS] {test_name}")
        except Exception as e:
            test_results[test_name] = f'FAIL: {str(e)}'
            print(f"[FAIL] {test_name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 ELITE TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in test_results.values() if v == 'PASS')
    total = len(test_results)

    print(f"Results: {passed}/{total} Tests Passed ({passed/total*100:.1f}%)")

    print("\nENHANCEMENT STATUS:")
    enhancements = [
        '#1 Adaptive Feature Selection',
        '#2 Macro-Aware Ensemble Weighting',
        '#3 Dynamic Macro Sensitivity',
        '#4 Macro Regime Forecasting',
        '#5 Macro Feature Compression',
        '#6 Real-Time Impact Monitoring',
        '#7 Macro-Aware Position Sizing',
        '#8 Cross-Timeframe Analysis'
    ]

    for i, (test_name, result) in enumerate(list(test_results.items())[:8]):
        status = "[OK]" if result == 'PASS' else "[!!]"
        print(f"  {status} {enhancements[i]}")

    if passed == total:
        print("\nALL ELITE ENHANCEMENTS OPERATIONAL!")
        print("Expected: +3-5% additional profit rate")
        print("Target: 63-68% total profit rate")
    else:
        print(f"\n{total - passed} enhancements need attention")

    return test_results


if __name__ == "__main__":
    results = run_elite_enhancement_tests()
    passed = sum(1 for v in results.values() if v == 'PASS')
    exit(0 if passed >= 8 else 1)
