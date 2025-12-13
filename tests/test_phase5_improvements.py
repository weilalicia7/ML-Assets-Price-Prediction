"""
Test suite for Phase 5 Macro Enhancements.

Tests all 6 improvements from 'phase5 improvement advice on C model.pdf':
1. Macro Context Integration into Dynamic Weighting
2. Macro-Informed Bayesian Priors
3. Regime-Aware Kelly Criterion
4. Advanced Cross-Asset Diversification Penalty
5. Dynamic Timeframe Weights
6. Ensemble Staleness Detection

Run with: python -m pytest tests/test_phase5_improvements.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ensemble.phase5_macro_enhanced import (
    macro_adjusted_composite_score,
    get_macro_informed_prior,
    regime_aware_kelly,
    dynamic_timeframe_weights,
    ensemble_staleness_detection,
    AdvancedDiversificationPenalty
)


class TestPhase5Improvements:
    """Test individual Phase 5 improvement functions."""

    def setup_method(self):
        """Setup test data."""
        self.sample_returns = np.array([0.01, -0.02, 0.015, 0.008, -0.005, 0.02, -0.01])
        self.sample_weights = np.array([0.2, 0.3, 0.15, 0.25, 0.1])

        # Mock macro context data
        self.risk_off_context = {
            'risk_off': True,
            'vix': 35.0,
            'spy_trend': -0.05,
            'market_regime': 'high_vol'
        }

        self.risk_on_context = {
            'risk_off': False,
            'vix': 12.0,
            'spy_trend': 0.08,
            'market_regime': 'low_vol'
        }

        self.normal_context = {
            'risk_off': False,
            'vix': 18.0,
            'spy_trend': 0.02,
            'market_regime': 'normal'
        }

    def test_macro_adjusted_composite_score_risk_off(self):
        """Test composite score adjustment in risk-off regime."""
        base_score = 0.75
        asset_class = 'bond'

        adjusted_score = macro_adjusted_composite_score(
            base_score, asset_class, self.risk_off_context
        )

        # Bonds should get boosted in risk-off
        expected_score = base_score * 1.3
        assert adjusted_score == expected_score
        assert adjusted_score > base_score

    def test_macro_adjusted_composite_score_risk_on(self):
        """Test composite score adjustment in risk-on regime."""
        base_score = 0.75
        asset_class = 'equity'

        adjusted_score = macro_adjusted_composite_score(
            base_score, asset_class, self.risk_on_context
        )

        # Equity should get boosted in risk-on
        expected_score = base_score * 1.3
        assert adjusted_score == expected_score
        assert adjusted_score > base_score

    def test_macro_adjusted_composite_score_crypto_reduction(self):
        """Test crypto reduction in risk-off regimes."""
        base_score = 0.75
        asset_class = 'crypto'

        adjusted_score = macro_adjusted_composite_score(
            base_score, asset_class, self.risk_off_context
        )

        # Crypto should be significantly reduced in risk-off
        expected_score = base_score * 0.5
        assert adjusted_score == expected_score
        assert adjusted_score < base_score

    def test_get_macro_informed_prior_high_fear(self):
        """Test prior parameters in high fear regime."""
        high_fear_context = {'vix': 35.0, 'spy_trend': -0.08}

        prior = get_macro_informed_prior(high_fear_context)

        # Should be more skeptical (higher beta) in high fear
        assert prior['alpha'] == 2.0
        assert prior['beta'] == 3.0
        assert prior['beta'] > prior['alpha']  # More skeptical

    def test_get_macro_informed_prior_bull_market(self):
        """Test prior parameters in bull market."""
        bull_context = {'vix': 12.0, 'spy_trend': 0.10}

        prior = get_macro_informed_prior(bull_context)

        # Should be more optimistic in bull markets
        assert prior['alpha'] == 3.0
        assert prior['beta'] == 2.0
        assert prior['alpha'] > prior['beta']  # More optimistic

    def test_get_macro_informed_prior_normal(self):
        """Test prior parameters in normal conditions."""
        # Normal: VIX <= 30 and spy_trend <= 0 (not bullish)
        normal_context = {'vix': 18.0, 'spy_trend': 0.0}

        prior = get_macro_informed_prior(normal_context)

        # Should be balanced in normal conditions
        assert prior['alpha'] == 2.0
        assert prior['beta'] == 2.0

    def test_regime_aware_kelly_low_vol(self):
        """Test Kelly sizing in low volatility regime."""
        win_rate = 0.6
        win_loss_ratio = 2.0
        regime = 'low_vol'

        position = regime_aware_kelly(
            win_rate, win_loss_ratio, regime, self.risk_on_context
        )

        base_kelly = win_rate - (1 - win_rate) / win_loss_ratio
        expected_quarter = base_kelly * 0.25

        # Position should be boosted in low vol (1.2x multiplier)
        assert position > expected_quarter
        assert position > 0

    def test_regime_aware_kelly_crisis(self):
        """Test Kelly sizing in crisis regime."""
        win_rate = 0.6
        win_loss_ratio = 2.0
        regime = 'crisis'

        position = regime_aware_kelly(
            win_rate, win_loss_ratio, regime, self.risk_off_context
        )

        base_kelly = win_rate - (1 - win_rate) / win_loss_ratio
        expected_quarter = base_kelly * 0.25

        # Position should be significantly reduced in crisis (0.2x multiplier)
        assert position < expected_quarter
        assert position >= 0

    def test_regime_aware_kelly_vix_adjustment(self):
        """Test VIX-based adjustment in Kelly sizing."""
        win_rate = 0.6
        win_loss_ratio = 2.0
        regime = 'normal'

        # Test with high VIX
        high_vix_context = {'vix': 40.0, 'spy_trend': 0.0}
        position_high_vix = regime_aware_kelly(
            win_rate, win_loss_ratio, regime, high_vix_context
        )

        # Test with low VIX
        low_vix_context = {'vix': 12.0, 'spy_trend': 0.0}
        position_low_vix = regime_aware_kelly(
            win_rate, win_loss_ratio, regime, low_vix_context
        )

        # Higher VIX should result in smaller position
        assert position_high_vix < position_low_vix

    def test_advanced_diversification_penalty_same_cluster(self):
        """Test penalty for assets in same correlation cluster."""
        div_penalty = AdvancedDiversificationPenalty()

        portfolio = {'AAPL': 0.1, 'MSFT': 0.15, 'GOOGL': 0.05}
        new_ticker = 'NVDA'  # Also in Technology cluster

        penalty = div_penalty.calculate_penalty(
            new_ticker, portfolio, self.normal_context
        )

        # Should have significant penalty for same cluster
        assert penalty > 0.05
        assert penalty <= 0.6  # Should respect cap

    def test_advanced_diversification_penalty_different_clusters(self):
        """Test penalty for assets in different clusters."""
        div_penalty = AdvancedDiversificationPenalty()

        portfolio = {'AAPL': 0.1, 'GLD': 0.1, 'TLT': 0.1}
        new_ticker = 'XOM'  # Energy cluster

        penalty = div_penalty.calculate_penalty(
            new_ticker, portfolio, self.normal_context
        )

        # Should have lower penalty for diversified portfolio
        assert penalty < 0.4

    def test_advanced_diversification_penalty_empty_portfolio(self):
        """Test diversification penalty with empty portfolio."""
        div_penalty = AdvancedDiversificationPenalty()

        penalty = div_penalty.calculate_penalty(
            'AAPL', {}, self.normal_context
        )

        # No portfolio should mean no penalty
        assert penalty == 0.0

    def test_advanced_diversification_penalty_crisis_amplification(self):
        """Test penalty amplification during crisis."""
        div_penalty = AdvancedDiversificationPenalty()

        portfolio = {'AAPL': 0.1, 'MSFT': 0.1}
        new_ticker = 'GOOGL'

        crisis_context = {'vix': 45.0}  # Crisis levels
        normal_context = {'vix': 18.0}  # Normal levels

        penalty_crisis = div_penalty.calculate_penalty(
            new_ticker, portfolio, crisis_context
        )

        penalty_normal = div_penalty.calculate_penalty(
            new_ticker, portfolio, normal_context
        )

        # Should apply crisis amplification
        assert penalty_crisis > penalty_normal

    def test_dynamic_timeframe_weights_high_volatility(self):
        """Test timeframe weight adjustment in high volatility."""
        volatility_regime = 'high'
        macro_trend = 0.0
        recent_performance = {'short_term': 0.02, 'long_term': 0.03}

        weights = dynamic_timeframe_weights(
            volatility_regime, macro_trend, recent_performance
        )

        # In high volatility, should reduce short-term weights
        assert weights['1h'] < 0.15  # Reduced from base 0.15
        assert weights['1w'] > 0.25  # Increased from base 0.25

        # Should still sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_dynamic_timeframe_weights_low_volatility(self):
        """Test timeframe weight adjustment in low volatility."""
        volatility_regime = 'low'
        macro_trend = 0.05
        recent_performance = {'short_term': 0.04, 'long_term': 0.02}

        weights = dynamic_timeframe_weights(
            volatility_regime, macro_trend, recent_performance
        )

        # In low volatility with short-term outperforming, should boost short-term weights
        assert weights['1h'] > 0.15  # Increased from base 0.15

        # Should still sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_dynamic_timeframe_weights_performance_momentum(self):
        """Test timeframe weight adjustment based on recent performance."""
        volatility_regime = 'normal'
        macro_trend = 0.0

        # Short-term outperforming
        recent_performance_good_short = {'short_term': 0.05, 'long_term': 0.02}
        weights_good_short = dynamic_timeframe_weights(
            volatility_regime, macro_trend, recent_performance_good_short
        )

        # Long-term outperforming
        recent_performance_good_long = {'short_term': 0.01, 'long_term': 0.04}
        weights_good_long = dynamic_timeframe_weights(
            volatility_regime, macro_trend, recent_performance_good_long
        )

        # Should boost timeframes that are performing well
        assert weights_good_short['1h'] > weights_good_long['1h']

    def test_ensemble_staleness_detection_significant_degradation(self):
        """Test staleness detection with significant performance degradation."""
        # Simulate performance degradation
        historical_perf = list(np.array([0.01, 0.02, 0.015, 0.008, 0.012] * 10))  # Good performance
        recent_perf = list(np.array([-0.02, -0.015, -0.01, 0.005, -0.008] * 4))  # Poor performance
        ensemble_performance = historical_perf + recent_perf

        result = ensemble_staleness_detection(ensemble_performance, lookback_days=20)

        assert result['needs_refresh'] == True
        assert result['degradation_ratio'] < 0.7
        assert result['recommended_action'] == 'recalculate_weights'

    def test_ensemble_staleness_detection_stable_performance(self):
        """Test staleness detection with stable performance."""
        # Simulate stable performance
        np.random.seed(42)
        stable_perf = list(np.random.normal(0.01, 0.005, 60))  # Consistent returns
        ensemble_performance = stable_perf

        result = ensemble_staleness_detection(ensemble_performance, lookback_days=21)

        assert result['needs_refresh'] == False
        assert result['degradation_ratio'] >= 0.7
        assert result['recommended_action'] == 'maintain_weights'

    def test_ensemble_staleness_detection_improving_performance(self):
        """Test staleness detection with improving performance."""
        # Simulate improving performance
        np.random.seed(42)
        early_perf = list(np.random.normal(0.005, 0.008, 30))  # Lower returns
        later_perf = list(np.random.normal(0.015, 0.006, 30))  # Higher returns
        ensemble_performance = early_perf + later_perf

        result = ensemble_staleness_detection(ensemble_performance, lookback_days=15)

        # Should not trigger refresh when performance is improving
        assert result['needs_refresh'] == False
        assert result['degradation_ratio'] > 1.0  # Improvement ratio

    def test_ensemble_staleness_detection_insufficient_data(self):
        """Test staleness detection with insufficient data."""
        short_perf = [0.01, 0.02, 0.015]  # Too short

        result = ensemble_staleness_detection(short_perf, lookback_days=21)

        assert result['needs_refresh'] == False
        assert result['recommended_action'] == 'insufficient_data'


class TestIntegrationScenarios:
    """Integration tests simulating real-world scenarios."""

    def setup_method(self):
        """Setup test data."""
        self.normal_context = {
            'risk_off': False,
            'vix': 18.0,
            'spy_trend': 0.02,
            'market_regime': 'normal'
        }

    def test_full_risk_off_scenario(self):
        """Test complete risk-off market scenario."""
        # Simulate crisis conditions
        macro_context = {
            'risk_off': True,
            'vix': 38.0,
            'spy_trend': -0.12,
            'market_regime': 'crisis'
        }

        # Test ensemble weighting
        base_scores = {
            'equity': 0.80,
            'bond': 0.65,
            'crypto': 0.75,
            'commodity': 0.70
        }

        adjusted_scores = {}
        for asset_class, score in base_scores.items():
            adjusted_scores[asset_class] = macro_adjusted_composite_score(
                score, asset_class, macro_context
            )

        # Verify risk-off behavior
        assert adjusted_scores['bond'] > base_scores['bond']  # Bonds boosted
        assert adjusted_scores['equity'] < base_scores['equity']  # Equity reduced
        assert adjusted_scores['crypto'] < base_scores['crypto'] * 0.6  # Crypto heavily reduced

    def test_volatility_regime_transition(self):
        """Test behavior during volatility regime transition."""
        # Transition from low to high volatility
        low_vol_context = {'vix': 12.0, 'market_regime': 'low_vol'}
        high_vol_context = {'vix': 28.0, 'market_regime': 'high_vol'}

        # Test position sizing changes
        win_rate, win_loss_ratio = 0.55, 1.8

        low_vol_position = regime_aware_kelly(
            win_rate, win_loss_ratio, 'low_vol', low_vol_context
        )

        high_vol_position = regime_aware_kelly(
            win_rate, win_loss_ratio, 'high_vol', high_vol_context
        )

        # Should reduce position size in high volatility
        assert high_vol_position < low_vol_position

    def test_correlation_breakdown_scenario(self):
        """Test behavior during correlation breakdown (crisis)."""
        div_penalty = AdvancedDiversificationPenalty()

        # Portfolio with larger weights for higher penalty
        portfolio = {'SPY': 0.25, 'QQQ': 0.25, 'IWM': 0.20}
        new_ticker = 'DIA'
        crisis_context = {'vix': 42.0}

        penalty = div_penalty.calculate_penalty(
            new_ticker, portfolio, crisis_context
        )

        # Crisis amplification: base penalty * 1.5
        # Same cluster penalty = 0.3 * (0.25 + 0.25 + 0.20) = 0.21, * 1.5 = 0.315
        assert penalty >= 0.3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup test data."""
        self.normal_context = {
            'risk_off': False,
            'vix': 18.0,
            'spy_trend': 0.02,
            'market_regime': 'normal'
        }

    def test_extreme_vix_values(self):
        """Test with extreme VIX values."""
        extreme_contexts = [
            {'vix': 5.0, 'spy_trend': 0.10},   # Extremely low VIX
            {'vix': 80.0, 'spy_trend': -0.20},  # Extremely high VIX
        ]

        for context in extreme_contexts:
            prior = get_macro_informed_prior(context)
            # Should handle extremes without crashing
            assert prior['alpha'] > 0
            assert prior['beta'] > 0

    def test_zero_win_rate_kelly(self):
        """Test Kelly with zero win rate."""
        position = regime_aware_kelly(0.0, 2.0, 'normal', self.normal_context)
        # Should handle gracefully (zero or minimal position)
        assert position == 0.0

    def test_very_high_win_rate_kelly(self):
        """Test Kelly with very high win rate."""
        position = regime_aware_kelly(0.95, 3.0, 'normal', self.normal_context)
        # regime_aware_kelly returns raw calculated position (not capped)
        # RegimeAwarePositionSizer class applies min/max caps
        # Raw calculation: kelly = 0.95 - 0.05/3 = 0.933, * 0.25 = 0.233
        # With normal regime (1.0) and VIX adjustment (~0.97) = ~0.226
        assert position > 0.15  # High win rate gives larger raw position
        assert position < 0.30  # But still within reasonable bounds

    def test_negative_win_loss_ratio(self):
        """Test Kelly with edge case win/loss ratio."""
        position = regime_aware_kelly(0.5, 0.5, 'normal', self.normal_context)
        # Should handle gracefully
        assert position >= 0

    def test_timeframe_weights_sum_to_one(self):
        """Test that timeframe weights always sum to 1.0."""
        test_cases = [
            ('high', 0.0, {'short_term': 0.5, 'long_term': 0.5}),
            ('low', 0.05, {'short_term': 0.6, 'long_term': 0.4}),
            ('normal', -0.02, {'short_term': 0.4, 'long_term': 0.6}),
            ('crisis', -0.1, {'short_term': 0.3, 'long_term': 0.7}),
        ]

        for regime, trend, perf in test_cases:
            weights = dynamic_timeframe_weights(regime, trend, perf)
            assert abs(sum(weights.values()) - 1.0) < 0.01, f"Failed for regime={regime}"
            assert all(w > 0 for w in weights.values()), f"Negative weight for regime={regime}"


def run_quick_validation():
    """Run quick validation of Phase 5 improvements."""
    print("=" * 60)
    print("Phase 5 Improvements - Quick Validation")
    print("=" * 60)

    # Test 1: Macro composite scoring
    print("\n1. Testing macro composite scoring...")
    risk_off = {'risk_off': True, 'vix': 35}
    risk_on = {'risk_off': False, 'vix': 15}

    bond_risk_off = macro_adjusted_composite_score(0.5, 'bond', risk_off)
    bond_risk_on = macro_adjusted_composite_score(0.5, 'bond', risk_on)
    print(f"   Bond in risk-off: {bond_risk_off:.3f} (expected > 0.5)")
    print(f"   Bond in risk-on: {bond_risk_on:.3f} (expected < 0.5)")
    assert bond_risk_off > bond_risk_on
    print("   [OK]")

    # Test 2: Macro priors
    print("\n2. Testing macro-informed priors...")
    high_fear = get_macro_informed_prior({'vix': 40, 'spy_trend': -0.1})
    bull = get_macro_informed_prior({'vix': 12, 'spy_trend': 0.1})
    print(f"   High fear prior: alpha={high_fear['alpha']}, beta={high_fear['beta']}")
    print(f"   Bull market prior: alpha={bull['alpha']}, beta={bull['beta']}")
    assert high_fear['beta'] > high_fear['alpha']
    assert bull['alpha'] > bull['beta']
    print("   [OK]")

    # Test 3: Regime-aware Kelly
    print("\n3. Testing regime-aware Kelly...")
    low_vol_pos = regime_aware_kelly(0.6, 1.5, 'low_vol', {'vix': 12})
    crisis_pos = regime_aware_kelly(0.6, 1.5, 'crisis', {'vix': 40})
    print(f"   Low vol position: {low_vol_pos:.4f}")
    print(f"   Crisis position: {crisis_pos:.4f}")
    assert low_vol_pos > crisis_pos
    print("   [OK]")

    # Test 4: Dynamic timeframe weights
    print("\n4. Testing dynamic timeframe weights...")
    high_vol_weights = dynamic_timeframe_weights('high', 0, {'short_term': 0.5, 'long_term': 0.5})
    low_vol_weights = dynamic_timeframe_weights('low', 0, {'short_term': 0.5, 'long_term': 0.5})
    print(f"   High vol 1h weight: {high_vol_weights['1h']:.3f}")
    print(f"   Low vol 1h weight: {low_vol_weights['1h']:.3f}")
    assert high_vol_weights['1h'] < low_vol_weights['1h']
    print("   [OK]")

    # Test 5: Staleness detection
    print("\n5. Testing staleness detection...")
    # Need variability for sharpe calculation (std > 0)
    np.random.seed(42)
    # Stable: consistent positive returns with some variability
    good_perf = list(np.random.normal(0.01, 0.005, 50))  # Mean 1%, std 0.5%
    # Degraded: good historical, bad recent (sharpe drops significantly)
    historical_good = list(np.random.normal(0.015, 0.005, 25))  # High sharpe period
    recent_bad = list(np.random.normal(-0.01, 0.02, 25))  # Low/negative sharpe period
    bad_recent = historical_good + recent_bad

    stable_result = ensemble_staleness_detection(good_perf, 20)
    degraded_result = ensemble_staleness_detection(bad_recent, 20)
    print(f"   Stable: needs_refresh={stable_result['needs_refresh']}, ratio={stable_result['degradation_ratio']:.2f}")
    print(f"   Degraded: needs_refresh={degraded_result['needs_refresh']}, ratio={degraded_result['degradation_ratio']:.2f}")
    assert stable_result['needs_refresh'] == False
    assert degraded_result['needs_refresh'] == True
    print("   [OK]")

    # Test 6: Diversification penalty
    print("\n6. Testing diversification penalty...")
    div_calc = AdvancedDiversificationPenalty()
    same_cluster = div_calc.calculate_penalty('MSFT', {'AAPL': 0.2}, {'vix': 20})
    empty_portfolio = div_calc.calculate_penalty('AAPL', {}, {'vix': 20})
    print(f"   Same cluster penalty: {same_cluster:.3f}")
    print(f"   Empty portfolio penalty: {empty_portfolio:.3f}")
    assert same_cluster > 0
    assert empty_portfolio == 0
    print("   [OK]")

    print("\n" + "=" * 60)
    print("Phase 5 Improvements Validation PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Run quick validation first
    if run_quick_validation():
        print("\n\nRunning full test suite...\n")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("\nQuick validation failed. Please fix issues before running full tests.")
        sys.exit(1)
