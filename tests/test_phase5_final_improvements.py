"""
Tests for Phase 5 Final Strategic Improvements

Tests all 7 final improvements:
1. MarketMicrostructureEnhancer
2. TransactionCostOptimizer
3. RegimeTransitionSmoother
4. DynamicCorrelationManager
5. TailRiskAdjustedSizing
6. RealTimeModelUpdater
7. RobustCrossValidator
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ensemble.phase5_final_improvements import (
    MarketMicrostructureEnhancer,
    TransactionCostOptimizer,
    RegimeTransitionSmoother,
    DynamicCorrelationManager,
    TailRiskAdjustedSizing,
    RealTimeModelUpdater,
    RobustCrossValidator,
    StressTestResult,
    Phase5FinalImprovementsSystem,
    create_phase5_final_system,
    PRODUCTION_READINESS_CHECKLIST
)


# =============================================================================
# Test MarketMicrostructureEnhancer
# =============================================================================

class TestMarketMicrostructureEnhancer:
    """Tests for market microstructure adjustments."""

    def setup_method(self):
        self.enhancer = MarketMicrostructureEnhancer()

    def test_initialization(self):
        """Test enhancer initializes with correct defaults."""
        assert self.enhancer.volatility_threshold == 0.25
        assert self.enhancer.high_vol_penalty == 0.8
        assert 'large_cap_equity' in self.enhancer.liquidity_multipliers

    def test_adjust_for_liquidity_large_cap(self):
        """Test liquidity adjustment for large cap equity."""
        adjusted = self.enhancer.adjust_for_liquidity(
            'large_cap_equity', 0.1, 0.15
        )
        # Large cap, low vol -> no adjustment
        assert adjusted == 0.1

    def test_adjust_for_liquidity_small_cap(self):
        """Test liquidity adjustment for small cap equity."""
        adjusted = self.enhancer.adjust_for_liquidity(
            'small_cap_equity', 0.1, 0.15
        )
        # Small cap -> 0.7x multiplier
        assert adjusted == pytest.approx(0.07)

    def test_adjust_for_liquidity_high_volatility(self):
        """Test liquidity adjustment in high volatility."""
        adjusted = self.enhancer.adjust_for_liquidity(
            'large_cap_equity', 0.1, 0.30  # High vol
        )
        # High vol -> 0.8x penalty
        assert adjusted == pytest.approx(0.08)

    def test_adjust_for_liquidity_crypto_minor(self):
        """Test liquidity adjustment for minor crypto."""
        adjusted = self.enhancer.adjust_for_liquidity(
            'crypto_minor', 0.1, 0.15
        )
        # Crypto minor -> 0.4x multiplier
        assert adjusted == pytest.approx(0.04)

    def test_calculate_slippage_small_position(self):
        """Test slippage for small position."""
        slippage = self.enhancer.calculate_slippage_estimate(
            1000, 10_000_000  # 0.01% of volume
        )
        assert slippage == 0.0005  # 5 bps

    def test_calculate_slippage_medium_position(self):
        """Test slippage for medium position."""
        slippage = self.enhancer.calculate_slippage_estimate(
            30_000, 10_000_000  # 0.3% of volume
        )
        assert slippage == 0.001  # 10 bps

    def test_calculate_slippage_large_position(self):
        """Test slippage for large position."""
        slippage = self.enhancer.calculate_slippage_estimate(
            150_000, 10_000_000  # 1.5% of volume
        )
        assert slippage == 0.005  # 50 bps

    def test_calculate_slippage_zero_volume(self):
        """Test slippage with zero volume."""
        slippage = self.enhancer.calculate_slippage_estimate(1000, 0)
        assert slippage == 0.01  # Default 100 bps

    def test_get_optimal_execution_size(self):
        """Test optimal execution size calculation."""
        chunk, num_trades = self.enhancer.get_optimal_execution_size(
            100_000, 10_000_000, max_slippage=0.001
        )
        assert chunk > 0
        assert num_trades >= 1


# =============================================================================
# Test TransactionCostOptimizer
# =============================================================================

class TestTransactionCostOptimizer:
    """Tests for transaction cost optimization."""

    def setup_method(self):
        self.optimizer = TransactionCostOptimizer()

    def test_initialization(self):
        """Test optimizer initializes with cost models."""
        assert 'equity' in self.optimizer.cost_models
        assert 'crypto' in self.optimizer.cost_models
        assert self.optimizer.risk_premium == 0.002

    def test_calculate_breakeven_equity(self):
        """Test breakeven calculation for equity."""
        breakeven = self.optimizer.calculate_breakeven_threshold(
            0.1, 'equity', 10
        )
        # Commission 0.0005*2 + spread 0.0002 + risk premium 0.002
        expected = 0.0012 + 0.002
        assert breakeven == pytest.approx(expected)

    def test_calculate_breakeven_short_term(self):
        """Test breakeven for short-term trade."""
        breakeven = self.optimizer.calculate_breakeven_threshold(
            0.1, 'equity', 3  # Short term
        )
        # Short term -> 1.5x cost multiplier
        expected = (0.0012 * 1.5) + 0.002
        assert breakeven == pytest.approx(expected)

    def test_calculate_breakeven_crypto(self):
        """Test breakeven for crypto."""
        breakeven = self.optimizer.calculate_breakeven_threshold(
            0.1, 'crypto', 10
        )
        # Crypto: commission 0.001*2 + spread 0.0005 + risk premium 0.002
        expected = 0.0025 + 0.002
        assert breakeven == pytest.approx(expected)

    def test_should_execute_trade_profitable(self):
        """Test trade execution decision for profitable trade."""
        should_trade, analysis = self.optimizer.should_execute_trade(
            0.05,  # 5% expected return
            0.1,
            'equity',
            {'confidence': 0.7, 'expected_holding_period': 10}
        )
        assert should_trade is True
        assert analysis['expected_return'] == 0.05
        assert analysis['should_execute'] == 1.0

    def test_should_execute_trade_unprofitable(self):
        """Test trade execution decision for unprofitable trade."""
        should_trade, analysis = self.optimizer.should_execute_trade(
            0.001,  # 0.1% expected return - below breakeven
            0.1,
            'equity',
            {'confidence': 0.5, 'expected_holding_period': 10}
        )
        assert should_trade is False
        assert analysis['should_execute'] == 0.0

    def test_calculate_cost_adjusted_return(self):
        """Test cost-adjusted return calculation."""
        net_return = self.optimizer.calculate_cost_adjusted_return(
            0.05, 'equity', 1
        )
        # 5% - costs
        expected = 0.05 - 0.0012
        assert net_return == pytest.approx(expected)


# =============================================================================
# Test RegimeTransitionSmoother
# =============================================================================

class TestRegimeTransitionSmoother:
    """Tests for regime transition smoothing."""

    def setup_method(self):
        self.smoother = RegimeTransitionSmoother(
            transition_threshold=0.7,
            history_size=10
        )

    def test_initialization(self):
        """Test smoother initializes correctly."""
        assert self.smoother.transition_threshold == 0.7
        assert self.smoother.history_size == 10
        assert len(self.smoother.regime_history) == 0

    def test_get_smoothed_regime_single(self):
        """Test smoothed regime with single observation."""
        regime = self.smoother.get_smoothed_regime('bull', 0.8)
        assert regime == 'bull'

    def test_get_smoothed_regime_consensus(self):
        """Test regime transition with consensus."""
        # Add consistent bull signals
        for _ in range(8):
            self.smoother.get_smoothed_regime('bull', 0.8)

        regime = self.smoother.get_smoothed_regime('bull', 0.8)
        assert regime == 'bull'
        assert self.smoother.current_confirmed_regime == 'bull'

    def test_get_smoothed_regime_no_consensus(self):
        """Test regime stays when no consensus."""
        # Establish bull regime first
        for _ in range(5):
            self.smoother.get_smoothed_regime('bull', 0.8)

        # Try to switch to bear without enough consensus
        self.smoother.get_smoothed_regime('bear', 0.6)
        regime = self.smoother.get_smoothed_regime('bear', 0.6)

        # Should still be bull (not enough bear consensus)
        assert regime == 'bull'

    def test_calculate_regime_momentum_strong(self):
        """Test regime momentum when consistent."""
        for _ in range(5):
            self.smoother.get_smoothed_regime('bull', 0.8)

        momentum = self.smoother.calculate_regime_momentum()
        assert momentum == 1.0

    def test_calculate_regime_momentum_weak(self):
        """Test regime momentum when mixed."""
        self.smoother.get_smoothed_regime('bull', 0.8)
        self.smoother.get_smoothed_regime('bear', 0.8)
        self.smoother.get_smoothed_regime('neutral', 0.8)

        momentum = self.smoother.calculate_regime_momentum()
        assert momentum == 0.0

    def test_get_transition_probability(self):
        """Test transition probability calculation."""
        # Build history with transitions
        self.smoother.get_smoothed_regime('bull', 0.8)
        self.smoother.get_smoothed_regime('bull', 0.8)
        self.smoother.get_smoothed_regime('bear', 0.8)
        self.smoother.get_smoothed_regime('bull', 0.8)

        prob = self.smoother.get_transition_probability('bull', 'bear')
        # 1 bull->bear transition out of 2 bull states
        assert prob == pytest.approx(0.5)

    def test_reset(self):
        """Test smoother reset."""
        self.smoother.get_smoothed_regime('bull', 0.8)
        self.smoother.reset()

        assert len(self.smoother.regime_history) == 0
        assert self.smoother.current_confirmed_regime is None


# =============================================================================
# Test DynamicCorrelationManager
# =============================================================================

class TestDynamicCorrelationManager:
    """Tests for dynamic correlation management."""

    def setup_method(self):
        self.manager = DynamicCorrelationManager()

    def test_initialization(self):
        """Test manager initializes with regime correlations."""
        assert 'low_vol' in self.manager.regime_correlations
        assert 'high_vol' in self.manager.regime_correlations
        assert 'crisis' in self.manager.regime_correlations

    def test_get_regime_aware_correlations_low_vol(self):
        """Test correlations in low volatility regime."""
        pairs = [('equity', 'bonds'), ('equity', 'gold')]
        corrs = self.manager.get_regime_aware_correlations('low_vol', pairs)

        assert ('equity', 'bonds') in corrs
        assert corrs[('equity', 'bonds')] == -0.3
        assert corrs[('equity', 'gold')] == -0.2

    def test_get_regime_aware_correlations_crisis(self):
        """Test correlations in crisis regime."""
        pairs = [('equity', 'bonds'), ('crypto', 'equity')]
        corrs = self.manager.get_regime_aware_correlations('crisis', pairs)

        # Crisis: correlations increase
        assert corrs[('equity', 'bonds')] == 0.5
        assert corrs[('crypto', 'equity')] == 0.8

    def test_detect_correlation_regime_stable(self):
        """Test correlation regime detection - stable."""
        corrs = pd.Series([0.3, 0.31, 0.29, 0.30, 0.32, 0.28])
        regime = self.manager.detect_correlation_regime(corrs)
        assert regime == 'correlation_stable'

    def test_detect_correlation_regime_breakdown(self):
        """Test correlation regime detection - breakdown."""
        corrs = pd.Series([0.3, 0.5, 0.1, 0.7, -0.2, 0.8])  # High volatility
        regime = self.manager.detect_correlation_regime(corrs)
        assert regime == 'correlation_breakdown'

    def test_update_correlation_history(self):
        """Test correlation history update."""
        corr_matrix = np.array([[1, 0.5], [0.5, 1]])
        self.manager.update_correlation_history(corr_matrix)

        assert len(self.manager.correlation_history) == 1
        assert self.manager.correlation_history[0]['avg_correlation'] == 0.5

    def test_get_correlation_trend(self):
        """Test correlation trend calculation."""
        # Add increasing correlations
        for i in range(5):
            corr_matrix = np.array([[1, 0.3 + i * 0.1], [0.3 + i * 0.1, 1]])
            self.manager.update_correlation_history(corr_matrix)

        trend = self.manager.get_correlation_trend()
        assert trend > 0  # Increasing correlations


# =============================================================================
# Test TailRiskAdjustedSizing
# =============================================================================

class TestTailRiskAdjustedSizing:
    """Tests for tail risk adjusted position sizing."""

    def setup_method(self):
        self.sizer = TailRiskAdjustedSizing()
        np.random.seed(42)

    def test_initialization(self):
        """Test sizer initializes with correct confidence levels."""
        assert self.sizer.var_confidence == 0.95
        assert self.sizer.expected_shortfall_confidence == 0.975

    def test_calculate_var(self):
        """Test VaR calculation."""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        var = self.sizer.calculate_var(returns, 0.95)
        # 5th percentile should be near -0.05
        assert var < 0

    def test_calculate_expected_shortfall(self):
        """Test Expected Shortfall (CVaR) calculation."""
        returns = np.array([-0.10, -0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.05])
        es = self.sizer.calculate_expected_shortfall(returns, 0.95)
        var = self.sizer.calculate_var(returns, 0.95)

        # ES should be <= VaR (more negative)
        assert es <= var

    def test_detect_fat_tails_normal(self):
        """Test fat tail detection - normal distribution."""
        returns = np.random.normal(0, 0.02, 1000)
        tail_type = self.sizer.detect_fat_tails(returns)
        assert tail_type == 'normal_tails'

    def test_detect_fat_tails_fat(self):
        """Test fat tail detection - fat tails."""
        # Create fat-tailed distribution (t-distribution with low df)
        returns = np.random.standard_t(df=3, size=1000) * 0.02
        tail_type = self.sizer.detect_fat_tails(returns)
        assert tail_type in ['fat_tails', 'very_fat_tails']

    def test_calculate_var_adjusted_position_normal(self):
        """Test position adjustment for normal returns."""
        returns = np.random.normal(0, 0.02, 100)
        adjusted = self.sizer.calculate_var_adjusted_position(0.1, returns)

        # Normal tails -> no significant adjustment
        assert adjusted >= 0.08  # Within 20% of original

    def test_calculate_var_adjusted_position_fat_tails(self):
        """Test position adjustment for fat-tailed returns."""
        # Create fat-tailed returns
        returns = np.random.standard_t(df=2, size=100) * 0.02
        adjusted = self.sizer.calculate_var_adjusted_position(0.1, returns)

        # Fat tails -> reduced position
        assert adjusted <= 0.1

    def test_get_tail_risk_multiplier(self):
        """Test tail risk multiplier."""
        # Normal returns
        normal_returns = np.random.normal(0, 0.02, 100)
        mult_normal = self.sizer.get_tail_risk_multiplier(normal_returns)
        assert mult_normal == 1.0

        # Fat-tailed returns
        fat_returns = np.random.standard_t(df=2, size=100) * 0.02
        mult_fat = self.sizer.get_tail_risk_multiplier(fat_returns)
        assert mult_fat < 1.0


# =============================================================================
# Test RealTimeModelUpdater
# =============================================================================

class TestRealTimeModelUpdater:
    """Tests for real-time model updating."""

    def setup_method(self):
        self.updater = RealTimeModelUpdater()

    def test_initialization(self):
        """Test updater initializes with correct parameters."""
        assert self.updater.performance_window == 252
        assert self.updater.update_frequency == 21
        assert self.updater.degradation_threshold == 0.1

    def test_calculate_sharpe(self):
        """Test Sharpe ratio calculation."""
        np.random.seed(42)
        # Use returns with some variance around positive mean
        returns = list(np.random.normal(0.01, 0.02, 252))
        sharpe = self.updater.calculate_sharpe(returns)
        # Should be positive (positive mean returns)
        assert sharpe > 0

    def test_calculate_sharpe_negative(self):
        """Test Sharpe ratio for negative returns."""
        np.random.seed(42)
        # Use returns with some variance around negative mean
        returns = list(np.random.normal(-0.01, 0.02, 252))
        sharpe = self.updater.calculate_sharpe(returns)
        assert sharpe < 0

    def test_should_update_model_insufficient_data(self):
        """Test update decision with insufficient data."""
        returns = [0.01] * 100  # Not enough data
        should_update = self.updater.should_update_model(returns)
        assert should_update is False

    def test_should_update_model_no_degradation(self):
        """Test update decision when no degradation."""
        # Consistent performance
        returns = [0.01] * 300
        should_update = self.updater.should_update_model(returns)
        assert should_update is False

    def test_calculate_adaptive_learning_rate_high_vol(self):
        """Test learning rate in high volatility."""
        # Create data with std > 0.25 threshold
        high_vol_data = pd.Series([0.3, -0.3, 0.35, -0.35, 0.4, -0.4] * 20)
        lr = self.updater.calculate_adaptive_learning_rate(high_vol_data)
        assert lr == 0.01  # Slow learning

    def test_calculate_adaptive_learning_rate_low_vol(self):
        """Test learning rate in low volatility."""
        low_vol_data = pd.Series(np.random.normal(0, 0.05, 100))
        lr = self.updater.calculate_adaptive_learning_rate(low_vol_data)
        assert lr == 0.05  # Fast learning

    def test_record_performance(self):
        """Test performance recording."""
        for i in range(10):
            self.updater.record_performance(0.01)

        assert len(self.updater.performance_history) == 10

    def test_get_performance_summary(self):
        """Test performance summary."""
        np.random.seed(42)
        for _ in range(50):
            self.updater.record_performance(np.random.normal(0.001, 0.02))

        summary = self.updater.get_performance_summary()
        assert 'sharpe_ratio' in summary
        assert 'total_return' in summary
        assert 'volatility' in summary
        assert 'max_drawdown' in summary


# =============================================================================
# Test RobustCrossValidator
# =============================================================================

class TestRobustCrossValidator:
    """Tests for cross-validation framework."""

    def setup_method(self):
        self.validator = RobustCrossValidator()

    def test_initialization(self):
        """Test validator initializes with stress scenarios."""
        assert 'flash_crash' in self.validator.stress_scenarios
        assert 'liquidity_crisis' in self.validator.stress_scenarios
        assert 'slow_bleed' in self.validator.stress_scenarios

    def test_cross_validate_improvements(self):
        """Test cross-validation of improvements."""
        periods = {
            'bull_market': pd.DataFrame({'returns': np.random.normal(0.001, 0.01, 252)}),
            'bear_market': pd.DataFrame({'returns': np.random.normal(-0.001, 0.02, 252)})
        }

        results = self.validator.cross_validate_improvements(periods)

        assert 'bull_market' in results
        assert 'bear_market' in results
        assert 'improvements' in results['bull_market']
        assert 'overall_impact' in results['bull_market']

    def test_stress_test_extreme_conditions(self):
        """Test stress testing."""
        base_perf = {'sharpe_ratio': 1.5, 'volatility': 0.15}
        results = self.validator.stress_test_extreme_conditions(base_perf)

        assert 'flash_crash' in results
        assert isinstance(results['flash_crash'], StressTestResult)
        assert results['flash_crash'].scenario_name == 'flash_crash'

    def test_get_validation_summary_empty(self):
        """Test validation summary when not validated."""
        summary = self.validator.get_validation_summary()
        assert summary['status'] == 'not_validated'

    def test_get_validation_summary_after_validation(self):
        """Test validation summary after validation."""
        periods = {
            'period1': pd.DataFrame({'returns': np.random.normal(0, 0.02, 100)})
        }
        self.validator.cross_validate_improvements(periods)

        summary = self.validator.get_validation_summary()
        assert summary['status'] == 'validated'
        assert summary['num_periods'] == 1


# =============================================================================
# Test Phase5FinalImprovementsSystem
# =============================================================================

class TestPhase5FinalImprovementsSystem:
    """Tests for integrated final improvements system."""

    def setup_method(self):
        self.system = create_phase5_final_system(enable_all=True)
        np.random.seed(42)

    def test_initialization(self):
        """Test system initializes all components."""
        assert self.system.microstructure is not None
        assert self.system.cost_optimizer is not None
        assert self.system.regime_smoother is not None
        assert self.system.correlation_manager is not None
        assert self.system.tail_risk_sizer is not None
        assert self.system.model_updater is not None
        assert self.system.cross_validator is not None

    def test_process_signal_basic(self):
        """Test basic signal processing."""
        raw_signal = {
            'direction': 1,
            'confidence': 0.7,
            'position_size': 0.1,
            'expected_return': 0.05
        }
        market_context = {
            'regime': 'bull',
            'regime_confidence': 0.8,
            'asset_class': 'equity',
            'volatility': 0.15
        }

        enhanced = self.system.process_signal(raw_signal, market_context)

        assert 'adjustments_applied' in enhanced
        assert 'smoothed_regime' in enhanced
        assert 'passes_cost_filter' in enhanced

    def test_process_signal_with_tail_risk(self):
        """Test signal processing with tail risk adjustment."""
        raw_signal = {
            'direction': 1,
            'confidence': 0.7,
            'position_size': 0.1,
            'expected_return': 0.05
        }
        market_context = {
            'regime': 'bull',
            'regime_confidence': 0.8,
            'asset_class': 'equity',
            'volatility': 0.15
        }
        returns = np.random.normal(0, 0.02, 100)

        enhanced = self.system.process_signal(
            raw_signal, market_context, returns
        )

        assert 'tail_risk_multiplier' in enhanced
        assert 'tail_type' in enhanced

    def test_process_signal_cost_filter(self):
        """Test signal filtered by cost analysis."""
        raw_signal = {
            'direction': 1,
            'confidence': 0.4,  # Low confidence
            'position_size': 0.1,
            'expected_return': 0.001  # Low return - below breakeven
        }
        market_context = {
            'regime': 'neutral',
            'asset_class': 'equity',
            'volatility': 0.15
        }

        enhanced = self.system.process_signal(raw_signal, market_context)

        assert enhanced['passes_cost_filter'] is False
        assert enhanced['position_size'] == 0.0

    def test_get_system_status(self):
        """Test system status retrieval."""
        # Process some signals
        for _ in range(5):
            self.system.process_signal(
                {'position_size': 0.1, 'expected_return': 0.05, 'confidence': 0.7},
                {'regime': 'bull', 'asset_class': 'equity', 'volatility': 0.15}
            )

        status = self.system.get_system_status()

        assert status['total_signals_processed'] == 5
        assert 'components_enabled' in status
        assert status['components_enabled']['microstructure'] is True

    def test_run_stress_tests(self):
        """Test stress test execution."""
        results = self.system.run_stress_tests({'sharpe_ratio': 1.5})

        assert 'flash_crash' in results
        assert 'liquidity_crisis' in results

    def test_reset(self):
        """Test system reset."""
        self.system.process_signal(
            {'position_size': 0.1, 'expected_return': 0.05},
            {'regime': 'bull', 'asset_class': 'equity', 'volatility': 0.15}
        )

        self.system.reset()

        assert self.system.total_signals_processed == 0
        assert self.system.trades_filtered_by_cost == 0


class TestCreatePhase5FinalSystem:
    """Tests for factory function."""

    def test_create_with_all_enabled(self):
        """Test creating system with all components."""
        system = create_phase5_final_system(enable_all=True)

        assert system.enable_microstructure is True
        assert system.enable_transaction_costs is True
        assert system.enable_regime_smoothing is True

    def test_create_with_selective_components(self):
        """Test creating system with selective components."""
        system = create_phase5_final_system(
            enable_all=False,
            enable_microstructure=True,
            enable_tail_risk=True
        )

        assert system.enable_microstructure is True
        assert system.enable_tail_risk is True
        assert system.enable_transaction_costs is False


class TestProductionReadinessChecklist:
    """Tests for production readiness checklist."""

    def test_checklist_structure(self):
        """Test checklist has all categories."""
        assert 'risk_management' in PRODUCTION_READINESS_CHECKLIST
        assert 'monitoring' in PRODUCTION_READINESS_CHECKLIST
        assert 'operational' in PRODUCTION_READINESS_CHECKLIST

    def test_checklist_items(self):
        """Test checklist has required items."""
        risk_items = PRODUCTION_READINESS_CHECKLIST['risk_management']
        assert 'Circuit breakers implemented' in risk_items
        assert 'Tail risk protection enabled' in risk_items


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_full_trading_cycle(self):
        """Test complete trading cycle through all improvements."""
        system = create_phase5_final_system()
        np.random.seed(42)

        # Simulate 20 trading days
        results = []
        for day in range(20):
            signal = {
                'direction': 1 if np.random.random() > 0.5 else -1,
                'confidence': np.random.uniform(0.5, 0.9),
                'position_size': np.random.uniform(0.05, 0.15),
                'expected_return': np.random.uniform(0.01, 0.05)
            }
            context = {
                'regime': np.random.choice(['bull', 'bear', 'neutral']),
                'regime_confidence': np.random.uniform(0.6, 0.9),
                'asset_class': 'equity',
                'volatility': np.random.uniform(0.10, 0.25),
                'daily_return': np.random.normal(0.001, 0.02)
            }
            returns = np.random.normal(0, 0.02, 100)

            enhanced = system.process_signal(signal, context, returns)
            results.append(enhanced)

        # Check all signals processed
        assert len(results) == 20
        assert system.total_signals_processed == 20

    def test_crisis_scenario(self):
        """Test system behavior during crisis."""
        system = create_phase5_final_system()
        np.random.seed(42)

        # Fat-tailed crisis returns
        crisis_returns = np.random.standard_t(df=2, size=100) * 0.05

        signal = {
            'direction': -1,
            'confidence': 0.6,
            'position_size': 0.1,
            'expected_return': 0.03
        }
        context = {
            'regime': 'crisis',
            'regime_confidence': 0.9,
            'asset_class': 'equity',
            'volatility': 0.35  # High vol
        }

        enhanced = system.process_signal(signal, context, crisis_returns)

        # Position should be reduced
        assert enhanced['position_size'] < signal['position_size']
        assert enhanced['tail_type'] in ['fat_tails', 'very_fat_tails']

    def test_low_confidence_signal_filtered(self):
        """Test low confidence signals are filtered by cost."""
        system = create_phase5_final_system()

        signal = {
            'direction': 1,
            'confidence': 0.4,
            'position_size': 0.1,
            'expected_return': 0.002  # Very low expected return
        }
        context = {
            'regime': 'neutral',
            'asset_class': 'equity',
            'volatility': 0.15
        }

        enhanced = system.process_signal(signal, context)

        assert enhanced['passes_cost_filter'] is False
        assert system.trades_filtered_by_cost == 1


# =============================================================================
# Quick Validation Function
# =============================================================================

def run_quick_validation():
    """Run quick validation of all Phase 5 Final Improvements components."""
    print("=" * 60)
    print("Phase 5 Final Improvements - Quick Validation")
    print("=" * 60)

    # Test 1: Market Microstructure
    print("\n1. Testing Market Microstructure...")
    enhancer = MarketMicrostructureEnhancer()
    adjusted = enhancer.adjust_for_liquidity('small_cap_equity', 0.1, 0.30)
    print(f"   Small cap + high vol adjustment: {adjusted:.3f}")
    assert adjusted < 0.1
    print("   [OK]")

    # Test 2: Transaction Cost Optimizer
    print("\n2. Testing Transaction Cost Optimizer...")
    optimizer = TransactionCostOptimizer()
    should_trade, analysis = optimizer.should_execute_trade(
        0.05, 0.1, 'equity', {'confidence': 0.7}
    )
    print(f"   Should execute 5% expected return: {should_trade}")
    assert should_trade is True
    print("   [OK]")

    # Test 3: Regime Transition Smoother
    print("\n3. Testing Regime Transition Smoother...")
    smoother = RegimeTransitionSmoother()
    for _ in range(8):
        smoother.get_smoothed_regime('bull', 0.8)
    momentum = smoother.calculate_regime_momentum()
    print(f"   Regime momentum (8 bull): {momentum:.2f}")
    assert momentum == 1.0
    print("   [OK]")

    # Test 4: Dynamic Correlation Manager
    print("\n4. Testing Dynamic Correlation Manager...")
    manager = DynamicCorrelationManager()
    corrs = manager.get_regime_aware_correlations(
        'crisis', [('equity', 'bonds'), ('crypto', 'equity')]
    )
    print(f"   Crisis equity-bonds correlation: {corrs[('equity', 'bonds')]:.2f}")
    assert corrs[('equity', 'bonds')] > 0  # Positive in crisis
    print("   [OK]")

    # Test 5: Tail Risk Adjusted Sizing
    print("\n5. Testing Tail Risk Adjusted Sizing...")
    sizer = TailRiskAdjustedSizing()
    np.random.seed(42)
    fat_returns = np.random.standard_t(df=2, size=100) * 0.02
    adjusted = sizer.calculate_var_adjusted_position(0.1, fat_returns)
    print(f"   Fat-tail adjusted position: {adjusted:.3f}")
    assert adjusted <= 0.1
    print("   [OK]")

    # Test 6: Real-Time Model Updater
    print("\n6. Testing Real-Time Model Updater...")
    updater = RealTimeModelUpdater()
    for _ in range(50):
        updater.record_performance(np.random.normal(0.001, 0.02))
    summary = updater.get_performance_summary()
    print(f"   Sharpe ratio: {summary['sharpe_ratio']:.2f}")
    print("   [OK]")

    # Test 7: Robust Cross Validator
    print("\n7. Testing Robust Cross Validator...")
    validator = RobustCrossValidator()
    stress_results = validator.stress_test_extreme_conditions({'sharpe_ratio': 1.5})
    print(f"   Flash crash max drawdown: {stress_results['flash_crash'].max_drawdown:.2%}")
    print("   [OK]")

    # Test 8: Integrated System
    print("\n8. Testing Integrated System...")
    system = create_phase5_final_system()
    enhanced = system.process_signal(
        {'position_size': 0.1, 'expected_return': 0.05, 'confidence': 0.7},
        {'regime': 'bull', 'regime_confidence': 0.8, 'asset_class': 'equity', 'volatility': 0.15},
        np.random.normal(0, 0.02, 100)
    )
    print(f"   Adjustments applied: {len(enhanced['adjustments_applied'])}")
    assert len(enhanced['adjustments_applied']) > 0
    print("   [OK]")

    print("\n" + "=" * 60)
    print("Phase 5 Final Improvements Validation PASSED")
    print("=" * 60)

    return True


if __name__ == '__main__':
    # Run quick validation first
    if run_quick_validation():
        print("\n\nRunning full test suite...\n")
        pytest.main([__file__, '-v', '--tb=short'])
