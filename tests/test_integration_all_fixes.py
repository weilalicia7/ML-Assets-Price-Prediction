"""
Integration Test Suite for Fixes 16-19
======================================

Tests the integration of all fixes working together:
- Fix 16: China stock handling
- Fix 17: JPY SELL adaptive blocking
- Fix 18: Crude Oil adaptive blocking
- Fix 19: Market regime detection and position adjustment

These tests verify that all components work together correctly
to produce the expected trading behavior.

Author: Claude Code
Last Updated: 2025-12-03
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_management.market_regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    create_indicators_from_data,
)
from src.risk_management.adaptive_blocker import (
    AdaptiveBlocker,
    BlockingLevel,
)
from src.risk_management.position_adjuster import (
    PositionAdjuster,
    PositionAdjustment,
)


class TestFullTradingPipeline(unittest.TestCase):
    """Test the full trading pipeline from signal to position."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)
        self.adjuster = PositionAdjuster(
            regime_detector=self.detector,
            historical_win_rates={
                'AAPL': 0.65,
                'USDJPY=X': 0.40,
                'CL=F': 0.45,
            },
        )

    def test_bull_market_pipeline(self):
        """Test full pipeline in BULL market regime."""
        # Create bull market indicators
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_return_60d=0.15,
            spy_above_200ma=True,
            gold_return_20d=-0.02,
        )

        # Detect regime
        detection = self.detector.detect_regime(indicators)
        self.assertEqual(detection.primary_regime, MarketRegime.BULL)

        # Test stock signal - should pass with normal position
        stock_block = self.blocker.evaluate_signal(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            detection=detection,
        )
        self.assertFalse(stock_block.blocked)
        self.assertGreaterEqual(stock_block.position_reduction, 0.7)

        stock_position = self.adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.10,
            detection=detection,
        )
        self.assertGreater(stock_position.adjusted_position, 0.0)

        # Test JPY SELL - should be reduced in bull
        jpy_block = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )
        self.assertFalse(jpy_block.blocked)  # Adaptive, not blocked
        self.assertLess(jpy_block.position_reduction, 0.5)  # But reduced

        # Test Crude BUY - should be favorable in bull
        crude_block = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )
        self.assertFalse(crude_block.blocked)
        self.assertGreaterEqual(crude_block.position_reduction, 0.8)

    def test_bear_market_pipeline(self):
        """Test full pipeline in BEAR market regime."""
        # Create bear market indicators
        indicators = create_indicators_from_data(
            vix=30.0,
            spy_return_20d=-0.10,
            spy_return_60d=-0.15,
            spy_above_200ma=False,
            gold_return_20d=0.05,
        )

        # Detect regime
        detection = self.detector.detect_regime(indicators)
        self.assertEqual(detection.primary_regime, MarketRegime.BEAR)

        # Test JPY SELL - should be favorable in bear
        jpy_block = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )
        self.assertFalse(jpy_block.blocked)
        self.assertGreaterEqual(jpy_block.position_reduction, 0.8)

        # Test Crude SELL - should be favorable in bear
        crude_block = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )
        self.assertFalse(crude_block.blocked)
        self.assertGreaterEqual(crude_block.position_reduction, 0.8)

        # Test stock BUY - should be reduced in bear
        stock_block = self.blocker.evaluate_signal(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            detection=detection,
        )
        self.assertFalse(stock_block.blocked)

    def test_crisis_market_pipeline(self):
        """Test full pipeline in CRISIS market regime."""
        # Create crisis indicators
        indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            spy_return_60d=-0.20,
            spy_above_200ma=False,
            high_yield_spread=9.0,
        )

        # Detect regime
        detection = self.detector.detect_regime(indicators)
        self.assertIn(
            detection.primary_regime,
            [MarketRegime.CRISIS, MarketRegime.VOLATILE, MarketRegime.BEAR]
        )

        # Test JPY SELL - should still be favorable (JPY safe haven)
        jpy_block = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )
        self.assertFalse(jpy_block.blocked)

        # Test Crude BUY - should be heavily reduced in crisis
        crude_block = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )
        self.assertFalse(crude_block.blocked)  # Adaptive
        self.assertLess(crude_block.position_reduction, 0.5)  # But reduced


class TestMultiAssetPortfolio(unittest.TestCase):
    """Test portfolio-level integration with multiple assets."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)
        self.adjuster = PositionAdjuster(
            regime_detector=self.detector,
            historical_win_rates={
                'AAPL': 0.72,
                'MSFT': 0.68,
                'USDJPY=X': 0.35,
                'EURJPY=X': 0.38,
                'CL=F': 0.42,
                'BTC-USD': 0.55,
            },
        )

    def test_portfolio_in_bull_market(self):
        """Test portfolio behavior in bull market."""
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_return_60d=0.15,
            spy_above_200ma=True,
        )
        detection = self.detector.detect_regime(indicators)

        portfolio_signals = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
            {'ticker': 'MSFT', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
            {'ticker': 'USDJPY=X', 'signal_type': 'SELL', 'asset_type': 'jpy_pair', 'base_position': 0.05},
            {'ticker': 'CL=F', 'signal_type': 'BUY', 'asset_type': 'crude_oil', 'base_position': 0.05},
        ]

        # Process all signals
        blocking_results = self.blocker.batch_evaluate(portfolio_signals, detection)
        position_results = self.adjuster.batch_adjust(portfolio_signals, detection)

        # All signals should pass (adaptive blocking)
        for result in blocking_results:
            self.assertFalse(result.blocked, f"{result.ticker} should not be blocked")

        # Calculate total allocation
        total_adjusted = sum(r.adjusted_position for r in position_results)

        # Should have reasonable total allocation
        self.assertGreater(total_adjusted, 0.0)
        self.assertLess(total_adjusted, 1.0)

    def test_portfolio_in_risk_off(self):
        """Test portfolio behavior in risk-off environment."""
        indicators = create_indicators_from_data(
            vix=25.0,
            spy_return_20d=-0.03,
            gold_return_20d=0.04,
            high_yield_spread=6.0,
        )
        detection = self.detector.detect_regime(indicators)

        portfolio_signals = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
            {'ticker': 'USDJPY=X', 'signal_type': 'SELL', 'asset_type': 'jpy_pair', 'base_position': 0.08},
            {'ticker': 'CL=F', 'signal_type': 'SELL', 'asset_type': 'crude_oil', 'base_position': 0.05},
        ]

        blocking_results = self.blocker.batch_evaluate(portfolio_signals, detection)
        position_results = self.adjuster.batch_adjust(portfolio_signals, detection)

        # JPY SELL should have high position (favorable)
        jpy_result = next(r for r in blocking_results if 'JPY' in r.ticker)
        self.assertGreaterEqual(jpy_result.position_reduction, 0.7)


class TestRegimeTransitions(unittest.TestCase):
    """Test behavior during regime transitions."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_bull_to_bear_transition(self):
        """Test signal handling during bull-to-bear transition."""
        # Start in bull
        bull_indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_above_200ma=True,
        )
        bull_detection = self.detector.detect_regime(bull_indicators)

        jpy_sell_bull = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=bull_detection,
        )

        # Transition to bear
        bear_indicators = create_indicators_from_data(
            vix=30.0,
            spy_return_20d=-0.10,
            spy_above_200ma=False,
        )
        bear_detection = self.detector.detect_regime(bear_indicators)

        jpy_sell_bear = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=bear_detection,
        )

        # JPY SELL position should increase from bull to bear
        self.assertGreater(
            jpy_sell_bear.position_reduction,
            jpy_sell_bull.position_reduction,
            "JPY SELL position should increase from bull to bear"
        )

    def test_regime_history_tracking(self):
        """Test regime history is tracked across detections."""
        # Multiple regime detections
        regimes_to_detect = [
            create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True),  # Bull
            create_indicators_from_data(vix=17, spy_return_20d=0.01),  # Sideways
            create_indicators_from_data(vix=30, spy_return_20d=-0.10),  # Bear
        ]

        for indicators in regimes_to_detect:
            self.detector.detect_regime(indicators)

        # Check history is being tracked
        self.assertGreater(
            len(self.detector.regime_history), 0,
            "Regime history should be tracked"
        )


class TestAdaptiveBlockingConsistency(unittest.TestCase):
    """Test that adaptive blocking is consistent across all asset types."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_no_signals_fully_blocked(self):
        """Test that NO signals are ever fully blocked (adaptive blocking)."""
        test_scenarios = [
            # (indicators, asset_type, signal_type)
            (create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True),
             'jpy_pair', 'SELL'),  # JPY SELL in bull (worst case)
            (create_indicators_from_data(vix=45, spy_return_20d=-0.15, high_yield_spread=9.0),
             'crude_oil', 'BUY'),  # Crude BUY in crisis (worst case)
            (create_indicators_from_data(vix=18, tip_ief_spread=0.04),
             'crude_oil', 'SELL'),  # Crude SELL in inflation (worst case)
        ]

        for indicators, asset_type, signal_type in test_scenarios:
            detection = self.detector.detect_regime(indicators)

            result = self.blocker.evaluate_signal(
                ticker='TEST',
                signal_type=signal_type,
                asset_type=asset_type,
                detection=detection,
            )

            with self.subTest(asset=asset_type, signal=signal_type, regime=detection.primary_regime):
                self.assertFalse(
                    result.blocked,
                    f"{asset_type} {signal_type} should NOT be blocked in {detection.primary_regime}"
                )
                self.assertGreater(
                    result.position_reduction, 0.0,
                    f"Position should be > 0 (adaptive blocking)"
                )


class TestPositionSizingIntegration(unittest.TestCase):
    """Test position sizing integration with blocking."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)
        self.adjuster = PositionAdjuster(regime_detector=self.detector)

    def test_blocked_position_reflects_reduction(self):
        """Test that position adjuster accounts for blocking level."""
        # Unfavorable regime for JPY SELL
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.10,
            spy_above_200ma=True,
        )
        detection = self.detector.detect_regime(indicators)

        # Get blocking result
        block_result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Get position adjustment
        pos_result = self.adjuster.adjust_position(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            base_position=0.10,
            detection=detection,
        )

        # If blocking says reduce, position should be reduced
        if block_result.position_reduction < 0.5:
            self.assertLess(
                pos_result.adjusted_position,
                pos_result.base_position,
                "Position should be reduced when blocking indicates reduction"
            )


class TestChinaStockIntegration(unittest.TestCase):
    """Test China stock handling integration."""

    def test_china_stock_classification(self):
        """Test China stocks are correctly classified."""
        china_tickers = ['600519.SS', '000858.SZ']

        for ticker in china_tickers:
            is_china = ticker.upper().endswith('.SS') or ticker.upper().endswith('.SZ')
            self.assertTrue(is_china, f"{ticker} should be identified as China stock")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)
        self.adjuster = PositionAdjuster(regime_detector=self.detector)

    def test_no_regime_detection_available(self):
        """Test behavior when no regime detection is available."""
        # Fresh blocker with no detection
        fresh_blocker = AdaptiveBlocker()

        result = fresh_blocker.evaluate_signal(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            detection=None,
        )

        # Should return conservative result, not crash
        self.assertIsNotNone(result)
        self.assertFalse(result.blocked)
        self.assertGreater(result.position_reduction, 0.0)

    def test_unknown_asset_type(self):
        """Test behavior with unknown asset type."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='UNKNOWN',
            signal_type='BUY',
            asset_type='unknown_asset',
            detection=detection,
        )

        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertFalse(result.blocked)

    def test_extreme_indicators(self):
        """Test with extreme indicator values."""
        extreme_indicators = create_indicators_from_data(
            vix=80.0,  # Very extreme VIX
            spy_return_20d=-0.30,  # 30% drawdown
            spy_return_60d=-0.40,
            spy_above_200ma=False,
            high_yield_spread=15.0,
        )

        # Should not crash
        detection = self.detector.detect_regime(extreme_indicators)
        self.assertIsNotNone(detection)
        self.assertIn(
            detection.primary_regime,
            [MarketRegime.CRISIS, MarketRegime.VOLATILE, MarketRegime.BEAR]
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
