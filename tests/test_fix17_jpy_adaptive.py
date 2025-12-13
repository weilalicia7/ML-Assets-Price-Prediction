"""
Test Suite for Fix 17: JPY SELL Adaptive Blocking
==================================================

Tests for adaptive blocking of JPY SELL signals based on market regime.

Key Insight: JPY SELL has 0% historical win rate in BULL/RISK_ON regimes
because JPY weakens in risk-on environments (SELL loses).

IMPORTANT: This is ADAPTIVE blocking - positions are REDUCED, not fully blocked.
- Favorable regimes: Full position (100%)
- Unfavorable regimes: Heavily reduced (30%)

Author: Claude Code
Last Updated: 2025-12-03
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_management.adaptive_blocker import (
    AdaptiveBlocker,
    BlockingLevel,
    BlockingResult,
)
from src.risk_management.market_regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeDetection,
    RegimeIndicators,
    create_indicators_from_data,
)


class TestJPYPairClassification(unittest.TestCase):
    """Test JPY pair identification."""

    def setUp(self):
        """Set up test fixtures."""
        self.jpy_tickers = [
            'USDJPY=X',
            'EURJPY=X',
            'GBPJPY=X',
            'AUDJPY=X',
            'CHFJPY=X',
            'CADJPY=X',
        ]

        self.non_jpy_tickers = [
            'EURUSD=X',
            'GBPUSD=X',
            'AUDUSD=X',
            'USDCHF=X',
            'AAPL',
            'CL=F',
        ]

    def test_jpy_pair_identification(self):
        """Test identification of JPY currency pairs."""
        def is_jpy_pair(ticker: str) -> bool:
            """Check if ticker is a JPY pair."""
            return 'JPY' in ticker.upper()

        for ticker in self.jpy_tickers:
            with self.subTest(ticker=ticker):
                self.assertTrue(
                    is_jpy_pair(ticker),
                    f"{ticker} should be identified as JPY pair"
                )

        for ticker in self.non_jpy_tickers:
            with self.subTest(ticker=ticker):
                self.assertFalse(
                    is_jpy_pair(ticker),
                    f"{ticker} should NOT be identified as JPY pair"
                )


class TestJPYSELLRegimeCompatibility(unittest.TestCase):
    """Test JPY SELL signal compatibility with market regimes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_jpy_sell_favorable_in_bear(self):
        """Test JPY SELL is favorable in BEAR regime.

        JPY strengthens (gains value) in risk-off/bear markets.
        SELL JPY wins when JPY strengthens.
        """
        indicators = create_indicators_from_data(
            vix=30.0,
            spy_return_20d=-0.10,
            spy_return_60d=-0.15,
            spy_above_200ma=False,
            gold_return_20d=0.05,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be favorable (full or near-full position)
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "JPY SELL should have >= 80% position in BEAR regime"
        )
        self.assertFalse(result.blocked, "JPY SELL should NOT be blocked in BEAR")

    def test_jpy_sell_favorable_in_risk_off(self):
        """Test JPY SELL is favorable in RISK_OFF regime."""
        indicators = create_indicators_from_data(
            vix=25.0,
            spy_return_20d=-0.03,
            gold_return_20d=0.04,
            high_yield_spread=6.0,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be favorable
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "JPY SELL should have >= 80% position in RISK_OFF regime"
        )
        self.assertFalse(result.blocked)

    def test_jpy_sell_favorable_in_crisis(self):
        """Test JPY SELL is favorable in CRISIS regime.

        JPY is a safe haven - strengthens during crisis.
        """
        indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            spy_return_60d=-0.20,
            spy_above_200ma=False,
            high_yield_spread=9.0,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be favorable even in crisis (JPY safe haven)
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "JPY SELL should have >= 80% position in CRISIS (JPY safe haven)"
        )
        self.assertFalse(result.blocked)

    def test_jpy_sell_reduced_in_bull(self):
        """Test JPY SELL is reduced in BULL regime.

        ADAPTIVE: Not blocked, but heavily reduced.
        Historical 0% win rate in BULL, but we reduce instead of block.
        """
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_return_60d=0.15,
            spy_above_200ma=True,
            gold_return_20d=-0.02,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be heavily reduced (NOT blocked)
        self.assertLess(
            result.position_reduction, 0.5,
            "JPY SELL should be reduced to < 50% in BULL regime"
        )
        self.assertFalse(result.blocked, "JPY SELL should NOT be blocked (adaptive)")

    def test_jpy_sell_reduced_in_risk_on(self):
        """Test JPY SELL is reduced in RISK_ON regime."""
        indicators = create_indicators_from_data(
            vix=16.0,
            spy_return_20d=0.05,
            spy_above_200ma=True,
            gold_return_20d=-0.03,
            high_yield_spread=3.5,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be heavily reduced (NOT blocked)
        self.assertLess(
            result.position_reduction, 0.5,
            "JPY SELL should be reduced to < 50% in RISK_ON regime"
        )
        self.assertFalse(result.blocked, "JPY SELL should NOT be blocked (adaptive)")


class TestJPYBUYRegimeCompatibility(unittest.TestCase):
    """Test JPY BUY signal compatibility with market regimes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_jpy_buy_favorable_in_bull(self):
        """Test JPY BUY is favorable in BULL regime.

        JPY weakens in risk-on environments.
        BUY JPY is a contrarian bet that may work in strong bull.
        """
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_return_60d=0.15,
            spy_above_200ma=True,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='BUY',
            asset_type='jpy_pair',
            detection=detection,
        )

        # JPY BUY favorable in BULL
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "JPY BUY should have >= 80% position in BULL regime"
        )
        self.assertFalse(result.blocked)

    def test_jpy_buy_reduced_in_crisis(self):
        """Test JPY BUY is reduced in CRISIS regime.

        Don't fight the safe haven flow.
        """
        indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            spy_return_60d=-0.20,
            spy_above_200ma=False,
            high_yield_spread=9.0,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='BUY',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should be reduced in crisis (JPY strengthening)
        self.assertLess(
            result.position_reduction, 0.5,
            "JPY BUY should be reduced to < 50% in CRISIS (JPY safe haven)"
        )
        self.assertFalse(result.blocked, "JPY BUY should NOT be blocked (adaptive)")


class TestJPYPositionReduction(unittest.TestCase):
    """Test position reduction logic for JPY pairs."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_position_reduction_range(self):
        """Test position reduction is within valid range."""
        # Test various regimes
        test_regimes = [
            create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True),  # Bull
            create_indicators_from_data(vix=30, spy_return_20d=-0.10, spy_above_200ma=False),  # Bear
            create_indicators_from_data(vix=17, spy_return_20d=0.01, spy_return_60d=0.02),  # Sideways
            create_indicators_from_data(vix=45, spy_return_20d=-0.15, high_yield_spread=9.0),  # Crisis
        ]

        for indicators in test_regimes:
            detection = self.detector.detect_regime(indicators)

            result = self.blocker.evaluate_signal(
                ticker='USDJPY=X',
                signal_type='SELL',
                asset_type='jpy_pair',
                detection=detection,
            )

            with self.subTest(regime=detection.primary_regime):
                # Position reduction should always be > 0 (adaptive blocking)
                self.assertGreater(
                    result.position_reduction, 0.0,
                    f"Position reduction should be > 0 in {detection.primary_regime}"
                )
                # Position reduction should be <= 1.0
                self.assertLessEqual(
                    result.position_reduction, 1.0,
                    f"Position reduction should be <= 1.0 in {detection.primary_regime}"
                )

    def test_never_fully_blocked(self):
        """Test JPY signals are never fully blocked (adaptive behavior)."""
        # Even in most unfavorable regime
        indicators = create_indicators_from_data(
            vix=14,
            spy_return_20d=0.10,
            spy_return_60d=0.20,
            spy_above_200ma=True,
            gold_return_20d=-0.05,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        # Should NEVER be blocked (adaptive)
        self.assertFalse(
            result.blocked,
            "JPY SELL should NEVER be blocked (adaptive blocking only reduces position)"
        )
        # Position should still be > 0
        self.assertGreater(
            result.position_reduction, 0.0,
            "Position reduction should be > 0 even in unfavorable regime"
        )


class TestJPYBlockingResultDetails(unittest.TestCase):
    """Test blocking result details for JPY pairs."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_blocking_result_contains_regime(self):
        """Test that blocking result contains regime information."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        self.assertIsInstance(result, BlockingResult)
        self.assertIsInstance(result.regime, MarketRegime)
        self.assertIsInstance(result.blocking_level, BlockingLevel)
        self.assertIsNotNone(result.reason)

    def test_blocking_result_confidence(self):
        """Test that blocking result contains regime confidence."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=detection,
        )

        self.assertGreaterEqual(result.regime_confidence, 0.0)
        self.assertLessEqual(result.regime_confidence, 1.0)


class TestJPYBlockingLevels(unittest.TestCase):
    """Test blocking level assignments for JPY signals."""

    def setUp(self):
        """Set up test fixtures."""
        self.blocker = AdaptiveBlocker()

    def test_blocking_level_mapping(self):
        """Test that blocking levels map to correct position reductions."""
        # NO_BLOCK should give 100% position
        self.assertEqual(
            AdaptiveBlocker.POSITION_REDUCTIONS[BlockingLevel.NO_BLOCK],
            1.0
        )

        # WARNING should give 80% position
        self.assertEqual(
            AdaptiveBlocker.POSITION_REDUCTIONS[BlockingLevel.WARNING],
            0.8
        )

        # REDUCE_POSITION should give 50% position
        self.assertEqual(
            AdaptiveBlocker.POSITION_REDUCTIONS[BlockingLevel.REDUCE_POSITION],
            0.5
        )

        # Even FULL_BLOCK gives 10% (adaptive, not permanent)
        self.assertGreater(
            AdaptiveBlocker.POSITION_REDUCTIONS[BlockingLevel.FULL_BLOCK],
            0.0,
            "Even FULL_BLOCK should have > 0 position (adaptive blocking)"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
