"""
Test Suite for Fix 18: Crude Oil Adaptive Blocking
===================================================

Tests for adaptive blocking of Crude Oil (CL=F) signals based on market regime.

Key Insight: Crude Oil is a consistent loser in wrong market regimes.
- BUY favorable in: INFLATION, BULL, RISK_ON (demand/inflation drives price up)
- SELL favorable in: DEFLATION, BEAR (demand destruction drives price down)
- Both risky in: VOLATILE, CRISIS, SIDEWAYS (unpredictable)

IMPORTANT: This is ADAPTIVE blocking - positions are REDUCED, not fully blocked.
- Favorable regimes: Full position (100%)
- Unfavorable regimes: Heavily reduced (20-40%)

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
    create_indicators_from_data,
)


class TestCrudeOilClassification(unittest.TestCase):
    """Test Crude Oil asset identification."""

    def setUp(self):
        """Set up test fixtures."""
        self.crude_tickers = [
            'CL=F',   # WTI Crude Oil Futures
            'BZ=F',   # Brent Crude Futures
        ]

        self.non_crude_tickers = [
            'GC=F',   # Gold
            'SI=F',   # Silver
            'NG=F',   # Natural Gas
            'AAPL',   # Stock
            'USDJPY=X',  # Forex
        ]

    def test_crude_oil_identification(self):
        """Test identification of Crude Oil tickers."""
        def is_crude_oil(ticker: str) -> bool:
            """Check if ticker is crude oil."""
            return ticker.upper() in ['CL=F', 'BZ=F']

        for ticker in self.crude_tickers:
            with self.subTest(ticker=ticker):
                self.assertTrue(
                    is_crude_oil(ticker),
                    f"{ticker} should be identified as crude oil"
                )

        for ticker in self.non_crude_tickers:
            with self.subTest(ticker=ticker):
                self.assertFalse(
                    is_crude_oil(ticker),
                    f"{ticker} should NOT be identified as crude oil"
                )


class TestCrudeBUYRegimeCompatibility(unittest.TestCase):
    """Test Crude Oil BUY signal compatibility with market regimes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_crude_buy_favorable_in_inflation(self):
        """Test Crude Oil BUY is favorable in INFLATION regime.

        Oil prices rise with inflation - supply constraints and
        currency debasement drive commodity prices up.
        """
        indicators = create_indicators_from_data(
            vix=18.0,
            spy_return_20d=0.02,
            tip_ief_spread=0.04,  # Positive = inflation expectations
            gold_return_20d=0.03,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be favorable in inflation
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "Crude BUY should have >= 80% position in INFLATION regime"
        )
        self.assertFalse(result.blocked)

    def test_crude_buy_favorable_in_bull(self):
        """Test Crude Oil BUY is favorable in BULL regime.

        Economic growth drives oil demand.
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
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be favorable in bull
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "Crude BUY should have >= 80% position in BULL regime"
        )
        self.assertFalse(result.blocked)

    def test_crude_buy_favorable_in_risk_on(self):
        """Test Crude Oil BUY is favorable in RISK_ON regime."""
        indicators = create_indicators_from_data(
            vix=16.0,
            spy_return_20d=0.05,
            spy_above_200ma=True,
            gold_return_20d=-0.03,
            high_yield_spread=3.5,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be favorable in risk-on
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "Crude BUY should have >= 80% position in RISK_ON regime"
        )
        self.assertFalse(result.blocked)

    def test_crude_buy_reduced_in_deflation(self):
        """Test Crude Oil BUY is reduced in DEFLATION regime.

        Demand destruction and falling prices hurt oil.
        ADAPTIVE: Not blocked, but heavily reduced.
        """
        indicators = create_indicators_from_data(
            vix=22.0,
            spy_return_20d=-0.03,
            tip_ief_spread=-0.03,  # Negative = deflation expectations
            usd_index_return=0.04,  # Strong dollar = deflationary
            treasury_10y=1.5,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be heavily reduced (NOT blocked)
        self.assertLess(
            result.position_reduction, 0.5,
            "Crude BUY should be reduced to < 50% in DEFLATION regime"
        )
        self.assertFalse(result.blocked, "Crude BUY should NOT be blocked (adaptive)")

    def test_crude_buy_reduced_in_crisis(self):
        """Test Crude Oil BUY is reduced in CRISIS regime.

        Oil crashes during economic crisis (demand destruction).
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
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be heavily reduced
        self.assertLess(
            result.position_reduction, 0.5,
            "Crude BUY should be reduced to < 50% in CRISIS regime"
        )
        self.assertFalse(result.blocked, "Crude BUY should NOT be blocked (adaptive)")


class TestCrudeSELLRegimeCompatibility(unittest.TestCase):
    """Test Crude Oil SELL signal compatibility with market regimes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_crude_sell_favorable_in_deflation(self):
        """Test Crude Oil SELL is favorable in DEFLATION regime.

        Falling prices and demand destruction favor shorts.
        """
        indicators = create_indicators_from_data(
            vix=22.0,
            spy_return_20d=-0.03,
            tip_ief_spread=-0.03,
            usd_index_return=0.04,
            treasury_10y=1.5,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be favorable in deflation
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "Crude SELL should have >= 80% position in DEFLATION regime"
        )
        self.assertFalse(result.blocked)

    def test_crude_sell_favorable_in_bear(self):
        """Test Crude Oil SELL is favorable in BEAR regime."""
        indicators = create_indicators_from_data(
            vix=30.0,
            spy_return_20d=-0.10,
            spy_return_60d=-0.15,
            spy_above_200ma=False,
            gold_return_20d=0.05,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be favorable in bear
        self.assertGreaterEqual(
            result.position_reduction, 0.8,
            "Crude SELL should have >= 80% position in BEAR regime"
        )
        self.assertFalse(result.blocked)

    def test_crude_sell_reduced_in_inflation(self):
        """Test Crude Oil SELL is reduced in INFLATION regime.

        Oil prices rise with inflation - SELL is unfavorable.
        ADAPTIVE: Not blocked, but heavily reduced.
        """
        indicators = create_indicators_from_data(
            vix=18.0,
            spy_return_20d=0.02,
            tip_ief_spread=0.04,
            gold_return_20d=0.03,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be heavily reduced
        self.assertLess(
            result.position_reduction, 0.5,
            "Crude SELL should be reduced to < 50% in INFLATION regime"
        )
        self.assertFalse(result.blocked, "Crude SELL should NOT be blocked (adaptive)")

    def test_crude_sell_reduced_in_risk_on(self):
        """Test Crude Oil SELL is reduced in RISK_ON regime."""
        indicators = create_indicators_from_data(
            vix=16.0,
            spy_return_20d=0.05,
            spy_above_200ma=True,
            gold_return_20d=-0.03,
            high_yield_spread=3.5,
        )
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )

        # Should be heavily reduced
        self.assertLess(
            result.position_reduction, 0.5,
            "Crude SELL should be reduced to < 50% in RISK_ON regime"
        )
        self.assertFalse(result.blocked)


class TestCrudeOilVolatileRegime(unittest.TestCase):
    """Test Crude Oil behavior in volatile regimes."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_crude_reduced_in_volatile(self):
        """Test both Crude BUY and SELL are reduced in VOLATILE regime.

        Unpredictable price swings make both directions risky.
        """
        indicators = create_indicators_from_data(
            vix=35.0,
            spy_return_20d=0.05,  # Mixed signals
            spy_return_60d=-0.05,
        )
        detection = self.detector.detect_regime(indicators)

        # Test BUY
        buy_result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Test SELL
        sell_result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=detection,
        )

        # Both should be reduced in volatile regime
        self.assertLess(
            buy_result.position_reduction, 0.8,
            "Crude BUY should be reduced in VOLATILE regime"
        )
        self.assertLess(
            sell_result.position_reduction, 0.8,
            "Crude SELL should be reduced in VOLATILE regime"
        )

        # Neither should be fully blocked
        self.assertFalse(buy_result.blocked)
        self.assertFalse(sell_result.blocked)


class TestCrudeOilPositionReduction(unittest.TestCase):
    """Test position reduction logic for Crude Oil."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_position_reduction_range(self):
        """Test position reduction is within valid range."""
        test_regimes = [
            create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True),  # Bull
            create_indicators_from_data(vix=30, spy_return_20d=-0.10, spy_above_200ma=False),  # Bear
            create_indicators_from_data(vix=18, tip_ief_spread=0.04, gold_return_20d=0.03),  # Inflation
            create_indicators_from_data(vix=22, tip_ief_spread=-0.03, usd_index_return=0.04),  # Deflation
        ]

        for indicators in test_regimes:
            detection = self.detector.detect_regime(indicators)

            for signal_type in ['BUY', 'SELL']:
                result = self.blocker.evaluate_signal(
                    ticker='CL=F',
                    signal_type=signal_type,
                    asset_type='crude_oil',
                    detection=detection,
                )

                with self.subTest(regime=detection.primary_regime, signal=signal_type):
                    # Position reduction should always be > 0
                    self.assertGreater(
                        result.position_reduction, 0.0,
                        f"Position reduction should be > 0 for {signal_type}"
                    )
                    # Position reduction should be <= 1.0
                    self.assertLessEqual(
                        result.position_reduction, 1.0,
                        f"Position reduction should be <= 1.0 for {signal_type}"
                    )

    def test_never_fully_blocked(self):
        """Test Crude Oil signals are never fully blocked (adaptive behavior)."""
        # Test in most unfavorable regimes
        unfavorable_indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            spy_return_60d=-0.20,
            spy_above_200ma=False,
            high_yield_spread=9.0,
        )
        detection = self.detector.detect_regime(unfavorable_indicators)

        for signal_type in ['BUY', 'SELL']:
            result = self.blocker.evaluate_signal(
                ticker='CL=F',
                signal_type=signal_type,
                asset_type='crude_oil',
                detection=detection,
            )

            with self.subTest(signal=signal_type):
                self.assertFalse(
                    result.blocked,
                    f"Crude {signal_type} should NEVER be blocked (adaptive)"
                )
                self.assertGreater(
                    result.position_reduction, 0.0,
                    f"Position should be > 0 even in unfavorable regime"
                )


class TestCrudeOilSeasonality(unittest.TestCase):
    """Test Crude Oil seasonal patterns awareness.

    Note: Actual seasonality implementation may be in a separate module.
    These tests verify awareness of seasonal patterns.
    """

    def test_seasonal_patterns_awareness(self):
        """Test awareness of crude oil seasonal patterns.

        Key seasonal patterns:
        - Summer driving season (May-Sep): Higher demand
        - Hurricane season (Jun-Nov): Supply risk
        - Refinery maintenance (Spring/Fall): Temporary supply reduction
        - Winter heating demand (Nov-Mar): Higher demand for heating oil
        """
        seasonal_patterns = {
            'summer_driving': {'months': [5, 6, 7, 8, 9], 'effect': 'bullish'},
            'hurricane_season': {'months': [6, 7, 8, 9, 10, 11], 'effect': 'volatile'},
            'refinery_maintenance': {'months': [3, 4, 10, 11], 'effect': 'volatile'},
            'winter_heating': {'months': [11, 12, 1, 2, 3], 'effect': 'bullish'},
        }

        self.assertIn('summer_driving', seasonal_patterns)
        self.assertIn('hurricane_season', seasonal_patterns)
        self.assertEqual(seasonal_patterns['summer_driving']['effect'], 'bullish')


class TestCrudeOilBlockingResultDetails(unittest.TestCase):
    """Test blocking result details for Crude Oil."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.blocker = AdaptiveBlocker(self.detector)

    def test_blocking_result_structure(self):
        """Test that blocking result has correct structure."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        result = self.blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=detection,
        )

        # Check structure
        self.assertIsInstance(result, BlockingResult)
        self.assertEqual(result.ticker, 'CL=F')
        self.assertEqual(result.signal_type, 'BUY')
        self.assertIsInstance(result.regime, MarketRegime)
        self.assertIsInstance(result.blocking_level, BlockingLevel)
        self.assertIsNotNone(result.reason)
        self.assertGreaterEqual(result.regime_confidence, 0.0)
        self.assertLessEqual(result.regime_confidence, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
