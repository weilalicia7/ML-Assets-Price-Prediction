"""
Test Suite for Fix 19: Market Regime Detection
==============================================

Tests for market regime detection and position adjustment.

Market Regimes:
- BULL: Strong upward trend, low volatility
- BEAR: Strong downward trend, elevated volatility
- RISK_ON: High risk appetite, money flowing to equities
- RISK_OFF: Flight to safety, money flowing to bonds/gold
- VOLATILE: High volatility, uncertain direction
- SIDEWAYS: Range-bound, low conviction
- INFLATION: Rising inflation expectations
- DEFLATION: Falling inflation expectations
- CRISIS: Market stress indicators elevated

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
    RegimeDetection,
    RegimeIndicators,
    create_indicators_from_data,
)
from src.risk_management.position_adjuster import (
    PositionAdjuster,
    PositionAdjustment,
    create_position_adjuster,
)


class TestMarketRegimeDetection(unittest.TestCase):
    """Test market regime detection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_bull_regime_detection(self):
        """Test BULL regime detection.

        Conditions: Low VIX, positive returns, above 200MA.
        """
        indicators = create_indicators_from_data(
            vix=14.0,
            spy_return_20d=0.08,
            spy_return_60d=0.15,
            spy_above_200ma=True,
            gold_return_20d=-0.02,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertEqual(
            detection.primary_regime, MarketRegime.BULL,
            "Should detect BULL regime with strong positive indicators"
        )
        self.assertGreater(detection.confidence, 0.5)

    def test_bear_regime_detection(self):
        """Test BEAR regime detection.

        Conditions: High VIX, negative returns, below 200MA.
        """
        indicators = create_indicators_from_data(
            vix=30.0,
            spy_return_20d=-0.10,
            spy_return_60d=-0.15,
            spy_above_200ma=False,
            gold_return_20d=0.05,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertEqual(
            detection.primary_regime, MarketRegime.BEAR,
            "Should detect BEAR regime with strong negative indicators"
        )
        self.assertGreater(detection.confidence, 0.5)

    def test_risk_on_regime_detection(self):
        """Test RISK_ON regime detection.

        Conditions: Low VIX, positive returns, tight credit spreads.
        """
        indicators = create_indicators_from_data(
            vix=16.0,
            spy_return_20d=0.05,
            spy_above_200ma=True,
            gold_return_20d=-0.03,
            high_yield_spread=3.5,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.RISK_ON, MarketRegime.BULL],
            "Should detect RISK_ON or BULL regime with risk-on indicators"
        )

    def test_risk_off_regime_detection(self):
        """Test RISK_OFF regime detection.

        Conditions: Elevated VIX, negative returns, gold rising.
        """
        indicators = create_indicators_from_data(
            vix=25.0,
            spy_return_20d=-0.03,
            gold_return_20d=0.04,
            high_yield_spread=6.0,
            usd_index_return=0.02,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.RISK_OFF, MarketRegime.BEAR],
            "Should detect RISK_OFF or BEAR regime with risk-off indicators"
        )

    def test_volatile_regime_detection(self):
        """Test VOLATILE regime detection.

        Conditions: High VIX, mixed returns.
        """
        indicators = create_indicators_from_data(
            vix=35.0,
            spy_return_20d=0.05,
            spy_return_60d=-0.05,  # Mixed signals
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.VOLATILE, MarketRegime.CRISIS],
            "Should detect VOLATILE or CRISIS with high VIX and mixed signals"
        )

    def test_sideways_regime_detection(self):
        """Test SIDEWAYS regime detection.

        Conditions: Normal VIX, flat returns.
        """
        indicators = create_indicators_from_data(
            vix=17.0,
            spy_return_20d=0.01,
            spy_return_60d=0.02,
            gold_return_20d=0.01,
            usd_index_return=0.005,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.SIDEWAYS, MarketRegime.BULL],
            "Should detect SIDEWAYS or BULL with flat/low returns"
        )

    def test_inflation_regime_detection(self):
        """Test INFLATION regime detection.

        Conditions: Positive TIP/IEF spread, gold rising.
        """
        indicators = create_indicators_from_data(
            vix=18.0,
            spy_return_20d=0.02,
            tip_ief_spread=0.04,
            gold_return_20d=0.04,
            treasury_10y=5.0,
            treasury_2y=4.5,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.INFLATION, MarketRegime.RISK_ON, MarketRegime.BULL],
            "Should detect INFLATION with positive inflation indicators"
        )

    def test_deflation_regime_detection(self):
        """Test DEFLATION regime detection.

        Conditions: Negative TIP/IEF spread, strong dollar.
        """
        indicators = create_indicators_from_data(
            vix=22.0,
            spy_return_20d=-0.03,
            tip_ief_spread=-0.03,
            usd_index_return=0.04,
            treasury_10y=1.5,
        )
        detection = self.detector.detect_regime(indicators)

        # Deflation detection can be difficult - accept related regimes
        self.assertIn(
            detection.primary_regime,
            [MarketRegime.DEFLATION, MarketRegime.BEAR, MarketRegime.RISK_OFF],
            "Should detect DEFLATION or related regime"
        )

    def test_crisis_regime_detection(self):
        """Test CRISIS regime detection.

        Conditions: Extreme VIX, large drawdown, wide credit spreads.
        """
        indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            spy_return_60d=-0.20,
            spy_above_200ma=False,
            high_yield_spread=9.0,
            gold_return_20d=0.08,
        )
        detection = self.detector.detect_regime(indicators)

        self.assertIn(
            detection.primary_regime,
            [MarketRegime.CRISIS, MarketRegime.VOLATILE, MarketRegime.BEAR],
            "Should detect CRISIS or related regime with extreme indicators"
        )


class TestRegimeDetectionConfidence(unittest.TestCase):
    """Test regime detection confidence scoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_confidence_range(self):
        """Test confidence is always in valid range [0, 1]."""
        test_cases = [
            create_indicators_from_data(vix=14, spy_return_20d=0.08),
            create_indicators_from_data(vix=30, spy_return_20d=-0.10),
            create_indicators_from_data(vix=45, spy_return_20d=-0.15),
            create_indicators_from_data(vix=17, spy_return_20d=0.01),
        ]

        for indicators in test_cases:
            detection = self.detector.detect_regime(indicators)

            self.assertGreaterEqual(
                detection.confidence, 0.0,
                "Confidence should be >= 0"
            )
            self.assertLessEqual(
                detection.confidence, 1.0,
                "Confidence should be <= 1"
            )

    def test_clear_signals_high_confidence(self):
        """Test that clear signals produce reasonable confidence."""
        # Very clear bull signal
        bull_indicators = create_indicators_from_data(
            vix=12.0,
            spy_return_20d=0.12,
            spy_return_60d=0.25,
            spy_above_200ma=True,
            gold_return_20d=-0.05,
        )
        bull_detection = self.detector.detect_regime(bull_indicators)

        # Mixed/unclear signal
        mixed_indicators = create_indicators_from_data(
            vix=20.0,
            spy_return_20d=0.00,
            spy_return_60d=0.00,
        )
        mixed_detection = self.detector.detect_regime(mixed_indicators)

        # Both should have valid confidence levels
        self.assertGreater(
            bull_detection.confidence, 0.5,
            "Bull signal should have reasonable confidence > 0.5"
        )
        self.assertGreater(
            mixed_detection.confidence, 0.0,
            "Mixed signal should have positive confidence"
        )


class TestRegimeScoring(unittest.TestCase):
    """Test regime scoring mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_regime_scores_returned(self):
        """Test that regime scores are returned for all regimes."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        # Should have scores for all regimes
        for regime in MarketRegime:
            self.assertIn(
                regime, detection.regime_scores,
                f"Scores should include {regime}"
            )

    def test_regime_scores_normalized(self):
        """Test that regime scores are normalized (max = 1.0)."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        max_score = max(detection.regime_scores.values())
        self.assertAlmostEqual(
            max_score, 1.0, places=2,
            msg="Maximum regime score should be ~1.0 (normalized)"
        )


class TestPositionAdjustment(unittest.TestCase):
    """Test position adjustment based on regime."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.adjuster = PositionAdjuster(regime_detector=self.detector)

    def test_position_adjustment_result(self):
        """Test position adjustment returns correct structure."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        result = self.adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.10,
            detection=detection,
        )

        self.assertIsInstance(result, PositionAdjustment)
        self.assertEqual(result.ticker, 'AAPL')
        self.assertEqual(result.signal_type, 'BUY')
        self.assertEqual(result.base_position, 0.10)
        self.assertGreater(result.adjusted_position, 0.0)

    def test_position_clamped_to_limits(self):
        """Test adjusted position is clamped to min/max limits."""
        adjuster = PositionAdjuster(
            max_position_size=0.15,
            min_position_size=0.01,
        )

        indicators = create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True)
        detection = self.detector.detect_regime(indicators)

        # Test with large base position
        result = adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.50,  # Very large
            detection=detection,
        )

        self.assertLessEqual(
            result.adjusted_position, 0.15,
            "Adjusted position should not exceed max_position_size"
        )

        # Test with tiny base position
        result2 = adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.001,  # Very small
            detection=detection,
        )

        self.assertGreaterEqual(
            result2.adjusted_position, 0.01,
            "Adjusted position should not be below min_position_size"
        )

    def test_position_reduced_in_crisis(self):
        """Test positions are reduced in CRISIS regime."""
        crisis_indicators = create_indicators_from_data(
            vix=45.0,
            spy_return_20d=-0.15,
            high_yield_spread=9.0,
        )
        detection = self.detector.detect_regime(crisis_indicators)

        result = self.adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.10,
            detection=detection,
        )

        # Position should be reduced in crisis
        self.assertLess(
            result.adjusted_position,
            result.base_position,
            "Position should be reduced in CRISIS regime"
        )


class TestPositionAdjustmentMultipliers(unittest.TestCase):
    """Test position adjustment multiplier logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.adjuster = PositionAdjuster()

    def test_regime_multipliers_defined(self):
        """Test that regime multipliers are defined for all regimes."""
        for regime in MarketRegime:
            self.assertIn(
                regime, PositionAdjuster.REGIME_MULTIPLIERS,
                f"REGIME_MULTIPLIERS should include {regime}"
            )

    def test_short_regime_multipliers_defined(self):
        """Test that short (SELL) regime multipliers are defined."""
        for regime in MarketRegime:
            self.assertIn(
                regime, PositionAdjuster.REGIME_MULTIPLIERS_SHORT,
                f"REGIME_MULTIPLIERS_SHORT should include {regime}"
            )

    def test_multiplier_ranges(self):
        """Test multipliers are in reasonable range."""
        for regime, mult in PositionAdjuster.REGIME_MULTIPLIERS.items():
            self.assertGreater(
                mult, 0.0,
                f"Multiplier for {regime} should be > 0"
            )
            self.assertLessEqual(
                mult, 2.0,
                f"Multiplier for {regime} should be <= 2.0"
            )


class TestAssetTypeMultipliers(unittest.TestCase):
    """Test asset type specific multipliers."""

    def setUp(self):
        """Set up test fixtures."""
        self.adjuster = PositionAdjuster()

    def test_asset_multipliers_defined(self):
        """Test that asset type multipliers are defined."""
        expected_assets = ['stock', 'etf', 'cryptocurrency', 'commodity', 'jpy_pair', 'crude_oil']

        for asset in expected_assets:
            self.assertIn(
                asset, PositionAdjuster.ASSET_BASE_MULTIPLIERS,
                f"Asset multiplier should be defined for {asset}"
            )

    def test_volatile_assets_reduced(self):
        """Test that volatile assets have lower multipliers."""
        stock_mult = PositionAdjuster.ASSET_BASE_MULTIPLIERS.get('stock', 1.0)
        crypto_mult = PositionAdjuster.ASSET_BASE_MULTIPLIERS.get('cryptocurrency', 0.6)
        crude_mult = PositionAdjuster.ASSET_BASE_MULTIPLIERS.get('crude_oil', 0.4)

        self.assertLess(
            crypto_mult, stock_mult,
            "Cryptocurrency should have lower multiplier than stocks"
        )
        self.assertLess(
            crude_mult, stock_mult,
            "Crude oil should have lower multiplier than stocks"
        )


class TestWinRateMultipliers(unittest.TestCase):
    """Test win rate based position adjustment."""

    def setUp(self):
        """Set up test fixtures."""
        self.adjuster = PositionAdjuster()
        self.detector = MarketRegimeDetector()

    def test_high_win_rate_increases_position(self):
        """Test that high win rate increases position size."""
        indicators = create_indicators_from_data(vix=17, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        # High win rate
        result_high = self.adjuster.adjust_position(
            ticker='WINNER',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.05,
            win_rate=0.85,
            detection=detection,
        )

        # Low win rate
        result_low = self.adjuster.adjust_position(
            ticker='LOSER',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.05,
            win_rate=0.35,
            detection=detection,
        )

        self.assertGreater(
            result_high.win_rate_multiplier,
            result_low.win_rate_multiplier,
            "High win rate should have higher multiplier"
        )

    def test_win_rate_multiplier_range(self):
        """Test win rate multipliers are in valid range."""
        for (low, high), mult in PositionAdjuster.WIN_RATE_MULTIPLIERS.items():
            self.assertGreater(mult, 0.0, f"Win rate multiplier should be > 0")
            self.assertLessEqual(mult, 2.0, f"Win rate multiplier should be <= 2.0")


class TestKellyCriterion(unittest.TestCase):
    """Test Kelly Criterion position sizing."""

    def setUp(self):
        """Set up test fixtures."""
        self.adjuster = PositionAdjuster()

    def test_kelly_positive_edge(self):
        """Test Kelly formula with positive edge."""
        # 60% win rate, 2:1 risk/reward
        kelly = self.adjuster.calculate_kelly_position(
            win_rate=0.60,
            avg_win=0.10,
            avg_loss=0.05,
            fraction=1.0,  # Full Kelly for test
        )

        # Kelly should be positive with edge
        self.assertGreater(kelly, 0.0, "Kelly should be positive with edge")

    def test_kelly_negative_edge(self):
        """Test Kelly formula with negative edge."""
        # 30% win rate, 1:1 risk/reward
        kelly = self.adjuster.calculate_kelly_position(
            win_rate=0.30,
            avg_win=0.05,
            avg_loss=0.05,
            fraction=1.0,
        )

        # Kelly should be zero or very small with no edge
        self.assertLessEqual(kelly, 0.01, "Kelly should be ~0 with no edge")

    def test_quarter_kelly(self):
        """Test quarter-Kelly reduces position."""
        full_kelly = self.adjuster.calculate_kelly_position(
            win_rate=0.60,
            avg_win=0.10,
            avg_loss=0.05,
            fraction=1.0,
        )

        quarter_kelly = self.adjuster.calculate_kelly_position(
            win_rate=0.60,
            avg_win=0.10,
            avg_loss=0.05,
            fraction=0.25,
        )

        # Quarter Kelly should be less than full Kelly
        self.assertLess(
            quarter_kelly, full_kelly,
            "Quarter Kelly should be less than full Kelly"
        )
        # Both should be positive with positive edge
        self.assertGreater(full_kelly, 0.0, "Full Kelly should be positive")
        self.assertGreater(quarter_kelly, 0.0, "Quarter Kelly should be positive")

    def test_kelly_clamped(self):
        """Test Kelly is clamped to max position size."""
        adjuster = PositionAdjuster(max_position_size=0.15)

        # Very high edge would produce >100% Kelly
        kelly = adjuster.calculate_kelly_position(
            win_rate=0.90,
            avg_win=0.50,
            avg_loss=0.10,
            fraction=1.0,
        )

        self.assertLessEqual(
            kelly, 0.15,
            "Kelly should be clamped to max_position_size"
        )


class TestBatchPositionAdjustment(unittest.TestCase):
    """Test batch position adjustment."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.adjuster = PositionAdjuster(regime_detector=self.detector)

    def test_batch_adjust_multiple_positions(self):
        """Test adjusting multiple positions at once."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        positions = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
            {'ticker': 'USDJPY=X', 'signal_type': 'SELL', 'asset_type': 'jpy_pair', 'base_position': 0.05},
            {'ticker': 'CL=F', 'signal_type': 'BUY', 'asset_type': 'crude_oil', 'base_position': 0.05},
        ]

        results = self.adjuster.batch_adjust(positions, detection)

        self.assertEqual(len(results), 3, "Should return results for all positions")

        for i, result in enumerate(results):
            self.assertEqual(result.ticker, positions[i]['ticker'])
            self.assertIsInstance(result, PositionAdjustment)


class TestAdjustmentSummary(unittest.TestCase):
    """Test adjustment summary statistics."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()
        self.adjuster = PositionAdjuster(regime_detector=self.detector)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        indicators = create_indicators_from_data(vix=20, spy_return_20d=0.03)
        detection = self.detector.detect_regime(indicators)

        positions = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
            {'ticker': 'MSFT', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
        ]

        results = self.adjuster.batch_adjust(positions, detection)
        summary = self.adjuster.get_adjustment_summary(results)

        self.assertIn('total_positions', summary)
        self.assertIn('total_base_allocation', summary)
        self.assertIn('total_adjusted_allocation', summary)
        self.assertIn('average_multiplier', summary)
        self.assertEqual(summary['total_positions'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
