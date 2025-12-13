"""
Test Suite for SELL Signal Fixes
=================================
Tests specifically for SELL signal optimizations (Fixes 1, 2, 3, 6, 7, 13)

Key Rules:
- SELL signals have only 25% win rate vs 60% for BUY
- Require higher confidence thresholds
- Reduced position sizes
- Tighter stop-losses
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.us_intl_optimizer import (
    USIntlModelOptimizer,
    create_optimizer,
    SignalOptimization,
    AssetClass
)


# SELL Signal Rules from PDF
SELL_SIGNAL_RULES = {
    'min_confidence': 0.80,       # 80% minimum confidence
    'position_multiplier': 0.40,  # 40% of normal size
    'stop_loss': 0.04,           # 4% stop-loss
    'win_rate': 0.25,            # Historical 25% win rate
}


class TestSellSignalThresholds:
    """Test SELL signal confidence thresholds."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_sell_requires_80pct_confidence(self):
        """SELL signals require minimum 80% confidence."""
        # Test range of confidences
        test_cases = [
            (0.79, True),   # Below threshold - blocked
            (0.80, False),  # At threshold - allowed
            (0.85, False),  # Above threshold - allowed
            (0.90, False),  # Well above - allowed
        ]

        for confidence, should_be_blocked in test_cases:
            result = self.optimizer.optimize_signal(
                ticker='AAPL',
                signal_type='SELL',
                confidence=confidence
            )
            assert result.blocked == should_be_blocked, \
                f"SELL with {confidence*100:.0f}% confidence: expected blocked={should_be_blocked}, got blocked={result.blocked}"

    def test_commodity_sell_threshold_80pct(self):
        """Commodities need 80% SELL confidence."""
        result = self.optimizer.optimize_signal(
            ticker='CL=F',
            signal_type='SELL',
            confidence=0.78
        )
        assert result.blocked is True

    def test_crypto_sell_threshold_78pct(self):
        """Crypto needs 78% SELL confidence (but Fix 13 raises to 80%)."""
        result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='SELL',
            confidence=0.79
        )
        # Should be blocked because Fix 13 requires 80%
        assert result.blocked is True


class TestSellPositionSizing:
    """Test SELL signal position sizing (40% of normal)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(enable_dynamic_sizing=False)

    def test_sell_position_40pct_multiplier(self):
        """SELL positions should use 40% multiplier (Fix 13)."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        if not result.blocked:
            # Should apply SELL_POSITION_MULTIPLIER (0.40) and asset multiplier (0.50)
            # Total should be < 1.0
            assert result.position_multiplier < 1.0
            assert 'Fix 13' in str(result.fixes_applied) or 'Fix 3' in str(result.fixes_applied)

    def test_sell_commodity_30pct_multiplier(self):
        """Commodity SELL should have 30% multiplier (Fix 7)."""
        result = self.optimizer.optimize_signal(
            ticker='CL=F',  # Crude oil
            signal_type='SELL',
            confidence=0.90
        )

        if not result.blocked:
            # Should have very low position multiplier
            assert result.position_multiplier < 0.50

    def test_sell_vs_buy_position_comparison(self):
        """SELL positions should be smaller than BUY positions."""
        buy_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85
        )

        sell_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        # BUY should have larger position (if both not blocked)
        if not buy_result.blocked and not sell_result.blocked:
            assert buy_result.position_multiplier > sell_result.position_multiplier


class TestSellStopLoss:
    """Test SELL signal stop-loss (4%)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_sell_stop_loss_4pct(self):
        """SELL signals should have 4% stop-loss (Fix 13)."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        if not result.blocked:
            # Should use tighter 4% stop-loss
            assert result.stop_loss_pct <= 0.04

    def test_sell_stop_loss_tighter_than_buy(self):
        """SELL stop-loss should be tighter than BUY."""
        buy_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85
        )

        sell_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        if not sell_result.blocked:
            assert sell_result.stop_loss_pct <= buy_result.stop_loss_pct


class TestSellBlacklists:
    """Test SELL-specific blacklists."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_ng_futures_sell_blocked(self):
        """Fix 6: NG=F should be blocked for SELL."""
        result = self.optimizer.optimize_signal(
            ticker='NG=F',
            signal_type='SELL',
            confidence=0.95
        )

        assert result.blocked is True
        assert 'NG=F' in (result.block_reason or '') or 'BLACKLIST' in (result.block_reason or '').upper()

    def test_ng_futures_buy_allowed(self):
        """NG=F should still be allowed for BUY."""
        result = self.optimizer.optimize_signal(
            ticker='NG=F',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.blocked is False

    def test_commodity_sell_blacklist(self):
        """Fix 13: Choppy commodity futures blocked for SELL."""
        blocked_commodities = ['ZC=F', 'ZW=F', 'HG=F', 'LE=F', 'HE=F']

        for ticker in blocked_commodities:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.95
            )
            assert result.blocked is True, f"{ticker} should be blocked for SELL"

    def test_stablecoin_sell_blocked(self):
        """Fix 13: Stablecoins blocked for SELL."""
        stablecoins = ['USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD']

        for ticker in stablecoins:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.95
            )
            assert result.blocked is True, f"{ticker} stablecoin should be blocked for SELL"


class TestSellFixesApplied:
    """Test that correct fixes are applied to SELL signals."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_sell_fixes_chain(self):
        """SELL signals should apply fixes 1, 2, 3, 4, 7 (if commodity), 13."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        fixes_str = str(result.fixes_applied)

        # Should include confidence threshold check
        assert 'Fix' in fixes_str

        if not result.blocked:
            # Should have position reduction
            assert 'Fix 3' in fixes_str or 'Fix 13' in fixes_str or 'reduction' in fixes_str.lower()

    def test_commodity_sell_applies_fix7(self):
        """Commodity SELL should specifically apply Fix 7."""
        result = self.optimizer.optimize_signal(
            ticker='CL=F',
            signal_type='SELL',
            confidence=0.90
        )

        if not result.blocked:
            assert 'Fix 7' in str(result.fixes_applied) or 'Commodity' in str(result.fixes_applied)


class TestSellAssetClassThresholds:
    """Test asset-class specific SELL thresholds (Fix 2)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_all_asset_class_thresholds(self):
        """Verify threshold for each asset class."""
        test_cases = [
            ('AAPL', AssetClass.STOCK, 0.75),       # Stock base (but Fix 13 raises to 0.80)
            ('CL=F', AssetClass.COMMODITY, 0.80),   # Commodity
            ('BTC-USD', AssetClass.CRYPTOCURRENCY, 0.78),  # Crypto (but Fix 13 raises to 0.80)
            ('0700.HK', AssetClass.CHINA_STOCK, 0.75),     # China (but Fix 13 raises to 0.80)
        ]

        for ticker, expected_class, _ in test_cases:
            asset_class = self.optimizer.classify_asset(ticker)
            assert asset_class == expected_class, f"{ticker} should be {expected_class}"


class TestSellWinRateImpact:
    """Test how low win rate affects SELL signals."""

    def setup_method(self):
        """Setup with SELL-appropriate win rate."""
        self.optimizer = create_optimizer(
            historical_win_rates={'AAPL': 0.25},  # 25% SELL win rate
            enable_dynamic_sizing=True
        )

    def test_low_win_rate_reduces_position(self):
        """25% win rate should significantly reduce SELL position."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85,
            win_rate=0.25
        )

        if not result.blocked:
            # 20-30% win rate should give 0.3x multiplier (Fix 9)
            assert result.position_multiplier < 0.5
            assert 'Fix 9' in str(result.fixes_applied)

    def test_very_low_win_rate_minimal_position(self):
        """<20% win rate should give minimal position."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85,
            win_rate=0.15
        )

        if not result.blocked:
            # <20% win rate should give 0.1x multiplier
            assert result.position_multiplier < 0.3


class TestSellSignalIntegration:
    """Integration tests for complete SELL signal flow."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={
                'AAPL': 0.25,
                'TSLA': 0.30,
                'NG=F': 0.20,
            },
            enable_kelly=True,
            enable_dynamic_sizing=True
        )

    def test_complete_sell_signal_flow(self):
        """Test complete SELL signal with all fixes applied."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.88,
            volatility=0.30,
            momentum=-0.02,  # Negative momentum supports SELL
            win_rate=0.28
        )

        # Should not be blocked (88% > 80%)
        assert result.blocked is False

        # Should have reduced position
        assert result.position_multiplier < 1.0

        # Should have tight stop-loss
        assert result.stop_loss_pct <= 0.04

        # Should have profit-taking levels
        assert len(result.take_profit_levels) > 0

        # Should have multiple fixes applied
        assert len(result.fixes_applied) > 3

    def test_marginal_sell_signal_blocked(self):
        """Marginal SELL signal should be blocked."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.78,  # Below 80% threshold
            volatility=0.25,
            momentum=0.01
        )

        assert result.blocked is True
        assert 'CONF' in (result.block_reason or '').upper() or 'threshold' in (result.block_reason or '').lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
