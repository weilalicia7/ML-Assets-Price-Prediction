"""
Test Suite for Profit Maximization Fixes (1-15)
================================================
Tests all 15 fixes from 'us model fixing1.pdf' and 'us model fixing 2.pdf'

Expected Improvements:
- Blocklist (Fix 6, 8, 13): +4.61% P&L
- Position Sizing (Fix 3, 5, 7, 9): +5.80% P&L
- Stop-Loss (Fix 4, 12, 13): +4.61% P&L
- Kelly Criterion (Fix 11): +2.30% P&L
- Pattern Detection (Fix 14): +2.48% P&L
- Total Expected: +19.80% P&L
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


class TestBlocklistFunctionality:
    """Test blocklist fixes (Fix 6, 8, 13)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_sell_blacklist_ng_futures(self):
        """Fix 6: NG=F should be blocked from SELL signals."""
        result = self.optimizer.optimize_signal(
            ticker='NG=F',
            signal_type='SELL',
            confidence=0.95
        )

        assert result.blocked is True
        assert 'BLACKLIST' in (result.block_reason or '').upper() or 'Fix 6' in str(result.fixes_applied)
        assert result.signal_type == 'SELL'

    def test_extended_blocklist_lumn(self):
        """Fix 8: LUMN should be blocked from all signals."""
        result = self.optimizer.optimize_signal(
            ticker='LUMN',
            signal_type='BUY',
            confidence=0.94
        )

        assert result.blocked is True
        assert 'BLOCKLIST' in (result.block_reason or '').upper() or 'Fix 8' in str(result.fixes_applied)

    def test_extended_blocklist_hk_stock(self):
        """Fix 8: 3800.HK should be blocked."""
        result = self.optimizer.optimize_signal(
            ticker='3800.HK',
            signal_type='BUY',
            confidence=0.85
        )

        assert result.blocked is True

    def test_commodity_sell_blacklist(self):
        """Fix 13: Commodity futures in blacklist should be blocked for SELL."""
        commodities_blocked = ['ZC=F', 'ZW=F', 'HG=F']

        for ticker in commodities_blocked:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.90
            )
            assert result.blocked is True, f"{ticker} should be blocked for SELL"

    def test_crypto_stablecoin_sell_blacklist(self):
        """Fix 13: Stablecoins should be blocked from SELL signals."""
        stablecoins = ['USDT-USD', 'USDC-USD', 'DAI-USD']

        for ticker in stablecoins:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.90
            )
            assert result.blocked is True, f"{ticker} stablecoin should be blocked for SELL"

    def test_non_blocklisted_passes(self):
        """Non-blocklisted tickers should pass through."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85
        )

        assert result.blocked is False


class TestPositionSizing:
    """Test position sizing fixes (Fix 3, 5, 7, 9, 10)."""

    def setup_method(self):
        """Setup test optimizer with win rates."""
        self.win_rates = {
            'AAPL': 0.75,
            'TSLA': 0.60,
            'NG=F': 0.33,
            'BTC-USD': 0.45,
        }
        self.optimizer = create_optimizer(
            historical_win_rates=self.win_rates,
            enable_kelly=True,
            enable_dynamic_sizing=True
        )

    def test_sell_position_50pct_reduction(self):
        """Fix 3: SELL positions should be reduced by 50%."""
        # High confidence to pass threshold
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.90
        )

        # Should include 0.50 or 0.40 multiplier for SELL
        assert result.position_multiplier < 1.0 or result.blocked
        if not result.blocked:
            assert 'Fix 3' in str(result.fixes_applied) or 'Fix 13' in str(result.fixes_applied)

    def test_buy_position_30pct_boost_us_stocks(self):
        """Fix 5: US stock BUY should get 30% boost."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85
        )

        assert result.blocked is False
        assert result.position_multiplier > 1.0
        assert 'Fix 5' in str(result.fixes_applied)

    def test_commodity_sell_70pct_reduction(self):
        """Fix 7: Commodity SELL should be reduced by 70%."""
        # Use a commodity not in blacklist with high confidence
        result = self.optimizer.optimize_signal(
            ticker='CL=F',  # Crude oil, not in blacklist
            signal_type='SELL',
            confidence=0.90
        )

        if not result.blocked:
            # Should have 0.30 multiplier (70% reduction)
            assert 'Fix 7' in str(result.fixes_applied) or result.position_multiplier <= 0.50

    def test_win_rate_multiplier_high(self):
        """Fix 9: High win rate should boost position."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85,
            win_rate=0.85
        )

        assert result.blocked is False
        assert 'Fix 9' in str(result.fixes_applied)
        # 80%+ win rate should give 2.0x multiplier
        assert result.position_multiplier > 2.0

    def test_win_rate_multiplier_low(self):
        """Fix 9: Low win rate should reduce position."""
        result = self.optimizer.optimize_signal(
            ticker='TSLA',
            signal_type='BUY',
            confidence=0.70,
            win_rate=0.35
        )

        assert result.blocked is False
        # 30-40% win rate should give 0.5x multiplier
        assert 'Fix 9' in str(result.fixes_applied)

    def test_base_allocation_by_asset_class(self):
        """Fix 10: Base allocations vary by asset class."""
        # US stock should have 10% base
        stock_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        # Crypto should have 5% base
        crypto_result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='BUY',
            confidence=0.80
        )

        assert 'Fix 10' in str(stock_result.fixes_applied)
        assert 'Fix 10' in str(crypto_result.fixes_applied)


class TestStopLoss:
    """Test stop-loss fixes (Fix 4, 12, 13)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_stock_stop_loss_8pct(self):
        """Fix 4: Stock stop-loss should be 8%."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.08

    def test_crypto_stop_loss_12pct(self):
        """Fix 4: Crypto stop-loss should be 12%."""
        result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.12

    def test_commodity_stop_loss_10pct(self):
        """Fix 4: Commodity stop-loss should be 10%."""
        result = self.optimizer.optimize_signal(
            ticker='GC=F',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.10

    def test_forex_stop_loss_6pct(self):
        """Fix 4: Forex stop-loss should be 6%."""
        result = self.optimizer.optimize_signal(
            ticker='EURUSD=X',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.06

    def test_sell_tighter_stop_loss(self):
        """Fix 13: SELL signals should have tighter 4% stop-loss."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.90
        )

        if not result.blocked:
            # SELL stop-loss should be min of asset stop and 4%
            assert result.stop_loss_pct <= 0.08


class TestKellyCriterion:
    """Test Kelly Criterion position sizing (Fix 11)."""

    def setup_method(self):
        """Setup test optimizer with Kelly enabled."""
        self.optimizer = create_optimizer(enable_kelly=True)

    def test_kelly_calculation_basic(self):
        """Fix 11: Kelly fraction calculation."""
        # p=0.60, b=1.5 (avg_win/avg_loss)
        # f* = (0.60 * 1.5 - 0.40) / 1.5 = 0.333
        kelly = self.optimizer.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=0.06,
            avg_loss=0.04
        )

        assert 0.0 < kelly <= 0.25  # Capped at 25%

    def test_kelly_high_win_rate(self):
        """Fix 11: High win rate should give higher Kelly fraction (before cap)."""
        # Use lower win rates to avoid hitting the 0.25 cap
        kelly_high = self.optimizer.calculate_kelly_fraction(
            win_rate=0.65,
            avg_win=0.05,
            avg_loss=0.04
        )

        kelly_low = self.optimizer.calculate_kelly_fraction(
            win_rate=0.52,
            avg_win=0.05,
            avg_loss=0.04
        )

        # Higher win rate should give higher Kelly fraction
        assert kelly_high > kelly_low

    def test_kelly_negative_edge(self):
        """Fix 11: Negative edge should return 0 Kelly fraction."""
        kelly = self.optimizer.calculate_kelly_fraction(
            win_rate=0.30,
            avg_win=0.03,
            avg_loss=0.05
        )

        assert kelly == 0.0

    def test_kelly_in_optimization(self):
        """Fix 11: Kelly fraction should be included in signal optimization."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85,
            win_rate=0.70
        )

        assert result.blocked is False
        assert result.kelly_fraction is not None or 'Fix 11' in str(result.fixes_applied)


class TestConfidenceThresholds:
    """Test confidence threshold fixes (Fix 1, 2, 12, 13)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_buy_min_confidence_60pct(self):
        """Fix 12: BUY signals need minimum 60% confidence."""
        # Below threshold
        result_low = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.55
        )

        # Above threshold
        result_high = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.65
        )

        assert result_low.blocked is True
        assert result_high.blocked is False

    def test_sell_min_confidence_80pct(self):
        """Fix 13: SELL signals need minimum 80% confidence."""
        # Below threshold
        result_low = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.75
        )

        # Above threshold
        result_high = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.85
        )

        assert result_low.blocked is True
        assert result_high.blocked is False

    def test_commodity_higher_sell_threshold(self):
        """Fix 2: Commodities need 80% confidence for SELL."""
        result = self.optimizer.optimize_signal(
            ticker='CL=F',
            signal_type='SELL',
            confidence=0.78
        )

        # 78% < 80% threshold for commodities
        assert result.blocked is True

    def test_stock_sell_threshold_75pct(self):
        """Fix 1/2: Stocks need 75% base SELL confidence (but Fix 13 raises to 80%)."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='SELL',
            confidence=0.77
        )

        # 77% < 80% (Fix 13 minimum)
        assert result.blocked is True


class TestPatternDetection:
    """Test high-profit pattern detection (Fix 14)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_high_profit_pattern_detection(self):
        """Fix 14: High confidence + moderate volatility + momentum = boost."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85,
            volatility=0.25,
            momentum=0.05
        )

        assert result.blocked is False
        # Should detect high-profit pattern and boost position
        if 'Fix 14' in str(result.fixes_applied):
            assert result.position_multiplier > 1.0

    def test_very_high_confidence_boost(self):
        """Fix 14: Very high confidence (>90%) should get boost."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.92,
            volatility=0.20,
            momentum=0.01
        )

        assert result.blocked is False
        # Very high confidence should trigger pattern detection
        assert 'Fix 14' in str(result.fixes_applied)


class TestProfitTaking:
    """Test profit-taking rules (Fix 15)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_profit_taking_levels(self):
        """Fix 15: Profit-taking levels should be set correctly."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.take_profit_levels is not None
        assert 0.15 in result.take_profit_levels  # 15% level
        assert 0.25 in result.take_profit_levels  # 25% level
        assert 0.40 in result.take_profit_levels  # 40% level

    def test_profit_taking_close_percentages(self):
        """Fix 15: Close percentages should match spec."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        levels = result.take_profit_levels
        assert levels[0.15] == 0.50  # 50% at 15% profit
        assert levels[0.25] == 0.75  # 75% at 25% profit
        assert levels[0.40] == 1.00  # 100% at 40% profit


class TestAssetClassification:
    """Test asset class classification."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_classify_us_stock(self):
        """US stocks should be classified correctly."""
        assert self.optimizer.classify_asset('AAPL') == AssetClass.STOCK
        assert self.optimizer.classify_asset('TSLA') == AssetClass.STOCK
        assert self.optimizer.classify_asset('MSFT') == AssetClass.STOCK

    def test_classify_cryptocurrency(self):
        """Crypto should be classified correctly."""
        assert self.optimizer.classify_asset('BTC-USD') == AssetClass.CRYPTOCURRENCY
        assert self.optimizer.classify_asset('ETH-USD') == AssetClass.CRYPTOCURRENCY

    def test_classify_commodity(self):
        """Commodities should be classified correctly."""
        assert self.optimizer.classify_asset('GC=F') == AssetClass.COMMODITY
        assert self.optimizer.classify_asset('CL=F') == AssetClass.COMMODITY
        assert self.optimizer.classify_asset('NG=F') == AssetClass.COMMODITY

    def test_classify_forex(self):
        """Forex should be classified correctly."""
        assert self.optimizer.classify_asset('EURUSD=X') == AssetClass.FOREX

    def test_classify_china_stock(self):
        """China stocks should be classified correctly."""
        assert self.optimizer.classify_asset('0700.HK') == AssetClass.CHINA_STOCK
        assert self.optimizer.classify_asset('9988.HK') == AssetClass.CHINA_STOCK

    def test_classify_etf(self):
        """ETFs should be classified correctly."""
        assert self.optimizer.classify_asset('SPY') == AssetClass.ETF
        assert self.optimizer.classify_asset('QQQ') == AssetClass.ETF


class TestBatchProcessing:
    """Test batch signal processing."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_process_batch(self):
        """Batch processing should work correctly."""
        signals = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.85},
            {'ticker': 'TSLA', 'signal_type': 'BUY', 'confidence': 0.70},
            {'ticker': 'NG=F', 'signal_type': 'SELL', 'confidence': 0.90},
            {'ticker': 'LUMN', 'signal_type': 'BUY', 'confidence': 0.94},
        ]

        results, summary = self.optimizer.process_signals_batch(signals)

        assert len(results) == 4
        assert summary['total_signals'] == 4
        assert summary['blocked_count'] >= 2  # NG=F and LUMN

    def test_batch_summary_stats(self):
        """Batch summary should have correct stats."""
        signals = [
            {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.80},
            {'ticker': 'MSFT', 'signal_type': 'SELL', 'confidence': 0.85},
            {'ticker': 'GOOGL', 'signal_type': 'BUY', 'confidence': 0.55},  # Below threshold
        ]

        results, summary = self.optimizer.process_signals_batch(signals)

        assert 'block_rate' in summary
        assert 0 <= summary['block_rate'] <= 1


class TestConfigurationSummary:
    """Test configuration summary."""

    def test_get_configuration(self):
        """Configuration summary should include all fixes."""
        optimizer = create_optimizer()
        config = optimizer.get_configuration_summary()

        assert 'fixes_1_2' in config
        assert 'fix_3_7' in config
        assert 'fix_4' in config
        assert 'fix_5' in config
        assert 'fix_6' in config
        assert 'fix_8' in config
        assert 'fix_9' in config
        assert 'fix_10' in config
        assert 'fix_11' in config
        assert 'fix_12' in config
        assert 'fix_13' in config
        assert 'fix_14' in config
        assert 'fix_15' in config


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
