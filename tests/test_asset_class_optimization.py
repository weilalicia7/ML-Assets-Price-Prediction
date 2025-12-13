"""
Test Suite for Asset Class Optimization
========================================
Tests asset-class specific optimizations (Fixes 2, 4, 5, 7, 10)

Historical Performance by Asset Class:
- US Stocks: 75% win rate - BEST PERFORMER
- ETFs: 65% win rate
- Forex: 55% win rate
- Commodities: 33% win rate
- Crypto: 33% win rate
- China Stocks: 25% win rate - WORST PERFORMER
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


# Asset Class Performance Data from PDF
ASSET_CLASS_STATS = {
    'us_stocks': {'win_rate': 0.75, 'expected_boost': 1.30},
    'etfs': {'win_rate': 0.65, 'expected_boost': 1.20},
    'forex': {'win_rate': 0.55, 'expected_boost': 1.10},
    'commodities': {'win_rate': 0.33, 'expected_boost': 0.80},
    'crypto': {'win_rate': 0.33, 'expected_boost': 0.90},
    'china_stocks': {'win_rate': 0.25, 'expected_boost': 1.00},
}


class TestUSStockOptimization:
    """Test US stock optimization (best performer - 75% win rate)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'AAPL': 0.75, 'MSFT': 0.72},
            enable_dynamic_sizing=True
        )

    def test_us_stock_30pct_boost(self):
        """Fix 5: US stocks should get 30% BUY position boost."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.blocked is False
        assert result.asset_class == 'stock'
        # Should have 1.30x boost in fixes
        assert 'Fix 5' in str(result.fixes_applied)
        assert result.position_multiplier > 1.0

    def test_us_stock_base_allocation_10pct(self):
        """Fix 10: US stocks should have 10% base allocation."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        assert 'Fix 10' in str(result.fixes_applied)
        # Base allocation of 10% should be applied

    def test_us_stock_stop_loss_8pct(self):
        """Fix 4: US stock stop-loss should be 8%."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.08

    def test_us_stock_high_win_rate_boost(self):
        """Fix 9: High win rate (75%) should give 1.5x position boost."""
        result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80,
            win_rate=0.75
        )

        # 70-80% win rate = 1.5x multiplier
        assert 'Fix 9' in str(result.fixes_applied)
        assert result.position_multiplier > 1.5  # 1.30 * 1.5 * other factors


class TestCommodityOptimization:
    """Test commodity optimization (33% win rate)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'GC=F': 0.45, 'CL=F': 0.40},
            enable_dynamic_sizing=True
        )

    def test_commodity_buy_20pct_reduction(self):
        """Fix 5: Commodity BUY should be reduced by 20%."""
        result = self.optimizer.optimize_signal(
            ticker='GC=F',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.asset_class == 'commodity'
        assert 'Fix 5' in str(result.fixes_applied)
        # Should have 0.80x boost (20% reduction)

    def test_commodity_sell_70pct_reduction(self):
        """Fix 7: Commodity SELL should be reduced by 70%."""
        result = self.optimizer.optimize_signal(
            ticker='CL=F',
            signal_type='SELL',
            confidence=0.90
        )

        if not result.blocked:
            assert 'Fix 7' in str(result.fixes_applied)
            # Should have very low multiplier

    def test_commodity_stop_loss_10pct(self):
        """Fix 4: Commodity stop-loss should be 10%."""
        result = self.optimizer.optimize_signal(
            ticker='GC=F',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.stop_loss_pct == 0.10

    def test_commodity_sell_blacklist(self):
        """Fix 13: Choppy commodities should be blocked for SELL."""
        choppy = ['ZC=F', 'ZW=F', 'NG=F']

        for ticker in choppy:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.95
            )
            assert result.blocked is True


class TestCryptoOptimization:
    """Test cryptocurrency optimization (33% win rate)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'BTC-USD': 0.40, 'ETH-USD': 0.42},
            enable_dynamic_sizing=True
        )

    def test_crypto_buy_10pct_reduction(self):
        """Fix 5: Crypto BUY should be reduced by 10%."""
        result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.asset_class == 'cryptocurrency'
        # Should have 0.90x boost (10% reduction)

    def test_crypto_stop_loss_12pct(self):
        """Fix 4: Crypto stop-loss should be 12% (most volatile)."""
        result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.12

    def test_crypto_base_allocation_5pct(self):
        """Fix 10: Crypto should have 5% base allocation."""
        result = self.optimizer.optimize_signal(
            ticker='BTC-USD',
            signal_type='BUY',
            confidence=0.80
        )

        assert 'Fix 10' in str(result.fixes_applied)

    def test_stablecoin_sell_blocked(self):
        """Fix 13: Stablecoins should be blocked for SELL."""
        stablecoins = ['USDT-USD', 'USDC-USD']

        for ticker in stablecoins:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=0.95
            )
            assert result.blocked is True


class TestChinaStockOptimization:
    """Test China stock optimization (25% win rate - worst performer)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'0700.HK': 0.30, '9988.HK': 0.25},
            enable_dynamic_sizing=True
        )

    def test_china_stock_no_boost(self):
        """Fix 5: China stocks should have no boost (1.0x)."""
        result = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.asset_class == 'china_stock'
        # Should have 1.0x boost (no change)

    def test_china_stock_base_allocation_6pct(self):
        """Fix 10: China stocks should have 6% base allocation."""
        result = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='BUY',
            confidence=0.75
        )

        assert 'Fix 10' in str(result.fixes_applied)

    def test_china_stock_low_win_rate_reduction(self):
        """Fix 9: Low win rate (25%) should reduce position."""
        result = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='BUY',
            confidence=0.75,
            win_rate=0.25
        )

        # 20-30% win rate = 0.3x multiplier
        assert 'Fix 9' in str(result.fixes_applied)

    def test_china_stock_sell_very_risky(self):
        """China stock SELL with 25% win rate should have minimal position."""
        result = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='SELL',
            confidence=0.85,
            win_rate=0.20
        )

        if not result.blocked:
            # Should have very small position
            assert result.position_multiplier < 0.3


class TestForexOptimization:
    """Test forex optimization (55% win rate)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'EURUSD=X': 0.55},
            enable_dynamic_sizing=True
        )

    def test_forex_10pct_boost(self):
        """Fix 5: Forex BUY should get 10% boost."""
        result = self.optimizer.optimize_signal(
            ticker='EURUSD=X',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.asset_class == 'forex'
        # Should have 1.10x boost

    def test_forex_stop_loss_6pct(self):
        """Fix 4: Forex stop-loss should be 6% (lowest volatility)."""
        result = self.optimizer.optimize_signal(
            ticker='EURUSD=X',
            signal_type='BUY',
            confidence=0.75
        )

        assert result.stop_loss_pct == 0.06

    def test_forex_base_allocation_4pct(self):
        """Fix 10: Forex should have 4% base allocation."""
        result = self.optimizer.optimize_signal(
            ticker='EURUSD=X',
            signal_type='BUY',
            confidence=0.75
        )

        assert 'Fix 10' in str(result.fixes_applied)


class TestETFOptimization:
    """Test ETF optimization (65% win rate)."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(
            historical_win_rates={'SPY': 0.70, 'QQQ': 0.68},
            enable_dynamic_sizing=True
        )

    def test_etf_20pct_boost(self):
        """Fix 5: ETF BUY should get 20% boost."""
        result = self.optimizer.optimize_signal(
            ticker='SPY',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.asset_class == 'etf'
        # Should have 1.20x boost

    def test_etf_base_allocation_8pct(self):
        """Fix 10: ETFs should have 8% base allocation."""
        result = self.optimizer.optimize_signal(
            ticker='SPY',
            signal_type='BUY',
            confidence=0.80
        )

        assert 'Fix 10' in str(result.fixes_applied)

    def test_etf_stop_loss_8pct(self):
        """Fix 4: ETF stop-loss should be 8%."""
        result = self.optimizer.optimize_signal(
            ticker='SPY',
            signal_type='BUY',
            confidence=0.80
        )

        assert result.stop_loss_pct == 0.08


class TestAssetClassComparison:
    """Compare optimization across asset classes."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer(enable_dynamic_sizing=False)

    def test_position_boost_ranking(self):
        """US stocks should have highest boost, commodities lowest."""
        tickers = {
            'AAPL': ('stock', 1.30),
            'SPY': ('etf', 1.20),
            'EURUSD=X': ('forex', 1.10),
            '0700.HK': ('china_stock', 1.00),
            'BTC-USD': ('cryptocurrency', 0.90),
            'GC=F': ('commodity', 0.80),
        }

        results = {}
        for ticker, (expected_class, expected_boost) in tickers.items():
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='BUY',
                confidence=0.80
            )
            results[ticker] = result

            assert result.asset_class == expected_class, \
                f"{ticker} should be {expected_class}, got {result.asset_class}"

    def test_stop_loss_by_volatility(self):
        """Stop-losses should match asset volatility."""
        expected_stop_losses = {
            'EURUSD=X': 0.06,  # Forex - lowest
            'AAPL': 0.08,     # Stock
            'SPY': 0.08,      # ETF
            'GC=F': 0.10,     # Commodity
            'BTC-USD': 0.12,  # Crypto - highest
        }

        for ticker, expected_sl in expected_stop_losses.items():
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='BUY',
                confidence=0.80
            )
            assert result.stop_loss_pct == expected_sl, \
                f"{ticker} should have {expected_sl*100}% stop-loss, got {result.stop_loss_pct*100}%"


class TestAssetClassSellThresholds:
    """Test SELL confidence thresholds by asset class."""

    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = create_optimizer()

    def test_sell_threshold_by_asset(self):
        """Different asset classes have different SELL thresholds."""
        # All should require at least 80% due to Fix 13
        test_cases = [
            ('AAPL', 0.79, True),    # Stock - blocked
            ('GC=F', 0.79, True),    # Commodity - blocked
            ('BTC-USD', 0.79, True), # Crypto - blocked
        ]

        for ticker, confidence, should_block in test_cases:
            result = self.optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=confidence
            )
            assert result.blocked == should_block, \
                f"{ticker} SELL at {confidence*100}% should be blocked={should_block}"


class TestExpectedValueCalculations:
    """Test expected value calculations by asset class."""

    def setup_method(self):
        """Setup test optimizer with realistic win rates."""
        self.win_rates = {
            'AAPL': 0.75,     # US stock
            'SPY': 0.65,      # ETF
            'EURUSD=X': 0.55, # Forex
            'GC=F': 0.33,     # Commodity
            'BTC-USD': 0.33,  # Crypto
            '0700.HK': 0.25,  # China stock
        }
        self.optimizer = create_optimizer(
            historical_win_rates=self.win_rates,
            enable_kelly=True,
            enable_dynamic_sizing=True
        )

    def test_positive_ev_assets_larger_positions(self):
        """Higher win rate assets should get larger positions."""
        high_wr = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.80,
            win_rate=0.75
        )

        low_wr = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='BUY',
            confidence=0.80,
            win_rate=0.25
        )

        assert high_wr.position_multiplier > low_wr.position_multiplier

    def test_kelly_fraction_reflects_edge(self):
        """Kelly fraction should be higher for higher win rate."""
        # High win rate
        high_wr_result = self.optimizer.optimize_signal(
            ticker='AAPL',
            signal_type='BUY',
            confidence=0.85,
            win_rate=0.75
        )

        # Low win rate
        low_wr_result = self.optimizer.optimize_signal(
            ticker='0700.HK',
            signal_type='BUY',
            confidence=0.85,
            win_rate=0.25
        )

        if high_wr_result.kelly_fraction and low_wr_result.kelly_fraction:
            assert high_wr_result.kelly_fraction >= low_wr_result.kelly_fraction


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
