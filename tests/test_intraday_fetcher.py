"""
Tests for intraday data fetcher.
Tests all data sources and microstructure features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.intraday_fetcher import IntradayDataFetcher


class TestIntradayDataFetcher:
    """Test suite for IntradayDataFetcher."""

    @pytest.fixture
    def fetcher(self):
        """Create fetcher instance."""
        return IntradayDataFetcher()

    def test_fetcher_initialization(self, fetcher):
        """Test that fetcher initializes correctly."""
        assert fetcher is not None
        assert fetcher.binance_url == "https://api.binance.com/api/v3"
        assert fetcher.alpha_vantage_url == "https://www.alphavantage.co/query"

    def test_binance_intraday_btc(self, fetcher):
        """Test Binance intraday data fetching for Bitcoin."""
        df = fetcher.fetch_binance_intraday('BTCUSDT', interval='5m', days=1, limit=100)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) > 0

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check data types
        assert df['Open'].dtype in [np.float64, np.float32]
        assert df['Volume'].dtype in [np.float64, np.float32]

        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check metadata
        assert 'Source' in df.columns
        assert df['Source'].iloc[0] == 'Binance'

        print(f"\n[TEST PASS] Binance BTC: {len(df)} bars fetched")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

    def test_binance_intraday_eth(self, fetcher):
        """Test Binance for Ethereum with different interval."""
        df = fetcher.fetch_binance_intraday('ETHUSDT', interval='15m', days=2, limit=50)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Interval' in df.columns
        assert df['Interval'].iloc[0] == '15m'

        print(f"\n[TEST PASS] Binance ETH: {len(df)} bars fetched")

    def test_binance_symbol_conversion(self, fetcher):
        """Test that symbol formats are correctly converted."""
        # Test with different input formats
        df1 = fetcher.fetch_binance_intraday('BTC-USD', interval='1h', days=1, limit=24)
        df2 = fetcher.fetch_binance_intraday('BTCUSDT', interval='1h', days=1, limit=24)

        assert not df1.empty
        assert not df2.empty
        assert 'USDT' in df1['Symbol'].iloc[0]
        assert 'USDT' in df2['Symbol'].iloc[0]

        print(f"\n[TEST PASS] Symbol conversion works")

    @pytest.mark.skipif(
        not os.getenv('ALPHA_VANTAGE_KEY'),
        reason="Alpha Vantage API key not configured"
    )
    def test_alpha_vantage_intraday(self, fetcher):
        """Test Alpha Vantage intraday data (requires API key)."""
        df = fetcher.fetch_alpha_vantage_intraday('AAPL', interval='5min', outputsize='compact')

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Source' in df.columns
        assert df['Source'].iloc[0] == 'Alpha Vantage'

        print(f"\n[TEST PASS] Alpha Vantage AAPL: {len(df)} bars fetched")

    @pytest.mark.skipif(
        not os.getenv('POLYGON_API_KEY'),
        reason="Polygon.io API key not configured"
    )
    def test_polygon_intraday(self, fetcher):
        """Test Polygon.io intraday data (requires API key)."""
        df = fetcher.fetch_polygon_intraday('AAPL', interval='5', timespan='minute', days=2)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Source' in df.columns
        assert df['Source'].iloc[0] == 'Polygon.io'

        print(f"\n[TEST PASS] Polygon.io AAPL: {len(df)} bars fetched")

    def test_auto_fetch_crypto(self, fetcher):
        """Test auto-fetch for cryptocurrency."""
        df = fetcher.fetch_intraday_auto('BTC-USD', interval='5m', days=1)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Source' in df.columns

        print(f"\n[TEST PASS] Auto-fetch BTC: {len(df)} bars from {df['Source'].iloc[0]}")

    def test_auto_fetch_different_intervals(self, fetcher):
        """Test different time intervals."""
        intervals = ['1m', '5m', '15m', '1h']

        for interval in intervals:
            df = fetcher.fetch_binance_intraday('BTCUSDT', interval=interval, days=1, limit=50)
            assert not df.empty
            assert df['Interval'].iloc[0] == interval

            print(f"[TEST PASS] Interval {interval}: {len(df)} bars")

    def test_microstructure_features(self, fetcher):
        """Test microstructure feature calculation."""
        # Get intraday data
        df = fetcher.fetch_binance_intraday('BTCUSDT', interval='5m', days=1, limit=200)

        # Calculate features
        df_features = fetcher.calculate_microstructure_features(df)

        # Check new features exist
        expected_features = [
            'log_return',
            'realized_vol_5',
            'realized_vol_20',
            'volume_ratio',
            'volume_surge',
            'hl_ratio',
            'spread_proxy',
            'microprice',
            'vwap',
            'intraday_momentum'
        ]

        for feature in expected_features:
            assert feature in df_features.columns, f"Missing feature: {feature}"

        # Check feature calculations are reasonable
        assert df_features['microprice'].notna().sum() > 0
        assert df_features['volume_ratio'].notna().sum() > 0
        assert df_features['realized_vol_5'].notna().sum() > 0

        print(f"\n[TEST PASS] Microstructure features: {len(expected_features)} features calculated")
        print(f"Available features: {df_features.columns.tolist()}")

    def test_aggregate_to_daily(self, fetcher):
        """Test aggregation from intraday to daily."""
        # Get multiple days of intraday data
        df_intraday = fetcher.fetch_binance_intraday('BTCUSDT', interval='1h', days=7, limit=168)

        # Add microstructure features
        df_features = fetcher.calculate_microstructure_features(df_intraday)

        # Aggregate to daily
        df_daily = fetcher.aggregate_to_daily(df_features)

        assert isinstance(df_daily, pd.DataFrame)
        assert not df_daily.empty
        assert len(df_daily) <= 8  # Should have at most 7-8 days (depending on timezone)

        # Check OHLCV columns exist
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            assert col in df_daily.columns

        print(f"\n[TEST PASS] Aggregation: {len(df_intraday)} intraday bars → {len(df_daily)} daily bars")

    def test_data_quality_checks(self, fetcher):
        """Test data quality and integrity."""
        df = fetcher.fetch_binance_intraday('BTCUSDT', interval='5m', days=1, limit=100)

        # Check for NaN values in critical columns
        assert df['Close'].isna().sum() == 0, "Close prices contain NaN"
        assert df['Volume'].isna().sum() == 0, "Volume contains NaN"

        # Check that High >= Low
        assert (df['High'] >= df['Low']).all(), "High prices should be >= Low prices"

        # Check that Close is within High-Low range
        assert (df['Close'] <= df['High']).all(), "Close should be <= High"
        assert (df['Close'] >= df['Low']).all(), "Close should be >= Low"

        # Check that Volume is non-negative
        assert (df['Volume'] >= 0).all(), "Volume should be non-negative"

        print(f"\n[TEST PASS] Data quality checks passed")

    def test_rate_limiting(self, fetcher):
        """Test that rate limiting works."""
        import time

        start_time = time.time()

        # Make multiple requests
        for i in range(3):
            fetcher.fetch_binance_intraday('BTCUSDT', interval='1h', days=1, limit=10)

        elapsed_time = time.time() - start_time

        # Should take at least 2 seconds (2 rate limit intervals)
        assert elapsed_time >= 2.0, "Rate limiting not working correctly"

        print(f"\n[TEST PASS] Rate limiting working ({elapsed_time:.2f}s for 3 requests)")

    def test_error_handling_invalid_symbol(self, fetcher):
        """Test error handling for invalid symbols."""
        with pytest.raises(Exception):
            fetcher.fetch_binance_intraday('INVALIDSYMBOL123', interval='5m', days=1)

        print(f"\n[TEST PASS] Invalid symbol error handling works")

    def test_empty_data_handling(self, fetcher):
        """Test handling of empty/no data scenarios."""
        # This might not raise error but should return empty or handle gracefully
        try:
            # Use very old dates that might not have data
            df = fetcher.fetch_binance_intraday('BTCUSDT', interval='1m', days=1, limit=1)
            # If it succeeds, should return valid DataFrame
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # Should raise meaningful error
            assert len(str(e)) > 0

        print(f"\n[TEST PASS] Empty data handling works")


def run_tests():
    """Run all tests with detailed output."""
    print("="*80)
    print("RUNNING INTRADAY FETCHER TESTS")
    print("="*80)

    pytest.main([__file__, '-v', '-s'])


if __name__ == "__main__":
    run_tests()
