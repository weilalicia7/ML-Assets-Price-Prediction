"""
Unit tests for data fetching module.
Tests stock and crypto data retrieval.
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.fetch_data import DataFetcher


class TestDataFetcher:
    """Test cases for DataFetcher class."""

    def test_single_stock_fetch(self):
        """Test fetching data for a single stock."""
        print("\n" + "="*60)
        print("TEST 1: Fetching single stock (AAPL)")
        print("="*60)

        fetcher = DataFetcher(
            tickers='AAPL',
            start_date='2024-01-01',
            end_date='2024-02-01'
        )

        data = fetcher.fetch_single_ticker('AAPL')

        # Assertions
        assert not data.empty, "Data should not be empty"
        assert 'Ticker' in data.columns, "Should have Ticker column"
        assert 'AssetType' in data.columns, "Should have AssetType column"
        assert data['AssetType'].iloc[0] == 'stock', "Should be classified as stock"
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), \
            "Should have OHLC + Volume columns"

        print(f"✓ Successfully fetched {len(data)} rows")
        print(f"✓ Columns: {list(data.columns)}")
        print(f"✓ Date range: {data.index.min()} to {data.index.max()}")
        print(data.head())

    def test_single_crypto_fetch(self):
        """Test fetching data for a single cryptocurrency."""
        print("\n" + "="*60)
        print("TEST 2: Fetching single crypto (BTC-USD)")
        print("="*60)

        fetcher = DataFetcher(
            tickers='BTC-USD',
            start_date='2024-01-01',
            end_date='2024-02-01'
        )

        data = fetcher.fetch_single_ticker('BTC-USD')

        # Assertions
        assert not data.empty, "Data should not be empty"
        assert data['Ticker'].iloc[0] == 'BTC-USD', "Ticker should be BTC-USD"
        assert data['AssetType'].iloc[0] == 'crypto', "Should be classified as crypto"

        print(f"✓ Successfully fetched {len(data)} rows")
        print(f"✓ Asset type: {data['AssetType'].iloc[0]}")
        print(data.head())

    def test_multiple_assets_fetch(self):
        """Test fetching data for multiple assets (stocks + crypto)."""
        print("\n" + "="*60)
        print("TEST 3: Fetching multiple assets (AAPL, MSFT, BTC-USD)")
        print("="*60)

        tickers = ['AAPL', 'MSFT', 'BTC-USD']
        fetcher = DataFetcher(
            tickers=tickers,
            start_date='2024-01-01',
            end_date='2024-02-01'
        )

        data = fetcher.fetch_all()

        # Assertions
        assert not data.empty, "Data should not be empty"
        unique_tickers = data['Ticker'].unique()
        assert len(unique_tickers) == 3, f"Should have 3 tickers, got {len(unique_tickers)}"

        # Check asset types
        stocks = data[data['AssetType'] == 'stock']
        crypto = data[data['AssetType'] == 'crypto']

        assert len(stocks) > 0, "Should have stock data"
        assert len(crypto) > 0, "Should have crypto data"

        print(f"✓ Successfully fetched data for {len(unique_tickers)} assets")
        print(f"✓ Total rows: {len(data)}")
        print(f"✓ Stocks: {len(stocks)} rows")
        print(f"✓ Crypto: {len(crypto)} rows")
        print("\nRows per ticker:")
        print(data.groupby('Ticker').size())

    def test_data_save_load(self):
        """Test saving and loading data."""
        print("\n" + "="*60)
        print("TEST 4: Save and load data")
        print("="*60)

        fetcher = DataFetcher(
            tickers='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-15'
        )

        # Fetch and save
        data = fetcher.fetch_all()
        test_path = 'data/raw/test_data.csv'
        fetcher.save_data(data, test_path)

        # Load
        loaded_data = DataFetcher.load_data(test_path)

        # Assertions
        assert len(data) == len(loaded_data), "Loaded data should have same length"
        assert list(data.columns) == list(loaded_data.columns), "Columns should match"

        print(f"✓ Successfully saved and loaded {len(loaded_data)} rows")
        print(f"✓ File: {test_path}")

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            print("✓ Cleaned up test file")

    def test_invalid_ticker(self):
        """Test handling of invalid ticker."""
        print("\n" + "="*60)
        print("TEST 5: Invalid ticker handling")
        print("="*60)

        fetcher = DataFetcher(
            tickers='INVALIDTICKER123',
            start_date='2024-01-01',
            end_date='2024-01-15'
        )

        try:
            data = fetcher.fetch_single_ticker('INVALIDTICKER123')
            print("✗ Should have raised an error for invalid ticker")
            assert False, "Should raise error for invalid ticker"
        except ValueError as e:
            print(f"✓ Correctly raised error: {str(e)}")
            assert True

    def test_date_range(self):
        """Test custom date range."""
        print("\n" + "="*60)
        print("TEST 6: Custom date range")
        print("="*60)

        start = '2023-06-01'
        end = '2023-07-01'

        fetcher = DataFetcher(
            tickers='MSFT',
            start_date=start,
            end_date=end
        )

        data = fetcher.fetch_single_ticker('MSFT')

        # Check date range
        min_date = data.index.min()
        max_date = data.index.max()

        print(f"✓ Requested range: {start} to {end}")
        print(f"✓ Actual range: {min_date} to {max_date}")
        print(f"✓ Rows fetched: {len(data)}")

        assert min_date >= pd.Timestamp(start), "Min date should be >= start"
        assert max_date <= pd.Timestamp(end), "Max date should be <= end"


def run_all_tests():
    """Run all tests manually."""
    print("\n" + "="*80)
    print("RUNNING DATA FETCHER TESTS")
    print("="*80)

    tester = TestDataFetcher()

    try:
        tester.test_single_stock_fetch()
        print("\n✓ TEST 1 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")

    try:
        tester.test_single_crypto_fetch()
        print("\n✓ TEST 2 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")

    try:
        tester.test_multiple_assets_fetch()
        print("\n✓ TEST 3 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")

    try:
        tester.test_data_save_load()
        print("\n✓ TEST 4 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")

    try:
        tester.test_invalid_ticker()
        print("\n✓ TEST 5 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")

    try:
        tester.test_date_range()
        print("\n✓ TEST 6 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Run tests manually
    run_all_tests()

    # Or use pytest
    # pytest.main([__file__, '-v'])
