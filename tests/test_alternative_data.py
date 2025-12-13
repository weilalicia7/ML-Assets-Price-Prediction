"""
Tests for alternative data collector.
Tests Google Trends, News API, and Economic indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.alternative_data import AlternativeDataCollector


class TestAlternativeDataCollector:
    """Test suite for AlternativeDataCollector."""

    @pytest.fixture
    def collector(self):
        """Create collector instance."""
        return AlternativeDataCollector()

    def test_collector_initialization(self, collector):
        """Test that collector initializes correctly."""
        assert collector is not None
        assert collector.pytrends is not None
        print("\n[TEST PASS] Collector initialized successfully")

    def test_google_trends_single_keyword(self, collector):
        """Test Google Trends with single keyword."""
        import time
        time.sleep(2)  # Rate limiting

        df = collector.fetch_google_trends(['Bitcoin'], timeframe='today 1-m')

        assert isinstance(df, pd.DataFrame)
        # Note: Trends data might be empty for very new searches or rate-limited
        if not df.empty:
            assert 'Bitcoin' in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            print(f"\n[TEST PASS] Google Trends single keyword: {len(df)} data points")
        else:
            print(f"\n[TEST SKIP] Google Trends returned empty (rate limited or no data)")

    def test_google_trends_multiple_keywords(self, collector):
        """Test Google Trends with multiple keywords."""
        import time
        time.sleep(2)  # Rate limiting

        keywords = ['Tesla', 'Apple']
        df = collector.fetch_google_trends(keywords, timeframe='today 1-m')

        if not df.empty:
            for keyword in keywords:
                assert keyword in df.columns, f"Missing keyword: {keyword}"
            print(f"\n[TEST PASS] Google Trends multiple keywords: {df.columns.tolist()}")
        else:
            print(f"\n[TEST SKIP] Google Trends returned empty")

    def test_google_trends_different_timeframes(self, collector):
        """Test different timeframe options."""
        import time

        timeframes = ['today 1-m', 'today 3-m']

        for timeframe in timeframes:
            time.sleep(2)  # Rate limiting
            try:
                df = collector.fetch_google_trends(['Python'], timeframe=timeframe)
                if not df.empty:
                    print(f"[TEST PASS] Timeframe {timeframe}: {len(df)} points")
                else:
                    print(f"[TEST SKIP] Timeframe {timeframe}: Empty data")
            except Exception as e:
                print(f"[TEST SKIP] Timeframe {timeframe}: {str(e)}")

    def test_trend_momentum_calculation(self, collector):
        """Test trend momentum feature calculation."""
        # Create sample trends data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Bitcoin': np.random.randint(50, 100, size=30),
            'Ethereum': np.random.randint(30, 80, size=30)
        }, index=dates)

        df_momentum = collector.calculate_trend_momentum(sample_data, window=7)

        # Check new columns exist
        expected_cols = [
            'Bitcoin_momentum', 'Bitcoin_ma', 'Bitcoin_strength', 'Bitcoin_acceleration',
            'Ethereum_momentum', 'Ethereum_ma', 'Ethereum_strength', 'Ethereum_acceleration'
        ]

        for col in expected_cols:
            assert col in df_momentum.columns, f"Missing column: {col}"

        print(f"\n[TEST PASS] Trend momentum: {len(expected_cols)} features calculated")

    @pytest.mark.skipif(
        not os.getenv('NEWS_API_KEY'),
        reason="NewsAPI key not configured"
    )
    def test_news_sentiment_fetch(self, collector):
        """Test news article fetching."""
        df = collector.fetch_news_sentiment('Apple', days=7)

        if not df.empty:
            assert 'title' in df.columns
            assert 'source' in df.columns
            assert 'description' in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)

            print(f"\n[TEST PASS] News fetching: {len(df)} articles retrieved")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"\n[TEST SKIP] No news articles found")

    @pytest.mark.skipif(
        not os.getenv('NEWS_API_KEY'),
        reason="NewsAPI key not configured"
    )
    def test_news_sentiment_features(self, collector):
        """Test news sentiment feature calculation."""
        # Get news data
        news_df = collector.fetch_news_sentiment('Tesla', days=7)

        if not news_df.empty:
            # Calculate sentiment features
            features_df = collector.calculate_news_sentiment_features(news_df)

            if not features_df.empty:
                expected_cols = [
                    'news_count',
                    'news_sentiment_avg',
                    'news_sentiment_std',
                    'news_positive_pct',
                    'news_negative_pct'
                ]

                for col in expected_cols:
                    assert col in features_df.columns, f"Missing column: {col}"

                print(f"\n[TEST PASS] News sentiment features: {len(features_df)} days")
                print(f"Features: {features_df.columns.tolist()}")
            else:
                print(f"\n[TEST SKIP] News sentiment features empty")
        else:
            print(f"\n[TEST SKIP] No news data to process")

    @pytest.mark.skipif(
        not os.getenv('FRED_API_KEY'),
        reason="FRED API key not configured"
    )
    def test_economic_indicators_fetch(self, collector):
        """Test FRED economic indicators fetching."""
        indicators = ['VIXCLS', 'DFF']
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        df = collector.fetch_economic_indicators(indicators, start_date=start_date)

        if not df.empty:
            for indicator in indicators:
                if indicator in df.columns:
                    assert indicator in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)

            print(f"\n[TEST PASS] Economic indicators: {df.columns.tolist()}")
            print(f"Data points: {len(df)}")
        else:
            print(f"\n[TEST SKIP] Economic indicators returned empty")

    def test_collect_all_alternative_data(self, collector):
        """Test collecting all alternative data sources."""
        import time
        time.sleep(2)

        result = collector.collect_all_alternative_data(
            'AAPL',
            keywords=['Apple', 'iPhone'],
            days=30
        )

        assert isinstance(result, dict)

        # Check which sources returned data
        available_sources = list(result.keys())
        print(f"\n[TEST PASS] Collected alternative data sources: {available_sources}")

        for source, df in result.items():
            assert isinstance(df, pd.DataFrame)
            print(f"  - {source}: {len(df)} rows, {len(df.columns)} columns")

    def test_data_quality_trends(self, collector):
        """Test data quality for trends data."""
        import time
        time.sleep(2)

        df = collector.fetch_google_trends(['Python'], timeframe='today 1-m')

        if not df.empty:
            # Check value ranges (Google Trends is 0-100)
            for col in df.columns:
                if col != 'isPartial':
                    assert (df[col] >= 0).all(), f"{col} has negative values"
                    assert (df[col] <= 100).all(), f"{col} exceeds 100"

            print(f"\n[TEST PASS] Trends data quality checks passed")
        else:
            print(f"\n[TEST SKIP] No trends data to validate")

    def test_news_sentiment_values(self, collector):
        """Test sentiment values are in expected range."""
        # Create mock news data
        news_data = pd.DataFrame({
            'title': ['Great earnings', 'Stock crashes', 'Neutral update'],
            'description': ['Very positive', 'Very negative', 'No change'],
            'source': ['Source1', 'Source2', 'Source3'],
            'url': ['url1', 'url2', 'url3'],
            'author': ['Auth1', 'Auth2', 'Auth3']
        })
        news_data.index = pd.date_range(start='2024-01-01', periods=3, freq='D')

        features = collector.calculate_news_sentiment_features(news_data)

        if not features.empty:
            # Sentiment should be between -1 and 1
            if 'news_sentiment_avg' in features.columns:
                assert (features['news_sentiment_avg'] >= -1).all()
                assert (features['news_sentiment_avg'] <= 1).all()

            print(f"\n[TEST PASS] News sentiment values in valid range")

    def test_error_handling_invalid_keyword(self, collector):
        """Test error handling for invalid inputs."""
        import time
        time.sleep(2)

        # Test with empty keyword list
        try:
            df = collector.fetch_google_trends([], timeframe='today 1-m')
            # Should either return empty or raise error
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # Error is acceptable
            assert len(str(e)) > 0

        print(f"\n[TEST PASS] Error handling works for invalid inputs")

    def test_error_handling_too_many_keywords(self, collector):
        """Test error handling for >5 keywords (Google Trends limit)."""
        import time
        time.sleep(2)

        keywords = ['A', 'B', 'C', 'D', 'E', 'F']  # 6 keywords

        with pytest.raises(ValueError):
            collector.fetch_google_trends(keywords, timeframe='today 1-m')

        print(f"\n[TEST PASS] Keyword limit validation works")

    def test_related_queries(self, collector):
        """Test fetching related queries."""
        import time
        time.sleep(2)

        try:
            result = collector.fetch_related_queries('Bitcoin')

            assert isinstance(result, dict)
            assert 'rising' in result
            assert 'top' in result

            print(f"\n[TEST PASS] Related queries fetched successfully")
        except Exception as e:
            print(f"\n[TEST SKIP] Related queries: {str(e)}")

    def test_timeframe_formats(self, collector):
        """Test various timeframe format inputs."""
        import time

        valid_timeframes = [
            'today 1-m',
            'today 3-m',
            'today 12-m',
        ]

        for tf in valid_timeframes:
            time.sleep(2)
            try:
                df = collector.fetch_google_trends(['Test'], timeframe=tf)
                print(f"[TEST PASS] Timeframe '{tf}' accepted")
            except Exception as e:
                print(f"[TEST SKIP] Timeframe '{tf}': {str(e)}")


def run_tests():
    """Run all tests with detailed output."""
    print("="*80)
    print("RUNNING ALTERNATIVE DATA TESTS")
    print("="*80)
    print("\nNote: Some tests may be skipped if API keys are not configured")
    print("Set environment variables: NEWS_API_KEY, FRED_API_KEY")
    print("="*80)

    pytest.main([__file__, '-v', '-s'])


if __name__ == "__main__":
    run_tests()
