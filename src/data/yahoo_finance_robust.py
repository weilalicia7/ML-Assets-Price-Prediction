"""
Robust Yahoo Finance Data Access Layer

Provides reliable real-time data fetching with:
- Exponential backoff retry logic
- Data validation and freshness checks
- In-memory caching to reduce API calls
- Graceful error handling and fallbacks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_TIMEOUT = 30  # seconds
CACHE_TTL_SECONDS = 300  # 5 minutes cache TTL
MAX_DATA_AGE_DAYS = 5  # Max age for "fresh" data


# ============================================================================
# IN-MEMORY CACHE
# ============================================================================

class DataCache:
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = CACHE_TTL_SECONDS):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    self.hits += 1
                    return value
                else:
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        with self._lock:
            expiry = time.time() + (ttl or self.default_ttl)
            self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate*100:.1f}%",
                'cached_items': len(self._cache)
            }


# Global cache instance
_data_cache = DataCache()


# ============================================================================
# ROBUST DATA FETCHER
# ============================================================================

class YahooFinanceRobust:
    """
    Robust Yahoo Finance data fetcher with error handling and retries.

    Features:
    - Exponential backoff retry logic
    - Data validation and freshness checks
    - In-memory caching
    - Graceful fallbacks
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        timeout: int = DEFAULT_TIMEOUT,
        use_cache: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.use_cache = use_cache
        self.last_error = None
        self.fetch_count = 0
        self.success_count = 0

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
        # Add jitter (10-30% random variation)
        jitter = delay * (0.1 + 0.2 * np.random.random())
        return delay + jitter

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
        """
        Validate fetched data for completeness and freshness.

        Returns:
            (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Empty dataframe"

        required_columns = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"

        # Check for all-NaN data
        if df['Close'].isna().all():
            return False, "All Close prices are NaN"

        # Check data freshness (for daily data)
        if len(df) > 0:
            latest_date = df.index[-1]
            if hasattr(latest_date, 'date'):
                latest_date = latest_date.date()
            elif hasattr(latest_date, 'to_pydatetime'):
                latest_date = latest_date.to_pydatetime().date()

            today = datetime.now().date()
            days_old = (today - latest_date).days

            # Allow up to MAX_DATA_AGE_DAYS (weekends, holidays)
            if days_old > MAX_DATA_AGE_DAYS:
                return False, f"Data is {days_old} days old (max: {MAX_DATA_AGE_DAYS})"

        return True, ""

    def get_history(
        self,
        ticker: str,
        period: str = "60d",
        interval: str = "1d",
        start: str = None,
        end: str = None,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with retry logic and caching.

        Args:
            ticker: Stock ticker symbol
            period: Data period (e.g., "1d", "5d", "1mo", "3mo", "1y")
            interval: Data interval (e.g., "1m", "5m", "1h", "1d")
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            validate: Whether to validate data freshness

        Returns:
            DataFrame with OHLCV data or None on failure
        """
        self.fetch_count += 1

        # Build cache key
        cache_key = f"history:{ticker}:{period}:{interval}:{start}:{end}"

        # Check cache first
        if self.use_cache:
            cached = _data_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"[YF] Cache hit for {ticker}")
                return cached

        # Fetch with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"[YF] Fetching {ticker} (attempt {attempt + 1}/{self.max_retries})")

                ticker_obj = yf.Ticker(ticker)

                if start and end:
                    df = ticker_obj.history(start=start, end=end, interval=interval)
                else:
                    df = ticker_obj.history(period=period, interval=interval)

                # Validate data
                if validate:
                    is_valid, error_msg = self._validate_data(df, ticker)
                    if not is_valid:
                        raise ValueError(f"Data validation failed: {error_msg}")

                # Success - cache and return
                if self.use_cache and df is not None and not df.empty:
                    _data_cache.set(cache_key, df)

                self.success_count += 1
                logger.debug(f"[YF] Successfully fetched {len(df)} rows for {ticker}")
                return df

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[YF] Attempt {attempt + 1} failed for {ticker}: {e}")

                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.debug(f"[YF] Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        # All retries failed
        self.last_error = last_error
        logger.error(f"[YF] All {self.max_retries} attempts failed for {ticker}: {last_error}")
        return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current/latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price or None on failure
        """
        cache_key = f"price:{ticker}"

        # Check cache (shorter TTL for prices)
        if self.use_cache:
            cached = _data_cache.get(cache_key)
            if cached is not None:
                return cached

        # Try intraday first, then daily
        for period, interval in [("1d", "1m"), ("5d", "1d")]:
            try:
                df = self.get_history(ticker, period=period, interval=interval, validate=False)
                if df is not None and not df.empty:
                    price = float(df['Close'].iloc[-1])
                    if self.use_cache:
                        _data_cache.set(cache_key, price, ttl=60)  # 1 minute TTL for prices
                    return price
            except Exception as e:
                logger.debug(f"[YF] Price fetch failed for {ticker}: {e}")
                continue

        return None

    def get_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker info/metadata.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker info or None on failure
        """
        cache_key = f"info:{ticker}"

        if self.use_cache:
            cached = _data_cache.get(cache_key)
            if cached is not None:
                return cached

        for attempt in range(self.max_retries):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info

                if info and len(info) > 0:
                    if self.use_cache:
                        _data_cache.set(cache_key, info, ttl=3600)  # 1 hour TTL for info
                    return info

            except Exception as e:
                logger.debug(f"[YF] Info fetch attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self._exponential_backoff(attempt))

        return None

    def batch_get_prices(self, tickers: list, parallel: bool = True) -> Dict[str, Optional[float]]:
        """
        Fetch current prices for multiple tickers.

        Args:
            tickers: List of ticker symbols
            parallel: Whether to fetch in parallel (faster but more API calls)

        Returns:
            Dictionary of ticker -> price
        """
        prices = {}

        if parallel and len(tickers) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=min(5, len(tickers))) as executor:
                future_to_ticker = {
                    executor.submit(self.get_current_price, ticker): ticker
                    for ticker in tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        prices[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"[YF] Batch price failed for {ticker}: {e}")
                        prices[ticker] = None
        else:
            for ticker in tickers:
                prices[ticker] = self.get_current_price(ticker)

        return prices

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        success_rate = self.success_count / self.fetch_count if self.fetch_count > 0 else 0
        return {
            'fetch_count': self.fetch_count,
            'success_count': self.success_count,
            'success_rate': f"{success_rate*100:.1f}%",
            'last_error': self.last_error,
            'cache_stats': _data_cache.get_stats()
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global fetcher instance
_fetcher = YahooFinanceRobust()


def get_realtime_yahoo_data(
    ticker: str,
    period: str = "60d",
    max_retries: int = 3,
    validate: bool = True
) -> Optional[pd.DataFrame]:
    """
    Robust Yahoo Finance data fetcher with error handling and retries.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "9999.HK", "600519.SS")
        period: Data period (default "60d")
        max_retries: Maximum retry attempts (default 3)
        validate: Validate data freshness (default True)

    Returns:
        DataFrame with OHLCV data or None on failure

    Example:
        >>> df = get_realtime_yahoo_data("9999.HK")
        >>> if df is not None:
        ...     print(f"Latest close: {df['Close'].iloc[-1]}")
    """
    fetcher = YahooFinanceRobust(max_retries=max_retries)
    return fetcher.get_history(ticker, period=period, validate=validate)


def get_current_price(ticker: str) -> Optional[float]:
    """
    Get current/latest price for a ticker with caching.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Current price or None on failure
    """
    return _fetcher.get_current_price(ticker)


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate that a ticker exists and has data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        (is_valid, error_message or ticker_info)
    """
    try:
        info = _fetcher.get_info(ticker)
        if info and info.get('regularMarketPrice'):
            name = info.get('longName') or info.get('shortName') or ticker
            return True, name

        # Fallback: try to get price
        price = _fetcher.get_current_price(ticker)
        if price is not None:
            return True, f"{ticker} (price: {price:.2f})"

        return False, f"No data available for {ticker}"

    except Exception as e:
        return False, str(e)


def clear_cache():
    """Clear all cached data."""
    _data_cache.clear()
    logger.info("[YF] Cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _data_cache.get_stats()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("YAHOO FINANCE ROBUST DATA LAYER - TEST")
    print("=" * 60)

    # Test tickers
    test_tickers = ["AAPL", "9999.HK", "600519.SS", "INVALID_TICKER"]

    for ticker in test_tickers:
        print(f"\nTesting {ticker}...")

        # Validate
        is_valid, msg = validate_ticker(ticker)
        print(f"  Valid: {is_valid} - {msg}")

        if is_valid:
            # Get price
            price = get_current_price(ticker)
            print(f"  Price: {price}")

            # Get history
            df = get_realtime_yahoo_data(ticker, period="5d")
            if df is not None:
                print(f"  History: {len(df)} rows")

    # Show stats
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Cache stats: {get_cache_stats()}")
    print(f"Fetcher stats: {_fetcher.get_stats()}")
