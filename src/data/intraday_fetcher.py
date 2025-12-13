"""
Intraday Data Fetcher - Professional-grade high-frequency data collection
Supports multiple sources for 1-minute to 1-hour bars

Sources:
1. Binance - Crypto (1m, 5m, 15m, 1h, 4h) - Free, no API key
2. Alpha Vantage - Stocks (1min, 5min, 15min, 30min, 60min) - Free tier
3. Polygon.io - Stocks/Crypto (1min, 5min, 15min, 1hour) - Free tier available

Critical for:
- Real volatility estimation
- Microstructure features
- Realistic trading simulation
- Better signal generation
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import os


class IntradayDataFetcher:
    """
    Fetches intraday (high-frequency) financial data from multiple sources.

    Enables professional-level features:
    - Real volatility estimation
    - Market microstructure analysis
    - Intraday pattern recognition
    - High-frequency trading simulation
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        polygon_key: Optional[str] = None
    ):
        """
        Initialize intraday data fetcher.

        Args:
            alpha_vantage_key: Alpha Vantage API key (free at https://www.alphavantage.co/)
            polygon_key: Polygon.io API key (free tier at https://polygon.io/)
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY')
        self.polygon_key = polygon_key or os.getenv('POLYGON_API_KEY')

        # API endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.polygon_url = "https://api.polygon.io"
        self.binance_url = "https://api.binance.com/api/v3"

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests

    def _rate_limit(self, source: str):
        """Implement rate limiting to avoid API throttling."""
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

        self.last_request_time[source] = time.time()

    # ===========================
    # BINANCE (Crypto - Free, No API Key)
    # ===========================

    def fetch_binance_intraday(
        self,
        symbol: str,
        interval: str = '1m',
        days: int = 7,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch intraday crypto data from Binance (free, no API key needed).

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Timeframe - '1m', '5m', '15m', '30m', '1h', '4h', '1d'
            days: Number of days of historical data
            limit: Max number of candles (max 1000 per request)

        Returns:
            DataFrame with OHLCV data at specified interval
        """
        print(f"[Binance] Fetching {symbol} ({interval} bars)...")

        # Convert common formats to Binance format
        if symbol.endswith('-USD'):
            symbol = symbol.replace('-USD', 'USDT')
        elif symbol.endswith('-USDT'):
            symbol = symbol.replace('-', '')
        elif not symbol.endswith('USDT') and not symbol.endswith('USD'):
            symbol = symbol + 'USDT'

        # Map interval formats
        interval_map = {
            '1min': '1m', '5min': '5m', '15min': '15m', '30min': '30m', '60min': '1h',
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h'
        }
        interval = interval_map.get(interval, interval)

        try:
            self._rate_limit('binance')

            # Calculate start time
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            url = f"{self.binance_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': min(limit, 1000)
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError(f"No data returned from Binance for {symbol}")

            # Parse klines data
            df = pd.DataFrame(data, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base',
                'Taker_buy_quote', 'Ignore'
            ])

            # Convert to proper types
            df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
            df = df.set_index('Date')

            # Select and convert main OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']].astype(float)

            # Add metadata
            df['Source'] = 'Binance'
            df['Interval'] = interval
            df['Symbol'] = symbol

            print(f"[OK] Retrieved {len(df)} {interval} bars from Binance")
            print(f"     Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"[ERROR] Binance error: {str(e)}")
            raise

    # ===========================
    # ALPHA VANTAGE (Stocks - Intraday)
    # ===========================

    def fetch_alpha_vantage_intraday(
        self,
        ticker: str,
        interval: str = '5min',
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Fetch intraday stock data from Alpha Vantage.

        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
            interval: '1min', '5min', '15min', '30min', '60min'
            outputsize: 'compact' (latest 100 data points) or 'full' (20+ days)

        Returns:
            DataFrame with intraday OHLCV data
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key required. Get free key at https://www.alphavantage.co/")

        print(f"[Alpha Vantage] Fetching {ticker} ({interval} bars)...")

        try:
            self._rate_limit('alpha_vantage')

            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': ticker,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.alpha_vantage_key
            }

            response = requests.get(self.alpha_vantage_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            if 'Note' in data:
                raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

            # Parse time series data
            ts_key = f'Time Series ({interval})'
            if ts_key not in data:
                raise ValueError(f"No intraday data from Alpha Vantage for {ticker}")

            ts_data = data[ts_key]
            df = pd.DataFrame.from_dict(ts_data, orient='index')

            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            df = df.astype(float)
            df.sort_index(inplace=True)

            # Add metadata
            df['Source'] = 'Alpha Vantage'
            df['Interval'] = interval
            df['Symbol'] = ticker

            print(f"[OK] Retrieved {len(df)} {interval} bars from Alpha Vantage")
            print(f"     Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"[ERROR] Alpha Vantage error: {str(e)}")
            raise

    # ===========================
    # POLYGON.IO (Stocks/Crypto - Intraday)
    # ===========================

    def fetch_polygon_intraday(
        self,
        ticker: str,
        interval: str = '5',
        timespan: str = 'minute',
        days: int = 2
    ) -> pd.DataFrame:
        """
        Fetch intraday data from Polygon.io (supports stocks and crypto).

        Args:
            ticker: Stock symbol (e.g., 'AAPL') or crypto pair (e.g., 'X:BTCUSD')
            interval: Number of intervals (e.g., '1', '5', '15')
            timespan: 'minute' or 'hour'
            days: Number of days of historical data

        Returns:
            DataFrame with intraday OHLCV data
        """
        if not self.polygon_key:
            raise ValueError("Polygon.io API key required. Get free key at https://polygon.io/")

        print(f"[Polygon.io] Fetching {ticker} ({interval}{timespan} bars)...")

        try:
            self._rate_limit('polygon')

            # Format dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.polygon_url}/v2/aggs/ticker/{ticker}/range/{interval}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            params = {
                'apiKey': self.polygon_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK' or 'results' not in data:
                raise ValueError(f"No data from Polygon.io for {ticker}: {data.get('message', 'Unknown error')}")

            # Parse results
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('Date')

            # Rename columns
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'n': 'Trades'
            })

            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']]

            # Add metadata
            df['Source'] = 'Polygon.io'
            df['Interval'] = f'{interval}{timespan}'
            df['Symbol'] = ticker

            print(f"[OK] Retrieved {len(df)} bars from Polygon.io")
            print(f"     Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"[ERROR] Polygon.io error: {str(e)}")
            raise

    # ===========================
    # AUTO-SELECT BEST SOURCE
    # ===========================

    def fetch_intraday_auto(
        self,
        ticker: str,
        interval: str = '5min',
        days: int = 7
    ) -> pd.DataFrame:
        """
        Automatically fetch intraday data from the best available source.

        Args:
            ticker: Ticker symbol (stocks/crypto)
            interval: Time interval ('1min', '5min', '15min', '30min', '1h')
            days: Number of days of data

        Returns:
            DataFrame with intraday data from first successful source
        """
        print(f"\n[AUTO] Fetching intraday data for {ticker}...")

        # Determine if crypto
        is_crypto = (
            '-USD' in ticker or
            ticker.upper() in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'MATIC'] or
            ticker.endswith('USDT')
        )

        if is_crypto:
            # Try Binance first for crypto (free, reliable)
            try:
                return self.fetch_binance_intraday(ticker, interval, days)
            except Exception as e:
                print(f"[WARN] Binance failed: {str(e)}")

            # Try Polygon.io for crypto
            if self.polygon_key:
                try:
                    # Format ticker for Polygon crypto
                    crypto_ticker = ticker.replace('-USD', '').replace('USDT', '')
                    polygon_ticker = f"X:{crypto_ticker}USD"

                    interval_num = interval.replace('min', '').replace('m', '')
                    return self.fetch_polygon_intraday(polygon_ticker, interval_num, 'minute', days)
                except Exception as e:
                    print(f"[WARN] Polygon.io failed: {str(e)}")

        else:
            # For stocks, try Alpha Vantage first
            if self.alpha_vantage_key:
                try:
                    av_interval = interval.replace('m', 'min')
                    outputsize = 'full' if days > 2 else 'compact'
                    return self.fetch_alpha_vantage_intraday(ticker, av_interval, outputsize)
                except Exception as e:
                    print(f"[WARN] Alpha Vantage failed: {str(e)}")

            # Try Polygon.io for stocks
            if self.polygon_key:
                try:
                    interval_num = interval.replace('min', '').replace('m', '')
                    return self.fetch_polygon_intraday(ticker, interval_num, 'minute', days)
                except Exception as e:
                    print(f"[WARN] Polygon.io failed: {str(e)}")

        raise ValueError(f"Failed to fetch intraday data from any source for {ticker}")

    # ===========================
    # MICROSTRUCTURE FEATURES
    # ===========================

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure features from intraday data.

        Critical for professional-level trading:
        - Realized volatility
        - Volume profiles
        - Intraday patterns
        - Trade intensity

        Args:
            df: DataFrame with intraday OHLCV data

        Returns:
            DataFrame with added microstructure features
        """
        print(f"\n[Microstructure] Calculating features...")

        df = df.copy()

        # 1. Realized Volatility (5-min, 1-hour windows)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['realized_vol_5'] = df['log_return'].rolling(5).std() * np.sqrt(252 * 78)  # Annualized
        df['realized_vol_20'] = df['log_return'].rolling(20).std() * np.sqrt(252 * 78)

        # 2. Volume Profile
        df['volume_ma_10'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
        df['volume_surge'] = (df['volume_ratio'] > 2).astype(int)

        # 3. Price Range Features (High-Low dynamics)
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['hl_ratio_ma'] = df['hl_ratio'].rolling(10).mean()

        # 4. Bid-Ask Spread Proxy (using High-Low)
        df['spread_proxy'] = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)

        # 5. Trade Intensity (if available)
        if 'Trades' in df.columns:
            df['trade_intensity'] = df['Trades'] / df['Trades'].rolling(10).mean()

        # 6. Microprice (mid-point estimator)
        df['microprice'] = (df['High'] + df['Low']) / 2

        # 7. VWAP (Volume-Weighted Average Price)
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # 8. Intraday Momentum
        df['intraday_momentum'] = df['Close'] / df['Open'] - 1

        # 9. Volatility Regime Indicator
        df['vol_regime'] = pd.qcut(
            df['realized_vol_20'].fillna(df['realized_vol_20'].median()),
            q=3,
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )

        print(f"[OK] Added {9} microstructure feature groups")

        return df

    # ===========================
    # AGGREGATION TO DAILY
    # ===========================

    def aggregate_to_daily(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate intraday data to daily bars with microstructure features.

        Args:
            intraday_df: DataFrame with intraday data

        Returns:
            Daily DataFrame with aggregated features
        """
        print(f"\n[Aggregation] Converting intraday to daily...")

        daily = pd.DataFrame()

        # Standard OHLCV aggregation
        daily['Open'] = intraday_df.groupby(intraday_df.index.date)['Open'].first()
        daily['High'] = intraday_df.groupby(intraday_df.index.date)['High'].max()
        daily['Low'] = intraday_df.groupby(intraday_df.index.date)['Low'].min()
        daily['Close'] = intraday_df.groupby(intraday_df.index.date)['Close'].last()
        daily['Volume'] = intraday_df.groupby(intraday_df.index.date)['Volume'].sum()

        # Aggregate microstructure features
        if 'realized_vol_20' in intraday_df.columns:
            daily['intraday_realized_vol'] = intraday_df.groupby(intraday_df.index.date)['realized_vol_20'].mean()

        if 'volume_surge' in intraday_df.columns:
            daily['volume_surge_count'] = intraday_df.groupby(intraday_df.index.date)['volume_surge'].sum()

        if 'Trades' in intraday_df.columns:
            daily['total_trades'] = intraday_df.groupby(intraday_df.index.date)['Trades'].sum()

        daily.index = pd.to_datetime(daily.index)
        daily.index.name = 'Date'

        print(f"[OK] Aggregated to {len(daily)} daily bars")

        return daily


def main():
    """
    Example usage of IntradayDataFetcher.
    """
    print("="*80)
    print("INTRADAY DATA FETCHER - EXAMPLES")
    print("="*80)

    fetcher = IntradayDataFetcher()

    # Example 1: Binance crypto (free, no API key)
    print("\n\nExample 1: Binance - Bitcoin 5-minute bars")
    print("-" * 60)
    try:
        btc_intraday = fetcher.fetch_binance_intraday('BTCUSDT', interval='5m', days=2)
        print(btc_intraday.head())
        print(f"\nShape: {btc_intraday.shape}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Auto-select source
    print("\n\nExample 2: Auto-select - Ethereum 15-minute bars")
    print("-" * 60)
    try:
        eth_intraday = fetcher.fetch_intraday_auto('ETH-USD', interval='15min', days=1)
        print(eth_intraday.tail())
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Microstructure features
    print("\n\nExample 3: Microstructure Features")
    print("-" * 60)
    try:
        btc_data = fetcher.fetch_binance_intraday('BTCUSDT', interval='5m', days=1)
        btc_features = fetcher.calculate_microstructure_features(btc_data)
        print("\nAvailable features:")
        print(btc_features.columns.tolist())
        print("\nSample data:")
        print(btc_features[['Close', 'Volume', 'realized_vol_5', 'volume_ratio', 'microprice']].tail())
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Aggregate to daily
    print("\n\nExample 4: Aggregate Intraday to Daily")
    print("-" * 60)
    try:
        eth_intraday = fetcher.fetch_binance_intraday('ETHUSDT', interval='1h', days=7)
        eth_intraday_features = fetcher.calculate_microstructure_features(eth_intraday)
        eth_daily = fetcher.aggregate_to_daily(eth_intraday_features)
        print(eth_daily.tail())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
