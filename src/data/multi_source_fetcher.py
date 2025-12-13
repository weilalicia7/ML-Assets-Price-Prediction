"""
Multi-source data fetcher for stocks and cryptocurrencies.
Supports multiple reliable public data sources beyond Yahoo Finance.

Data Sources:
1. Yahoo Finance (yfinance) - Stocks, indices, commodities
2. Alpha Vantage API - Stocks, forex, crypto (requires free API key)
3. CoinGecko API - Comprehensive crypto data (free, no API key needed)
4. FRED (Federal Reserve Economic Data) - Economic indicators
5. Quandl/Nasdaq Data Link - Financial & economic data
6. Polygon.io - Stocks, forex, crypto (free tier available)
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict
import os
import time


class MultiSourceDataFetcher:
    """
    Fetches financial data from multiple public sources.

    Supports Yahoo Finance, Alpha Vantage, CoinGecko, FRED, and more.
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        fred_key: Optional[str] = None,
        polygon_key: Optional[str] = None,
        quandl_key: Optional[str] = None
    ):
        """
        Initialize multi-source data fetcher.

        Args:
            alpha_vantage_key: Alpha Vantage API key (get free at https://www.alphavantage.co/)
            fred_key: FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
            polygon_key: Polygon.io API key (get free at https://polygon.io/)
            quandl_key: Quandl/Nasdaq Data Link API key
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY')
        self.fred_key = fred_key or os.getenv('FRED_API_KEY')
        self.polygon_key = polygon_key or os.getenv('POLYGON_API_KEY')
        self.quandl_key = quandl_key or os.getenv('QUANDL_API_KEY')

        # API endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.fred_url = "https://api.stlouisfed.org/fred/series/observations"
        self.polygon_url = "https://api.polygon.io/v2"

    # ===========================
    # YAHOO FINANCE (Primary)
    # ===========================
    def fetch_yahoo(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance using yfinance.

        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"[Yahoo Finance] Fetching {ticker}...")

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if data.empty:
                raise ValueError(f"No data from Yahoo Finance for {ticker}")

            data['Source'] = 'Yahoo Finance'
            print(f"[OK] Retrieved {len(data)} rows from Yahoo Finance")
            return data

        except Exception as e:
            print(f"[ERROR] Yahoo Finance error: {str(e)}")
            raise

    # ===========================
    # ALPHA VANTAGE
    # ===========================
    def fetch_alpha_vantage_daily(
        self,
        ticker: str,
        outputsize: str = 'full'
    ) -> pd.DataFrame:
        """
        Fetch daily stock data from Alpha Vantage.

        Args:
            ticker: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key required. Get free key at https://www.alphavantage.co/")

        print(f"[Alpha Vantage] Fetching {ticker}...")

        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': outputsize,
            'apikey': self.alpha_vantage_key
        }

        try:
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No data from Alpha Vantage for {ticker}")

            # Parse time series data
            ts_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')

            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume', 'Dividend', 'Split']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.sort_index(inplace=True)

            # Select main columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['Source'] = 'Alpha Vantage'

            print(f"[OK] Retrieved {len(df)} rows from Alpha Vantage")
            return df

        except Exception as e:
            print(f"[ERROR] Alpha Vantage error: {str(e)}")
            raise

    def fetch_alpha_vantage_crypto(
        self,
        symbol: str,
        market: str = 'USD'
    ) -> pd.DataFrame:
        """
        Fetch daily crypto data from Alpha Vantage.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            market: Market currency (default: 'USD')

        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key required")

        print(f"[Alpha Vantage] Fetching {symbol}-{market}...")

        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': market,
            'apikey': self.alpha_vantage_key
        }

        try:
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Digital Currency Daily)' not in data:
                raise ValueError(f"No crypto data from Alpha Vantage for {symbol}")

            ts_data = data['Time Series (Digital Currency Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')

            # Rename columns (use USD values)
            df = df[[f'1a. open ({market})', f'2a. high ({market})',
                     f'3a. low ({market})', f'4a. close ({market})', '5. volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.sort_index(inplace=True)
            df['Source'] = 'Alpha Vantage Crypto'

            print(f"[OK] Retrieved {len(df)} rows from Alpha Vantage Crypto")
            return df

        except Exception as e:
            print(f"[ERROR] Alpha Vantage Crypto error: {str(e)}")
            raise

    # ===========================
    # COINGECKO (Crypto - Free, No API Key)
    # ===========================
    def fetch_coingecko(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch crypto data from CoinGecko (free, no API key needed).

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of data (1, 7, 14, 30, 90, 180, 365, max)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"[CoinGecko] Fetching {coin_id}...")

        url = f"{self.coingecko_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError(f"No data from CoinGecko for {coin_id}")

            # Parse OHLC data
            df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df.drop('Timestamp', axis=1, inplace=True)
            df['Source'] = 'CoinGecko'

            print(f"[OK] Retrieved {len(df)} rows from CoinGecko")
            return df

        except Exception as e:
            print(f"[ERROR] CoinGecko error: {str(e)}")
            raise

    def fetch_coingecko_market_data(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch detailed market data from CoinGecko including volume and market cap.

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency (default: 'usd')
            days: Days of data

        Returns:
            DataFrame with price, volume, market cap
        """
        print(f"[CoinGecko Market] Fetching {coin_id} market data...")

        url = f"{self.coingecko_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse price, volume, market cap
            prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['Timestamp', 'Volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['Timestamp', 'MarketCap'])

            # Merge on timestamp
            df = prices.merge(volumes, on='Timestamp').merge(market_caps, on='Timestamp')
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df.drop('Timestamp', axis=1, inplace=True)
            df['Source'] = 'CoinGecko Market'

            print(f"[OK] Retrieved {len(df)} rows from CoinGecko Market")
            return df

        except Exception as e:
            print(f"[ERROR] CoinGecko Market error: {str(e)}")
            raise

    # ===========================
    # FRED (Economic Data)
    # ===========================
    def fetch_fred(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch economic data from FRED (Federal Reserve Economic Data).

        Common series:
        - 'DFF': Federal Funds Rate
        - 'DGS10': 10-Year Treasury Rate
        - 'UNRATE': Unemployment Rate
        - 'CPIAUCSL': Consumer Price Index
        - 'GDP': Gross Domestic Product

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with economic data
        """
        if not self.fred_key:
            raise ValueError("FRED API key required. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html")

        print(f"[FRED] Fetching {series_id}...")

        params = {
            'series_id': series_id,
            'api_key': self.fred_key,
            'file_type': 'json'
        }

        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date

        try:
            response = requests.get(self.fred_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'observations' not in data:
                raise ValueError(f"No data from FRED for {series_id}")

            # Parse observations
            obs = data['observations']
            df = pd.DataFrame(obs)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['value']].dropna()
            df.columns = [series_id]
            df['Source'] = 'FRED'

            print(f"[OK] Retrieved {len(df)} rows from FRED")
            return df

        except Exception as e:
            print(f"[ERROR] FRED error: {str(e)}")
            raise

    # ===========================
    # POLYGON.IO
    # ===========================
    def fetch_polygon_stocks(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = 'day'
    ) -> pd.DataFrame:
        """
        Fetch stock data from Polygon.io.

        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: 'minute', 'hour', 'day', 'week', 'month'

        Returns:
            DataFrame with OHLCV data
        """
        if not self.polygon_key:
            raise ValueError("Polygon.io API key required. Get free key at https://polygon.io/")

        print(f"[Polygon.io] Fetching {ticker}...")

        url = f"{self.polygon_url}/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
        params = {'apiKey': self.polygon_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data or not data['results']:
                raise ValueError(f"No data from Polygon.io for {ticker}")

            # Parse results
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)

            # Rename columns
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['Source'] = 'Polygon.io'

            print(f"[OK] Retrieved {len(df)} rows from Polygon.io")
            return df

        except Exception as e:
            print(f"[ERROR] Polygon.io error: {str(e)}")
            raise

    # ===========================
    # COINGECKO COIN ID MAPPER
    # ===========================
    @staticmethod
    def get_coingecko_id(symbol: str) -> str:
        """
        Map common crypto symbols to CoinGecko IDs.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')

        Returns:
            CoinGecko coin ID
        """
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'MATIC': 'matic-network',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'AVAX': 'avalanche-2',
            'ATOM': 'cosmos',
            'AAVE': 'aave',
            'MKR': 'maker',
            'COMP': 'compound-governance-token'
        }

        return mapping.get(symbol.upper(), symbol.lower())

    # ===========================
    # AUTO-SELECT BEST SOURCE
    # ===========================
    def fetch_auto(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        prefer_source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Automatically fetch from the best available source.

        Tries sources in order: Yahoo -> Alpha Vantage -> CoinGecko -> Polygon

        Args:
            ticker: Ticker/symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            prefer_source: Preferred source to try first

        Returns:
            DataFrame with data from first successful source
        """
        print(f"\n[AUTO] Fetching {ticker} from best available source...")

        sources = []

        # Determine if crypto
        is_crypto = '-USD' in ticker or ticker.upper() in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE']

        if is_crypto:
            # Try CoinGecko first (free, reliable for crypto)
            sources = ['coingecko', 'yahoo', 'alpha_vantage']
        else:
            # For stocks, try Yahoo first
            sources = ['yahoo', 'alpha_vantage', 'polygon']

        # Try preferred source first
        if prefer_source:
            sources = [prefer_source] + [s for s in sources if s != prefer_source]

        for source in sources:
            try:
                if source == 'yahoo':
                    return self.fetch_yahoo(ticker, start_date, end_date)

                elif source == 'alpha_vantage' and self.alpha_vantage_key:
                    if is_crypto:
                        symbol = ticker.replace('-USD', '')
                        return self.fetch_alpha_vantage_crypto(symbol)
                    else:
                        return self.fetch_alpha_vantage_daily(ticker)

                elif source == 'coingecko' and is_crypto:
                    symbol = ticker.replace('-USD', '')
                    coin_id = self.get_coingecko_id(symbol)
                    # Calculate days
                    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
                    return self.fetch_coingecko(coin_id, days=min(days, 365))

                elif source == 'polygon' and self.polygon_key:
                    return self.fetch_polygon_stocks(ticker, start_date, end_date)

            except Exception as e:
                print(f"[WARN] {source} failed: {str(e)}")
                continue

        raise ValueError(f"Failed to fetch data from any source for {ticker}")


def main():
    """
    Example usage of MultiSourceDataFetcher.
    """
    print("="*80)
    print("MULTI-SOURCE DATA FETCHER - EXAMPLES")
    print("="*80)

    fetcher = MultiSourceDataFetcher()

    # Example 1: Yahoo Finance (default)
    print("\n\nExample 1: Yahoo Finance Stock Data")
    print("-" * 60)
    try:
        data = fetcher.fetch_yahoo('AAPL', '2024-01-01', '2024-02-01')
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: CoinGecko (no API key needed)
    print("\n\nExample 2: CoinGecko Crypto Data")
    print("-" * 60)
    try:
        data = fetcher.fetch_coingecko('bitcoin', days=30)
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Auto-select best source
    print("\n\nExample 3: Auto-select Source (BTC)")
    print("-" * 60)
    try:
        data = fetcher.fetch_auto('BTC-USD', '2024-01-01', '2024-02-01')
        print(data.head())
        print(f"\nData source used: {data['Source'].iloc[0]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
