"""
Data fetching module for stocks and cryptocurrencies.
Downloads OHLC data from Yahoo Finance using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Union
import os


class DataFetcher:
    """
    Fetches historical OHLC data for stocks and cryptocurrencies.

    Attributes:
        tickers (List[str]): List of ticker symbols to fetch
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD)
    """

    def __init__(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize DataFetcher.

        Args:
            tickers: Single ticker or list of tickers (e.g., 'AAPL' or ['AAPL', 'BTC-USD'])
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
        """
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        self.start_date = start_date
        self.end_date = end_date

    def fetch_single_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Fetch data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL', 'BTC-USD')

        Returns:
            DataFrame with OHLC data

        Raises:
            ValueError: If no data is retrieved
        """
        print(f"Fetching data for {ticker}...")

        try:
            # Download data
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )

            if data.empty:
                raise ValueError(f"No data retrieved for {ticker}")

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Add ticker column
            data['Ticker'] = ticker

            # Detect asset type
            if '-USD' in ticker or ticker in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA']:
                data['AssetType'] = 'crypto'
            else:
                data['AssetType'] = 'stock'

            print(f"[OK] Retrieved {len(data)} rows for {ticker}")
            return data

        except Exception as e:
            print(f"[ERROR] Error fetching {ticker}: {str(e)}")
            raise

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch data for all tickers.

        Returns:
            Combined DataFrame with all tickers
        """
        all_data = []

        for ticker in self.tickers:
            try:
                data = self.fetch_single_ticker(ticker)
                all_data.append(data)
            except Exception as e:
                print(f"Skipping {ticker} due to error: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data was successfully fetched for any ticker")

        # Combine all data
        combined = pd.concat(all_data)
        combined.sort_index(inplace=True)

        print(f"\n[OK] Total data fetched: {len(combined)} rows across {len(all_data)} assets")
        return combined

    def save_data(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save fetched data to CSV.

        Args:
            data: DataFrame to save
            output_path: Path to save CSV file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path)
        print(f"[OK] Data saved to {output_path}")

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"[OK] Loaded {len(data)} rows from {file_path}")
        return data

    def get_info(self, ticker: str) -> dict:
        """
        Get ticker information.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with ticker info
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Extract relevant info
            relevant_info = {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName', info.get('name', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }

            return relevant_info

        except Exception as e:
            print(f"Could not fetch info for {ticker}: {str(e)}")
            return {'symbol': ticker, 'error': str(e)}


def main():
    """
    Example usage of DataFetcher.
    """
    # Define assets
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    crypto = ['BTC-USD', 'ETH-USD']
    all_assets = stocks + crypto

    # Create fetcher
    fetcher = DataFetcher(
        tickers=all_assets,
        start_date='2020-01-01'
    )

    # Fetch data
    data = fetcher.fetch_all()

    # Save data
    fetcher.save_data(data, 'data/raw/historical_data.csv')

    # Display summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Total rows: {len(data)}")
    print(f"\nRows per asset:")
    print(data.groupby('Ticker').size())
    print(f"\nAsset types:")
    print(data.groupby('AssetType').size())
    print("\nFirst few rows:")
    print(data.head())


if __name__ == "__main__":
    main()
