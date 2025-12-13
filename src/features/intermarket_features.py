"""
Intermarket Correlation Features

Analyzes relationships between assets and major market indices/indicators:
- S&P 500 (SPY) - Overall market direction
- VIX - Volatility/fear gauge
- US Dollar (DXY) - Currency strength
- Treasury yields (TLT) - Interest rate proxy
- Gold (GLD) - Safe haven demand

Used to understand how assets behave relative to broader market conditions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class IntermarketAnalyzer:
    """
    Analyze correlations and relationships with major market indices.

    Indices tracked:
    - SPY: S&P 500 ETF (market beta)
    - ^VIX: CBOE Volatility Index (fear gauge)
    - DX-Y.NYB: US Dollar Index (currency strength)
    - TLT: 20+ Year Treasury ETF (interest rate proxy)
    - GLD: Gold ETF (safe haven)
    """

    def __init__(self, lookback_periods=[20, 60, 120]):
        """
        Initialize intermarket analyzer.

        Args:
            lookback_periods: Windows for rolling correlations and betas
        """
        self.lookback_periods = lookback_periods
        self.market_data = None
        self.indices = {
            'SPY': 'S&P 500',
            '^VIX': 'VIX',
            'DX-Y.NYB': 'DXY',
            'TLT': '20Y Treasury',
            'GLD': 'Gold'
        }

    def fetch_market_data(self, start_date='2020-01-01', end_date=None):
        """
        Fetch major market indices data.

        Args:
            start_date: Start date for data fetch
            end_date: End date (default: today)

        Returns:
            DataFrame with market indices data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[INFO] Fetching market indices data from {start_date} to {end_date}...")

        all_data = {}

        for ticker, name in self.indices.items():
            try:
                print(f"  Downloading {ticker} ({name})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if len(data) > 0:
                    all_data[ticker] = data['Close']
                else:
                    print(f"  [WARNING] No data for {ticker}")

            except Exception as e:
                print(f"  [ERROR] Failed to fetch {ticker}: {e}")

        if len(all_data) == 0:
            raise ValueError("Failed to fetch any market data!")

        # Combine into single DataFrame
        self.market_data = pd.DataFrame(all_data)

        print(f"[OK] Fetched {len(self.market_data)} days of market data")
        print(f"     Indices available: {list(self.market_data.columns)}")

        return self.market_data

    def align_data(self, df: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Align asset data with market data by date.

        Args:
            df: Asset DataFrame with datetime index
            market_data: Market indices DataFrame (optional, uses cached if None)

        Returns:
            Combined DataFrame with both asset and market data
        """
        if market_data is None:
            if self.market_data is None:
                raise ValueError("No market data available! Call fetch_market_data() first.")
            market_data = self.market_data

        # Merge on date index
        df = df.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        # Merge (left join to keep all asset dates)
        combined = df.join(market_data, how='left', rsuffix='_market')

        # Forward fill market data (for weekends/holidays)
        market_cols = [col for col in combined.columns if col in market_data.columns or '_market' in col]
        combined[market_cols] = combined[market_cols].fillna(method='ffill')

        return combined

    def calculate_rolling_correlation(self, asset_returns: pd.Series,
                                      market_returns: pd.Series,
                                      window: int) -> pd.Series:
        """
        Calculate rolling correlation between asset and market.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        corr = asset_returns.rolling(window=window, min_periods=int(window*0.7)).corr(market_returns)
        return corr

    def calculate_rolling_beta(self, asset_returns: pd.Series,
                               market_returns: pd.Series,
                               window: int) -> pd.Series:
        """
        Calculate rolling beta (sensitivity to market).

        Beta = Cov(asset, market) / Var(market)
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        Beta < 0: Inverse relationship

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window size

        Returns:
            Series of rolling betas
        """
        # Rolling covariance
        cov = asset_returns.rolling(window=window, min_periods=int(window*0.7)).cov(market_returns)

        # Rolling variance of market
        market_var = market_returns.rolling(window=window, min_periods=int(window*0.7)).var()

        # Beta
        beta = cov / (market_var + 1e-10)

        return beta

    def calculate_relative_strength(self, asset_price: pd.Series,
                                    market_price: pd.Series,
                                    window: int = 60) -> pd.Series:
        """
        Calculate relative strength (outperformance vs market).

        RS = (Asset Return - Market Return) over window
        Positive = outperforming, negative = underperforming

        Args:
            asset_price: Asset price series
            market_price: Market price series
            window: Lookback window

        Returns:
            Series of relative strength values
        """
        asset_return = asset_price.pct_change(window)
        market_return = market_price.pct_change(window)

        relative_strength = asset_return - market_return

        return relative_strength

    def get_correlation_features(self, df: pd.DataFrame, price_col='Close') -> pd.DataFrame:
        """
        Generate correlation-based features.

        Args:
            df: DataFrame with asset data and market data (aligned)
            price_col: Name of price column

        Returns:
            DataFrame with correlation features added
        """
        print("[INFO] Calculating correlation features...")
        df = df.copy()

        # Calculate returns
        asset_returns = df[price_col].pct_change()

        feature_count = 0

        # Correlations with each index
        for ticker in self.indices.keys():
            if ticker not in df.columns:
                print(f"  [WARNING] {ticker} not available in data, skipping...")
                continue

            market_returns = df[ticker].pct_change()

            # Rolling correlations at different windows
            for window in self.lookback_periods:
                col_name = f'corr_{ticker}_{window}d'
                df[col_name] = self.calculate_rolling_correlation(
                    asset_returns, market_returns, window
                )
                feature_count += 1

            # Rolling beta (for SPY only - primary market benchmark)
            if ticker == 'SPY':
                for window in self.lookback_periods:
                    col_name = f'beta_SPY_{window}d'
                    df[col_name] = self.calculate_rolling_beta(
                        asset_returns, market_returns, window
                    )
                    feature_count += 1

                # Relative strength vs SPY
                for window in [20, 60, 120]:
                    col_name = f'rel_strength_SPY_{window}d'
                    df[col_name] = self.calculate_relative_strength(
                        df[price_col], df[ticker], window
                    )
                    feature_count += 1

        # Correlation change (acceleration/deceleration)
        if 'corr_SPY_60d' in df.columns:
            df['corr_SPY_change'] = df['corr_SPY_60d'].diff(20)
            feature_count += 1

        # Beta stability (low std = stable beta)
        if 'beta_SPY_60d' in df.columns:
            df['beta_stability'] = 1 / (df['beta_SPY_60d'].rolling(60).std() + 1e-10)
            feature_count += 1

        print(f"[OK] Added {feature_count} correlation features")

        return df

    def get_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market regime features based on intermarket relationships.

        Regimes:
        - Risk-On: SPY up, VIX down, GLD flat/down
        - Risk-Off: SPY down, VIX up, GLD/TLT up
        - Currency Volatility: DXY high volatility
        - Defensive: TLT/GLD outperforming SPY

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with market regime features
        """
        print("[INFO] Calculating market regime features...")
        df = df.copy()

        feature_count = 0

        # SPY momentum (market direction)
        if 'SPY' in df.columns:
            for window in [5, 10, 20]:
                df[f'SPY_momentum_{window}d'] = df['SPY'].pct_change(window)
                feature_count += 1

            # SPY trend (positive/negative)
            df['SPY_trend'] = np.sign(df['SPY_momentum_20d'])
            feature_count += 1

        # VIX level and changes (fear gauge)
        if '^VIX' in df.columns:
            # VIX level categories
            df['VIX_level'] = df['^VIX']
            df['VIX_high'] = (df['^VIX'] > 25).astype(int)  # High fear
            df['VIX_extreme'] = (df['^VIX'] > 35).astype(int)  # Extreme fear
            feature_count += 3

            # VIX momentum
            for window in [5, 10, 20]:
                df[f'VIX_change_{window}d'] = df['^VIX'].pct_change(window)
                feature_count += 1

            # VIX spike detection
            vix_sma = df['^VIX'].rolling(window=20, min_periods=1).mean()
            df['VIX_spike'] = (df['^VIX'] > vix_sma * 1.5).astype(int)
            feature_count += 1

        # DXY (dollar strength)
        if 'DX-Y.NYB' in df.columns:
            for window in [10, 20, 60]:
                df[f'DXY_momentum_{window}d'] = df['DX-Y.NYB'].pct_change(window)
                feature_count += 1

            # DXY strength indicator
            df['DXY_strong'] = (df['DXY_momentum_60d'] > 0).astype(int)
            feature_count += 1

        # TLT momentum (interest rate direction)
        if 'TLT' in df.columns:
            for window in [10, 20, 60]:
                df[f'TLT_momentum_{window}d'] = df['TLT'].pct_change(window)
                feature_count += 1

            # TLT/SPY ratio (flight to safety indicator)
            if 'SPY' in df.columns:
                df['TLT_SPY_ratio'] = df['TLT'] / (df['SPY'] + 1e-10)
                df['TLT_SPY_ratio_change'] = df['TLT_SPY_ratio'].pct_change(20)
                feature_count += 2

        # GLD momentum (safe haven demand)
        if 'GLD' in df.columns:
            for window in [10, 20, 60]:
                df[f'GLD_momentum_{window}d'] = df['GLD'].pct_change(window)
                feature_count += 1

            # GLD strength
            df['GLD_strong'] = (df['GLD_momentum_60d'] > 0).astype(int)
            feature_count += 1

        # Risk-On/Risk-Off Classification
        risk_on_conditions = 0
        risk_off_conditions = 0

        if 'SPY_momentum_20d' in df.columns:
            risk_on_conditions += (df['SPY_momentum_20d'] > 0).astype(int)
            risk_off_conditions += (df['SPY_momentum_20d'] < 0).astype(int)

        if 'VIX_change_20d' in df.columns:
            risk_on_conditions += (df['VIX_change_20d'] < 0).astype(int)
            risk_off_conditions += (df['VIX_change_20d'] > 0).astype(int)

        if 'GLD_momentum_20d' in df.columns:
            risk_on_conditions += (df['GLD_momentum_20d'] < 0).astype(int)
            risk_off_conditions += (df['GLD_momentum_20d'] > 0).astype(int)

        if 'TLT_momentum_20d' in df.columns:
            risk_on_conditions += (df['TLT_momentum_20d'] < 0).astype(int)
            risk_off_conditions += (df['TLT_momentum_20d'] > 0).astype(int)

        # Risk regime score (-1 to +1)
        # Positive = Risk-On, Negative = Risk-Off
        df['risk_regime_score'] = (risk_on_conditions - risk_off_conditions) / 4.0
        feature_count += 1

        # Categorical regime
        df['market_regime'] = 'neutral'
        df.loc[df['risk_regime_score'] > 0.3, 'market_regime'] = 'risk_on'
        df.loc[df['risk_regime_score'] < -0.3, 'market_regime'] = 'risk_off'
        feature_count += 1

        print(f"[OK] Added {feature_count} market regime features")

        return df

    def get_all_features(self, df: pd.DataFrame, price_col='Close',
                        fetch_market_data=True) -> pd.DataFrame:
        """
        Generate all intermarket features.

        Args:
            df: Asset DataFrame
            price_col: Price column name
            fetch_market_data: Whether to fetch fresh market data

        Returns:
            DataFrame with all intermarket features
        """
        # Fetch market data if needed
        if fetch_market_data or self.market_data is None:
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            self.fetch_market_data(start_date, end_date)

        # Align data
        df = self.align_data(df)

        # Generate features
        df = self.get_correlation_features(df, price_col)
        df = self.get_market_regime_features(df)

        return df


class SectorAnalyzer:
    """
    Analyze sector-specific correlations and relative strength.

    Useful for understanding if a stock is outperforming its sector.
    """

    def __init__(self):
        """Initialize sector analyzer."""
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLC': 'Communication',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLB': 'Materials'
        }

    def fetch_sector_data(self, sector_ticker: str, start_date: str, end_date: str = None):
        """
        Fetch sector ETF data.

        Args:
            sector_ticker: Sector ETF ticker (e.g., 'XLK')
            start_date: Start date
            end_date: End date (default: today)

        Returns:
            DataFrame with sector data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[INFO] Fetching {sector_ticker} ({self.sector_etfs.get(sector_ticker, 'Unknown')})...")

        try:
            data = yf.download(sector_ticker, start=start_date, end=end_date, progress=False)
            return data['Close']
        except Exception as e:
            print(f"[ERROR] Failed to fetch {sector_ticker}: {e}")
            return None

    def get_sector_features(self, df: pd.DataFrame, sector_ticker: str,
                           price_col='Close') -> pd.DataFrame:
        """
        Generate sector-relative features.

        Args:
            df: Asset DataFrame
            sector_ticker: Sector ETF ticker
            price_col: Price column name

        Returns:
            DataFrame with sector features
        """
        print(f"[INFO] Calculating sector features for {sector_ticker}...")
        df = df.copy()

        # Fetch sector data
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        sector_data = self.fetch_sector_data(sector_ticker, start_date, end_date)

        if sector_data is None:
            print(f"[WARNING] No sector data available")
            return df

        # Align with asset data
        df[f'sector_{sector_ticker}'] = sector_data

        # Forward fill
        df[f'sector_{sector_ticker}'] = df[f'sector_{sector_ticker}'].fillna(method='ffill')

        # Relative strength vs sector
        for window in [20, 60]:
            asset_return = df[price_col].pct_change(window)
            sector_return = df[f'sector_{sector_ticker}'].pct_change(window)
            df[f'rel_strength_sector_{window}d'] = asset_return - sector_return

        # Correlation with sector
        asset_returns = df[price_col].pct_change()
        sector_returns = df[f'sector_{sector_ticker}'].pct_change()

        for window in [20, 60]:
            df[f'corr_sector_{window}d'] = asset_returns.rolling(
                window=window, min_periods=int(window*0.7)
            ).corr(sector_returns)

        print(f"[OK] Added 4 sector features")

        return df


def main():
    """Test intermarket features on sample data."""
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("INTERMARKET ANALYSIS TEST")
    print("="*60)

    # Fetch sample data
    print("\n[INFO] Fetching AAPL data...")
    fetcher = DataFetcher(['AAPL'], start_date='2023-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    print(f"[OK] Loaded {len(aapl)} days of data")

    # Test intermarket analyzer
    print("\n[1/2] Testing Intermarket Analyzer...")
    analyzer = IntermarketAnalyzer(lookback_periods=[20, 60, 120])
    aapl = analyzer.get_all_features(aapl)

    print(f"\nSample correlation features:")
    corr_cols = [col for col in aapl.columns if 'corr_' in col or 'beta_' in col][:5]
    if len(corr_cols) > 0:
        print(aapl[corr_cols].tail(10))

    print(f"\nMarket regime (last 10 days):")
    if 'market_regime' in aapl.columns:
        print(aapl['market_regime'].tail(10).value_counts())
        print(f"\nCurrent risk regime score: {aapl['risk_regime_score'].iloc[-1]:.2f}")
        print("  (+1 = Full Risk-On, -1 = Full Risk-Off)")

    # Test sector analyzer
    print("\n[2/2] Testing Sector Analyzer...")
    sector_analyzer = SectorAnalyzer()
    aapl = sector_analyzer.get_sector_features(aapl, 'XLK')  # Technology sector

    print(f"\nSample sector features:")
    sector_cols = [col for col in aapl.columns if 'sector' in col or 'rel_strength' in col]
    if len(sector_cols) > 0:
        print(aapl[sector_cols].tail(10))

    print("\n[SUCCESS] Intermarket analysis test complete!")

    # Count total features
    intermarket_cols = [col for col in aapl.columns if
                       'corr_' in col or 'beta_' in col or 'rel_strength' in col or
                       'SPY_' in col or 'VIX_' in col or 'DXY_' in col or
                       'TLT_' in col or 'GLD_' in col or 'risk_' in col or
                       'market_regime' in col or 'sector' in col]
    print(f"Total intermarket features: {len(intermarket_cols)}")


if __name__ == "__main__":
    main()
