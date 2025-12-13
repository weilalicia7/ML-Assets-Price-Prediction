"""
Commodity Features Module

Specialized features for commodities and commodity-related stocks:
- Oil & Gas (CL=F, NG=F, XOM, CVX)
- Gold/Silver (GC=F, SI=F, GLD)
- Supply/demand proxies
- Seasonal patterns
- Inventory indicators
- Geopolitical risk features
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CommodityFeatures:
    """
    Generate features specific to commodities and commodity stocks.

    Handles:
    - Energy: Oil (CL=F, USO), Natural Gas (NG=F, UNG)
    - Precious Metals: Gold (GC=F, GLD), Silver (SI=F, SLV)
    - Energy Stocks: XOM, CVX, COP, SLB
    - Mining Stocks: NEM, GOLD, FCX
    """

    # Commodity mappings
    COMMODITY_MAP = {
        # Direct commodities
        'CL=F': 'oil', 'USO': 'oil',
        'NG=F': 'natgas', 'UNG': 'natgas',
        'GC=F': 'gold', 'GLD': 'gold',
        'SI=F': 'silver', 'SLV': 'silver',
        # Energy stocks
        'XOM': 'oil_stock', 'CVX': 'oil_stock',
        'COP': 'oil_stock', 'SLB': 'oil_services',
        'HAL': 'oil_services', 'OXY': 'oil_stock',
        # Mining stocks
        'NEM': 'gold_mining', 'GOLD': 'gold_mining',
        'AEM': 'gold_mining', 'KGC': 'gold_mining',
        'FCX': 'copper_mining',
    }

    # Reference commodities
    COMMODITY_TICKERS = {
        'oil': 'CL=F',
        'natgas': 'NG=F',
        'gold': 'GC=F',
        'silver': 'SI=F',
        'copper': 'HG=F',
    }

    # Dollar index (impacts all commodities)
    DXY_TICKER = 'DX-Y.NYB'

    def __init__(self, lookback: int = 60):
        """
        Initialize commodity features generator.

        Args:
            lookback: Rolling window for calculations
        """
        self.lookback = lookback
        self.commodity_cache = {}

    def get_commodity_type(self, ticker: str) -> str:
        """Get commodity type for a ticker."""
        return self.COMMODITY_MAP.get(ticker, 'other')

    def fetch_commodity_data(self, commodity: str, period: str = '1y') -> pd.DataFrame:
        """Fetch commodity price data."""
        cache_key = f"{commodity}_{period}"
        if cache_key in self.commodity_cache:
            return self.commodity_cache[cache_key]

        try:
            ticker = self.COMMODITY_TICKERS.get(commodity, commodity)
            data = yf.download(ticker, period=period, progress=False)
            if len(data) > 0:
                self.commodity_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch {commodity} data: {e}")

        return pd.DataFrame()

    def fetch_dollar_index(self, period: str = '1y') -> pd.DataFrame:
        """Fetch US Dollar Index data."""
        cache_key = f"dxy_{period}"
        if cache_key in self.commodity_cache:
            return self.commodity_cache[cache_key]

        try:
            data = yf.download(self.DXY_TICKER, period=period, progress=False)
            if len(data) > 0:
                self.commodity_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch DXY data: {e}")

        return pd.DataFrame()

    def create_oil_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create oil-specific features.

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with oil features added
        """
        df = stock_data.copy()

        # Fetch oil data
        oil_data = self.fetch_commodity_data('oil')

        if len(oil_data) == 0:
            for col in ['oil_return_1d', 'oil_return_5d', 'oil_return_20d',
                       'oil_volatility', 'oil_correlation', 'oil_contango']:
                df[col] = 0.0
            return df

        # Align dates
        oil_close = oil_data['Close'].reindex(df.index, method='ffill')

        # Oil returns at different horizons
        df['oil_return_1d'] = oil_close.pct_change(1)
        df['oil_return_5d'] = oil_close.pct_change(5)
        df['oil_return_20d'] = oil_close.pct_change(20)

        # Oil volatility
        oil_returns = oil_close.pct_change()
        df['oil_volatility'] = oil_returns.rolling(20).std() * np.sqrt(252)

        # Stock-Oil correlation
        stock_returns = df['Close'].pct_change()
        df['oil_correlation'] = stock_returns.rolling(self.lookback).corr(oil_returns)

        # Contango/backwardation proxy (based on momentum)
        # Positive = contango (far months more expensive)
        # Negative = backwardation (near months more expensive)
        oil_ma_20 = oil_close.rolling(20).mean()
        oil_ma_60 = oil_close.rolling(60).mean()
        df['oil_contango'] = (oil_ma_60 - oil_ma_20) / oil_ma_20

        # Fill NaN values
        for col in ['oil_return_1d', 'oil_return_5d', 'oil_return_20d',
                   'oil_volatility', 'oil_correlation', 'oil_contango']:
            df[col] = df[col].fillna(0)

        return df

    def create_gold_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create gold-specific features.

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with gold features added
        """
        df = stock_data.copy()

        # Fetch gold data
        gold_data = self.fetch_commodity_data('gold')

        if len(gold_data) == 0:
            for col in ['gold_return_1d', 'gold_return_5d', 'gold_return_20d',
                       'gold_volatility', 'gold_correlation', 'gold_safe_haven']:
                df[col] = 0.0
            return df

        # Align dates
        gold_close = gold_data['Close'].reindex(df.index, method='ffill')

        # Gold returns at different horizons
        df['gold_return_1d'] = gold_close.pct_change(1)
        df['gold_return_5d'] = gold_close.pct_change(5)
        df['gold_return_20d'] = gold_close.pct_change(20)

        # Gold volatility
        gold_returns = gold_close.pct_change()
        df['gold_volatility'] = gold_returns.rolling(20).std() * np.sqrt(252)

        # Stock-Gold correlation
        stock_returns = df['Close'].pct_change()
        df['gold_correlation'] = stock_returns.rolling(self.lookback).corr(gold_returns)

        # Safe haven indicator (gold rises when stocks fall)
        # Negative correlation = strong safe haven
        try:
            spy_data = yf.download('SPY', period='1y', progress=False)
            if len(spy_data) > 0:
                spy_returns = spy_data['Close'].pct_change().reindex(df.index, method='ffill')
                gold_spy_corr = gold_returns.rolling(20).corr(spy_returns)
                df['gold_safe_haven'] = -gold_spy_corr  # Negative = more safe haven
            else:
                df['gold_safe_haven'] = 0.0
        except:
            df['gold_safe_haven'] = 0.0

        # Fill NaN values
        for col in ['gold_return_1d', 'gold_return_5d', 'gold_return_20d',
                   'gold_volatility', 'gold_correlation', 'gold_safe_haven']:
            df[col] = df[col].fillna(0)

        return df

    def create_dollar_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create US Dollar features (inversely correlated with commodities).

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with dollar features added
        """
        df = stock_data.copy()

        # Fetch dollar index
        dxy_data = self.fetch_dollar_index()

        if len(dxy_data) == 0:
            for col in ['dxy_return_1d', 'dxy_return_5d', 'dxy_momentum',
                       'dxy_correlation', 'dxy_strength']:
                df[col] = 0.0
            return df

        # Align dates
        dxy_close = dxy_data['Close'].reindex(df.index, method='ffill')

        # DXY returns
        df['dxy_return_1d'] = dxy_close.pct_change(1)
        df['dxy_return_5d'] = dxy_close.pct_change(5)

        # DXY momentum
        dxy_ma_20 = dxy_close.rolling(20).mean()
        dxy_ma_50 = dxy_close.rolling(50).mean()
        df['dxy_momentum'] = (dxy_ma_20 - dxy_ma_50) / dxy_ma_50

        # Stock-DXY correlation (usually negative for commodities)
        stock_returns = df['Close'].pct_change()
        dxy_returns = dxy_close.pct_change()
        df['dxy_correlation'] = stock_returns.rolling(self.lookback).corr(dxy_returns)

        # Dollar strength regime
        dxy_52w_high = dxy_close.rolling(252).max()
        dxy_52w_low = dxy_close.rolling(252).min()
        df['dxy_strength'] = (dxy_close - dxy_52w_low) / (dxy_52w_high - dxy_52w_low + 0.01)

        # Fill NaN values
        for col in ['dxy_return_1d', 'dxy_return_5d', 'dxy_momentum',
                   'dxy_correlation', 'dxy_strength']:
            df[col] = df[col].fillna(0)

        return df

    def create_seasonal_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create seasonal pattern features.

        Important for:
        - Natural gas (heating/cooling seasons)
        - Oil (driving season, refinery maintenance)
        - Agriculture (planting/harvest)

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with seasonal features added
        """
        df = stock_data.copy()

        # Get date components
        dates = pd.to_datetime(df.index)
        df['month'] = dates.month
        df['day_of_year'] = dates.dayofyear

        # Seasonal factors
        # Natural gas: high demand Dec-Feb (heating), Jun-Aug (cooling)
        natgas_season = np.where(
            df['month'].isin([12, 1, 2]), 1.0,  # Winter peak
            np.where(df['month'].isin([6, 7, 8]), 0.5,  # Summer cooling
                    0.0)  # Shoulder season
        )
        df['natgas_seasonal'] = natgas_season

        # Oil: high demand May-Sep (driving season)
        oil_season = np.where(
            df['month'].isin([5, 6, 7, 8, 9]), 1.0,  # Driving season
            np.where(df['month'].isin([3, 4, 10]), 0.5,  # Shoulder
                    0.0)  # Winter (refinery maintenance)
        )
        df['oil_seasonal'] = oil_season

        # Gold: often strong Sep-Feb (wedding/festival season in Asia)
        gold_season = np.where(
            df['month'].isin([9, 10, 11, 12, 1, 2]), 1.0,  # Strong season
            0.0  # Weak season
        )
        df['gold_seasonal'] = gold_season

        # Quarterly effects (inventory reports, earnings)
        quarter_end = df['month'].isin([3, 6, 9, 12])
        df['quarter_end'] = quarter_end.astype(int)

        # Day of year normalized (0-1 for cyclical pattern)
        df['day_normalized'] = df['day_of_year'] / 365.0

        # Sine/cosine encoding for cyclical patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Clean up
        df = df.drop(columns=['month', 'day_of_year'], errors='ignore')

        return df

    def create_supply_demand_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create supply/demand proxy features.

        Note: Real inventory data requires EIA API or similar.
        These are proxy features based on price patterns.

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with supply/demand features added
        """
        df = stock_data.copy()

        returns = df['Close'].pct_change()
        volume = df['Volume']

        # Volume-price divergence (supply/demand imbalance)
        price_ma = df['Close'].rolling(20).mean()
        volume_ma = volume.rolling(20).mean()

        price_trend = (df['Close'] - price_ma) / price_ma
        volume_trend = (volume - volume_ma) / volume_ma

        # High volume + falling price = oversupply
        # High volume + rising price = excess demand
        df['supply_demand_signal'] = price_trend * volume_trend.abs()

        # Inventory proxy: based on term structure
        # Backwardation (falling prices over time) = tight supply
        # Contango (rising prices over time) = excess supply
        ma_short = df['Close'].rolling(5).mean()
        ma_long = df['Close'].rolling(60).mean()
        df['inventory_proxy'] = (ma_short - ma_long) / ma_long

        # Open interest proxy (based on volume patterns)
        volume_change = volume.pct_change(5)
        price_change = returns.rolling(5).sum()

        # Rising volume + rising price = new longs (bullish)
        # Rising volume + falling price = new shorts (bearish)
        df['positioning_signal'] = volume_change * np.sign(price_change)

        # Mean-reversion signal (commodities often mean-revert)
        zscore = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
        df['mean_reversion_signal'] = -zscore  # Negative = buy signal when oversold

        # Fill NaN values
        for col in ['supply_demand_signal', 'inventory_proxy',
                   'positioning_signal', 'mean_reversion_signal']:
            df[col] = df[col].fillna(0)

        return df

    def create_all_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create all commodity features for a ticker.

        Args:
            stock_data: Stock/commodity OHLCV data
            ticker: Ticker symbol

        Returns:
            DataFrame with all commodity features added
        """
        df = stock_data.copy()
        commodity_type = self.get_commodity_type(ticker)

        # Add oil features (for oil-related assets)
        if commodity_type in ['oil', 'oil_stock', 'oil_services', 'natgas', 'other']:
            df = self.create_oil_features(df, ticker)

        # Add gold features (for gold-related assets)
        if commodity_type in ['gold', 'gold_mining', 'silver', 'copper_mining', 'other']:
            df = self.create_gold_features(df, ticker)

        # Add dollar features (impacts all commodities)
        df = self.create_dollar_features(df, ticker)

        # Add seasonal features
        df = self.create_seasonal_features(df, ticker)

        # Add supply/demand features
        df = self.create_supply_demand_features(df, ticker)

        # Feature count
        commodity_cols = [col for col in df.columns if col not in stock_data.columns]
        print(f"[OK] Created {len(commodity_cols)} commodity features for {ticker}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all commodity feature names."""
        return [
            # Oil features
            'oil_return_1d', 'oil_return_5d', 'oil_return_20d',
            'oil_volatility', 'oil_correlation', 'oil_contango',
            # Gold features
            'gold_return_1d', 'gold_return_5d', 'gold_return_20d',
            'gold_volatility', 'gold_correlation', 'gold_safe_haven',
            # Dollar features
            'dxy_return_1d', 'dxy_return_5d', 'dxy_momentum',
            'dxy_correlation', 'dxy_strength',
            # Seasonal features
            'natgas_seasonal', 'oil_seasonal', 'gold_seasonal',
            'quarter_end', 'day_normalized', 'month_sin', 'month_cos',
            # Supply/demand features
            'supply_demand_signal', 'inventory_proxy',
            'positioning_signal', 'mean_reversion_signal'
        ]


def main():
    """Test commodity features."""
    print("=" * 60)
    print("COMMODITY FEATURES TEST")
    print("=" * 60)

    # Test with Gold ETF
    ticker = 'GLD'
    print(f"\n[INFO] Testing with {ticker}...")

    # Fetch data
    data = yf.download(ticker, period='1y', progress=False)
    print(f"[OK] Loaded {len(data)} days of data")

    # Create features
    commodity_features = CommodityFeatures()
    enhanced_data = commodity_features.create_all_features(data, ticker)

    # Show sample
    print(f"\nSample commodity features:")
    feature_cols = commodity_features.get_feature_names()
    available_cols = [col for col in feature_cols if col in enhanced_data.columns]
    print(enhanced_data[available_cols].tail(10))

    print("\n[SUCCESS] Commodity features test complete!")


if __name__ == "__main__":
    main()
