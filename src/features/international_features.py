"""
International Stock Features Module

Specialized features for international stocks (ADRs, European, Asian):
- FX exposure features
- ADR premium/discount
- Home market correlation
- Geopolitical risk indicators
- Regional economic indicators
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class InternationalFeatures:
    """
    Generate features specific to international stocks.

    Handles:
    - Chinese ADRs (BABA, JD, PDD, etc.)
    - Japanese stocks (SONY, TM)
    - European stocks (SAP, ASML)
    - Taiwan (TSM)
    """

    # Currency mappings for major international stocks
    CURRENCY_MAP = {
        # Chinese ADRs
        'BABA': 'CNY', 'JD': 'CNY', 'PDD': 'CNY', 'NIO': 'CNY',
        'BIDU': 'CNY', 'BILI': 'CNY', 'LI': 'CNY', 'XPEV': 'CNY',
        # Japanese
        'SONY': 'JPY', 'TM': 'JPY', 'HMC': 'JPY', 'MUFG': 'JPY',
        # European
        'SAP': 'EUR', 'ASML': 'EUR', 'NVO': 'EUR', 'AZN': 'GBP',
        'BP': 'GBP', 'SHEL': 'GBP', 'UL': 'GBP',
        # Taiwan
        'TSM': 'TWD',
        # Korean
        'SSNLF': 'KRW', 'PKX': 'KRW',
    }

    # FX pairs to USD
    FX_PAIRS = {
        'CNY': 'CNYUSD=X',
        'JPY': 'JPYUSD=X',
        'EUR': 'EURUSD=X',
        'GBP': 'GBPUSD=X',
        'TWD': 'TWDUSD=X',
        'KRW': 'KRWUSD=X',
    }

    # Home market indices
    HOME_INDICES = {
        'CNY': '000001.SS',  # Shanghai Composite
        'JPY': '^N225',       # Nikkei 225
        'EUR': '^STOXX50E',   # Euro Stoxx 50
        'GBP': '^FTSE',       # FTSE 100
        'TWD': '^TWII',       # Taiwan Weighted
        'KRW': '^KS11',       # KOSPI
    }

    def __init__(self, lookback: int = 60):
        """
        Initialize international features generator.

        Args:
            lookback: Rolling window for calculations
        """
        self.lookback = lookback
        self.fx_cache = {}
        self.index_cache = {}

    def get_currency(self, ticker: str) -> str:
        """Get home currency for a ticker."""
        return self.CURRENCY_MAP.get(ticker, 'USD')

    def fetch_fx_data(self, currency: str, period: str = '1y') -> pd.DataFrame:
        """Fetch FX rate data."""
        if currency == 'USD' or currency not in self.FX_PAIRS:
            return pd.DataFrame()

        cache_key = f"{currency}_{period}"
        if cache_key in self.fx_cache:
            return self.fx_cache[cache_key]

        try:
            fx_ticker = self.FX_PAIRS[currency]
            data = yf.download(fx_ticker, period=period, progress=False)
            if len(data) > 0:
                self.fx_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch FX data for {currency}: {e}")

        return pd.DataFrame()

    def fetch_home_index(self, currency: str, period: str = '1y') -> pd.DataFrame:
        """Fetch home market index data."""
        if currency == 'USD' or currency not in self.HOME_INDICES:
            return pd.DataFrame()

        cache_key = f"idx_{currency}_{period}"
        if cache_key in self.index_cache:
            return self.index_cache[cache_key]

        try:
            idx_ticker = self.HOME_INDICES[currency]
            data = yf.download(idx_ticker, period=period, progress=False)
            if len(data) > 0:
                self.index_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch index data for {currency}: {e}")

        return pd.DataFrame()

    def create_fx_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create FX exposure features.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with FX features added
        """
        df = stock_data.copy()
        currency = self.get_currency(ticker)

        if currency == 'USD':
            # No FX exposure for US stocks
            df['fx_return_1d'] = 0.0
            df['fx_return_5d'] = 0.0
            df['fx_return_20d'] = 0.0
            df['fx_volatility'] = 0.0
            df['fx_momentum'] = 0.0
            df['fx_correlation'] = 0.0
            return df

        # Fetch FX data
        fx_data = self.fetch_fx_data(currency)

        if len(fx_data) == 0:
            # Return zeros if no FX data
            for col in ['fx_return_1d', 'fx_return_5d', 'fx_return_20d',
                       'fx_volatility', 'fx_momentum', 'fx_correlation']:
                df[col] = 0.0
            return df

        # Align dates
        fx_close = fx_data['Close'].reindex(df.index, method='ffill')

        # FX returns at different horizons
        df['fx_return_1d'] = fx_close.pct_change(1)
        df['fx_return_5d'] = fx_close.pct_change(5)
        df['fx_return_20d'] = fx_close.pct_change(20)

        # FX volatility
        df['fx_volatility'] = fx_close.pct_change().rolling(20).std()

        # FX momentum (trend direction)
        fx_ma_20 = fx_close.rolling(20).mean()
        fx_ma_60 = fx_close.rolling(60).mean()
        df['fx_momentum'] = (fx_ma_20 - fx_ma_60) / fx_ma_60

        # Stock-FX correlation
        stock_returns = df['Close'].pct_change()
        fx_returns = fx_close.pct_change()
        df['fx_correlation'] = stock_returns.rolling(self.lookback).corr(fx_returns)

        # Fill NaN values
        for col in ['fx_return_1d', 'fx_return_5d', 'fx_return_20d',
                   'fx_volatility', 'fx_momentum', 'fx_correlation']:
            df[col] = df[col].fillna(0)

        return df

    def create_home_market_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create home market correlation features.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with home market features added
        """
        df = stock_data.copy()
        currency = self.get_currency(ticker)

        # Fetch home market index
        index_data = self.fetch_home_index(currency)

        if len(index_data) == 0:
            df['home_index_return_1d'] = 0.0
            df['home_index_return_5d'] = 0.0
            df['home_index_correlation'] = 0.0
            df['home_index_beta'] = 1.0
            df['relative_strength'] = 0.0
            return df

        # Align dates
        idx_close = index_data['Close'].reindex(df.index, method='ffill')

        # Home index returns
        df['home_index_return_1d'] = idx_close.pct_change(1)
        df['home_index_return_5d'] = idx_close.pct_change(5)

        # Correlation with home index
        stock_returns = df['Close'].pct_change()
        idx_returns = idx_close.pct_change()
        df['home_index_correlation'] = stock_returns.rolling(self.lookback).corr(idx_returns)

        # Beta to home index
        covariance = stock_returns.rolling(self.lookback).cov(idx_returns)
        variance = idx_returns.rolling(self.lookback).var()
        df['home_index_beta'] = covariance / variance
        df['home_index_beta'] = df['home_index_beta'].clip(-3, 3)  # Limit extreme values

        # Relative strength vs home index
        stock_cum = (1 + stock_returns).rolling(20).apply(lambda x: x.prod(), raw=True) - 1
        idx_cum = (1 + idx_returns).rolling(20).apply(lambda x: x.prod(), raw=True) - 1
        df['relative_strength'] = stock_cum - idx_cum

        # Fill NaN values
        for col in ['home_index_return_1d', 'home_index_return_5d',
                   'home_index_correlation', 'home_index_beta', 'relative_strength']:
            df[col] = df[col].fillna(0)

        return df

    def create_adr_premium_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create ADR premium/discount features.

        Note: True ADR premium requires home market price data.
        This creates proxy features based on price behavior.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with ADR features added
        """
        df = stock_data.copy()

        # ADR premium proxy: deviation from fair value based on FX
        # True ADR premium = (ADR_price * FX_rate) / home_price - 1
        # Proxy: volatility premium and momentum divergence

        returns = df['Close'].pct_change()

        # Overnight gap (often reflects home market movement)
        df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Gap vs return correlation (higher for ADRs)
        df['gap_return_corr'] = df['overnight_gap'].rolling(self.lookback).corr(returns)

        # Intraday range relative to gap (price discovery)
        intraday_range = (df['High'] - df['Low']) / df['Open']
        df['range_vs_gap'] = intraday_range / (df['overnight_gap'].abs() + 0.001)
        df['range_vs_gap'] = df['range_vs_gap'].clip(0, 10)

        # Volume around market open (higher volume = more price discovery)
        volume_ma = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma

        # Fill NaN values
        for col in ['overnight_gap', 'gap_return_corr', 'range_vs_gap', 'volume_ratio']:
            df[col] = df[col].fillna(0)

        return df

    def create_geopolitical_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create geopolitical risk proxy features.

        Uses volatility clustering and regime changes as proxies
        for geopolitical events.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with geopolitical features added
        """
        df = stock_data.copy()

        returns = df['Close'].pct_change()

        # Volatility regime changes (proxy for event risk)
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        df['vol_regime_ratio'] = vol_20 / vol_60

        # Tail risk (frequency of large moves)
        large_move_threshold = returns.rolling(252).std() * 2
        df['tail_risk_freq'] = (returns.abs() > large_move_threshold).rolling(20).mean()

        # Downside skewness (geopolitical risk tends to be asymmetric)
        df['return_skewness'] = returns.rolling(self.lookback).skew()

        # Cross-asset contagion (correlation spike = systemic event)
        # Proxy: correlation with US market
        try:
            spy_data = yf.download('SPY', period='1y', progress=False)
            if len(spy_data) > 0:
                spy_returns = spy_data['Close'].pct_change().reindex(df.index, method='ffill')
                df['us_correlation'] = returns.rolling(20).corr(spy_returns)
                df['correlation_change'] = df['us_correlation'] - df['us_correlation'].rolling(60).mean()
            else:
                df['us_correlation'] = 0.5
                df['correlation_change'] = 0.0
        except:
            df['us_correlation'] = 0.5
            df['correlation_change'] = 0.0

        # Fill NaN values
        for col in ['vol_regime_ratio', 'tail_risk_freq', 'return_skewness',
                   'us_correlation', 'correlation_change']:
            df[col] = df[col].fillna(0)

        return df

    def create_all_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create all international features for a stock.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with all international features added
        """
        df = stock_data.copy()

        # Add FX features
        df = self.create_fx_features(df, ticker)

        # Add home market features
        df = self.create_home_market_features(df, ticker)

        # Add ADR features
        df = self.create_adr_premium_features(df, ticker)

        # Add geopolitical features
        df = self.create_geopolitical_features(df, ticker)

        # Feature count
        intl_cols = [col for col in df.columns if col not in stock_data.columns]
        print(f"[OK] Created {len(intl_cols)} international features for {ticker}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all international feature names."""
        return [
            # FX features
            'fx_return_1d', 'fx_return_5d', 'fx_return_20d',
            'fx_volatility', 'fx_momentum', 'fx_correlation',
            # Home market features
            'home_index_return_1d', 'home_index_return_5d',
            'home_index_correlation', 'home_index_beta', 'relative_strength',
            # ADR features
            'overnight_gap', 'gap_return_corr', 'range_vs_gap', 'volume_ratio',
            # Geopolitical features
            'vol_regime_ratio', 'tail_risk_freq', 'return_skewness',
            'us_correlation', 'correlation_change'
        ]


def main():
    """Test international features."""
    print("=" * 60)
    print("INTERNATIONAL FEATURES TEST")
    print("=" * 60)

    # Test with a Chinese ADR
    ticker = 'BABA'
    print(f"\n[INFO] Testing with {ticker}...")

    # Fetch data
    data = yf.download(ticker, period='1y', progress=False)
    print(f"[OK] Loaded {len(data)} days of data")

    # Create features
    intl_features = InternationalFeatures()
    enhanced_data = intl_features.create_all_features(data, ticker)

    # Show sample
    print(f"\nSample international features:")
    feature_cols = intl_features.get_feature_names()
    available_cols = [col for col in feature_cols if col in enhanced_data.columns]
    print(enhanced_data[available_cols].tail(10))

    print("\n[SUCCESS] International features test complete!")


if __name__ == "__main__":
    main()
