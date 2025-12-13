"""
Crypto-Related Stock Features Module

Specialized features for crypto-correlated stocks (COIN, MSTR, etc.):
- Bitcoin/Ethereum correlation
- Crypto market sentiment proxies
- On-chain proxy metrics
- Crypto volatility features
- Event detection (halving, ETF news, etc.)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CryptoFeatures:
    """
    Generate features specific to crypto-related stocks.

    Handles:
    - Crypto exchanges (COIN)
    - Bitcoin treasury companies (MSTR, MARA, RIOT)
    - Bitcoin miners
    - Crypto-adjacent tech
    """

    # Crypto-correlated stocks
    CRYPTO_STOCKS = {
        'COIN': 'exchange',      # Coinbase
        'MSTR': 'treasury',      # MicroStrategy (BTC holder)
        'MARA': 'miner',         # Marathon Digital
        'RIOT': 'miner',         # Riot Platforms
        'CLSK': 'miner',         # CleanSpark
        'HUT': 'miner',          # Hut 8
        'BITF': 'miner',         # Bitfarms
        'CIFR': 'miner',         # Cipher Mining
        'SQ': 'payment',         # Block (Cash App Bitcoin)
        'PYPL': 'payment',       # PayPal (crypto trading)
        'HOOD': 'exchange',      # Robinhood (crypto trading)
    }

    # Crypto reference assets
    CRYPTO_TICKERS = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
    }

    def __init__(self, lookback: int = 60):
        """
        Initialize crypto features generator.

        Args:
            lookback: Rolling window for calculations
        """
        self.lookback = lookback
        self.crypto_cache = {}

    def get_stock_type(self, ticker: str) -> str:
        """Get crypto stock type."""
        return self.CRYPTO_STOCKS.get(ticker, 'other')

    def fetch_crypto_data(self, crypto: str = 'BTC', period: str = '1y') -> pd.DataFrame:
        """Fetch cryptocurrency price data."""
        cache_key = f"{crypto}_{period}"
        if cache_key in self.crypto_cache:
            return self.crypto_cache[cache_key]

        try:
            ticker = self.CRYPTO_TICKERS.get(crypto, f'{crypto}-USD')
            data = yf.download(ticker, period=period, progress=False)
            if len(data) > 0:
                self.crypto_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch {crypto} data: {e}")

        return pd.DataFrame()

    def create_bitcoin_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create Bitcoin correlation features.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with Bitcoin features added
        """
        df = stock_data.copy()

        # Fetch Bitcoin data
        btc_data = self.fetch_crypto_data('BTC')

        if len(btc_data) == 0:
            # Return zeros if no BTC data
            for col in ['btc_return_1d', 'btc_return_5d', 'btc_return_20d',
                       'btc_volatility', 'btc_correlation', 'btc_beta',
                       'btc_momentum', 'btc_ma_cross']:
                df[col] = 0.0
            return df

        # Align dates
        btc_close = btc_data['Close'].reindex(df.index, method='ffill')

        # BTC returns at different horizons
        df['btc_return_1d'] = btc_close.pct_change(1)
        df['btc_return_5d'] = btc_close.pct_change(5)
        df['btc_return_20d'] = btc_close.pct_change(20)

        # BTC volatility
        btc_returns = btc_close.pct_change()
        df['btc_volatility'] = btc_returns.rolling(20).std() * np.sqrt(365)  # Annualized

        # Stock-BTC correlation
        stock_returns = df['Close'].pct_change()
        df['btc_correlation'] = stock_returns.rolling(self.lookback).corr(btc_returns)

        # Beta to Bitcoin
        covariance = stock_returns.rolling(self.lookback).cov(btc_returns)
        variance = btc_returns.rolling(self.lookback).var()
        df['btc_beta'] = covariance / variance
        df['btc_beta'] = df['btc_beta'].clip(-5, 5)  # Limit extreme values

        # BTC momentum
        btc_ma_20 = btc_close.rolling(20).mean()
        btc_ma_50 = btc_close.rolling(50).mean()
        df['btc_momentum'] = (btc_close - btc_ma_50) / btc_ma_50

        # BTC MA crossover signal
        df['btc_ma_cross'] = (btc_ma_20 > btc_ma_50).astype(int)

        # Fill NaN values
        for col in ['btc_return_1d', 'btc_return_5d', 'btc_return_20d',
                   'btc_volatility', 'btc_correlation', 'btc_beta',
                   'btc_momentum', 'btc_ma_cross']:
            df[col] = df[col].fillna(0)

        return df

    def create_ethereum_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create Ethereum correlation features.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with Ethereum features added
        """
        df = stock_data.copy()

        # Fetch Ethereum data
        eth_data = self.fetch_crypto_data('ETH')

        if len(eth_data) == 0:
            df['eth_return_1d'] = 0.0
            df['eth_correlation'] = 0.0
            df['eth_btc_ratio'] = 0.0
            return df

        # Align dates
        eth_close = eth_data['Close'].reindex(df.index, method='ffill')

        # ETH returns
        df['eth_return_1d'] = eth_close.pct_change(1)

        # Stock-ETH correlation
        stock_returns = df['Close'].pct_change()
        eth_returns = eth_close.pct_change()
        df['eth_correlation'] = stock_returns.rolling(self.lookback).corr(eth_returns)

        # ETH/BTC ratio (altcoin season indicator)
        btc_data = self.fetch_crypto_data('BTC')
        if len(btc_data) > 0:
            btc_close = btc_data['Close'].reindex(df.index, method='ffill')
            df['eth_btc_ratio'] = eth_close / btc_close
            df['eth_btc_ratio'] = df['eth_btc_ratio'].pct_change(20)  # 20-day change
        else:
            df['eth_btc_ratio'] = 0.0

        # Fill NaN values
        for col in ['eth_return_1d', 'eth_correlation', 'eth_btc_ratio']:
            df[col] = df[col].fillna(0)

        return df

    def create_sentiment_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create crypto sentiment proxy features.

        Note: Uses price/volume patterns as sentiment proxies
        (real sentiment would require external APIs).

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with sentiment features added
        """
        df = stock_data.copy()

        # Fetch Bitcoin data for crypto-wide sentiment
        btc_data = self.fetch_crypto_data('BTC')

        if len(btc_data) == 0:
            for col in ['crypto_fear_greed_proxy', 'crypto_momentum_rank',
                       'btc_drawdown', 'btc_recovery_rate']:
                df[col] = 0.0
            return df

        btc_close = btc_data['Close'].reindex(df.index, method='ffill')
        btc_returns = btc_close.pct_change()

        # Fear & Greed proxy: based on volatility and momentum
        btc_vol = btc_returns.rolling(14).std() * np.sqrt(365)
        btc_mom = btc_close.pct_change(14)

        # Normalize to 0-100 scale (like Fear & Greed Index)
        # High volatility + negative returns = Fear (0-50)
        # Low volatility + positive returns = Greed (50-100)
        vol_norm = 1 - (btc_vol / btc_vol.rolling(90).max()).clip(0, 1)
        mom_norm = ((btc_mom / btc_mom.rolling(90).std()) + 2) / 4  # Normalize around 0.5
        mom_norm = mom_norm.clip(0, 1)

        df['crypto_fear_greed_proxy'] = (vol_norm * 0.4 + mom_norm * 0.6) * 100

        # Crypto momentum rank (where are we in the cycle)
        btc_52w_high = btc_close.rolling(252).max()
        btc_52w_low = btc_close.rolling(252).min()
        df['crypto_momentum_rank'] = (btc_close - btc_52w_low) / (btc_52w_high - btc_52w_low + 0.01)

        # Bitcoin drawdown from ATH
        btc_ath = btc_close.cummax()
        df['btc_drawdown'] = (btc_close - btc_ath) / btc_ath

        # Recovery rate from recent low
        btc_recent_low = btc_close.rolling(30).min()
        df['btc_recovery_rate'] = (btc_close - btc_recent_low) / btc_recent_low

        # Fill NaN values
        for col in ['crypto_fear_greed_proxy', 'crypto_momentum_rank',
                   'btc_drawdown', 'btc_recovery_rate']:
            df[col] = df[col].fillna(0)

        return df

    def create_volatility_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create crypto-specific volatility features.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with volatility features added
        """
        df = stock_data.copy()

        stock_returns = df['Close'].pct_change()
        stock_vol = stock_returns.rolling(20).std()

        # Fetch Bitcoin for comparison
        btc_data = self.fetch_crypto_data('BTC')

        if len(btc_data) > 0:
            btc_close = btc_data['Close'].reindex(df.index, method='ffill')
            btc_returns = btc_close.pct_change()
            btc_vol = btc_returns.rolling(20).std()

            # Relative volatility to Bitcoin
            df['vol_vs_btc'] = stock_vol / btc_vol
            df['vol_vs_btc'] = df['vol_vs_btc'].clip(0, 5)

            # Volatility regime: is crypto in high or low vol period?
            btc_vol_percentile = btc_vol.rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001) if len(x) > 1 else 0.5,
                raw=False
            )
            df['crypto_vol_regime'] = btc_vol_percentile
        else:
            df['vol_vs_btc'] = 1.0
            df['crypto_vol_regime'] = 0.5

        # Stock-specific volatility clustering
        df['vol_cluster'] = stock_vol / stock_vol.rolling(60).mean()

        # Weekend effect proxy (crypto trades 24/7)
        # Higher Monday volatility often reflects weekend crypto moves
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        monday_vol = stock_returns.abs() * df['is_monday']
        df['monday_vol_ratio'] = monday_vol.rolling(20).mean() / stock_returns.abs().rolling(20).mean()

        # Clean up
        df = df.drop(columns=['day_of_week', 'is_monday'], errors='ignore')

        # Fill NaN values
        for col in ['vol_vs_btc', 'crypto_vol_regime', 'vol_cluster', 'monday_vol_ratio']:
            df[col] = df[col].fillna(0)

        return df

    def create_onchain_proxy_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create on-chain metric proxy features.

        Note: Uses price patterns as proxies for on-chain metrics
        (real on-chain data would require blockchain APIs).

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with on-chain proxy features added
        """
        df = stock_data.copy()

        # Fetch Bitcoin data
        btc_data = self.fetch_crypto_data('BTC')

        if len(btc_data) == 0:
            for col in ['hodl_proxy', 'whale_activity_proxy', 'network_growth_proxy']:
                df[col] = 0.0
            return df

        btc_close = btc_data['Close'].reindex(df.index, method='ffill')
        btc_volume = btc_data['Volume'].reindex(df.index, method='ffill')

        # HODL proxy: long-term holder behavior
        # Based on realized cap approximation (using volume-weighted price)
        vwap_90 = (btc_close * btc_volume).rolling(90).sum() / btc_volume.rolling(90).sum()
        df['hodl_proxy'] = (btc_close - vwap_90) / vwap_90

        # Whale activity proxy: large volume days
        volume_ma = btc_volume.rolling(20).mean()
        volume_std = btc_volume.rolling(20).std()
        whale_threshold = volume_ma + 2 * volume_std
        df['whale_activity_proxy'] = (btc_volume > whale_threshold).rolling(10).mean()

        # Network growth proxy: new ATH attempts (proxy for new adoption)
        btc_ath = btc_close.cummax()
        ath_distance = (btc_close / btc_ath)
        df['network_growth_proxy'] = (ath_distance > 0.95).rolling(30).mean()

        # Fill NaN values
        for col in ['hodl_proxy', 'whale_activity_proxy', 'network_growth_proxy']:
            df[col] = df[col].fillna(0)

        return df

    def create_all_features(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create all crypto features for a stock.

        Args:
            stock_data: Stock OHLCV data
            ticker: Stock ticker

        Returns:
            DataFrame with all crypto features added
        """
        df = stock_data.copy()

        # Add Bitcoin features
        df = self.create_bitcoin_features(df, ticker)

        # Add Ethereum features
        df = self.create_ethereum_features(df, ticker)

        # Add sentiment features
        df = self.create_sentiment_features(df, ticker)

        # Add volatility features
        df = self.create_volatility_features(df, ticker)

        # Add on-chain proxy features
        df = self.create_onchain_proxy_features(df, ticker)

        # Feature count
        crypto_cols = [col for col in df.columns if col not in stock_data.columns]
        print(f"[OK] Created {len(crypto_cols)} crypto features for {ticker}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all crypto feature names."""
        return [
            # Bitcoin features
            'btc_return_1d', 'btc_return_5d', 'btc_return_20d',
            'btc_volatility', 'btc_correlation', 'btc_beta',
            'btc_momentum', 'btc_ma_cross',
            # Ethereum features
            'eth_return_1d', 'eth_correlation', 'eth_btc_ratio',
            # Sentiment features
            'crypto_fear_greed_proxy', 'crypto_momentum_rank',
            'btc_drawdown', 'btc_recovery_rate',
            # Volatility features
            'vol_vs_btc', 'crypto_vol_regime', 'vol_cluster', 'monday_vol_ratio',
            # On-chain proxy features
            'hodl_proxy', 'whale_activity_proxy', 'network_growth_proxy'
        ]


def main():
    """Test crypto features."""
    print("=" * 60)
    print("CRYPTO FEATURES TEST")
    print("=" * 60)

    # Test with Coinbase
    ticker = 'COIN'
    print(f"\n[INFO] Testing with {ticker}...")

    # Fetch data
    data = yf.download(ticker, period='1y', progress=False)
    print(f"[OK] Loaded {len(data)} days of data")

    # Create features
    crypto_features = CryptoFeatures()
    enhanced_data = crypto_features.create_all_features(data, ticker)

    # Show sample
    print(f"\nSample crypto features:")
    feature_cols = crypto_features.get_feature_names()
    available_cols = [col for col in feature_cols if col in enhanced_data.columns]
    print(enhanced_data[available_cols].tail(10))

    print("\n[SUCCESS] Crypto features test complete!")


if __name__ == "__main__":
    main()
