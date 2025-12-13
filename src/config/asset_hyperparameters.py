"""
Asset-Specific Hyperparameters (ENHANCEMENT #7)

Different asset classes have different characteristics:
- Stocks: Lower volatility, slower trends
- Crypto: High volatility, rapid changes
- Forex: Very smooth, mean-reverting
- Commodities: Seasonal patterns, external factors

This module provides optimized hyperparameters for each asset class.
"""

import re
from typing import Dict, Any


class AssetHyperparameters:
    """
    ENHANCEMENT #7: Asset-specific hyperparameters

    Provides optimal hyperparameters based on asset class.
    """

    # Default hyperparameters (conservative, works for most assets)
    DEFAULT_PARAMS = {
        'lookback': 20,
        'cnn_channels': [32, 64, 32],
        'kernel_sizes': [3, 5, 7],
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 40,  # Reduced from 100 for faster training
        'confidence_threshold_percentile': 40,
    }

    # Stock-specific parameters (lower volatility, longer trends)
    STOCK_PARAMS = {
        'lookback': 30,  # Longer lookback for slower trends
        'cnn_channels': [32, 64, 32],
        'kernel_sizes': [5, 7, 9],  # Larger kernels for longer patterns
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.25,  # Less dropout (stocks more predictable)
        'learning_rate': 0.0008,  # Slightly lower LR
        'batch_size': 32,
        'epochs': 40,  # Reduced from 120 for faster training
        'confidence_threshold_percentile': 35,  # Lower threshold (more conservative)
    }

    # Crypto-specific parameters (high volatility, rapid changes)
    CRYPTO_PARAMS = {
        'lookback': 15,  # Shorter lookback (rapid changes)
        'cnn_channels': [64, 128, 64],  # Larger capacity for volatility
        'kernel_sizes': [3, 5, 7],  # Smaller kernels for fast patterns
        'lstm_hidden_size': 128,  # Larger LSTM for complexity
        'lstm_num_layers': 3,  # More layers
        'dropout': 0.4,  # Higher dropout (prevent overfitting to noise)
        'learning_rate': 0.001,
        'batch_size': 64,  # Larger batches (more data)
        'epochs': 30,  # Reduced from 80 for faster training
        'confidence_threshold_percentile': 50,  # Higher threshold (more selective)
    }

    # Forex-specific parameters (smooth, mean-reverting)
    FOREX_PARAMS = {
        'lookback': 25,
        'cnn_channels': [32, 48, 32],  # Smaller capacity (simpler patterns)
        'kernel_sizes': [5, 7, 9],  # Larger kernels for smooth trends
        'lstm_hidden_size': 48,
        'lstm_num_layers': 2,
        'dropout': 0.2,  # Low dropout (forex very smooth)
        'learning_rate': 0.0005,  # Lower LR (careful updates)
        'batch_size': 32,
        'epochs': 40,  # Reduced from 100 for faster training
        'confidence_threshold_percentile': 30,  # Very selective
    }

    # Commodity-specific parameters (seasonal patterns)
    COMMODITY_PARAMS = {
        'lookback': 40,  # Longer lookback for seasonal patterns
        'cnn_channels': [32, 64, 32],
        'kernel_sizes': [7, 11, 15],  # Very large kernels for seasonal patterns
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.0008,
        'batch_size': 32,
        'epochs': 40,  # Reduced from 100 for faster training
        'confidence_threshold_percentile': 40,
    }

    @staticmethod
    def detect_asset_class(ticker: str) -> str:
        """
        Detect asset class from ticker symbol.

        Args:
            ticker: Asset ticker (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')

        Returns:
            Asset class: 'stock', 'crypto', 'forex', or 'commodity'
        """
        ticker = ticker.upper()

        # Crypto detection
        if any(crypto in ticker for crypto in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'DOGE', 'XRP', 'LTC']):
            return 'crypto'

        # Forex detection
        if '=X' in ticker or re.match(r'^[A-Z]{6}=X$', ticker):
            return 'forex'

        # Commodity detection
        commodities = ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F']  # Gold, Silver, Oil, Gas, Copper
        if ticker in commodities or '=F' in ticker:
            return 'commodity'

        # Default: stock
        return 'stock'

    @classmethod
    def get_params(cls, ticker: str) -> Dict[str, Any]:
        """
        Get optimal hyperparameters for an asset.

        Args:
            ticker: Asset ticker

        Returns:
            Dictionary of hyperparameters
        """
        asset_class = cls.detect_asset_class(ticker)

        params_map = {
            'stock': cls.STOCK_PARAMS,
            'crypto': cls.CRYPTO_PARAMS,
            'forex': cls.FOREX_PARAMS,
            'commodity': cls.COMMODITY_PARAMS
        }

        params = params_map.get(asset_class, cls.DEFAULT_PARAMS).copy()

        return {
            'asset_class': asset_class,
            **params
        }

    @classmethod
    def print_params(cls, ticker: str):
        """Print hyperparameters for an asset."""
        params = cls.get_params(ticker)
        asset_class = params.pop('asset_class')

        print(f"\nHyperparameters for {ticker} ({asset_class.upper()}):")
        print("="*50)
        for key, value in params.items():
            print(f"  {key:30s}: {value}")


def test_asset_hyperparameters():
    """Test asset-specific hyperparameters."""
    print("="*60)
    print("ASSET-SPECIFIC HYPERPARAMETERS TEST")
    print("="*60)

    test_tickers = [
        'AAPL',       # Stock
        'BTC-USD',    # Crypto
        'EURUSD=X',   # Forex
        'GC=F',       # Commodity (Gold)
    ]

    for ticker in test_tickers:
        AssetHyperparameters.print_params(ticker)

    print("\n" + "="*60)
    print("ASSET CLASS DETECTION TEST")
    print("="*60)

    more_tickers = [
        ('MSFT', 'stock'),
        ('ETH-USD', 'crypto'),
        ('GBPUSD=X', 'forex'),
        ('CL=F', 'commodity'),
        ('TSLA', 'stock'),
        ('DOGE-USD', 'crypto'),
    ]

    print("\nAsset Class Detection:")
    for ticker, expected in more_tickers:
        detected = AssetHyperparameters.detect_asset_class(ticker)
        status = "OK" if detected == expected else "FAIL"
        print(f"  {ticker:15s} -> {detected:10s} (expected: {expected:10s}) [{status}]")

    print("\n[SUCCESS] Asset-specific hyperparameters ready!")


if __name__ == "__main__":
    test_asset_hyperparameters()
