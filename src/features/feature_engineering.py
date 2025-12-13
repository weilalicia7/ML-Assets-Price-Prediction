"""
Feature Engineering Module

Provides unified interface for creating features from raw price data.
Generates 15 base features commonly used across all predictors.
"""

import pandas as pd
import numpy as np


def create_features(data):
    """
    Create features from raw OHLCV data.

    Args:
        data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
              with DatetimeIndex

    Returns:
        DataFrame with features and target variable:
        - Features: 15 technical indicators (price, volume, volatility, position)
        - Target: 'target' column (1 = next day up, 0 = next day down)
    """
    df = data.copy()

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")

    # Handle multi-level columns from yfinance (e.g., ('Close', 'AAPL'))
    # Convert DataFrame columns to Series if needed
    for col in required_cols:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    # =================================================================
    # BASE FEATURES (15 total)
    # =================================================================

    # 1-5: Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['close_vs_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-8)

    # 6-10: Moving averages and momentum
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    sma_5 = close.rolling(window=5).mean()
    sma_20 = close.rolling(window=20).mean()

    df['sma_5'] = sma_5
    df['sma_20'] = sma_20
    df['price_vs_sma5'] = close / (sma_5 + 1e-8)
    df['price_vs_sma20'] = close / (sma_20 + 1e-8)
    df['momentum_5'] = close / close.shift(5) - 1

    # 11-13: Volatility features
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)

    # 14-15: Volume features
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['volume_change'] = df['Volume'].pct_change()

    # =================================================================
    # TARGET VARIABLE
    # =================================================================
    # Predict next day's direction: 1 if Close tomorrow > Close today, else 0
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # =================================================================
    # CLEANUP
    # =================================================================
    # Drop rows with NaN values (from rolling windows and pct_change)
    df = df.dropna()

    # Feature columns (exclude OHLCV and intermediate calculations)
    feature_cols = [
        'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio', 'close_vs_high',
        'price_vs_sma5', 'price_vs_sma20', 'momentum_5',
        'volatility_5', 'volatility_20', 'volatility_ratio',
        'volume_ratio', 'volume_change',
        # Also include raw price position for context
        'sma_5', 'sma_20'
    ]

    # Return only feature columns + target
    return df[feature_cols + ['target']]


def create_features_for_prediction(data):
    """
    Create features for prediction (no target variable).

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with features only (no target column)
    """
    df = create_features(data)
    if 'target' in df.columns:
        df = df.drop('target', axis=1)
    return df
