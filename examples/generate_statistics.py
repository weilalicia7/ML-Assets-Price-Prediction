# examples/generate_statistics.py
"""
Generate comprehensive descriptive statistics for the dataset.

This script demonstrates:
1. Fetching historical market data
2. Computing basic descriptive statistics (mean, std, min, max)
3. Generating correlation matrices
4. Analyzing statistics by ticker
5. Engineering features and analyzing their distributions
"""

from src.data.fetch_data import DataFetcher
from src.features.feature_engineering import create_features
from src.evaluation.metrics import VolatilityMetrics
import pandas as pd
import numpy as np

# 1. Fetch dataset
print("Fetching dataset...")
fetcher = DataFetcher(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD'],
    start_date='2020-01-01'
)
data = fetcher.fetch_all()

# 2. Generate descriptive statistics
print("\n=== DESCRIPTIVE STATISTICS ===\n")

# Basic statistics
print("Dataset Shape:", data.shape)
print("\nSummary Statistics:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# Correlation matrix
print("\nCorrelation Matrix:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].corr())

# Statistics by ticker
print("\nStatistics by Ticker:")
print(data.groupby('Ticker').agg({
    'Close': ['mean', 'std', 'min', 'max'],
    'Volume': ['mean', 'std']
}))

# 3. Feature engineering and analysis
features = create_features(data)
print("\nEngineered Features Statistics:")
print(features.describe())

# 4. Volatility metrics
if len(features) > 100:
    # Split data for demonstration
    train_size = int(len(features) * 0.8)
    y_true = features['next_day_direction'].iloc[train_size:].values
    y_pred = features['next_day_direction'].iloc[train_size-1:-1].values  # Example

    metrics = VolatilityMetrics()
    print("\nVolatility Analysis Metrics:")
    stats = metrics.calculate_all_metrics(y_true[:len(y_pred)], y_pred)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

print("\n=== Statistics generation complete ===")
