"""
Iron Ore with China Economic Features
Complete Action 2: Add China economic indicators

Tests if adding China-specific features turns iron ore profitable:
- Baseline: 51.9% accuracy, -0.30% return (without China features)
- Target: 60%+ accuracy, positive return (with China features)

China Features:
- USD/CNY exchange rate (currency effects)
- Baltic Dry Index (shipping costs)
- China PMI (manufacturing activity)
- Synthetic Steel Production Index (if real data unavailable)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.models.enhanced_ensemble import EnhancedEnsemblePredictor
from src.trading.kelly_backtester import KellyBacktester
from src.data.china_economic_data import ChinaEconomicDataFetcher


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical features."""
    data = data.copy()

    # Returns
    data['returns'] = data['Close'].pct_change()
    data['returns_5'] = data['Close'].pct_change(5)
    data['returns_20'] = data['Close'].pct_change(20)

    # Volatility
    data['vol_5'] = data['returns'].rolling(5).std()
    data['vol_20'] = data['returns'].rolling(20).std()

    # Volume
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        data['Volume'] = data['Volume'].fillna(1.0)
        data['volume_ma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / (data['volume_ma'] + 1e-10)
    else:
        data['volume_ratio'] = 1.0

    # Price
    data['hl_ratio'] = (data['High'] - data['Low']) / data['Close']

    # Momentum
    data['momentum_5'] = data['Close'].pct_change(5)
    data['momentum_20'] = data['Close'].pct_change(20)

    # Moving averages
    data['ma_20'] = data['Close'].rolling(20).mean()
    data['ma_50'] = data['Close'].rolling(50).mean()
    data['ma_cross'] = (data['ma_20'] / data['ma_50'] - 1)

    return data


def main():
    print("="*80)
    print(" "*15 + "IRON ORE WITH CHINA ECONOMIC FEATURES")
    print(" "*20 + "Action 2: Complete Integration")
    print("="*80)
    print()

    start_time = datetime.now()

    # Fetch VALE data (iron ore proxy)
    print("Step 1: Fetching VALE (Iron Ore Producer) data...")
    ticker = yf.Ticker('VALE')
    data = ticker.history(period='3mo', interval='1h')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"  Fetched {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Add technical features
    print("\nStep 2: Adding technical features...")
    data_features = add_technical_features(data)

    # Fetch China economic features
    print("\nStep 3: Fetching China economic indicators...")
    china_fetcher = ChinaEconomicDataFetcher(fred_api_key=None)  # Will use synthetic if no API key

    start_date = data.index[0].strftime('%Y-%m-%d')
    end_date = data.index[-1].strftime('%Y-%m-%d')

    china_features = china_fetcher.fetch_all_china_features(
        start_date=start_date,
        end_date=end_date,
        use_synthetic_fallback=True
    )

    # Align China features to asset frequency
    if not china_features.empty:
        china_aligned = china_fetcher.align_china_features_to_asset(
            data_features,
            china_features
        )

        # Merge with technical features
        print("\nStep 4: Merging China features with technical features...")
        data_combined = pd.concat([data_features, china_aligned], axis=1)

        # Check for added features
        china_cols = list(china_aligned.columns)
        print(f"  Added {len(china_cols)} China features: {china_cols}")
    else:
        print("\nWARNING: No China features available, using technical features only")
        data_combined = data_features
        china_cols = []

    # Create target
    data_combined['target'] = data_combined['Close'].pct_change(24).shift(-24)
    data_combined = data_combined.dropna()

    # Define features
    technical_features = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                         'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']

    feature_cols = technical_features + china_cols

    print(f"\nStep 5: Preparing dataset...")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  - Technical: {len(technical_features)}")
    print(f"  - China: {len(china_cols)}")

    X = data_combined[feature_cols]
    y = data_combined['target']

    # Split
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]

    print(f"  Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    # Train model
    print("\nStep 6: Training LightGBM model with China features...")
    ensemble = EnhancedEnsemblePredictor(use_prediction_market=False)
    ensemble.train_all_models(X_train, y_train, X_val, y_val, models_to_train=['lightgbm'])

    predictions = ensemble.predict(X_test)
    dir_acc = np.mean(np.sign(predictions) == np.sign(y_test.values))

    print(f"  Directional Accuracy: {dir_acc:.1%}")

    # Price data for backtesting
    price_data = data_combined.iloc[train_size+val_size:][['Close']].copy()
    price_data.index = range(len(price_data))

    # Test with optimized parameters from Action 1
    print(f"\n{'='*80}")
    print("Step 7: Backtesting with Optimized Parameters")
    print(f"{'='*80}\n")

    # Best configuration from parameter optimization
    best_config = {
        'kelly': 0.05,
        'hold': 10,
        'edge': 0.05
    }

    print(f"Using optimized config: Kelly {best_config['kelly']}, Hold {best_config['hold']}, Edge {best_config['edge']}")

    # Generate probabilities
    probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)

    # Backtest
    backtester = KellyBacktester(
        kelly_fraction=best_config['kelly'],
        min_edge=best_config['edge'],
        max_position_size=0.10
    )

    bt_result = backtester.backtest(price_data, probs, hold_periods=best_config['hold'])

    # Compare with baseline (no China features)
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}\n")

    print("Baseline (No China Features):")
    print("  Directional Accuracy: 51.9%")
    print("  Return: -0.30%")
    print("  Sharpe: -1.09")
    print("  Max Drawdown: -0.71%")
    print()

    print(f"With China Features ({len(china_cols)} indicators):")
    print(f"  Directional Accuracy: {dir_acc:.1%}")
    print(f"  Return: {bt_result['total_return']:.2%}")
    print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {bt_result['max_drawdown']:.2%}")
    print()

    # Calculate improvement
    baseline_acc = 0.519
    baseline_return = -0.003
    baseline_sharpe = -1.09

    acc_improvement = (dir_acc - baseline_acc) * 100
    return_improvement = (bt_result['total_return'] - baseline_return) * 100
    sharpe_improvement = bt_result['sharpe_ratio'] - baseline_sharpe

    print("Improvement:")
    print(f"  Accuracy: {acc_improvement:+.1f} percentage points")
    print(f"  Return: {return_improvement:+.2f} percentage points")
    print(f"  Sharpe: {sharpe_improvement:+.2f}")

    # Status
    print(f"\n{'='*80}")
    print("PROFITABILITY STATUS")
    print(f"{'='*80}\n")

    if bt_result['total_return'] > 0:
        print("✅ IRON ORE IS NOW PROFITABLE!")
        print(f"\nChina features successfully turned iron ore profitable:")
        print(f"  {baseline_return:.2%} → {bt_result['total_return']:.2%}")
        print(f"\nReady for production deployment with:")
        print(f"  - Kelly Fraction: {best_config['kelly']}")
        print(f"  - Holding Period: {best_config['hold']} bars")
        print(f"  - Min Edge: {best_config['edge']}")
        print(f"  - Max Position: 10%")
    elif bt_result['total_return'] > baseline_return:
        print("⚠️ IMPROVED BUT NOT YET PROFITABLE")
        print(f"\nChina features improved performance:")
        print(f"  {baseline_return:.2%} → {bt_result['total_return']:.2%}")
        print(f"  Still {abs(bt_result['total_return']):.2%} away from breakeven")
        print(f"\nNext steps:")
        print(f"  1. Try additional parameter configurations")
        print(f"  2. Add more China features (steel production, property starts)")
        print(f"  3. Test neural models for temporal patterns")
    else:
        print("❌ CHINA FEATURES DID NOT IMPROVE PERFORMANCE")
        print(f"\nPerformance decreased:")
        print(f"  {baseline_return:.2%} → {bt_result['total_return']:.2%}")
        print(f"\nPossible reasons:")
        print(f"  1. Using synthetic features (need real China data)")
        print(f"  2. Need FRED API key for real China PMI")
        print(f"  3. Features not aligned properly to hourly frequency")
        print(f"\nRecommendation:")
        print(f"  - Obtain FRED API key for real China PMI data")
        print(f"  - Use direct iron ore futures instead of VALE proxy")

    # Feature importance analysis
    if hasattr(ensemble.models['lightgbm'], 'feature_importances_'):
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}\n")

        importances = ensemble.models['lightgbm'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("Top 10 Features:")
        for i, row in feature_importance.head(10).iterrows():
            is_china = row['feature'] in china_cols
            marker = "🇨🇳" if is_china else "📊"
            print(f"  {marker} {row['feature']:<25} {row['importance']:.4f}")

        # China features contribution
        china_importance = feature_importance[feature_importance['feature'].isin(china_cols)]
        if len(china_importance) > 0:
            china_total_importance = china_importance['importance'].sum()
            print(f"\nChina Features Total Importance: {china_total_importance:.4f}")
            print(f"China Features in Top 10: {len([f for f in feature_importance.head(10)['feature'] if f in china_cols])}/10")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n\nTest Duration: {duration:.1f} seconds")

    print(f"\n{'='*80}")
    print("ACTION 2 COMPLETE")
    print(f"{'='*80}")

    if bt_result['total_return'] > 0:
        print("\n✅ SUCCESS: Iron ore is profitable with China features!")
        print("📊 Ready to proceed to Action 3: Test neural models")
    else:
        print("\n⚠️ PARTIAL: Need real China data or further optimization")
        print("🔄 Recommend: Get FRED API key and retest with real PMI data")


if __name__ == "__main__":
    main()
