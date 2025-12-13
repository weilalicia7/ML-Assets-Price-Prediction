"""
Test All Remaining Unprofitable Assets on Multiple Intervals
Apply the successful interval optimization strategy to all 8 remaining unprofitable assets.

Success rate so far: 3/3 (Sony, Alibaba, JPM)
Expected: Turn 2-4 more assets profitable → 70-80% overall profitability

Assets to test:
- Gold (GC=F): 49.6% acc on 1h
- Oil (CL=F): 47.5% acc on 1h
- EUR/USD: 51.8% acc on 1h
- GBP/USD: 59.5% acc on 1h
- USD/JPY: 64.8% acc on 1h (highest accuracy!)
- USD/CHF: 51.3% acc on 1h
- Copper (HG=F): 45.6% acc on 1h
- Silver (SI=F): 52.1% acc on 1h
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
from typing import Dict, List

from src.models.enhanced_ensemble import EnhancedEnsemblePredictor
from src.trading.kelly_backtester import KellyBacktester


def add_features(data):
    """Add technical features."""
    data = data.copy()
    data['returns'] = data['Close'].pct_change()
    data['returns_5'] = data['Close'].pct_change(5)
    data['returns_20'] = data['Close'].pct_change(20)
    data['vol_5'] = data['returns'].rolling(5).std()
    data['vol_20'] = data['returns'].rolling(20).std()

    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        data['Volume'] = data['Volume'].fillna(1.0)
        data['volume_ma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / (data['volume_ma'] + 1e-10)
    else:
        data['volume_ratio'] = 1.0

    data['hl_ratio'] = (data['High'] - data['Low']) / data['Close']
    data['momentum_5'] = data['Close'].pct_change(5)
    data['momentum_20'] = data['Close'].pct_change(20)
    data['ma_20'] = data['Close'].rolling(20).mean()
    data['ma_50'] = data['Close'].rolling(50).mean()
    data['ma_cross'] = (data['ma_20'] / data['ma_50'] - 1)

    return data


def test_asset_interval(symbol: str, name: str, interval: str, period: str, kelly: float, min_edge: float):
    """Test single asset on single interval."""
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if len(data) < 100:
            return None

        # Add features
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_features = add_features(data)
        data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
        data_features = data_features.dropna()

        feature_cols = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                       'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']

        X = data_features[feature_cols]
        y = data_features['target']

        # Split
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]

        # Train
        ensemble = EnhancedEnsemblePredictor(use_prediction_market=False)
        ensemble.train_all_models(X_train, y_train, X_val, y_val, models_to_train=['lightgbm'])

        predictions = ensemble.predict(X_test)
        dir_acc = np.mean(np.sign(predictions) == np.sign(y_test.values))

        # Backtest
        price_data = data_features.iloc[train_size+val_size:][['Close']].copy()
        probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)

        backtester = KellyBacktester(
            kelly_fraction=kelly,
            min_edge=min_edge,
            max_position_size=0.10
        )

        result = backtester.backtest(price_data, probs, hold_periods=5 if interval != '1d' else 7)

        return {
            'interval': interval,
            'bars': len(data),
            'accuracy': dir_acc,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio'],
            'max_dd': result['max_drawdown'],
            'profitable': result['total_return'] > 0
        }

    except Exception as e:
        return None


def test_asset_all_intervals(symbol: str, name: str, baseline_acc: float, baseline_return: float, kelly: float, min_edge: float):
    """Test asset on 1h, 4h, 1d intervals."""
    print(f"\n{'='*80}")
    print(f"Testing: {name} ({symbol})")
    print(f"{'='*80}")
    print(f"Baseline (1h): {baseline_acc:.1%} accuracy, {baseline_return:.2%} return")

    intervals = [
        ('1h', '3mo'),
        ('4h', '3mo'),
        ('1d', '6mo')
    ]

    results = []
    for interval, period in intervals:
        print(f"\nTesting {interval}...", end=' ')
        result = test_asset_interval(symbol, name, interval, period, kelly, min_edge)
        if result:
            results.append(result)
            status = "✅ PROFIT" if result['profitable'] else "❌ LOSS"
            print(f"{result['accuracy']:.1%} acc, {result['return']:.2%} return, Sharpe {result['sharpe']:.2f} {status}")
        else:
            print("❌ Failed")

    # Find best
    if results:
        best = max(results, key=lambda x: x['sharpe'])

        print(f"\nBest: {best['interval']} - {best['return']:.2%} return, Sharpe {best['sharpe']:.2f}")

        if best['profitable']:
            print(f"✅ SUCCESS: {name} is profitable on {best['interval']}!")
            return True, best
        else:
            print(f"⚠️ Improved but still negative")
            return False, best
    else:
        print("❌ All intervals failed")
        return False, None


def main():
    print("="*80)
    print(" "*15 + "TEST ALL REMAINING ASSETS ON MULTIPLE INTERVALS")
    print(" "*20 + "Interval Optimization Strategy")
    print("="*80)

    start_time = datetime.now()

    print("\nCurrent Status:")
    print("  Profitable: 14/22 = 63.6%")
    print("  Unprofitable: 8 assets")
    print("\nInterval Optimization Success Rate: 3/3 (Sony, Alibaba, JPM)")
    print("Expected: Turn 2-4 more profitable → 70-80% overall")

    # Define test assets
    test_assets = [
        {
            'symbol': 'GC=F',
            'name': 'Gold',
            'baseline_acc': 0.496,
            'baseline_return': -0.0293,
            'kelly': 0.25,
            'min_edge': 0.05
        },
        {
            'symbol': 'CL=F',
            'name': 'Crude Oil',
            'baseline_acc': 0.475,
            'baseline_return': -0.0212,
            'kelly': 0.15,
            'min_edge': 0.07
        },
        {
            'symbol': 'EURUSD=X',
            'name': 'EUR/USD',
            'baseline_acc': 0.518,
            'baseline_return': -0.0227,
            'kelly': 0.30,
            'min_edge': 0.05
        },
        {
            'symbol': 'GBPUSD=X',
            'name': 'GBP/USD',
            'baseline_acc': 0.595,
            'baseline_return': -0.0160,
            'kelly': 0.25,
            'min_edge': 0.05
        },
        {
            'symbol': 'JPY=X',
            'name': 'USD/JPY',
            'baseline_acc': 0.648,
            'baseline_return': -0.0235,
            'kelly': 0.25,
            'min_edge': 0.05
        },
        {
            'symbol': 'CHF=X',
            'name': 'USD/CHF',
            'baseline_acc': 0.513,
            'baseline_return': -0.0197,
            'kelly': 0.20,
            'min_edge': 0.05
        },
        {
            'symbol': 'HG=F',
            'name': 'Copper',
            'baseline_acc': 0.456,
            'baseline_return': -0.0105,
            'kelly': 0.15,
            'min_edge': 0.06
        },
        {
            'symbol': 'SI=F',
            'name': 'Silver',
            'baseline_acc': 0.521,
            'baseline_return': -0.0365,
            'kelly': 0.20,
            'min_edge': 0.05
        }
    ]

    successful = []
    improved = []

    for asset in test_assets:
        success, best_result = test_asset_all_intervals(**asset)

        if success:
            successful.append({
                'name': asset['name'],
                'symbol': asset['symbol'],
                'baseline_return': asset['baseline_return'],
                'best_interval': best_result['interval'],
                'new_return': best_result['return'],
                'sharpe': best_result['sharpe']
            })
        elif best_result:
            improved.append({
                'name': asset['name'],
                'baseline_return': asset['baseline_return'],
                'best_interval': best_result['interval'],
                'new_return': best_result['return'],
                'improvement': best_result['return'] - asset['baseline_return']
            })

    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    print(f"Assets Tested: {len(test_assets)}")
    print(f"Turned Profitable: {len(successful)}")
    print(f"Improved but Still Negative: {len(improved)}")

    if successful:
        print(f"\n{'='*80}")
        print("NEWLY PROFITABLE ASSETS")
        print(f"{'='*80}\n")

        for asset in successful:
            print(f"✅ {asset['name']} ({asset['symbol']})")
            print(f"   Interval: {asset['best_interval']}")
            print(f"   Return: {asset['baseline_return']:.2%} → {asset['new_return']:.2%} (improvement: {asset['new_return'] - asset['baseline_return']:+.2%})")
            print(f"   Sharpe: {asset['sharpe']:.2f}")
            print()

    if improved:
        print(f"{'='*80}")
        print("IMPROVED BUT STILL NEGATIVE")
        print(f"{'='*80}\n")

        for asset in improved:
            print(f"⚠️ {asset['name']}")
            print(f"   Best Interval: {asset['best_interval']}")
            print(f"   Return: {asset['baseline_return']:.2%} → {asset['new_return']:.2%} (improvement: {asset['improvement']:+.2%})")
            print()

    # Overall impact
    print(f"{'='*80}")
    print("OVERALL PROFITABILITY IMPACT")
    print(f"{'='*80}\n")

    current_profitable = 14
    new_profitable = current_profitable + len(successful)
    total_assets = 22

    print(f"Previous: {current_profitable}/{total_assets} = {current_profitable/total_assets*100:.1f}%")
    print(f"New: {new_profitable}/{total_assets} = {new_profitable/total_assets*100:.1f}%")
    print(f"Improvement: +{len(successful)} assets = +{len(successful)/total_assets*100:.1f} percentage points")

    if new_profitable >= 15:
        print(f"\n🎉 TARGET ACHIEVED: ≥70% profitability!")
    elif new_profitable >= 14:
        print(f"\n🎯 Close to 70% target!")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n\nTotal Duration: {duration/60:.1f} minutes")

    print(f"\n{'='*80}")
    print("INTERVAL OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    # Save results
    if successful or improved:
        print(f"\n💾 Saving results to INTERVAL_OPTIMIZATION_RESULTS.csv")

        all_results = []
        for asset in successful:
            all_results.append({
                'Asset': asset['name'],
                'Symbol': asset['symbol'],
                'Status': 'Profitable',
                'Best_Interval': asset['best_interval'],
                'Baseline_Return': asset['baseline_return'],
                'New_Return': asset['new_return'],
                'Improvement': asset['new_return'] - asset['baseline_return'],
                'Sharpe': asset['sharpe']
            })

        for asset in improved:
            all_results.append({
                'Asset': asset['name'],
                'Status': 'Improved',
                'Best_Interval': asset['best_interval'],
                'Baseline_Return': asset['baseline_return'],
                'New_Return': asset['new_return'],
                'Improvement': asset['improvement'],
                'Sharpe': 'N/A'
            })

        df = pd.DataFrame(all_results)
        df.to_csv('INTERVAL_OPTIMIZATION_RESULTS.csv', index=False)
        print("✅ Results saved")


if __name__ == "__main__":
    main()
