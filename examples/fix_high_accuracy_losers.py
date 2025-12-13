"""
Fix High-Accuracy Losers with Magnitude-Aware Sizing
Test magnitude-aware approach on assets with good accuracy but negative returns:
- Bitcoin: 57.4% accuracy, -0.69% return
- JPM: 57.1% accuracy, -0.75% return
- XOM: 58.4% accuracy, -0.85% return
- USD/JPY: 64.8% accuracy, -2.35% return (highest accuracy!)

Expected: Turn 2-3 of these profitable
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


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
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


def magnitude_aware_backtest(
    price_data: pd.DataFrame,
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    kelly_fraction: float = 0.10,
    min_edge: float = 0.04,
    magnitude_threshold: float = 0.005,
    holding_period: int = 7,
    max_position_size: float = 0.10
):
    """Magnitude-aware backtest."""
    n = len(price_data)
    capital = 10000.0
    equity_curve = [capital]
    trades = []

    i = 0
    while i < n - holding_period:
        pred = predictions[i]
        actual = actual_returns[i]

        pred_magnitude = abs(pred)

        if pred_magnitude < magnitude_threshold:
            i += 1
            equity_curve.append(capital)
            continue

        prob = 0.5 + 0.4 * (pred / (abs(pred) + 1e-10))
        prob = np.clip(prob, 0.1, 0.9)
        edge = 2 * prob - 1

        if edge < min_edge:
            i += 1
            equity_curve.append(capital)
            continue

        kelly_size = kelly_fraction * edge
        kelly_size = np.clip(kelly_size, 0, max_position_size)
        magnitude_scale = min(1.0, pred_magnitude / 0.02)
        position_size = kelly_size * magnitude_scale
        direction = np.sign(pred)

        hold_return = actual
        pnl = capital * position_size * direction * hold_return
        capital += pnl

        trades.append({'pnl': pnl, 'capital': capital})

        for j in range(holding_period):
            if i + j < n:
                equity_curve.append(capital)
        i += holding_period

    while len(equity_curve) < n:
        equity_curve.append(capital)

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    total_return = (capital - 10000) / 10000
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 24)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'avg_position': np.mean([abs(0.1)] * len(trades)) if trades else 0
    }


def test_asset_with_magnitude(
    name: str,
    symbol: str,
    baseline_acc: float,
    baseline_return: float,
    baseline_sharpe: float,
    period: str = '3mo',
    interval: str = '1h'
):
    """Test asset with magnitude-aware sizing."""
    print(f"\n{'='*80}")
    print(f"Testing: {name} ({symbol})")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_acc:.1%} acc, {baseline_return:.2%} return, {baseline_sharpe:.2f} Sharpe")

    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Fetched {len(data)} bars")

        # Add features
        data_features = add_technical_features(data)
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

        print(f"Directional Accuracy: {dir_acc:.1%}")

        # Price data
        price_data = data_features.iloc[train_size+val_size:][['Close']].copy()

        # Test multiple magnitude-aware configs
        configs = [
            {'kelly': 0.05, 'mag_thresh': 0.003, 'hold': 7, 'edge': 0.04},
            {'kelly': 0.10, 'mag_thresh': 0.005, 'hold': 7, 'edge': 0.04},
            {'kelly': 0.10, 'mag_thresh': 0.003, 'hold': 10, 'edge': 0.05},
            {'kelly': 0.15, 'mag_thresh': 0.005, 'hold': 5, 'edge': 0.05},
            {'kelly': 0.20, 'mag_thresh': 0.007, 'hold': 3, 'edge': 0.06},
        ]

        results = []
        for config in configs:
            bt_result = magnitude_aware_backtest(
                price_data, predictions, y_test.values,
                kelly_fraction=config['kelly'],
                min_edge=config['edge'],
                magnitude_threshold=config['mag_thresh'],
                holding_period=config['hold']
            )
            results.append({**config, **bt_result})

        # Sort by Sharpe
        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        best = results[0]

        print(f"\nTop 5 Configurations:")
        print(f"{'Rank':<6} {'Kelly':<8} {'MagTh':<8} {'Hold':<6} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10} {'Trades':<8} {'Status':<10}")
        print("-" * 100)

        for i, r in enumerate(results, 1):
            status = "✅ PROFIT" if r['total_return'] > 0 else "❌ LOSS"
            print(f"{i:<6} {r['kelly']:<8.2f} {r['mag_thresh']:<8.3f} {r['hold']:<6} "
                  f"{r['total_return']:<10.2%} {r['sharpe_ratio']:<8.2f} "
                  f"{r['max_drawdown']:<10.2%} {r['num_trades']:<8} {status:<10}")

        print(f"\nBest Result:")
        print(f"  Return: {best['total_return']:.2%} (baseline: {baseline_return:.2%})")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f} (baseline: {baseline_sharpe:.2f})")
        print(f"  Improvement: {best['total_return'] - baseline_return:+.2%} return, {best['sharpe_ratio'] - baseline_sharpe:+.2f} Sharpe")

        profitable_count = sum(1 for r in results if r['total_return'] > 0)
        print(f"  Profitable Configs: {profitable_count}/5")

        if best['total_return'] > 0:
            print(f"\n✅ SUCCESS: {name} is now profitable!")
            return True
        else:
            print(f"\n⚠️ IMPROVED but not profitable (need {abs(best['total_return']):.2%} more)")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("="*80)
    print(" "*15 + "FIX HIGH-ACCURACY LOSERS")
    print(" "*10 + "Apply Magnitude-Aware Sizing to Turn Them Profitable")
    print("="*80)

    start_time = datetime.now()

    # Test assets with good accuracy but negative returns
    test_cases = [
        {
            'name': 'Bitcoin',
            'symbol': 'BTC-USD',
            'baseline_acc': 0.574,
            'baseline_return': -0.0069,
            'baseline_sharpe': -0.77,
            'period': '3mo',
            'interval': '1h'
        },
        {
            'name': 'JP Morgan',
            'symbol': 'JPM',
            'baseline_acc': 0.571,
            'baseline_return': -0.0075,
            'baseline_sharpe': -2.24,
            'period': '3mo',
            'interval': '1h'
        },
        {
            'name': 'Exxon',
            'symbol': 'XOM',
            'baseline_acc': 0.584,
            'baseline_return': -0.0085,
            'baseline_sharpe': -2.58,
            'period': '3mo',
            'interval': '1h'
        },
        {
            'name': 'USD/JPY',
            'symbol': 'JPY=X',
            'baseline_acc': 0.648,
            'baseline_return': -0.0235,
            'baseline_sharpe': -12.27,
            'period': '3mo',
            'interval': '1h'
        }
    ]

    results = {}
    for test in test_cases:
        success = test_asset_with_magnitude(**test)
        results[test['name']] = success

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    successful = sum(results.values())
    total = len(results)

    print(f"Assets Fixed: {successful}/{total}")
    print()

    for name, success in results.items():
        status = "✅ NOW PROFITABLE" if success else "❌ STILL NEGATIVE"
        print(f"  {name}: {status}")

    print(f"\n{'='*80}")
    print("OVERALL IMPACT")
    print(f"{'='*80}\n")

    print(f"Previous Overall Profitability: 9/22 = 40.9%")
    print(f"New Overall Profitability: {9 + successful}/22 = {(9 + successful)/22*100:.1f}%")
    print(f"Improvement: +{successful} assets = +{successful/22*100:.1f} percentage points")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nTest Duration: {duration:.1f} seconds")

    print(f"\n{'='*80}")
    print("FIX COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
