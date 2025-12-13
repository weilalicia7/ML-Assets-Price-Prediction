"""
Comprehensive Regime Testing
Test all assets across all regimes with latest improvements:
- Stocks (US, Europe, Asia)
- Commodities (Gold, Oil, Iron Ore)
- Forex (Major pairs)
- Crypto (Bitcoin, Ethereum)

Applies magnitude-aware sizing where beneficial
Documents accuracy and profitability for each regime
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
from typing import Dict, List, Tuple

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

        # Predicted magnitude
        pred_magnitude = abs(pred)

        # Skip if magnitude too small
        if pred_magnitude < magnitude_threshold:
            i += 1
            equity_curve.append(capital)
            continue

        # Predicted probability
        prob = 0.5 + 0.4 * (pred / (abs(pred) + 1e-10))
        prob = np.clip(prob, 0.1, 0.9)

        # Edge
        edge = 2 * prob - 1

        if edge < min_edge:
            i += 1
            equity_curve.append(capital)
            continue

        # Kelly position size
        kelly_size = kelly_fraction * edge
        kelly_size = np.clip(kelly_size, 0, max_position_size)

        # Scale by magnitude
        magnitude_scale = min(1.0, pred_magnitude / 0.02)
        position_size = kelly_size * magnitude_scale

        # Direction
        direction = np.sign(pred)

        # PnL over holding period
        hold_return = actual
        pnl = capital * position_size * direction * hold_return
        capital += pnl

        trades.append({
            'pnl': pnl,
            'capital': capital
        })

        # Skip to next period
        for j in range(holding_period):
            if i + j < n:
                equity_curve.append(capital)

        i += holding_period

    # Fill remaining
    while len(equity_curve) < n:
        equity_curve.append(capital)

    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (capital - 10000) / 10000
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 24)

    # Max drawdown
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }


def test_asset(
    name: str,
    symbol: str,
    regime: str,
    period: str = '3mo',
    interval: str = '1h',
    use_magnitude_aware: bool = False,
    use_china_features: bool = False,
    kelly: float = 0.15,
    holding: int = 5,
    min_edge: float = 0.07,
    mag_threshold: float = 0.005
) -> Dict:
    """Test single asset."""
    try:
        print(f"\n{'='*80}")
        print(f"Testing: {name} ({symbol})")
        print(f"Regime: {regime}")
        print(f"{'='*80}")

        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if len(data) < 100:
            print(f"  ⚠️ Insufficient data ({len(data)} bars)")
            return None

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"  Fetched {len(data)} bars")

        # Add technical features
        data_features = add_technical_features(data)

        # Add China features if requested
        china_cols = []
        if use_china_features:
            china_fetcher = ChinaEconomicDataFetcher(fred_api_key=None)
            start_date = data.index[0].strftime('%Y-%m-%d')
            end_date = data.index[-1].strftime('%Y-%m-%d')
            china_features = china_fetcher.fetch_all_china_features(
                start_date, end_date, use_synthetic_fallback=True
            )
            if not china_features.empty:
                china_aligned = china_fetcher.align_china_features_to_asset(
                    data_features, china_features
                )
                data_features = pd.concat([data_features, china_aligned], axis=1)
                china_cols = list(china_aligned.columns)
                print(f"  Added {len(china_cols)} China features")

        # Create target
        data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
        data_features = data_features.dropna()

        # Features
        technical_features = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                             'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']
        feature_cols = technical_features + china_cols

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

        if len(X_test) < 20:
            print(f"  ⚠️ Insufficient test data ({len(X_test)} samples)")
            return None

        # Train model
        ensemble = EnhancedEnsemblePredictor(use_prediction_market=False)
        ensemble.train_all_models(X_train, y_train, X_val, y_val, models_to_train=['lightgbm'])

        predictions = ensemble.predict(X_test)
        dir_acc = np.mean(np.sign(predictions) == np.sign(y_test.values))

        # Backtest
        price_data = data_features.iloc[train_size+val_size:][['Close']].copy()

        if use_magnitude_aware:
            bt_result = magnitude_aware_backtest(
                price_data, predictions, y_test.values,
                kelly_fraction=kelly, min_edge=min_edge,
                magnitude_threshold=mag_threshold,
                holding_period=holding, max_position_size=0.10
            )
        else:
            # Standard Kelly backtest
            probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)
            backtester = KellyBacktester(
                kelly_fraction=kelly,
                min_edge=min_edge,
                max_position_size=0.10
            )
            bt_result = backtester.backtest(price_data, probs, hold_periods=holding)

        # Calculate volatility
        volatility = data_features['returns'].std() * np.sqrt(252 * 24) * 100

        result = {
            'name': name,
            'symbol': symbol,
            'regime': regime,
            'dir_acc': dir_acc,
            'return': bt_result['total_return'],
            'sharpe': bt_result['sharpe_ratio'],
            'max_dd': bt_result['max_drawdown'],
            'volatility': volatility,
            'num_trades': bt_result.get('num_trades', 0),
            'profitable': bt_result['total_return'] > 0,
            'magnitude_aware': use_magnitude_aware,
            'china_features': use_china_features
        }

        print(f"  Directional Accuracy: {dir_acc:.1%}")
        print(f"  Return: {bt_result['total_return']:.2%}")
        print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f}")
        print(f"  Status: {'✅ PROFIT' if result['profitable'] else '❌ LOSS'}")

        return result

    except Exception as e:
        print(f"  ❌ Error testing {name}: {e}")
        return None


def main():
    print("="*80)
    print(" "*20 + "COMPREHENSIVE REGIME TESTING")
    print(" "*15 + "All Asset Classes with Latest Improvements")
    print("="*80)
    print()

    start_time = datetime.now()

    # Define test assets by regime
    test_configs = [
        # US STOCKS
        {'name': 'Apple (Tech)', 'symbol': 'AAPL', 'regime': 'US Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'Microsoft (Tech)', 'symbol': 'MSFT', 'regime': 'US Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'JP Morgan (Finance)', 'symbol': 'JPM', 'regime': 'US Stock',
         'kelly': 0.15, 'holding': 7, 'min_edge': 0.07, 'mag_aware': False},
        {'name': 'Exxon (Energy)', 'symbol': 'XOM', 'regime': 'US Stock',
         'kelly': 0.15, 'holding': 7, 'min_edge': 0.07, 'mag_aware': False},
        {'name': 'Johnson & Johnson (Healthcare)', 'symbol': 'JNJ', 'regime': 'US Stock',
         'kelly': 0.10, 'holding': 7, 'min_edge': 0.04, 'mag_aware': True, 'mag_threshold': 0.005},

        # EUROPEAN STOCKS
        {'name': 'ASML (Europe Tech)', 'symbol': 'ASML', 'regime': 'Europe Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'SAP (Europe Tech)', 'symbol': 'SAP', 'regime': 'Europe Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'Nestle (Europe Consumer)', 'symbol': 'NSRGY', 'regime': 'Europe Stock',
         'kelly': 0.10, 'holding': 7, 'min_edge': 0.04, 'mag_aware': True, 'mag_threshold': 0.005},

        # ASIAN STOCKS
        {'name': 'Toyota (Asia Auto)', 'symbol': 'TM', 'regime': 'Asia Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'Sony (Asia Tech)', 'symbol': 'SONY', 'regime': 'Asia Stock',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},
        {'name': 'Alibaba (Asia Tech)', 'symbol': 'BABA', 'regime': 'Asia Stock',
         'kelly': 0.15, 'holding': 5, 'min_edge': 0.07, 'mag_aware': False},

        # COMMODITIES
        {'name': 'Gold', 'symbol': 'GC=F', 'regime': 'Commodity',
         'kelly': 0.25, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},
        {'name': 'Crude Oil', 'symbol': 'CL=F', 'regime': 'Commodity',
         'kelly': 0.15, 'holding': 5, 'min_edge': 0.07, 'mag_aware': False},
        {'name': 'Iron Ore (VALE)', 'symbol': 'VALE', 'regime': 'Commodity',
         'kelly': 0.05, 'holding': 7, 'min_edge': 0.04, 'mag_aware': True,
         'mag_threshold': 0.003, 'china_features': True},
        {'name': 'Silver', 'symbol': 'SI=F', 'regime': 'Commodity',
         'kelly': 0.20, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},
        {'name': 'Copper', 'symbol': 'HG=F', 'regime': 'Commodity',
         'kelly': 0.15, 'holding': 5, 'min_edge': 0.06, 'mag_aware': False},

        # FOREX
        {'name': 'EUR/USD', 'symbol': 'EURUSD=X', 'regime': 'Forex',
         'kelly': 0.30, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},
        {'name': 'GBP/USD', 'symbol': 'GBPUSD=X', 'regime': 'Forex',
         'kelly': 0.25, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},
        {'name': 'USD/JPY', 'symbol': 'JPY=X', 'regime': 'Forex',
         'kelly': 0.25, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},
        {'name': 'USD/CHF', 'symbol': 'CHF=X', 'regime': 'Forex',
         'kelly': 0.20, 'holding': 3, 'min_edge': 0.05, 'mag_aware': False},

        # CRYPTO
        {'name': 'Bitcoin', 'symbol': 'BTC-USD', 'regime': 'Crypto',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.10, 'mag_aware': False},
        {'name': 'Ethereum', 'symbol': 'ETH-USD', 'regime': 'Crypto',
         'kelly': 0.10, 'holding': 5, 'min_edge': 0.10, 'mag_aware': False},
    ]

    results = []

    for config in test_configs:
        result = test_asset(
            name=config['name'],
            symbol=config['symbol'],
            regime=config['regime'],
            kelly=config['kelly'],
            holding=config['holding'],
            min_edge=config['min_edge'],
            use_magnitude_aware=config.get('mag_aware', False),
            use_china_features=config.get('china_features', False),
            mag_threshold=config.get('mag_threshold', 0.005)
        )
        if result:
            results.append(result)

    # Create results DataFrame
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*80}\n")

    # Summary by regime
    print("RESULTS BY REGIME")
    print(f"{'='*80}\n")

    for regime in ['US Stock', 'Europe Stock', 'Asia Stock', 'Commodity', 'Forex', 'Crypto']:
        regime_data = df[df['regime'] == regime]
        if len(regime_data) == 0:
            continue

        profitable = regime_data['profitable'].sum()
        total = len(regime_data)
        avg_acc = regime_data['dir_acc'].mean()
        avg_return = regime_data['return'].mean()
        avg_sharpe = regime_data['sharpe'].mean()

        print(f"{regime}:")
        print(f"  Assets Tested: {total}")
        print(f"  Profitable: {profitable}/{total} ({profitable/total*100:.1f}%)")
        print(f"  Avg Directional Accuracy: {avg_acc:.1%}")
        print(f"  Avg Return: {avg_return:.2%}")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print()

    # Detailed results table
    print(f"\n{'='*80}")
    print("DETAILED RESULTS (Sorted by Sharpe Ratio)")
    print(f"{'='*80}\n")

    df_sorted = df.sort_values('sharpe', ascending=False)

    print(f"{'Asset':<30} {'Regime':<15} {'Acc':<8} {'Return':<10} {'Sharpe':<10} {'MaxDD':<10} {'Status':<10}")
    print("-" * 110)

    for _, row in df_sorted.iterrows():
        status = "✅ PROFIT" if row['profitable'] else "❌ LOSS"
        mag_marker = " 🎯" if row['magnitude_aware'] else ""
        china_marker = " 🇨🇳" if row['china_features'] else ""
        name = row['name'][:28] + mag_marker + china_marker

        print(f"{name:<30} {row['regime']:<15} {row['dir_acc']:<8.1%} "
              f"{row['return']:<10.2%} {row['sharpe']:<10.2f} "
              f"{row['max_dd']:<10.2%} {status:<10}")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")

    total_assets = len(df)
    total_profitable = df['profitable'].sum()
    profitability_rate = total_profitable / total_assets * 100

    print(f"Total Assets Tested: {total_assets}")
    print(f"Profitable Assets: {total_profitable}")
    print(f"Profitability Rate: {profitability_rate:.1f}%")
    print()
    print(f"Average Directional Accuracy: {df['dir_acc'].mean():.1%}")
    print(f"Average Return: {df['return'].mean():.2%}")
    print(f"Average Sharpe: {df['sharpe'].mean():.2f}")
    print()

    # Best performers
    print(f"{'='*80}")
    print("TOP 10 PERFORMERS (By Sharpe Ratio)")
    print(f"{'='*80}\n")

    top10 = df_sorted.head(10)
    print(f"{'Rank':<6} {'Asset':<30} {'Regime':<15} {'Return':<10} {'Sharpe':<10}")
    print("-" * 80)

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<6} {row['name']:<30} {row['regime']:<15} "
              f"{row['return']:<10.2%} {row['sharpe']:<10.2f}")

    # Profitability by regime
    print(f"\n{'='*80}")
    print("PROFITABILITY BY REGIME")
    print(f"{'='*80}\n")

    regime_summary = df.groupby('regime').agg({
        'profitable': ['sum', 'count'],
        'dir_acc': 'mean',
        'return': 'mean',
        'sharpe': 'mean'
    })

    for regime in regime_summary.index:
        profitable = regime_summary.loc[regime, ('profitable', 'sum')]
        total = regime_summary.loc[regime, ('profitable', 'count')]
        rate = profitable / total * 100 if total > 0 else 0

        print(f"{regime}: {profitable}/{total} = {rate:.1f}% profitable")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n\nTest Duration: {duration:.1f} seconds")

    # Save results
    output_file = "COMPREHENSIVE_REGIME_RESULTS.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    print(f"\n{'='*80}")
    print("COMPREHENSIVE TESTING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
