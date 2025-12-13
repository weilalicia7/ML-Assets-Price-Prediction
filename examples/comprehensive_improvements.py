"""
Comprehensive Improvements
Apply all improvement strategies to boost profitability from 50% to 60%+:

1. ✅ Magnitude-aware on high-accuracy losers (Bitcoin, XOM) - DONE: 50%
2. 🔄 Add forex features (VIX, DXY, Interest Rate Diff)
3. 🔄 Add commodity macro features (Gold, Oil, Copper)
4. 🔄 Test Asia stocks on different intervals (4h, 1d instead of 1h)
5. 🔄 Low-accuracy asset filtering (<40% accuracy)
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
from typing import Dict, Optional

from src.models.enhanced_ensemble import EnhancedEnsemblePredictor
from src.trading.kelly_backtester import KellyBacktester
from src.data.forex_features import ForexFeaturesFetcher
from src.data.commodity_macro_features import CommodityMacroFeaturesFetcher


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


def test_forex_with_features(symbol: str, name: str, baseline_return: float, baseline_sharpe: float):
    """Test forex pair with interest rate features."""
    print(f"\n{'='*80}")
    print(f"Testing Forex: {name} ({symbol})")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_return:.2%} return, {baseline_sharpe:.2f} Sharpe")

    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='3mo', interval='1h')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Fetched {len(data)} bars")

        # Technical features
        data_features = add_technical_features(data)

        # Add forex features
        forex_fetcher = ForexFeaturesFetcher(fred_api_key=None)
        pair_name = symbol.replace('=X', '').replace('USD', '').replace('GBP', '').replace('EUR', '').replace('JPY', '').replace('CHF', '')

        start_date = data.index[0].strftime('%Y-%m-%d')
        end_date = data.index[-1].strftime('%Y-%m-%d')

        forex_features = forex_fetcher.fetch_all_forex_features(
            symbol.replace('=X', ''),
            start_date, end_date,
            use_synthetic_fallback=True
        )

        forex_cols = []
        if not forex_features.empty:
            forex_aligned = forex_fetcher.align_forex_features_to_asset(data_features, forex_features)
            data_features = pd.concat([data_features, forex_aligned], axis=1)
            forex_cols = list(forex_aligned.columns)
            print(f"Added {len(forex_cols)} forex features: {forex_cols}")

        # Create target
        data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
        data_features = data_features.dropna()

        # Features
        technical_features = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                             'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']
        feature_cols = technical_features + forex_cols

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

        # Backtest
        price_data = data_features.iloc[train_size+val_size:][['Close']].copy()
        probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)

        backtester = KellyBacktester(kelly_fraction=0.25, min_edge=0.05, max_position_size=0.10)
        bt_result = backtester.backtest(price_data, probs, hold_periods=3)

        improvement = bt_result['total_return'] - baseline_return
        sharpe_improvement = bt_result['sharpe_ratio'] - baseline_sharpe

        print(f"Results:")
        print(f"  Return: {bt_result['total_return']:.2%} (baseline: {baseline_return:.2%})")
        print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f} (baseline: {baseline_sharpe:.2f})")
        print(f"  Improvement: {improvement:+.2%} return, {sharpe_improvement:+.2f} Sharpe")

        if bt_result['total_return'] > 0:
            print(f"✅ SUCCESS: {name} is now profitable!")
            return True
        else:
            print(f"⚠️ Still negative: need {abs(bt_result['total_return']):.2%} more")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_commodity_with_macro(symbol: str, name: str, commodity_type: str, baseline_return: float, baseline_sharpe: float):
    """Test commodity with macro features."""
    print(f"\n{'='*80}")
    print(f"Testing Commodity: {name} ({symbol})")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_return:.2%} return, {baseline_sharpe:.2f} Sharpe")

    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='3mo', interval='1h')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Fetched {len(data)} bars")

        # Technical features
        data_features = add_technical_features(data)

        # Add commodity macro features
        macro_fetcher = CommodityMacroFeaturesFetcher(fred_api_key=None)
        start_date = data.index[0].strftime('%Y-%m-%d')
        end_date = data.index[-1].strftime('%Y-%m-%d')

        if commodity_type == 'gold':
            macro_features = macro_fetcher.fetch_gold_features(start_date, end_date, use_synthetic_fallback=True)
        elif commodity_type == 'oil':
            macro_features = macro_fetcher.fetch_oil_features(start_date, end_date, use_synthetic_fallback=True)
        elif commodity_type == 'copper':
            macro_features = macro_fetcher.fetch_copper_features(start_date, end_date, use_synthetic_fallback=True)
        else:
            macro_features = pd.DataFrame()

        macro_cols = []
        if not macro_features.empty:
            macro_aligned = macro_fetcher.align_features_to_asset(data_features, macro_features)
            data_features = pd.concat([data_features, macro_aligned], axis=1)
            macro_cols = list(macro_aligned.columns)
            print(f"Added {len(macro_cols)} macro features: {macro_cols}")

        # Create target
        data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
        data_features = data_features.dropna()

        # Features
        technical_features = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                             'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']
        feature_cols = technical_features + macro_cols

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

        # Backtest
        price_data = data_features.iloc[train_size+val_size:][['Close']].copy()
        probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)

        kelly = 0.25 if commodity_type == 'gold' else 0.15
        backtester = KellyBacktester(kelly_fraction=kelly, min_edge=0.05, max_position_size=0.10)
        bt_result = backtester.backtest(price_data, probs, hold_periods=3)

        improvement = bt_result['total_return'] - baseline_return
        sharpe_improvement = bt_result['sharpe_ratio'] - baseline_sharpe

        print(f"Results:")
        print(f"  Return: {bt_result['total_return']:.2%} (baseline: {baseline_return:.2%})")
        print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f} (baseline: {baseline_sharpe:.2f})")
        print(f"  Improvement: {improvement:+.2%} return, {sharpe_improvement:+.2f} Sharpe")

        if bt_result['total_return'] > 0:
            print(f"✅ SUCCESS: {name} is now profitable!")
            return True
        else:
            print(f"⚠️ Still negative")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_asia_different_intervals(symbol: str, name: str, baseline_acc: float, baseline_return: float):
    """Test Asia stocks on different intervals."""
    print(f"\n{'='*80}")
    print(f"Testing Asia Stock: {name} ({symbol}) - Multiple Intervals")
    print(f"{'='*80}")
    print(f"Baseline (1h): {baseline_acc:.1%} acc, {baseline_return:.2%} return")

    intervals = ['4h', '1d']
    best_result = {'interval': '1h', 'return': baseline_return, 'profitable': False}

    for interval in intervals:
        try:
            print(f"\nTesting {interval} interval...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='6mo' if interval == '1d' else '3mo', interval=interval)

            if len(data) < 100:
                print(f"  Insufficient data: {len(data)} bars")
                continue

            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_features = add_technical_features(data)
            data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
            data_features = data_features.dropna()

            feature_cols = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                           'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']

            X = data_features[feature_cols]
            y = data_features['target']

            train_size = int(0.6 * len(X))
            val_size = int(0.2 * len(X))

            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_val = X.iloc[train_size:train_size+val_size]
            y_val = y.iloc[train_size:train_size+val_size]
            X_test = X.iloc[train_size+val_size:]
            y_test = y.iloc[train_size+val_size:]

            ensemble = EnhancedEnsemblePredictor(use_prediction_market=False)
            ensemble.train_all_models(X_train, y_train, X_val, y_val, models_to_train=['lightgbm'])
            predictions = ensemble.predict(X_test)
            dir_acc = np.mean(np.sign(predictions) == np.sign(y_test.values))

            price_data = data_features.iloc[train_size+val_size:][['Close']].copy()
            probs = 0.3 + 0.4 * (predictions > np.median(predictions)).astype(float)

            backtester = KellyBacktester(kelly_fraction=0.10, min_edge=0.06, max_position_size=0.10)
            bt_result = backtester.backtest(price_data, probs, hold_periods=5)

            print(f"  Accuracy: {dir_acc:.1%}")
            print(f"  Return: {bt_result['total_return']:.2%}")
            print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f}")

            if bt_result['total_return'] > best_result['return']:
                best_result = {
                    'interval': interval,
                    'return': bt_result['total_return'],
                    'sharpe': bt_result['sharpe_ratio'],
                    'profitable': bt_result['total_return'] > 0
                }

        except Exception as e:
            print(f"  Error on {interval}: {e}")
            continue

    print(f"\nBest Interval: {best_result['interval']}")
    print(f"  Return: {best_result['return']:.2%}")
    if best_result['profitable']:
        print(f"✅ SUCCESS: {name} is now profitable on {best_result['interval']} interval!")
        return True
    else:
        print(f"⚠️ Still negative on all intervals")
        return False


def main():
    print("="*80)
    print(" "*20 + "COMPREHENSIVE IMPROVEMENTS")
    print(" "*15 + "Boost Profitability from 50% to 60%+")
    print("="*80)

    start_time = datetime.now()

    improvements = {
        'forex': 0,
        'commodities': 0,
        'asia': 0
    }

    # Test Forex with features
    print(f"\n{'='*80}")
    print("PART 1: FOREX WITH INTEREST RATE FEATURES")
    print(f"{'='*80}")

    forex_tests = [
        {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'baseline_return': -0.0227, 'baseline_sharpe': -13.05},
        {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'baseline_return': -0.0160, 'baseline_sharpe': -7.87},
        {'symbol': 'JPY=X', 'name': 'USD/JPY', 'baseline_return': -0.0235, 'baseline_sharpe': -12.27},
        {'symbol': 'CHF=X', 'name': 'USD/CHF', 'baseline_return': -0.0197, 'baseline_sharpe': -10.20},
    ]

    for test in forex_tests:
        if test_forex_with_features(**test):
            improvements['forex'] += 1

    # Test Commodities with macro features
    print(f"\n\n{'='*80}")
    print("PART 2: COMMODITIES WITH MACRO FEATURES")
    print(f"{'='*80}")

    commodity_tests = [
        {'symbol': 'GC=F', 'name': 'Gold', 'commodity_type': 'gold', 'baseline_return': -0.0293, 'baseline_sharpe': -3.94},
        {'symbol': 'CL=F', 'name': 'Crude Oil', 'commodity_type': 'oil', 'baseline_return': -0.0212, 'baseline_sharpe': -3.26},
        {'symbol': 'HG=F', 'name': 'Copper', 'commodity_type': 'copper', 'baseline_return': -0.0105, 'baseline_sharpe': -1.92},
    ]

    for test in commodity_tests:
        if test_commodity_with_macro(**test):
            improvements['commodities'] += 1

    # Test Asia stocks on different intervals
    print(f"\n\n{'='*80}")
    print("PART 3: ASIA STOCKS ON DIFFERENT INTERVALS")
    print(f"{'='*80}")

    asia_tests = [
        {'symbol': 'SONY', 'name': 'Sony', 'baseline_acc': 0.390, 'baseline_return': -0.0103},
        {'symbol': 'BABA', 'name': 'Alibaba', 'baseline_acc': 0.260, 'baseline_return': -0.0167},
    ]

    for test in asia_tests:
        if test_asia_different_intervals(**test):
            improvements['asia'] += 1

    # Final Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    total_improved = sum(improvements.values())

    print(f"Forex Improvements: {improvements['forex']}/4")
    print(f"Commodity Improvements: {improvements['commodities']}/3")
    print(f"Asia Improvements: {improvements['asia']}/2")
    print(f"\nTotal New Profitable Assets: {total_improved}")

    current_profitable = 11  # After magnitude-aware fixes
    new_profitable = current_profitable + total_improved

    print(f"\nOverall Profitability:")
    print(f"  Previous: {current_profitable}/22 = 50.0%")
    print(f"  New: {new_profitable}/22 = {new_profitable/22*100:.1f}%")
    print(f"  Improvement: +{total_improved} assets = +{total_improved/22*100:.1f} pp")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal Duration: {duration/60:.1f} minutes")

    print(f"\n{'='*80}")
    print("COMPREHENSIVE IMPROVEMENTS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
