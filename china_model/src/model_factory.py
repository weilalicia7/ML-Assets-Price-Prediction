"""
Automated Model Factory with Strategy Detection

Creates production-ready prediction models for robust performers with:
    - Automatic strategy detection (momentum, mean reversion, regime)
    - Optimized hyperparameters per strategy type
    - Ensemble methods for improved robustness
    - Model persistence and versioning
"""

import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import json
import os
import pickle
import warnings
import time
import threading
warnings.filterwarnings('ignore')

from china_stock_screener import detect_market_type, get_market_config

# ============================================================================
# YAHOO FINANCE RATE LIMITER
# ============================================================================
_MODEL_YF_LOCK = threading.Lock()
_MODEL_YF_LAST_REQUEST = 0
_MODEL_YF_MIN_INTERVAL = 0.5  # Minimum 0.5 seconds between requests
_MODEL_YF_CACHE = {}  # Cache for stock data
_MODEL_YF_CACHE_TTL = 300  # 5 minutes cache TTL


def _rate_limited_history(symbol, start=None, end=None, period=None, max_retries=3):
    """
    Rate-limited Yahoo Finance history fetch with caching.
    Prevents 401 errors during concurrent model training.
    """
    global _MODEL_YF_LAST_REQUEST

    # Create cache key
    cache_key = f"{symbol}_{start}_{end}_{period}"

    # Check cache first
    if cache_key in _MODEL_YF_CACHE:
        cached_data, cached_time = _MODEL_YF_CACHE[cache_key]
        if time.time() - cached_time < _MODEL_YF_CACHE_TTL:
            return cached_data

    # Rate limiting
    with _MODEL_YF_LOCK:
        now = time.time()
        elapsed = now - _MODEL_YF_LAST_REQUEST
        if elapsed < _MODEL_YF_MIN_INTERVAL:
            time.sleep(_MODEL_YF_MIN_INTERVAL - elapsed)
        _MODEL_YF_LAST_REQUEST = time.time()

    # Try to fetch with retries
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            if period:
                df = ticker.history(period=period)
            else:
                df = ticker.history(start=start, end=end)

            if df is not None and not df.empty:
                # Cache the result
                _MODEL_YF_CACHE[cache_key] = (df, time.time())
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  [RATE LIMIT] Retry {attempt + 1}/{max_retries} for {symbol} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  [RATE LIMIT] All retries failed for {symbol}: {e}")

    return pd.DataFrame()


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_START = '2023-01-01'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END = '2025-11-27'

# Strategy-specific configurations
STRATEGY_CONFIG = {
    'momentum': {
        'iterations': 200,
        'depth': 5,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
        'feature_weights': {
            'momentum': 1.5,
            'returns': 1.3,
            'trend': 1.2,
        }
    },
    'mean_reversion': {
        'iterations': 180,
        'depth': 4,
        'learning_rate': 0.025,
        'l2_leaf_reg': 5,
        'feature_weights': {
            'rsi': 1.5,
            'bollinger': 1.4,
            'distance_from_mean': 1.3,
        }
    },
    'regime': {
        'iterations': 220,
        'depth': 6,
        'learning_rate': 0.02,
        'l2_leaf_reg': 4,
        'feature_weights': {
            'volatility': 1.4,
            'volume': 1.3,
            'trend_strength': 1.2,
        }
    }
}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_comprehensive_features(df):
    """Full feature set for production models (30+ features)"""
    df = df.copy()

    # Price returns (multiple timeframes)
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'returns_{period}d'] = df['Close'].pct_change(period)

    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'price_vs_sma{period}'] = df['Close'] / df[f'sma_{period}']

    # MA crossovers
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['sma_10_50_cross'] = (df['sma_10'] > df['sma_50']).astype(int)

    # Volume features
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['volume_ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        df['vol_price_corr'] = df['returns_1d'].rolling(20).corr(df['Volume'].pct_change())
    else:
        df['volume_ratio_5'] = 1.0
        df['volume_ratio_20'] = 1.0
        df['volume_trend'] = 1.0
        df['vol_price_corr'] = 0.0

    # Volatility features
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns_1d'].rolling(period).std()

    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)

    # Momentum features
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

    # Bollinger Bands
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bollinger_upper'] = sma + 2 * std
    df['bollinger_lower'] = sma - 2 * std
    df['bollinger_pct'] = (df['Close'] - sma) / (2 * std + 1e-10)
    df['bollinger_width'] = (4 * std) / (sma + 1e-10)

    # Mean reversion
    df['distance_from_mean_10'] = (df['Close'] - df['Close'].rolling(10).mean()) / (df['Close'].rolling(10).std() + 1e-10)
    df['distance_from_mean_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-10)

    # Trend strength
    df['trend_strength'] = abs(df['momentum_20']) / (df['volatility_20'] + 1e-10)

    # High/Low range
    df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    df['range_vs_avg'] = df['daily_range'] / df['daily_range'].rolling(20).mean()

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


FEATURE_COLUMNS = [
    'returns_1d', 'returns_2d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
    'rsi_7', 'rsi_14', 'rsi_21',
    'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
    'sma_5_20_cross', 'sma_10_50_cross',
    'volume_ratio_5', 'volume_ratio_20', 'volume_trend', 'vol_price_corr',
    'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio',
    'momentum_5', 'momentum_10', 'momentum_20',
    'bollinger_pct', 'bollinger_width',
    'distance_from_mean_10', 'distance_from_mean_20',
    'trend_strength',
    'daily_range', 'range_vs_avg',
    'macd', 'macd_signal', 'macd_hist'
]


# ============================================================================
# STRATEGY DETECTION
# ============================================================================

class StrategyDetector:
    """Detect optimal trading strategy based on stock characteristics"""

    def __init__(self):
        self.strategy_scores = {}

    def detect_strategy(self, df, symbol):
        """
        Analyze price behavior to determine optimal strategy

        Returns: 'momentum', 'mean_reversion', or 'regime'
        """
        # Calculate key metrics
        returns = df['Close'].pct_change().dropna()

        # Momentum score: autocorrelation of returns
        autocorr_1 = returns.autocorr(1) if len(returns) > 10 else 0
        autocorr_5 = returns.autocorr(5) if len(returns) > 10 else 0
        momentum_score = max(0, autocorr_1 + autocorr_5 * 0.5)

        # Mean reversion score: negative autocorrelation
        mean_rev_score = max(0, -autocorr_1 - autocorr_5 * 0.5)

        # RSI extremes frequency
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi_extreme_freq = ((rsi < 30) | (rsi > 70)).mean()
        mean_rev_score += rsi_extreme_freq * 2

        # Regime score: volatility clustering
        volatility = returns.rolling(20).std()
        vol_of_vol = volatility.std() / (volatility.mean() + 1e-10)
        regime_score = vol_of_vol * 2

        # Trend persistence
        sma_20 = df['Close'].rolling(20).mean()
        trend_periods = (df['Close'] > sma_20).astype(int).diff().abs().sum() / len(df)
        regime_score += (1 - trend_periods) * 2  # Lower = more persistent trends

        self.strategy_scores[symbol] = {
            'momentum': float(momentum_score),
            'mean_reversion': float(mean_rev_score),
            'regime': float(regime_score)
        }

        # Return highest scoring strategy
        scores = {
            'momentum': momentum_score,
            'mean_reversion': mean_rev_score,
            'regime': regime_score
        }

        return max(scores, key=scores.get)

    def get_detailed_analysis(self, symbol):
        """Get detailed strategy analysis for a symbol"""
        if symbol in self.strategy_scores:
            return self.strategy_scores[symbol]
        return None


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """
    Factory for creating production-ready prediction models
    """

    def __init__(self, models_dir=None):
        """
        Args:
            models_dir: Directory to save/load models
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.strategy_detector = StrategyDetector()
        self.models = {}
        self.model_metadata = {}

    def build_model(self, symbol, verbose=True):
        """
        Build optimized prediction model for a symbol

        Args:
            symbol: Stock ticker
            verbose: Print progress

        Returns:
            dict with model, metadata, and performance metrics
        """
        market_type = detect_market_type(symbol)
        market_config = get_market_config(market_type)

        if verbose:
            print(f"\nBuilding model for {symbol} ({market_type})...")

        # Download data (rate-limited to prevent 401 errors)
        try:
            df = _rate_limited_history(symbol, start=TRAIN_START, end=TEST_END)

            if df.empty or len(df) < 100:
                print(f"  Insufficient data: {len(df) if not df.empty else 0} days")
                return None
        except Exception as e:
            print(f"  Download error: {e}")
            return None

        # Detect optimal strategy
        strategy = self.strategy_detector.detect_strategy(df, symbol)
        strategy_config = STRATEGY_CONFIG[strategy]

        if verbose:
            print(f"  Detected strategy: {strategy}")
            scores = self.strategy_detector.get_detailed_analysis(symbol)
            print(f"  Strategy scores: momentum={scores['momentum']:.2f}, mean_rev={scores['mean_reversion']:.2f}, regime={scores['regime']:.2f}")

        # Add features
        df = add_comprehensive_features(df)

        # Create target
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['next_return'] = df['Close'].pct_change().shift(-1)

        # Drop NaN
        df = df.dropna()

        if len(df) < 100:
            print(f"  Insufficient data after feature engineering")
            return None

        # Split
        train_df = df[df.index < TEST_START]
        test_df = df[df.index >= TEST_START]

        if len(train_df) < 50 or len(test_df) < 20:
            print(f"  Insufficient train/test data")
            return None

        # Get available features
        available_features = [f for f in FEATURE_COLUMNS if f in train_df.columns]

        X_train = train_df[available_features]
        y_train = train_df['target']
        X_test = test_df[available_features]
        test_returns = test_df['next_return']

        # Build ensemble (5 seeds for robustness)
        seeds = [42, 123, 456, 789, 1011]
        ensemble_models = []
        seed_results = []

        for seed in seeds:
            params = {
                'iterations': strategy_config['iterations'],
                'learning_rate': strategy_config['learning_rate'],
                'depth': strategy_config['depth'],
                'l2_leaf_reg': strategy_config['l2_leaf_reg'],
                'subsample': 0.8,
                'random_seed': seed,
                'verbose': False,
            }

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            ensemble_models.append(model)

            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            confident_mask = y_pred_proba >= market_config['confidence_threshold']

            if confident_mask.sum() >= market_config['min_trades']:
                avg_return = test_returns[confident_mask].mean()
                num_trades = int(confident_mask.sum())
                passes = avg_return >= market_config['profit_gate']
            else:
                avg_return = 0.0
                num_trades = 0
                passes = False

            seed_results.append({
                'seed': seed,
                'avg_return': float(avg_return),
                'num_trades': num_trades,
                'passes': bool(passes)
            })

        # Calculate ensemble prediction
        ensemble_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in ensemble_models], axis=0)
        confident_mask = ensemble_proba >= market_config['confidence_threshold']

        if confident_mask.sum() >= market_config['min_trades']:
            ensemble_return = test_returns[confident_mask].mean()
            ensemble_trades = int(confident_mask.sum())
        else:
            ensemble_return = 0.0
            ensemble_trades = 0

        pass_rate = np.mean([r['passes'] for r in seed_results])

        # Feature importance
        feature_importance = dict(zip(
            available_features,
            np.mean([m.feature_importances_ for m in ensemble_models], axis=0)
        ))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        if verbose:
            print(f"  Ensemble return: {ensemble_return*100:.2f}%")
            print(f"  Pass rate: {pass_rate*100:.0f}%")
            print(f"  Trades: {ensemble_trades}")
            print(f"  Top features: {[f[0] for f in top_features[:5]]}")

        # Store model
        model_data = {
            'symbol': symbol,
            'market_type': market_type,
            'strategy': strategy,
            'strategy_scores': self.strategy_detector.get_detailed_analysis(symbol),
            'ensemble_models': ensemble_models,
            'features': available_features,
            'confidence_threshold': market_config['confidence_threshold'],
            'metrics': {
                'ensemble_return': float(ensemble_return),
                'ensemble_trades': ensemble_trades,
                'pass_rate': float(pass_rate),
                'seed_results': seed_results,
                'feature_importance': feature_importance,
                'top_features': [f[0] for f in top_features],
            },
            'created_at': datetime.now().isoformat(),
            'train_period': f"{TRAIN_START} to {TRAIN_END}",
            'test_period': f"{TEST_START} to {TEST_END}",
        }

        self.models[symbol] = model_data

        return model_data

    def build_all_models(self, symbols, verbose=True):
        """Build models for all symbols"""
        results = []

        if verbose:
            print("=" * 70)
            print("MODEL FACTORY: Building Production Models")
            print("=" * 70)
            print(f"Symbols: {len(symbols)}")

        for symbol in symbols:
            result = self.build_model(symbol, verbose=verbose)
            if result:
                results.append(result)

        if verbose:
            print("\n" + "=" * 70)
            print("MODEL FACTORY SUMMARY")
            print("=" * 70)
            print(f"Models built: {len(results)}/{len(symbols)}")

            # Strategy distribution
            strategies = [r['strategy'] for r in results]
            print(f"\nStrategy distribution:")
            for s in ['momentum', 'mean_reversion', 'regime']:
                count = strategies.count(s)
                print(f"  {s}: {count} ({count/len(results)*100:.0f}%)")

            # Best performers
            print(f"\nTop performers by return:")
            sorted_results = sorted(results, key=lambda x: x['metrics']['ensemble_return'], reverse=True)
            for r in sorted_results[:5]:
                print(f"  {r['symbol']}: {r['metrics']['ensemble_return']*100:.2f}% ({r['strategy']})")

        return results

    def save_models(self, filepath=None):
        """Save all models to disk"""
        if filepath is None:
            filepath = os.path.join(self.models_dir, 'production_models.pkl')

        # Save models
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)

        # Save metadata separately as JSON (without models)
        metadata = {}
        for symbol, data in self.models.items():
            metadata[symbol] = {
                'symbol': data['symbol'],
                'market_type': data['market_type'],
                'strategy': data['strategy'],
                'strategy_scores': data['strategy_scores'],
                'features': data['features'],
                'confidence_threshold': data['confidence_threshold'],
                'metrics': data['metrics'],
                'created_at': data['created_at'],
                'train_period': data['train_period'],
                'test_period': data['test_period'],
            }

        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModels saved to: {filepath}")
        print(f"Metadata saved to: {metadata_path}")

    def load_models(self, filepath=None):
        """Load models from disk"""
        if filepath is None:
            filepath = os.path.join(self.models_dir, 'production_models.pkl')

        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)

        print(f"Loaded {len(self.models)} models from {filepath}")
        return self.models

    def predict(self, symbol, df=None):
        """
        Make prediction for a symbol using its trained model

        Args:
            symbol: Stock ticker
            df: Optional dataframe with features (downloads latest if None)

        Returns:
            dict with prediction, confidence, and signal
        """
        if symbol not in self.models:
            raise ValueError(f"No model found for {symbol}")

        model_data = self.models[symbol]

        # Download latest data if not provided (rate-limited)
        if df is None:
            df = _rate_limited_history(symbol, period='60d')

        # Add features
        df = add_comprehensive_features(df)
        df = df.dropna()

        if len(df) == 0:
            return None

        # Get latest row
        latest = df.iloc[[-1]][model_data['features']]

        # Ensemble prediction
        ensemble_models = model_data['ensemble_models']
        probas = [m.predict_proba(latest)[0, 1] for m in ensemble_models]
        avg_proba = np.mean(probas)
        std_proba = np.std(probas)

        # Determine signal
        threshold = model_data['confidence_threshold']

        if avg_proba >= threshold:
            signal = 'BUY'
        elif avg_proba <= (1 - threshold):
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'probability': float(avg_proba),
            'confidence_std': float(std_proba),
            'signal': signal,
            'strategy': model_data['strategy'],
            'threshold': threshold,
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Build production models for robust performers"""
    print("=" * 70)
    print("CHINA MODEL FACTORY")
    print("=" * 70)
    print()

    # Load robust performers from tiered screening
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tiered_screening_results.json')

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            screening_results = json.load(f)

        robust_symbols = [r['symbol'] for r in screening_results['robust_performers']]
        print(f"Loaded {len(robust_symbols)} robust performers from screening results")
    else:
        # Default robust performers
        robust_symbols = [
            '9999.HK',   # NetEase
            '9888.HK',   # Baidu
            '0981.HK',   # SMIC
            '3993.HK',   # CMOC
            '2600.HK',   # Aluminum Corp
            '6160.HK',   # Beigene
            '0175.HK',   # Geely
            '1299.HK',   # AIA
            '1093.HK',   # CSPC Pharma
            '2828.HK',   # HS China ETF
            '300274.SZ', # Sungrow Power
        ]
        print(f"Using default {len(robust_symbols)} robust performers")

    print()

    # Build models
    factory = ModelFactory()
    results = factory.build_all_models(robust_symbols, verbose=True)

    # Save models
    factory.save_models()

    # Generate predictions for all models
    print("\n" + "=" * 70)
    print("CURRENT SIGNALS")
    print("=" * 70)
    print(f"{'Symbol':<12} {'Strategy':<15} {'Prob':>8} {'Signal':<6}")
    print("-" * 45)

    for symbol in robust_symbols:
        if symbol in factory.models:
            pred = factory.predict(symbol)
            if pred:
                print(f"{pred['symbol']:<12} {pred['strategy']:<15} {pred['probability']*100:>7.1f}% {pred['signal']:<6}")


if __name__ == '__main__':
    main()
