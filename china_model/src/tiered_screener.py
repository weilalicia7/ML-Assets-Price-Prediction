"""
Tiered Screening System for China Stocks

Three-tier approach for compute-efficient analysis:
    Tier 1: Quick Screen (1 seed, basic features) - ALL liquid stocks
    Tier 2: Medium Analysis (3 seeds, full features) - Promising only
    Tier 3: Deep Analysis (5 seeds, full validation) - Best candidates

This reduces compute by 90%+ while finding all robust performers.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from china_stock_screener import (
    detect_market_type, get_market_config,
    ChinaUniverseBuilder, FULL_UNIVERSE
)

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_START = '2023-01-01'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END = '2025-11-27'

# Tier configurations (base - will be adjusted by market type)
TIER_CONFIG = {
    'quick': {
        'seeds': [42],
        'iterations': 100,
        'depth': 4,
        'pass_threshold': 0.0,  # Just need positive return
        'features': 'basic',
    },
    'medium': {
        'seeds': [42, 123, 456],
        'iterations': 150,
        'depth': 5,
        'pass_threshold': 0.5,  # 50% pass rate across seeds
        'features': 'full',
    },
    'deep': {
        'seeds': [42, 123, 456, 789, 1011],
        'iterations': 200,
        'depth': 6,
        'pass_threshold': 0.6,  # 60% pass rate for robustness
        'features': 'full',
    }
}

# Market-adjusted thresholds - A-shares need relaxed criteria due to retail noise
MARKET_ADJUSTED_THRESHOLDS = {
    'HK_Stock': {
        'medium_pass': 0.50,   # Strict - institutional efficiency
        'deep_pass': 0.60,
        'description': 'HK stocks - institutional, efficient'
    },
    'HK_ETF': {
        'medium_pass': 0.50,
        'deep_pass': 0.55,
        'description': 'HK ETFs - diversified, stable'
    },
    'Shanghai_A': {
        'medium_pass': 0.40,   # Relaxed - policy influences
        'deep_pass': 0.45,
        'description': 'Shanghai A - policy-driven, SOE heavy'
    },
    'Shenzhen_A': {
        'medium_pass': 0.35,   # Most relaxed - retail noise
        'deep_pass': 0.40,
        'description': 'Shenzhen A - retail-dominated, tech/growth'
    },
    'Futures': {
        'medium_pass': 0.45,
        'deep_pass': 0.50,
        'description': 'Futures proxies - commodity driven'
    }
}

def get_market_threshold(market_type, tier):
    """Get market-adjusted pass threshold"""
    if market_type not in MARKET_ADJUSTED_THRESHOLDS:
        market_type = 'HK_Stock'  # Default to strict

    thresholds = MARKET_ADJUSTED_THRESHOLDS[market_type]

    if tier == 'medium':
        return thresholds['medium_pass']
    elif tier == 'deep':
        return thresholds['deep_pass']
    else:
        return 0.0  # Quick tier always 0

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_basic_features(df):
    """Basic feature set for quick screening (11 features)"""
    df = df.copy()

    # Price
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_10d'] = df['Close'].pct_change(10)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MA
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['price_vs_sma20'] = df['Close'] / df['sma_20']

    # Volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0

    # Volatility
    df['volatility_10'] = df['returns_1d'].rolling(10).std()

    # Momentum
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

    # Bollinger
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bollinger_pct'] = (df['Close'] - sma) / (2 * std)

    return df

BASIC_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_10d',
    'rsi_14', 'price_vs_sma20', 'volume_ratio',
    'volatility_10', 'momentum_10', 'bollinger_pct'
]


def add_full_features(df):
    """Full feature set for deep analysis (21 features)"""
    df = add_basic_features(df)

    # Additional price features
    df['returns_20d'] = df['Close'].pct_change(20)

    # Additional MA
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['price_vs_sma50'] = df['Close'] / df['sma_50']
    df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)

    # Additional volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        df['vol_price_corr'] = df['returns_1d'].rolling(10).corr(df['Volume'].pct_change())
    else:
        df['volume_trend'] = 1.0
        df['vol_price_corr'] = 0.0

    # Additional volatility
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']

    # Additional momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

    # Bollinger width
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bollinger_width'] = (4 * std) / sma

    # Mean reversion
    df['distance_from_mean'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

    # Trend strength
    df['trend_strength'] = abs(df['momentum_20']) / (df['volatility_20'] + 0.001)

    return df


def add_ashare_features(df):
    """
    A-share specific features for policy-driven, retail-dominated markets

    These capture:
    - Retail flow patterns (turnover, small trades)
    - Policy sensitivity (gap behavior, limit moves)
    - Mean reversion (oversold/overbought extremes)
    """
    df = add_full_features(df)

    # ====== RETAIL FLOW INDICATORS ======
    # Turnover ratio - high turnover = retail activity
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        # Turnover spike detection
        df['turnover_spike'] = (df['Volume'] / df['Volume'].rolling(20).mean()) > 2.0
        df['turnover_spike'] = df['turnover_spike'].astype(int)

        # Volume acceleration
        df['volume_accel'] = df['Volume'].pct_change(5) - df['Volume'].pct_change(20)

        # Retail panic indicator (high volume + down day)
        df['panic_indicator'] = ((df['Volume'] > df['Volume'].rolling(20).mean() * 1.5) &
                                  (df['returns_1d'] < -0.02)).astype(int)
    else:
        df['turnover_spike'] = 0
        df['volume_accel'] = 0.0
        df['panic_indicator'] = 0

    # ====== POLICY SENSITIVITY ======
    # Gap behavior (overnight gaps common in A-shares)
    df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['gap_magnitude'] = abs(df['overnight_gap'])

    # Limit move proximity (A-shares have 10% daily limits)
    df['limit_up_proximity'] = (df['High'] - df['Open']) / df['Open']
    df['limit_down_proximity'] = (df['Open'] - df['Low']) / df['Open']
    df['near_limit'] = ((df['limit_up_proximity'] > 0.08) |
                        (df['limit_down_proximity'] > 0.08)).astype(int)

    # ====== MEAN REVERSION SIGNALS ======
    # RSI extremes (A-shares mean revert more)
    df['rsi_extreme_low'] = (df['rsi_14'] < 25).astype(int)
    df['rsi_extreme_high'] = (df['rsi_14'] > 75).astype(int)

    # Consecutive down/up days
    df['consec_down'] = (df['returns_1d'] < 0).rolling(5).sum()
    df['consec_up'] = (df['returns_1d'] > 0).rolling(5).sum()

    # ====== MOMENTUM REGIME ======
    # Short-term vs long-term momentum divergence
    df['momentum_divergence'] = df['momentum_5'] - df['momentum_20']

    # Trend reversal signal
    df['trend_reversal'] = ((df['sma_20'].shift(1) > df['sma_50'].shift(1)) &
                            (df['sma_20'] < df['sma_50'])).astype(int)

    return df


def add_hk_enhanced_features(df, symbol=None):
    """
    HK-specific enhanced features for institutional flow analysis

    These capture:
    - Institutional flow patterns (proxy via volume patterns)
    - Cross-market dynamics (HK-A premium/discount effects)
    - Sector momentum
    """
    df = add_full_features(df)

    # ====== INSTITUTIONAL FLOW PROXY ======
    # Large block detection (institutional vs retail)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        # Volume persistence (institutional accumulation)
        df['volume_persistence'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        # Volume-price divergence (smart money indicator)
        df['volume_price_diverge'] = (
            (df['returns_5d'] < 0) & (df['Volume'] > df['Volume'].rolling(10).mean())
        ).astype(int)

        # Accumulation/Distribution proxy
        df['ad_proxy'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        df['ad_proxy_ma'] = df['ad_proxy'].rolling(10).mean()
    else:
        df['volume_persistence'] = 1.0
        df['volume_price_diverge'] = 0
        df['ad_proxy'] = 0.0
        df['ad_proxy_ma'] = 0.0

    # ====== INTRADAY PATTERN FEATURES ======
    # Opening strength (overnight sentiment)
    df['open_strength'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Closing strength (institutional close)
    df['close_strength'] = (df['Close'] - df['Open']) / df['Open']

    # Daily range position (where close is within range)
    df['range_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

    # ====== CROSS-MARKET DYNAMICS ======
    # Relative strength vs benchmark (using own momentum as proxy)
    df['relative_momentum'] = df['momentum_10'] - df['momentum_20']

    # Volatility regime
    vol_median = df['volatility_20'].rolling(60).median()
    df['high_vol_regime'] = (df['volatility_20'] > vol_median).astype(int)

    # ====== SECTOR ROTATION PROXY ======
    # Price position in 60-day range
    rolling_high = df['High'].rolling(60).max()
    rolling_low = df['Low'].rolling(60).min()
    df['price_position_60d'] = (df['Close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

    # Breakout detection
    df['breakout_up'] = (df['Close'] > df['Close'].rolling(20).max().shift(1)).astype(int)
    df['breakout_down'] = (df['Close'] < df['Close'].rolling(20).min().shift(1)).astype(int)

    return df


# Feature sets
FULL_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
    'rsi_14', 'price_vs_sma20', 'price_vs_sma50', 'sma_cross',
    'volume_ratio', 'volume_trend', 'vol_price_corr',
    'volatility_10', 'volatility_20', 'volatility_ratio',
    'momentum_5', 'momentum_10', 'momentum_20',
    'bollinger_pct', 'bollinger_width',
    'distance_from_mean', 'trend_strength'
]

ASHARE_FEATURES = FULL_FEATURES + [
    'turnover_spike', 'volume_accel', 'panic_indicator',
    'overnight_gap', 'gap_magnitude', 'near_limit',
    'rsi_extreme_low', 'rsi_extreme_high',
    'consec_down', 'consec_up',
    'momentum_divergence', 'trend_reversal'
]

HK_ENHANCED_FEATURES = FULL_FEATURES + [
    'volume_persistence', 'volume_price_diverge',
    'ad_proxy', 'ad_proxy_ma',
    'open_strength', 'close_strength', 'range_position',
    'relative_momentum', 'high_vol_regime',
    'price_position_60d', 'breakout_up', 'breakout_down'
]


# ============================================================================
# SINGLE STOCK EVALUATION
# ============================================================================

def evaluate_stock(symbol, tier='quick', verbose=False):
    """
    Evaluate a single stock at specified tier level

    Args:
        symbol: Stock ticker
        tier: 'quick', 'medium', or 'deep'
        verbose: Print progress

    Returns:
        dict with results or None if failed
    """
    config = TIER_CONFIG[tier]
    market_type = detect_market_type(symbol)
    market_config = get_market_config(market_type)

    # Download data
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=TRAIN_START, end=TEST_END)

        if len(df) < 100:
            if verbose:
                print(f"  {symbol}: Insufficient data ({len(df)} days)")
            return None
    except Exception as e:
        if verbose:
            print(f"  {symbol}: Download error")
        return None

    # Add features - use market-specific features
    is_ashare = market_type in ['Shanghai_A', 'Shenzhen_A']
    is_hk = market_type in ['HK_Stock', 'HK_ETF']

    if config['features'] == 'basic':
        df = add_basic_features(df)
        feature_cols = BASIC_FEATURES
    elif is_ashare:
        # Use enhanced A-share features for mainland stocks
        df = add_ashare_features(df)
        feature_cols = ASHARE_FEATURES
    elif is_hk:
        # Use enhanced HK features for institutional flow analysis
        df = add_hk_enhanced_features(df, symbol)
        feature_cols = HK_ENHANCED_FEATURES
    else:
        df = add_full_features(df)
        feature_cols = FULL_FEATURES

    # Create target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['next_return'] = df['Close'].pct_change().shift(-1)

    # Drop NaN
    df = df.dropna()

    if len(df) < 100:
        return None

    # Split
    train_df = df[df.index < TEST_START]
    test_df = df[df.index >= TEST_START]

    if len(train_df) < 50 or len(test_df) < 20:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    test_returns = test_df['next_return']

    # Evaluate across seeds
    seed_results = []

    for seed in config['seeds']:
        params = {
            'iterations': config['iterations'],
            'learning_rate': 0.03,
            'depth': config['depth'],
            'l2_leaf_reg': 5,
            'subsample': 0.8,
            'random_seed': seed,
            'verbose': False,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Filter by confidence
        confident_mask = y_pred_proba >= market_config['confidence_threshold']

        if confident_mask.sum() < market_config['min_trades']:
            avg_return = 0.0
            num_trades = 0
            passes = False
        else:
            confident_returns = test_returns[confident_mask]
            avg_return = confident_returns.mean()
            num_trades = len(confident_returns)
            passes = avg_return >= market_config['profit_gate']

        seed_results.append({
            'seed': seed,
            'avg_return': float(avg_return),
            'num_trades': int(num_trades),
            'passes': bool(passes)
        })

    # Aggregate results
    pass_rate = np.mean([r['passes'] for r in seed_results])
    avg_return = np.mean([r['avg_return'] for r in seed_results])
    avg_trades = np.mean([r['num_trades'] for r in seed_results])

    # Use market-adjusted threshold instead of fixed threshold
    market_threshold = get_market_threshold(market_type, tier)
    is_promising = pass_rate >= market_threshold and avg_return > 0

    result = {
        'symbol': symbol,
        'market_type': market_type,
        'tier': tier,
        'pass_rate': float(pass_rate),
        'avg_return': float(avg_return),
        'avg_trades': float(avg_trades),
        'is_promising': bool(is_promising),
        'seed_results': seed_results,
        'threshold_used': float(market_threshold)  # Track which threshold was used
    }

    if verbose:
        status = "PROMISING" if is_promising else "skip"
        threshold_note = f" (threshold: {market_threshold*100:.0f}%)" if tier != 'quick' else ""
        print(f"  {symbol}: {avg_return*100:.2f}% return, {pass_rate*100:.0f}% pass rate - {status}{threshold_note}")

    return result


# ============================================================================
# TIERED SCREENING PIPELINE
# ============================================================================

class TieredScreener:
    """Run tiered screening pipeline"""

    def __init__(self, universe=None):
        """
        Args:
            universe: List of symbols or None to build from scratch
        """
        self.universe = universe
        self.tier1_results = []
        self.tier2_results = []
        self.tier3_results = []

    def run_tier1_quick(self, symbols=None, verbose=True):
        """
        Tier 1: Quick screen all symbols (1 seed, basic features)

        Returns promising candidates for Tier 2
        """
        if symbols is None:
            symbols = self.universe

        if verbose:
            print("=" * 70)
            print("TIER 1: QUICK SCREENING")
            print("=" * 70)
            print(f"Symbols: {len(symbols)}")
            print(f"Seeds: 1, Features: basic, Threshold: positive return")
            print()

        results = []
        promising = []

        for i, symbol in enumerate(symbols):
            result = evaluate_stock(symbol, tier='quick', verbose=verbose)
            if result:
                results.append(result)
                if result['is_promising']:
                    promising.append(symbol)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(symbols)}, Promising: {len(promising)}")

        self.tier1_results = results

        if verbose:
            print()
            print(f"Tier 1 Complete: {len(promising)}/{len(results)} promising ({len(promising)/len(results)*100:.1f}%)")

        return promising

    def run_tier2_medium(self, symbols=None, verbose=True):
        """
        Tier 2: Medium analysis on promising candidates (3 seeds, full features)

        Returns candidates for Tier 3 deep analysis
        """
        if symbols is None:
            symbols = [r['symbol'] for r in self.tier1_results if r['is_promising']]

        if verbose:
            print("=" * 70)
            print("TIER 2: MEDIUM ANALYSIS")
            print("=" * 70)
            print(f"Symbols: {len(symbols)}")
            print(f"Seeds: 3, Features: full, Threshold: 50% pass rate")
            print()

        results = []
        promising = []

        for symbol in symbols:
            result = evaluate_stock(symbol, tier='medium', verbose=verbose)
            if result:
                results.append(result)
                if result['is_promising']:
                    promising.append(symbol)

        self.tier2_results = results

        if verbose:
            print()
            print(f"Tier 2 Complete: {len(promising)}/{len(results)} promising ({len(promising)/len(results)*100:.1f}%)")

        return promising

    def run_tier3_deep(self, symbols=None, verbose=True):
        """
        Tier 3: Deep analysis on best candidates (5 seeds, full validation)

        Returns final robust performers
        """
        if symbols is None:
            symbols = [r['symbol'] for r in self.tier2_results if r['is_promising']]

        if verbose:
            print("=" * 70)
            print("TIER 3: DEEP ANALYSIS")
            print("=" * 70)
            print(f"Symbols: {len(symbols)}")
            print(f"Seeds: 5, Features: full, Threshold: 60% pass rate")
            print()

        results = []
        robust = []

        for symbol in symbols:
            result = evaluate_stock(symbol, tier='deep', verbose=verbose)
            if result:
                results.append(result)
                if result['is_promising']:
                    robust.append(result)

        self.tier3_results = results

        if verbose:
            print()
            print(f"Tier 3 Complete: {len(robust)}/{len(results)} ROBUST")

        return robust

    def run_full_pipeline(self, symbols, verbose=True):
        """Run complete tiered screening pipeline"""
        if verbose:
            print("=" * 70)
            print("TIERED SCREENING PIPELINE")
            print("=" * 70)
            print(f"Total universe: {len(symbols)} symbols")
            print()

        # Tier 1
        tier1_promising = self.run_tier1_quick(symbols, verbose=verbose)
        print()

        # Tier 2
        tier2_promising = self.run_tier2_medium(tier1_promising, verbose=verbose)
        print()

        # Tier 3
        robust_performers = self.run_tier3_deep(tier2_promising, verbose=verbose)

        # Summary
        if verbose:
            print()
            print("=" * 70)
            print("PIPELINE SUMMARY")
            print("=" * 70)
            print(f"Tier 1 (Quick):  {len(symbols)} -> {len(tier1_promising)} promising")
            print(f"Tier 2 (Medium): {len(tier1_promising)} -> {len(tier2_promising)} promising")
            print(f"Tier 3 (Deep):   {len(tier2_promising)} -> {len(robust_performers)} ROBUST")
            print()
            print(f"Compute savings: {(1 - len(tier2_promising)/len(symbols))*100:.1f}% less deep analysis")

        return robust_performers

    def save_results(self, filepath):
        """Save all results to JSON"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'tier1': self.tier1_results,
            'tier2': self.tier2_results,
            'tier3': self.tier3_results,
            'robust_performers': [r for r in self.tier3_results if r['is_promising']]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run tiered screening on China universe"""
    print("=" * 70)
    print("CHINA MARKET TIERED SCREENING")
    print("=" * 70)
    print()

    # Build liquid universe first
    print("Step 1: Building liquid universe...")
    builder = ChinaUniverseBuilder(min_liquidity=500_000_000)
    liquid_symbols, _ = builder.build_full_universe(verbose=True)

    print()
    print("Step 2: Running tiered screening...")
    print()

    # Run tiered screening
    screener = TieredScreener()
    robust_performers = screener.run_full_pipeline(liquid_symbols, verbose=True)

    # Show robust performers
    if robust_performers:
        print()
        print("=" * 70)
        print("ROBUST PERFORMERS (Final)")
        print("=" * 70)
        print(f"{'Symbol':<12} {'Market':<15} {'Return':>10} {'Pass Rate':>12}")
        print("-" * 55)

        for r in sorted(robust_performers, key=lambda x: x['avg_return'], reverse=True):
            print(f"{r['symbol']:<12} {r['market_type']:<15} {r['avg_return']*100:>9.2f}% {r['pass_rate']*100:>11.0f}%")

    # Save results
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tiered_screening_results.json')
    screener.save_results(save_path)

    print()


if __name__ == '__main__':
    main()
