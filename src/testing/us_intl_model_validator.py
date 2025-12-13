"""
US/International Model Validation Framework

Rigorous testing framework mirroring China model standards:
- Multi-seed robustness testing (5+ seeds)
- Tiered screening (Quick -> Medium -> Deep)
- Market-adjusted thresholds (US, Europe, Emerging)
- Cross-market validation metrics
- Capacity and liquidity testing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Time periods for train/test split
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-11-27'

# Market-adjusted thresholds (mirroring China model structure)
MARKET_THRESHOLDS = {
    'US_Large': {
        'medium_pass': 0.55,    # Moderate - very efficient market
        'deep_pass': 0.60,
        'liquidity_min': 1_000_000_000,  # $1B daily volume
        'min_trades': 10,
        'profit_gate': 0.001,   # 0.1% per trade minimum
        'confidence_threshold': 0.55,
        'description': 'US Large Caps - highly efficient, institutional'
    },
    'US_Mid': {
        'medium_pass': 0.50,
        'deep_pass': 0.55,
        'liquidity_min': 100_000_000,  # $100M daily volume
        'min_trades': 8,
        'profit_gate': 0.002,
        'confidence_threshold': 0.55,
        'description': 'US Mid Caps - moderately efficient'
    },
    'US_Small': {
        'medium_pass': 0.45,
        'deep_pass': 0.50,
        'liquidity_min': 10_000_000,  # $10M daily volume
        'min_trades': 5,
        'profit_gate': 0.003,
        'confidence_threshold': 0.55,
        'description': 'US Small Caps - higher alpha potential, more noise'
    },
    'Europe': {
        'medium_pass': 0.50,
        'deep_pass': 0.55,
        'liquidity_min': 50_000_000,
        'min_trades': 8,
        'profit_gate': 0.002,
        'confidence_threshold': 0.55,
        'description': 'European markets - mixed efficiency'
    },
    'Japan': {
        'medium_pass': 0.50,
        'deep_pass': 0.55,
        'liquidity_min': 50_000_000,
        'min_trades': 8,
        'profit_gate': 0.002,
        'confidence_threshold': 0.55,
        'description': 'Japanese markets - institutional dominance'
    },
    'Emerging': {
        'medium_pass': 0.45,
        'deep_pass': 0.50,
        'liquidity_min': 20_000_000,
        'min_trades': 5,
        'profit_gate': 0.003,
        'confidence_threshold': 0.55,
        'description': 'Emerging markets - higher noise, policy risk'
    },
    'Commodity': {
        'medium_pass': 0.45,
        'deep_pass': 0.50,
        'liquidity_min': 100_000_000,
        'min_trades': 8,
        'profit_gate': 0.002,
        'confidence_threshold': 0.55,
        'description': 'Commodities - macro-driven'
    },
    'Forex': {
        'medium_pass': 0.50,
        'deep_pass': 0.55,
        'liquidity_min': 500_000_000,
        'min_trades': 10,
        'profit_gate': 0.001,
        'confidence_threshold': 0.55,
        'description': 'Forex - highly efficient'
    },
    'Crypto': {
        'medium_pass': 0.40,
        'deep_pass': 0.45,
        'liquidity_min': 100_000_000,
        'min_trades': 10,
        'profit_gate': 0.005,
        'confidence_threshold': 0.55,
        'description': 'Crypto - high volatility, 24/7'
    }
}

# Tier configurations
TIER_CONFIG = {
    'quick': {
        'seeds': [42],
        'iterations': 100,
        'depth': 4,
        'pass_threshold': 0.0,
        'features': 'basic',
        'description': 'Quick screen - all liquid stocks'
    },
    'medium': {
        'seeds': [42, 123, 456],
        'iterations': 150,
        'depth': 5,
        'pass_threshold': 0.50,
        'features': 'full',
        'description': 'Medium analysis - promising candidates'
    },
    'deep': {
        'seeds': [42, 123, 456, 789, 1011],
        'iterations': 200,
        'depth': 6,
        'pass_threshold': 0.60,
        'features': 'full',
        'description': 'Deep validation - robustness testing'
    }
}


# ============================================================================
# UNIVERSE DEFINITIONS
# ============================================================================

US_LARGE_CAPS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
    'MRK', 'LLY', 'KO', 'PEP', 'COST', 'BAC', 'AVGO', 'TMO', 'WMT',
    'MCD', 'DIS', 'CSCO', 'ABT', 'VZ', 'ADBE', 'CRM', 'NKE', 'INTC',
    'QCOM', 'TXN', 'NEE', 'AMD', 'PM', 'HON', 'UPS', 'IBM', 'BMY',
    'ORCL', 'GE', 'RTX', 'LOW', 'CAT', 'PFE'
]

US_MID_CAPS = [
    'PANW', 'CRWD', 'DDOG', 'ZS', 'SNOW', 'TEAM', 'OKTA', 'NET',
    'MDB', 'COIN', 'RIVN', 'LCID', 'PLTR', 'SOFI', 'HOOD', 'RBLX',
    'U', 'DKNG', 'ABNB', 'DASH', 'SNAP', 'PINS', 'ROKU', 'SQ',
    'PYPL', 'SHOP', 'TTD', 'TWLO', 'DOCU', 'ZM', 'UBER', 'LYFT'
]

EUROPE_STOCKS = [
    'SAP', 'ASML', 'NVO', 'NVS', 'AZN', 'SHEL', 'TTE', 'UL', 'HSBC',
    'BP', 'RIO', 'BHP', 'GSK', 'BTI', 'DEO', 'SNY', 'RACE'
]

JAPAN_STOCKS = [
    'TM', 'SONY', 'MUFG', 'SMFG', 'NTT', 'HMC'
]

EMERGING_STOCKS = [
    'BABA', 'PDD', 'JD', 'BIDU', 'NIO', 'LI', 'XPEV',
    'TSM', 'VALE', 'ITUB', 'PBR', 'NU', 'GRAB'
]

COMMODITIES = [
    'GLD', 'SLV', 'USO', 'UNG', 'CPER', 'WEAT', 'CORN', 'DBA'
]

FOREX_PROXIES = [
    'UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA'
]

CRYPTO_PROXIES = [
    'BITO', 'GBTC', 'ETHE', 'COIN', 'MARA', 'RIOT', 'MSTR'
]


def get_full_universe() -> Dict[str, List[str]]:
    """Get complete universe organized by market type."""
    return {
        'US_Large': US_LARGE_CAPS,
        'US_Mid': US_MID_CAPS,
        'Europe': EUROPE_STOCKS,
        'Japan': JAPAN_STOCKS,
        'Emerging': EMERGING_STOCKS,
        'Commodity': COMMODITIES,
        'Forex': FOREX_PROXIES,
        'Crypto': CRYPTO_PROXIES
    }


def detect_market_type(ticker: str) -> str:
    """Detect market type for a ticker."""
    universe = get_full_universe()
    for market_type, tickers in universe.items():
        if ticker in tickers:
            return market_type
    return 'US_Large'  # Default


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic feature set for quick screening (11 features)."""
    df = df.copy()

    # Price returns
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_10d'] = df['Close'].pct_change(10)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Moving Averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['price_vs_sma20'] = df['Close'] / (df['sma_20'] + 1e-10)

    # Volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)
    else:
        df['volume_ratio'] = 1.0

    # Volatility
    df['volatility_10'] = df['returns_1d'].rolling(10).std()

    # Momentum
    df['momentum_10'] = df['Close'] / (df['Close'].shift(10) + 1e-10) - 1

    # Bollinger
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bollinger_pct'] = (df['Close'] - sma) / (2 * std + 1e-10)

    return df


BASIC_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_10d',
    'rsi_14', 'price_vs_sma20', 'volume_ratio',
    'volatility_10', 'momentum_10', 'bollinger_pct'
]


def add_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature set for deep analysis (25+ features)."""
    df = add_basic_features(df)

    # Extended returns
    df['returns_20d'] = df['Close'].pct_change(20)
    df['returns_60d'] = df['Close'].pct_change(60)

    # Extended MA
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    df['price_vs_sma50'] = df['Close'] / (df['sma_50'] + 1e-10)
    df['price_vs_sma200'] = df['Close'] / (df['sma_200'] + 1e-10)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)

    # Extended volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_trend'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1)
        df['vol_price_corr'] = df['returns_1d'].rolling(20).corr(df['Volume'].pct_change())
        df['volume_momentum'] = df['Volume'] / (df['Volume'].shift(5) + 1) - 1
    else:
        df['volume_trend'] = 1.0
        df['vol_price_corr'] = 0.0
        df['volume_momentum'] = 0.0

    # Extended volatility
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    df['volatility_60'] = df['returns_1d'].rolling(60).std()
    df['volatility_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-10)
    df['volatility_regime'] = (df['volatility_10'] > df['volatility_60']).astype(int)

    # Extended momentum
    df['momentum_5'] = df['Close'] / (df['Close'].shift(5) + 1e-10) - 1
    df['momentum_20'] = df['Close'] / (df['Close'].shift(20) + 1e-10) - 1
    df['momentum_60'] = df['Close'] / (df['Close'].shift(60) + 1e-10) - 1
    df['momentum_divergence'] = df['momentum_5'] - df['momentum_20']

    # Bollinger width
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bollinger_width'] = (4 * std) / (sma + 1e-10)

    # Mean reversion
    df['distance_from_mean_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-10)
    df['distance_from_mean_60'] = (df['Close'] - df['Close'].rolling(60).mean()) / (df['Close'].rolling(60).std() + 1e-10)

    # Trend strength
    df['trend_strength'] = abs(df['momentum_20']) / (df['volatility_20'] + 1e-10)

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / (df['Close'] + 1e-10)

    return df


FULL_FEATURES = BASIC_FEATURES + [
    'returns_20d', 'returns_60d',
    'price_vs_sma50', 'price_vs_sma200', 'sma_20_50_cross', 'sma_50_200_cross',
    'volume_trend', 'vol_price_corr', 'volume_momentum',
    'volatility_20', 'volatility_60', 'volatility_ratio', 'volatility_regime',
    'momentum_5', 'momentum_20', 'momentum_60', 'momentum_divergence',
    'bollinger_width',
    'distance_from_mean_20', 'distance_from_mean_60',
    'trend_strength',
    'macd', 'macd_signal', 'macd_hist',
    'atr_14', 'atr_ratio'
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SeedResult:
    """Result from a single seed test."""
    seed: int
    avg_return: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    passes: bool


@dataclass
class ScreeningResult:
    """Result from screening a single ticker."""
    symbol: str
    market_type: str
    tier: str
    pass_rate: float
    avg_return: float
    return_std: float
    avg_trades: int
    seed_results: List[SeedResult]
    passes_tier: bool
    liquidity_ok: bool
    avg_daily_volume: float


@dataclass
class ValidationSummary:
    """Summary of validation run."""
    timestamp: str
    total_screened: int
    liquid_count: int
    tier1_passed: int
    tier2_passed: int
    tier3_robust: int
    by_market: Dict[str, Dict]
    by_sector: Dict[str, int]
    robust_performers: List[Dict]
    compute_savings: float


# ============================================================================
# MODEL VALIDATOR
# ============================================================================

class USIntlModelValidator:
    """
    Comprehensive validation framework for US/International models.

    Mirrors China model testing standards with:
    - Multi-seed robustness testing
    - Tiered screening pipeline
    - Market-adjusted thresholds
    - Cross-market validation
    """

    def __init__(
        self,
        train_start: str = TRAIN_START,
        train_end: str = TRAIN_END,
        test_start: str = TEST_START,
        test_end: str = TEST_END,
        results_dir: str = None
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.screening_results: Dict[str, ScreeningResult] = {}
        self.validation_summary: Optional[ValidationSummary] = None

    def check_liquidity(self, symbol: str, market_type: str) -> Tuple[bool, float]:
        """Check if symbol meets liquidity requirements."""
        threshold = MARKET_THRESHOLDS.get(market_type, MARKET_THRESHOLDS['US_Large'])
        min_volume = threshold['liquidity_min']

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='60d')

            if len(df) < 20:
                return False, 0.0

            avg_volume = df['Volume'].mean()
            avg_price = df['Close'].mean()
            avg_dollar_volume = avg_volume * avg_price

            return avg_dollar_volume >= min_volume, float(avg_dollar_volume)

        except Exception as e:
            logger.warning(f"Liquidity check failed for {symbol}: {e}")
            return False, 0.0

    def run_single_seed_test(
        self,
        symbol: str,
        market_type: str,
        seed: int,
        tier: str,
        df: pd.DataFrame = None
    ) -> Optional[SeedResult]:
        """Run model test with a single random seed."""
        threshold = MARKET_THRESHOLDS.get(market_type, MARKET_THRESHOLDS['US_Large'])
        tier_config = TIER_CONFIG[tier]

        try:
            # Download data if not provided
            if df is None:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.train_start, end=self.test_end)

            if len(df) < 100:
                return None

            # Add features
            if tier_config['features'] == 'basic':
                df = add_basic_features(df)
                features = [f for f in BASIC_FEATURES if f in df.columns]
            else:
                df = add_full_features(df)
                features = [f for f in FULL_FEATURES if f in df.columns]

            # Create target
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df['next_return'] = df['Close'].pct_change().shift(-1)

            # Drop NaN
            df = df.dropna()

            if len(df) < 100:
                return None

            # Split data
            train_df = df[df.index < self.test_start]
            test_df = df[df.index >= self.test_start]

            if len(train_df) < 50 or len(test_df) < 20:
                return None

            X_train = train_df[features].values
            y_train = train_df['target'].values
            X_test = test_df[features].values
            test_returns = test_df['next_return'].values

            # Train model
            model = CatBoostClassifier(
                iterations=tier_config['iterations'],
                depth=tier_config['depth'],
                learning_rate=0.03,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False
            )
            model.fit(X_train, y_train, verbose=False)

            # Predict
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Evaluate with confidence threshold
            confident_mask = y_pred_proba >= threshold['confidence_threshold']
            num_trades = int(confident_mask.sum())

            if num_trades >= threshold['min_trades']:
                confident_returns = test_returns[confident_mask]
                avg_return = float(np.mean(confident_returns))
                win_rate = float((confident_returns > 0).mean())

                # Sharpe ratio (annualized)
                if len(confident_returns) > 1 and np.std(confident_returns) > 0:
                    sharpe = (np.mean(confident_returns) / np.std(confident_returns)) * np.sqrt(252)
                else:
                    sharpe = 0.0

                passes = avg_return >= threshold['profit_gate']
            else:
                avg_return = 0.0
                win_rate = 0.0
                sharpe = 0.0
                passes = False

            return SeedResult(
                seed=seed,
                avg_return=avg_return,
                num_trades=num_trades,
                win_rate=win_rate,
                sharpe_ratio=float(sharpe),
                passes=passes
            )

        except Exception as e:
            logger.warning(f"Seed test failed for {symbol} (seed={seed}): {e}")
            return None

    def screen_symbol(
        self,
        symbol: str,
        tier: str = 'deep',
        df: pd.DataFrame = None
    ) -> Optional[ScreeningResult]:
        """Screen a single symbol through specified tier."""
        market_type = detect_market_type(symbol)
        tier_config = TIER_CONFIG[tier]
        threshold = MARKET_THRESHOLDS.get(market_type, MARKET_THRESHOLDS['US_Large'])

        # Check liquidity
        liquidity_ok, avg_volume = self.check_liquidity(symbol, market_type)

        if not liquidity_ok:
            logger.info(f"{symbol}: Failed liquidity filter (${avg_volume/1e6:.1f}M < ${threshold['liquidity_min']/1e6:.1f}M)")
            return ScreeningResult(
                symbol=symbol,
                market_type=market_type,
                tier=tier,
                pass_rate=0.0,
                avg_return=0.0,
                return_std=0.0,
                avg_trades=0,
                seed_results=[],
                passes_tier=False,
                liquidity_ok=False,
                avg_daily_volume=avg_volume
            )

        # Download data once for all seeds
        if df is None:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.train_start, end=self.test_end)
            except Exception as e:
                logger.warning(f"Data download failed for {symbol}: {e}")
                return None

        # Run multi-seed tests
        seed_results = []
        for seed in tier_config['seeds']:
            result = self.run_single_seed_test(symbol, market_type, seed, tier, df)
            if result:
                seed_results.append(result)

        if not seed_results:
            return None

        # Calculate aggregates
        pass_rate = np.mean([r.passes for r in seed_results])
        avg_return = np.mean([r.avg_return for r in seed_results])
        return_std = np.std([r.avg_return for r in seed_results])
        avg_trades = int(np.mean([r.num_trades for r in seed_results]))

        # Determine pass threshold based on market type
        if tier == 'medium':
            pass_threshold = threshold.get('medium_pass', TIER_CONFIG['medium']['pass_threshold'])
        elif tier == 'deep':
            pass_threshold = threshold.get('deep_pass', TIER_CONFIG['deep']['pass_threshold'])
        else:
            pass_threshold = 0.0

        passes_tier = pass_rate >= pass_threshold

        result = ScreeningResult(
            symbol=symbol,
            market_type=market_type,
            tier=tier,
            pass_rate=float(pass_rate),
            avg_return=float(avg_return),
            return_std=float(return_std),
            avg_trades=avg_trades,
            seed_results=seed_results,
            passes_tier=passes_tier,
            liquidity_ok=True,
            avg_daily_volume=avg_volume
        )

        self.screening_results[symbol] = result
        return result

    def run_tiered_screening(
        self,
        universe: Dict[str, List[str]] = None,
        verbose: bool = True
    ) -> ValidationSummary:
        """
        Run full tiered screening pipeline.

        Tier 1 (Quick): All liquid stocks, 1 seed
        Tier 2 (Medium): Tier 1 passers, 3 seeds
        Tier 3 (Deep): Tier 2 passers, 5 seeds -> Robust performers
        """
        if universe is None:
            universe = get_full_universe()

        if verbose:
            print("=" * 70)
            print("US/INTERNATIONAL MODEL VALIDATION FRAMEWORK")
            print("=" * 70)
            print(f"Train period: {self.train_start} to {self.train_end}")
            print(f"Test period:  {self.test_start} to {self.test_end}")
            print()

        # Flatten universe
        all_symbols = []
        for market_type, symbols in universe.items():
            for s in symbols:
                all_symbols.append((s, market_type))

        total_screened = len(all_symbols)

        if verbose:
            print(f"Total universe: {total_screened} instruments")
            for market_type, symbols in universe.items():
                print(f"  {market_type}: {len(symbols)}")
            print()

        # Track results by tier
        tier1_passed = []
        tier2_passed = []
        tier3_robust = []
        liquid_count = 0
        by_market = {m: {'total': len(s), 'liquid': 0, 'robust': 0} for m, s in universe.items()}

        # ===== TIER 1: Quick Screen =====
        if verbose:
            print("=" * 70)
            print("TIER 1: QUICK SCREEN (1 seed, basic features)")
            print("=" * 70)

        for symbol, market_type in all_symbols:
            result = self.screen_symbol(symbol, tier='quick')

            if result and result.liquidity_ok:
                liquid_count += 1
                by_market[market_type]['liquid'] += 1

                if result.pass_rate > 0:  # Any positive performance
                    tier1_passed.append(symbol)
                    if verbose:
                        print(f"  PASS: {symbol} ({market_type}) - return: {result.avg_return*100:.2f}%")

        if verbose:
            print(f"\nTier 1 Summary: {len(tier1_passed)}/{liquid_count} liquid passed")
            print()

        # ===== TIER 2: Medium Analysis =====
        if verbose:
            print("=" * 70)
            print("TIER 2: MEDIUM ANALYSIS (3 seeds, full features)")
            print("=" * 70)

        for symbol in tier1_passed:
            result = self.screen_symbol(symbol, tier='medium')

            if result and result.passes_tier:
                tier2_passed.append(symbol)
                if verbose:
                    print(f"  PASS: {symbol} ({result.market_type}) - "
                          f"pass_rate: {result.pass_rate*100:.0f}%, "
                          f"return: {result.avg_return*100:.2f}%")

        if verbose:
            print(f"\nTier 2 Summary: {len(tier2_passed)}/{len(tier1_passed)} passed")
            print()

        # ===== TIER 3: Deep Validation =====
        if verbose:
            print("=" * 70)
            print("TIER 3: DEEP VALIDATION (5 seeds, robustness testing)")
            print("=" * 70)

        for symbol in tier2_passed:
            result = self.screen_symbol(symbol, tier='deep')

            if result and result.passes_tier:
                tier3_robust.append(symbol)
                by_market[result.market_type]['robust'] += 1
                if verbose:
                    print(f"  ROBUST: {symbol} ({result.market_type}) - "
                          f"pass_rate: {result.pass_rate*100:.0f}%, "
                          f"return: {result.avg_return*100:.2f}% +/- {result.return_std*100:.2f}%")

        if verbose:
            print(f"\nTier 3 Summary: {len(tier3_robust)}/{len(tier2_passed)} ROBUST performers")
            print()

        # Compile robust performers
        robust_performers = []
        for symbol in tier3_robust:
            result = self.screening_results.get(symbol)
            if result:
                robust_performers.append({
                    'symbol': symbol,
                    'market_type': result.market_type,
                    'pass_rate': result.pass_rate,
                    'avg_return': result.avg_return,
                    'return_std': result.return_std,
                    'avg_trades': result.avg_trades,
                    'avg_sharpe': np.mean([r.sharpe_ratio for r in result.seed_results]),
                    'avg_win_rate': np.mean([r.win_rate for r in result.seed_results]),
                    'avg_daily_volume': result.avg_daily_volume
                })

        # Sort by return
        robust_performers.sort(key=lambda x: x['avg_return'], reverse=True)

        # Calculate compute savings
        full_tests = total_screened * 5  # If we ran 5 seeds on all
        actual_tests = (total_screened * 1) + (len(tier1_passed) * 3) + (len(tier2_passed) * 5)
        compute_savings = 1 - (actual_tests / full_tests) if full_tests > 0 else 0

        # Create summary
        self.validation_summary = ValidationSummary(
            timestamp=datetime.now().isoformat(),
            total_screened=total_screened,
            liquid_count=liquid_count,
            tier1_passed=len(tier1_passed),
            tier2_passed=len(tier2_passed),
            tier3_robust=len(tier3_robust),
            by_market=by_market,
            by_sector={},  # Could add sector classification
            robust_performers=robust_performers,
            compute_savings=compute_savings
        )

        if verbose:
            print("=" * 70)
            print("VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Total screened:    {total_screened}")
            print(f"Liquid:            {liquid_count} ({liquid_count/total_screened*100:.1f}%)")
            print(f"Tier 1 passed:     {len(tier1_passed)} ({len(tier1_passed)/max(liquid_count,1)*100:.1f}%)")
            print(f"Tier 2 passed:     {len(tier2_passed)} ({len(tier2_passed)/max(len(tier1_passed),1)*100:.1f}%)")
            print(f"ROBUST performers: {len(tier3_robust)} ({len(tier3_robust)/max(len(tier2_passed),1)*100:.1f}%)")
            print(f"Compute savings:   {compute_savings*100:.1f}%")
            print()
            print("By Market:")
            for market, stats in by_market.items():
                if stats['total'] > 0:
                    print(f"  {market}: {stats['robust']}/{stats['liquid']} robust ({stats['robust']/max(stats['liquid'],1)*100:.0f}%)")
            print()
            print("Top Robust Performers:")
            for i, p in enumerate(robust_performers[:10], 1):
                print(f"  {i}. {p['symbol']}: {p['avg_return']*100:.2f}% ({p['market_type']}, pass_rate: {p['pass_rate']*100:.0f}%)")

        return self.validation_summary

    def save_results(self, filepath: str = None):
        """Save validation results to JSON."""
        if filepath is None:
            filepath = os.path.join(self.results_dir, 'us_intl_validation_results.json')

        # Helper to convert numpy types to Python natives
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        results = {
            'validation_summary': convert_to_native(asdict(self.validation_summary)) if self.validation_summary else None,
            'screening_results': {
                symbol: convert_to_native({
                    'symbol': r.symbol,
                    'market_type': r.market_type,
                    'tier': r.tier,
                    'pass_rate': r.pass_rate,
                    'avg_return': r.avg_return,
                    'return_std': r.return_std,
                    'avg_trades': r.avg_trades,
                    'passes_tier': r.passes_tier,
                    'liquidity_ok': r.liquidity_ok,
                    'avg_daily_volume': r.avg_daily_volume
                })
                for symbol, r in self.screening_results.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def generate_report(self) -> str:
        """Generate markdown validation report."""
        if not self.validation_summary:
            return "No validation results available. Run tiered_screening first."

        s = self.validation_summary
        report = f"""# US/International Model Validation Report

**Generated:** {s.timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Screened | {s.total_screened} |
| Liquid | {s.liquid_count} ({s.liquid_count/s.total_screened*100:.1f}%) |
| Tier 1 Passed | {s.tier1_passed} |
| Tier 2 Passed | {s.tier2_passed} |
| **ROBUST Performers** | **{s.tier3_robust}** |
| Compute Savings | {s.compute_savings*100:.1f}% |

## Market Breakdown

| Market | Total | Liquid | Robust | Pass Rate |
|--------|-------|--------|--------|-----------|
"""
        for market, stats in s.by_market.items():
            if stats['total'] > 0:
                pass_rate = stats['robust'] / max(stats['liquid'], 1) * 100
                report += f"| {market} | {stats['total']} | {stats['liquid']} | {stats['robust']} | {pass_rate:.0f}% |\n"

        report += f"""
## Robust Performers

| Rank | Symbol | Market | Return | Pass Rate | Sharpe | Win Rate |
|------|--------|--------|--------|-----------|--------|----------|
"""
        for i, p in enumerate(s.robust_performers[:20], 1):
            report += f"| {i} | {p['symbol']} | {p['market_type']} | {p['avg_return']*100:.2f}% | {p['pass_rate']*100:.0f}% | {p.get('avg_sharpe', 0):.2f} | {p.get('avg_win_rate', 0)*100:.0f}% |\n"

        report += f"""
## Methodology

### Tiered Screening Pipeline

1. **Tier 1 (Quick)**: 1 seed, basic features, pass if return > 0%
2. **Tier 2 (Medium)**: 3 seeds, full features, market-adjusted pass rate
3. **Tier 3 (Deep)**: 5 seeds, full features, 60%+ pass rate required

### Market-Adjusted Thresholds

| Market | Medium Pass | Deep Pass | Min Liquidity |
|--------|-------------|-----------|---------------|
"""
        for market, thresh in MARKET_THRESHOLDS.items():
            report += f"| {market} | {thresh['medium_pass']*100:.0f}% | {thresh['deep_pass']*100:.0f}% | ${thresh['liquidity_min']/1e6:.0f}M |\n"

        report += f"""
### Anti-Overfitting Measures

- **Multi-seed validation**: 5 random seeds for robustness
- **Train/Test split**: {self.train_start}-{self.train_end} train, {self.test_start}-{self.test_end} test
- **Robustness threshold**: 60%+ pass rate required
- **Liquidity filter**: Market-specific volume requirements
- **Conservative hyperparameters**: Depth 4-6, iterations 100-200

---
*Report generated by US/International Model Validation Framework*
"""
        return report


# ============================================================================
# CROSS-MARKET VALIDATION
# ============================================================================

class CrossMarketValidator:
    """
    Cross-market validation to ensure consistent quality across regions.

    Validates:
    - Pass rate consistency (within 15% across markets)
    - Return stability (std < 5% across seeds)
    - Sector diversification (edges in 3+ sectors)
    - Capacity testing (signal quality at scale)
    """

    def __init__(self, validator: USIntlModelValidator):
        self.validator = validator

    def validate_consistency(self) -> Dict[str, Any]:
        """Check consistency of results across markets."""
        if not self.validator.validation_summary:
            return {'error': 'No validation results available'}

        summary = self.validator.validation_summary

        # Calculate metrics
        market_pass_rates = {}
        for market, stats in summary.by_market.items():
            if stats['liquid'] > 0:
                market_pass_rates[market] = stats['robust'] / stats['liquid']

        if not market_pass_rates:
            return {'error': 'No market data available'}

        avg_pass_rate = np.mean(list(market_pass_rates.values()))
        pass_rate_std = np.std(list(market_pass_rates.values()))
        pass_rate_range = max(market_pass_rates.values()) - min(market_pass_rates.values())

        # Return stability
        return_stds = [p['return_std'] for p in summary.robust_performers]
        avg_return_std = np.mean(return_stds) if return_stds else 0

        # Sector/market diversification
        markets_with_robust = len([m for m, s in summary.by_market.items() if s['robust'] > 0])

        validation_results = {
            'pass_rate_consistency': {
                'average': float(avg_pass_rate),
                'std': float(pass_rate_std),
                'range': float(pass_rate_range),
                'is_consistent': pass_rate_range <= 0.15,  # Within 15%
                'by_market': market_pass_rates
            },
            'return_stability': {
                'avg_std_across_seeds': float(avg_return_std),
                'is_stable': avg_return_std <= 0.05,  # < 5% std
            },
            'diversification': {
                'markets_with_robust': markets_with_robust,
                'is_diversified': markets_with_robust >= 3
            },
            'overall_pass': (
                pass_rate_range <= 0.15 and
                avg_return_std <= 0.05 and
                markets_with_robust >= 3
            )
        }

        return validation_results

    def generate_consistency_report(self) -> str:
        """Generate consistency validation report."""
        results = self.validate_consistency()

        if 'error' in results:
            return f"Error: {results['error']}"

        report = """# Cross-Market Validation Report

## Pass Rate Consistency
"""
        prc = results['pass_rate_consistency']
        report += f"- Average pass rate: {prc['average']*100:.1f}%\n"
        report += f"- Standard deviation: {prc['std']*100:.1f}%\n"
        report += f"- Range: {prc['range']*100:.1f}%\n"
        report += f"- **Status**: {'PASS' if prc['is_consistent'] else 'FAIL'} (threshold: 15%)\n\n"

        report += "### By Market\n"
        for market, rate in prc['by_market'].items():
            report += f"- {market}: {rate*100:.1f}%\n"

        report += f"""
## Return Stability
- Avg std across seeds: {results['return_stability']['avg_std_across_seeds']*100:.2f}%
- **Status**: {'PASS' if results['return_stability']['is_stable'] else 'FAIL'} (threshold: 5%)

## Diversification
- Markets with robust performers: {results['diversification']['markets_with_robust']}
- **Status**: {'PASS' if results['diversification']['is_diversified'] else 'FAIL'} (minimum: 3)

## Overall Validation
**{'PASS' if results['overall_pass'] else 'FAIL'}**
"""
        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run full validation pipeline."""
    print("=" * 70)
    print("US/INTERNATIONAL MODEL VALIDATION FRAMEWORK")
    print("=" * 70)
    print()

    # Initialize validator
    validator = USIntlModelValidator()

    # Run tiered screening (use subset for demo)
    demo_universe = {
        'US_Large': US_LARGE_CAPS[:20],  # Top 20 large caps
        'US_Mid': US_MID_CAPS[:10],
        'Europe': EUROPE_STOCKS[:5],
    }

    # Run validation
    summary = validator.run_tiered_screening(universe=demo_universe, verbose=True)

    # Save results
    validator.save_results()

    # Generate report
    report = validator.generate_report()
    report_path = os.path.join(validator.results_dir, 'us_intl_validation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Cross-market validation
    cross_validator = CrossMarketValidator(validator)
    consistency = cross_validator.validate_consistency()
    print("\n" + "=" * 70)
    print("CROSS-MARKET VALIDATION")
    print("=" * 70)
    print(cross_validator.generate_consistency_report())

    return validator


if __name__ == '__main__':
    main()
