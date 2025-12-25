"""
US Model Fixing9 Implementation
================================
Implements critical fixes for US/International model based on fixing9.pdf

P0 (Critical): Fix missing features, reduce feature count, lower SNR thresholds
P1 (High): New profit score, aggressive position sizing, dynamic profit targets
P2 (Medium): Sector rotation, correlation limits

Last Updated: 2025-12-24
"""

import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# =============================================================================
# P0 FIX 1: SAFE FEATURE DEFAULTS (No more zeros!)
# =============================================================================

# Historical means for critical features - NEVER use 0 for these
SAFE_FEATURE_DEFAULTS = {
    # US Macro features
    'vix': 20.0,              # Normal VIX level
    'vix_return': 0.0,        # No change (centered)
    'vix_relative': 1.0,      # Normal relative level
    'spy_return': 0.001,      # Small positive drift (market bias)
    'spy_beta': 1.0,          # Market beta
    'dxy_return': 0.0,        # USD index neutral
    'gld_return': 0.0,        # Gold neutral

    # Technical indicators
    'rsi_14': 50.0,           # Neutral RSI
    'rsi_21': 50.0,           # Neutral RSI
    'macd': 0.0,              # Neutral MACD
    'macd_signal': 0.0,       # Neutral signal
    'macd_hist': 0.0,         # Neutral histogram
    'stoch_k': 50.0,          # Neutral stochastic
    'stoch_d': 50.0,          # Neutral stochastic

    # Volatility features
    'parkinson_vol_20': 0.015,    # 1.5% daily vol (typical)
    'yz_vol_20': 0.015,           # Yang-Zhang vol
    'hist_vol_20': 0.015,         # Historical vol
    'volatility': 0.015,          # General vol
    'volatility_ratio_5_20': 1.0, # Normal ratio
    'vol_regime': 0.5,            # Neutral regime
    'atr_14': 0.02,               # 2% ATR typical

    # Volume features
    'volume_ratio': 1.0,          # Normal volume
    'volume_sma_ratio_20': 1.0,   # Normal relative volume
    'volume_price_correlation': 0.0,  # No correlation

    # Price action
    'returns_1d': 0.0,            # No return (centered)
    'returns_5d': 0.0,            # No return
    'returns_20d': 0.0,           # No return
    'high_low_ratio': 1.02,       # 2% typical range
    'close_open_ratio': 1.0,      # Neutral close
    'price_sma_ratio_20': 1.0,    # At SMA

    # Market structure
    'advance_decline_ratio': 1.0, # Balanced
    'market_breadth': 0.5,        # Neutral
    'sector_strength': 0.0,       # Neutral
    'regime_score': 0.5,          # Neutral regime
    'regime': 0.5,                # Neutral

    # Seasonal
    'day_of_week': 2,             # Wednesday (mid-week)
    'month_of_year': 6,           # June (mid-year)
    'trend_strength': 0.0,        # No trend
}


def safe_fill_missing_features(df: pd.DataFrame,
                                feature_means: Optional[Dict[str, float]] = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Fill missing features with safe historical means instead of zeros.

    CRITICAL FIX: Using 0 for features like VIX (normally 10-30) completely
    breaks the model. This function uses realistic defaults.

    Args:
        df: DataFrame with features
        feature_means: Optional custom means (uses SAFE_FEATURE_DEFAULTS if None)
        verbose: Log which features were filled

    Returns:
        DataFrame with NaN filled using safe values
    """
    if feature_means is None:
        feature_means = SAFE_FEATURE_DEFAULTS.copy()

    filled_features = []

    for col in df.columns:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()

            if col in feature_means:
                # Use known safe default
                fill_value = feature_means[col]
                df[col] = df[col].fillna(fill_value)
                filled_features.append((col, fill_value, nan_count))
            else:
                # For unknown features, use smart defaults based on name
                if 'return' in col.lower() or 'change' in col.lower():
                    fill_value = 0.0  # Returns should be centered at 0
                elif 'vol' in col.lower():
                    fill_value = 0.015  # Typical volatility
                elif 'rsi' in col.lower():
                    fill_value = 50.0  # Neutral RSI
                elif 'ratio' in col.lower():
                    fill_value = 1.0  # Neutral ratio
                elif 'score' in col.lower() or 'regime' in col.lower():
                    fill_value = 0.5  # Neutral score
                else:
                    # Last resort: use column median if available, else 0
                    col_median = df[col].median()
                    fill_value = col_median if not pd.isna(col_median) else 0.0

                df[col] = df[col].fillna(fill_value)
                filled_features.append((col, fill_value, nan_count))

    if verbose and filled_features:
        logger.info(f"[FIXING9] Filled {len(filled_features)} features with safe defaults:")
        for feat, val, count in filled_features[:10]:  # Show first 10
            logger.info(f"  - {feat}: {val} ({count} NaN values)")
        if len(filled_features) > 10:
            logger.info(f"  ... and {len(filled_features) - 10} more features")

    return df


# =============================================================================
# P0 FIX 2: CORE FEATURES (30 instead of 100+)
# =============================================================================

US_CORE_FEATURES = [
    # 1. PRICE ACTION (7 features)
    'returns_1d', 'returns_5d', 'returns_20d',
    'high_low_ratio', 'close_open_ratio',
    'atr_14', 'price_sma_ratio_20',

    # 2. MOMENTUM (5 features)
    'rsi_14', 'macd', 'macd_signal',
    'stoch_k', 'stoch_d',

    # 3. VOLATILITY (4 features)
    'parkinson_vol_20', 'yz_vol_20',
    'volatility_ratio_5_20', 'vol_regime',

    # 4. VOLUME (3 features)
    'volume_ratio', 'volume_sma_ratio_20',
    'volume_price_correlation',

    # 5. US MACRO (5 features)
    'vix', 'vix_return', 'vix_relative',
    'spy_return', 'spy_beta',

    # 6. MARKET STRUCTURE (4 features)
    'advance_decline_ratio', 'market_breadth',
    'sector_strength', 'regime_score',

    # 7. SEASONAL/TREND (2 features)
    'day_of_week', 'trend_strength',
]  # Total: 30 features


def select_core_features(df: pd.DataFrame,
                         core_features: List[str] = None,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Select only core features to reduce overfitting.

    Current model uses 100+ features which causes massive overfitting.
    China model uses 10-30 features and works much better.

    Args:
        df: DataFrame with all features
        core_features: List of feature names to keep (uses US_CORE_FEATURES if None)
        verbose: Log feature selection info

    Returns:
        DataFrame with only core features
    """
    if core_features is None:
        core_features = US_CORE_FEATURES

    # Find which core features exist in the data
    available_features = [f for f in core_features if f in df.columns]
    missing_features = [f for f in core_features if f not in df.columns]

    if verbose:
        logger.info(f"[FIXING9] Feature selection: {len(available_features)}/{len(core_features)} core features available")
        if missing_features:
            logger.warning(f"[FIXING9] Missing core features: {missing_features[:5]}...")

    # If too few core features available, fall back to all features
    if len(available_features) < 10:
        logger.warning(f"[FIXING9] Only {len(available_features)} core features found, using all features")
        return df

    # Select only core features
    result = df[available_features].copy()

    if verbose:
        logger.info(f"[FIXING9] Reduced features from {len(df.columns)} to {len(result.columns)}")

    return result


# =============================================================================
# P0 FIX 3: IMPROVED SNR THRESHOLDS & CONFIDENCE
# =============================================================================

# New SNR thresholds (lower = more signals)
SNR_THRESHOLDS_FIXING9 = {
    'stock': 0.3,          # Was 0.5 - lowered for more signals
    'china_stock': 0.3,    # Same as stock
    'crypto': 0.5,         # Was 0.8 - lowered
    'forex': 0.2,          # Was 0.3 - lowered
    'commodity': 0.3,      # Was 0.5 - lowered
    'etf': 0.3,            # Was 0.5 - lowered
}

# Steepness of sigmoid curve (higher = sharper transition)
SNR_STEEPNESS = {
    'stock': 1.8,          # Was 1.5 - steeper for stocks
    'china_stock': 1.8,
    'crypto': 1.5,
    'forex': 2.0,          # Steeper for forex (small moves matter)
    'commodity': 1.8,
    'etf': 1.8,
}


def calculate_confidence_fixing9(predicted_return: float,
                                  volatility: float,
                                  asset_class: str = 'stock') -> Tuple[float, float]:
    """
    Improved confidence calculation with lower thresholds.

    Changes from original:
    - Lower SNR thresholds (0.3 vs 0.5 for stocks)
    - Steeper sigmoid (1.8 vs 1.5)
    - Higher minimum confidence (0.40 vs 0.30)
    - Better range utilization (0.40-0.95)

    Args:
        predicted_return: Model's predicted return
        volatility: Historical volatility
        asset_class: Type of asset

    Returns:
        (confidence, snr) tuple
    """
    # Avoid division by zero - minimum 0.5% volatility
    volatility = max(volatility, 0.005)

    # Calculate signal-to-noise ratio
    snr = abs(predicted_return) / volatility

    # Get asset-specific parameters
    threshold = SNR_THRESHOLDS_FIXING9.get(asset_class, 0.3)
    steepness = SNR_STEEPNESS.get(asset_class, 1.8)

    # Sigmoid transformation
    sigmoid_input = steepness * (snr - threshold)
    sigmoid = 1 / (1 + math.exp(-sigmoid_input))

    # Map to confidence range: 0.40 to 0.95 (higher floor than before)
    confidence = 0.40 + 0.55 * sigmoid

    # Ensure bounds
    confidence = min(0.95, max(0.40, confidence))

    return confidence, snr


# =============================================================================
# P1 FIX 4: PROFIT-MAXIMIZING SCORE
# =============================================================================

def calculate_profit_score_fixing9(predicted_return: float,
                                    confidence: float,
                                    volatility: float,
                                    volume_ratio: float = 1.0,
                                    rsi: float = 50.0,
                                    regime_multiplier: float = 1.0) -> float:
    """
    Less conservative profit score calculation.

    Changes from original:
    - Less aggressive volatility penalty
    - More generous volume and RSI factors
    - Multiplicative scoring (not additive)
    - Non-linear scaling for high scores

    Args:
        predicted_return: Model's predicted return
        confidence: Confidence level (0-1)
        volatility: Historical volatility
        volume_ratio: Current volume vs average
        rsi: RSI indicator value
        regime_multiplier: Market regime adjustment

    Returns:
        Profit score (0-20 scale)
    """
    # 1. Base expected profit (weighted by confidence)
    expected_profit = abs(predicted_return) * confidence

    # 2. Volatility adjustment (less aggressive penalty than before)
    if volatility < 0.02:      # <2% vol -> bonus
        vol_factor = 1.3
    elif volatility < 0.04:    # 2-4% -> neutral
        vol_factor = 1.0
    elif volatility < 0.08:    # 4-8% -> slight penalty
        vol_factor = 0.8
    else:                      # >8% -> bigger penalty
        vol_factor = 0.6

    # 3. Volume confirmation (reward high volume on moves)
    if volume_ratio > 1.5:
        volume_factor = 1.2
    elif volume_ratio > 1.2:
        volume_factor = 1.1
    elif volume_ratio > 0.8:
        volume_factor = 1.0
    else:
        volume_factor = 0.9

    # 4. RSI positioning (wider sweet spot than before)
    if 35 < rsi < 75:          # Wider range (was 40-70)
        rsi_factor = 1.2
    elif 25 < rsi < 85:
        rsi_factor = 1.0
    else:
        rsi_factor = 0.85      # Less penalty (was 0.8)

    # 5. Regime multiplier
    regime_factor = max(0.5, min(1.5, regime_multiplier))

    # 6. Combine all factors (multiplicative)
    score = (expected_profit * 100) * vol_factor * volume_factor * rsi_factor * regime_factor

    # 7. Non-linear scaling for high scores (diminishing returns above 10)
    if score > 10:
        score = 10 + (score - 10) * 0.5

    # Cap at 20
    return min(20, max(0, score))


# =============================================================================
# P1: AGGRESSIVE POSITION SIZING (Kelly-based)
# =============================================================================

def calculate_position_size_fixing9(predicted_return: float,
                                     confidence: float,
                                     volatility: float,
                                     max_position: float = 0.10,
                                     min_position: float = 0.02) -> float:
    """
    Kelly-criterion inspired position sizing.

    More aggressive than fixed position sizes, but still bounded.

    Args:
        predicted_return: Model's predicted return
        confidence: Confidence level (0-1)
        volatility: Historical volatility
        max_position: Maximum position size (default 10%)
        min_position: Minimum position size (default 2%)

    Returns:
        Position size as fraction of portfolio (0-1)
    """
    # Avoid division by zero
    volatility = max(volatility, 0.01)

    # Win probability from confidence
    win_prob = confidence

    # Win/loss ratio from predicted return vs volatility
    win_loss_ratio = abs(predicted_return) / volatility

    # Modified Kelly fraction
    # Kelly formula: f = (p * b - q) / b where p=win prob, b=win/loss ratio, q=1-p
    if win_loss_ratio > 0:
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    else:
        kelly_fraction = 0.0

    # Apply fractional Kelly based on confidence
    if confidence > 0.75 and win_loss_ratio > 2:
        position_size = kelly_fraction * 0.8    # 80% Kelly for high confidence
    elif confidence > 0.65:
        position_size = kelly_fraction * 0.5    # 50% Kelly
    else:
        position_size = kelly_fraction * 0.3    # 30% Kelly (conservative)

    # Apply bounds
    position_size = max(min_position, min(max_position, position_size))

    return position_size


# =============================================================================
# P1: DYNAMIC PROFIT TARGETS
# =============================================================================

def calculate_profit_targets_fixing9(predicted_return: float,
                                      volatility: float,
                                      regime: str = 'neutral') -> Dict[str, float]:
    """
    Dynamic take-profit and stop-loss based on prediction and volatility.

    Args:
        predicted_return: Model's predicted return
        volatility: Historical volatility
        regime: Market regime ('bull_momentum', 'bear_momentum', 'neutral')

    Returns:
        Dict with 'take_profit', 'stop_loss', 'time_exit_days'
    """
    # Base target: aim for 1.5x predicted return
    base_target = abs(predicted_return) * 1.5

    # Adjust based on volatility
    if volatility < 0.02:
        # Low vol -> tighter targets, shorter time
        take_profit = base_target * 0.8
        stop_loss = -abs(predicted_return) * 1.0
        time_exit = 5
    elif volatility < 0.05:
        # Medium vol
        take_profit = base_target * 1.0
        stop_loss = -abs(predicted_return) * 1.5
        time_exit = 4
    else:
        # High vol -> wider targets
        take_profit = base_target * 1.2
        stop_loss = -abs(predicted_return) * 2.0
        time_exit = 3

    # Regime adjustments
    if regime == 'bull_momentum':
        take_profit *= 1.2   # Let winners run
        stop_loss *= 0.8     # Wider stops
    elif regime == 'bear_momentum':
        take_profit *= 0.8   # Take profits faster
        stop_loss *= 1.2     # Tighter stops

    # Minimum bounds
    take_profit = max(take_profit, 0.01)   # At least 1%
    stop_loss = min(stop_loss, -0.02)      # At least -2%

    return {
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'time_exit_days': time_exit
    }


# =============================================================================
# P2: SECTOR ROTATION WEIGHTS
# =============================================================================

def get_sector_rotation_weights(transition_day: int = 0,
                                 vix_level: float = 20.0) -> Dict[str, float]:
    """
    Optimal sector allocation based on market transition phase.

    Args:
        transition_day: Days into the transition (0 = just started)
        vix_level: Current VIX level

    Returns:
        Dict of sector -> weight
    """
    # High VIX override: defensive sectors
    if vix_level > 30:
        return {
            'Healthcare': 0.25,
            'Consumer Staples': 0.25,
            'Utilities': 0.20,
            'Technology': 0.15,
            'Others': 0.15
        }

    if transition_day < 3:
        # EARLY TRANSITION (Days 0-3): Growth sectors lead
        return {
            'Technology': 0.30,
            'Consumer Discretionary': 0.25,
            'Financials': 0.20,
            'Industrials': 0.15,
            'Others': 0.10
        }
    elif transition_day < 7:
        # MID TRANSITION (Days 4-7): Broadening
        return {
            'Technology': 0.25,
            'Communication Services': 0.20,
            'Healthcare': 0.18,
            'Consumer Staples': 0.17,
            'Energy': 0.10,
            'Others': 0.10
        }
    else:
        # LATE TRANSITION (Days 8+): Defensive rotation
        return {
            'Healthcare': 0.25,
            'Consumer Staples': 0.20,
            'Utilities': 0.18,
            'Real Estate': 0.15,
            'Technology': 0.12,
            'Others': 0.10
        }


# =============================================================================
# P2: CORRELATION LIMITS
# =============================================================================

def check_correlation_limits(positions: List[Dict[str, Any]],
                              new_ticker: str,
                              new_sector: str,
                              max_sector_weight: float = 0.40,
                              max_correlation: float = 0.7) -> Tuple[bool, str]:
    """
    Check if adding a new position violates correlation/concentration limits.

    Args:
        positions: Current positions with 'ticker', 'sector', 'weight' keys
        new_ticker: Ticker to potentially add
        new_sector: Sector of the new ticker
        max_sector_weight: Maximum weight in any single sector
        max_correlation: Maximum allowed correlation (future enhancement)

    Returns:
        (can_add, reason) tuple
    """
    if not positions:
        return True, "No existing positions"

    # Calculate current sector weights
    sector_weights = {}
    for pos in positions:
        sector = pos.get('sector', 'Unknown')
        weight = pos.get('weight', 0)
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

    # Check if adding new position would exceed sector limit
    current_sector_weight = sector_weights.get(new_sector, 0)
    if current_sector_weight >= max_sector_weight:
        return False, f"Sector {new_sector} already at {current_sector_weight:.1%} (max {max_sector_weight:.1%})"

    # Check for duplicate ticker
    existing_tickers = [pos.get('ticker') for pos in positions]
    if new_ticker in existing_tickers:
        return False, f"Already holding {new_ticker}"

    # Check maximum number of positions per sector
    sector_counts = {}
    for pos in positions:
        sector = pos.get('sector', 'Unknown')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if sector_counts.get(new_sector, 0) >= 3:
        return False, f"Already have 3 positions in {new_sector}"

    return True, "Passed all checks"


# =============================================================================
# VALIDATION & TESTING
# =============================================================================

def validate_fixing9_implementation(X_train: pd.DataFrame = None,
                                     X_predict: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Validate that fixing9 is properly implemented.

    Returns:
        Dict with test results
    """
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'details': []
    }

    # Test 1: Safe fill doesn't use zeros for critical features
    test_df = pd.DataFrame({
        'vix': [np.nan, np.nan],
        'spy_return': [np.nan, np.nan],
        'volume_ratio': [np.nan, np.nan],
    })
    filled_df = safe_fill_missing_features(test_df.copy(), verbose=False)

    if filled_df['vix'].iloc[0] == 20.0:
        results['tests_passed'] += 1
        results['details'].append("PASS: VIX filled with 20.0 (not 0)")
    else:
        results['tests_failed'] += 1
        results['details'].append(f"FAIL: VIX filled with {filled_df['vix'].iloc[0]} (expected 20.0)")

    # Test 2: Confidence calculation works with new thresholds
    conf, snr = calculate_confidence_fixing9(0.02, 0.015, 'stock')
    if 0.40 <= conf <= 0.95:
        results['tests_passed'] += 1
        results['details'].append(f"PASS: Confidence in valid range: {conf:.2%}")
    else:
        results['tests_failed'] += 1
        results['details'].append(f"FAIL: Confidence out of range: {conf:.2%}")

    # Test 3: New threshold is lower than old
    if SNR_THRESHOLDS_FIXING9['stock'] < 0.5:
        results['tests_passed'] += 1
        results['details'].append(f"PASS: Stock SNR threshold lowered to {SNR_THRESHOLDS_FIXING9['stock']}")
    else:
        results['tests_failed'] += 1
        results['details'].append(f"FAIL: Stock SNR threshold not lowered")

    # Test 4: Profit score calculation
    score = calculate_profit_score_fixing9(0.03, 0.70, 0.02, 1.2, 55)
    if 0 < score <= 20:
        results['tests_passed'] += 1
        results['details'].append(f"PASS: Profit score calculated: {score:.2f}")
    else:
        results['tests_failed'] += 1
        results['details'].append(f"FAIL: Profit score invalid: {score}")

    # Test 5: Position sizing
    pos_size = calculate_position_size_fixing9(0.025, 0.72, 0.02)
    if 0.02 <= pos_size <= 0.10:
        results['tests_passed'] += 1
        results['details'].append(f"PASS: Position size in range: {pos_size:.1%}")
    else:
        results['tests_failed'] += 1
        results['details'].append(f"FAIL: Position size out of range: {pos_size:.1%}")

    return results


# =============================================================================
# FEATURE CONSISTENCY CHECK
# =============================================================================

def enforce_feature_consistency(X_train: pd.DataFrame,
                                 X_predict: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Ensure training and prediction use exactly the same features.

    This is CRITICAL - the main cause of R² = -0.0001 was feature mismatch.

    Args:
        X_train: Training features
        X_predict: Prediction features

    Returns:
        (X_train_aligned, X_predict_aligned, common_features)
    """
    train_cols = set(X_train.columns)
    predict_cols = set(X_predict.columns)

    common_features = list(train_cols & predict_cols)
    missing_in_train = predict_cols - train_cols
    missing_in_predict = train_cols - predict_cols

    if missing_in_train or missing_in_predict:
        logger.warning(f"[FIXING9] Feature mismatch detected!")
        if missing_in_train:
            logger.warning(f"  Missing in train: {list(missing_in_train)[:5]}...")
        if missing_in_predict:
            logger.warning(f"  Missing in predict: {list(missing_in_predict)[:5]}...")

    logger.info(f"[FIXING9] Using {len(common_features)} common features")

    return X_train[common_features], X_predict[common_features], common_features


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def apply_fixing9_to_prediction(ticker: str,
                                 predicted_return: float,
                                 volatility: float,
                                 volume_ratio: float = 1.0,
                                 rsi: float = 50.0,
                                 asset_class: str = 'stock',
                                 regime: str = 'neutral') -> Dict[str, Any]:
    """
    Apply all fixing9 improvements to a single prediction.

    Args:
        ticker: Stock ticker
        predicted_return: Model's raw prediction
        volatility: Historical volatility
        volume_ratio: Current volume ratio
        rsi: RSI indicator
        asset_class: Type of asset
        regime: Market regime

    Returns:
        Enhanced prediction dict with fixing9 improvements
    """
    # P0: Calculate confidence with new thresholds
    confidence, snr = calculate_confidence_fixing9(predicted_return, volatility, asset_class)

    # P1: Calculate profit score
    profit_score = calculate_profit_score_fixing9(
        predicted_return, confidence, volatility, volume_ratio, rsi, regime_multiplier=1.0
    )

    # P1: Calculate position size
    position_size = calculate_position_size_fixing9(predicted_return, confidence, volatility)

    # P1: Calculate profit targets
    targets = calculate_profit_targets_fixing9(predicted_return, volatility, regime)

    # Determine signal
    if predicted_return > 0 and confidence >= 0.60:
        signal = 'BUY'
        direction = 1
    elif predicted_return < 0 and confidence >= 0.60:
        signal = 'SELL'
        direction = -1
    else:
        signal = 'HOLD'
        direction = 0

    return {
        'ticker': ticker,
        'predicted_return': predicted_return,
        'confidence': confidence,
        'snr': snr,
        'signal': signal,
        'direction': direction,
        'profit_score': profit_score,
        'position_size': position_size,
        'take_profit': targets['take_profit'],
        'stop_loss': targets['stop_loss'],
        'time_exit_days': targets['time_exit_days'],
        'fixing9_applied': True,
        'asset_class': asset_class,
        'regime': regime,
    }


# Make key functions available at module level
__all__ = [
    'SAFE_FEATURE_DEFAULTS',
    'US_CORE_FEATURES',
    'SNR_THRESHOLDS_FIXING9',
    'safe_fill_missing_features',
    'select_core_features',
    'calculate_confidence_fixing9',
    'calculate_profit_score_fixing9',
    'calculate_position_size_fixing9',
    'calculate_profit_targets_fixing9',
    'get_sector_rotation_weights',
    'check_correlation_limits',
    'validate_fixing9_implementation',
    'enforce_feature_consistency',
    'apply_fixing9_to_prediction',
]
