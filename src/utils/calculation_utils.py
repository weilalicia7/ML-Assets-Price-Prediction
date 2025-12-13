"""
Phase 1-6 Calculation Utilities
Complete implementation with error handling, input validation, and numerical stability.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union

# ============================================================================
# CONSTANTS
# ============================================================================

TRADING_DAYS = 252
ANNUALIZE_VOL = np.sqrt(252)
ANNUALIZE_RET = 252

PHASE_PARAMETERS = {
    'phase1': {
        'turnover_threshold': 0.05,
        'correlation_penalty': 0.85,
        'benchmark_vol': 0.15
    },
    'phase2': {
        'tanh_scale': 10,
        'signal_periods': {'momentum': 20, 'volatility': 20, 'mean_reversion': 20}
    },
    'phase3': {
        'n_regimes': 4,
        'hurst_lookback': 100,
        'mfi_period': 14,
        'vwap_period': 20
    },
    'phase4': {
        'vix_threshold': 25,
        'asset_sensitivities': {
            'equity': 1.0, 'crypto': 1.3, 'commodity': 1.1,
            'forex': 0.8, 'bond': 0.7, 'real_estate': 0.9
        }
    },
    'phase5': {
        'decay_factor': 0.95,
        'metric_weights': [0.4, 0.3, 0.2, 0.1],
        'bayesian_prior': [1, 1]
    },
    'phase6': {
        'risk_aversion_range': [1.0, 2.5],
        'max_position': 0.25,
        'turnover_limit': 0.1,
        'regime_multipliers': {
            'crisis': 0.5, 'high_vol': 0.75, 'normal': 1.0, 'low_vol': 1.15
        }
    }
}


# ============================================================================
# PHASE 1: CORE FEATURES & TRADING
# ============================================================================

def calculate_volatility_scaling(returns: Union[pd.Series, np.ndarray],
                                  lookback: int = 20,
                                  benchmark_vol: float = 0.15) -> float:
    """
    Calculate volatility scaling adjustment for position sizing.

    Args:
        returns: Return series
        lookback: Lookback period for recent volatility
        benchmark_vol: Benchmark volatility level

    Returns:
        Scaling adjustment factor [0.5, 1.5]
    """
    if len(returns) < lookback:
        return 1.0

    if isinstance(returns, pd.Series):
        recent_vol = returns.tail(lookback).std() * ANNUALIZE_VOL
    else:
        recent_vol = np.std(returns[-lookback:]) * ANNUALIZE_VOL

    recent_vol = max(recent_vol, 0.01)  # Avoid division by zero

    vol_ratio = recent_vol / benchmark_vol
    adjustment = 1.0 / (1.0 + max(vol_ratio - 1.0, 0) * 0.5)

    return float(np.clip(adjustment, 0.5, 1.5))


def correlation_aware_sizing(base_size: float,
                              sector_value: float,
                              total_portfolio_value: float,
                              num_same_sector_positions: int) -> float:
    """
    Adjust position size based on sector correlation.

    Args:
        base_size: Base position size [0, 1]
        sector_value: Value of positions in same sector
        total_portfolio_value: Total portfolio value
        num_same_sector_positions: Number of positions in same sector

    Returns:
        Adjusted position size
    """
    base_size = max(0, min(1, base_size))
    sector_value = max(0, sector_value)
    total_portfolio_value = max(0.01, total_portfolio_value)
    num_same_sector_positions = max(0, int(num_same_sector_positions))

    # Correlation penalty (15% per same-sector position)
    correlation_penalty = 0.85 ** num_same_sector_positions
    correlation_penalty = max(correlation_penalty, 0.70)

    adjusted_size = base_size * correlation_penalty

    return float(np.clip(adjusted_size, 0, 1))


def risk_parity_allocation(volatilities: np.ndarray,
                           correlations: np.ndarray,
                           budget: float = 1.0) -> np.ndarray:
    """
    Risk parity allocation across assets.

    Args:
        volatilities: Asset volatilities
        correlations: Correlation to portfolio
        budget: Total allocation budget

    Returns:
        Allocation weights
    """
    n_assets = len(volatilities)

    if n_assets == 0:
        return np.array([])

    volatilities = np.array(volatilities)
    correlations = np.array(correlations)

    volatilities = np.maximum(volatilities, 0.01)

    risk_contrib = volatilities * (0.5 + correlations * 0.5)
    total_risk = np.sum(risk_contrib)

    if total_risk <= 0:
        return np.ones(n_assets) / n_assets

    allocation = budget * (risk_contrib / total_risk)
    allocation = allocation / np.sum(allocation)

    return allocation


def calculate_turnover(current_weights: np.ndarray,
                       target_weights: np.ndarray) -> float:
    """
    Calculate portfolio turnover.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights

    Returns:
        One-way turnover
    """
    current_weights = np.array(current_weights)
    target_weights = np.array(target_weights)

    if len(current_weights) != len(target_weights):
        raise ValueError("Weight arrays must have same length")

    turnover = np.sum(np.abs(current_weights - target_weights)) / 2
    return max(0, turnover)


# ============================================================================
# PHASE 2: ASSET CLASS ENSEMBLES
# ============================================================================

def safe_data_extraction(data, column: str = 'Close', default_value: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Safely extract data from various input types (DataFrame, dict, array, Series).

    Args:
        data: Input data (DataFrame, dict, Series, or array-like)
        column: Column name to extract if data is DataFrame/dict
        default_value: Value to return if extraction fails (default: None)

    Returns:
        Numpy array of values or default_value if extraction fails
    """
    try:
        if data is None:
            return default_value

        # Handle pandas Series - extract values directly
        if isinstance(data, pd.Series):
            return data.values.astype(float)

        # Handle dict with column key
        if isinstance(data, dict) and column in data:
            return np.array(data[column], dtype=float)

        # Handle DataFrame with columns attribute
        if hasattr(data, 'columns') and column in data.columns:
            return data[column].values.astype(float)

        # Handle 2D numpy array - extract column by index or flatten if single column
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # If 2D array with single column, flatten it
                if data.shape[1] == 1:
                    return data.flatten().astype(float)
                # Otherwise, try to extract first column (assume it's Close/price data)
                return data[:, 0].astype(float)
            else:
                # 1D array
                return data.astype(float)

        # Handle array-like (list, tuple)
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=float)
            # Handle nested list (2D)
            if arr.ndim == 2:
                if arr.shape[1] == 1:
                    return arr.flatten()
                return arr[:, 0]
            return arr

        # Fallback: try direct conversion
        return np.array(data, dtype=float)

    except Exception as e:
        print(f"Data extraction error: {e}")
        return default_value


def momentum_signal(data: Union[pd.DataFrame, np.ndarray, dict], period: int = 20) -> float:
    """
    Calculate momentum signal using price changes.

    Args:
        data: Price data (DataFrame with 'Close', dict with 'Close', or array)
        period: Lookback period

    Returns:
        Momentum signal [-1, 1]
    """
    # Safe data extraction handling multiple input types
    close_prices = safe_data_extraction(data, 'Close')

    if close_prices is None or len(close_prices) < period + 1:
        return 0.0

    # Remove NaN values
    close_prices = close_prices[~np.isnan(close_prices)]

    if len(close_prices) < period + 1:
        return 0.0

    if close_prices[-period-1] == 0:
        return 0.0

    returns_20d = (close_prices[-1] - close_prices[-period-1]) / close_prices[-period-1]

    if np.isnan(returns_20d) or np.isinf(returns_20d):
        return 0.0

    momentum = np.tanh(returns_20d * 10)
    return float(momentum)


def volatility_signal(data: Union[pd.DataFrame, np.ndarray], period: int = 20) -> float:
    """
    Calculate volatility-based signal (lower vol = higher signal).

    Args:
        data: Price data
        period: Lookback period

    Returns:
        Volatility signal [0, 1]
    """
    if len(data) < period + 1:
        return 0.0

    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        close_prices = data['Close'].values
    else:
        close_prices = np.array(data)

    returns = np.diff(close_prices) / np.where(close_prices[:-1] == 0, 1, close_prices[:-1])

    if len(returns) < period:
        return 0.0

    recent_returns = returns[-period:]
    vol = np.std(recent_returns)

    if vol < 1e-8:
        return 1.0

    vol_signal = 1 - min(vol * ANNUALIZE_VOL * 10, 1)
    return float(vol_signal)


def mean_reversion_signal(data: Union[pd.DataFrame, np.ndarray], period: int = 20) -> float:
    """
    Calculate mean reversion signal.

    Args:
        data: Price data
        period: Lookback period

    Returns:
        Mean reversion signal [-1, 1]
    """
    if len(data) < period:
        return 0.0

    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        close_prices = data['Close'].values
    else:
        close_prices = np.array(data)

    current_price = close_prices[-1]
    ma_20 = np.mean(close_prices[-period:])

    if ma_20 < 1e-8:
        return 0.0

    deviation = (current_price - ma_20) / ma_20
    mean_rev = -np.tanh(deviation * 5)
    return float(mean_rev)


def combine_signals(signals_dict: Dict[str, float]) -> Tuple[float, float]:
    """
    Combine multiple signals with confidence weighting.

    Args:
        signals_dict: Dictionary of signal name -> value

    Returns:
        Tuple of (combined_signal, confidence)
    """
    if not signals_dict:
        return 0.0, 0.0

    signals = list(signals_dict.values())
    weights = {k: 1.0 / len(signals_dict) for k in signals_dict}

    weighted_sum = sum(signals_dict[k] * weights[k] for k in signals_dict)
    combined = np.clip(weighted_sum, -1, 1)

    # Confidence from agreement
    agreement = sum(1 for s in signals if np.sign(s) == np.sign(combined)) / len(signals)
    confidence = np.clip(0.3 + 0.5 * agreement, 0, 1)

    return float(combined), float(confidence)


# ============================================================================
# PHASE 3: REGIME DETECTION & ORDER FLOW
# ============================================================================

def detect_volatility_regime(volatility_data: np.ndarray, n_components: int = 4) -> np.ndarray:
    """
    Detect volatility regimes using Gaussian Mixture Models.

    Args:
        volatility_data: Volatility time series
        n_components: Number of regime states

    Returns:
        Regime labels
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return np.zeros(len(volatility_data))

    if len(volatility_data) < n_components * 10:
        return np.zeros(len(volatility_data))

    volatility_array = np.array(volatility_data).reshape(-1, 1)

    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(volatility_array)
    regimes = model.predict(volatility_array)

    return regimes


def calculate_hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Calculate Hurst exponent using rescaled range analysis.

    Args:
        ts: Time series
        max_lag: Maximum lag for analysis

    Returns:
        Trend score [-1, 1] (H < 0.5: mean-reverting, H > 0.5: trending)
    """
    if len(ts) < max_lag:
        max_lag = len(ts) // 2

    if max_lag < 10:
        return 0.5

    lags = list(range(2, max_lag))
    tau = []

    for lag in lags:
        if lag < len(ts):
            diff_std = np.std(ts[lag:] - ts[:-lag])
            if diff_std > 0:
                tau.append(diff_std)

    if len(tau) < 5:
        return 0.5

    lags_array = np.array(lags[:len(tau)])
    tau_array = np.array(tau)

    # Avoid log of zero
    valid_mask = (lags_array > 0) & (tau_array > 0)
    if np.sum(valid_mask) < 5:
        return 0.5

    hurst = np.polyfit(np.log(lags_array[valid_mask]), np.log(tau_array[valid_mask]), 1)[0]
    trend_score = np.clip((hurst - 0.5) * 2, -1, 1)

    return float(trend_score)


def calculate_obv(close_prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Calculate On-Balance Volume.

    Args:
        close_prices: Close price series
        volumes: Volume series

    Returns:
        OBV series
    """
    if len(close_prices) != len(volumes):
        raise ValueError("Close prices and volumes must have same length")

    close_changes = np.diff(close_prices)
    volumes_aligned = volumes[1:]

    obv = np.where(close_changes > 0, volumes_aligned,
                   np.where(close_changes < 0, -volumes_aligned, 0))
    obv_cumulative = np.cumsum(obv)

    return obv_cumulative


def accumulation_distribution(high_prices: np.ndarray,
                               low_prices: np.ndarray,
                               close_prices: np.ndarray,
                               volumes: np.ndarray) -> np.ndarray:
    """
    Calculate Accumulation/Distribution Line.

    Args:
        high_prices: High price series
        low_prices: Low price series
        close_prices: Close price series
        volumes: Volume series

    Returns:
        A/D line series
    """
    if not all(len(arr) == len(high_prices) for arr in [low_prices, close_prices, volumes]):
        raise ValueError("All price arrays must have same length")

    high_low_range = high_prices - low_prices
    high_low_range = np.where(high_low_range < 1e-8, 1e-8, high_low_range)

    clv = ((close_prices - low_prices) - (high_prices - close_prices)) / high_low_range
    clv = np.clip(clv, -1, 1)

    mfv = clv * volumes
    ad_line = np.cumsum(mfv)

    return ad_line


def money_flow_index(high_prices: np.ndarray,
                     low_prices: np.ndarray,
                     close_prices: np.ndarray,
                     volumes: np.ndarray,
                     period: int = 14) -> np.ndarray:
    """
    Calculate Money Flow Index.

    Args:
        high_prices: High price series
        low_prices: Low price series
        close_prices: Close price series
        volumes: Volume series
        period: MFI period

    Returns:
        MFI series [0, 100]
    """
    if len(high_prices) < period:
        return np.full(len(high_prices), 50.0)

    typical_price = (high_prices + low_prices + close_prices) / 3
    raw_mf = typical_price * volumes

    price_changes = np.diff(typical_price)
    raw_mf_changes = raw_mf[1:]

    positive_flow = np.where(price_changes > 0, raw_mf_changes, 0)
    negative_flow = np.where(price_changes < 0, raw_mf_changes, 0)

    positive_mf = pd.Series(positive_flow).rolling(period).sum().values
    negative_mf = pd.Series(negative_flow).rolling(period).sum().values

    money_ratio = np.divide(positive_mf, negative_mf,
                            out=np.ones_like(positive_mf),
                            where=negative_mf != 0)

    mfi = 100 - (100 / (1 + money_ratio))
    mfi = np.where(np.isnan(mfi), 50.0, mfi)

    return mfi


def calculate_vwap(high_prices: np.ndarray,
                   low_prices: np.ndarray,
                   close_prices: np.ndarray,
                   volumes: np.ndarray,
                   period: int = 20) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price.

    Args:
        high_prices: High price series
        low_prices: Low price series
        close_prices: Close price series
        volumes: Volume series
        period: VWAP period

    Returns:
        VWAP series
    """
    if len(high_prices) < period:
        return np.full(len(high_prices), np.nan)

    typical_price = (high_prices + low_prices + close_prices) / 3

    vwap_numerator = pd.Series(typical_price * volumes).rolling(period).sum()
    vwap_denominator = pd.Series(volumes).rolling(period).sum()

    vwap = vwap_numerator / vwap_denominator
    return vwap.values


def force_index(close_prices: np.ndarray, volumes: np.ndarray, span: int = 13) -> np.ndarray:
    """
    Calculate Force Index.

    Args:
        close_prices: Close price series
        volumes: Volume series
        span: EMA span

    Returns:
        Force Index series
    """
    if len(close_prices) != len(volumes):
        raise ValueError("Close prices and volumes must have same length")

    close_diff = np.diff(close_prices)
    volumes_trimmed = volumes[1:]

    force = close_diff * volumes_trimmed
    force_ema = pd.Series(force).ewm(span=span).mean()

    return force_ema.values


def detect_smart_money(close_prices: np.ndarray,
                       volumes: np.ndarray,
                       obv: np.ndarray,
                       ad_line: np.ndarray,
                       volatility_threshold: float = 0.02,
                       volume_threshold: float = 1.5) -> np.ndarray:
    """
    Detect smart money accumulation/distribution.

    Args:
        close_prices: Close price series
        volumes: Volume series
        obv: OBV series
        ad_line: A/D line series
        volatility_threshold: Price stability threshold
        volume_threshold: High volume threshold multiplier

    Returns:
        Smart money pressure [-1, 1]
    """
    if len(close_prices) < 20:
        return np.zeros(len(close_prices))

    price_volatility = pd.Series(close_prices).pct_change().rolling(20).std().values
    volume_ma = pd.Series(volumes).rolling(20).mean().values

    price_stable = price_volatility < volatility_threshold
    volume_high = volumes > volume_ma * volume_threshold
    obv_rising = np.concatenate([[False], np.diff(obv) > 0])
    ad_rising = np.concatenate([[False], np.diff(ad_line) > 0])

    accumulation = (price_stable & volume_high & obv_rising & ad_rising)
    distribution = (price_stable & volume_high & ~obv_rising & ~ad_rising)

    smart_money_pressure = accumulation.astype(int) - distribution.astype(int)
    smart_money_pressure = np.clip(smart_money_pressure, -1, 1)

    return smart_money_pressure


# ============================================================================
# PHASE 4: MACRO INTEGRATION
# ============================================================================

def asset_specific_multiplier(base_mult: float,
                               asset_class: str,
                               vix: Optional[float] = None,
                               vix_sensitivity: float = 1.0) -> float:
    """
    Apply asset-specific multipliers and VIX adjustments.

    Args:
        base_mult: Base multiplier
        asset_class: Asset class name
        vix: Current VIX level
        vix_sensitivity: VIX sensitivity factor

    Returns:
        Adjusted multiplier [0, 1.5]
    """
    base_mult = float(base_mult)
    asset_class = str(asset_class).lower()

    sensitivity_map = {
        'equity': 1.0,
        'crypto': 1.3,
        'commodity': 1.1,
        'forex': 0.8,
        'bond': 0.7,
        'real_estate': 0.9
    }

    if asset_class not in sensitivity_map:
        sensitivity = 1.0  # Default sensitivity
    else:
        sensitivity = sensitivity_map[asset_class]

    if base_mult < 1.0:
        adjusted = 1.0 - (1.0 - base_mult) * sensitivity
    else:
        adjusted = 1.0 + (base_mult - 1.0) * sensitivity

    if vix is not None:
        vix = float(vix)
        if vix > 25:
            vix_effect = (vix - 25) / 100 * vix_sensitivity
            adjusted *= (1 - vix_effect)

    return float(np.clip(adjusted, 0.0, 1.5))


def regime_persistence(regime_history: np.ndarray, n_regimes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate regime persistence and expected duration.

    Args:
        regime_history: Historical regime labels
        n_regimes: Number of regimes

    Returns:
        Tuple of (transition_matrix, expected_duration)
    """
    if len(regime_history) < 2:
        return np.ones((n_regimes, n_regimes)) / n_regimes, np.ones(n_regimes)

    transition_matrix = np.zeros((n_regimes, n_regimes))

    for i in range(len(regime_history) - 1):
        current_regime = int(regime_history[i])
        next_regime = int(regime_history[i + 1])
        if current_regime < n_regimes and next_regime < n_regimes:
            transition_matrix[current_regime, next_regime] += 1

    row_sums = transition_matrix.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    P = transition_matrix / row_sums[:, np.newaxis]

    persistence = np.diag(P)
    persistence_safe = np.clip(persistence, 0.01, 0.99)
    expected_duration = -1 / np.log(persistence_safe)

    return P, expected_duration


# ============================================================================
# PHASE 5: DYNAMIC WEIGHTING & BAYESIAN COMBINATION
# ============================================================================

def decay_weighted_sharpe(returns: np.ndarray, decay_factor: float = 0.95) -> float:
    """
    Calculate decay-weighted Sharpe ratio.

    Args:
        returns: Return series
        decay_factor: Decay factor for weighting

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    returns = np.array(returns)
    n = len(returns)

    decay_weights = np.array([decay_factor ** (n-1-i) for i in range(n)])
    decay_weights /= decay_weights.sum()

    weighted_mean = np.sum(returns * decay_weights)
    weighted_variance = np.sum(decay_weights * (returns - weighted_mean)**2)
    weighted_std = np.sqrt(weighted_variance)

    if weighted_std < 1e-8:
        return 0.0

    sharpe = (weighted_mean / weighted_std) * ANNUALIZE_VOL
    return float(sharpe)


def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """
    Calculate Calmar ratio (return to max drawdown).

    Args:
        returns: Return series
        periods: Annualization factor

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = np.array(returns)

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative / running_max) - 1
    max_drawdown = np.min(drawdowns)

    if abs(max_drawdown) < 1e-8:
        return 0.0

    total_return = cumulative[-1] - 1
    annualized_return = (1 + total_return) ** (periods / len(returns)) - 1

    calmar = annualized_return / abs(max_drawdown)
    return float(calmar)


def composite_performance_score(returns: np.ndarray,
                                 decay_factor: float = 0.95) -> Tuple[float, float]:
    """
    Calculate composite performance score.

    Args:
        returns: Return series
        decay_factor: Decay factor for weighting

    Returns:
        Tuple of (score, confidence)
    """
    if len(returns) < 20:
        return 0.0, 0.0

    returns = np.array(returns)

    sharpe = decay_weighted_sharpe(returns, decay_factor)

    wins = np.sum(returns > 0)
    win_rate = wins / len(returns) if len(returns) > 0 else 0.5

    calmar = calmar_ratio(returns)

    return_std = np.std(returns)
    consistency = 1 / (1 + return_std * ANNUALIZE_VOL) if return_std > 0 else 1.0

    score = (
        sharpe * 0.40 +
        win_rate * 0.30 +
        calmar * 0.20 +
        consistency * 0.10
    )

    confidence = min(1.0, len(returns) / 100)
    final_score = score * confidence

    return float(final_score), float(confidence)


class BayesianSignalUpdater:
    """Bayesian updater for signal reliability using Beta-Bernoulli model with improved decay."""

    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1, decay: float = 0.99):
        self.alpha = float(prior_alpha)
        self.beta = float(prior_beta)
        self.decay = float(decay)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def update(self, signal: float, actual_return: float) -> float:
        """
        Update reliability estimates using improved Beta-Bernoulli model.

        The improved decay formula blends towards prior rather than subtractive decay:
        new_value = prior * (1 - decay) + current * decay

        This prevents overly aggressive decay that could destabilize estimates.
        """
        try:
            signal = float(signal)
            actual_return = float(actual_return)

            # Improved decay: blend towards prior rather than subtractive decay
            # Old formula: prior + (current - prior) * decay (too aggressive)
            # New formula: prior * (1 - decay) + current * decay (smoother blend)
            self.alpha = self.prior_alpha * (1 - self.decay) + self.alpha * self.decay
            self.beta = self.prior_beta * (1 - self.decay) + self.beta * self.decay

            # Check if signal direction was correct
            signal_correct = (np.sign(signal) == np.sign(actual_return))

            # Update parameters based on correctness
            if signal_correct:
                self.alpha += 1
            else:
                self.beta += 1

            # Ensure parameters stay positive (minimum bounds)
            self.alpha = max(0.1, self.alpha)
            self.beta = max(0.1, self.beta)

            # Calculate reliability
            total = self.alpha + self.beta
            reliability = self.alpha / total if total > 0 else 0.5

            return float(reliability)

        except Exception as e:
            print(f"Bayesian update error: {e}")
            return 0.5

    def get_reliability(self) -> float:
        """Get current reliability estimate."""
        try:
            total = self.alpha + self.beta
            return float(self.alpha / total) if total > 0 else 0.5
        except:
            return 0.5

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Bayesian credible interval for reliability estimate.

        Uses the Beta distribution's quantile function to compute exact
        credible intervals based on current alpha and beta parameters.

        Args:
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) for the reliability estimate
        """
        try:
            alpha_level = (1 - confidence) / 2
            lower = stats.beta.ppf(alpha_level, self.alpha, self.beta)
            upper = stats.beta.ppf(1 - alpha_level, self.alpha, self.beta)
            return (float(lower), float(upper))
        except Exception as e:
            print(f"Confidence interval calculation error: {e}")
            # Fallback to wide interval
            return (0.0, 1.0)

    def reset(self):
        """Reset to prior values."""
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta


def bayesian_signal_combination(signals_dict: Dict[str, float],
                                 reliability_dict: Dict[str, float],
                                 correlation_matrix: Optional[np.ndarray] = None) -> float:
    """
    Combine signals using Bayesian weighting.

    Args:
        signals_dict: Dictionary of signal name -> value
        reliability_dict: Dictionary of signal name -> reliability
        correlation_matrix: Optional correlation matrix between signals

    Returns:
        Combined signal [-1, 1]
    """
    if not signals_dict:
        return 0.0

    signal_names = list(signals_dict.keys())
    signals = np.array([signals_dict[name] for name in signal_names])
    reliabilities = np.array([reliability_dict.get(name, 0.5) for name in signal_names])

    weights = reliabilities.copy()

    if correlation_matrix is not None and len(signals) == correlation_matrix.shape[0]:
        for i, signal in enumerate(signals):
            correlation_boost = 0
            for j, other_signal in enumerate(signals):
                if i != j and correlation_matrix[i, j] > 0.1:
                    agreement = 1 if np.sign(signal) == np.sign(other_signal) else -1
                    correlation_boost += correlation_matrix[i, j] * agreement

            weights[i] *= (1 + abs(signal) * correlation_boost)

    if np.sum(weights) < 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / np.sum(weights)

    combined_signal = np.sum(signals * weights)
    combined_signal = np.clip(combined_signal, -1, 1)

    return float(combined_signal)


# ============================================================================
# PHASE 6: PORTFOLIO OPTIMIZATION
# ============================================================================

def risk_contribution_analysis(weights: np.ndarray,
                                covariance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze risk contribution of portfolio components.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix

    Returns:
        Tuple of (component_contrib, risk_percent)
    """
    weights = np.array(weights)
    cov = np.array(covariance_matrix)

    if weights.shape[0] != cov.shape[0]:
        raise ValueError("Weights and covariance matrix dimensions don't match")

    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    portfolio_vol = np.sqrt(weights @ cov @ weights)

    if portfolio_vol < 1e-8:
        return np.zeros_like(weights), np.zeros_like(weights)

    marginal_contrib = cov @ weights / portfolio_vol
    component_contrib = weights * marginal_contrib
    risk_percent = component_contrib / portfolio_vol

    return component_contrib, risk_percent


def _calculate_parametric_es(volatility: float, confidence: float = 0.95) -> float:
    """
    Calculate parametric Expected Shortfall assuming normal distribution.

    For a standard normal distribution:
    - ES = -sigma * phi(z_alpha) / (1 - alpha)

    Where:
    - phi is the PDF of standard normal
    - z_alpha is the VaR quantile
    - alpha is the confidence level

    At 95% confidence: ES ≈ -2.063 * sigma
    At 99% confidence: ES ≈ -2.665 * sigma

    Args:
        volatility: Portfolio volatility (standard deviation)
        confidence: Confidence level (default 0.95)

    Returns:
        Parametric Expected Shortfall (negative value indicating loss)
    """
    try:
        alpha = 1 - confidence  # Tail probability
        z_alpha = stats.norm.ppf(alpha)  # VaR quantile
        pdf_at_var = stats.norm.pdf(z_alpha)

        # ES formula for normal distribution
        es_multiplier = -pdf_at_var / alpha
        return float(es_multiplier * volatility)
    except Exception as e:
        print(f"Parametric ES calculation error: {e}")
        # Fallback to approximate values
        if confidence >= 0.99:
            return -2.665 * volatility
        return -2.063 * volatility


def expected_shortfall(returns: Optional[np.ndarray],
                       weights: np.ndarray,
                       confidence: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (CVaR).

    Args:
        returns: Return matrix (n_periods x n_assets) or portfolio returns
        weights: Portfolio weights
        confidence: Confidence level

    Returns:
        Expected shortfall value
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = np.array(returns)
    weights = np.array(weights)

    if returns.ndim == 2:
        portfolio_returns = returns @ weights
    else:
        portfolio_returns = returns

    var_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

    if len(tail_returns) == 0:
        return float(var_threshold)

    es = np.mean(tail_returns)
    return float(es)


def regime_aware_risk_budget(base_risk: float,
                              current_regime: str,
                              vix: Optional[float] = None,
                              regime_mults: Optional[Dict[str, float]] = None,
                              vix_base: float = 15) -> float:
    """
    Adjust risk budget based on market regime and VIX.

    Args:
        base_risk: Base risk budget
        current_regime: Current market regime
        vix: Current VIX level
        regime_mults: Regime multipliers
        vix_base: Base VIX level

    Returns:
        Adjusted risk budget
    """
    if regime_mults is None:
        regime_mults = {
            'crisis': 0.5,
            'high_vol': 0.75,
            'normal': 1.0,
            'low_vol': 1.15
        }

    regime_mult = regime_mults.get(current_regime, 1.0)

    vix_adj = 1.0
    if vix is not None:
        vix_adj = np.clip(1.0 - (vix - vix_base) / 100, 0.5, 1.5)

    adjusted_budget = base_risk * regime_mult * vix_adj
    return float(adjusted_budget)


def portfolio_optimization(expected_returns: np.ndarray,
                            covariance_matrix: np.ndarray,
                            current_weights: Optional[np.ndarray] = None,
                            risk_aversion: float = 1.5,
                            max_position: float = 0.25,
                            turnover_limit: float = 0.1) -> np.ndarray:
    """
    Portfolio optimization with turnover constraints.

    Args:
        expected_returns: Expected asset returns
        covariance_matrix: Asset covariance matrix
        current_weights: Current portfolio weights
        risk_aversion: Risk aversion parameter
        max_position: Maximum single position size
        turnover_limit: Maximum turnover allowed

    Returns:
        Optimal portfolio weights
    """
    n_assets = len(expected_returns)

    expected_returns = np.array(expected_returns)
    covariance_matrix = np.array(covariance_matrix)

    if current_weights is None:
        current_weights = np.ones(n_assets) / n_assets
    else:
        current_weights = np.array(current_weights)

    def objective(weights):
        ret = weights @ expected_returns
        var = weights @ covariance_matrix @ weights
        turnover = np.sum(np.abs(weights - current_weights)) / 2
        turnover_penalty = 0.01 * max(0, turnover - turnover_limit)**2
        return -(ret - 0.5 * risk_aversion * var - turnover_penalty)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, max_position)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    try:
        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            return x0
    except:
        return x0


def multi_timeframe_blending(timeframe_results: Dict[str, Dict[str, float]],
                              blend_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Blend portfolio weights from multiple timeframes.

    Args:
        timeframe_results: Dictionary of timeframe -> {asset: weight}
        blend_weights: Timeframe blending weights

    Returns:
        Blended portfolio weights
    """
    if blend_weights is None:
        blend_weights = {
            'intraday': 0.1,
            'daily': 0.3,
            'weekly': 0.4,
            'monthly': 0.2
        }

    all_assets = set()
    for tf_weights in timeframe_results.values():
        all_assets.update(tf_weights.keys())

    blended = {asset: 0.0 for asset in all_assets}

    for tf, tf_weights in timeframe_results.items():
        if tf in blend_weights:
            weight = blend_weights[tf]
            for asset, w in tf_weights.items():
                blended[asset] += weight * w

    total = sum(blended.values())
    if total > 1e-8:
        blended = {asset: w / total for asset, w in blended.items()}

    return blended


def adjust_covariance_matrix(covariance_matrix: np.ndarray,
                              method: str = 'ewm',
                              returns: Optional[np.ndarray] = None,
                              lambda_: float = 0.94,
                              regime: Optional[str] = None,
                              historical_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Adjust covariance matrix using various methods.

    Args:
        covariance_matrix: Base covariance matrix
        method: Adjustment method ('ewm', 'regime_blend')
        returns: Historical returns for EWM
        lambda_: Decay factor for EWM
        regime: Current regime for regime blending
        historical_cov: Historical covariance for regime blending

    Returns:
        Adjusted covariance matrix
    """
    cov = np.array(covariance_matrix)

    if method == 'ewm' and returns is not None:
        returns = np.array(returns)
        n = len(returns)
        weights = np.array([lambda_ ** (n-1-i) for i in range(n)])
        weights = weights / weights.sum()

        weighted_returns = returns * weights[:, np.newaxis]
        cov_ewm = weighted_returns.T @ weighted_returns
        return cov_ewm

    elif method == 'regime_blend' and regime == 'crisis' and historical_cov is not None:
        return 0.8 * cov + 0.2 * historical_cov

    else:
        return cov


def stress_test_portfolio(weights: np.ndarray,
                           covariance_matrix: np.ndarray,
                           scenarios: Optional[Dict[str, Dict[str, float]]] = None,
                           historical_returns: Optional[np.ndarray] = None,
                           n_simulations: int = 1000) -> Dict[str, Dict]:
    """
    Comprehensive stress test portfolio with simulated returns for realistic ES calculation.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        scenarios: Stress scenarios {name: {vol_mult, corr_boost}}
        historical_returns: Historical returns matrix (n_periods x n_assets) for ES calc
        n_simulations: Number of Monte Carlo simulations for ES

    Returns:
        Stress test results including Expected Shortfall
    """
    if scenarios is None:
        scenarios = {
            '2008_lehman': {'vol_mult': 3.0, 'corr_boost': 0.4},
            '2020_covid': {'vol_mult': 2.5, 'corr_boost': 0.3},
            'inflation_shock': {'vol_mult': 2.0, 'corr_boost': 0.2},
            'normal': {'vol_mult': 1.0, 'corr_boost': 0.0}
        }

    weights = np.array(weights, dtype=float)
    base_cov = np.array(covariance_matrix, dtype=float)

    # Extract base volatilities and correlation matrix
    vol = np.sqrt(np.diag(base_cov))
    vol = np.where(vol < 1e-8, 1e-8, vol)  # Avoid division by zero
    corr = base_cov / np.outer(vol, vol)
    corr = np.where(np.isnan(corr), 0, corr)

    results = {}

    for scenario_name, params in scenarios.items():
        # Adjust volatilities
        stressed_vol = vol * params['vol_mult']

        # Adjust correlations (increase towards 1 during stress)
        stressed_corr = corr + params['corr_boost'] * (1 - corr)

        # Ensure valid correlation matrix (symmetric with unit diagonal)
        stressed_corr = (stressed_corr + stressed_corr.T) / 2
        np.fill_diagonal(stressed_corr, 1.0)

        # Reconstruct covariance matrix
        stressed_cov = np.outer(stressed_vol, stressed_vol) * stressed_corr

        # Calculate stressed portfolio volatility
        stressed_portfolio_vol = np.sqrt(weights @ stressed_cov @ weights)

        # Calculate Expected Shortfall using simulated returns
        stressed_es = 0.0

        if historical_returns is not None and len(historical_returns) > 0:
            # Use historical returns transformed to stressed regime
            try:
                historical_array = np.array(historical_returns, dtype=float)
                if historical_array.ndim == 2 and historical_array.shape[1] == len(weights):
                    # Scale historical returns by stress multiplier
                    scaled_returns = historical_array * params['vol_mult']
                    portfolio_returns = scaled_returns @ weights
                    stressed_es = expected_shortfall(portfolio_returns, weights, confidence=0.95)
            except Exception as e:
                print(f"Historical ES calculation error: {e}")

        # If no historical data or ES calc failed, use Monte Carlo simulation
        if stressed_es == 0.0:
            try:
                # Monte Carlo simulation using stressed covariance
                # Ensure covariance is positive semi-definite
                eigvals = np.linalg.eigvalsh(stressed_cov)
                if np.min(eigvals) < 0:
                    # Add small value to diagonal for numerical stability
                    stressed_cov += np.eye(len(weights)) * abs(np.min(eigvals)) * 1.01

                # Cholesky decomposition for correlated random returns
                L = np.linalg.cholesky(stressed_cov)

                # Generate random standard normal returns
                np.random.seed(42)  # For reproducibility
                random_returns = np.random.standard_normal((n_simulations, len(weights)))

                # Transform to correlated returns with stressed covariance
                simulated_returns = random_returns @ L.T

                # Calculate portfolio returns
                portfolio_sim_returns = simulated_returns @ weights

                # Calculate ES from simulated returns
                var_threshold = np.percentile(portfolio_sim_returns, 5)  # 95% confidence
                tail_returns = portfolio_sim_returns[portfolio_sim_returns <= var_threshold]

                if len(tail_returns) > 0:
                    stressed_es = float(np.mean(tail_returns))
                else:
                    # Fallback: use parametric ES helper
                    stressed_es = _calculate_parametric_es(stressed_portfolio_vol, confidence=0.95)

            except Exception as e:
                print(f"Monte Carlo ES calculation error: {e}")
                # Fallback: use parametric ES helper
                stressed_es = _calculate_parametric_es(stressed_portfolio_vol, confidence=0.95)

        results[scenario_name] = {
            'portfolio_volatility': float(stressed_portfolio_vol),
            'expected_shortfall': float(stressed_es),
            'vol_multiplier': params['vol_mult'],
            'correlation_boost': params['corr_boost']
        }

    return results
