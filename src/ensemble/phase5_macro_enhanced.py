"""
Phase 5 Macro-Enhanced Module

Implements the 6 critical improvements from 'phase5 improvement advice on C model.pdf':
1. Integrate Phase 4 Macro Context into Dynamic Weighting (+2-3%)
2. Enhance Bayesian Signal Combination with Macro Priors (+0.5-1%)
3. Add Regime-Aware Position Multipliers to Kelly Criterion (+2-3%)
4. Implement Cross-Asset Correlation in Diversification Penalty (+1-2%)
5. Add Time-Varying Feature Importance to MTF Ensemble (+1-2%)
6. Add Performance-Based Ensemble Staleness Detection (+0.5-1%)

Total Expected Improvement: +5-8% additional profit rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque


# =============================================================================
# 1. MACRO CONTEXT INTEGRATION INTO DYNAMIC WEIGHTING
# =============================================================================

def macro_adjusted_composite_score(
    composite_score: float,
    asset_class: str,
    macro_context: Dict[str, Any]
) -> float:
    """
    Adjust ensemble weights based on macro regime from Phase 4.

    Args:
        composite_score: Base composite score from dynamic weighter
        asset_class: Asset class name
        macro_context: Dictionary with 'risk_off', 'vix', 'spy_trend', etc.

    Returns:
        Macro-adjusted composite score
    """
    # Risk-off regime: favor defensive ensembles
    defensive_boost = {
        'bond': 1.3,        # Boost bond ensembles
        'commodity': 1.2,   # Boost commodities (gold)
        'equity': 0.7,      # Reduce equity exposure
        'crypto': 0.5,      # Significantly reduce crypto
        'etf': 0.9,
        'forex': 1.1,
        'international': 0.8
    }

    # Risk-on regime: favor growth ensembles
    growth_boost = {
        'equity': 1.3,
        'crypto': 1.2,
        'etf': 1.1,
        'international': 1.1,
        'bond': 0.8,
        'commodity': 0.9,
        'forex': 0.95
    }

    # Determine regime
    is_risk_off = macro_context.get('risk_off', False)

    if is_risk_off:
        regime_boost = defensive_boost
    else:
        regime_boost = growth_boost

    return composite_score * regime_boost.get(asset_class, 1.0)


class MacroAwareDynamicWeighter:
    """
    Enhanced Dynamic Ensemble Weighter with Phase 4 macro integration.
    """

    def __init__(
        self,
        base_weighter: Any,
        vix_warning_level: float = 25.0,
        vix_crisis_level: float = 35.0
    ):
        """
        Initialize macro-aware weighter.

        Args:
            base_weighter: Base DynamicEnsembleWeighter instance
            vix_warning_level: VIX level for warning regime
            vix_crisis_level: VIX level for crisis regime
        """
        self.base_weighter = base_weighter
        self.vix_warning_level = vix_warning_level
        self.vix_crisis_level = vix_crisis_level

    def get_macro_adjusted_weights(
        self,
        macro_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get dynamic weights adjusted for macro context.

        Args:
            macro_context: Dictionary with macro indicators

        Returns:
            Macro-adjusted weights by asset class
        """
        # Get base weights
        base_weights = self.base_weighter.get_dynamic_weights()

        # Adjust each weight based on macro context
        adjusted_weights = {}
        for asset_class, weight in base_weights.items():
            # Get composite score (approximate from weight)
            composite_score = weight * len(base_weights)

            # Apply macro adjustment
            adjusted_score = macro_adjusted_composite_score(
                composite_score, asset_class, macro_context
            )

            adjusted_weights[asset_class] = adjusted_score

        # Normalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def determine_macro_regime(
        self,
        vix: float,
        spy_trend: float,
        gld_trend: float = 0,
        dxy_trend: float = 0
    ) -> Dict[str, Any]:
        """
        Determine macro regime from indicators.

        Args:
            vix: Current VIX level
            spy_trend: SPY trend (positive = bullish)
            gld_trend: Gold trend
            dxy_trend: Dollar index trend

        Returns:
            Macro context dictionary
        """
        # Determine risk-off conditions
        risk_off_signals = 0

        if vix > self.vix_warning_level:
            risk_off_signals += 1
        if vix > self.vix_crisis_level:
            risk_off_signals += 1
        if spy_trend < -0.02:  # SPY down more than 2%
            risk_off_signals += 1
        if gld_trend > 0.02:   # Gold up more than 2% (flight to safety)
            risk_off_signals += 1
        if dxy_trend > 0.01:   # Dollar strengthening
            risk_off_signals += 0.5

        is_risk_off = risk_off_signals >= 2

        return {
            'risk_off': is_risk_off,
            'vix': vix,
            'spy_trend': spy_trend,
            'gld_trend': gld_trend,
            'dxy_trend': dxy_trend,
            'risk_score': risk_off_signals,
            'regime': 'risk_off' if is_risk_off else 'risk_on'
        }


# =============================================================================
# 2. MACRO-INFORMED BAYESIAN PRIORS
# =============================================================================

def get_macro_informed_prior(macro_context: Dict[str, Any]) -> Dict[str, float]:
    """
    Set Bayesian prior based on current market regime.

    Args:
        macro_context: Dictionary with 'vix', 'spy_trend', etc.

    Returns:
        Dictionary with 'alpha' and 'beta' prior parameters
    """
    vix_level = macro_context.get('vix', 20)
    market_trend = macro_context.get('spy_trend', 0)

    if vix_level > 30:  # High fear regime
        # More skeptical prior in volatile markets
        return {'alpha': 2.0, 'beta': 3.0}
    elif market_trend > 0:  # Bull market
        # More optimistic prior in uptrends
        return {'alpha': 3.0, 'beta': 2.0}
    else:  # Normal conditions
        return {'alpha': 2.0, 'beta': 2.0}


class MacroAwareBayesianCombiner:
    """
    Bayesian Signal Combiner with macro-informed priors.
    """

    def __init__(
        self,
        base_combiner: Any,
        decay_factor: float = 0.99
    ):
        """
        Initialize macro-aware Bayesian combiner.

        Args:
            base_combiner: Base BayesianSignalCombiner instance
            decay_factor: Decay factor for historical observations
        """
        self.base_combiner = base_combiner
        self.decay_factor = decay_factor
        self.macro_priors = {}

    def update_macro_priors(self, macro_context: Dict[str, Any]) -> None:
        """
        Update priors based on current macro context.

        Args:
            macro_context: Current macro indicators
        """
        self.macro_priors = get_macro_informed_prior(macro_context)

        # Update base combiner's priors
        if hasattr(self.base_combiner, 'prior_alpha'):
            self.base_combiner.prior_alpha = self.macro_priors['alpha']
            self.base_combiner.prior_beta = self.macro_priors['beta']

    def combine_signals_with_macro(
        self,
        signals: Dict[str, float],
        macro_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine signals using macro-informed Bayesian approach.

        Args:
            signals: Dictionary of signal name -> signal value
            macro_context: Current macro indicators

        Returns:
            Combined signal result
        """
        # Update priors based on macro
        self.update_macro_priors(macro_context)

        # Use base combiner
        result = self.base_combiner.combine_signals_bayesian(signals)

        # Apply additional macro adjustment to confidence
        vix = macro_context.get('vix', 20)
        if vix > 30:
            # Reduce confidence in high volatility
            result['confidence'] *= 0.8
        elif vix < 15:
            # Boost confidence in low volatility
            result['confidence'] *= 1.1

        result['confidence'] = min(result['confidence'], 1.0)
        result['macro_prior'] = self.macro_priors

        return result


# =============================================================================
# 3. REGIME-AWARE KELLY CRITERION
# =============================================================================

def regime_aware_kelly(
    win_rate: float,
    win_loss_ratio: float,
    regime: str,
    macro_context: Dict[str, Any],
    kelly_fraction: float = 0.25
) -> float:
    """
    Apply regime-specific adjustments to Kelly sizing.

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / average loss
        regime: Volatility regime ('low_vol', 'normal', 'high_vol', 'crisis')
        macro_context: Dictionary with macro indicators
        kelly_fraction: Base Kelly fraction (default: 0.25 = quarter-Kelly)

    Returns:
        Regime-adjusted position size
    """
    # Base Kelly calculation
    base_kelly = win_rate - (1 - win_rate) / win_loss_ratio

    # Ensure non-negative
    if base_kelly <= 0:
        return 0.0

    quarter_kelly = base_kelly * kelly_fraction

    # Regime-based multipliers from Phase 3
    regime_multipliers = {
        'low_vol': 1.2,     # Low volatility: increase position
        'low': 1.2,
        'normal': 1.0,      # Normal: standard position
        'high_vol': 0.6,    # High volatility: reduce position
        'high': 0.6,
        'crisis': 0.2       # Crisis: minimal position
    }

    regime_mult = regime_multipliers.get(regime, 1.0)

    # Additional macro adjustment using VIX
    vix = macro_context.get('vix', 20)
    vix_adjustment = 1.0 - (vix - 15) / 100  # Normalize VIX impact
    vix_adjustment = max(0.5, min(1.5, vix_adjustment))

    return quarter_kelly * regime_mult * vix_adjustment


class RegimeAwarePositionSizer:
    """
    Position sizer with regime-aware Kelly criterion adjustments.
    """

    def __init__(
        self,
        base_sizer: Any,
        min_position: float = 0.02,
        max_position: float = 0.15
    ):
        """
        Initialize regime-aware position sizer.

        Args:
            base_sizer: Base ConfidenceAwarePositionSizer instance
            min_position: Minimum position size
            max_position: Maximum position size
        """
        self.base_sizer = base_sizer
        self.min_position = min_position
        self.max_position = max_position

    def get_regime_adjusted_position(
        self,
        signal_data: Dict[str, Any],
        regime: str,
        macro_context: Dict[str, Any],
        portfolio: Dict[str, float] = None,
        current_exposure: float = 0.0
    ) -> float:
        """
        Get position size with regime adjustment.

        Args:
            signal_data: Signal information
            regime: Current volatility regime
            macro_context: Macro indicators
            portfolio: Current portfolio holdings
            current_exposure: Current total exposure

        Returns:
            Adjusted position size
        """
        # Get base position
        base_position = self.base_sizer.get_position_size(
            signal_data, portfolio or {}, current_exposure
        )

        # Apply regime-aware Kelly adjustment
        win_rate = signal_data.get('win_rate', 0.55)
        win_loss_ratio = signal_data.get('win_loss_ratio', 1.3)

        regime_adjusted = regime_aware_kelly(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            regime=regime,
            macro_context=macro_context,
            kelly_fraction=self.base_sizer.kelly_fraction
        )

        # Blend base and regime-adjusted (60% regime, 40% base)
        final_position = 0.6 * regime_adjusted + 0.4 * base_position

        # Apply bounds
        final_position = max(self.min_position, min(self.max_position, final_position))

        return final_position


# =============================================================================
# 4. ADVANCED CROSS-ASSET CORRELATION DIVERSIFICATION PENALTY
# =============================================================================

@dataclass
class CorrelationCluster:
    """Correlation cluster information."""
    cluster_id: int
    members: List[str]
    avg_correlation: float


class AdvancedDiversificationPenalty:
    """
    Enhanced diversification penalty using PCA-based correlation clusters.
    """

    def __init__(self, correlation_lookback: int = 60):
        """
        Initialize advanced diversification penalty calculator.

        Args:
            correlation_lookback: Days for rolling correlation
        """
        self.correlation_lookback = correlation_lookback
        self.correlation_matrix = {}
        self.clusters = {}

        # Default sector clusters
        self.sector_clusters = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'COIN'],
            'International': ['BABA', 'TSM', '0700.HK', '9988.HK'],
            'Bonds': ['TLT', 'IEF', 'BND', 'AGG'],
            'Commodities': ['GLD', 'SLV', 'USO', 'UNG']
        }

    def get_cluster(self, ticker: str) -> str:
        """
        Get cluster for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Cluster name
        """
        ticker_upper = ticker.upper()

        for cluster_name, members in self.sector_clusters.items():
            if ticker_upper in members:
                return cluster_name

        # Default cluster based on patterns
        if '-USD' in ticker_upper or ticker_upper in ['BTC', 'ETH']:
            return 'Crypto'
        elif '.HK' in ticker_upper or '.SS' in ticker_upper:
            return 'International'
        elif '=X' in ticker_upper:
            return 'Forex'
        else:
            return 'General'

    def get_correlation(self, ticker1: str, ticker2: str) -> float:
        """
        Get correlation between two tickers.

        Args:
            ticker1: First ticker
            ticker2: Second ticker

        Returns:
            Correlation coefficient
        """
        # Check cached correlation
        key = tuple(sorted([ticker1.upper(), ticker2.upper()]))
        if key in self.correlation_matrix:
            return self.correlation_matrix[key]

        # Default correlations by cluster
        cluster1 = self.get_cluster(ticker1)
        cluster2 = self.get_cluster(ticker2)

        if cluster1 == cluster2:
            return 0.7  # Same cluster: high correlation

        # Cross-cluster default correlations
        cross_correlations = {
            ('Technology', 'Consumer'): 0.5,
            ('Technology', 'Finance'): 0.4,
            ('Finance', 'Energy'): 0.3,
            ('Crypto', 'Technology'): 0.4,
            ('Bonds', 'Equity'): -0.2,
            ('Commodities', 'Equity'): 0.2,
        }

        key_pair = tuple(sorted([cluster1, cluster2]))
        return cross_correlations.get(key_pair, 0.3)

    def update_correlation(
        self,
        ticker1: str,
        ticker2: str,
        correlation: float
    ) -> None:
        """
        Update correlation between two tickers.

        Args:
            ticker1: First ticker
            ticker2: Second ticker
            correlation: Calculated correlation
        """
        key = tuple(sorted([ticker1.upper(), ticker2.upper()]))
        self.correlation_matrix[key] = correlation

    def calculate_penalty(
        self,
        new_ticker: str,
        portfolio: Dict[str, float],
        macro_context: Dict[str, Any]
    ) -> float:
        """
        Calculate advanced diversification penalty.

        Args:
            new_ticker: Ticker to add
            portfolio: Current portfolio {ticker: weight}
            macro_context: Macro indicators

        Returns:
            Penalty factor (0 to 0.6)
        """
        if not portfolio:
            return 0.0

        new_cluster = self.get_cluster(new_ticker)
        penalty = 0.0

        for holding_ticker, holding_weight in portfolio.items():
            holding_cluster = self.get_cluster(holding_ticker)

            # Intra-cluster penalty (higher for same cluster)
            if new_cluster == holding_cluster:
                cluster_penalty = 0.3
            # Cross-cluster but correlated
            elif self.get_correlation(new_ticker, holding_ticker) > 0.7:
                cluster_penalty = 0.2
            else:
                cluster_penalty = 0.05

            penalty += cluster_penalty * holding_weight

        # Macro adjustment: reduce diversification benefits in crisis
        vix = macro_context.get('vix', 20)
        if vix > 30:
            penalty = min(0.7, penalty * 1.5)  # Increase penalty in crises

        return min(penalty, 0.6)  # Cap at 60% reduction


# =============================================================================
# 5. TIME-VARYING TIMEFRAME WEIGHTS FOR MTF ENSEMBLE
# =============================================================================

def dynamic_timeframe_weights(
    volatility_regime: str,
    macro_trend: float,
    recent_performance: Dict[str, float]
) -> Dict[str, float]:
    """
    Adjust timeframe weights based on market conditions.

    Args:
        volatility_regime: Current volatility regime
        macro_trend: Overall macro trend direction
        recent_performance: Recent performance by timeframe

    Returns:
        Adjusted timeframe weights
    """
    base_weights = {
        '1h': 0.15,
        '4h': 0.25,
        '1d': 0.35,
        '1w': 0.25
    }

    # Volatility-based adjustments
    if volatility_regime in ['high', 'high_vol']:
        # Reduce short-term noise in high volatility
        adjustments = {'1h': 0.8, '4h': 0.9, '1d': 1.1, '1w': 1.2}
    elif volatility_regime in ['low', 'low_vol']:
        # Emphasize short-term in calm markets
        adjustments = {'1h': 1.2, '4h': 1.1, '1d': 0.9, '1w': 0.8}
    elif volatility_regime == 'crisis':
        # Crisis: heavily favor longer timeframes
        adjustments = {'1h': 0.5, '4h': 0.7, '1d': 1.2, '1w': 1.4}
    else:
        adjustments = {'1h': 1.0, '4h': 1.0, '1d': 1.0, '1w': 1.0}

    # Apply recent performance momentum
    short_term_perf = recent_performance.get('short_term', 0)
    long_term_perf = recent_performance.get('long_term', 0)

    if short_term_perf > long_term_perf:
        # Boost shorter timeframes when they're performing well
        adjustments['1h'] *= 1.1
        adjustments['4h'] *= 1.05
    elif long_term_perf > short_term_perf:
        # Boost longer timeframes when they're performing well
        adjustments['1d'] *= 1.05
        adjustments['1w'] *= 1.1

    # Calculate final weights
    final_weights = {
        tf: base_weights[tf] * adjustments[tf]
        for tf in base_weights
    }

    # Normalize
    total = sum(final_weights.values())
    return {tf: w / total for tf, w in final_weights.items()}


class DynamicMTFWeighter:
    """
    Multi-timeframe ensemble with dynamic weight adjustment.
    """

    def __init__(self, base_mtf_ensemble: Any):
        """
        Initialize dynamic MTF weighter.

        Args:
            base_mtf_ensemble: Base MultiTimeframeEnsemble instance
        """
        self.base_ensemble = base_mtf_ensemble
        self.performance_history = {
            '1h': deque(maxlen=50),
            '4h': deque(maxlen=50),
            '1d': deque(maxlen=50),
            '1w': deque(maxlen=50)
        }

    def update_performance(
        self,
        timeframe: str,
        accuracy: float
    ) -> None:
        """
        Update performance history for a timeframe.

        Args:
            timeframe: Timeframe identifier
            accuracy: Prediction accuracy (0-1)
        """
        if timeframe in self.performance_history:
            self.performance_history[timeframe].append(accuracy)

    def get_recent_performance(self) -> Dict[str, float]:
        """
        Get recent performance summary.

        Returns:
            Performance metrics
        """
        short_term_acc = []
        long_term_acc = []

        for tf in ['1h', '4h']:
            if self.performance_history[tf]:
                short_term_acc.extend(list(self.performance_history[tf])[-10:])

        for tf in ['1d', '1w']:
            if self.performance_history[tf]:
                long_term_acc.extend(list(self.performance_history[tf])[-10:])

        return {
            'short_term': np.mean(short_term_acc) if short_term_acc else 0.5,
            'long_term': np.mean(long_term_acc) if long_term_acc else 0.5
        }

    def get_dynamic_weights(
        self,
        volatility_regime: str,
        macro_trend: float
    ) -> Dict[str, float]:
        """
        Get dynamically adjusted timeframe weights.

        Args:
            volatility_regime: Current volatility regime
            macro_trend: Macro trend direction

        Returns:
            Adjusted weights
        """
        recent_perf = self.get_recent_performance()

        return dynamic_timeframe_weights(
            volatility_regime=volatility_regime,
            macro_trend=macro_trend,
            recent_performance=recent_perf
        )


# =============================================================================
# 6. ENSEMBLE STALENESS DETECTION
# =============================================================================

def calculate_sharpe(returns: List[float]) -> float:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: List of returns

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 5:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return < 0.0001:
        return 0.0

    return (mean_return * 252) / (std_return * np.sqrt(252))


def ensemble_staleness_detection(
    ensemble_performance: List[float],
    lookback_days: int = 21
) -> Dict[str, Any]:
    """
    Detect when ensemble weights need refreshing.

    Args:
        ensemble_performance: List of daily returns
        lookback_days: Days for comparison

    Returns:
        Staleness detection result
    """
    if len(ensemble_performance) < lookback_days * 2:
        return {
            'needs_refresh': False,
            'degradation_ratio': 1.0,
            'recommended_action': 'insufficient_data'
        }

    recent_perf = ensemble_performance[-lookback_days:]
    historical_perf = ensemble_performance[-(lookback_days * 2):-lookback_days]

    # Calculate performance degradation
    recent_sharpe = calculate_sharpe(recent_perf)
    historical_sharpe = calculate_sharpe(historical_perf)

    if historical_sharpe > 0:
        degradation_ratio = recent_sharpe / historical_sharpe
    else:
        degradation_ratio = 1.0 if recent_sharpe >= 0 else 0.5

    # Trigger refresh if significant degradation
    refresh_threshold = 0.7  # 30% degradation
    needs_refresh = degradation_ratio < refresh_threshold

    return {
        'needs_refresh': needs_refresh,
        'degradation_ratio': degradation_ratio,
        'recent_sharpe': recent_sharpe,
        'historical_sharpe': historical_sharpe,
        'recommended_action': 'recalculate_weights' if needs_refresh else 'maintain_weights'
    }


class EnsembleStalenessMonitor:
    """
    Monitor for ensemble weight staleness.
    """

    def __init__(
        self,
        lookback_days: int = 21,
        refresh_threshold: float = 0.7,
        min_samples: int = 42
    ):
        """
        Initialize staleness monitor.

        Args:
            lookback_days: Days for performance comparison
            refresh_threshold: Degradation ratio to trigger refresh
            min_samples: Minimum samples before monitoring
        """
        self.lookback_days = lookback_days
        self.refresh_threshold = refresh_threshold
        self.min_samples = min_samples
        self.performance_history = deque(maxlen=252)  # 1 year
        self.last_refresh_date = None
        self.refresh_count = 0

    def add_performance(self, daily_return: float) -> None:
        """
        Add daily performance observation.

        Args:
            daily_return: Daily return
        """
        self.performance_history.append(daily_return)

    def check_staleness(self) -> Dict[str, Any]:
        """
        Check if ensemble weights are stale.

        Returns:
            Staleness check result
        """
        return ensemble_staleness_detection(
            list(self.performance_history),
            self.lookback_days
        )

    def record_refresh(self) -> None:
        """Record that weights were refreshed."""
        from datetime import datetime
        self.last_refresh_date = datetime.now()
        self.refresh_count += 1


# =============================================================================
# INTEGRATED MACRO-ENHANCED PHASE 5 SYSTEM
# =============================================================================

class MacroEnhancedPhase5System:
    """
    Fully integrated Phase 5 system with all 6 macro enhancements.
    """

    def __init__(
        self,
        base_phase5_system: Any,
        vix_warning_level: float = 25.0,
        vix_crisis_level: float = 35.0
    ):
        """
        Initialize macro-enhanced Phase 5 system.

        Args:
            base_phase5_system: Base Phase5DynamicWeightingSystem
            vix_warning_level: VIX warning threshold
            vix_crisis_level: VIX crisis threshold
        """
        self.base_system = base_phase5_system
        self.vix_warning_level = vix_warning_level
        self.vix_crisis_level = vix_crisis_level

        # Initialize enhancement components
        self.macro_weighter = MacroAwareDynamicWeighter(
            base_phase5_system.dynamic_weighter,
            vix_warning_level,
            vix_crisis_level
        )

        self.macro_bayesian = MacroAwareBayesianCombiner(
            base_phase5_system.signal_combiner
        )

        self.regime_sizer = RegimeAwarePositionSizer(
            base_phase5_system.position_sizer
        )

        self.diversification = AdvancedDiversificationPenalty()

        self.mtf_weighter = DynamicMTFWeighter(
            base_phase5_system.mtf_ensemble
        )

        self.staleness_monitor = EnsembleStalenessMonitor()

    def generate_enhanced_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
        raw_signals: Dict[str, float],
        macro_context: Dict[str, Any],
        portfolio: Dict[str, float] = None,
        current_exposure: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate trading signal with all macro enhancements.

        Args:
            ticker: Stock ticker
            data: OHLCV data
            raw_signals: Raw signals from models
            macro_context: Macro indicators
            portfolio: Current portfolio
            current_exposure: Current exposure

        Returns:
            Enhanced trading signal
        """
        # 1. Get macro-adjusted weights
        macro_weights = self.macro_weighter.get_macro_adjusted_weights(macro_context)

        # 2. Combine signals with macro-aware Bayesian
        bayesian_result = self.macro_bayesian.combine_signals_with_macro(
            raw_signals, macro_context
        )

        # 3. Get MTF signal with dynamic weights
        volatility_regime = self.base_system.mtf_ensemble.get_volatility_regime(data)
        macro_trend = macro_context.get('spy_trend', 0)

        mtf_weights = self.mtf_weighter.get_dynamic_weights(
            volatility_regime, macro_trend
        )

        # Update MTF ensemble weights
        for tf, weight in mtf_weights.items():
            if tf in self.base_system.mtf_ensemble.timeframes:
                self.base_system.mtf_ensemble.timeframes[tf]['weight'] = weight

        mtf_result = self.base_system.mtf_ensemble.generate_multi_timeframe_signal(
            ticker, data
        )

        # 4. Calculate diversification penalty
        div_penalty = self.diversification.calculate_penalty(
            ticker, portfolio or {}, macro_context
        )

        # 5. Get regime-aware position size
        signal_data = {
            'ticker': ticker,
            'signal_strength': abs(bayesian_result['combined_signal']),
            'win_rate': 0.55,
            'win_loss_ratio': 1.3,
            'confidence': bayesian_result['confidence']
        }

        base_position = self.regime_sizer.get_regime_adjusted_position(
            signal_data=signal_data,
            regime=volatility_regime,
            macro_context=macro_context,
            portfolio=portfolio,
            current_exposure=current_exposure
        )

        # Apply diversification penalty
        final_position = base_position * (1 - div_penalty)

        # 6. Check staleness
        staleness_check = self.staleness_monitor.check_staleness()

        # Combine all results
        combined_direction = (
            0.5 * bayesian_result['combined_signal'] +
            0.5 * mtf_result.combined_signal
        )

        combined_confidence = (
            0.5 * bayesian_result['confidence'] +
            0.5 * mtf_result.combined_confidence
        )

        # Adjust confidence if staleness detected
        if staleness_check['needs_refresh']:
            combined_confidence *= 0.8

        # Get asset class weight
        asset_class = self.base_system.dynamic_weighter.get_asset_class(ticker)
        asset_weight = macro_weights.get(asset_class, 0.15)

        return {
            'ticker': ticker,
            'direction': combined_direction,
            'confidence': combined_confidence,
            'position_size': final_position,
            'asset_class': asset_class,
            'asset_weight': asset_weight,
            'volatility_regime': volatility_regime,
            'macro_regime': macro_context.get('regime', 'unknown'),
            'diversification_penalty': div_penalty,
            'staleness_check': staleness_check,
            'mtf_agreement': mtf_result.agreement_score,
            'mtf_weights': mtf_weights,
            'bayesian_prior': bayesian_result.get('macro_prior', {})
        }

    def update_performance(self, daily_return: float) -> None:
        """
        Update performance tracking.

        Args:
            daily_return: Daily portfolio return
        """
        self.staleness_monitor.add_performance(daily_return)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_macro_enhanced_phase5(
    base_system: Any,
    config: Dict[str, Any] = None
) -> MacroEnhancedPhase5System:
    """
    Factory function to create macro-enhanced Phase 5 system.

    Args:
        base_system: Base Phase5DynamicWeightingSystem
        config: Configuration options

    Returns:
        MacroEnhancedPhase5System instance
    """
    config = config or {}

    return MacroEnhancedPhase5System(
        base_phase5_system=base_system,
        vix_warning_level=config.get('vix_warning_level', 25.0),
        vix_crisis_level=config.get('vix_crisis_level', 35.0)
    )
