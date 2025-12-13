"""
Phase 5 Dynamic Weighting Integration Module

This module integrates all Phase 5 components:
- Dynamic Ensemble Weighting (DynamicEnsembleWeighter)
- Confidence-Calibrated Position Sizing (ConfidenceAwarePositionSizer)
- Bayesian Signal Combination (BayesianSignalCombiner)
- Multi-Timeframe Ensemble (MultiTimeframeEnsemble)

Expected total improvement: +3-5% profit rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .dynamic_weighter import DynamicEnsembleWeighter, RegimeAwareWeighter
from .confidence_position_sizer import ConfidenceAwarePositionSizer, AdaptivePositionSizer
from .bayesian_combiner import BayesianSignalCombiner, EnhancedBayesianCombiner
from .multi_timeframe_ensemble import MultiTimeframeEnsemble, AdaptiveMultiTimeframeEnsemble


class TradingAction(Enum):
    """Trading action recommendations"""
    STRONG_BUY = 'STRONG_BUY'
    BUY = 'BUY'
    HOLD = 'HOLD'
    SELL = 'SELL'
    STRONG_SELL = 'STRONG_SELL'
    NO_TRADE = 'NO_TRADE'


@dataclass
class Phase5TradingSignal:
    """Complete trading signal from Phase 5 system"""
    ticker: str
    action: TradingAction
    direction: float          # -1 to 1
    confidence: float         # 0 to 1
    position_size: float      # 0 to max_position

    # Component signals
    ensemble_weight: float
    bayesian_confidence: float
    mtf_agreement: float

    # Risk metrics
    risk_adjusted_size: float
    volatility_regime: str

    # Metadata
    signals_used: Dict[str, float]
    reasoning: List[str]


@dataclass
class Phase5PortfolioDecision:
    """Portfolio-level decision from Phase 5"""
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int

    recommended_trades: List[Phase5TradingSignal]
    total_exposure: float
    risk_status: str

    portfolio_confidence: float


class Phase5DynamicWeightingSystem:
    """
    Master integration class for Phase 5 Dynamic Weighting System.

    Combines all Phase 5 components into a unified trading signal
    generation and position sizing system.

    Components:
    1. DynamicEnsembleWeighter - Sharpe-based weight optimization
    2. ConfidenceAwarePositionSizer - Kelly criterion sizing
    3. BayesianSignalCombiner - Signal reliability estimation
    4. MultiTimeframeEnsemble - Multi-timeframe signal agreement
    """

    def __init__(
        self,
        # Dynamic Weighter params
        lookback_period: int = 63,
        min_weight: float = 0.05,
        max_weight: float = 0.35,

        # Position Sizer params
        kelly_fraction: float = 0.25,
        min_position: float = 0.02,
        max_position: float = 0.15,
        max_total_exposure: float = 0.30,

        # Bayesian params
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,

        # Multi-timeframe params
        agreement_threshold: float = 0.6,

        # System params
        confidence_threshold: float = 0.5,
        use_adaptive_components: bool = True
    ):
        """
        Initialize Phase 5 Dynamic Weighting System.

        Args:
            lookback_period: Lookback for performance calculation
            min_weight: Minimum ensemble weight
            max_weight: Maximum ensemble weight
            kelly_fraction: Fraction of Kelly criterion to use
            min_position: Minimum position size
            max_position: Maximum position size
            max_total_exposure: Maximum total portfolio exposure
            prior_alpha: Bayesian prior alpha
            prior_beta: Bayesian prior beta
            agreement_threshold: MTF agreement threshold
            confidence_threshold: Minimum confidence to trade
            use_adaptive_components: Use adaptive versions of components
        """
        self.confidence_threshold = confidence_threshold
        self.use_adaptive = use_adaptive_components

        # Initialize components
        self.dynamic_weighter = DynamicEnsembleWeighter(
            lookback_period=lookback_period,
            min_weight=min_weight,
            max_weight=max_weight
        )

        self.regime_weighter = RegimeAwareWeighter(
            lookback_period=lookback_period,
            min_weight=min_weight,
            max_weight=max_weight
        )

        if use_adaptive_components:
            self.position_sizer = AdaptivePositionSizer(
                kelly_fraction=kelly_fraction,
                min_position=min_position,
                max_position=max_position,
                max_total_exposure=max_total_exposure
            )

            self.signal_combiner = EnhancedBayesianCombiner(
                prior_alpha=prior_alpha,
                prior_beta=prior_beta
            )

            self.mtf_ensemble = AdaptiveMultiTimeframeEnsemble(
                agreement_threshold=agreement_threshold
            )
        else:
            self.position_sizer = ConfidenceAwarePositionSizer(
                kelly_fraction=kelly_fraction,
                min_position=min_position,
                max_position=max_position,
                max_total_exposure=max_total_exposure
            )

            self.signal_combiner = BayesianSignalCombiner(
                prior_alpha=prior_alpha,
                prior_beta=prior_beta
            )

            self.mtf_ensemble = MultiTimeframeEnsemble(
                agreement_threshold=agreement_threshold
            )

        # Performance tracking
        self.trade_history = []
        self.daily_performance = []

    def generate_trading_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
        raw_signals: Dict[str, float],
        current_portfolio: Dict[str, float] = None,
        current_exposure: float = 0.0
    ) -> Phase5TradingSignal:
        """
        Generate complete trading signal using all Phase 5 components.

        Args:
            ticker: Stock ticker
            data: OHLCV data for the ticker
            raw_signals: Raw signals from base models
            current_portfolio: Current portfolio holdings
            current_exposure: Current total exposure

        Returns:
            Phase5TradingSignal with action, position size, and metadata
        """
        reasoning = []

        # 1. Get asset class and dynamic weights
        asset_class = self.dynamic_weighter.get_asset_class(ticker)
        ensemble_weights = self.dynamic_weighter.get_dynamic_weights()
        asset_weight = ensemble_weights.get(asset_class, 0.15)
        reasoning.append(f"Asset class: {asset_class}, weight: {asset_weight:.2%}")

        # 2. Get multi-timeframe signal
        mtf_result = self.mtf_ensemble.generate_multi_timeframe_signal(ticker, data)
        mtf_direction = mtf_result.combined_signal
        mtf_confidence = mtf_result.combined_confidence
        mtf_agreement = mtf_result.agreement_score
        reasoning.append(f"MTF signal: {mtf_direction:.3f}, agreement: {mtf_agreement:.2%}")

        # 3. Combine raw signals with Bayesian weighting
        bayesian_result = self.signal_combiner.combine_signals_bayesian(raw_signals)
        bayesian_signal = bayesian_result['combined_signal']
        bayesian_confidence = bayesian_result['confidence']
        reasoning.append(f"Bayesian signal: {bayesian_signal:.3f}, confidence: {bayesian_confidence:.2%}")

        # 4. Combine MTF and Bayesian signals
        # Weight: 60% MTF (for consistency), 40% Bayesian (for accuracy)
        combined_direction = 0.6 * mtf_direction + 0.4 * bayesian_signal
        combined_confidence = 0.6 * mtf_confidence + 0.4 * bayesian_confidence

        # Adjust confidence based on MTF agreement
        if mtf_agreement >= 0.8:
            combined_confidence *= 1.2  # High agreement bonus
            reasoning.append("High MTF agreement: confidence boosted")
        elif mtf_agreement < 0.5:
            combined_confidence *= 0.8  # Low agreement penalty
            reasoning.append("Low MTF agreement: confidence reduced")

        combined_confidence = min(combined_confidence, 1.0)

        # 5. Get volatility regime
        volatility_regime = self.mtf_ensemble.get_volatility_regime(data)
        reasoning.append(f"Volatility regime: {volatility_regime}")

        # 6. Calculate position size
        signal_data = {
            'ticker': ticker,
            'signal_strength': abs(combined_direction),
            'win_rate': self._estimate_win_rate(ticker, asset_class),
            'win_loss_ratio': self._estimate_win_loss_ratio(ticker, asset_class),
            'confidence': combined_confidence,
            'volatility_regime': volatility_regime
        }

        base_position = self.position_sizer.get_position_size(
            signal_data,
            current_portfolio or {},
            current_exposure
        )

        # 7. Apply regime and asset class adjustments
        regime_multipliers = {
            'low': 1.2,
            'normal': 1.0,
            'high': 0.6,
            'crisis': 0.2
        }
        regime_mult = regime_multipliers.get(volatility_regime, 1.0)

        risk_adjusted_size = base_position * regime_mult * asset_weight / 0.15  # Normalize
        risk_adjusted_size = min(risk_adjusted_size, self.position_sizer.max_position)

        reasoning.append(f"Position: {base_position:.2%} -> {risk_adjusted_size:.2%} (regime adjusted)")

        # 8. Determine action
        action = self._determine_action(combined_direction, combined_confidence)

        if action == TradingAction.NO_TRADE:
            risk_adjusted_size = 0.0
            reasoning.append("Signal below confidence threshold - no trade")

        return Phase5TradingSignal(
            ticker=ticker,
            action=action,
            direction=combined_direction,
            confidence=combined_confidence,
            position_size=risk_adjusted_size,
            ensemble_weight=asset_weight,
            bayesian_confidence=bayesian_confidence,
            mtf_agreement=mtf_agreement,
            risk_adjusted_size=risk_adjusted_size,
            volatility_regime=volatility_regime,
            signals_used=raw_signals,
            reasoning=reasoning
        )

    def _determine_action(
        self,
        direction: float,
        confidence: float
    ) -> TradingAction:
        """Determine trading action from direction and confidence."""
        if confidence < self.confidence_threshold:
            return TradingAction.NO_TRADE

        if direction > 0.3:
            if confidence > 0.7:
                return TradingAction.STRONG_BUY
            return TradingAction.BUY
        elif direction < -0.3:
            if confidence > 0.7:
                return TradingAction.STRONG_SELL
            return TradingAction.SELL
        else:
            return TradingAction.HOLD

    def _estimate_win_rate(self, ticker: str, asset_class: str) -> float:
        """Estimate win rate for position sizing (use historical if available)."""
        # Check trade history
        ticker_trades = [t for t in self.trade_history if t.get('ticker') == ticker]

        if len(ticker_trades) >= 10:
            wins = sum(1 for t in ticker_trades if t.get('profitable', False))
            return wins / len(ticker_trades)

        # Default rates by asset class
        default_rates = {
            'equity': 0.55,
            'crypto': 0.52,
            'commodity': 0.54,
            'forex': 0.53,
            'bond': 0.56,
            'etf': 0.55,
            'international': 0.53
        }
        return default_rates.get(asset_class, 0.54)

    def _estimate_win_loss_ratio(self, ticker: str, asset_class: str) -> float:
        """Estimate win/loss ratio for position sizing."""
        ticker_trades = [t for t in self.trade_history if t.get('ticker') == ticker]

        if len(ticker_trades) >= 10:
            wins = [t['return'] for t in ticker_trades if t.get('profitable', False)]
            losses = [abs(t['return']) for t in ticker_trades if not t.get('profitable', False)]

            if wins and losses:
                return np.mean(wins) / np.mean(losses)

        # Default ratios by asset class
        default_ratios = {
            'equity': 1.3,
            'crypto': 1.5,
            'commodity': 1.2,
            'forex': 1.1,
            'bond': 1.4,
            'etf': 1.3,
            'international': 1.2
        }
        return default_ratios.get(asset_class, 1.25)

    def generate_portfolio_decisions(
        self,
        tickers: List[str],
        data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, Dict[str, float]],
        current_portfolio: Dict[str, float] = None
    ) -> Phase5PortfolioDecision:
        """
        Generate trading decisions for entire portfolio.

        Args:
            tickers: List of tickers to analyze
            data_dict: Dictionary of ticker -> OHLCV data
            signals_dict: Dictionary of ticker -> raw signals
            current_portfolio: Current holdings

        Returns:
            Phase5PortfolioDecision with all recommendations
        """
        current_portfolio = current_portfolio or {}
        recommended_trades = []
        current_exposure = sum(current_portfolio.values())

        for ticker in tickers:
            if ticker not in data_dict or ticker not in signals_dict:
                continue

            signal = self.generate_trading_signal(
                ticker=ticker,
                data=data_dict[ticker],
                raw_signals=signals_dict[ticker],
                current_portfolio=current_portfolio,
                current_exposure=current_exposure
            )

            if signal.action != TradingAction.NO_TRADE:
                recommended_trades.append(signal)

                # Update exposure for next iteration
                if signal.action in [TradingAction.BUY, TradingAction.STRONG_BUY]:
                    current_exposure += signal.risk_adjusted_size

        # Categorize signals
        buy_signals = sum(
            1 for s in recommended_trades
            if s.action in [TradingAction.BUY, TradingAction.STRONG_BUY]
        )
        sell_signals = sum(
            1 for s in recommended_trades
            if s.action in [TradingAction.SELL, TradingAction.STRONG_SELL]
        )
        hold_signals = sum(
            1 for s in recommended_trades
            if s.action == TradingAction.HOLD
        )

        # Calculate portfolio confidence
        if recommended_trades:
            portfolio_confidence = np.mean([s.confidence for s in recommended_trades])
        else:
            portfolio_confidence = 0.5

        # Determine risk status
        if current_exposure > 0.25:
            risk_status = 'HIGH_EXPOSURE'
        elif current_exposure > 0.15:
            risk_status = 'MODERATE'
        else:
            risk_status = 'NORMAL'

        return Phase5PortfolioDecision(
            total_signals=len(recommended_trades),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            recommended_trades=recommended_trades,
            total_exposure=current_exposure,
            risk_status=risk_status,
            portfolio_confidence=portfolio_confidence
        )

    def update_performance(
        self,
        ticker: str,
        actual_return: float,
        signal: Phase5TradingSignal
    ) -> None:
        """
        Update performance tracking with actual trade result.

        Args:
            ticker: Stock ticker
            actual_return: Actual return from trade
            signal: Original signal that generated the trade
        """
        # Update component trackers
        asset_class = self.dynamic_weighter.get_asset_class(ticker)

        # Update dynamic weighter
        predicted_direction = 1 if signal.direction > 0 else -1
        self.dynamic_weighter.update_performance(
            asset_class=asset_class,
            daily_return=actual_return,
            prediction=predicted_direction,
            actual_direction=1 if actual_return > 0 else -1
        )

        # Update Bayesian combiner
        for signal_name, signal_value in signal.signals_used.items():
            self.signal_combiner.update_signal_reliability(
                signal_name=signal_name,
                signal_value=signal_value,
                actual_return=actual_return
            )

        # Update MTF ensemble
        for tf, tf_signal in signal.signals_used.items():
            if tf in ['1h', '4h', '1d', '1w']:
                self.mtf_ensemble.update_performance(
                    timeframe=tf,
                    predicted_direction=tf_signal,
                    actual_return=actual_return
                )

        # Record trade
        self.trade_history.append({
            'ticker': ticker,
            'return': actual_return,
            'profitable': actual_return > 0,
            'signal_direction': signal.direction,
            'confidence': signal.confidence,
            'position_size': signal.position_size
        })

        # Trim history to last 500 trades
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary of statistics from all components
        """
        stats = {
            'phase5_version': '1.0',
            'components': {
                'dynamic_weighter': {
                    'current_weights': self.dynamic_weighter.get_dynamic_weights(),
                    'performance_metrics': {
                        ac: self.dynamic_weighter.calculate_performance_metrics(ac)
                        for ac in self.dynamic_weighter.performance_history.keys()
                    }
                },
                'position_sizer': {
                    'kelly_fraction': self.position_sizer.kelly_fraction,
                    'min_position': self.position_sizer.min_position,
                    'max_position': self.position_sizer.max_position
                },
                'bayesian_combiner': {
                    'signal_reliabilities': {
                        name: self.signal_combiner.get_signal_reliability(name)
                        for name in self.signal_combiner.signal_priors.keys()
                    }
                },
                'mtf_ensemble': {
                    'timeframe_stats': self.mtf_ensemble.get_timeframe_statistics(),
                    'current_weights': {
                        tf: config['weight']
                        for tf, config in self.mtf_ensemble.timeframes.items()
                    }
                }
            },
            'trade_history': {
                'total_trades': len(self.trade_history),
                'profitable_trades': sum(
                    1 for t in self.trade_history if t.get('profitable', False)
                ),
                'win_rate': (
                    sum(1 for t in self.trade_history if t.get('profitable', False))
                    / len(self.trade_history)
                ) if self.trade_history else 0,
                'avg_return': np.mean(
                    [t['return'] for t in self.trade_history]
                ) if self.trade_history else 0
            }
        }

        return stats


def create_phase5_system(
    config: Dict[str, Any] = None
) -> Phase5DynamicWeightingSystem:
    """
    Factory function to create Phase 5 system with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Phase5DynamicWeightingSystem
    """
    default_config = {
        'lookback_period': 63,
        'min_weight': 0.05,
        'max_weight': 0.35,
        'kelly_fraction': 0.25,
        'min_position': 0.02,
        'max_position': 0.15,
        'max_total_exposure': 0.30,
        'prior_alpha': 1.0,
        'prior_beta': 1.0,
        'agreement_threshold': 0.6,
        'confidence_threshold': 0.5,
        'use_adaptive_components': True
    }

    if config:
        default_config.update(config)

    return Phase5DynamicWeightingSystem(**default_config)


def integrate_phase5_with_trading_system(
    trading_system: Any,
    phase5_system: Phase5DynamicWeightingSystem = None
) -> Any:
    """
    Integrate Phase 5 system with existing trading system.

    Args:
        trading_system: Existing trading system instance
        phase5_system: Phase 5 system (created if not provided)

    Returns:
        Enhanced trading system
    """
    if phase5_system is None:
        phase5_system = create_phase5_system()

    # Store Phase 5 system reference
    trading_system.phase5 = phase5_system

    # Monkey-patch enhanced signal generation if trading system has generate_signal
    if hasattr(trading_system, 'generate_signal'):
        original_generate_signal = trading_system.generate_signal

        def enhanced_generate_signal(ticker, data, **kwargs):
            # Get original signal
            original_signal = original_generate_signal(ticker, data, **kwargs)

            # Enhance with Phase 5
            if isinstance(original_signal, dict):
                raw_signals = original_signal.get('signals', {ticker: original_signal.get('signal', 0)})
            else:
                raw_signals = {ticker: float(original_signal)}

            phase5_signal = phase5_system.generate_trading_signal(
                ticker=ticker,
                data=data,
                raw_signals=raw_signals
            )

            return {
                'original': original_signal,
                'phase5': phase5_signal,
                'final_action': phase5_signal.action.value,
                'final_position': phase5_signal.risk_adjusted_size
            }

        trading_system.generate_signal = enhanced_generate_signal

    return trading_system
