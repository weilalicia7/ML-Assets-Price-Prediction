"""
Trading System Conflict Resolver - Master Integration

Integrates all Phase 2 conflict resolutions into a unified system:
- UnifiedDrawdownManager (Phase 2 thresholds: 5%/10%/15%)
- ResolvedPositionSizer (Quarter-Kelly: 0.25)
- UnifiedRegimeDetector (GMM + Transition)

This class ensures all components work together without conflicts.

Based on: phase2 fixing on C model_conflict resolutions.pdf
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging

from .unified_drawdown_manager import UnifiedDrawdownManager
from .resolved_position_sizer import ResolvedPositionSizer
from .unified_regime_detector import UnifiedRegimeDetector

logger = logging.getLogger(__name__)


class TradingSystemConflictResolver:
    """
    Master conflict resolver integrating all Phase 2 improvements.

    Ensures consistent behavior across:
    - Drawdown management (Phase 2 conservative thresholds)
    - Position sizing (Quarter-Kelly)
    - Regime detection (Combined GMM + transition)

    All conflicts are resolved in favor of Phase 2 approaches.
    """

    def __init__(
        self,
        # Drawdown parameters (Phase 2)
        warning_threshold: float = 0.05,
        danger_threshold: float = 0.10,
        max_drawdown: float = 0.15,
        # Position sizing parameters (Phase 2)
        kelly_fraction: float = 0.25,
        max_position: float = 0.30,
        # Regime parameters
        n_regimes: int = 4,
        # General
        initial_capital: float = 100000.0
    ):
        """
        Initialize conflict resolver with all components.

        Args:
            warning_threshold: Drawdown warning level (5%)
            danger_threshold: Drawdown danger level (10%)
            max_drawdown: Maximum drawdown before stopping (15%)
            kelly_fraction: Kelly fraction for sizing (0.25 = quarter-Kelly)
            max_position: Maximum position size (30%)
            n_regimes: Number of regime states
            initial_capital: Starting capital
        """
        # Initialize components
        self.drawdown_manager = UnifiedDrawdownManager(
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            max_threshold=max_drawdown
        )

        self.position_sizer = ResolvedPositionSizer(
            kelly_fraction=kelly_fraction,
            max_position=max_position
        )

        self.regime_detector = UnifiedRegimeDetector(
            n_regimes=n_regimes
        )

        # Portfolio state
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital

        # Configuration
        self.config = {
            'warning_threshold': warning_threshold,
            'danger_threshold': danger_threshold,
            'max_drawdown': max_drawdown,
            'kelly_fraction': kelly_fraction,
            'max_position': max_position,
            'n_regimes': n_regimes
        }

        logger.info("Initialized TradingSystemConflictResolver")
        logger.info(f"  Drawdown: {warning_threshold:.1%}/{danger_threshold:.1%}/{max_drawdown:.1%}")
        logger.info(f"  Kelly: {kelly_fraction:.0%}, Max position: {max_position:.1%}")

    def fit_regime_detector(self, data: pd.DataFrame, vol_col: str = 'volatility'):
        """
        Fit regime detector on historical data.

        Args:
            data: Historical data with volatility
            vol_col: Volatility column name
        """
        self.regime_detector.fit(data, vol_col)
        logger.info("Regime detector fitted on historical data")

    def update_portfolio(self, portfolio_value: float, daily_return: float = 0.0):
        """
        Update portfolio state.

        Args:
            portfolio_value: Current portfolio value
            daily_return: Today's return
        """
        self.portfolio_value = portfolio_value
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Update drawdown manager
        self.drawdown_manager.update(portfolio_value, daily_return)

    def record_trade_outcome(
        self,
        ticker: str,
        was_profitable: bool,
        profit_pct: float = 0.0,
        loss_pct: float = 0.0
    ):
        """
        Record a completed trade for statistics.

        Args:
            ticker: Stock ticker
            was_profitable: Whether trade was profitable
            profit_pct: Profit percentage
            loss_pct: Loss percentage
        """
        self.position_sizer.record_trade(
            ticker=ticker,
            was_profitable=was_profitable,
            profit_pct=profit_pct,
            loss_pct=loss_pct
        )

    def get_trading_decision(
        self,
        ticker: str,
        signal_confidence: float,
        signal_direction: str,
        current_volatility: float,
        current_price: float,
        daily_return: float = 0.0
    ) -> Dict:
        """
        Get comprehensive trading decision with all conflict resolutions applied.

        This is the main entry point that combines all components.

        Args:
            ticker: Stock ticker
            signal_confidence: Model confidence (0-1)
            signal_direction: 'LONG', 'SHORT', or 'HOLD'
            current_volatility: Current volatility
            current_price: Current price
            daily_return: Today's return

        Returns:
            Dict with complete trading decision
        """
        # 1. Update drawdown state
        drawdown_state = self.drawdown_manager.update(
            self.portfolio_value,
            daily_return
        )

        # 2. Get regime state
        regime_state = self.regime_detector.update(current_volatility)

        # 3. Check if trading is allowed
        can_trade = (
            drawdown_state['can_trade'] and
            regime_state['should_trade'] and
            signal_direction != 'HOLD'
        )

        # 4. Calculate combined multiplier
        combined_multiplier = (
            drawdown_state['position_multiplier'] *
            regime_state['position_multiplier']
        )

        # 5. Calculate position size (if trading)
        if can_trade and combined_multiplier > 0:
            position_value, sizing_details = self.position_sizer.get_position_value(
                portfolio_value=self.portfolio_value,
                confidence=signal_confidence,
                ticker=ticker,
                volatility=current_volatility,
                drawdown_multiplier=combined_multiplier
            )
            shares = int(position_value / current_price) if current_price > 0 else 0
        else:
            position_value = 0
            shares = 0
            sizing_details = {
                'position_size': 0,
                'kelly_size': 0,
                'win_rate': self.position_sizer.get_win_rate(ticker),
                'can_trade': False
            }

        # 6. Build decision
        decision = {
            # Action
            'action': signal_direction if can_trade and shares > 0 else 'HOLD',
            'can_trade': can_trade,
            'shares': shares,
            'position_value': position_value,

            # Position sizing (Phase 2 quarter-Kelly)
            'position_pct': sizing_details['position_size'],
            'kelly_size': sizing_details.get('kelly_size', 0),
            'win_rate': sizing_details.get('win_rate', 0.5),

            # Drawdown state (Phase 2 thresholds)
            'drawdown': drawdown_state['drawdown'],
            'drawdown_state': drawdown_state['state'],
            'drawdown_multiplier': drawdown_state['position_multiplier'],

            # Regime state (Combined GMM + transition)
            'regime': regime_state['regime'],
            'regime_name': regime_state['regime_name'],
            'regime_multiplier': regime_state['position_multiplier'],
            'transition_warning': regime_state['transition_warning'],
            'regime_stability': regime_state['stability_score'],

            # Combined
            'combined_multiplier': combined_multiplier,
            'signal_confidence': signal_confidence,
            'current_volatility': current_volatility,

            # Reasons if not trading
            'hold_reasons': self._get_hold_reasons(
                drawdown_state, regime_state, signal_direction, shares
            )
        }

        # Log decision
        if can_trade and shares > 0:
            logger.info(
                f"{ticker}: {decision['action']} {shares} shares "
                f"(${position_value:.0f}, {sizing_details['position_size']:.1%})"
            )
        elif signal_direction != 'HOLD':
            logger.info(
                f"{ticker}: HOLD - {', '.join(decision['hold_reasons'])}"
            )

        return decision

    def _get_hold_reasons(
        self,
        drawdown_state: Dict,
        regime_state: Dict,
        signal_direction: str,
        shares: int
    ) -> List[str]:
        """Get list of reasons for not trading."""
        reasons = []

        if signal_direction == 'HOLD':
            reasons.append("No signal")

        if not drawdown_state['can_trade']:
            reasons.append(f"Drawdown limit ({drawdown_state['drawdown']:.1%})")

        if drawdown_state['circuit_breaker']:
            reasons.append("Circuit breaker triggered")

        if not regime_state['should_trade']:
            reasons.append(f"Crisis regime ({regime_state['regime_name']})")

        if regime_state['transition_warning']:
            reasons.append("Regime transition warning")

        if shares == 0 and signal_direction != 'HOLD':
            reasons.append("Position too small")

        return reasons if reasons else ["None"]

    def get_system_state(self) -> Dict:
        """Get current state of all components."""
        return {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': self.drawdown_manager.current_drawdown,
            'drawdown_state': self.drawdown_manager._get_drawdown_state(),
            'circuit_breaker': self.drawdown_manager.circuit_breaker_triggered,
            'overall_win_rate': self.position_sizer.get_win_rate(),
            'kelly_fraction': self.position_sizer.kelly_fraction,
            'config': self.config
        }

    def reset(self):
        """Reset all components."""
        self.drawdown_manager.reset()
        self.position_sizer.reset()
        self.regime_detector.reset()
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        logger.info("Conflict resolver reset")


# Convenience function
def get_conflict_resolver(**kwargs) -> TradingSystemConflictResolver:
    """Get a configured TradingSystemConflictResolver instance."""
    return TradingSystemConflictResolver(**kwargs)


# Quick integration with existing Phase 2 system
def integrate_with_phase2_system(phase2_system, conflict_resolver: TradingSystemConflictResolver):
    """
    Integrate conflict resolver with existing Phase2ImprovementSystem.

    Args:
        phase2_system: Existing Phase2ImprovementSystem instance
        conflict_resolver: Configured TradingSystemConflictResolver

    Returns:
        Modified phase2_system with resolved conflicts
    """
    # Override drawdown protection with unified manager
    if hasattr(phase2_system, 'drawdown_protection'):
        phase2_system.drawdown_protection = conflict_resolver.drawdown_manager

    # Override position sizer with resolved version
    if hasattr(phase2_system, 'position_sizer'):
        phase2_system.position_sizer = conflict_resolver.position_sizer

    # Override regime detector with unified version
    if hasattr(phase2_system, 'regime_detector'):
        phase2_system.regime_detector = conflict_resolver.regime_detector

    logger.info("Integrated conflict resolver with Phase2ImprovementSystem")

    return phase2_system
