"""
Phase 4 Macro-Enhanced Conflict Resolver

Extends TradingSystemConflictResolver with Phase 4 macro integration:
- Macro context awareness (VIX, GLD, SPY, DXY)
- Cross-market correlation regime
- Risk-on/Risk-off position adjustment
- Correlation breakdown protection

Expected Impact: +5-8% profit rate improvement

Based on: phase future roadmap.pdf
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging

from .conflict_resolver import TradingSystemConflictResolver

logger = logging.getLogger(__name__)


class Phase4MacroResolver(TradingSystemConflictResolver):
    """
    Phase 4 enhanced conflict resolver with macro integration.

    Extends the base TradingSystemConflictResolver with:
    - Macro regime awareness (VIX, GLD, SPY, DXY)
    - Risk-on/Risk-off position adjustment
    - Correlation breakdown protection
    - Safe haven flow detection

    All Phase 2/3 functionality is preserved.
    """

    def __init__(
        self,
        # Inherit Phase 2 parameters
        warning_threshold: float = 0.05,
        danger_threshold: float = 0.10,
        max_drawdown: float = 0.15,
        kelly_fraction: float = 0.25,
        max_position: float = 0.30,
        n_regimes: int = 4,
        initial_capital: float = 100000.0,
        # Phase 4 macro parameters
        vix_warning_level: float = 25.0,
        vix_crisis_level: float = 35.0,
        risk_off_reduction: float = 0.5,
        correlation_breakdown_reduction: float = 0.3
    ):
        """
        Initialize Phase 4 macro-enhanced resolver.

        Inherits all Phase 2 parameters plus:
        Args:
            vix_warning_level: VIX level triggering caution (default 25)
            vix_crisis_level: VIX level triggering crisis mode (default 35)
            risk_off_reduction: Position reduction in risk-off regime (0.5 = 50%)
            correlation_breakdown_reduction: Reduction on correlation breakdown (0.3 = 30%)
        """
        # Initialize parent class (Phase 2/3)
        super().__init__(
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            max_drawdown=max_drawdown,
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            n_regimes=n_regimes,
            initial_capital=initial_capital
        )

        # Phase 4 macro parameters
        self.vix_warning_level = vix_warning_level
        self.vix_crisis_level = vix_crisis_level
        self.risk_off_reduction = risk_off_reduction
        self.correlation_breakdown_reduction = correlation_breakdown_reduction

        # Current macro state
        self.macro_state = {
            'vix_level': None,
            'vix_regime': 'normal',
            'risk_regime': 'neutral',
            'risk_score': 0.0,
            'correlation_breakdown': False,
            'macro_multiplier': 1.0,
            'gld_strength': 'neutral',
            'spy_trend': 'neutral'
        }

        # Add Phase 4 to config
        self.config.update({
            'vix_warning_level': vix_warning_level,
            'vix_crisis_level': vix_crisis_level,
            'risk_off_reduction': risk_off_reduction,
            'correlation_breakdown_reduction': correlation_breakdown_reduction,
            'phase': 4
        })

        logger.info("Initialized Phase4MacroResolver")
        logger.info(f"  VIX thresholds: {vix_warning_level}/{vix_crisis_level}")
        logger.info(f"  Risk-off reduction: {risk_off_reduction:.0%}")

    def update_macro_context(
        self,
        vix_level: Optional[float] = None,
        risk_regime: str = 'neutral',
        risk_score: float = 0.0,
        correlation_breakdown: bool = False,
        gld_momentum: float = 0.0,
        spy_momentum: float = 0.0,
        regime_position_mult: float = 1.0
    ):
        """
        Update macro context from feature data.

        This should be called with the latest macro features before
        making trading decisions.

        Args:
            vix_level: Current VIX level
            risk_regime: Risk regime ('risk_on', 'neutral', 'risk_off', etc.)
            risk_score: Risk score (-1 to +1)
            correlation_breakdown: Whether correlation breakdown detected
            gld_momentum: GLD 20-day momentum
            spy_momentum: SPY 20-day momentum
            regime_position_mult: Position multiplier from regime (0.3 to 1.2)
        """
        # VIX regime
        if vix_level is not None:
            self.macro_state['vix_level'] = vix_level
            if vix_level >= self.vix_crisis_level:
                self.macro_state['vix_regime'] = 'crisis'
            elif vix_level >= self.vix_warning_level:
                self.macro_state['vix_regime'] = 'elevated'
            else:
                self.macro_state['vix_regime'] = 'normal'

        # Risk regime
        self.macro_state['risk_regime'] = risk_regime
        self.macro_state['risk_score'] = risk_score
        self.macro_state['correlation_breakdown'] = correlation_breakdown

        # Trend indicators
        self.macro_state['gld_strength'] = 'bullish' if gld_momentum > 0 else 'bearish'
        self.macro_state['spy_trend'] = 'up' if spy_momentum > 0 else 'down'

        # Calculate combined macro multiplier
        self.macro_state['macro_multiplier'] = self._calculate_macro_multiplier(
            regime_position_mult
        )

        logger.debug(f"Macro context updated: {self.macro_state}")

    def _calculate_macro_multiplier(self, regime_position_mult: float = 1.0) -> float:
        """
        Calculate combined macro position multiplier.

        Combines:
        - VIX regime impact
        - Risk-on/risk-off impact
        - Correlation breakdown impact

        Returns:
            Combined multiplier (0.0 to 1.2)
        """
        multiplier = 1.0

        # VIX impact
        if self.macro_state['vix_regime'] == 'crisis':
            multiplier *= 0.3  # 70% reduction in crisis
        elif self.macro_state['vix_regime'] == 'elevated':
            multiplier *= 0.7  # 30% reduction when elevated

        # Risk regime impact
        if 'risk_off' in self.macro_state['risk_regime']:
            multiplier *= self.risk_off_reduction
        elif 'risk_on' in self.macro_state['risk_regime']:
            multiplier *= 1.1  # Slight increase in risk-on

        # Correlation breakdown protection
        if self.macro_state['correlation_breakdown']:
            multiplier *= (1 - self.correlation_breakdown_reduction)

        # Apply regime-based multiplier
        multiplier *= regime_position_mult

        # Clamp to valid range
        multiplier = max(0.0, min(1.2, multiplier))

        return multiplier

    def get_trading_decision(
        self,
        ticker: str,
        signal_confidence: float,
        signal_direction: str,
        current_volatility: float,
        current_price: float,
        daily_return: float = 0.0,
        macro_context: Optional[Dict] = None
    ) -> Dict:
        """
        Get trading decision with Phase 4 macro integration.

        Extends parent method with macro context awareness.

        Args:
            ticker: Stock ticker
            signal_confidence: Model confidence (0-1)
            signal_direction: 'LONG', 'SHORT', or 'HOLD'
            current_volatility: Current volatility
            current_price: Current price
            daily_return: Today's return
            macro_context: Optional dict with macro features (from Phase4MacroIntegration)

        Returns:
            Dict with complete trading decision including macro context
        """
        # Update macro context if provided
        if macro_context is not None:
            self.update_macro_context(
                vix_level=macro_context.get('vix_level'),
                risk_regime=macro_context.get('risk_regime', 'neutral'),
                risk_score=macro_context.get('risk_score', 0.0),
                correlation_breakdown=macro_context.get('correlation_breakdown', False),
                gld_momentum=macro_context.get('gld_momentum', 0.0),
                spy_momentum=macro_context.get('spy_momentum', 0.0),
                regime_position_mult=macro_context.get('position_multiplier', 1.0)
            )

        # Get base decision from parent (Phase 2/3)
        decision = super().get_trading_decision(
            ticker=ticker,
            signal_confidence=signal_confidence,
            signal_direction=signal_direction,
            current_volatility=current_volatility,
            current_price=current_price,
            daily_return=daily_return
        )

        # Apply Phase 4 macro multiplier
        macro_mult = self.macro_state['macro_multiplier']

        # Adjust position size with macro context
        if decision['can_trade'] and decision['shares'] > 0:
            # Apply macro multiplier
            adjusted_value = decision['position_value'] * macro_mult
            adjusted_shares = int(adjusted_value / current_price) if current_price > 0 else 0

            decision['position_value_pre_macro'] = decision['position_value']
            decision['position_value'] = adjusted_value
            decision['shares_pre_macro'] = decision['shares']
            decision['shares'] = adjusted_shares

            # Update action if shares reduced to 0
            if adjusted_shares == 0 and decision['action'] != 'HOLD':
                decision['action'] = 'HOLD'
                decision['hold_reasons'].append(f"Macro context reduction ({macro_mult:.1%})")

        # Add Phase 4 macro context to decision
        decision['macro_context'] = {
            'vix_level': self.macro_state['vix_level'],
            'vix_regime': self.macro_state['vix_regime'],
            'risk_regime': self.macro_state['risk_regime'],
            'risk_score': self.macro_state['risk_score'],
            'correlation_breakdown': self.macro_state['correlation_breakdown'],
            'macro_multiplier': macro_mult,
            'gld_strength': self.macro_state['gld_strength'],
            'spy_trend': self.macro_state['spy_trend']
        }

        # Update combined multiplier to include macro
        decision['combined_multiplier'] = (
            decision['drawdown_multiplier'] *
            decision['regime_multiplier'] *
            macro_mult
        )

        # Add macro-specific hold reasons if applicable
        if self.macro_state['vix_regime'] == 'crisis':
            decision['hold_reasons'].append("VIX crisis mode")
        if self.macro_state['correlation_breakdown']:
            decision['hold_reasons'].append("Correlation breakdown detected")
        if 'strong_risk_off' in self.macro_state['risk_regime']:
            decision['hold_reasons'].append("Strong risk-off environment")

        # Log decision with macro context
        if decision['can_trade'] and decision['shares'] > 0:
            logger.info(
                f"{ticker}: {decision['action']} {decision['shares']} shares "
                f"(macro_mult={macro_mult:.1%}, risk={self.macro_state['risk_regime']})"
            )

        return decision

    def get_system_state(self) -> Dict:
        """Get current state including Phase 4 macro context."""
        state = super().get_system_state()

        # Add Phase 4 macro state
        state['macro_context'] = self.macro_state.copy()
        state['phase'] = 4

        return state

    def should_trade_in_current_macro(self) -> Tuple[bool, str]:
        """
        Check if trading is advisable in current macro environment.

        Returns:
            Tuple of (should_trade, reason)
        """
        # VIX crisis = no trading
        if self.macro_state['vix_regime'] == 'crisis':
            return False, f"VIX crisis ({self.macro_state['vix_level']:.1f})"

        # Strong risk-off = very limited trading
        if self.macro_state['risk_regime'] == 'strong_risk_off':
            return False, "Strong risk-off environment"

        # Correlation breakdown = proceed with caution
        if self.macro_state['correlation_breakdown']:
            return True, "Correlation breakdown - reduced positions"

        # Normal conditions
        return True, "Normal macro environment"


# Convenience function
def get_phase4_resolver(**kwargs) -> Phase4MacroResolver:
    """Get a configured Phase4MacroResolver instance."""
    return Phase4MacroResolver(**kwargs)


def integrate_phase4_with_trading_system(trading_system, macro_features_df: pd.DataFrame):
    """
    Integrate Phase 4 macro context with existing trading system.

    Args:
        trading_system: Existing trading system with conflict resolver
        macro_features_df: DataFrame with Phase 4 macro features

    Returns:
        Updated trading system with Phase 4 integration
    """
    if not hasattr(trading_system, 'conflict_resolver'):
        logger.warning("Trading system has no conflict_resolver attribute")
        return trading_system

    # Get latest macro context
    if len(macro_features_df) == 0:
        return trading_system

    latest = macro_features_df.iloc[-1]

    macro_context = {
        'vix_level': latest.get('VIX', None),
        'risk_regime': latest.get('risk_regime', 'neutral'),
        'risk_score': latest.get('risk_regime_score', 0.0),
        'correlation_breakdown': bool(latest.get('corr_breakdown', 0)),
        'gld_momentum': latest.get('GLD_momentum_20d', 0.0),
        'spy_momentum': latest.get('SPY_momentum_20d', 0.0),
        'position_multiplier': latest.get('regime_position_mult', 1.0)
    }

    # Update resolver with macro context
    if isinstance(trading_system.conflict_resolver, Phase4MacroResolver):
        trading_system.conflict_resolver.update_macro_context(**macro_context)
    else:
        logger.info("Upgrading conflict resolver to Phase4MacroResolver")
        # Create new Phase 4 resolver with existing config
        old_config = trading_system.conflict_resolver.config
        trading_system.conflict_resolver = Phase4MacroResolver(
            warning_threshold=old_config.get('warning_threshold', 0.05),
            danger_threshold=old_config.get('danger_threshold', 0.10),
            max_drawdown=old_config.get('max_drawdown', 0.15),
            kelly_fraction=old_config.get('kelly_fraction', 0.25),
            max_position=old_config.get('max_position', 0.30)
        )
        trading_system.conflict_resolver.update_macro_context(**macro_context)

    return trading_system
