"""
Risk Management Module - Phase 2 Conflict Resolution

This module contains unified risk management components that resolve
conflicts between Phase 1 and Phase 2 implementations.

Key resolutions:
1. Drawdown thresholds: Phase 2 (5%/10%/15%) takes precedence over Phase 1 (8%/20%)
2. Position sizing: Quarter-Kelly (0.25) takes precedence over confidence-based
3. Regime detection: Combined GMM (Phase 1) + Transition (Phase 2)

Usage:
    from src.risk import TradingSystemConflictResolver

    # Initialize resolver
    resolver = TradingSystemConflictResolver(
        warning_threshold=0.05,   # Phase 2 threshold
        danger_threshold=0.10,    # Phase 2 threshold
        max_drawdown=0.15,        # Phase 2 threshold
        kelly_fraction=0.25       # Quarter-Kelly
    )

    # Fit regime detector on historical data
    resolver.fit_regime_detector(historical_data)

    # Get trading decision
    decision = resolver.get_trading_decision(
        ticker='AAPL',
        signal_confidence=0.75,
        signal_direction='LONG',
        current_volatility=0.02,
        current_price=150.0
    )

    # Record trade outcome for Kelly calculation
    resolver.record_trade_outcome(
        ticker='AAPL',
        was_profitable=True,
        profit_pct=0.03
    )
"""

from .unified_drawdown_manager import (
    UnifiedDrawdownManager,
    get_unified_drawdown_manager
)

from .resolved_position_sizer import (
    ResolvedPositionSizer,
    get_resolved_position_sizer
)

from .unified_regime_detector import (
    UnifiedRegimeDetector,
    get_unified_regime_detector
)

from .conflict_resolver import (
    TradingSystemConflictResolver,
    get_conflict_resolver,
    integrate_with_phase2_system
)

# Phase 4 Macro Integration
from .phase4_macro_resolver import (
    Phase4MacroResolver,
    get_phase4_resolver,
    integrate_phase4_with_trading_system
)

__all__ = [
    # Drawdown management
    'UnifiedDrawdownManager',
    'get_unified_drawdown_manager',

    # Position sizing
    'ResolvedPositionSizer',
    'get_resolved_position_sizer',

    # Regime detection
    'UnifiedRegimeDetector',
    'get_unified_regime_detector',

    # Master resolver
    'TradingSystemConflictResolver',
    'get_conflict_resolver',
    'integrate_with_phase2_system',

    # Phase 4 Macro Integration
    'Phase4MacroResolver',
    'get_phase4_resolver',
    'integrate_phase4_with_trading_system'
]

__version__ = '4.0.0'
__author__ = 'Phase 4 Macro Integration'
