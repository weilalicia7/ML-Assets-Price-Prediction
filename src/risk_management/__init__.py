"""
Risk Management Module for US/INTL Model
=========================================

This module implements Fixes 16-19 from the model fixing documentation:
    - Fix 16: China stock blocking (routing to China model)
    - Fix 17: Adaptive blocking for JPY SELL signals
    - Fix 18: Adaptive blocking for Crude Oil CL=F
    - Fix 19: Dynamic position adjustment based on market regime

Components:
    - MarketRegimeDetector: Detects current market regime (Fix 19)
    - AdaptiveBlocker: Blocks/reduces signals based on regime (Fixes 17 & 18)
    - PositionAdjuster: Adjusts position sizes based on regime (Fix 19)

Last Updated: 2025-12-03
"""

from .market_regime_detector import MarketRegimeDetector, MarketRegime
from .adaptive_blocker import AdaptiveBlocker, BlockingLevel, BlockingResult
from .position_adjuster import PositionAdjuster, PositionAdjustment

__all__ = [
    'MarketRegimeDetector',
    'MarketRegime',
    'AdaptiveBlocker',
    'BlockingLevel',
    'BlockingResult',
    'PositionAdjuster',
    'PositionAdjustment',
]
