"""
Utility modules for stock prediction model.

Includes:
- enhanced_signal_validator: Enhanced signal validation for China model
  - Prevents catastrophic shorts by filtering dangerous patterns
  - Smart position sizing with volatility-based controls
  - Enhanced regime detection
"""

from .enhanced_signal_validator import (
    EnhancedSignalValidator,
    SmartPositionSizer,
    EnhancedRegimeDetector,
    EnhancedPhaseSystem,
    PerformanceTracker,
    validate_china_signal,
)

__all__ = [
    'EnhancedSignalValidator',
    'SmartPositionSizer',
    'EnhancedRegimeDetector',
    'EnhancedPhaseSystem',
    'PerformanceTracker',
    'validate_china_signal',
]
