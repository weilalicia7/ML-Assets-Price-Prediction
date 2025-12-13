"""Trading modules for daily execution."""

from .risk_manager import RiskManager, TradingSignalGenerator

# Phase 1 Integration
try:
    from .phase1_integration import (
        Phase1TradingSystem,
        Phase1APIEndpoints,
        get_phase1_system,
        get_phase1_api,
    )
except ImportError:
    Phase1TradingSystem = None

# Phase 2 + 3 Unified Integration
try:
    from .phase2_phase3_integration import (
        UnifiedTradingSystem,
        UnifiedAPIEndpoints,
        get_unified_system,
        get_unified_api,
    )
except ImportError:
    UnifiedTradingSystem = None

__all__ = [
    'RiskManager',
    'TradingSignalGenerator',
    'Phase1TradingSystem',
    'Phase1APIEndpoints',
    'get_phase1_system',
    'get_phase1_api',
    'UnifiedTradingSystem',
    'UnifiedAPIEndpoints',
    'get_unified_system',
    'get_unified_api',
]
