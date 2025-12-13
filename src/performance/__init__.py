"""
Performance Module for Stock Prediction Model
==============================================

This module provides comprehensive performance tracking for the trading system.
It integrates with the existing risk management framework (Fixes 16-19).

Components:
    - TradeResult: Enum for trade outcomes (WIN, LOSS, BREAKEVEN, PENDING)
    - TradeRecord: Data class for individual trade records
    - PerformanceTracker: Main class for tracking and analyzing trade performance
    - EnhancedPositionAdjuster: Position adjuster with performance-based sizing
    - RiskManagerWithPerformance: Complete risk manager with performance tracking

Key Features:
    - update_performance(): Records trade outcomes with full metrics
    - performance_history: Maintains historical performance data by period
    - Regime-based performance analysis
    - Asset-specific tracking
    - Adaptive position multiplier calculation
    - Data persistence (JSON and pickle formats)

Author: Claude Code
Last Updated: 2025-12-03
"""

from .performance_tracker import (
    TradeResult,
    TradeRecord,
    PerformanceTracker,
    create_performance_tracker
)

from .integration import (
    EnhancedPositionAdjuster,
    RiskManagerWithPerformance,
    create_risk_manager_with_performance
)

__all__ = [
    # Core classes
    'TradeResult',
    'TradeRecord',
    'PerformanceTracker',

    # Integration classes
    'EnhancedPositionAdjuster',
    'RiskManagerWithPerformance',

    # Factory functions
    'create_performance_tracker',
    'create_risk_manager_with_performance'
]
