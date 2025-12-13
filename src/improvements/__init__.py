"""
Phase 2 Improvements Module

This module contains all 15 improvements from the Phase 2 enhancement plan:

1. Dynamic Ensemble Weighting - Adapt weights based on recent performance
2. Regime-Aware Feature Selection - Select features based on market regime
3. Advanced Cross-Asset Correlations - Dynamic correlation networks
4. Multi-Timeframe Ensemble - Combine signals from multiple timeframes
5. Real-Time Feature Engineering - Incremental feature calculation
6. Confidence-Calibrated Position Sizing - Kelly-based sizing with confidence
7. Regime Transition Detection - Early warning for regime changes
8. Feature Importance Over Time - Track how importance changes
9. Bayesian Signal Combination - Bayesian model averaging
10. Dynamic Drawdown Protection - Adaptive position reduction during drawdown
11. Information-Theoretic Model Selection - Select based on prediction entropy
12. Adaptive Feature Thresholds - Dynamic decision thresholds
13. Cross-Market Signal Validation - Validate against correlated markets
14. Profit-Maximizing Loss Functions - Custom loss functions
15. Walk-Forward Validation Framework - Proper backtesting methodology

Usage:
    from src.improvements import Phase2ImprovementSystem, apply_all_improvements

    # Option 1: Use the full system
    system = Phase2ImprovementSystem()
    system.initialize()
    signal = system.generate_enhanced_signal(ticker, data, portfolio, market_conditions)
    position_size = system.get_position_size(signal, portfolio)

    # Option 2: Quick apply
    result = apply_all_improvements(ticker, data, portfolio, market_conditions)

Expected Impact:
    - +3-5% improvement in profit rate (to ~65-70%)
    - -10-20% reduction in maximum drawdown
    - +15-25% improvement in risk-adjusted returns (Sharpe ratio)
"""

from .phase2_fifteen_improvements import (
    # Core improvement classes
    DynamicEnsembleWeighter,
    RegimeAwareFeatureSelector,
    AdvancedCorrelationAnalyzer,
    MultiTimeframeEnsemble,
    StreamingFeatureEngine,
    ConfidenceAwarePositionSizer,
    RegimeTransitionDetector,
    TimeVaryingFeatureImportance,
    BayesianSignalCombiner,
    AdaptiveDrawdownProtection,

    # Additional improvements
    InformationTheoreticModelSelector,
    AdaptiveFeatureThresholds,
    CrossMarketSignalValidator,
    ProfitMaximizingLoss,
    WalkForwardValidator,

    # Master integration
    Phase2ImprovementSystem,

    # Convenience functions
    get_phase2_system,
    apply_all_improvements
)

__all__ = [
    # Core classes (1-10)
    'DynamicEnsembleWeighter',
    'RegimeAwareFeatureSelector',
    'AdvancedCorrelationAnalyzer',
    'MultiTimeframeEnsemble',
    'StreamingFeatureEngine',
    'ConfidenceAwarePositionSizer',
    'RegimeTransitionDetector',
    'TimeVaryingFeatureImportance',
    'BayesianSignalCombiner',
    'AdaptiveDrawdownProtection',

    # Additional classes (11-15)
    'InformationTheoreticModelSelector',
    'AdaptiveFeatureThresholds',
    'CrossMarketSignalValidator',
    'ProfitMaximizingLoss',
    'WalkForwardValidator',

    # Master integration
    'Phase2ImprovementSystem',

    # Convenience functions
    'get_phase2_system',
    'apply_all_improvements'
]

__version__ = '2.0.0'
__author__ = 'Phase 2 Improvement System'
