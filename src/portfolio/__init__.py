"""
Phase 6 Portfolio Optimization Module

This module contains all Phase 6 components for portfolio-level optimization:
- Cross-Portfolio Risk Budgeting (+0.5-1% profit rate)
- Liquidity-Aware Rebalancing (+0.5-1% profit rate)
- Tax-Efficient Optimization (+1-2% net returns)
- Multi-Objective Optimization (+0.5-1% profit rate)
- Regulatory Compliance (Risk reduction)

Expected Total Phase 6 Impact: +2-4% profit rate
Expected Total After Phase 6: +21-36% from baseline
"""

# Risk Budgeting
from .risk_budgeting import (
    # Enums
    AllocationMethod,
    CorrelationRegime,

    # Data Classes
    RiskBudgetConfig,
    RiskAllocation,
    PositionRiskMetrics,

    # Core Classes
    MarginalRiskCalculator,
    CorrelationAwareAllocator,
    RiskBudgetAllocator,

    # Factory
    create_risk_budget_allocator,
)

# Liquidity-Aware Rebalancing
from .liquidity_rebalancer import (
    # Enums
    TriggerType,
    ExecutionStrategy,
    UrgencyLevel,

    # Data Classes
    RebalanceConfig,
    TradeOrder,
    ExecutionResult,
    RebalanceDecision,

    # Core Classes
    SlippageTracker,
    ExecutionOptimizer,
    RebalanceTrigger,
    LiquidityAwareRebalancer,

    # Factory
    create_liquidity_aware_rebalancer,
)

# Tax Optimization
from .tax_optimizer import (
    # Enums
    LotSelectionMethod,
    HarvestAction,

    # Data Classes
    TaxConfig,
    TaxLot,
    HarvestOpportunity,
    TaxImpact,

    # Core Classes
    TaxLotManager,
    TaxLossHarvester,
    CapitalGainsOptimizer,
    HoldPeriodOptimizer,
    TaxOptimizer,

    # Factory
    create_tax_optimizer,
)

# Multi-Objective Optimization
from .multi_objective_optimizer import (
    # Enums
    OptimizationMethod,
    ConstraintType,

    # Data Classes
    OptimizationConfig,
    Constraint,
    OptimizationObjective,
    OptimizationResult,

    # Core Classes
    ConstraintManager,
    UtilityFunction,
    ObjectiveWeightOptimizer,
    MultiObjectiveOptimizer,

    # Factory
    create_multi_objective_optimizer,
)

# Compliance Management
from .compliance_manager import (
    # Enums
    ComplianceStatus,
    AlertSeverity,

    # Data Classes
    ComplianceConfig,
    ComplianceAlert,
    TradeComplianceResult,
    AuditEntry,

    # Core Classes
    PositionLimitManager,
    ConcentrationMonitor,
    ComplianceChecker,
    AuditTrailManager,
    ComplianceManager,

    # Factory
    create_compliance_manager,
)

# Phase 6 Integration
from .phase6_integration import (
    # Enums
    PortfolioAction,

    # Data Classes
    Phase6Config,
    PortfolioState,
    Phase6Decision,
    Phase6Status,

    # Main System
    Phase6PortfolioOptimizationSystem,

    # Integration
    integrate_phase6_with_phase5,

    # Factory
    create_phase6_system,

    # Checklist
    PHASE6_PRODUCTION_CHECKLIST,
)

# Phase 6 Improvements (Enhanced Calculations)
from .phase6_improvements import (
    # Enums
    MarketRegime,
    ExecutionStrategy as ImprovedExecutionStrategy,
    UrgencyLevel as ImprovedUrgencyLevel,
    HarvestAction as ImprovedHarvestAction,

    # Configurations
    RegimeConfig,
    TaxConfig as ImprovedTaxConfig,
    HarvestOpportunity as ImprovedHarvestOpportunity,

    # Core Improvement Classes
    RegimeAwareRiskBudgeter,
    ExpectedShortfallCalculator,
    AdaptiveExecutionSelector,
    LiquidityAdjustedRisk,
    TaxAwareRebalancer,
    TaxLossHarvestOptimizer,
    RegimeAwareUtility,
    RobustCovarianceEstimator,
    Phase5SignalIntegrator,
    OptimizationDiagnostics,

    # Main System
    Phase6ImprovementsSystem,

    # Factory
    create_phase6_improvements_system,

    # Validation
    validate_improvements_module,
)

# Phase 6 Final Improvements (Production-Grade Enhancements)
from .phase6_final_improvements import (
    # Core Classes
    PortfolioOptimizationMonitor,
    MultiTimeframeOptimizer,
    AdaptiveConstraintManager,
    CrossPortfolioCorrelationManager,
    PortfolioStressTester,

    # Main System
    Phase6FinalImprovementsSystem,

    # Factories
    create_phase6_final_system,
    create_portfolio_monitor,
    create_stress_tester,

    # Checklist
    PHASE6_FINAL_PRODUCTION_CHECKLIST,

    # Validation
    validate_final_improvements,
)


__all__ = [
    # Risk Budgeting
    'AllocationMethod',
    'CorrelationRegime',
    'RiskBudgetConfig',
    'RiskAllocation',
    'PositionRiskMetrics',
    'MarginalRiskCalculator',
    'CorrelationAwareAllocator',
    'RiskBudgetAllocator',
    'create_risk_budget_allocator',

    # Liquidity-Aware Rebalancing
    'TriggerType',
    'ExecutionStrategy',
    'UrgencyLevel',
    'RebalanceConfig',
    'TradeOrder',
    'ExecutionResult',
    'RebalanceDecision',
    'SlippageTracker',
    'ExecutionOptimizer',
    'RebalanceTrigger',
    'LiquidityAwareRebalancer',
    'create_liquidity_aware_rebalancer',

    # Tax Optimization
    'LotSelectionMethod',
    'HarvestAction',
    'TaxConfig',
    'TaxLot',
    'HarvestOpportunity',
    'TaxImpact',
    'TaxLotManager',
    'TaxLossHarvester',
    'CapitalGainsOptimizer',
    'HoldPeriodOptimizer',
    'TaxOptimizer',
    'create_tax_optimizer',

    # Multi-Objective Optimization
    'OptimizationMethod',
    'ConstraintType',
    'OptimizationConfig',
    'Constraint',
    'OptimizationObjective',
    'OptimizationResult',
    'ConstraintManager',
    'UtilityFunction',
    'ObjectiveWeightOptimizer',
    'MultiObjectiveOptimizer',
    'create_multi_objective_optimizer',

    # Compliance Management
    'ComplianceStatus',
    'AlertSeverity',
    'ComplianceConfig',
    'ComplianceAlert',
    'TradeComplianceResult',
    'AuditEntry',
    'PositionLimitManager',
    'ConcentrationMonitor',
    'ComplianceChecker',
    'AuditTrailManager',
    'ComplianceManager',
    'create_compliance_manager',

    # Phase 6 Integration
    'PortfolioAction',
    'Phase6Config',
    'PortfolioState',
    'Phase6Decision',
    'Phase6Status',
    'Phase6PortfolioOptimizationSystem',
    'integrate_phase6_with_phase5',
    'create_phase6_system',
    'PHASE6_PRODUCTION_CHECKLIST',

    # Phase 6 Improvements
    'MarketRegime',
    'ImprovedExecutionStrategy',
    'ImprovedUrgencyLevel',
    'ImprovedHarvestAction',
    'RegimeConfig',
    'ImprovedTaxConfig',
    'ImprovedHarvestOpportunity',
    'RegimeAwareRiskBudgeter',
    'ExpectedShortfallCalculator',
    'AdaptiveExecutionSelector',
    'LiquidityAdjustedRisk',
    'TaxAwareRebalancer',
    'TaxLossHarvestOptimizer',
    'RegimeAwareUtility',
    'RobustCovarianceEstimator',
    'Phase5SignalIntegrator',
    'OptimizationDiagnostics',
    'Phase6ImprovementsSystem',
    'create_phase6_improvements_system',
    'validate_improvements_module',

    # Phase 6 Final Improvements
    'PortfolioOptimizationMonitor',
    'MultiTimeframeOptimizer',
    'AdaptiveConstraintManager',
    'CrossPortfolioCorrelationManager',
    'PortfolioStressTester',
    'Phase6FinalImprovementsSystem',
    'create_phase6_final_system',
    'create_portfolio_monitor',
    'create_stress_tester',
    'PHASE6_FINAL_PRODUCTION_CHECKLIST',
    'validate_final_improvements',
]

__version__ = '6.2.0'
__phase__ = 'Phase 6 - Portfolio Optimization with Final Improvements'
