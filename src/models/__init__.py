"""ML models for volatility prediction."""

from .base_models import VolatilityPredictor
from .ensemble_model import EnsemblePredictor
from .regime_detector import RegimeDetector, RegimeSwitchingModel

# Phase 2 Asset Class Ensembles
try:
    from .asset_class_ensembles import (
        AssetClassEnsembleFactory,
        EquitySpecificEnsemble,
        ForexSpecificEnsemble,
        CryptoSpecificEnsemble,
        CommoditySpecificEnsemble,
        InternationalEnsemble,
        BondSpecificEnsemble,
        ETFSpecificEnsemble,
    )
except ImportError:
    AssetClassEnsembleFactory = None

# Phase 2 Meta Ensemble
try:
    from .meta_ensemble import (
        MetaEnsembleCombiner,
        EnhancedTradingSystem,
        get_meta_ensemble,
        get_enhanced_system,
    )
except ImportError:
    MetaEnsembleCombiner = None

# US/Intl Model Optimizer (Fixes 1-15)
try:
    from .us_intl_optimizer import (
        USIntlModelOptimizer,
        create_optimizer,
        SignalOptimization,
        AssetClass,
    )
except ImportError:
    USIntlModelOptimizer = None
    create_optimizer = None

# Hybrid Ensemble with integrated optimizer
try:
    from .hybrid_ensemble import HybridEnsemblePredictor
except ImportError:
    HybridEnsemblePredictor = None

__all__ = [
    'VolatilityPredictor',
    'EnsemblePredictor',
    'RegimeDetector',
    'RegimeSwitchingModel',
    'AssetClassEnsembleFactory',
    'EquitySpecificEnsemble',
    'ForexSpecificEnsemble',
    'CryptoSpecificEnsemble',
    'CommoditySpecificEnsemble',
    'InternationalEnsemble',
    'BondSpecificEnsemble',
    'ETFSpecificEnsemble',
    'MetaEnsembleCombiner',
    'EnhancedTradingSystem',
    'get_meta_ensemble',
    'get_enhanced_system',
    # US/Intl Optimizer
    'USIntlModelOptimizer',
    'create_optimizer',
    'SignalOptimization',
    'AssetClass',
    'HybridEnsemblePredictor',
]
