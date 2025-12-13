"""Feature engineering modules."""

from .technical_features import TechnicalFeatureEngineer
from .volatility_features import VolatilityFeatureEngineer

# Phase 3 specialized features
try:
    from .international_features import InternationalFeatures
except ImportError:
    InternationalFeatures = None

try:
    from .crypto_features import CryptoFeatures
except ImportError:
    CryptoFeatures = None

try:
    from .commodity_features import CommodityFeatures
except ImportError:
    CommodityFeatures = None

__all__ = [
    'TechnicalFeatureEngineer',
    'VolatilityFeatureEngineer',
    'InternationalFeatures',
    'CryptoFeatures',
    'CommodityFeatures',
]
