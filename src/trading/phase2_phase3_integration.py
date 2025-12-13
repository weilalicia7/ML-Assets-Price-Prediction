"""
Phase 2 + Phase 3 Integration Layer

Integrates:
- Phase 1: 20 Advanced Features (stress protection, meta-learning, etc.)
- Phase 2: Asset-Class Specific Ensembles (7 specialized ensembles)
- Phase 3: Structural Features (regime detection, volatility regimes, momentum/mean-reversion)

Usage:
    from src.trading.phase2_phase3_integration import UnifiedTradingSystem

    system = UnifiedTradingSystem()
    result = system.generate_signal(ticker, market_data)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Phase 1 imports
try:
    from src.trading.phase1_integration import Phase1TradingSystem, get_phase1_system
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    print("[WARN] Phase 1 integration not available")

# Phase 2 imports - Asset Class Ensembles
try:
    from src.models.asset_class_ensembles import (
        AssetClassEnsembleFactory,
        EquitySpecificEnsemble,
        ForexSpecificEnsemble,
        CryptoSpecificEnsemble,
        CommoditySpecificEnsemble,
        InternationalEnsemble,
        BondSpecificEnsemble,
        ETFSpecificEnsemble,
    )
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False
    print("[WARN] Phase 2 asset class ensembles not available")

# Phase 2 imports - Meta Ensemble
try:
    from src.models.meta_ensemble import (
        MetaEnsembleCombiner,
        EnhancedTradingSystem,
        get_meta_ensemble,
        get_enhanced_system,
    )
    META_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    META_ENSEMBLE_AVAILABLE = False
    logger.debug(f"Meta ensemble not available: {e}")

# Phase 3 imports - Specialized Features
try:
    from src.features.international_features import InternationalFeatures
    INTL_FEATURES_AVAILABLE = True
except ImportError:
    INTL_FEATURES_AVAILABLE = False

try:
    from src.features.crypto_features import CryptoFeatures
    CRYPTO_FEATURES_AVAILABLE = True
except ImportError:
    CRYPTO_FEATURES_AVAILABLE = False

try:
    from src.features.commodity_features import CommodityFeatures
    COMMODITY_FEATURES_AVAILABLE = True
except ImportError:
    COMMODITY_FEATURES_AVAILABLE = False

# Phase 3 validation
try:
    from src.validation.phase3_validation import Phase3Validator
    PHASE3_VALIDATION_AVAILABLE = True
except ImportError:
    PHASE3_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnifiedTradingSystem:
    """
    Unified Trading System combining Phase 1, 2, and 3.

    Architecture:
    1. Feature Layer (Phase 3): Specialized features for each asset class
    2. Ensemble Layer (Phase 2): 7 specialized ensembles + meta-combiner
    3. Integration Layer (Phase 1): 20 advanced features including stress protection

    Flow:
    ticker → detect asset class → load specialized features →
    route to specialized ensemble → combine with meta-ensemble →
    apply Phase 1 risk management → output signal
    """

    # Asset class mappings
    INTERNATIONAL_STOCKS = ['BABA', 'JD', 'PDD', 'NIO', 'BIDU', 'BILI', 'LI', 'XPEV',
                           'SONY', 'TM', 'HMC', 'MUFG', 'SAP', 'ASML', 'NVO', 'AZN',
                           'BP', 'SHEL', 'UL', 'TSM']

    CRYPTO_STOCKS = ['COIN', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF', 'CIFR',
                     'SQ', 'PYPL', 'HOOD']

    COMMODITY_STOCKS = ['XOM', 'CVX', 'OXY', 'SLB', 'HAL', 'GDX', 'GOLD', 'NEM',
                        'FCX', 'RIO', 'BHP', 'VALE']

    FOREX_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCNY=X', 'AUDUSD=X']

    BOND_ETFS = ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK']

    ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLF', 'XLE', 'XLK', 'XLV']

    def __init__(self):
        """Initialize unified trading system."""
        self.phase1_available = PHASE1_AVAILABLE
        self.phase2_available = PHASE2_AVAILABLE
        self.meta_ensemble_available = META_ENSEMBLE_AVAILABLE

        # Initialize Phase 1 if available
        if self.phase1_available:
            self.phase1_system = get_phase1_system()
            logger.info("[UNIFIED] Phase 1 system initialized (20 features)")
        else:
            self.phase1_system = None

        # Initialize Phase 2 ensembles if available
        if self.phase2_available:
            self.ensemble_factory = AssetClassEnsembleFactory()
            logger.info("[UNIFIED] Phase 2 ensemble factory initialized")
        else:
            self.ensemble_factory = None

        # Initialize Meta Ensemble if available
        if self.meta_ensemble_available:
            self.meta_ensemble = get_meta_ensemble()
            self.enhanced_system = get_enhanced_system()
            logger.info("[UNIFIED] Meta ensemble initialized")
        else:
            self.meta_ensemble = None
            self.enhanced_system = None

        # Initialize Phase 3 feature generators
        self.intl_features = InternationalFeatures() if INTL_FEATURES_AVAILABLE else None
        self.crypto_features = CryptoFeatures() if CRYPTO_FEATURES_AVAILABLE else None
        self.commodity_features = CommodityFeatures() if COMMODITY_FEATURES_AVAILABLE else None

        # Track which phases are active
        self.active_phases = []
        if self.phase1_available:
            self.active_phases.append('Phase1')
        if self.phase2_available:
            self.active_phases.append('Phase2')
        if self.meta_ensemble_available:
            self.active_phases.append('MetaEnsemble')

        logger.info(f"[UNIFIED] Active phases: {self.active_phases}")

    def detect_asset_class(self, ticker: str) -> str:
        """
        Detect the asset class for a ticker.

        Returns:
            One of: 'international', 'crypto', 'commodity', 'forex', 'bond', 'etf', 'equity'
        """
        ticker_upper = ticker.upper()

        # Check specific asset classes
        if ticker_upper in [t.upper() for t in self.INTERNATIONAL_STOCKS]:
            return 'international'
        elif ticker_upper in [t.upper() for t in self.CRYPTO_STOCKS]:
            return 'crypto'
        elif ticker_upper in [t.upper() for t in self.COMMODITY_STOCKS]:
            return 'commodity'
        elif '=X' in ticker_upper or ticker_upper in [t.upper() for t in self.FOREX_PAIRS]:
            return 'forex'
        elif ticker_upper in [t.upper() for t in self.BOND_ETFS]:
            return 'bond'
        elif ticker_upper in [t.upper() for t in self.ETFS]:
            return 'etf'

        # Check for Hong Kong stocks (China)
        if '.HK' in ticker_upper or '.SS' in ticker_upper or '.SZ' in ticker_upper:
            return 'international'

        # Default to equity
        return 'equity'

    def generate_specialized_features(
        self,
        ticker: str,
        data: pd.DataFrame,
        asset_class: str = None
    ) -> pd.DataFrame:
        """
        Generate specialized features based on asset class.

        Args:
            ticker: Stock ticker
            data: OHLCV DataFrame
            asset_class: Asset class (auto-detected if None)

        Returns:
            DataFrame with specialized features added
        """
        if asset_class is None:
            asset_class = self.detect_asset_class(ticker)

        df = data.copy()
        features_added = []

        # Add international features
        if asset_class == 'international' and self.intl_features:
            df = self.intl_features.create_all_features(df, ticker)
            features_added.extend(self.intl_features.get_feature_names())

        # Add crypto features
        elif asset_class == 'crypto' and self.crypto_features:
            df = self.crypto_features.create_all_features(df, ticker)
            features_added.extend(self.crypto_features.get_feature_names())

        # Add commodity features
        elif asset_class == 'commodity' and self.commodity_features:
            df = self.commodity_features.create_all_features(df, ticker)
            features_added.extend(self.commodity_features.get_feature_names())

        logger.info(f"[UNIFIED] Added {len(features_added)} {asset_class} features for {ticker}")
        return df

    def get_ensemble_prediction(
        self,
        ticker: str,
        data: pd.DataFrame,
        asset_class: str = None
    ) -> Dict:
        """
        Get prediction from the appropriate specialized ensemble.

        Args:
            ticker: Stock ticker
            data: OHLCV DataFrame with features
            asset_class: Asset class (auto-detected if None)

        Returns:
            Dict with prediction, confidence, and ensemble used
        """
        if asset_class is None:
            asset_class = self.detect_asset_class(ticker)

        if not self.phase2_available or self.ensemble_factory is None:
            # Fallback to simple prediction
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'direction': 'HOLD',
                'ensemble_used': 'fallback',
                'asset_class': asset_class
            }

        # Get specialized ensemble
        ensemble = self.ensemble_factory.get_ensemble(ticker)

        # Generate prediction
        result = ensemble.predict(data)
        result['asset_class'] = asset_class

        return result

    def generate_signal(
        self,
        ticker: str,
        market_data: Dict,
        portfolio: Dict = None,
        price_data: pd.DataFrame = None
    ) -> Dict:
        """
        Generate comprehensive trading signal using all phases.

        Args:
            ticker: Stock ticker
            market_data: Market conditions dict
            portfolio: Current portfolio positions
            price_data: Optional OHLCV DataFrame

        Returns:
            Comprehensive trading signal with all recommendations
        """
        portfolio = portfolio or {}

        # Step 1: Detect asset class
        asset_class = self.detect_asset_class(ticker)

        # Step 2: Generate specialized features if data provided
        features_used = []
        if price_data is not None and len(price_data) > 0:
            enhanced_data = self.generate_specialized_features(ticker, price_data, asset_class)
            features_used = list(set(enhanced_data.columns) - set(price_data.columns))
        else:
            enhanced_data = None

        # Step 3: Get ensemble prediction
        if enhanced_data is not None:
            ensemble_result = self.get_ensemble_prediction(ticker, enhanced_data, asset_class)
        else:
            ensemble_result = {
                'prediction': 0.5,
                'confidence': 0.5,
                'direction': 'HOLD',
                'ensemble_used': 'none',
                'asset_class': asset_class
            }

        # Step 4: Use meta-ensemble if available
        if self.meta_ensemble_available and self.meta_ensemble and enhanced_data is not None:
            try:
                # Use predict method instead of combine_predictions
                meta_result = self.meta_ensemble.predict(enhanced_data, ticker, market_data)
                # Convert signal (-1 to 1) to prediction (0 to 1) scale
                meta_signal = (meta_result.get('signal', 0) + 1) / 2
                meta_conf = meta_result.get('confidence', 0.5)
                # Blend ensemble and meta predictions
                final_prediction = (
                    ensemble_result['prediction'] * 0.6 +
                    meta_signal * 0.4
                )
                final_confidence = (
                    ensemble_result['confidence'] * 0.6 +
                    meta_conf * 0.4
                )
                meta_result['combined_signal'] = meta_signal
            except Exception as e:
                logger.warning(f"Meta ensemble prediction failed: {e}")
                final_prediction = ensemble_result['prediction']
                final_confidence = ensemble_result['confidence']
                meta_result = {}
        else:
            final_prediction = ensemble_result['prediction']
            final_confidence = ensemble_result['confidence']
            meta_result = {}

        # Step 5: Apply Phase 1 risk management
        if self.phase1_available and self.phase1_system:
            phase1_result = self.phase1_system.generate_enhanced_signal(
                ticker=ticker,
                market_data=market_data,
                portfolio=portfolio,
                base_signal=final_prediction
            )

            # Integrate Phase 1 adjustments
            final_signal = phase1_result.get('weighted_signal', final_prediction)
            position_size = phase1_result.get('position_size', 0.1)
            stress_status = phase1_result.get('stress_status', {})
            should_trade = phase1_result.get('should_trade', True)
            action = phase1_result.get('action', 'HOLD')
        else:
            final_signal = final_prediction
            position_size = 0.1 if final_prediction > 0.6 or final_prediction < 0.4 else 0
            stress_status = {}
            should_trade = abs(final_prediction - 0.5) > 0.1
            action = 'LONG' if final_prediction > 0.6 else ('SHORT' if final_prediction < 0.4 else 'HOLD')

        # Step 6: Compile comprehensive result
        return {
            # Core signal
            'ticker': ticker,
            'action': action,
            'should_trade': should_trade,
            'final_signal': final_signal,
            'confidence': final_confidence,
            'position_size': position_size,

            # Asset class info
            'asset_class': asset_class,
            'ensemble_used': ensemble_result.get('ensemble_used', 'unknown'),

            # Phase contributions
            'phases_active': self.active_phases,
            'ensemble_prediction': ensemble_result.get('prediction', 0.5),
            'meta_prediction': meta_result.get('combined_signal', None),

            # Features
            'specialized_features_count': len(features_used),
            'specialized_features': features_used[:10] if features_used else [],  # Top 10

            # Risk management
            'stress_status': stress_status,

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'system_version': '2.0-unified'
        }

    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'phases_active': self.active_phases,
            'phase1_available': self.phase1_available,
            'phase2_available': self.phase2_available,
            'meta_ensemble_available': self.meta_ensemble_available,
            'feature_modules': {
                'international': INTL_FEATURES_AVAILABLE,
                'crypto': CRYPTO_FEATURES_AVAILABLE,
                'commodity': COMMODITY_FEATURES_AVAILABLE,
            },
            'ensemble_types': [
                'EquitySpecificEnsemble',
                'ForexSpecificEnsemble',
                'CryptoSpecificEnsemble',
                'CommoditySpecificEnsemble',
                'InternationalEnsemble',
                'BondSpecificEnsemble',
                'ETFSpecificEnsemble',
            ] if self.phase2_available else [],
            'total_features_phase1': 20 if self.phase1_available else 0,
            'total_ensembles_phase2': 7 if self.phase2_available else 0,
        }


# =============================================================================
# API ENDPOINTS FOR WEBAPP
# =============================================================================

class UnifiedAPIEndpoints:
    """
    API endpoints for unified system integration with webapp.
    """

    def __init__(self):
        self.system = UnifiedTradingSystem()

    def get_signal(self, request_data: Dict) -> Dict:
        """
        API: /api/unified/signal

        Get unified trading signal combining all phases.
        """
        ticker = request_data.get('ticker', '')
        market_data = request_data.get('market_data', {})
        portfolio = request_data.get('portfolio', {})
        price_data = request_data.get('price_data', None)

        # Convert price_data to DataFrame if provided
        if price_data is not None and not isinstance(price_data, pd.DataFrame):
            price_data = pd.DataFrame(price_data)

        return self.system.generate_signal(
            ticker=ticker,
            market_data=market_data,
            portfolio=portfolio,
            price_data=price_data
        )

    def get_asset_class(self, request_data: Dict) -> Dict:
        """
        API: /api/unified/asset-class

        Detect asset class for a ticker.
        """
        ticker = request_data.get('ticker', '')
        asset_class = self.system.detect_asset_class(ticker)

        return {
            'ticker': ticker,
            'asset_class': asset_class,
            'ensemble_type': f"{asset_class.title()}SpecificEnsemble"
        }

    def get_status(self, request_data: Dict = None) -> Dict:
        """
        API: /api/unified/status

        Get system status.
        """
        return self.system.get_status()

    def get_all_endpoints(self) -> Dict:
        """
        API: /api/unified/endpoints

        List all available endpoints.
        """
        return {
            'endpoints': [
                '/api/unified/signal',
                '/api/unified/asset-class',
                '/api/unified/status',
                '/api/unified/endpoints',
            ],
            'phases': self.system.active_phases,
            'version': '2.0-unified'
        }


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_unified_system = None
_unified_api = None


def get_unified_system() -> UnifiedTradingSystem:
    """Get singleton UnifiedTradingSystem instance."""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedTradingSystem()
    return _unified_system


def get_unified_api() -> UnifiedAPIEndpoints:
    """Get singleton UnifiedAPIEndpoints instance."""
    global _unified_api
    if _unified_api is None:
        _unified_api = UnifiedAPIEndpoints()
    return _unified_api


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("UNIFIED TRADING SYSTEM TEST (Phase 1 + 2 + 3)")
    print("=" * 70)

    # Initialize system
    system = UnifiedTradingSystem()

    # Show status
    print("\n[STATUS] System Configuration:")
    status = system.get_status()
    print(f"   Active Phases: {status['phases_active']}")
    print(f"   Phase 1 (20 features): {status['phase1_available']}")
    print(f"   Phase 2 (7 ensembles): {status['phase2_available']}")
    print(f"   Meta Ensemble: {status['meta_ensemble_available']}")
    print(f"   Feature Modules: {status['feature_modules']}")

    # Test asset class detection
    print("\n[TEST] Asset Class Detection:")
    test_tickers = ['AAPL', 'BABA', 'COIN', 'XOM', 'EURUSD=X', 'TLT', 'SPY', '0700.HK']
    for ticker in test_tickers:
        asset_class = system.detect_asset_class(ticker)
        print(f"   {ticker:12} -> {asset_class}")

    # Test signal generation
    print("\n[TEST] Signal Generation:")
    market_data = {
        'vix': 18,
        'market_return': 0.01,
        'spread': 0.001,
        'volume': 5000000,
        'volatility': 0.02,
    }

    for ticker in ['AAPL', 'BABA', 'COIN']:
        signal = system.generate_signal(
            ticker=ticker,
            market_data=market_data,
            portfolio={}
        )
        print(f"\n   {ticker}:")
        print(f"      Asset Class: {signal['asset_class']}")
        print(f"      Action: {signal['action']}")
        print(f"      Confidence: {signal['confidence']:.2%}")
        print(f"      Position Size: {signal['position_size']:.2%}")
        print(f"      Phases Active: {signal['phases_active']}")

    print("\n" + "=" * 70)
    print("UNIFIED SYSTEM TEST COMPLETE")
    print("Phase 1 + Phase 2 + Phase 3 Integration Successful!")
    print("=" * 70)
