"""
China Model Sector Router

Routes Chinese stock predictions to sector-specific models based on test results.

Routing Logic:
- Pharma/Biotech stocks → ChinaPharmaPredictor (70-81% accuracy)
- All other sectors → Fallback to US/Intl model (China model only 40-46% accuracy)

This ensures we only use the China model where it performs well.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.models.china_sector_classifier import ChinaSectorClassifier
from src.models.china_pharma_predictor import ChinaPharmaPredictor
from src.models.china_predictor import ChinaMarketPredictor

logger = logging.getLogger(__name__)


class ChinaSectorRouter:
    """
    Routes Chinese stocks to appropriate models based on sector.

    Uses sector-specific models where performance is proven,
    otherwise returns None to trigger fallback to US/Intl model.
    """

    def __init__(self):
        """Initialize sector router."""
        self.sector_classifier = ChinaSectorClassifier
        self.pharma_model = None  # Lazy loading
        self.general_model = None  # Lazy loading for non-pharma (if needed)

        logger.info("[SECTOR ROUTER] China sector-based routing initialized")

    def _get_pharma_model(self) -> ChinaPharmaPredictor:
        """Get or create pharma model (lazy loading)."""
        if self.pharma_model is None:
            logger.info("[SECTOR ROUTER] Loading pharma-optimized model...")
            self.pharma_model = ChinaPharmaPredictor()
        return self.pharma_model

    def should_use_china_model(self, ticker: str) -> bool:
        """
        Determine if China model should be used for this ticker.

        Args:
            ticker: Stock ticker

        Returns:
            True if should use China model, False to use fallback
        """
        return self.sector_classifier.should_use_china_model(ticker)

    def get_sector_info(self, ticker: str) -> dict:
        """
        Get sector information and expected performance.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with sector info and performance metrics
        """
        sector = self.sector_classifier.get_sector(ticker)
        sector_name = self.sector_classifier.get_sector_name(ticker)
        performance = self.sector_classifier.get_sector_performance(sector)

        return {
            'ticker': ticker,
            'sector': sector,
            'sector_name': sector_name,
            'use_china_model': self.should_use_china_model(ticker),
            'expected_accuracy': performance['expected_accuracy'],
            'recommendation': performance['recommendation']
        }

    def route_prediction(self, ticker: str, X: pd.DataFrame, training_mode: bool = False) -> Optional[dict]:
        """
        Route prediction to appropriate model based on sector.

        Args:
            ticker: Stock ticker
            X: Feature DataFrame
            training_mode: If True, return model reference for training

        Returns:
            Dict with model and routing info, or None to use fallback
        """
        # Get sector
        sector = self.sector_classifier.get_sector(ticker)
        sector_name = self.sector_classifier.get_sector_name(ticker)

        logger.info(f"[SECTOR ROUTER] Routing {ticker} ({sector_name})...")

        # Route based on sector
        if sector == 'pharma':
            # Use pharma-optimized model
            logger.info(f"  ✓ Using PHARMA model (expected 70-81% accuracy)")
            model = self._get_pharma_model()

            return {
                'model': model,
                'sector': sector,
                'sector_name': sector_name,
                'model_type': 'pharma_optimized',
                'expected_accuracy': 0.76,
                'use_china_model': True
            }

        else:
            # Don't use China model for other sectors
            logger.info(f"  ⚠ Skipping China model for {sector_name} (use fallback)")
            logger.info(f"    Reason: China model only 40-46% accuracy for this sector")

            return None  # Signal to use fallback (US/Intl model)

    def add_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add features for Chinese stocks using sector-appropriate method.

        Args:
            df: Raw OHLC DataFrame
            ticker: Stock ticker

        Returns:
            DataFrame with features added
        """
        # Get sector
        sector = self.sector_classifier.get_sector(ticker)

        # Use sector-specific feature engineering
        if sector == 'pharma':
            model = self._get_pharma_model()
            return model.add_features(df, ticker)
        else:
            # For non-pharma, use general China model (though we won't use predictions)
            if self.general_model is None:
                self.general_model = ChinaMarketPredictor()
            return self.general_model.add_features(df, ticker)

    def fit(self, ticker: str, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train sector-appropriate model.

        Args:
            ticker: Stock ticker
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        # Get routing info
        routing_info = self.route_prediction(ticker, X, training_mode=True)

        if routing_info is None:
            logger.info(f"[SECTOR ROUTER] No China model training for {ticker} (will use fallback)")
            return None

        # Train the appropriate model
        model = routing_info['model']
        logger.info(f"[SECTOR ROUTER] Training {routing_info['model_type']} for {ticker}...")

        model.fit(X, y, X_val, y_val)

        logger.info(f"[SECTOR ROUTER] Training complete for {ticker}")

    def predict(self, ticker: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using sector-appropriate model.

        Args:
            ticker: Stock ticker
            X: Feature DataFrame

        Returns:
            Predictions array, or None to use fallback
        """
        # Get routing info
        routing_info = self.route_prediction(ticker, X)

        if routing_info is None:
            # No China model for this sector
            return None

        # Make predictions with appropriate model
        model = routing_info['model']
        predictions = model.predict(X)

        logger.info(f"[SECTOR ROUTER] Predictions made using {routing_info['model_type']}")

        return predictions


if __name__ == '__main__':
    # Test the sector router
    print("China Sector Router - Test\n")
    print("=" * 70)

    router = ChinaSectorRouter()

    test_tickers = [
        ('1177.HK', 'Pharma'),
        ('2269.HK', 'Pharma'),
        ('0700.HK', 'Tech'),
        ('9988.HK', 'Tech'),
        ('2319.HK', 'Consumer'),
        ('0939.HK', 'Finance'),
    ]

    print("\nSector Routing Decisions:\n")

    for ticker, expected_sector in test_tickers:
        info = router.get_sector_info(ticker)

        print(f"Ticker: {ticker}")
        print(f"  Sector: {info['sector_name']}")
        print(f"  Use China Model: {'YES' if info['use_china_model'] else 'NO (use fallback)'}")
        print(f"  Expected Accuracy: {info['expected_accuracy']:.1%}")
        print(f"  Recommendation: {info['recommendation']}")
        print()

    print("\nSummary:")
    print("  [+] Pharma stocks -> China Pharma Model (70-81% accuracy)")
    print("  [-] Other sectors -> Fallback to US/Intl Model")
