"""
Market Classifier

Determines which prediction model to use based on asset ticker and market characteristics.

Models:
- US/International Model: Optimized for US stocks, forex, commodities
  - Uses VIX, SPY, DXY, GLD macro features
  - Phase 1+2 + Selective Phase 4 (107 features)

- Chinese Model: Optimized for Chinese A-shares and HK-listed stocks
  - Uses CSI300, SSEC, HSI, CNY macro features
  - Phase 1+2 + China-specific macro features
  - Regime detection and validation-based weighting

Rules:
- .HK (Hong Kong) → Chinese model
- .SS (Shanghai) → Chinese model
- .SZ (Shenzhen) → Chinese model
- Everything else → US/International model
"""

import re
from typing import Literal


class MarketClassifier:
    """
    Classifies assets into market regions to select appropriate prediction model.
    """

    # Market suffixes
    HONG_KONG_SUFFIXES = ['.HK', '.HKG']
    SHANGHAI_SUFFIXES = ['.SS', '.SSE']
    SHENZHEN_SUFFIXES = ['.SZ', '.SZE']

    # All Chinese market suffixes
    CHINESE_SUFFIXES = HONG_KONG_SUFFIXES + SHANGHAI_SUFFIXES + SHENZHEN_SUFFIXES

    @classmethod
    def get_market(cls, ticker: str) -> Literal['us_international', 'chinese']:
        """
        Determine which market/model should be used for this ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', '0700.HK', '600519.SS')

        Returns:
            'us_international' or 'chinese'
        """
        ticker_upper = ticker.upper()

        # Check for Chinese market suffixes
        for suffix in cls.CHINESE_SUFFIXES:
            if ticker_upper.endswith(suffix):
                return 'chinese'

        # Default to US/International model
        return 'us_international'

    @classmethod
    def get_market_details(cls, ticker: str) -> dict:
        """
        Get detailed market information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with market classification details
        """
        market = cls.get_market(ticker)
        ticker_upper = ticker.upper()

        details = {
            'ticker': ticker,
            'market': market,
            'exchange': None,
            'region': None,
            'model_type': None,
            'macro_features': None
        }

        if market == 'chinese':
            # Determine specific exchange
            if any(ticker_upper.endswith(s) for s in cls.HONG_KONG_SUFFIXES):
                details['exchange'] = 'HKEX'
                details['region'] = 'Hong Kong'
            elif any(ticker_upper.endswith(s) for s in cls.SHANGHAI_SUFFIXES):
                details['exchange'] = 'SSE'
                details['region'] = 'Shanghai'
            elif any(ticker_upper.endswith(s) for s in cls.SHENZHEN_SUFFIXES):
                details['exchange'] = 'SZSE'
                details['region'] = 'Shenzhen'

            details['model_type'] = 'Chinese Markets Model'
            details['macro_features'] = ['CSI300', 'SSEC', 'HSI', 'CNY', 'GLD']

        else:
            details['region'] = 'US/International'
            details['model_type'] = 'US/International Model'
            details['macro_features'] = ['VIX', 'SPY', 'DXY', 'GLD']

        return details

    @classmethod
    def should_use_chinese_model(cls, ticker: str) -> bool:
        """
        Quick check if Chinese model should be used.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if Chinese model should be used, False otherwise
        """
        return cls.get_market(ticker) == 'chinese'

    @classmethod
    def is_chinese_a_share(cls, ticker: str) -> bool:
        """
        Check if ticker is a Chinese A-share (Shanghai or Shenzhen).

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if A-share, False otherwise
        """
        ticker_upper = ticker.upper()
        return any(ticker_upper.endswith(s) for s in cls.SHANGHAI_SUFFIXES + cls.SHENZHEN_SUFFIXES)

    @classmethod
    def is_hong_kong_stock(cls, ticker: str) -> bool:
        """
        Check if ticker is Hong Kong listed.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if HK-listed, False otherwise
        """
        ticker_upper = ticker.upper()
        return any(ticker_upper.endswith(s) for s in cls.HONG_KONG_SUFFIXES)

    @classmethod
    def get_performance_expectations(cls, ticker: str) -> dict:
        """
        Get expected model performance based on market classification.

        Based on test results:
        - US/International: 48.6% profitable (17/35 assets)
        - Hong Kong: ~50% profitable
        - Chinese A-shares: 0% profitable (NOT RECOMMENDED)

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with performance expectations
        """
        market = cls.get_market(ticker)

        if market == 'chinese':
            if cls.is_chinese_a_share(ticker):
                return {
                    'expected_profitability': 0.0,
                    'confidence': 'very_low',
                    'recommendation': 'NOT RECOMMENDED - Model performs poorly on A-shares',
                    'baseline_accuracy': '40-46%',
                    'notes': 'Shanghai/Shenzhen A-shares require specialized model'
                }
            elif cls.is_hong_kong_stock(ticker):
                return {
                    'expected_profitability': 0.50,
                    'confidence': 'medium',
                    'recommendation': 'ACCEPTABLE - Use with caution',
                    'baseline_accuracy': '50%',
                    'notes': 'Hong Kong stocks perform reasonably well'
                }

        # US/International
        return {
            'expected_profitability': 0.486,
            'confidence': 'high',
            'recommendation': 'RECOMMENDED - Model optimized for this market',
            'baseline_accuracy': '48.6%',
            'notes': 'Best performance on US Tech (70%) and US Finance (62.5%)'
        }


class ModelRouter:
    """
    Routes prediction requests to appropriate model based on market classification.
    """

    def __init__(self, us_model=None, chinese_model=None):
        """
        Initialize model router.

        Args:
            us_model: US/International prediction model (optional, for lazy loading)
            chinese_model: Chinese markets prediction model (optional, for lazy loading)
        """
        self.us_model = us_model
        self.chinese_model = chinese_model
        self.classifier = MarketClassifier()

    def predict(self, ticker: str, features):
        """
        Route prediction to appropriate model.

        Args:
            ticker: Stock ticker symbol
            features: Feature data (DataFrame or array)

        Returns:
            Prediction from appropriate model
        """
        market = self.classifier.get_market(ticker)

        if market == 'chinese':
            return self.chinese_model.predict(features)
        else:
            return self.us_model.predict(features)

    def get_model_info(self, ticker: str) -> dict:
        """
        Get information about which model will be used.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with model routing information
        """
        market = self.classifier.get_market(ticker)
        details = self.classifier.get_market_details(ticker)
        performance = self.classifier.get_performance_expectations(ticker)

        return {
            **details,
            **performance,
            'model_used': 'chinese_model' if market == 'chinese' else 'us_model'
        }


if __name__ == "__main__":
    # Test market classification
    print("="*60)
    print("MARKET CLASSIFIER TEST")
    print("="*60)

    test_tickers = [
        # US stocks
        'AAPL', 'MSFT', 'GOOGL', 'TSLA',
        # Hong Kong
        '0700.HK', '9988.HK', '2318.HK',
        # Shanghai A-shares
        '600519.SS', '601318.SS',
        # Shenzhen A-shares
        '000858.SZ', '002594.SZ',
        # Forex/Commodities
        'EURUSD=X', 'GC=F'
    ]

    classifier = MarketClassifier()

    for ticker in test_tickers:
        market = classifier.get_market(ticker)
        details = classifier.get_market_details(ticker)
        performance = classifier.get_performance_expectations(ticker)

        print(f"\n{ticker}:")
        print(f"  Market: {market}")
        print(f"  Exchange: {details['exchange']}")
        print(f"  Region: {details['region']}")
        print(f"  Model: {details['model_type']}")
        print(f"  Macro Features: {', '.join(details['macro_features'])}")
        print(f"  Expected Profitability: {performance['expected_profitability']:.1%}")
        print(f"  Confidence: {performance['confidence']}")
        print(f"  Recommendation: {performance['recommendation']}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
