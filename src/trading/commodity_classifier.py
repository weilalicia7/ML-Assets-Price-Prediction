"""
Commodity Dominance Classifier - Priority 2 Fix
===============================================
Fixes the -91% mining sector loss by separating commodity-dominant stocks.

From 'profit test fixing1 full code.pdf':
- Mining stocks (BHP, RIO, VALE) were showing -91% cumulative loss
- Root cause: Model treating them as equities when they're commodity proxies
- Solution: Classify stocks by commodity dominance and adjust signals accordingly

Key Insight:
- Some stocks are >70% driven by underlying commodity prices
- Standard equity signals fail because the model can't predict commodity moves
- Need to classify: Pure Equity vs Commodity Dominant vs Hybrid

Classification:
1. Pure Equity (0-30% commodity correlation): Standard equity strategies
2. Hybrid (30-70%): Blend commodity/equity signals
3. Commodity Dominant (70-100%): Commodity-first strategies

"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CommodityDominance(Enum):
    """Stock's commodity dominance classification."""
    PURE_EQUITY = "pure_equity"           # <30% commodity driven
    HYBRID = "hybrid"                      # 30-70% commodity driven
    COMMODITY_DOMINANT = "commodity_dominant"  # >70% commodity driven
    UNKNOWN = "unknown"


@dataclass
class CommoditySignalAdjustment:
    """Result of commodity-aware signal adjustment."""
    ticker: str
    dominance_class: CommodityDominance
    commodity_correlation: float
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    should_trade: bool
    adjustment_reason: str
    underlying_commodity: Optional[str] = None
    commodity_trend: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    equity_model_weight: float = 1.0       # How much to trust equity model


class CommodityDominanceClassifier:
    """
    Classifies stocks by their commodity price dominance and adjusts
    trading signals accordingly.

    Key Features:
    - Static commodity correlation mapping for known mining/resource stocks
    - Dynamic correlation estimation for unknown stocks
    - Commodity trend overlay (bullish/bearish commodity environment)
    - Position sizing based on commodity dominance
    - Equity model weight adjustment (trust model less for commodity-dominant stocks)
    """

    # Known commodity-dominant stocks with their underlying commodities
    # Format: ticker -> (commodity_symbol, base_correlation)
    COMMODITY_STOCKS = {
        # Major Diversified Miners (Iron Ore, Copper, Coal dominant)
        'BHP': ('IRON_ORE', 0.85),    # Iron ore ~60%, copper ~25%
        'RIO': ('IRON_ORE', 0.82),    # Iron ore ~65%
        'VALE': ('IRON_ORE', 0.88),   # Iron ore ~75%

        # Copper Miners
        'FCX': ('HG=F', 0.80),        # Freeport - copper dominant
        'SCCO': ('HG=F', 0.78),       # Southern Copper

        # Gold Miners
        'NEM': ('GC=F', 0.75),        # Newmont - gold
        'GOLD': ('GC=F', 0.80),       # Barrick - gold
        'AEM': ('GC=F', 0.72),        # Agnico Eagle

        # Steel / Iron Ore
        'CLF': ('IRON_ORE', 0.70),    # Cleveland-Cliffs
        'X': ('IRON_ORE', 0.65),      # US Steel
        'NUE': ('IRON_ORE', 0.55),    # Nucor (more hybrid)

        # Aluminum
        'AA': ('ALI=F', 0.72),        # Alcoa

        # Silver
        'PAAS': ('SI=F', 0.78),       # Pan American Silver
        'AG': ('SI=F', 0.75),         # First Majestic

        # Lithium / Battery Metals
        'ALB': ('LITHIUM', 0.70),     # Albemarle
        'SQM': ('LITHIUM', 0.72),     # SQM
        'LTHM': ('LITHIUM', 0.75),    # Livent

        # Uranium
        'CCJ': ('URANIUM', 0.80),     # Cameco

        # Coal
        'BTU': ('COAL', 0.75),        # Peabody
        'ARCH': ('COAL', 0.72),       # Arch Resources

        # Agricultural Commodities
        'ADM': ('CORN', 0.45),        # More hybrid
        'BG': ('SOYBEAN', 0.48),      # Bunge
        'CTVA': ('CORN', 0.40),       # Corteva

        # Timber
        'WY': ('LUMBER', 0.55),       # Weyerhaeuser
        'RYN': ('LUMBER', 0.52),      # Rayonier
    }

    # Commodity price proxies (Yahoo Finance symbols)
    COMMODITY_PROXIES = {
        'IRON_ORE': 'BHP',        # Use BHP as iron ore proxy (no direct futures)
        'LITHIUM': 'ALB',         # Use ALB as lithium proxy
        'URANIUM': 'URA',         # Uranium ETF
        'COAL': 'BTU',            # Use Peabody as proxy
        'CORN': 'CORN',           # Teucrium Corn ETF
        'SOYBEAN': 'SOYB',        # Teucrium Soybean
        'LUMBER': 'WOOD',         # iShares Global Timber
        'HG=F': 'HG=F',           # Copper futures
        'GC=F': 'GC=F',           # Gold futures
        'SI=F': 'SI=F',           # Silver futures
        'ALI=F': 'ALI=F',         # Aluminum futures
    }

    # Dominance class thresholds and parameters
    DOMINANCE_PARAMS = {
        CommodityDominance.PURE_EQUITY: {
            'correlation_range': (0.0, 0.30),
            'equity_model_weight': 1.0,      # Full trust in equity model
            'position_mult': 1.0,
            'confidence_threshold': 0.55,
        },
        CommodityDominance.HYBRID: {
            'correlation_range': (0.30, 0.70),
            'equity_model_weight': 0.6,      # Partial trust
            'position_mult': 0.8,
            'confidence_threshold': 0.60,
        },
        CommodityDominance.COMMODITY_DOMINANT: {
            'correlation_range': (0.70, 1.0),
            'equity_model_weight': 0.3,      # Low trust - commodity drives price
            'position_mult': 0.5,            # Small positions
            'confidence_threshold': 0.70,    # Higher bar to trade
        },
        CommodityDominance.UNKNOWN: {
            'correlation_range': (0.0, 1.0),
            'equity_model_weight': 0.5,
            'position_mult': 0.6,
            'confidence_threshold': 0.65,
        },
    }

    def __init__(self):
        """Initialize commodity dominance classifier."""
        self._commodity_trend_cache: Dict[str, Tuple[str, datetime]] = {}
        logger.info("Initialized CommodityDominanceClassifier for mining/resource stocks")

    def is_commodity_stock(self, ticker: str) -> bool:
        """Check if ticker is a known commodity-dominant stock."""
        return ticker.upper() in self.COMMODITY_STOCKS

    def get_commodity_correlation(self, ticker: str) -> Tuple[float, str]:
        """
        Get commodity correlation for a stock.

        Args:
            ticker: Stock ticker

        Returns:
            (correlation, underlying_commodity)
        """
        ticker_upper = ticker.upper()
        if ticker_upper in self.COMMODITY_STOCKS:
            commodity, correlation = self.COMMODITY_STOCKS[ticker_upper]
            return correlation, commodity
        return 0.0, None

    def classify_dominance(self, ticker: str) -> CommodityDominance:
        """
        Classify stock by commodity dominance.

        Args:
            ticker: Stock ticker

        Returns:
            CommodityDominance enum
        """
        correlation, _ = self.get_commodity_correlation(ticker)

        if correlation >= 0.70:
            return CommodityDominance.COMMODITY_DOMINANT
        elif correlation >= 0.30:
            return CommodityDominance.HYBRID
        elif correlation > 0:
            return CommodityDominance.PURE_EQUITY
        else:
            return CommodityDominance.UNKNOWN

    def get_commodity_trend(self, commodity: str) -> str:
        """
        Analyze commodity trend (bullish/bearish/neutral).

        Args:
            commodity: Commodity symbol

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        now = datetime.now()

        # Check cache
        if commodity in self._commodity_trend_cache:
            cached_trend, cache_time = self._commodity_trend_cache[commodity]
            if (now - cache_time).total_seconds() < 3600:  # 1 hour cache
                return cached_trend

        # Get proxy symbol
        proxy = self.COMMODITY_PROXIES.get(commodity, commodity)

        try:
            data = yf.Ticker(proxy).history(period='3mo')
            if len(data) < 20:
                return 'neutral'

            # Calculate 20-day and 50-day momentum
            current_price = data['Close'].iloc[-1]
            price_20d_ago = data['Close'].iloc[-20] if len(data) >= 20 else data['Close'].iloc[0]
            price_50d_ago = data['Close'].iloc[-50] if len(data) >= 50 else data['Close'].iloc[0]

            momentum_20d = (current_price - price_20d_ago) / price_20d_ago
            momentum_50d = (current_price - price_50d_ago) / price_50d_ago

            # Simple moving average trend
            sma_20 = data['Close'].iloc[-20:].mean()
            sma_50 = data['Close'].iloc[-50:].mean() if len(data) >= 50 else sma_20

            if momentum_20d > 0.05 and current_price > sma_20 > sma_50:
                trend = 'bullish'
            elif momentum_20d < -0.05 and current_price < sma_20 < sma_50:
                trend = 'bearish'
            else:
                trend = 'neutral'

            self._commodity_trend_cache[commodity] = (trend, now)
            return trend

        except Exception as e:
            logger.warning(f"Failed to get commodity trend for {commodity}: {e}")
            return 'neutral'

    def optimize_commodity_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.30,
        momentum: float = 0.0,
    ) -> CommoditySignalAdjustment:
        """
        Optimize signal for commodity-dominant stock.

        The key insight: If a stock is 80% correlated with iron ore prices,
        our equity model can only explain 20% of the price movement.
        We should:
        1. Reduce position size (we're essentially betting on commodities)
        2. Raise confidence threshold (need more certainty)
        3. Check if commodity trend aligns with signal

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility
            momentum: Recent price momentum

        Returns:
            CommoditySignalAdjustment with optimized parameters
        """
        ticker_upper = ticker.upper()
        commodity_corr, underlying_commodity = self.get_commodity_correlation(ticker_upper)
        dominance_class = self.classify_dominance(ticker_upper)
        params = self.DOMINANCE_PARAMS[dominance_class]

        # Get commodity trend if we have underlying commodity
        commodity_trend = None
        if underlying_commodity:
            commodity_trend = self.get_commodity_trend(underlying_commodity)

        # Initialize adjustments
        adjusted_confidence = confidence
        position_multiplier = params['position_mult']
        equity_model_weight = params['equity_model_weight']
        should_trade = True
        reasons = []

        # Key adjustment: Reduce confidence based on how much commodity drives price
        # If 80% commodity driven, model can only explain 20% of variance
        explainable_variance = 1.0 - commodity_corr
        adjusted_confidence *= (0.5 + 0.5 * explainable_variance)  # Scale down
        reasons.append(f"Commodity corr {commodity_corr:.0%}, model explains {explainable_variance:.0%}")

        # Commodity trend alignment check
        if commodity_trend and dominance_class == CommodityDominance.COMMODITY_DOMINANT:
            if signal_type == 'BUY' and commodity_trend == 'bullish':
                adjusted_confidence *= 1.15
                position_multiplier *= 1.2
                reasons.append(f"BUY aligned with bullish {underlying_commodity}")
            elif signal_type == 'BUY' and commodity_trend == 'bearish':
                adjusted_confidence *= 0.70
                position_multiplier *= 0.5
                reasons.append(f"BUY contradicts bearish {underlying_commodity}")
            elif signal_type == 'SELL' and commodity_trend == 'bearish':
                adjusted_confidence *= 1.15
                position_multiplier *= 1.2
                reasons.append(f"SELL aligned with bearish {underlying_commodity}")
            elif signal_type == 'SELL' and commodity_trend == 'bullish':
                adjusted_confidence *= 0.70
                position_multiplier *= 0.5
                reasons.append(f"SELL contradicts bullish {underlying_commodity}")

        # For hybrid stocks, blend adjustments
        if dominance_class == CommodityDominance.HYBRID:
            # Partial adjustment based on commodity trend
            if commodity_trend == 'bullish' and signal_type == 'BUY':
                adjusted_confidence *= 1.05
                reasons.append("Hybrid: commodity tailwind")
            elif commodity_trend == 'bearish' and signal_type == 'SELL':
                adjusted_confidence *= 1.05
                reasons.append("Hybrid: commodity headwind")

        # High volatility penalty for commodity stocks (they're already volatile)
        if volatility > 0.40 and dominance_class != CommodityDominance.PURE_EQUITY:
            position_multiplier *= 0.7
            reasons.append("High volatility reduction")

        # Apply confidence threshold
        conf_threshold = params['confidence_threshold']
        if adjusted_confidence < conf_threshold:
            should_trade = False
            reasons.append(f"Below threshold ({adjusted_confidence:.2f} < {conf_threshold:.2f})")

        # Cap and floor
        adjusted_confidence = max(0.0, min(0.95, adjusted_confidence))
        position_multiplier = max(0.2, min(1.5, position_multiplier))

        adjustment_reason = "; ".join(reasons) if reasons else "Standard processing"

        return CommoditySignalAdjustment(
            ticker=ticker_upper,
            dominance_class=dominance_class,
            commodity_correlation=commodity_corr,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_multiplier,
            should_trade=should_trade,
            adjustment_reason=adjustment_reason,
            underlying_commodity=underlying_commodity,
            commodity_trend=commodity_trend,
            equity_model_weight=equity_model_weight,
        )

    def get_commodity_stock_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked commodity stocks."""
        summary = {
            'dominant': [],
            'hybrid': [],
            'total_tracked': len(self.COMMODITY_STOCKS),
        }

        for ticker, (commodity, corr) in self.COMMODITY_STOCKS.items():
            if corr >= 0.70:
                summary['dominant'].append({
                    'ticker': ticker,
                    'commodity': commodity,
                    'correlation': corr,
                })
            elif corr >= 0.30:
                summary['hybrid'].append({
                    'ticker': ticker,
                    'commodity': commodity,
                    'correlation': corr,
                })

        return summary


# Factory function
def create_commodity_classifier() -> CommodityDominanceClassifier:
    """Create a CommodityDominanceClassifier instance."""
    return CommodityDominanceClassifier()
