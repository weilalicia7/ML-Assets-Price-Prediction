"""
Energy Specific Optimizer - Priority 1 Fix
==========================================
Fixes the -82% energy sector loss by implementing subsector-specific strategies.

From 'profit test fixing1 full code.pdf':
- Energy sector was showing -82% cumulative loss
- Root cause: Treating all energy stocks the same despite different drivers
- Solution: Classify into 6 subsectors with tailored strategies

Subsectors:
1. Integrated Oil (XOM, CVX) - Diversified, lower volatility, follows crude with lag
2. E&P (COP, EOG, PXD) - High leverage to oil prices, exploration risk
3. Oil Services (SLB, HAL, BKR) - Capex cycle driven, leading indicator
4. Refining (VLO, MPC, PSX) - Crack spread driven, inverse crude correlation
5. Midstream (KMI, WMB, ET) - Fee-based, dividend focus, rate sensitive
6. Renewables (FSLR, ENPH, NEE) - Policy driven, tech stock behavior

Author: Claude Code (US/Intl Model)
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


class EnergySubsector(Enum):
    """Energy stock subsector classification."""
    INTEGRATED_OIL = "integrated_oil"
    EXPLORATION_PRODUCTION = "e_and_p"
    OIL_SERVICES = "oil_services"
    REFINING = "refining"
    MIDSTREAM = "midstream"
    RENEWABLES = "renewables"
    UNKNOWN = "unknown"


@dataclass
class EnergySignalAdjustment:
    """Result of energy-specific signal adjustment."""
    ticker: str
    subsector: EnergySubsector
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    should_trade: bool
    adjustment_reason: str
    crack_spread: Optional[float] = None
    contango_signal: Optional[str] = None
    subsector_momentum: Optional[float] = None


class EnergySpecificOptimizer:
    """
    Energy sector specific optimizer that classifies stocks into subsectors
    and applies tailored trading strategies.

    Key Features:
    - 6-subsector classification
    - Crack spread analysis for refiners
    - Futures curve analysis (contango/backwardation)
    - Subsector-specific confidence thresholds
    - Position sizing by subsector volatility profile
    """

    # Subsector ticker mappings (US/INTL model only - no China stocks)
    SUBSECTOR_TICKERS = {
        EnergySubsector.INTEGRATED_OIL: ['XOM', 'CVX', 'SHEL', 'BP', 'TTE'],
        EnergySubsector.EXPLORATION_PRODUCTION: ['COP', 'EOG', 'PXD', 'DVN', 'OXY', 'FANG', 'HES'],
        EnergySubsector.OIL_SERVICES: ['SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'CHX'],
        EnergySubsector.REFINING: ['VLO', 'MPC', 'PSX', 'DINO', 'PBF', 'DK'],
        EnergySubsector.MIDSTREAM: ['KMI', 'WMB', 'ET', 'EPD', 'MPLX', 'OKE', 'TRGP'],
        EnergySubsector.RENEWABLES: ['FSLR', 'ENPH', 'NEE', 'AES', 'BE', 'RUN', 'SEDG'],
    }

    # Subsector-specific parameters
    SUBSECTOR_PARAMS = {
        EnergySubsector.INTEGRATED_OIL: {
            'confidence_threshold': 0.55,  # Lower threshold (more stable)
            'position_mult': 1.0,          # Normal position
            'volatility_scale': 0.8,       # Lower volatility expected
            'oil_correlation': 0.7,        # High but lagged correlation to crude
            'strategy': 'trend_following',
        },
        EnergySubsector.EXPLORATION_PRODUCTION: {
            'confidence_threshold': 0.65,  # Higher threshold (more volatile)
            'position_mult': 0.7,          # Smaller positions
            'volatility_scale': 1.3,       # Higher volatility
            'oil_correlation': 0.9,        # Highest correlation to crude
            'strategy': 'momentum',
        },
        EnergySubsector.OIL_SERVICES: {
            'confidence_threshold': 0.60,
            'position_mult': 0.8,
            'volatility_scale': 1.2,
            'oil_correlation': 0.6,        # Leading indicator, less direct correlation
            'strategy': 'leading_indicator',
        },
        EnergySubsector.REFINING: {
            'confidence_threshold': 0.55,
            'position_mult': 0.9,
            'volatility_scale': 1.0,
            'oil_correlation': -0.3,       # Often INVERSE correlation (crack spreads)
            'strategy': 'crack_spread',
        },
        EnergySubsector.MIDSTREAM: {
            'confidence_threshold': 0.50,  # Lowest threshold (most stable)
            'position_mult': 1.2,          # Larger positions OK
            'volatility_scale': 0.6,       # Low volatility, fee-based
            'oil_correlation': 0.3,        # Low correlation
            'strategy': 'dividend_capture',
        },
        EnergySubsector.RENEWABLES: {
            'confidence_threshold': 0.70,  # High threshold (policy risk)
            'position_mult': 0.6,          # Smaller positions
            'volatility_scale': 1.5,       # High volatility (tech-like)
            'oil_correlation': -0.2,       # Slight inverse correlation
            'strategy': 'tech_momentum',
        },
        EnergySubsector.UNKNOWN: {
            'confidence_threshold': 0.60,
            'position_mult': 0.5,
            'volatility_scale': 1.0,
            'oil_correlation': 0.5,
            'strategy': 'conservative',
        },
    }

    def __init__(self, enable_futures_analysis: bool = True):
        """
        Initialize energy optimizer.

        Args:
            enable_futures_analysis: Whether to analyze crude futures curve
        """
        self.enable_futures_analysis = enable_futures_analysis
        self._build_ticker_lookup()
        self._futures_cache: Dict[str, Tuple[str, datetime]] = {}
        self._crack_spread_cache: Dict[str, Tuple[float, datetime]] = {}

        logger.info("Initialized EnergySpecificOptimizer with 6 subsector strategies")

    def _build_ticker_lookup(self):
        """Build reverse lookup from ticker to subsector."""
        self.ticker_to_subsector = {}
        for subsector, tickers in self.SUBSECTOR_TICKERS.items():
            for ticker in tickers:
                self.ticker_to_subsector[ticker.upper()] = subsector

    def is_energy_stock(self, ticker: str) -> bool:
        """Check if ticker is a known energy stock."""
        return ticker.upper() in self.ticker_to_subsector

    def classify_subsector(self, ticker: str) -> EnergySubsector:
        """
        Classify energy stock into subsector.

        Args:
            ticker: Stock ticker

        Returns:
            EnergySubsector enum value
        """
        ticker_upper = ticker.upper()
        return self.ticker_to_subsector.get(ticker_upper, EnergySubsector.UNKNOWN)

    def get_crack_spread(self) -> Optional[float]:
        """
        Calculate crack spread (gasoline + heating oil - crude).
        Crack spread = 2*RBOB + 1*HO - 3*CL (simplified 3:2:1 crack)

        Returns:
            Crack spread in dollars per barrel, or None if data unavailable
        """
        cache_key = 'crack_spread'
        now = datetime.now()

        # Check cache (valid for 15 minutes)
        if cache_key in self._crack_spread_cache:
            cached_value, cache_time = self._crack_spread_cache[cache_key]
            if (now - cache_time).total_seconds() < 900:
                return cached_value

        try:
            # Fetch futures prices
            crude = yf.Ticker('CL=F').history(period='1d')
            gasoline = yf.Ticker('RB=F').history(period='1d')
            heating_oil = yf.Ticker('HO=F').history(period='1d')

            if crude.empty or gasoline.empty or heating_oil.empty:
                return None

            cl_price = crude['Close'].iloc[-1]
            rb_price = gasoline['Close'].iloc[-1] * 42  # Convert to $/barrel
            ho_price = heating_oil['Close'].iloc[-1] * 42

            # 3:2:1 crack spread
            crack_spread = (2 * rb_price + ho_price) / 3 - cl_price

            self._crack_spread_cache[cache_key] = (crack_spread, now)
            return crack_spread

        except Exception as e:
            logger.warning(f"Failed to calculate crack spread: {e}")
            return None

    def get_futures_curve_signal(self) -> str:
        """
        Analyze crude oil futures curve for contango/backwardation.

        Returns:
            'contango': Front month < back month (bearish near-term)
            'backwardation': Front month > back month (bullish near-term)
            'flat': Minimal difference
        """
        cache_key = 'futures_curve'
        now = datetime.now()

        if cache_key in self._futures_cache:
            cached_value, cache_time = self._futures_cache[cache_key]
            if (now - cache_time).total_seconds() < 900:
                return cached_value

        try:
            # CL=F is front month, CLJ24 etc for back months
            # Simplified: compare CL=F to BZ=F (Brent) as proxy
            front = yf.Ticker('CL=F').history(period='5d')
            brent = yf.Ticker('BZ=F').history(period='5d')

            if front.empty or brent.empty:
                return 'flat'

            front_price = front['Close'].iloc[-1]
            back_price = brent['Close'].iloc[-1]  # Brent typically at premium

            spread_pct = (back_price - front_price) / front_price

            if spread_pct > 0.02:
                signal = 'contango'
            elif spread_pct < -0.02:
                signal = 'backwardation'
            else:
                signal = 'flat'

            self._futures_cache[cache_key] = (signal, now)
            return signal

        except Exception as e:
            logger.warning(f"Failed to analyze futures curve: {e}")
            return 'flat'

    def calculate_subsector_momentum(self, subsector: EnergySubsector) -> float:
        """
        Calculate momentum for a subsector using representative tickers.

        Args:
            subsector: Energy subsector

        Returns:
            20-day momentum as decimal (e.g., 0.05 = 5% gain)
        """
        tickers = self.SUBSECTOR_TICKERS.get(subsector, [])
        if not tickers:
            return 0.0

        momentums = []
        for ticker in tickers[:3]:  # Use top 3 tickers
            try:
                data = yf.Ticker(ticker).history(period='1mo')
                if len(data) >= 20:
                    momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-20]) - 1
                    momentums.append(momentum)
            except Exception:
                pass

        return np.mean(momentums) if momentums else 0.0

    def optimize_energy_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.25,
        momentum: float = 0.0,
    ) -> EnergySignalAdjustment:
        """
        Optimize an energy stock signal using subsector-specific strategies.

        Args:
            ticker: Energy stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility
            momentum: Recent price momentum

        Returns:
            EnergySignalAdjustment with optimized parameters
        """
        ticker_upper = ticker.upper()
        subsector = self.classify_subsector(ticker_upper)
        params = self.SUBSECTOR_PARAMS[subsector]

        # Get market context
        crack_spread = None
        contango_signal = None
        if self.enable_futures_analysis:
            contango_signal = self.get_futures_curve_signal()
            if subsector == EnergySubsector.REFINING:
                crack_spread = self.get_crack_spread()

        subsector_momentum = self.calculate_subsector_momentum(subsector)

        # Initialize adjustments
        adjusted_confidence = confidence
        position_multiplier = params['position_mult']
        should_trade = True
        reasons = []

        # Strategy-specific adjustments
        strategy = params['strategy']

        if strategy == 'crack_spread' and subsector == EnergySubsector.REFINING:
            # Refiners: Crack spread is KEY driver
            if crack_spread is not None:
                if crack_spread > 25:  # Wide crack spread = bullish for refiners
                    if signal_type == 'BUY':
                        adjusted_confidence *= 1.15
                        position_multiplier *= 1.2
                        reasons.append(f"Wide crack spread (${crack_spread:.1f}) bullish")
                    else:
                        adjusted_confidence *= 0.85
                        reasons.append(f"Wide crack spread contradicts SELL")
                elif crack_spread < 10:  # Narrow crack = bearish
                    if signal_type == 'SELL':
                        adjusted_confidence *= 1.15
                        reasons.append(f"Narrow crack spread (${crack_spread:.1f}) bearish")
                    else:
                        adjusted_confidence *= 0.85
                        reasons.append(f"Narrow crack spread contradicts BUY")

        elif strategy == 'leading_indicator' and subsector == EnergySubsector.OIL_SERVICES:
            # Oil services lead the cycle - use momentum confirmation
            if momentum > 0.05 and signal_type == 'BUY':
                adjusted_confidence *= 1.10
                reasons.append("Momentum confirms oil services cycle upturn")
            elif momentum < -0.05 and signal_type == 'SELL':
                adjusted_confidence *= 1.10
                reasons.append("Momentum confirms oil services downturn")

        elif strategy == 'dividend_capture' and subsector == EnergySubsector.MIDSTREAM:
            # Midstream: Focus on stability, reduce SELL aggression
            if signal_type == 'SELL':
                adjusted_confidence *= 0.90
                position_multiplier *= 0.8
                reasons.append("Midstream SELL confidence reduced (dividend stability)")
            else:
                position_multiplier *= 1.1
                reasons.append("Midstream BUY favored (stable dividends)")

        elif strategy == 'tech_momentum' and subsector == EnergySubsector.RENEWABLES:
            # Renewables: High volatility, treat like tech
            if volatility > 0.35:
                position_multiplier *= 0.7
                reasons.append("High volatility - reduced position")
            if subsector_momentum > 0.10 and signal_type == 'BUY':
                adjusted_confidence *= 1.10
                reasons.append("Strong renewables momentum")

        elif strategy == 'momentum' and subsector == EnergySubsector.EXPLORATION_PRODUCTION:
            # E&P: High oil leverage, use momentum
            if contango_signal == 'backwardation' and signal_type == 'BUY':
                adjusted_confidence *= 1.10
                reasons.append("Backwardation bullish for E&P")
            elif contango_signal == 'contango' and signal_type == 'SELL':
                adjusted_confidence *= 1.10
                reasons.append("Contango bearish for E&P")

        # Apply confidence threshold
        conf_threshold = params['confidence_threshold']
        if adjusted_confidence < conf_threshold:
            should_trade = False
            reasons.append(f"Below threshold ({adjusted_confidence:.2f} < {conf_threshold:.2f})")

        # Cap confidence at 0.95
        adjusted_confidence = min(adjusted_confidence, 0.95)

        # Position multiplier bounds
        position_multiplier = max(0.3, min(1.5, position_multiplier))

        adjustment_reason = "; ".join(reasons) if reasons else f"{subsector.value} standard rules"

        return EnergySignalAdjustment(
            ticker=ticker_upper,
            subsector=subsector,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_multiplier,
            should_trade=should_trade,
            adjustment_reason=adjustment_reason,
            crack_spread=crack_spread,
            contango_signal=contango_signal,
            subsector_momentum=subsector_momentum,
        )

    def get_subsector_stats(self) -> Dict[str, Any]:
        """Get statistics for all subsectors."""
        stats = {}
        for subsector in EnergySubsector:
            if subsector == EnergySubsector.UNKNOWN:
                continue
            tickers = self.SUBSECTOR_TICKERS.get(subsector, [])
            momentum = self.calculate_subsector_momentum(subsector)
            stats[subsector.value] = {
                'tickers': tickers,
                'count': len(tickers),
                'momentum_20d': momentum,
                'params': self.SUBSECTOR_PARAMS[subsector],
            }
        return stats


# Factory function
def create_energy_optimizer(enable_futures: bool = True) -> EnergySpecificOptimizer:
    """Create an EnergySpecificOptimizer instance."""
    return EnergySpecificOptimizer(enable_futures_analysis=enable_futures)
