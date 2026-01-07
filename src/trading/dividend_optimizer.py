"""
Dividend Aware Optimizer - Priority 3 Fix
==========================================
Improves financial sector performance by accounting for dividend dynamics.

From 'profit test fixing1 full code.pdf':
- Financial stocks (JPM, GS, BAC) showing suboptimal performance
- Root cause: Momentum signals conflict with dividend capture strategies
- Solution: Classify by dividend profile and adjust signal timing

Key Insight:
- High dividend stocks often drop on ex-dividend date (expected)
- Model might generate SELL signals that are just ex-div price drops
- Need to track ex-dividend dates and yield profiles

Dividend Profiles:
1. High Yield (>4%): Dividend capture focus, less momentum-driven
2. Growth + Dividend (2-4%): Balance growth and income signals
3. Low/No Dividend (<2%): Standard momentum strategies
4. Dividend Aristocrats: Special handling (consistent dividend growers)

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


class DividendProfile(Enum):
    """Stock dividend profile classification."""
    HIGH_YIELD = "high_yield"           # >4% yield
    GROWTH_DIVIDEND = "growth_dividend"  # 2-4% yield
    LOW_DIVIDEND = "low_dividend"        # <2% yield
    ARISTOCRAT = "aristocrat"            # 25+ years of dividend increases
    NO_DIVIDEND = "no_dividend"          # No dividend
    UNKNOWN = "unknown"


@dataclass
class DividendSignalAdjustment:
    """Result of dividend-aware signal adjustment."""
    ticker: str
    dividend_profile: DividendProfile
    dividend_yield: float
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    should_trade: bool
    adjustment_reason: str
    near_ex_dividend: bool = False
    days_to_ex_div: Optional[int] = None
    is_rate_sensitive: bool = False


class DividendAwareOptimizer:
    """
    Dividend-aware optimizer for financial and high-yield stocks.

    Key Features:
    - Dividend yield classification
    - Ex-dividend date tracking
    - Rate sensitivity adjustment
    - Dividend aristocrat special handling
    - Financial sector specific rules
    """

    # Known dividend aristocrats (25+ years of dividend increases)
    DIVIDEND_ARISTOCRATS = [
        # Financial Aristocrats
        'AFL', 'CB', 'CINF', 'PFG', 'TROW',
        # Other sectors (for reference)
        'JNJ', 'PG', 'KO', 'PEP', 'CL', 'MMM', 'ABT', 'WMT', 'TGT',
        'MCD', 'SYY', 'GWW', 'ADP', 'SHW', 'ECL', 'ED', 'XEL',
    ]

    # Financial sector stocks (rate-sensitive)
    FINANCIAL_STOCKS = {
        # Major Banks (highly rate sensitive)
        'JPM': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'BAC': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'WFC': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'C': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'USB': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'PNC': {'sector': 'bank', 'rate_sensitivity': 'high'},
        'TFC': {'sector': 'bank', 'rate_sensitivity': 'high'},

        # Investment Banks (moderate rate sensitivity)
        'GS': {'sector': 'investment_bank', 'rate_sensitivity': 'moderate'},
        'MS': {'sector': 'investment_bank', 'rate_sensitivity': 'moderate'},

        # Insurance (less rate sensitive)
        'BRK-B': {'sector': 'insurance', 'rate_sensitivity': 'low'},
        'MET': {'sector': 'insurance', 'rate_sensitivity': 'moderate'},
        'PRU': {'sector': 'insurance', 'rate_sensitivity': 'moderate'},
        'AFL': {'sector': 'insurance', 'rate_sensitivity': 'moderate'},

        # Asset Managers
        'BLK': {'sector': 'asset_manager', 'rate_sensitivity': 'low'},
        'TROW': {'sector': 'asset_manager', 'rate_sensitivity': 'low'},
        'SCHW': {'sector': 'broker', 'rate_sensitivity': 'high'},

        # Credit Cards / Consumer Finance
        'V': {'sector': 'payments', 'rate_sensitivity': 'low'},
        'MA': {'sector': 'payments', 'rate_sensitivity': 'low'},
        'AXP': {'sector': 'consumer_finance', 'rate_sensitivity': 'moderate'},
        'COF': {'sector': 'consumer_finance', 'rate_sensitivity': 'moderate'},

        # REITs (highly rate sensitive)
        'PLD': {'sector': 'reit', 'rate_sensitivity': 'high'},
        'AMT': {'sector': 'reit', 'rate_sensitivity': 'high'},
        'CCI': {'sector': 'reit', 'rate_sensitivity': 'high'},
        'EQIX': {'sector': 'reit', 'rate_sensitivity': 'high'},
        'O': {'sector': 'reit', 'rate_sensitivity': 'high'},
        'SPG': {'sector': 'reit', 'rate_sensitivity': 'high'},
    }

    # Dividend profile parameters
    PROFILE_PARAMS = {
        DividendProfile.HIGH_YIELD: {
            'yield_range': (0.04, 1.0),
            'confidence_threshold': 0.50,  # Lower bar (stable stocks)
            'position_mult': 1.1,
            'sell_penalty': 0.85,          # Penalize SELL signals
            'ex_div_buffer_days': 7,       # Avoid trading 7 days before ex-div
        },
        DividendProfile.GROWTH_DIVIDEND: {
            'yield_range': (0.02, 0.04),
            'confidence_threshold': 0.55,
            'position_mult': 1.0,
            'sell_penalty': 0.95,
            'ex_div_buffer_days': 5,
        },
        DividendProfile.LOW_DIVIDEND: {
            'yield_range': (0.005, 0.02),
            'confidence_threshold': 0.55,
            'position_mult': 1.0,
            'sell_penalty': 1.0,           # No penalty
            'ex_div_buffer_days': 3,
        },
        DividendProfile.ARISTOCRAT: {
            'yield_range': (0.0, 1.0),     # Any yield
            'confidence_threshold': 0.45,  # Very low bar (quality stocks)
            'position_mult': 1.2,          # Larger positions
            'sell_penalty': 0.80,          # Strong SELL penalty
            'ex_div_buffer_days': 10,
        },
        DividendProfile.NO_DIVIDEND: {
            'yield_range': (0.0, 0.005),
            'confidence_threshold': 0.60,
            'position_mult': 0.9,
            'sell_penalty': 1.0,
            'ex_div_buffer_days': 0,
        },
        DividendProfile.UNKNOWN: {
            'yield_range': (0.0, 1.0),
            'confidence_threshold': 0.55,
            'position_mult': 0.8,
            'sell_penalty': 1.0,
            'ex_div_buffer_days': 0,
        },
    }

    def __init__(self):
        """Initialize dividend-aware optimizer."""
        self._dividend_cache: Dict[str, Tuple[Dict, datetime]] = {}
        logger.info("Initialized DividendAwareOptimizer for financial/dividend stocks")

    def is_financial_stock(self, ticker: str) -> bool:
        """Check if ticker is a known financial stock."""
        return ticker.upper() in self.FINANCIAL_STOCKS

    def is_dividend_aristocrat(self, ticker: str) -> bool:
        """Check if ticker is a dividend aristocrat."""
        return ticker.upper() in self.DIVIDEND_ARISTOCRATS

    def get_dividend_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get dividend information for a stock.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with dividend yield, ex-date, etc.
        """
        ticker_upper = ticker.upper()
        now = datetime.now()

        # Check cache (valid for 1 day)
        if ticker_upper in self._dividend_cache:
            cached_info, cache_time = self._dividend_cache[ticker_upper]
            if (now - cache_time).total_seconds() < 86400:
                return cached_info

        try:
            stock = yf.Ticker(ticker_upper)
            info = stock.info

            dividend_yield = info.get('dividendYield', 0) or 0
            ex_dividend_date = info.get('exDividendDate')
            dividend_rate = info.get('dividendRate', 0) or 0

            # Convert ex-dividend timestamp to date
            days_to_ex_div = None
            near_ex_div = False
            if ex_dividend_date:
                ex_date = datetime.fromtimestamp(ex_dividend_date)
                days_to_ex_div = (ex_date - now).days
                if 0 < days_to_ex_div <= 14:
                    near_ex_div = True

            result = {
                'dividend_yield': dividend_yield,
                'dividend_rate': dividend_rate,
                'ex_dividend_date': ex_dividend_date,
                'days_to_ex_div': days_to_ex_div,
                'near_ex_dividend': near_ex_div,
            }

            self._dividend_cache[ticker_upper] = (result, now)
            return result

        except Exception as e:
            logger.warning(f"Failed to get dividend info for {ticker}: {e}")
            return {
                'dividend_yield': 0,
                'dividend_rate': 0,
                'ex_dividend_date': None,
                'days_to_ex_div': None,
                'near_ex_dividend': False,
            }

    def classify_dividend_profile(self, ticker: str) -> Tuple[DividendProfile, float]:
        """
        Classify stock's dividend profile.

        Args:
            ticker: Stock ticker

        Returns:
            (DividendProfile, dividend_yield)
        """
        ticker_upper = ticker.upper()

        # Check aristocrat first
        if self.is_dividend_aristocrat(ticker_upper):
            div_info = self.get_dividend_info(ticker_upper)
            return DividendProfile.ARISTOCRAT, div_info['dividend_yield']

        # Get dividend info
        div_info = self.get_dividend_info(ticker_upper)
        div_yield = div_info['dividend_yield']

        if div_yield >= 0.04:
            return DividendProfile.HIGH_YIELD, div_yield
        elif div_yield >= 0.02:
            return DividendProfile.GROWTH_DIVIDEND, div_yield
        elif div_yield >= 0.005:
            return DividendProfile.LOW_DIVIDEND, div_yield
        elif div_yield > 0:
            return DividendProfile.LOW_DIVIDEND, div_yield
        else:
            return DividendProfile.NO_DIVIDEND, 0.0

    def get_rate_sensitivity(self, ticker: str) -> str:
        """
        Get rate sensitivity for financial stocks.

        Returns:
            'high', 'moderate', 'low', or 'none'
        """
        ticker_upper = ticker.upper()
        if ticker_upper in self.FINANCIAL_STOCKS:
            return self.FINANCIAL_STOCKS[ticker_upper].get('rate_sensitivity', 'none')
        return 'none'

    def optimize_dividend_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        rate_environment: str = 'neutral',  # 'rising', 'falling', 'neutral'
    ) -> DividendSignalAdjustment:
        """
        Optimize signal for dividend-paying stock.

        Key Logic:
        1. SELL signals on high-yield stocks should be penalized
           (often just dividend-related price drops)
        2. Avoid trading near ex-dividend dates (artificial price moves)
        3. Rate-sensitive financials need rate environment consideration

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility
            momentum: Recent price momentum
            rate_environment: Interest rate trend

        Returns:
            DividendSignalAdjustment with optimized parameters
        """
        ticker_upper = ticker.upper()
        dividend_profile, dividend_yield = self.classify_dividend_profile(ticker_upper)
        params = self.PROFILE_PARAMS[dividend_profile]
        div_info = self.get_dividend_info(ticker_upper)

        near_ex_div = div_info['near_ex_dividend']
        days_to_ex_div = div_info['days_to_ex_div']
        rate_sensitivity = self.get_rate_sensitivity(ticker_upper)
        is_rate_sensitive = rate_sensitivity in ['high', 'moderate']

        # Initialize adjustments
        adjusted_confidence = confidence
        position_multiplier = params['position_mult']
        should_trade = True
        reasons = []

        # Ex-dividend date handling
        ex_div_buffer = params['ex_div_buffer_days']
        if near_ex_div and days_to_ex_div is not None:
            if days_to_ex_div <= ex_div_buffer:
                if signal_type == 'SELL':
                    # Price drop near ex-div is expected, not a real signal
                    adjusted_confidence *= 0.70
                    reasons.append(f"Near ex-div ({days_to_ex_div}d) - SELL may be dividend drop")
                elif signal_type == 'BUY':
                    # BUY before ex-div could capture dividend
                    adjusted_confidence *= 1.05
                    reasons.append(f"Near ex-div ({days_to_ex_div}d) - dividend capture opportunity")

        # Dividend profile-specific adjustments
        if signal_type == 'SELL':
            sell_penalty = params['sell_penalty']
            adjusted_confidence *= sell_penalty
            if sell_penalty < 1.0:
                reasons.append(f"{dividend_profile.value}: SELL penalty applied")

        # Aristocrat boost for BUY
        if dividend_profile == DividendProfile.ARISTOCRAT and signal_type == 'BUY':
            adjusted_confidence *= 1.10
            position_multiplier *= 1.1
            reasons.append("Dividend aristocrat BUY boost")

        # High yield stocks - more defensive
        if dividend_profile == DividendProfile.HIGH_YIELD:
            if volatility > 0.25:
                position_multiplier *= 0.8
                reasons.append("High yield + high vol - reduced position")
            if signal_type == 'BUY' and dividend_yield > 0.06:
                # Very high yield might indicate distress
                adjusted_confidence *= 0.90
                reasons.append(f"High yield ({dividend_yield:.1%}) - check for distress")

        # Rate sensitivity adjustments for financials
        if is_rate_sensitive and rate_environment != 'neutral':
            if rate_environment == 'rising':
                if signal_type == 'BUY' and rate_sensitivity == 'high':
                    # Banks benefit from rising rates
                    if ticker_upper in ['JPM', 'BAC', 'WFC', 'C', 'USB']:
                        adjusted_confidence *= 1.10
                        reasons.append("Rising rates bullish for banks")
                elif signal_type == 'SELL' and rate_sensitivity == 'high':
                    # REITs suffer in rising rate environment
                    if self.FINANCIAL_STOCKS.get(ticker_upper, {}).get('sector') == 'reit':
                        adjusted_confidence *= 1.10
                        reasons.append("Rising rates bearish for REITs")
            elif rate_environment == 'falling':
                if signal_type == 'BUY' and self.FINANCIAL_STOCKS.get(ticker_upper, {}).get('sector') == 'reit':
                    adjusted_confidence *= 1.10
                    reasons.append("Falling rates bullish for REITs")

        # Apply confidence threshold
        conf_threshold = params['confidence_threshold']
        if adjusted_confidence < conf_threshold:
            should_trade = False
            reasons.append(f"Below threshold ({adjusted_confidence:.2f} < {conf_threshold:.2f})")

        # Cap and floor
        adjusted_confidence = max(0.0, min(0.95, adjusted_confidence))
        position_multiplier = max(0.3, min(1.5, position_multiplier))

        adjustment_reason = "; ".join(reasons) if reasons else f"{dividend_profile.value} standard rules"

        return DividendSignalAdjustment(
            ticker=ticker_upper,
            dividend_profile=dividend_profile,
            dividend_yield=dividend_yield,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_multiplier,
            should_trade=should_trade,
            adjustment_reason=adjustment_reason,
            near_ex_dividend=near_ex_div,
            days_to_ex_div=days_to_ex_div,
            is_rate_sensitive=is_rate_sensitive,
        )

    def get_financial_sector_summary(self) -> Dict[str, Any]:
        """Get summary of tracked financial stocks."""
        by_sector = {}
        for ticker, info in self.FINANCIAL_STOCKS.items():
            sector = info['sector']
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(ticker)

        return {
            'total_tracked': len(self.FINANCIAL_STOCKS),
            'by_sector': by_sector,
            'aristocrats': self.DIVIDEND_ARISTOCRATS,
        }


# Factory function
def create_dividend_optimizer() -> DividendAwareOptimizer:
    """Create a DividendAwareOptimizer instance."""
    return DividendAwareOptimizer()
