"""
RSI Risk Adapter - Enhanced Signal Filtering with RSI Risk Assessment
======================================================================
From 'us model fixing5.pdf':
- RSI-based risk classification (6 levels)
- Position sizing based on RSI
- Stop-loss optimization based on RSI and volatility
- Signal blocking for extreme RSI conditions

This module enhances signals by:
1. Classifying RSI into risk levels
2. Adjusting confidence based on RSI
3. Adjusting position size based on RSI + volatility
4. Setting appropriate stop-loss and take-profit targets
5. Blocking high-risk signals (e.g., BUY when RSI > 80)

Author: Claude Code
Date: 2025-12-16
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RSIRiskLevel(Enum):
    """RSI-based risk classification."""
    EXTREME_OVERSOLD = "extreme_oversold"      # RSI < 20
    OVERSOLD = "oversold"                       # RSI 20-30
    NEUTRAL_BUY_ZONE = "neutral_buy"           # RSI 30-50 (best for BUY)
    NEUTRAL_SELL_ZONE = "neutral_sell"         # RSI 50-70
    OVERBOUGHT = "overbought"                   # RSI 70-80
    EXTREME_OVERBOUGHT = "extreme_overbought"  # RSI > 80


class VolatilityProfile(Enum):
    """Stock volatility classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RSIEnhancedSignal:
    """Result from RSI-enhanced signal processing."""
    ticker: str
    signal_type: str
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    rsi: float
    rsi_risk_level: str
    volatility_profile: str
    should_trade: bool
    block_reason: Optional[str] = None

    # Adjustment details
    rsi_confidence_adj: float = 0.0
    rsi_position_adj: float = 1.0
    vol_position_adj: float = 1.0
    stop_loss_multiplier: float = 1.0


class RSIRiskAdapter:
    """
    Enhanced trend adaptation with RSI filtering and stop-loss optimization.

    Key Features:
    1. RSI Risk Classification - 6 levels from extreme oversold to extreme overbought
    2. Signal Confidence Adjustment - Boost/penalize based on RSI alignment
    3. Position Sizing - Reduce for risky RSI, increase for favorable RSI
    4. Stop-Loss Optimization - Wider stops for oversold, tighter for overbought
    5. Signal Blocking - Block extremely risky signals (BUY at RSI>80, SELL at RSI<20)
    """

    def __init__(self):
        # RSI risk adjustments for BUY and SELL signals
        self.rsi_risk_adjustments = {
            RSIRiskLevel.EXTREME_OVERSOLD: {
                'BUY': {
                    'confidence_boost': 0.15,      # +15% confidence for oversold BUY
                    'position_mult': 1.3,          # 30% larger position
                    'stop_loss_mult': 1.5,         # Wider stop (more room)
                    'take_profit_mult': 1.3,       # Higher target
                },
                'SELL': {
                    'confidence_penalty': -0.20,   # -20% confidence
                    'position_mult': 0.5,          # 50% smaller position
                    'block': True,                 # HIGH RISK - consider blocking
                    'block_reason': 'Extreme oversold - risky to short',
                },
            },
            RSIRiskLevel.OVERSOLD: {
                'BUY': {
                    'confidence_boost': 0.10,
                    'position_mult': 1.2,
                    'stop_loss_mult': 1.3,
                    'take_profit_mult': 1.2,
                },
                'SELL': {
                    'confidence_penalty': -0.15,
                    'position_mult': 0.7,
                },
            },
            RSIRiskLevel.NEUTRAL_BUY_ZONE: {
                'BUY': {
                    'confidence_boost': 0.05,      # Slight boost - good zone
                    'position_mult': 1.1,
                    'stop_loss_mult': 1.0,         # Normal stop
                    'take_profit_mult': 1.0,
                },
                'SELL': {
                    'confidence_penalty': -0.05,
                    'position_mult': 0.9,
                },
            },
            RSIRiskLevel.NEUTRAL_SELL_ZONE: {
                'BUY': {
                    'confidence_boost': 0.0,       # No adjustment
                    'position_mult': 1.0,
                    'stop_loss_mult': 0.9,         # Slightly tighter stop
                    'take_profit_mult': 1.0,
                },
                'SELL': {
                    'confidence_boost': 0.05,      # Slight boost - good zone
                    'position_mult': 1.1,
                },
            },
            RSIRiskLevel.OVERBOUGHT: {
                'BUY': {
                    'confidence_penalty': -0.10,   # -10% confidence
                    'position_mult': 0.7,          # 30% smaller position
                    'stop_loss_mult': 0.8,         # Tighter stop
                    'take_profit_mult': 0.8,       # Lower target
                },
                'SELL': {
                    'confidence_boost': 0.10,
                    'position_mult': 1.3,
                },
            },
            RSIRiskLevel.EXTREME_OVERBOUGHT: {
                'BUY': {
                    'confidence_penalty': -0.20,
                    'position_mult': 0.5,
                    'block': True,                 # HIGH RISK - consider blocking
                    'block_reason': 'Extreme overbought - risky to buy',
                },
                'SELL': {
                    'confidence_boost': 0.15,
                    'position_mult': 1.5,
                    'stop_loss_mult': 1.3,
                    'take_profit_mult': 1.3,
                },
            },
        }

        # Stock-specific volatility profiles
        self.volatility_profiles = {
            # High volatility stocks
            'TSLA': VolatilityProfile.HIGH,
            'NVDA': VolatilityProfile.HIGH,
            'AFRM': VolatilityProfile.HIGH,
            'CRCL': VolatilityProfile.HIGH,
            'UBER': VolatilityProfile.HIGH,
            'WIX': VolatilityProfile.MEDIUM,
            'MIRM': VolatilityProfile.MEDIUM,
            # Medium volatility
            'AAL': VolatilityProfile.MEDIUM,
            'CCL': VolatilityProfile.MEDIUM,
            # Low volatility
            'PFE': VolatilityProfile.LOW,
            'MRK': VolatilityProfile.LOW,
            'F': VolatilityProfile.LOW,
            # Crypto - always high
            'BTC-USD': VolatilityProfile.HIGH,
            'ETH-USD': VolatilityProfile.HIGH,
            'BNB-USD': VolatilityProfile.HIGH,
            'XRP-USD': VolatilityProfile.HIGH,
            'DOGE-USD': VolatilityProfile.HIGH,
        }

        # Volatility-based adjustments
        self.vol_adjustments = {
            VolatilityProfile.HIGH: {
                'stop_loss_mult': 1.3,      # Wider stops for high vol
                'position_mult': 0.8,       # Smaller positions
                'take_profit_mult': 1.2,    # Higher targets possible
            },
            VolatilityProfile.MEDIUM: {
                'stop_loss_mult': 1.0,
                'position_mult': 1.0,
                'take_profit_mult': 1.0,
            },
            VolatilityProfile.LOW: {
                'stop_loss_mult': 0.8,      # Tighter stops for low vol
                'position_mult': 1.2,       # Can take larger positions
                'take_profit_mult': 0.9,    # More conservative targets
            },
        }

        # Base stop-loss and take-profit percentages
        self.base_stop_loss = {
            'stock': 0.08,          # 8% stop for stocks
            'crypto': 0.12,         # 12% stop for crypto
            'commodity': 0.10,      # 10% stop for commodities
            'forex': 0.06,          # 6% stop for forex
        }

        self.base_take_profit = {
            'BUY': {
                'LOW': 0.05,        # 5% target for low vol
                'MEDIUM': 0.08,     # 8% target for medium vol
                'HIGH': 0.12,       # 12% target for high vol
            },
            'SELL': {
                'LOW': 0.04,
                'MEDIUM': 0.07,
                'HIGH': 0.10,
            }
        }

        # Statistics
        self.stats = {
            'total_processed': 0,
            'blocked_signals': 0,
            'confidence_boosted': 0,
            'confidence_reduced': 0,
            'position_increased': 0,
            'position_decreased': 0,
        }

        logger.info("RSIRiskAdapter initialized with 6-level RSI classification")

    def classify_rsi_risk(self, rsi: float) -> RSIRiskLevel:
        """Classify RSI into risk levels."""
        if rsi < 20:
            return RSIRiskLevel.EXTREME_OVERSOLD
        elif rsi < 30:
            return RSIRiskLevel.OVERSOLD
        elif rsi < 50:
            return RSIRiskLevel.NEUTRAL_BUY_ZONE
        elif rsi < 70:
            return RSIRiskLevel.NEUTRAL_SELL_ZONE
        elif rsi < 80:
            return RSIRiskLevel.OVERBOUGHT
        else:
            return RSIRiskLevel.EXTREME_OVERBOUGHT

    def get_volatility_profile(self, ticker: str, volatility: float = None) -> VolatilityProfile:
        """Get volatility profile for a ticker."""
        # Check known profiles first
        ticker_upper = ticker.upper()
        if ticker_upper in self.volatility_profiles:
            return self.volatility_profiles[ticker_upper]

        # Crypto detection
        if '-USD' in ticker_upper and ticker_upper not in ['GC=F', 'SI=F', 'CL=F']:
            return VolatilityProfile.HIGH

        # Futures detection
        if '=F' in ticker_upper:
            return VolatilityProfile.MEDIUM

        # Use provided volatility if available
        if volatility is not None:
            if volatility > 0.50:  # 50% annualized
                return VolatilityProfile.HIGH
            elif volatility > 0.25:  # 25% annualized
                return VolatilityProfile.MEDIUM
            else:
                return VolatilityProfile.LOW

        # Default to medium
        return VolatilityProfile.MEDIUM

    def get_asset_class(self, ticker: str) -> str:
        """Determine asset class for stop-loss calculation."""
        ticker_upper = ticker.upper()

        if '-USD' in ticker_upper:
            return 'crypto'
        elif '=F' in ticker_upper:
            return 'commodity'
        elif ticker_upper in ['EURUSD', 'GBPUSD', 'USDJPY', 'EUR=X', 'GBP=X']:
            return 'forex'
        else:
            return 'stock'

    def enhance_signal_with_rsi(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        rsi: float,
        volatility: float = None,
        position_multiplier: float = 1.0,
        stop_loss_pct: float = None,
        market_trend: str = "sideways",
        strict_blocking: bool = False,
    ) -> RSIEnhancedSignal:
        """
        Enhance a trading signal with RSI-based risk adjustments.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            rsi: Current RSI value (0-100)
            volatility: Optional volatility for profile detection
            position_multiplier: Base position multiplier
            stop_loss_pct: Base stop-loss percentage
            market_trend: Current market trend
            strict_blocking: If True, block high-risk signals completely

        Returns:
            RSIEnhancedSignal with all adjustments applied
        """
        self.stats['total_processed'] += 1

        # Classify RSI risk
        rsi_risk = self.classify_rsi_risk(rsi)
        rsi_adjustments = self.rsi_risk_adjustments[rsi_risk][signal_type]

        # Get volatility profile
        vol_profile = self.get_volatility_profile(ticker, volatility)
        vol_adjustments = self.vol_adjustments[vol_profile]

        # Get asset class for base stop-loss
        asset_class = self.get_asset_class(ticker)
        base_stop = stop_loss_pct if stop_loss_pct else self.base_stop_loss.get(asset_class, 0.08)

        # Check if signal should be blocked
        should_trade = True
        block_reason = None

        if rsi_adjustments.get('block', False):
            if strict_blocking:
                should_trade = False
                block_reason = rsi_adjustments.get('block_reason', f'RSI {rsi:.1f} too risky for {signal_type}')
                self.stats['blocked_signals'] += 1
            else:
                # Don't block but severely reduce position
                rsi_adjustments = {**rsi_adjustments, 'position_mult': 0.3}

        # Calculate confidence adjustment
        confidence_adj = rsi_adjustments.get('confidence_boost', 0)
        if 'confidence_penalty' in rsi_adjustments:
            confidence_adj = rsi_adjustments['confidence_penalty']

        adjusted_confidence = confidence * (1 + confidence_adj)
        adjusted_confidence = max(0.10, min(0.95, adjusted_confidence))  # Clamp to 10-95%

        if confidence_adj > 0:
            self.stats['confidence_boosted'] += 1
        elif confidence_adj < 0:
            self.stats['confidence_reduced'] += 1

        # Calculate position multiplier
        rsi_position_mult = rsi_adjustments.get('position_mult', 1.0)
        vol_position_mult = vol_adjustments['position_mult']
        final_position_mult = position_multiplier * rsi_position_mult * vol_position_mult

        if final_position_mult > 1.0:
            self.stats['position_increased'] += 1
        elif final_position_mult < 1.0:
            self.stats['position_decreased'] += 1

        # Calculate stop-loss
        rsi_stop_mult = rsi_adjustments.get('stop_loss_mult', 1.0)
        vol_stop_mult = vol_adjustments['stop_loss_mult']
        stop_loss_multiplier = rsi_stop_mult * vol_stop_mult
        final_stop_loss = base_stop * stop_loss_multiplier

        # For SELL (short) signals, cap stop-loss
        if signal_type == 'SELL':
            final_stop_loss = min(final_stop_loss, 0.10)  # Max 10% stop on shorts

        # Calculate take-profit target
        base_take_profit = self.base_take_profit[signal_type][vol_profile.value.upper()]
        rsi_tp_mult = rsi_adjustments.get('take_profit_mult', 1.0)
        vol_tp_mult = vol_adjustments['take_profit_mult']
        final_take_profit = base_take_profit * rsi_tp_mult * vol_tp_mult

        return RSIEnhancedSignal(
            ticker=ticker,
            signal_type=signal_type,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=final_position_mult,
            stop_loss_pct=final_stop_loss,
            take_profit_pct=final_take_profit,
            rsi=rsi,
            rsi_risk_level=rsi_risk.value,
            volatility_profile=vol_profile.value,
            should_trade=should_trade,
            block_reason=block_reason,
            rsi_confidence_adj=confidence_adj,
            rsi_position_adj=rsi_position_mult,
            vol_position_adj=vol_position_mult,
            stop_loss_multiplier=stop_loss_multiplier,
        )

    def get_rsi_risk_summary(self, rsi: float, signal_type: str) -> Dict[str, Any]:
        """Get a summary of RSI risk for display."""
        rsi_risk = self.classify_rsi_risk(rsi)
        adjustments = self.rsi_risk_adjustments[rsi_risk][signal_type]

        # Determine risk level for display
        if rsi_risk in [RSIRiskLevel.EXTREME_OVERSOLD, RSIRiskLevel.EXTREME_OVERBOUGHT]:
            if adjustments.get('block', False):
                risk_display = "HIGH RISK"
            else:
                risk_display = "ELEVATED"
        elif rsi_risk in [RSIRiskLevel.OVERSOLD, RSIRiskLevel.OVERBOUGHT]:
            risk_display = "MODERATE"
        else:
            risk_display = "LOW"

        # Determine recommendation
        if signal_type == 'BUY':
            if rsi < 30:
                recommendation = "Good entry - oversold"
            elif rsi > 70:
                recommendation = "Risky entry - overbought"
            else:
                recommendation = "Neutral entry zone"
        else:  # SELL
            if rsi > 70:
                recommendation = "Good short - overbought"
            elif rsi < 30:
                recommendation = "Risky short - oversold"
            else:
                recommendation = "Neutral short zone"

        return {
            'rsi': rsi,
            'rsi_risk_level': rsi_risk.value,
            'risk_display': risk_display,
            'recommendation': recommendation,
            'confidence_adjustment': adjustments.get('confidence_boost', adjustments.get('confidence_penalty', 0)),
            'position_adjustment': adjustments.get('position_mult', 1.0),
            'should_block': adjustments.get('block', False),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total = self.stats['total_processed']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'blocked_pct': self.stats['blocked_signals'] / total * 100,
            'boosted_pct': self.stats['confidence_boosted'] / total * 100,
            'reduced_pct': self.stats['confidence_reduced'] / total * 100,
            'increased_pct': self.stats['position_increased'] / total * 100,
            'decreased_pct': self.stats['position_decreased'] / total * 100,
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'total_processed': 0,
            'blocked_signals': 0,
            'confidence_boosted': 0,
            'confidence_reduced': 0,
            'position_increased': 0,
            'position_decreased': 0,
        }


# Factory function
def create_rsi_adapter() -> RSIRiskAdapter:
    """Create an RSI risk adapter instance."""
    return RSIRiskAdapter()


# Utility function for quick RSI assessment
def assess_rsi_risk(rsi: float, signal_type: str) -> Dict[str, Any]:
    """Quick RSI risk assessment without full adapter."""
    adapter = RSIRiskAdapter()
    return adapter.get_rsi_risk_summary(rsi, signal_type)
