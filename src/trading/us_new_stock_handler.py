"""
US New Stock/IPO Handler with SPAC Support

This module implements the tiered analysis for US IPOs, new stocks, and SPACs
from 'dual model fixing1.pdf'.

Key Features:
1. TIERED ANALYSIS BASED ON TRADING DAYS
   - 1-4 days: NO_TRADE (too volatile)
   - 5-10 days: BASIC (momentum-only)
   - 11-30 days: ENHANCED (momentum + limited quality)
   - 31-60 days: HYBRID (full analysis with adjustments)
   - 60+: FULL (standard analysis)

2. SPAC-SPECIFIC HANDLING
   - NAV floor detection ($10 typical)
   - Merger announcement detection
   - Pre/post merger differentiation
   - Redemption deadline tracking

3. OPTIONS-ENHANCED ANALYSIS (for liquid IPOs)
   - Implied volatility analysis
   - Put/call ratio
   - Options flow detection

4. US-SPECIFIC IPO SCORING
   - Lock-up expiration tracking
   - Institutional ownership
   - Underwriter reputation scoring
   - Sector momentum context

Author: Claude Code
Last Updated: 2025-12-20 (FIXING3 - Dual Model Integration)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


# ============================================================================
# SECTOR-SPECIFIC IPO THRESHOLDS
# ============================================================================

SECTOR_IPO_RULES = {
    'Technology': {
        'min_momentum_basic': 0.02,    # 2% 2-day for basic tier
        'min_momentum_enhanced': 0.0,  # No min for enhanced
        'allow_negative_eps': True,
        'max_de': 2.5,
        'position_mult_boost': 1.1     # 10% boost for tech IPOs
    },
    'Healthcare': {
        'min_momentum_basic': 0.025,
        'min_momentum_enhanced': 0.01,
        'allow_negative_eps': True,    # Biotech often unprofitable
        'max_de': 2.0,
        'position_mult_boost': 1.0
    },
    'Financials': {
        'min_momentum_basic': 0.015,
        'min_momentum_enhanced': 0.005,
        'allow_negative_eps': False,
        'max_de': 15.0,                # Banks have high leverage
        'position_mult_boost': 0.9     # Slightly lower for financials
    },
    'Consumer': {
        'min_momentum_basic': 0.02,
        'min_momentum_enhanced': 0.01,
        'allow_negative_eps': False,
        'max_de': 1.5,
        'position_mult_boost': 1.0
    },
    'Energy': {
        'min_momentum_basic': 0.03,    # Higher bar for energy
        'min_momentum_enhanced': 0.015,
        'allow_negative_eps': True,
        'max_de': 2.0,
        'position_mult_boost': 0.85
    },
    'Default': {
        'min_momentum_basic': 0.02,
        'min_momentum_enhanced': 0.005,
        'allow_negative_eps': False,
        'max_de': 3.0,
        'position_mult_boost': 1.0
    }
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class USIPOTierInfo:
    """Information about an IPO/new stock's tier classification."""
    tier: str
    trading_days: int
    can_trade: bool
    position_mult: float
    stop_mult: float
    analysis_type: str
    is_spac: bool = False
    is_direct_listing: bool = False
    lock_up_days_remaining: Optional[int] = None


@dataclass
class SPACInfo:
    """SPAC-specific information."""
    is_spac: bool
    nav_floor: float = 10.0
    has_target: bool = False
    target_announced_date: Optional[datetime] = None
    merger_completed: bool = False
    redemption_deadline: Optional[datetime] = None
    trust_value: Optional[float] = None
    can_trade: bool = True
    trade_reason: str = ''


# ============================================================================
# US NEW STOCK HANDLER
# ============================================================================

class USNewStockHandler:
    """
    Tiered analysis for US IPOs, new stocks, and SPACs.

    Trading Days Tiers:
    - 1-4 days: NO_TRADE (too volatile, insufficient data)
    - 5-10 days: BASIC (momentum-only analysis, tight stops)
    - 11-30 days: ENHANCED (momentum + limited quality, moderate stops)
    - 31-60 days: HYBRID (full analysis with adjustments)
    - 60+ days: FULL (standard analysis)
    """

    # Tier definitions
    TIERS = {
        'NO_TRADE': {'min_days': 1, 'max_days': 4, 'can_trade': False},
        'BASIC': {'min_days': 5, 'max_days': 10, 'can_trade': True, 'analysis': 'momentum_only'},
        'ENHANCED': {'min_days': 11, 'max_days': 30, 'can_trade': True, 'analysis': 'enhanced'},
        'HYBRID': {'min_days': 31, 'max_days': 60, 'can_trade': True, 'analysis': 'hybrid'},
        'FULL': {'min_days': 61, 'max_days': float('inf'), 'can_trade': True, 'analysis': 'full'}
    }

    # Position sizing by tier (relative to normal)
    TIER_POSITION_SIZING = {
        'NO_TRADE': 0.0,
        'BASIC': 0.25,     # 25% of normal position
        'ENHANCED': 0.50,   # 50% of normal position
        'HYBRID': 0.75,     # 75% of normal position
        'FULL': 1.0         # Full position
    }

    # Stop-loss multipliers by tier (tighter for newer stocks)
    TIER_STOP_MULTIPLIER = {
        'NO_TRADE': 0.0,
        'BASIC': 0.5,      # 50% of normal stop (tighter)
        'ENHANCED': 0.7,    # 70% of normal stop
        'HYBRID': 0.85,     # 85% of normal stop
        'FULL': 1.0         # Normal stop
    }

    def __init__(self):
        """Initialize the US new stock handler."""
        self.ipo_cache = {}
        self.spac_handler = USSPACHandler()

    def get_trading_days(self, signal: Dict[str, Any]) -> int:
        """Get number of trading days since IPO/listing."""
        trading_days = signal.get('trading_days', signal.get('days_listed', 0))

        if trading_days is None or trading_days == 0:
            # Try to calculate from first_trade_date
            first_trade = signal.get('first_trade_date', signal.get('ipo_date'))
            if first_trade:
                try:
                    if isinstance(first_trade, str):
                        first_trade = datetime.strptime(first_trade, '%Y-%m-%d')
                    trading_days = (datetime.now() - first_trade).days * 5 // 7  # Approximate
                except:
                    trading_days = 100  # Assume established if unknown

        return trading_days if trading_days else 100

    def get_tier(self, trading_days: int) -> str:
        """Determine the tier based on trading days."""
        if trading_days <= 4:
            return 'NO_TRADE'
        elif trading_days <= 10:
            return 'BASIC'
        elif trading_days <= 30:
            return 'ENHANCED'
        elif trading_days <= 60:
            return 'HYBRID'
        else:
            return 'FULL'

    def is_new_stock(self, signal: Dict[str, Any]) -> bool:
        """Check if this is a new stock (< 60 days)."""
        trading_days = self.get_trading_days(signal)
        return trading_days < 60

    def is_spac(self, signal: Dict[str, Any]) -> bool:
        """Check if this is a SPAC."""
        ticker = signal.get('ticker', '')
        name = signal.get('name', signal.get('company_name', '')).lower()

        # Common SPAC indicators
        spac_keywords = ['acquisition', 'spac', 'blank check', 'merger']
        if any(kw in name for kw in spac_keywords):
            return True

        # SPAC tickers often end with 'U' (units), 'W' (warrants)
        if ticker and (ticker.endswith('U') or ticker.endswith('W')):
            return True

        # Check explicit flag
        return signal.get('is_spac', False)

    def is_direct_listing(self, signal: Dict[str, Any]) -> bool:
        """Check if this is a direct listing."""
        return signal.get('is_direct_listing', False)

    def get_tier_info(self, signal: Dict[str, Any]) -> USIPOTierInfo:
        """Get comprehensive tier information."""
        trading_days = self.get_trading_days(signal)
        tier = self.get_tier(trading_days)
        is_spac = self.is_spac(signal)
        is_direct = self.is_direct_listing(signal)

        # Calculate lock-up days remaining
        lock_up_date = signal.get('lock_up_expiry', signal.get('lock_up_date'))
        lock_up_remaining = None
        if lock_up_date:
            try:
                if isinstance(lock_up_date, str):
                    lock_up_date = datetime.strptime(lock_up_date, '%Y-%m-%d')
                lock_up_remaining = (lock_up_date - datetime.now()).days
            except:
                pass

        return USIPOTierInfo(
            tier=tier,
            trading_days=trading_days,
            can_trade=self.TIERS[tier]['can_trade'],
            position_mult=self.TIER_POSITION_SIZING[tier],
            stop_mult=self.TIER_STOP_MULTIPLIER[tier],
            analysis_type=self.TIERS[tier].get('analysis', 'none'),
            is_spac=is_spac,
            is_direct_listing=is_direct,
            lock_up_days_remaining=lock_up_remaining
        )

    def can_trade(self, signal: Dict[str, Any]) -> Tuple[bool, str, Dict]:
        """
        Check if a new stock can be traded and return tier info.

        Returns:
            Tuple of (can_trade, tier_name, tier_adjustments)
        """
        tier_info = self.get_tier_info(signal)

        # Check SPAC eligibility separately
        if tier_info.is_spac:
            spac_info = self.spac_handler.analyze_spac(signal)
            if not spac_info.can_trade:
                return False, tier_info.tier, {
                    'tier': tier_info.tier,
                    'trading_days': tier_info.trading_days,
                    'can_trade': False,
                    'rejection_reason': spac_info.trade_reason
                }

        tier_dict = {
            'tier': tier_info.tier,
            'trading_days': tier_info.trading_days,
            'can_trade': tier_info.can_trade,
            'position_mult': tier_info.position_mult,
            'stop_mult': tier_info.stop_mult,
            'analysis_type': tier_info.analysis_type,
            'is_spac': tier_info.is_spac,
            'is_direct_listing': tier_info.is_direct_listing,
            'lock_up_days_remaining': tier_info.lock_up_days_remaining
        }

        signal['ipo_tier'] = tier_info.tier
        signal['ipo_trading_days'] = tier_info.trading_days
        signal['ipo_position_mult'] = tier_info.position_mult
        signal['ipo_stop_mult'] = tier_info.stop_mult

        return tier_info.can_trade, tier_info.tier, tier_dict

    def analyze_new_stock(self, signal: Dict[str, Any], tier: str) -> Dict[str, Any]:
        """
        Apply tier-specific analysis to a new stock.

        Args:
            signal: The stock signal
            tier: The determined tier

        Returns:
            Enhanced signal with tier-specific analysis
        """
        if tier == 'NO_TRADE':
            signal['trade_allowed'] = False
            signal['rejection_reason'] = 'IPO too new (< 5 trading days)'
            return signal

        if tier == 'BASIC':
            return self._analyze_basic(signal)
        elif tier == 'ENHANCED':
            return self._analyze_enhanced(signal)
        elif tier == 'HYBRID':
            return self._analyze_hybrid(signal)
        else:  # FULL
            return signal  # No modification needed

    def _analyze_basic(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        BASIC tier: Momentum-only analysis for 5-10 day old stocks.
        """
        signal['analysis_type'] = 'BASIC_MOMENTUM'

        # Get sector-specific thresholds
        sector = signal.get('sector', 'Default')
        sector_rules = SECTOR_IPO_RULES.get(sector, SECTOR_IPO_RULES['Default'])

        # Check momentum
        momentum_2d = signal.get('return_2d', signal.get('momentum_2d', 0))
        momentum_5d = signal.get('return_5d', signal.get('momentum_5d', 0))

        if momentum_2d is None:
            momentum_2d = 0
        if momentum_5d is None:
            momentum_5d = 0

        min_momentum = sector_rules['min_momentum_basic']

        # Require strong momentum for basic tier
        if momentum_2d < min_momentum and momentum_5d < min_momentum * 1.5:
            signal['trade_allowed'] = False
            signal['rejection_reason'] = f'Insufficient momentum for BASIC tier ({sector})'
            return signal

        # Skip quality checks for basic tier
        signal['quality_passed'] = True
        signal['quality_reason'] = 'basic_tier_skip'
        signal['quality_score'] = 0.5  # Neutral score

        # Calculate IPO momentum score
        signal['ipo_momentum_score'] = min(1.0, (momentum_2d * 20 + momentum_5d * 10) / 2)

        # Apply sector position boost
        signal['ipo_position_mult'] = signal.get('ipo_position_mult', 0.25) * sector_rules['position_mult_boost']

        signal['trade_allowed'] = True
        return signal

    def _analyze_enhanced(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED tier: Momentum + limited quality for 11-30 day old stocks.
        """
        signal['analysis_type'] = 'ENHANCED_MIXED'

        # Get sector-specific thresholds
        sector = signal.get('sector', 'Default')
        sector_rules = SECTOR_IPO_RULES.get(sector, SECTOR_IPO_RULES['Default'])

        # Check momentum (more relaxed than BASIC)
        momentum_2d = signal.get('return_2d', signal.get('momentum_2d', 0))
        if momentum_2d is None:
            momentum_2d = 0

        min_momentum = sector_rules['min_momentum_enhanced']

        if momentum_2d < -0.05:  # Only reject strong downtrends
            signal['trade_allowed'] = False
            signal['rejection_reason'] = 'Strong downtrend in ENHANCED tier'
            return signal

        # Limited quality check (sector-adjusted)
        eps = signal.get('eps', signal.get('trailing_eps', 0))
        if eps is None:
            eps = 0

        # Check EPS based on sector rules
        if not sector_rules['allow_negative_eps'] and eps < -0.2:
            signal['trade_allowed'] = False
            signal['rejection_reason'] = f'Negative EPS not allowed for {sector}'
            return signal

        signal['quality_passed'] = True
        signal['quality_reason'] = 'enhanced_tier_relaxed'
        signal['quality_score'] = 0.6

        signal['trade_allowed'] = True
        return signal

    def _analyze_hybrid(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        HYBRID tier: Full analysis with adjustments for 31-60 day old stocks.
        """
        signal['analysis_type'] = 'HYBRID_FULL'

        # Check lock-up expiry warning
        lock_up_remaining = signal.get('lock_up_days_remaining')
        if lock_up_remaining is not None and 0 < lock_up_remaining <= 14:
            # Within 2 weeks of lock-up expiry - be cautious
            signal['lock_up_warning'] = True
            signal['ipo_position_mult'] = signal.get('ipo_position_mult', 0.75) * 0.5

        signal['trade_allowed'] = True
        return signal

    def calculate_ipo_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate comprehensive IPO score (0-1).

        Components:
        - Momentum score (0-0.3)
        - Institutional ownership (0-0.2)
        - Underwriter quality (0-0.2)
        - Sector momentum (0-0.15)
        - Lock-up buffer (0-0.15)
        """
        score = 0.0

        # 1. Momentum score (0-0.3)
        momentum_2d = signal.get('return_2d', 0) or 0
        momentum_5d = signal.get('return_5d', 0) or 0
        momentum_score = min(0.3, max(0, (momentum_2d * 10 + momentum_5d * 5) / 2))
        score += momentum_score

        # 2. Institutional ownership (0-0.2)
        inst_ownership = signal.get('institutional_ownership', signal.get('inst_own', 0))
        if inst_ownership is None:
            inst_ownership = 0
        if inst_ownership > 0.5:
            score += 0.2
        elif inst_ownership > 0.3:
            score += 0.15
        elif inst_ownership > 0.1:
            score += 0.1

        # 3. Underwriter quality (0-0.2)
        underwriter = signal.get('lead_underwriter', signal.get('underwriter', ''))
        top_underwriters = ['goldman', 'morgan stanley', 'jp morgan', 'citi', 'bofa']
        if underwriter and any(u in underwriter.lower() for u in top_underwriters):
            score += 0.2
        elif underwriter:
            score += 0.1

        # 4. Sector momentum context (0-0.15)
        sector_momentum = signal.get('sector_momentum', signal.get('sector_return_5d', 0))
        if sector_momentum is None:
            sector_momentum = 0
        if sector_momentum > 0.05:
            score += 0.15
        elif sector_momentum > 0.02:
            score += 0.1
        elif sector_momentum > 0:
            score += 0.05

        # 5. Lock-up buffer (0-0.15)
        lock_up_remaining = signal.get('lock_up_days_remaining')
        if lock_up_remaining is not None:
            if lock_up_remaining > 90:
                score += 0.15
            elif lock_up_remaining > 60:
                score += 0.1
            elif lock_up_remaining > 30:
                score += 0.05
            # No bonus if < 30 days

        signal['ipo_score'] = min(1.0, score)
        return signal['ipo_score']

    def filter_new_stocks(self, signals: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        Filter and adjust new stocks based on their tier.

        Returns:
            List of signals that can be traded (with adjustments applied)
        """
        passed = []

        for signal in signals:
            if not self.is_new_stock(signal):
                # Not a new stock, pass through
                signal['is_new_stock'] = False
                passed.append(signal)
                continue

            signal['is_new_stock'] = True

            # Check trading eligibility
            can_trade, tier, tier_info = self.can_trade(signal)

            if not can_trade:
                if verbose:
                    print(f"  [US IPO HANDLER] Rejected {signal.get('ticker', 'UNKNOWN')}: "
                          f"tier={tier}, days={tier_info.get('trading_days', 0)}")
                continue

            # Apply tier-specific analysis
            analyzed = self.analyze_new_stock(signal, tier)

            if analyzed.get('trade_allowed', False):
                # Calculate IPO score
                self.calculate_ipo_score(analyzed)
                passed.append(analyzed)
            elif verbose:
                print(f"  [US IPO HANDLER] Rejected {signal.get('ticker', 'UNKNOWN')}: "
                      f"{analyzed.get('rejection_reason', 'Unknown')}")

        return passed


# ============================================================================
# US SPAC HANDLER
# ============================================================================

class USSPACHandler:
    """
    SPAC-specific handling with NAV floor and merger detection.

    Features:
    - NAV floor detection ($10 typical)
    - Pre-merger vs post-merger differentiation
    - Redemption deadline tracking
    - Trust value analysis
    """

    def __init__(self):
        """Initialize SPAC handler."""
        self.spac_cache = {}

    def analyze_spac(self, signal: Dict[str, Any]) -> SPACInfo:
        """
        Analyze a SPAC for trading eligibility.

        Returns:
            SPACInfo with trading eligibility and reason
        """
        ticker = signal.get('ticker', '')
        price = signal.get('price', signal.get('current_price', 0))

        # Default SPAC info
        nav_floor = signal.get('nav_floor', signal.get('trust_value', 10.0))
        has_target = signal.get('has_target', signal.get('target_announced', False))
        merger_completed = signal.get('merger_completed', False)

        # Check for warrant or unit (not tradeable in our system)
        if ticker.endswith('W') or ticker.endswith('U'):
            return SPACInfo(
                is_spac=True,
                nav_floor=nav_floor,
                has_target=has_target,
                merger_completed=merger_completed,
                can_trade=False,
                trade_reason='SPAC warrants/units not supported'
            )

        # Pre-merger SPACs trading near NAV floor
        if not merger_completed:
            if price and price < nav_floor * 0.95:
                # Trading below NAV - potential opportunity
                return SPACInfo(
                    is_spac=True,
                    nav_floor=nav_floor,
                    has_target=has_target,
                    merger_completed=merger_completed,
                    can_trade=True,
                    trade_reason='Below NAV floor - potential arbitrage'
                )
            elif price and price < nav_floor * 1.1:
                # Near NAV - low risk but limited upside
                return SPACInfo(
                    is_spac=True,
                    nav_floor=nav_floor,
                    has_target=has_target,
                    merger_completed=merger_completed,
                    can_trade=True,
                    trade_reason='Near NAV floor - limited downside'
                )
            else:
                # Trading at premium - higher risk
                return SPACInfo(
                    is_spac=True,
                    nav_floor=nav_floor,
                    has_target=has_target,
                    merger_completed=merger_completed,
                    can_trade=True,
                    trade_reason='Premium to NAV - speculative'
                )

        # Post-merger SPACs - treat more like regular IPOs
        return SPACInfo(
            is_spac=True,
            nav_floor=nav_floor,
            has_target=has_target,
            merger_completed=True,
            can_trade=True,
            trade_reason='Post-merger - standard IPO analysis'
        )

    def calculate_spac_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate SPAC-specific score (0-1).

        Components:
        - NAV floor proximity (0-0.3)
        - Target quality (0-0.3)
        - Sponsor quality (0-0.2)
        - Trust size (0-0.2)
        """
        score = 0.0

        price = signal.get('price', signal.get('current_price', 10))
        nav_floor = signal.get('nav_floor', 10.0)

        # 1. NAV floor proximity (0-0.3) - closer to NAV = safer
        if price and nav_floor:
            nav_ratio = price / nav_floor
            if nav_ratio < 0.98:
                score += 0.3  # Below NAV - very attractive
            elif nav_ratio < 1.05:
                score += 0.25  # Near NAV
            elif nav_ratio < 1.15:
                score += 0.15  # Slight premium
            elif nav_ratio < 1.3:
                score += 0.05  # High premium

        # 2. Target quality (0-0.3)
        has_target = signal.get('has_target', False)
        target_quality = signal.get('target_quality', signal.get('target_score', 0))
        if has_target:
            if target_quality and target_quality > 0.7:
                score += 0.3
            elif target_quality and target_quality > 0.5:
                score += 0.2
            else:
                score += 0.1

        # 3. Sponsor quality (0-0.2)
        sponsor = signal.get('sponsor', signal.get('sponsor_name', ''))
        top_sponsors = ['chamath', 'ackman', 'tilman', 'gary cohn']
        if sponsor and any(s in sponsor.lower() for s in top_sponsors):
            score += 0.2
        elif sponsor:
            score += 0.1

        # 4. Trust size (0-0.2) - larger trusts are more institutional
        trust_size = signal.get('trust_value', signal.get('trust_size', 0))
        if trust_size:
            if trust_size > 500e6:  # > $500M
                score += 0.2
            elif trust_size > 200e6:
                score += 0.15
            elif trust_size > 100e6:
                score += 0.1

        signal['spac_score'] = min(1.0, score)
        return signal['spac_score']


# ============================================================================
# OPTIONS-ENHANCED IPO ANALYZER (for liquid IPOs)
# ============================================================================

class USOptionsIPOAnalyzer:
    """
    Options-enhanced analysis for liquid IPOs.

    Features:
    - Implied volatility analysis
    - Put/call ratio
    - Options flow detection
    - Unusual options activity
    """

    def __init__(self):
        """Initialize options analyzer."""
        pass

    def has_options(self, signal: Dict[str, Any]) -> bool:
        """Check if the stock has liquid options."""
        has_options = signal.get('has_options', signal.get('options_available', False))
        if has_options:
            return True

        # Check for minimum volume/open interest
        options_volume = signal.get('options_volume', 0)
        open_interest = signal.get('options_open_interest', 0)

        return options_volume > 1000 or open_interest > 5000

    def analyze_options(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze options data for an IPO.

        Returns:
            Dict with options analysis results
        """
        if not self.has_options(signal):
            return {
                'has_options': False,
                'options_score': 0.5,  # Neutral
                'iv_rank': None,
                'put_call_ratio': None,
                'unusual_activity': False
            }

        iv = signal.get('implied_volatility', signal.get('iv', None))
        put_call_ratio = signal.get('put_call_ratio', signal.get('pcr', None))
        options_volume = signal.get('options_volume', 0)
        avg_volume = signal.get('options_avg_volume', options_volume)

        # Calculate IV rank if we have data
        iv_rank = signal.get('iv_rank', None)
        if iv_rank is None and iv is not None:
            # Approximate IV rank based on typical IPO IV levels
            # New IPOs typically have high IV (60-150%)
            if iv > 1.0:  # > 100%
                iv_rank = 90
            elif iv > 0.7:
                iv_rank = 75
            elif iv > 0.5:
                iv_rank = 50
            else:
                iv_rank = 25

        # Detect unusual activity
        unusual_activity = False
        if options_volume and avg_volume:
            if options_volume > avg_volume * 2:
                unusual_activity = True

        # Calculate options score
        options_score = 0.5  # Start neutral

        # Adjust for IV (lower IV is better for long positions)
        if iv_rank is not None:
            if iv_rank < 30:
                options_score += 0.2
            elif iv_rank > 70:
                options_score -= 0.1

        # Adjust for put/call ratio (lower = more bullish)
        if put_call_ratio is not None:
            if put_call_ratio < 0.5:
                options_score += 0.15
            elif put_call_ratio > 1.5:
                options_score -= 0.15

        # Unusual activity can go either way
        if unusual_activity:
            # Check if it's bullish or bearish
            call_volume = signal.get('call_volume', 0)
            put_volume = signal.get('put_volume', 0)
            if call_volume > put_volume * 1.5:
                options_score += 0.1  # Bullish flow
            elif put_volume > call_volume * 1.5:
                options_score -= 0.1  # Bearish flow

        signal['options_analysis'] = {
            'has_options': True,
            'options_score': max(0, min(1, options_score)),
            'iv_rank': iv_rank,
            'put_call_ratio': put_call_ratio,
            'unusual_activity': unusual_activity
        }

        return signal['options_analysis']


# ============================================================================
# MAIN US IPO ANALYZER (Combines all components)
# ============================================================================

class USIPOAnalyzer:
    """
    Main US IPO/New Stock Analyzer combining all components.

    Integrates:
    - USNewStockHandler (tiered analysis)
    - USSPACHandler (SPAC-specific)
    - USOptionsIPOAnalyzer (options-enhanced)
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.new_stock_handler = USNewStockHandler()
        self.spac_handler = USSPACHandler()
        self.options_analyzer = USOptionsIPOAnalyzer()

    def analyze(self, signal: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive IPO analysis.

        Args:
            signal: The stock signal
            verbose: Print detailed analysis

        Returns:
            Enhanced signal with all IPO analysis
        """
        # Check if new stock
        if not self.new_stock_handler.is_new_stock(signal):
            signal['is_new_stock'] = False
            signal['ipo_analysis_performed'] = False
            return signal

        signal['is_new_stock'] = True
        signal['ipo_analysis_performed'] = True

        # Get tier info
        can_trade, tier, tier_info = self.new_stock_handler.can_trade(signal)

        if not can_trade:
            signal['trade_allowed'] = False
            signal['rejection_reason'] = tier_info.get('rejection_reason', 'Tier not tradeable')
            return signal

        # Apply tier-specific analysis
        signal = self.new_stock_handler.analyze_new_stock(signal, tier)

        if not signal.get('trade_allowed', True):
            return signal

        # Check for SPAC
        if self.new_stock_handler.is_spac(signal):
            signal['is_spac'] = True
            self.spac_handler.calculate_spac_score(signal)

        # Apply options analysis if available
        if self.options_analyzer.has_options(signal):
            self.options_analyzer.analyze_options(signal)

        # Calculate overall IPO score
        self.new_stock_handler.calculate_ipo_score(signal)

        if verbose:
            print(f"  [US IPO ANALYZER] {signal.get('ticker', 'UNKNOWN')}: "
                  f"tier={tier}, score={signal.get('ipo_score', 0):.2f}, "
                  f"SPAC={signal.get('is_spac', False)}")

        return signal

    def filter_signals(self, signals: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        Filter and analyze all signals.

        Returns:
            List of analyzed and filtered signals
        """
        passed = []

        for signal in signals:
            analyzed = self.analyze(signal, verbose)

            if analyzed.get('trade_allowed', True):
                passed.append(analyzed)

        return passed


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def apply_us_ipo_handler(signals: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    Apply US IPO/New Stock handler to a list of signals.

    This is the main entry point for integrating with the trading system.

    Args:
        signals: List of trading signals
        verbose: Print detailed analysis

    Returns:
        Filtered and enhanced signals
    """
    analyzer = USIPOAnalyzer()
    return analyzer.filter_signals(signals, verbose)
