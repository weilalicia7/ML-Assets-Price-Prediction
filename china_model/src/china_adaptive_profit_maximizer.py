"""
China Adaptive Profit Maximizer with Lag-Free Regime Detection

This module implements the adaptive profit-maximizing strategy for China/DeepSeek model
from 'china model fixing1_final.pdf', 'china model fixing2.pdf', and 'dual model fixing1.pdf'.

Key Features (Fixing1 + Fixing2 + Fixing3):
1. ADAPTIVE REGIME DETECTION
   - BULL: Momentum focus, 95% invested, max 6 positions, 30% cap
   - BEAR: Quality focus, 70% invested, max 4 positions, 20% cap
   - HIGH_VOL: Low vol focus, 50% invested, max 3 positions
   - NEUTRAL: Balanced approach, 85% invested

2. LAG-FREE TRANSITION DETECTION (from US model Fix 62)
   - Early warning signals (higher lows, RSI divergence, volume, SMA breakout)
   - Gradual parameter blending (avoid sudden jumps)
   - Multi-timeframe confirmation
   - Regime probability scoring

3. REGIME-SPECIFIC PROFIT OPTIMIZATION (FIXING2 + FIXING3 adjustments)
   - Bull: Momentum + high-beta + volume expansion + EV >= 0.3 (was 0.75, too strict)
   - Bear: Quality + oversold + EV >= 0.6 (was 1.25, relaxed)
   - High Vol: Low volatility + mean reversion + EV >= 1.0 (was 2.5, relaxed)
   - Neutral: Balanced + EV >= 0.5 (was 1.0, relaxed)

4. DYNAMIC POSITION SIZING (FIXING2)
   - Volatility-adjusted position sizing
   - Correlation penalty for concentrated positions
   - EV-weighted concentration

5. TRAILING STOPS (FIXING2/3)
   - BULL: 15% trailing stop (wider for momentum plays)
   - NEUTRAL: 10% trailing stop
   - BEAR: 8% trailing stop
   - HIGH_VOL: 5% trailing stop

6. QUALITY FILTER (FIXING3: RELAXED + Composite Scoring)
   - Composite quality score (0-1) instead of hard filters
   - EPS > -0.1 (allow slightly negative for tech)
   - D/E < 3.0 (was 1.0, way too strict)
   - Market Cap > $500M (was $1B)
   - Sector-specific adjustments

7. MOMENTUM OVERRIDE RULE (FIXING3 NEW)
   - High momentum (>10% 5-day or >15% 10-day) can override quality filters
   - Regime-weighted entry: BULL=70/30 momentum/quality, BEAR=30/70

8. SHARPE-ADJUSTED EV (FIXING2 NEW)
   - EV_adj = (EV * sqrt(sharpe)) if sharpe > 0 else EV * 0.5

9. DYNAMIC STOP-LOSS (FIXING3: Wider initial stops)
   - BULL: 15% initial stop (was 8%)
   - NEUTRAL: 10% initial stop (was 6%)
   - BEAR: 8% initial stop (was 5%)

10. NEW STOCK/IPO HANDLER (FIXING3 NEW)
    - Tiered analysis based on trading days
    - 1-4 days: No trade (too volatile)
    - 5-10 days: Basic momentum-only analysis
    - 11-30 days: Enhanced analysis
    - 31-60 days: Hybrid analysis
    - 60+: Full analysis

11. SECTOR-SPECIFIC RULES (FIXING3 NEW)
    - Technology: D/E up to 2.5, EPS can be -0.1
    - Financials: D/E up to 15.0
    - Energy: D/E up to 2.0
    - Consumer: stricter D/E < 1.5

Last Updated: 2025-12-20 (Fixing3 Applied - Relaxed Filters + Momentum Override)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChinaRegimeOutput:
    """Container for China regime classification results."""

    # Primary classification
    primary_regime: str          # BULL, BEAR, HIGH_VOL, NEUTRAL
    transition_state: str        # TRANSITION_TO_BULL, TRANSITION_TO_BEAR, STABLE, etc.
    confidence: float            # 0.0 to 1.0

    # Early warning signals (lag-free detection)
    signals_detected: List[str]  # ['higher_lows', 'bullish_divergence', ...]
    transition_confidence: float # Confidence in transition (0.0 to 1.0)

    # Regime parameters
    regime_params: Dict[str, Any]  # Full regime parameters

    # Position sizing guidance
    total_allocation: float      # 0.5 to 0.95
    max_positions: int           # 3 to 6
    position_cap: float          # 0.2 to 0.3
    min_ev_threshold: float      # 0.5 to 2.0


@dataclass
class ChinaProfitGuidance:
    """Profit-maximizing guidance during transitions."""

    transition_type: str
    sector_focus: List[str]
    position_sizing: Dict[str, float]
    risk_adjustments: Dict[str, bool]
    recommended_actions: List[str]


# ============================================================================
# FIXING3: RELAXED QUALITY FILTER WITH COMPOSITE SCORING
# ============================================================================

# Sector-specific quality thresholds (FIXING3)
SECTOR_QUALITY_RULES = {
    'Technology': {'max_de': 2.5, 'min_eps': -0.1, 'min_mcap': 3e8},
    'Financials': {'max_de': 15.0, 'min_eps': 0.0, 'min_mcap': 1e9},
    'Energy': {'max_de': 2.0, 'min_eps': -0.05, 'min_mcap': 5e8},
    'Consumer': {'max_de': 1.5, 'min_eps': 0.0, 'min_mcap': 5e8},
    'Healthcare': {'max_de': 2.0, 'min_eps': -0.1, 'min_mcap': 3e8},
    'Industrials': {'max_de': 2.0, 'min_eps': 0.0, 'min_mcap': 5e8},
    'Materials': {'max_de': 1.8, 'min_eps': 0.0, 'min_mcap': 5e8},
    'Real Estate': {'max_de': 3.0, 'min_eps': 0.0, 'min_mcap': 1e9},
    'Utilities': {'max_de': 2.5, 'min_eps': 0.0, 'min_mcap': 1e9},
    'Default': {'max_de': 3.0, 'min_eps': -0.1, 'min_mcap': 5e8}  # FIXING3: Relaxed defaults
}

# Regime-specific entry weights (FIXING3)
REGIME_ENTRY_WEIGHTS = {
    'BULL': {'momentum': 0.7, 'quality': 0.3},
    'NEUTRAL': {'momentum': 0.5, 'quality': 0.5},
    'BEAR': {'momentum': 0.3, 'quality': 0.7},
    'HIGH_VOL': {'momentum': 0.2, 'quality': 0.8}
}


class ChinaQualityFilter:
    """
    FIXING3: Relaxed quality filter with composite scoring.

    Key changes from FIXING2:
    - Composite quality score (0-1) instead of hard filters
    - Relaxed thresholds: D/E from 1.0 to 3.0, EPS allows -0.1, MCap from $1B to $500M
    - Sector-specific adjustments
    - Momentum override rule for exceptional momentum
    """

    def __init__(self, regime_params: Dict[str, Any] = None, regime: str = 'NEUTRAL'):
        """Initialize quality filter with FIXING3 relaxed thresholds."""
        self.regime = regime

        # FIXING3: Relaxed default thresholds
        if regime_params:
            self.min_eps = regime_params.get('min_eps', -0.1)  # Was 0.0
            self.max_debt_equity = regime_params.get('max_debt_equity', 3.0)  # Was 1.0
            self.min_market_cap = regime_params.get('min_market_cap', 5e8)  # Was 1e9
        else:
            self.min_eps = -0.1
            self.max_debt_equity = 3.0
            self.min_market_cap = 5e8

        # FIXING3: Minimum quality score threshold (soft filter)
        self.min_quality_score = regime_params.get('min_quality_score', 0.3) if regime_params else 0.3

        # FIXING3: Momentum override thresholds
        self.momentum_override_5d = 0.10  # 10% 5-day return overrides quality
        self.momentum_override_10d = 0.15  # 15% 10-day return overrides quality

        # FIXING3: Get regime weights
        self.entry_weights = REGIME_ENTRY_WEIGHTS.get(regime, REGIME_ENTRY_WEIGHTS['NEUTRAL'])

    def calculate_composite_quality_score(self, signal: Dict[str, Any]) -> float:
        """
        FIXING3: Calculate composite quality score (0-1).

        Components:
        - EPS score (0-0.25): >0.1=0.25, >0=0.15, >-0.1=0.05, else=0
        - D/E score (0-0.25): <0.5=0.25, <1.0=0.2, <2.0=0.15, <3.0=0.1, else=0.05
        - Market cap score (0-0.25): >$5B=0.25, >$1B=0.2, >$500M=0.15, else=0.05
        - Profitability score (0-0.25): margin>5%=0.25, >2%=0.15, >0%=0.1
        """
        score = 0.0

        # Get sector-specific thresholds
        sector = signal.get('sector', 'Default')
        sector_rules = SECTOR_QUALITY_RULES.get(sector, SECTOR_QUALITY_RULES['Default'])

        # 1. EPS Score (0-0.25)
        eps = signal.get('eps', signal.get('trailing_eps', 0.0))
        if eps is None:
            eps = 0.0

        if eps > 0.1:
            score += 0.25
        elif eps > 0:
            score += 0.15
        elif eps > -0.1:
            score += 0.05
        # else: 0 for heavily negative EPS

        # 2. Debt-to-Equity Score (0-0.25) - sector adjusted
        debt_equity = signal.get('debt_equity', signal.get('debt_to_equity', 0.0))
        if debt_equity is None:
            debt_equity = 0.0

        # Adjust thresholds by sector
        de_mult = sector_rules['max_de'] / 3.0  # Normalize to default max

        if debt_equity < 0.5 * de_mult:
            score += 0.25
        elif debt_equity < 1.0 * de_mult:
            score += 0.20
        elif debt_equity < 2.0 * de_mult:
            score += 0.15
        elif debt_equity < 3.0 * de_mult:
            score += 0.10
        else:
            score += 0.05  # Still give small score for extreme D/E

        # 3. Market Cap Score (0-0.25)
        market_cap = signal.get('market_cap', signal.get('marketCap', 0))
        if market_cap is None:
            market_cap = 0

        if market_cap > 5e9:
            score += 0.25
        elif market_cap > 1e9:
            score += 0.20
        elif market_cap > 5e8:
            score += 0.15
        elif market_cap > 1e8:
            score += 0.10
        else:
            score += 0.05

        # 4. Profitability Score (0-0.25)
        profit_margin = signal.get('profit_margin', signal.get('profitMargins', 0.0))
        if profit_margin is None:
            profit_margin = 0.0

        if profit_margin > 0.05:
            score += 0.25
        elif profit_margin > 0.02:
            score += 0.15
        elif profit_margin > 0:
            score += 0.10

        return min(score, 1.0)

    def check_momentum_override(self, signal: Dict[str, Any]) -> bool:
        """
        FIXING3: Check if exceptional momentum overrides quality requirements.

        Returns True if momentum is high enough to override quality filters.
        """
        # Get momentum data
        return_5d = signal.get('return_5d', signal.get('momentum_5d', 0.0))
        return_10d = signal.get('return_10d', signal.get('momentum_10d', 0.0))

        if return_5d is None:
            return_5d = 0.0
        if return_10d is None:
            return_10d = 0.0

        # High momentum can override quality
        if return_5d >= self.momentum_override_5d:
            return True
        if return_10d >= self.momentum_override_10d:
            return True

        return False

    def passes_filter(self, signal: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        FIXING3: Check if signal passes quality filter with composite scoring.

        Returns:
            Tuple of (passes, quality_score, reason)
        """
        # Calculate composite quality score
        quality_score = self.calculate_composite_quality_score(signal)
        signal['quality_score'] = quality_score

        # Check momentum override first
        if self.check_momentum_override(signal):
            signal['momentum_override'] = True
            return True, quality_score, 'momentum_override'

        signal['momentum_override'] = False

        # Get sector-specific thresholds
        sector = signal.get('sector', 'Default')
        sector_rules = SECTOR_QUALITY_RULES.get(sector, SECTOR_QUALITY_RULES['Default'])

        # FIXING3: Use soft quality score threshold instead of hard filters
        if quality_score >= self.min_quality_score:
            return True, quality_score, 'quality_passed'

        # Additional hard filter check for extreme cases only
        eps = signal.get('eps', signal.get('trailing_eps', 0.0))
        if eps is None:
            eps = 0.0

        debt_equity = signal.get('debt_equity', signal.get('debt_to_equity', 0.0))
        if debt_equity is None:
            debt_equity = 0.0

        market_cap = signal.get('market_cap', signal.get('marketCap', 0))
        if market_cap is None:
            market_cap = 0

        # Only reject for extreme violations
        if eps < sector_rules['min_eps'] - 0.1:  # Very negative EPS
            return False, quality_score, f'eps_too_low ({eps:.2f})'

        if debt_equity > sector_rules['max_de'] * 1.5:  # Extreme debt
            return False, quality_score, f'de_too_high ({debt_equity:.1f})'

        if market_cap < sector_rules['min_mcap'] * 0.5:  # Very small cap
            return False, quality_score, f'mcap_too_low ({market_cap/1e9:.2f}B)'

        # Pass with low score if not extreme
        return True, quality_score, 'marginal_pass'

    def calculate_weighted_entry_score(self, signal: Dict[str, Any]) -> float:
        """
        FIXING3: Calculate regime-weighted entry score.

        BULL: 70% momentum, 30% quality
        NEUTRAL: 50% momentum, 50% quality
        BEAR: 30% momentum, 70% quality
        HIGH_VOL: 20% momentum, 80% quality
        """
        quality_score = signal.get('quality_score', self.calculate_composite_quality_score(signal))

        # Calculate momentum score (0-1)
        return_2d = signal.get('return_2d', signal.get('momentum_2d', 0.0))
        return_5d = signal.get('return_5d', signal.get('momentum_5d', 0.0))

        if return_2d is None:
            return_2d = 0.0
        if return_5d is None:
            return_5d = 0.0

        # Momentum score: normalize returns to 0-1 scale
        momentum_score = min(1.0, max(0.0, (return_2d * 10 + return_5d * 5) / 2))

        # Apply regime weights
        weights = self.entry_weights
        weighted_score = (
            weights['momentum'] * momentum_score +
            weights['quality'] * quality_score
        )

        signal['weighted_entry_score'] = weighted_score
        signal['momentum_score_raw'] = momentum_score

        return weighted_score

    def filter_signals(self, signals: List[Dict], verbose: bool = False) -> List[Dict]:
        """FIXING3: Filter signals with composite scoring and momentum override."""
        passed = []
        for s in signals:
            passes, quality_score, reason = self.passes_filter(s)

            if passes:
                s['quality_passed'] = True
                s['quality_reason'] = reason

                # Calculate weighted entry score
                self.calculate_weighted_entry_score(s)

                passed.append(s)
            elif verbose:
                print(f"  [QUALITY FILTER] Rejected {s.get('ticker', 'UNKNOWN')}: "
                      f"score={quality_score:.2f}, reason={reason}")

        return passed


# ============================================================================
# FIXING2: MOMENTUM CONFIRMATION
# ============================================================================

class ChinaMomentumConfirmation:
    """
    FIXING2: Momentum confirmation filter.

    Requires 2-day return > 1% for BUY signals (regime-dependent).
    """

    def __init__(self, confirm_days: int = 2, confirm_pct: float = 0.01):
        """
        Initialize momentum confirmation.

        Args:
            confirm_days: Number of days for momentum check (default: 2)
            confirm_pct: Minimum return required (default: 1%)
        """
        self.confirm_days = confirm_days
        self.confirm_pct = confirm_pct

    def confirm_momentum(self, signal: Dict[str, Any]) -> bool:
        """
        Check if signal has positive momentum confirmation.

        Args:
            signal: Dict with price history or momentum data

        Returns:
            True if momentum confirmed
        """
        if self.confirm_days == 0:
            return True  # No confirmation required (e.g., BEAR regime)

        # Check for recent return data
        recent_return = signal.get('return_2d', signal.get('momentum_2d', None))

        if recent_return is None:
            # Try to calculate from price history
            prices = signal.get('price_history', [])
            if len(prices) >= self.confirm_days + 1:
                recent_return = (prices[-1] / prices[-(self.confirm_days + 1)] - 1)
            else:
                # If no data, assume neutral (pass)
                return True

        return recent_return >= self.confirm_pct

    def filter_signals(self, signals: List[Dict], verbose: bool = False) -> List[Dict]:
        """Filter signals with momentum confirmation."""
        passed = []
        for s in signals:
            if self.confirm_momentum(s):
                s['momentum_confirmed'] = True
                passed.append(s)
            elif verbose:
                print(f"  [MOMENTUM FILTER] Rejected {s.get('ticker', 'UNKNOWN')}: "
                      f"2d return < {self.confirm_pct:.1%}")
        return passed


# ============================================================================
# FIXING3: CHINA IPO/NEW STOCK HANDLER
# ============================================================================

class ChinaNewStockHandler:
    """
    FIXING3: Tiered analysis for IPOs and new stocks.

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
        'BASIC': 0.25,    # 25% of normal position
        'ENHANCED': 0.50,  # 50% of normal position
        'HYBRID': 0.75,    # 75% of normal position
        'FULL': 1.0        # Full position
    }

    # Stop-loss multipliers by tier (tighter for newer stocks)
    TIER_STOP_MULTIPLIER = {
        'NO_TRADE': 0.0,
        'BASIC': 0.5,     # 50% of normal stop (tighter)
        'ENHANCED': 0.7,   # 70% of normal stop
        'HYBRID': 0.85,    # 85% of normal stop
        'FULL': 1.0        # Normal stop
    }

    def __init__(self):
        """Initialize the new stock handler."""
        self.ipo_cache = {}

    def get_trading_days(self, signal: Dict[str, Any]) -> int:
        """Get number of trading days since IPO/listing."""
        trading_days = signal.get('trading_days', signal.get('days_listed', 0))

        if trading_days is None or trading_days == 0:
            # Try to calculate from first_trade_date
            first_trade = signal.get('first_trade_date', signal.get('ipo_date'))
            if first_trade:
                from datetime import datetime
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

    def can_trade(self, signal: Dict[str, Any]) -> Tuple[bool, str, Dict]:
        """
        Check if a new stock can be traded and return tier info.

        Returns:
            Tuple of (can_trade, tier_name, tier_adjustments)
        """
        trading_days = self.get_trading_days(signal)
        tier = self.get_tier(trading_days)

        tier_info = {
            'tier': tier,
            'trading_days': trading_days,
            'can_trade': self.TIERS[tier]['can_trade'],
            'position_mult': self.TIER_POSITION_SIZING[tier],
            'stop_mult': self.TIER_STOP_MULTIPLIER[tier],
            'analysis_type': self.TIERS[tier].get('analysis', 'none')
        }

        signal['ipo_tier'] = tier
        signal['ipo_trading_days'] = trading_days
        signal['ipo_position_mult'] = tier_info['position_mult']
        signal['ipo_stop_mult'] = tier_info['stop_mult']

        return tier_info['can_trade'], tier, tier_info

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
            # Momentum-only analysis
            return self._analyze_basic(signal)
        elif tier == 'ENHANCED':
            # Momentum + limited quality
            return self._analyze_enhanced(signal)
        elif tier == 'HYBRID':
            # Full analysis with adjustments
            return self._analyze_hybrid(signal)
        else:  # FULL
            return signal  # No modification needed

    def _analyze_basic(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        BASIC tier: Momentum-only analysis for 5-10 day old stocks.

        - Only consider momentum signals
        - Require strong positive momentum (>2% 2-day)
        - Ignore quality metrics (too early)
        - Very tight stops
        """
        signal['analysis_type'] = 'BASIC_MOMENTUM'

        # Check momentum
        momentum_2d = signal.get('return_2d', signal.get('momentum_2d', 0))
        momentum_5d = signal.get('return_5d', signal.get('momentum_5d', 0))

        if momentum_2d is None:
            momentum_2d = 0
        if momentum_5d is None:
            momentum_5d = 0

        # Require strong momentum for basic tier
        if momentum_2d < 0.02 and momentum_5d < 0.03:
            signal['trade_allowed'] = False
            signal['rejection_reason'] = 'Insufficient momentum for BASIC tier'
            return signal

        # Skip quality checks for basic tier
        signal['quality_passed'] = True
        signal['quality_reason'] = 'basic_tier_skip'
        signal['quality_score'] = 0.5  # Neutral score

        # Calculate IPO momentum score
        signal['ipo_momentum_score'] = min(1.0, (momentum_2d * 20 + momentum_5d * 10) / 2)

        signal['trade_allowed'] = True
        return signal

    def _analyze_enhanced(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED tier: Momentum + limited quality for 11-30 day old stocks.

        - Consider momentum (primary)
        - Limited quality checks (relaxed thresholds)
        - Moderate stops
        """
        signal['analysis_type'] = 'ENHANCED_MIXED'

        # Check momentum (more relaxed than BASIC)
        momentum_2d = signal.get('return_2d', signal.get('momentum_2d', 0))
        if momentum_2d is None:
            momentum_2d = 0

        if momentum_2d < -0.05:  # Only reject strong downtrends
            signal['trade_allowed'] = False
            signal['rejection_reason'] = 'Strong downtrend in ENHANCED tier'
            return signal

        # Limited quality check (very relaxed)
        eps = signal.get('eps', signal.get('trailing_eps', 0))
        if eps is None:
            eps = 0

        # Only reject for severely negative EPS
        if eps < -0.5:
            signal['trade_allowed'] = False
            signal['rejection_reason'] = 'Severely negative EPS'
            return signal

        signal['quality_passed'] = True
        signal['quality_reason'] = 'enhanced_tier_relaxed'
        signal['quality_score'] = 0.6  # Slightly above neutral

        signal['trade_allowed'] = True
        return signal

    def _analyze_hybrid(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        HYBRID tier: Full analysis with adjustments for 31-60 day old stocks.

        - Full quality analysis (slightly relaxed)
        - Full momentum analysis
        - Standard stops with slight adjustment
        """
        signal['analysis_type'] = 'HYBRID_FULL'

        # Full analysis will be applied by main pipeline
        # Just mark as hybrid tier
        signal['trade_allowed'] = True
        return signal

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
                    print(f"  [IPO HANDLER] Rejected {signal.get('ticker', 'UNKNOWN')}: "
                          f"tier={tier}, days={tier_info['trading_days']}")
                continue

            # Apply tier-specific analysis
            analyzed = self.analyze_new_stock(signal, tier)

            if analyzed.get('trade_allowed', False):
                passed.append(analyzed)
            elif verbose:
                print(f"  [IPO HANDLER] Rejected {signal.get('ticker', 'UNKNOWN')}: "
                      f"{analyzed.get('rejection_reason', 'Unknown')}")

        return passed


# ============================================================================
# FIXING2: SHARPE-ADJUSTED EV CALCULATION
# ============================================================================

def calculate_sharpe_adjusted_ev(
    base_ev: float,
    sharpe_ratio: float,
    volatility: float = None,
    risk_free_rate: float = 0.03
) -> float:
    """
    FIXING2: Calculate Sharpe-adjusted Expected Value.

    Formula: EV_adj = EV * sqrt(sharpe) if sharpe > 0 else EV * 0.5

    Args:
        base_ev: Base expected value
        sharpe_ratio: Sharpe ratio of the strategy/stock
        volatility: Optional volatility for additional adjustment
        risk_free_rate: Risk-free rate (default 3%)

    Returns:
        Sharpe-adjusted EV
    """
    if sharpe_ratio is None or np.isnan(sharpe_ratio):
        sharpe_ratio = 0.5  # Assume neutral if unknown

    if sharpe_ratio > 0:
        # Positive Sharpe: boost EV by sqrt(sharpe)
        sharpe_mult = np.sqrt(min(sharpe_ratio, 3.0))  # Cap at sqrt(3) ~= 1.73
        ev_adjusted = base_ev * sharpe_mult
    else:
        # Negative Sharpe: penalize EV
        ev_adjusted = base_ev * 0.5

    # Additional volatility adjustment if provided
    if volatility is not None and volatility > 0:
        # Higher volatility = slight penalty
        vol_penalty = 1.0 / (1.0 + volatility * 5)  # e.g., 5% vol = 0.8x
        ev_adjusted *= vol_penalty

    return ev_adjusted


# ============================================================================
# FIXING2: DYNAMIC STOP-LOSS CALCULATOR
# ============================================================================

class ChinaDynamicStopLoss:
    """
    FIXING2: Dynamic stop-loss based on volatility.

    Formula: stop = base_stop * (1 + vol_percentile)
    - Tighter stops in high volatility
    - Wider stops in low volatility
    """

    @staticmethod
    def calculate_dynamic_stop(
        base_stop: float,
        volatility: float,
        vol_percentile: float = None,
        regime: str = 'NEUTRAL'
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and trailing stop.

        Args:
            base_stop: Base stop-loss percentage (e.g., 0.06 = 6%)
            volatility: Current 20-day volatility
            vol_percentile: Volatility percentile (0-1), calculated if None
            regime: Current regime for trailing stop lookup

        Returns:
            Tuple of (dynamic_stop, trailing_stop)
        """
        # Calculate vol_percentile if not provided
        if vol_percentile is None:
            # Approximate percentile based on typical China market volatility
            # Low vol: < 1.5%, Medium: 1.5-2.5%, High: > 2.5%
            if volatility < 0.015:
                vol_percentile = 0.25
            elif volatility < 0.025:
                vol_percentile = 0.50
            elif volatility < 0.035:
                vol_percentile = 0.75
            else:
                vol_percentile = 1.0

        # Dynamic stop-loss adjustment
        # Higher vol = tighter stop, lower vol = wider stop
        # Formula: base_stop * (1.2 - vol_percentile * 0.4)
        # At low vol (0.25): 1.2 - 0.1 = 1.1 (wider)
        # At high vol (1.0): 1.2 - 0.4 = 0.8 (tighter)
        vol_adjustment = 1.2 - vol_percentile * 0.4
        dynamic_stop = base_stop * vol_adjustment

        # Clamp to reasonable bounds
        dynamic_stop = max(0.02, min(0.15, dynamic_stop))

        # Get trailing stop from regime params (FIXING3: Wider stops)
        trailing_stops = {
            'BULL': 0.15,      # FIXING3: 15% (was 10%)
            'NEUTRAL': 0.10,   # FIXING3: 10% (was 8%)
            'BEAR': 0.08,      # FIXING3: 8% (was 5%)
            'HIGH_VOL': 0.05   # FIXING3: 5% (was 3%)
        }
        trailing_stop = trailing_stops.get(regime, 0.10)

        # Adjust trailing stop by volatility too
        trailing_stop = trailing_stop * vol_adjustment
        trailing_stop = max(0.02, min(0.12, trailing_stop))

        return dynamic_stop, trailing_stop


# ============================================================================
# CHINA ADAPTIVE REGIME DETECTOR (with Lag-Free Detection)
# ============================================================================

class ChinaAdaptiveRegimeDetector:
    """
    Adaptive regime detection for China markets with lag-free transition detection.

    Combines:
    1. China-specific market indicators (HSI, CSI300, CNY)
    2. Early warning signal detection (from US model Fix 62)
    3. Gradual transition blending
    4. Multi-timeframe confirmation
    """

    # Regime definitions with parameters (FIXING3: Relaxed EV + Quality + Wider Stops)
    REGIME_PARAMS = {
        'BULL': {
            'position_size_mult': 1.2,
            'max_positions': 6,
            'stop_loss': 0.15,           # FIXING3: 15% base stop (was 8%, wider for momentum)
            'trailing_stop': 0.15,       # FIXING3: 15% trailing stop (was 10%)
            'take_profit_mult': 2.5,
            'min_ev_threshold': 0.30,    # FIXING3: Relaxed EV (was 0.75, too strict)
            'heat_threshold': 0.15,
            'exit_days_no_profit': 10,
            'total_allocation': 0.95,     # 95% invested
            'position_cap': 0.30,         # Max 30% per position
            'rsi_buy_boost': 1.2,
            # FIXING3: Relaxed quality filter thresholds
            'min_eps': -0.1,              # Allow slightly negative (was 0.0)
            'max_debt_equity': 3.0,       # FIXING3: Relaxed D/E (was 1.0)
            'min_market_cap': 5e8,        # FIXING3: Relaxed MCap (was 1e9)
            'min_quality_score': 0.3,     # FIXING3: Composite score threshold
            # FIXING3: Relaxed momentum confirmation
            'momentum_confirm_days': 2,
            'momentum_confirm_pct': 0.005, # FIXING3: 0.5% (was 1%)
        },
        'BEAR': {
            'position_size_mult': 0.7,
            'max_positions': 4,
            'stop_loss': 0.08,           # FIXING3: 8% base stop (was 5%)
            'trailing_stop': 0.08,       # FIXING3: 8% trailing stop (was 5%)
            'take_profit_mult': 1.5,
            'min_ev_threshold': 0.60,    # FIXING3: Relaxed EV (was 1.25)
            'heat_threshold': 0.08,
            'exit_days_no_profit': 5,
            'total_allocation': 0.70,     # 70% invested
            'position_cap': 0.20,         # Max 20% per position
            'rsi_buy_boost': 0.8,
            # FIXING3: Relaxed quality filter for bear markets
            'min_eps': 0.0,               # Profitable (was 0.5)
            'max_debt_equity': 1.5,       # FIXING3: Relaxed (was 0.7)
            'min_market_cap': 1e9,        # Market cap $1B (was 2e9)
            'min_quality_score': 0.4,     # FIXING3: Higher quality score in bear
            # FIXING3: No momentum confirmation in bear (oversold focus)
            'momentum_confirm_days': 0,
            'momentum_confirm_pct': 0.0,
        },
        'HIGH_VOL': {
            'position_size_mult': 0.5,
            'max_positions': 3,
            'stop_loss': 0.06,           # FIXING3: 6% base stop (was 4%)
            'trailing_stop': 0.05,       # FIXING3: 5% trailing stop (was 3%)
            'take_profit_mult': 1.2,
            'min_ev_threshold': 1.0,     # FIXING3: Relaxed EV (was 2.5)
            'heat_threshold': 0.05,
            'exit_days_no_profit': 3,
            'total_allocation': 0.50,     # 50% invested
            'position_cap': 0.15,         # Max 15% per position
            'rsi_buy_boost': 0.6,
            # FIXING3: Relaxed quality filter for high vol
            'min_eps': 0.0,               # FIXING3: Relaxed (was 1.0)
            'max_debt_equity': 2.0,       # FIXING3: Relaxed (was 0.5)
            'min_market_cap': 1e9,        # FIXING3: Relaxed (was 5e9)
            'min_quality_score': 0.5,     # FIXING3: Higher quality score in high vol
            # FIXING3: No momentum confirmation in high vol
            'momentum_confirm_days': 0,
            'momentum_confirm_pct': 0.0,
        },
        'NEUTRAL': {
            'position_size_mult': 1.0,
            'max_positions': 5,
            'stop_loss': 0.10,           # FIXING3: 10% base stop (was 6%)
            'trailing_stop': 0.10,       # FIXING3: 10% trailing stop (was 8%)
            'take_profit_mult': 2.0,
            'min_ev_threshold': 0.50,    # FIXING3: Relaxed EV (was 1.0)
            'heat_threshold': 0.10,
            'exit_days_no_profit': 7,
            'total_allocation': 0.85,     # 85% invested
            'position_cap': 0.25,         # Max 25% per position
            'rsi_buy_boost': 1.0,
            # FIXING3: Relaxed quality filter
            'min_eps': -0.1,             # FIXING3: Allow slightly negative
            'max_debt_equity': 3.0,      # FIXING3: Relaxed (was 1.0)
            'min_market_cap': 5e8,       # FIXING3: Relaxed (was 1e9)
            'min_quality_score': 0.3,    # FIXING3: Composite score threshold
            # FIXING3: Relaxed momentum confirmation
            'momentum_confirm_days': 2,
            'momentum_confirm_pct': 0.003, # FIXING3: 0.3% (was 0.5%)
        }
    }

    # Transition thresholds (from PDF)
    TRANSITION_THRESHOLDS = {
        'bear_to_bull': {
            'minimum_signals': 3,          # At least 3 of 5 signals
            'confidence_threshold': 0.6,
            'confirmation_period': 3,      # Days to confirm
            'gradual_adjustment': True,    # Don't flip overnight
        },
        'bull_to_bear': {
            'minimum_signals': 2,          # More sensitive to downturns
            'confidence_threshold': 0.5,
            'confirmation_period': 2,
            'gradual_adjustment': False,   # Be faster to defend
        }
    }

    # Blending speeds (from PDF)
    BLENDING_SPEEDS = {
        'bear_to_bull': 0.3,    # 30% per day adjustment (patient)
        'bull_to_bear': 0.5,    # 50% per day adjustment (defend faster)
        'neutral_shifts': 0.4,  # 40% per day
    }

    # Early warning signal weights (from US model Fix 62)
    EARLY_SIGNALS_WEIGHTS = {
        'higher_lows': 0.25,
        'bullish_divergence': 0.25,
        'volume_expansion': 0.20,
        'sma_breakout': 0.20,
        'momentum_acceleration': 0.10
    }

    def __init__(self):
        """Initialize the adaptive regime detector."""
        self.current_regime = 'NEUTRAL'
        self.regime_history = deque(maxlen=20)
        self.transition_history = deque(maxlen=10)
        self.blended_params = self.REGIME_PARAMS['NEUTRAL'].copy()
        self.days_in_transition = 0
        self.pending_transition = None
        self.transition_start_params = None

    # =========================================================================
    # LAG-FREE EARLY WARNING SIGNAL DETECTION
    # =========================================================================

    def detect_early_signals(self, market_data: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        Detect early transition signals before traditional indicators.

        This is adapted from the US model Fix 62 LagFreeRegimeClassifier.

        Args:
            market_data: DataFrame with OHLCV data (HSI or CSI300 proxy)

        Returns:
            Tuple of (transition_type, confidence, signals_detected)
        """
        if len(market_data) < 30:
            return 'NO_TRANSITION', 0.0, []

        signals = []
        signal_scores = []

        # 1. Higher Lows Pattern (25% weight)
        try:
            lows_5d = market_data['Low'].rolling(5).min()
            if len(lows_5d) >= 15:
                low_now = float(lows_5d.iloc[-1])
                low_6d = float(lows_5d.iloc[-6])
                low_11d = float(lows_5d.iloc[-11])

                if not any(pd.isna([low_now, low_6d, low_11d])):
                    if low_now > low_6d > low_11d:
                        signals.append('higher_lows')
                        signal_scores.append(self.EARLY_SIGNALS_WEIGHTS['higher_lows'])
        except Exception:
            pass

        # 2. RSI Bullish Divergence (25% weight)
        try:
            rsi = self._calculate_rsi(market_data['Close'], 14)
            if len(rsi) >= 15:
                rsi_now = float(rsi.iloc[-1])
                rsi_10d = float(rsi.iloc[-10])
                price_now = float(market_data['Close'].iloc[-1])
                price_10d = float(market_data['Close'].iloc[-10])

                if not any(pd.isna([rsi_now, rsi_10d, price_now, price_10d])):
                    price_declining = price_now < price_10d
                    rsi_rising = rsi_now > rsi_10d + 3
                    if price_declining and rsi_rising:
                        signals.append('bullish_divergence')
                        signal_scores.append(self.EARLY_SIGNALS_WEIGHTS['bullish_divergence'])
        except Exception:
            pass

        # 3. Volume Expansion on Up Days (20% weight)
        if 'Volume' in market_data.columns:
            try:
                data_copy = market_data.copy()
                data_copy['Up_Day'] = data_copy['Close'] > data_copy['Open']
                up_days = data_copy[data_copy['Up_Day']]
                down_days = data_copy[~data_copy['Up_Day']]

                if len(up_days) >= 5 and len(down_days) >= 5:
                    up_volume = float(up_days['Volume'].tail(5).mean())
                    down_volume = float(down_days['Volume'].tail(5).mean())
                    if down_volume > 0 and up_volume > down_volume * 1.3:
                        signals.append('volume_expansion')
                        signal_scores.append(self.EARLY_SIGNALS_WEIGHTS['volume_expansion'])
            except Exception:
                pass

        # 4. SMA Breakout (20% weight)
        try:
            sma_20 = market_data['Close'].rolling(20).mean()
            if len(sma_20) >= 21:
                current_price = float(market_data['Close'].iloc[-1])
                prev_price = float(market_data['Close'].iloc[-2])
                current_sma = float(sma_20.iloc[-1])
                prev_sma = float(sma_20.iloc[-2])

                if not any(pd.isna([current_price, prev_price, current_sma, prev_sma])):
                    if current_price > current_sma and prev_price <= prev_sma:
                        signals.append('sma_breakout')
                        signal_scores.append(self.EARLY_SIGNALS_WEIGHTS['sma_breakout'])
        except Exception:
            pass

        # 5. Momentum Acceleration (10% weight)
        if len(market_data) >= 11:
            try:
                returns_5d = float(market_data['Close'].pct_change(5).iloc[-1])
                returns_10d = float(market_data['Close'].pct_change(10).iloc[-1])

                if not pd.isna(returns_5d) and not pd.isna(returns_10d):
                    if returns_5d > 0.02 and returns_5d > returns_10d:
                        signals.append('momentum_acceleration')
                        signal_scores.append(self.EARLY_SIGNALS_WEIGHTS['momentum_acceleration'])
            except Exception:
                pass

        # Calculate total confidence
        confidence = sum(signal_scores)

        # Determine transition type
        if confidence >= 0.75:
            return 'CONFIRMED_TRANSITION_TO_BULL', confidence, signals
        elif confidence >= 0.5:
            return 'EARLY_BULL_TRANSITION', confidence, signals
        elif confidence >= 0.25:
            return 'EARLY_BULL_SIGNAL', confidence, signals
        else:
            # Check for bear signals (inverse logic)
            bear_signals = self._detect_bear_signals(market_data)
            if bear_signals['confidence'] >= 0.5:
                return 'TRANSITION_TO_BEAR', bear_signals['confidence'], bear_signals['signals']
            return 'NO_TRANSITION', confidence, signals

    def _detect_bear_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect bearish transition signals."""
        signals = []
        confidence = 0.0

        try:
            # 1. Lower highs pattern
            highs_5d = market_data['High'].rolling(5).max()
            if len(highs_5d) >= 15:
                high_now = float(highs_5d.iloc[-1])
                high_6d = float(highs_5d.iloc[-6])
                high_11d = float(highs_5d.iloc[-11])

                if not any(pd.isna([high_now, high_6d, high_11d])):
                    if high_now < high_6d < high_11d:
                        signals.append('lower_highs')
                        confidence += 0.25

            # 2. Price below SMA 20
            sma_20 = market_data['Close'].rolling(20).mean()
            if len(sma_20) >= 21:
                current_price = float(market_data['Close'].iloc[-1])
                current_sma = float(sma_20.iloc[-1])
                if not pd.isna(current_sma) and current_price < current_sma * 0.97:
                    signals.append('below_sma20')
                    confidence += 0.25

            # 3. Negative momentum
            returns_10d = float(market_data['Close'].pct_change(10).iloc[-1])
            if not pd.isna(returns_10d) and returns_10d < -0.05:
                signals.append('negative_momentum')
                confidence += 0.25

            # 4. Volume on down days
            if 'Volume' in market_data.columns:
                data_copy = market_data.copy()
                data_copy['Down_Day'] = data_copy['Close'] < data_copy['Open']
                down_days = data_copy[data_copy['Down_Day']]
                up_days = data_copy[~data_copy['Down_Day']]

                if len(down_days) >= 5 and len(up_days) >= 5:
                    down_volume = float(down_days['Volume'].tail(5).mean())
                    up_volume = float(up_days['Volume'].tail(5).mean())
                    if up_volume > 0 and down_volume > up_volume * 1.3:
                        signals.append('volume_on_down')
                        confidence += 0.25

        except Exception:
            pass

        return {'signals': signals, 'confidence': min(confidence, 1.0)}

    # =========================================================================
    # REGIME CLASSIFICATION WITH GRADUAL BLENDING
    # =========================================================================

    def classify_regime(
        self,
        market_data: pd.DataFrame,
        volatility_20d: float = 0.02,
        hsi_return_20d: float = 0.0,
        hsi_return_5d: float = 0.0
    ) -> ChinaRegimeOutput:
        """
        Classify current China market regime with lag-free transition detection.

        Args:
            market_data: DataFrame with OHLCV data (individual stock or HSI proxy)
            volatility_20d: 20-day realized volatility
            hsi_return_20d: 20-day HSI returns
            hsi_return_5d: 5-day HSI returns

        Returns:
            ChinaRegimeOutput with full regime classification
        """
        # Step 1: Detect early warning signals (lag-free)
        transition_type, transition_conf, signals = self.detect_early_signals(market_data)

        # Step 2: Determine base regime from traditional indicators
        base_regime = self._classify_base_regime(volatility_20d, hsi_return_20d, hsi_return_5d)

        # Step 3: Handle transition with confirmation period
        final_regime = self._handle_transition(
            base_regime=base_regime,
            transition_type=transition_type,
            transition_confidence=transition_conf,
            signals=signals
        )

        # Step 4: Get blended parameters (gradual transition)
        regime_params = self._get_blended_params(final_regime, transition_type, transition_conf)

        # Update history
        self.regime_history.append(final_regime)
        self.current_regime = final_regime

        return ChinaRegimeOutput(
            primary_regime=final_regime,
            transition_state=transition_type,
            confidence=transition_conf,
            signals_detected=signals,
            transition_confidence=transition_conf,
            regime_params=regime_params,
            total_allocation=regime_params['total_allocation'],
            max_positions=regime_params['max_positions'],
            position_cap=regime_params['position_cap'],
            min_ev_threshold=regime_params['min_ev_threshold']
        )

    def _classify_base_regime(
        self,
        volatility_20d: float,
        hsi_return_20d: float,
        hsi_return_5d: float
    ) -> str:
        """Classify base regime from traditional indicators."""

        # High volatility override
        if volatility_20d > 0.03:  # > 3% daily vol = high volatility
            return 'HIGH_VOL'

        # Trend-based classification with adaptive thresholds
        # Use tighter thresholds in low vol, wider in high vol
        threshold = 0.05 if volatility_20d < 0.02 else 0.07

        if hsi_return_20d > threshold:
            if hsi_return_5d > threshold * 0.4:
                return 'BULL'
            else:
                return 'NEUTRAL'  # Bull but consolidating
        elif hsi_return_20d < -threshold:
            if hsi_return_5d < -threshold * 0.4:
                return 'BEAR'
            else:
                return 'NEUTRAL'  # Bear but rallying
        else:
            return 'NEUTRAL'

    def _handle_transition(
        self,
        base_regime: str,
        transition_type: str,
        transition_confidence: float,
        signals: List[str]
    ) -> str:
        """
        Handle regime transition with confirmation period.

        This prevents whipsaws by requiring confirmation before switching regimes.
        """
        # Check if we're in a pending transition
        if self.pending_transition is not None:
            self.days_in_transition += 1

            # Check if transition is confirmed
            if 'BULL' in transition_type and self.pending_transition == 'BULL':
                threshold = self.TRANSITION_THRESHOLDS['bear_to_bull']
                if (self.days_in_transition >= threshold['confirmation_period'] and
                    transition_confidence >= threshold['confidence_threshold'] and
                    len(signals) >= threshold['minimum_signals']):
                    # Transition confirmed
                    self.pending_transition = None
                    self.days_in_transition = 0
                    self.transition_history.append(('bear_to_bull', datetime.now()))
                    return 'BULL'

            elif 'BEAR' in transition_type and self.pending_transition == 'BEAR':
                threshold = self.TRANSITION_THRESHOLDS['bull_to_bear']
                if (self.days_in_transition >= threshold['confirmation_period'] and
                    transition_confidence >= threshold['confidence_threshold'] and
                    len(signals) >= threshold['minimum_signals']):
                    # Transition confirmed (faster for bear)
                    self.pending_transition = None
                    self.days_in_transition = 0
                    self.transition_history.append(('bull_to_bear', datetime.now()))
                    return 'BEAR'

            # Transition not yet confirmed, use current regime with blending
            return self.current_regime

        # Check for new potential transition
        if 'BULL' in transition_type and self.current_regime in ['BEAR', 'NEUTRAL', 'HIGH_VOL']:
            self.pending_transition = 'BULL'
            self.days_in_transition = 1
            self.transition_start_params = self.blended_params.copy()
            return self.current_regime  # Don't switch yet

        elif 'BEAR' in transition_type and self.current_regime in ['BULL', 'NEUTRAL']:
            self.pending_transition = 'BEAR'
            self.days_in_transition = 1
            self.transition_start_params = self.blended_params.copy()
            # For bear transitions, be faster
            if transition_confidence >= 0.5:
                self.transition_history.append(('bull_to_bear', datetime.now()))
                return 'BEAR'
            return self.current_regime

        return base_regime

    def _get_blended_params(
        self,
        target_regime: str,
        transition_type: str,
        transition_confidence: float
    ) -> Dict[str, Any]:
        """
        Get blended parameters during transitions.

        This provides smooth transitions instead of sudden parameter changes.
        """
        target_params = self.REGIME_PARAMS[target_regime]

        # If no transition in progress, use target params directly
        if self.pending_transition is None:
            self.blended_params = target_params.copy()
            return target_params.copy()

        # Determine blending speed
        if 'BULL' in transition_type:
            blend_speed = self.BLENDING_SPEEDS['bear_to_bull']
        elif 'BEAR' in transition_type:
            blend_speed = self.BLENDING_SPEEDS['bull_to_bear']
        else:
            blend_speed = self.BLENDING_SPEEDS['neutral_shifts']

        # Calculate blend factor based on days in transition
        blend_factor = min(self.days_in_transition * blend_speed, 1.0)

        # Adjust for transition confidence
        blend_factor *= transition_confidence

        # Blend parameters
        start_params = self.transition_start_params or self.blended_params
        blended = {}

        for key in target_params:
            if isinstance(target_params[key], (int, float)):
                start_val = start_params.get(key, target_params[key])
                blended[key] = start_val + (target_params[key] - start_val) * blend_factor
                # Keep int types as int
                if isinstance(target_params[key], int):
                    blended[key] = int(round(blended[key]))
            else:
                blended[key] = target_params[key]

        self.blended_params = blended
        return blended

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# ============================================================================
# PROFIT MAXIMIZATION LAYER (FIXING3 ENHANCED)
# ============================================================================

class ChinaProfitMaximizationLayer:
    """
    Applies profit-maximizing logic appropriate for each regime.

    FIXING3 Enhancements (Relaxed from FIXING2):
    - Relaxed EV thresholds (BULL: 0.3, BEAR: 0.6, HIGH_VOL: 1.0, NEUTRAL: 0.5)
    - Composite quality scoring (0-1) instead of hard filters
    - Momentum override for exceptional momentum (>10% 5-day or >15% 10-day)
    - Regime-weighted entry (BULL=70% momentum, BEAR=70% quality)
    - New stock/IPO handler with tiered analysis
    - Sector-specific quality rules
    - Sharpe-adjusted EV calculation
    - Dynamic stop-loss based on volatility
    """

    def __init__(self):
        """Initialize profit maximization layer with fixing3 enhancements."""
        self.quality_filter = None
        self.momentum_filter = None
        self.ipo_handler = ChinaNewStockHandler()  # FIXING3: IPO handler

    def optimize_for_regime(
        self,
        signals: List[Dict],
        regime_params: Dict[str, Any],
        regime_info: ChinaRegimeOutput,
        apply_fixing2: bool = True  # Now applies FIXING3 as well
    ) -> List[Dict]:
        """
        Apply regime-specific optimization to signals.

        FIXING3: Added IPO handler, composite quality scoring, momentum override.

        Args:
            signals: List of signal dicts with 'ticker', 'ev', 'confidence', etc.
            regime_params: Current regime parameters
            regime_info: Full regime classification output
            apply_fixing2: Whether to apply fixing2/3 enhancements (default: True)

        Returns:
            Optimized and filtered signals
        """
        regime = regime_info.primary_regime

        # FIXING3: Apply IPO/New Stock Handler first
        if apply_fixing2:
            signals = self.ipo_handler.filter_new_stocks(signals)

        # FIXING3: Apply quality filter with regime and composite scoring
        if apply_fixing2:
            self.quality_filter = ChinaQualityFilter(regime_params, regime=regime)
            signals = self.quality_filter.filter_signals(signals)

            # FIXING3: Apply momentum confirmation (relaxed thresholds)
            momentum_days = regime_params.get('momentum_confirm_days', 2)
            momentum_pct = regime_params.get('momentum_confirm_pct', 0.005)  # FIXING3: 0.5% default
            if momentum_days > 0:
                self.momentum_filter = ChinaMomentumConfirmation(momentum_days, momentum_pct)
                signals = self.momentum_filter.filter_signals(signals)

            # FIXING2/3: Apply Sharpe-adjusted EV
            for s in signals:
                base_ev = s.get('ev', 1.0)
                sharpe = s.get('sharpe_ratio', s.get('sharpe', 0.5))
                volatility = s.get('volatility', 0.02)
                s['ev_original'] = base_ev
                s['ev'] = calculate_sharpe_adjusted_ev(base_ev, sharpe, volatility)

        if regime == 'BULL':
            return self._optimize_bull(signals, regime_params, apply_fixing2)
        elif regime == 'BEAR':
            return self._optimize_bear(signals, regime_params, apply_fixing2)
        elif regime == 'HIGH_VOL':
            return self._optimize_high_vol(signals, regime_params, apply_fixing2)
        else:  # NEUTRAL
            return self._optimize_neutral(signals, regime_params, apply_fixing2)

    def _optimize_bull(self, signals: List[Dict], params: Dict, apply_fixing2: bool = True) -> List[Dict]:
        """
        BULL MARKET: Maximize upside capture.

        Focus on:
        - Momentum and breakouts
        - High-beta stocks
        - Volume expansion
        - FIXING3: EV >= 0.3 (relaxed from 0.75)
        """
        # 1. Score momentum
        for s in signals:
            s['momentum_score'] = s.get('momentum', 0) + s.get('rsi_score', 0.5)

        # 2. Score beta exposure (prefer higher beta in bull)
        for s in signals:
            beta = s.get('beta', 1.0)
            s['beta_score'] = min(beta / 1.5, 1.5)  # Target beta 1.5

        # 3. Score volume
        for s in signals:
            vol_ratio = s.get('volume_ratio', 1.0)
            s['volume_score'] = 1.0 if vol_ratio >= 1.5 else vol_ratio / 1.5

        # 4. Apply EV filter (FIXING3: relaxed threshold 0.3)
        min_ev = params.get('min_ev_threshold', 0.30)
        signals = [s for s in signals if s.get('ev', 0) >= min_ev]

        # 5. Sort by momentum × volume × EV
        signals.sort(key=lambda x: (
            x.get('momentum_score', 1) *
            x.get('volume_score', 1) *
            x.get('ev', 1)
        ), reverse=True)

        # FIXING2: Add dynamic stop-loss to each signal
        if apply_fixing2:
            for s in signals:
                volatility = s.get('volatility', 0.02)
                base_stop = params.get('stop_loss', 0.08)
                dynamic_stop, trailing_stop = ChinaDynamicStopLoss.calculate_dynamic_stop(
                    base_stop, volatility, regime='BULL'
                )
                s['dynamic_stop_loss'] = dynamic_stop
                s['trailing_stop'] = trailing_stop

        return signals[:params.get('max_positions', 6)]

    def _optimize_bear(self, signals: List[Dict], params: Dict, apply_fixing2: bool = True) -> List[Dict]:
        """
        BEAR MARKET: Preserve capital, selective opportunities.

        Focus on:
        - Quality stocks (strong fundamentals)
        - Oversold conditions
        - FIXING3: EV >= 0.6 (relaxed from 1.25)
        """
        # 1. Score quality (FIXING3: use composite quality score if available)
        for s in signals:
            # Use actual EPS and D/E if available
            eps = s.get('eps', s.get('trailing_eps', 0.5))
            if eps is None:
                eps = 0.5
            eps_score = min(eps / 2.0, 1.0) if eps > 0 else 0.3

            debt_equity = s.get('debt_equity', s.get('debt_to_equity', 0.5))
            if debt_equity is None:
                debt_equity = 0.5
            de_score = 1.0 - min(debt_equity, 1.0)

            earnings_stability = s.get('earnings_stability', 0.5)

            quality_factors = [eps_score, de_score, earnings_stability]
            s['quality_score'] = sum(quality_factors) / len(quality_factors)

        # 2. Score oversold conditions
        for s in signals:
            rsi = s.get('rsi', 50)
            if rsi < 30:
                s['oversold_score'] = 1.5
            elif rsi < 40:
                s['oversold_score'] = 1.2
            else:
                s['oversold_score'] = 1.0

        # 3. Require volume confirmation
        signals = [s for s in signals if s.get('volume_ratio', 0) >= 1.0]

        # 4. EV filter (FIXING3: relaxed to 0.6)
        min_ev = params.get('min_ev_threshold', 0.60)
        signals = [s for s in signals if s.get('ev', 0) >= min_ev]

        # 5. Sort by EV × quality × oversold
        signals.sort(key=lambda x: (
            x.get('ev', 1) *
            x.get('quality_score', 1) *
            x.get('oversold_score', 1)
        ), reverse=True)

        # FIXING2: Add dynamic stop-loss to each signal
        if apply_fixing2:
            for s in signals:
                volatility = s.get('volatility', 0.02)
                base_stop = params.get('stop_loss', 0.05)
                dynamic_stop, trailing_stop = ChinaDynamicStopLoss.calculate_dynamic_stop(
                    base_stop, volatility, regime='BEAR'
                )
                s['dynamic_stop_loss'] = dynamic_stop
                s['trailing_stop'] = trailing_stop

        return signals[:params.get('max_positions', 4)]

    def _optimize_high_vol(self, signals: List[Dict], params: Dict, apply_fixing2: bool = True) -> List[Dict]:
        """
        HIGH VOLATILITY: Risk-off, very selective.

        Focus on:
        - Low volatility stocks
        - Mean reversion setups
        - FIXING3: EV >= 1.0 (relaxed from 2.5)
        """
        # 1. Score low volatility (prefer lower vol stocks)
        for s in signals:
            stock_vol = s.get('volatility', 0.02)
            s['low_vol_score'] = 1.0 / (1.0 + stock_vol * 20)

        # 2. Score mean reversion
        for s in signals:
            # Look for oversold + quality
            rsi = s.get('rsi', 50)
            distance_from_mean = s.get('distance_from_sma', 0)
            if rsi < 35 and distance_from_mean < -0.05:
                s['mean_reversion_score'] = 1.5
            elif rsi < 45 and distance_from_mean < -0.03:
                s['mean_reversion_score'] = 1.2
            else:
                s['mean_reversion_score'] = 1.0

        # 3. EV filter (FIXING3: relaxed to 1.0)
        min_ev = params.get('min_ev_threshold', 1.0)
        signals = [s for s in signals if s.get('ev', 0) >= min_ev]

        # 4. Additional safety filters
        signals = [s for s in signals if s.get('beta', 1.0) < 1.2]
        signals = [s for s in signals if s.get('volume_ratio', 0) > 0.8]

        # 5. Sort by low_vol × mean_reversion × EV
        signals.sort(key=lambda x: (
            x.get('low_vol_score', 1) *
            x.get('mean_reversion_score', 1) *
            x.get('ev', 1)
        ), reverse=True)

        # FIXING2: Add dynamic stop-loss to each signal
        if apply_fixing2:
            for s in signals:
                volatility = s.get('volatility', 0.02)
                base_stop = params.get('stop_loss', 0.04)
                dynamic_stop, trailing_stop = ChinaDynamicStopLoss.calculate_dynamic_stop(
                    base_stop, volatility, regime='HIGH_VOL'
                )
                s['dynamic_stop_loss'] = dynamic_stop
                s['trailing_stop'] = trailing_stop

        return signals[:params.get('max_positions', 3)]

    def _optimize_neutral(self, signals: List[Dict], params: Dict, apply_fixing2: bool = True) -> List[Dict]:
        """
        NEUTRAL MARKET: Balanced approach.
        FIXING3: EV >= 0.5 (relaxed from 1.0)
        """
        # Apply EV filter (FIXING3: relaxed threshold 0.5)
        min_ev = params.get('min_ev_threshold', 0.50)
        signals = [s for s in signals if s.get('ev', 0) >= min_ev]

        # Sort by EV primarily
        signals.sort(key=lambda x: x.get('ev', 0), reverse=True)

        # FIXING2: Add dynamic stop-loss to each signal
        if apply_fixing2:
            for s in signals:
                volatility = s.get('volatility', 0.02)
                base_stop = params.get('stop_loss', 0.06)
                dynamic_stop, trailing_stop = ChinaDynamicStopLoss.calculate_dynamic_stop(
                    base_stop, volatility, regime='NEUTRAL'
                )
                s['dynamic_stop_loss'] = dynamic_stop
                s['trailing_stop'] = trailing_stop

        return signals[:params.get('max_positions', 5)]


# ============================================================================
# DYNAMIC POSITION SIZER (FIXING2 ENHANCED)
# ============================================================================

class ChinaDynamicPositionSizer:
    """
    Dynamic position sizing based on regime and transition state.

    FIXING2 Enhancements:
    - Volatility-adjusted position sizing
    - Correlation penalty for concentrated positions
    - Dynamic exit days based on regime

    From PDF:
    - BULL: 95% invested, max 30% per position
    - BEAR: 70% invested, max 20% per position
    - HIGH_VOL: 50% invested, max 15% per position
    - NEUTRAL: 85% invested, max 25% per position
    """

    # FIXING2: Dynamic exit days based on regime
    EXIT_DAYS = {
        'BULL': 10,      # Longer holding in uptrend
        'NEUTRAL': 7,    # Moderate holding
        'BEAR': 5,       # Shorter holding in downtrend
        'HIGH_VOL': 3    # Quick exit in high volatility
    }

    def size_positions(
        self,
        signals: List[Dict],
        regime_params: Dict[str, Any],
        regime_info: ChinaRegimeOutput,
        capital: float = 100000,
        apply_fixing2: bool = True
    ) -> List[Dict]:
        """
        Calculate position sizes based on regime and signal quality.

        FIXING2 Enhancements:
        - Volatility adjustment: reduce size for high-vol stocks
        - Correlation penalty: reduce size if correlated with existing positions
        - Dynamic stops: use calculated dynamic stop-loss

        Args:
            signals: Optimized signals from profit maximization layer
            regime_params: Current regime parameters
            regime_info: Full regime classification
            capital: Available capital
            apply_fixing2: Whether to apply fixing2 enhancements

        Returns:
            Signals with position sizing added
        """
        if not signals:
            return []

        positions = []

        base_multiplier = regime_params.get('position_size_mult', 1.0)
        max_positions = regime_params.get('max_positions', 5)
        total_allocation = regime_params.get('total_allocation', 0.85)
        position_cap = regime_params.get('position_cap', 0.25)

        # Adjust for transition confidence
        if regime_info.transition_confidence > 0.6:
            transition_multiplier = 1 + (regime_info.transition_confidence * 0.5)
        else:
            transition_multiplier = 1.0

        # Calculate allocations
        signals_to_process = signals[:max_positions]

        # FIXING2: Calculate sector/correlation penalty
        sector_counts = {}
        for s in signals_to_process:
            sector = s.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        for i, signal in enumerate(signals_to_process):
            # Base allocation
            base_size = total_allocation / len(signals_to_process)

            # Adjust for signal quality
            quality_mult = signal.get('confidence', 0.5) * 2  # 0-1 to 0-2

            # Adjust for EV
            ev = signal.get('ev', 1.0)
            ev_mult = min(ev / 5.0, 2.0)  # Cap at 2x

            # FIXING2: Volatility adjustment
            # Higher volatility = smaller position
            if apply_fixing2:
                stock_volatility = signal.get('volatility', 0.02)
                # Penalty for high-vol stocks: 1.0 at 2% vol, 0.5 at 5% vol
                vol_adjustment = max(0.5, 1.0 - (stock_volatility - 0.02) * 10)
            else:
                vol_adjustment = 1.0

            # FIXING2: Correlation/sector penalty
            # Reduce size if multiple stocks in same sector
            if apply_fixing2:
                sector = signal.get('sector', 'Unknown')
                sector_count = sector_counts.get(sector, 1)
                if sector_count > 1:
                    correlation_penalty = 1.0 / np.sqrt(sector_count)  # e.g., 2 stocks = 0.71x
                else:
                    correlation_penalty = 1.0
            else:
                correlation_penalty = 1.0

            # Calculate final size with FIXING2 adjustments
            final_size = (base_size * quality_mult * ev_mult * base_multiplier *
                         transition_multiplier * vol_adjustment * correlation_penalty)

            # Apply cap
            final_size = min(final_size, position_cap)

            # Ensure minimum position size (2% floor)
            final_size = max(final_size, 0.02)

            # Calculate shares/amount
            position_value = capital * final_size

            # FIXING2: Use dynamic stop-loss from signal if available
            if apply_fixing2 and 'dynamic_stop_loss' in signal:
                stop_loss = signal['dynamic_stop_loss']
                trailing_stop = signal.get('trailing_stop', regime_params.get('trailing_stop', 0.08))
            else:
                stop_loss = regime_params.get('stop_loss', 0.06)
                trailing_stop = regime_params.get('trailing_stop', 0.08)

            # FIXING2: Dynamic exit days
            regime = regime_info.primary_regime
            exit_days = self.EXIT_DAYS.get(regime, 7)

            positions.append({
                'ticker': signal.get('ticker', f'SIGNAL_{i}'),
                'allocation': final_size,
                'position_value': position_value,
                'stop_loss': stop_loss,
                'trailing_stop': trailing_stop,           # FIXING2: New field
                'take_profit_mult': regime_params.get('take_profit_mult', 2.0),
                'max_days': exit_days,                    # FIXING2: Dynamic exit days
                'signal': signal,
                'regime': regime_info.primary_regime,
                'transition_state': regime_info.transition_state,
                # FIXING2: Additional metadata
                'vol_adjustment': vol_adjustment,
                'correlation_penalty': correlation_penalty,
                'quality_passed': signal.get('quality_passed', False),
                'momentum_confirmed': signal.get('momentum_confirmed', False),
                'ev_original': signal.get('ev_original', signal.get('ev', 1.0)),
                'ev_adjusted': signal.get('ev', 1.0)
            })

        return positions


# ============================================================================
# MAIN ORCHESTRATOR: CHINA ADAPTIVE PROFIT MAXIMIZER (FIXING3)
# ============================================================================

class ChinaAdaptiveProfitMaximizer:
    """
    Main orchestrator combining all components.

    FIXING3 Enhancements (Relaxed from FIXING2):
    1. Relaxed EV thresholds (BULL: 0.3, BEAR: 0.6, HIGH_VOL: 1.0, NEUTRAL: 0.5)
    2. Composite quality scoring (0-1) instead of hard filters
    3. Relaxed quality thresholds (EPS > -0.1, D/E < 3.0, Market Cap > $500M)
    4. Momentum override (>10% 5-day or >15% 10-day overrides quality)
    5. Regime-weighted entry (BULL=70/30, BEAR=30/70 momentum/quality)
    6. New stock/IPO handler with tiered analysis
    7. Sector-specific quality rules
    8. Wider trailing stops (BULL: 15%, NEUTRAL: 10%, BEAR: 8%, HIGH_VOL: 5%)
    9. Volatility-adjusted position sizing
    10. Correlation penalty for concentrated positions
    11. Dynamic exit days based on regime

    Combines:
    1. US Model's EV-based filtering
    2. Adaptive regime detection (with lag-free transitions)
    3. Profit-maximizing concentration (regime-specific)
    4. China-specific optimizations

    Usage:
        maximizer = ChinaAdaptiveProfitMaximizer()

        # Execute strategy with FIXING3 enhancements
        result = maximizer.execute_strategy(signals, market_data, apply_fixing2=True)

        # Get positions
        positions = result['positions']
        regime = result['regime']
    """

    # FIXING3 version info
    VERSION = "3.0.0"
    FIXING3_DATE = "2025-12-20"

    def __init__(self, base_capital: float = 100000, apply_fixing2: bool = True):
        """
        Initialize the adaptive profit maximizer.

        Args:
            base_capital: Starting capital for position sizing
            apply_fixing2: Whether to enable fixing2/3 enhancements (default: True)
        """
        self.capital = base_capital
        self.current_regime = 'NEUTRAL'
        self.transition_confidence = 0.0
        self.apply_fixing2 = apply_fixing2

        # Initialize components
        self.regime_detector = ChinaAdaptiveRegimeDetector()
        self.profit_optimizer = ChinaProfitMaximizationLayer()
        self.position_sizer = ChinaDynamicPositionSizer()

        print(f"[CHINA MODEL] ChinaAdaptiveProfitMaximizer v{self.VERSION} initialized")
        print("  - Lag-free transition detection: ENABLED")
        print("  - Regime-specific profit optimization: ENABLED")
        print("  - Dynamic position sizing: ENABLED")
        if apply_fixing2:
            print("  - FIXING3 Enhancements: ENABLED")
            print("    * Relaxed EV thresholds (BULL:0.3, BEAR:0.6, HIGH_VOL:1.0, NEUTRAL:0.5)")
            print("    * Composite quality scoring + Momentum override")
            print("    * Relaxed filters (EPS>-0.1, D/E<3.0, MCap>$500M)")
            print("    * New stock/IPO tiered handler")
            print("    * Sector-specific rules")
            print("    * Wider trailing stops (BULL:15%, NEUTRAL:10%, BEAR:8%)")

    def execute_strategy(
        self,
        signals: List[Dict],
        market_data: pd.DataFrame,
        volatility_20d: float = 0.02,
        hsi_return_20d: float = 0.0,
        hsi_return_5d: float = 0.0,
        apply_fixing2: bool = None
    ) -> Dict[str, Any]:
        """
        Execute the complete adaptive profit-maximizing pipeline.

        FIXING2 Enhancements applied when apply_fixing2=True:
        - Quality filter (EPS > 0, D/E < 1.0, Market Cap > $1B)
        - Momentum confirmation (2-day return > 1%)
        - Sharpe-adjusted EV calculation
        - Dynamic stop-loss based on volatility
        - Trailing stops
        - Volatility-adjusted position sizing

        Args:
            signals: List of trading signals with 'ticker', 'ev', 'confidence', etc.
            market_data: DataFrame with OHLCV for regime detection (HSI or stock)
            volatility_20d: 20-day realized volatility
            hsi_return_20d: 20-day HSI returns
            hsi_return_5d: 5-day HSI returns
            apply_fixing2: Override instance setting (None uses instance default)

        Returns:
            Dict with regime info, positions, and expected returns
        """
        # Use instance setting if not overridden
        if apply_fixing2 is None:
            apply_fixing2 = self.apply_fixing2

        # 1. Detect regime and transitions (lag-free)
        regime_info = self.regime_detector.classify_regime(
            market_data=market_data,
            volatility_20d=volatility_20d,
            hsi_return_20d=hsi_return_20d,
            hsi_return_5d=hsi_return_5d
        )

        # 2. Get regime parameters
        params = regime_info.regime_params

        # 3. Apply profit-maximizing filters (regime-aware, with FIXING2)
        optimized_signals = self.profit_optimizer.optimize_for_regime(
            signals=signals,
            regime_params=params,
            regime_info=regime_info,
            apply_fixing2=apply_fixing2
        )

        # 4. Dynamic position sizing (with FIXING2)
        positions = self.position_sizer.size_positions(
            signals=optimized_signals,
            regime_params=params,
            regime_info=regime_info,
            capital=self.capital,
            apply_fixing2=apply_fixing2
        )

        # 5. Calculate expected return
        expected_return = self._calculate_expected_return(positions, params)

        # Update state
        self.current_regime = regime_info.primary_regime
        self.transition_confidence = regime_info.transition_confidence

        # FIXING2: Build enhanced result
        result = {
            'regime': regime_info.primary_regime,
            'transition_state': regime_info.transition_state,
            'confidence': regime_info.confidence,
            'signals_detected': regime_info.signals_detected,
            'parameters': params,
            'positions': positions,
            'expected_return': expected_return,
            'total_allocation': sum(p['allocation'] for p in positions),
            'num_positions': len(positions),
            # FIXING2: Additional metadata
            'fixing2_applied': apply_fixing2,
            'version': self.VERSION,
            'ev_threshold': params.get('min_ev_threshold', 1.0),
            'trailing_stop': params.get('trailing_stop', 0.08),
            'quality_filter_active': apply_fixing2,
            'momentum_filter_active': apply_fixing2 and params.get('momentum_confirm_days', 0) > 0,
        }

        return result

    def _calculate_expected_return(
        self,
        positions: List[Dict],
        params: Dict[str, Any]
    ) -> float:
        """Calculate expected return from positions."""
        if not positions:
            return 0.0

        total_expected = 0.0
        for pos in positions:
            ev = pos['signal'].get('ev', 1.0)
            allocation = pos['allocation']
            # Simplified expected return calculation
            expected = allocation * ev * params.get('take_profit_mult', 2.0) * 0.1
            total_expected += expected

        return total_expected

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get current regime summary."""
        return {
            'current_regime': self.current_regime,
            'transition_confidence': self.transition_confidence,
            'regime_history': list(self.regime_detector.regime_history),
            'pending_transition': self.regime_detector.pending_transition,
            'days_in_transition': self.regime_detector.days_in_transition,
            'blended_params': self.regime_detector.blended_params
        }


# ============================================================================
# MAIN EXECUTION / EXAMPLE
# ============================================================================

def main():
    """Example usage of ChinaAdaptiveProfitMaximizer."""
    print("=" * 70)
    print("CHINA ADAPTIVE PROFIT MAXIMIZER - EXAMPLE")
    print("=" * 70)

    # Create sample market data (simulating HSI)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')

    # Simulate bear-to-bull transition
    prices = [20000]  # HSI starting price
    for i in range(99):
        if i < 30:  # Bear phase
            change = np.random.normal(-0.003, 0.015)
        elif i < 50:  # Transition
            change = np.random.normal(0.002, 0.012)
        else:  # Bull phase
            change = np.random.normal(0.004, 0.010)
        prices.append(prices[-1] * (1 + change))

    market_data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000000, 5000000000, 100)
    }, index=dates)

    # Create sample signals
    sample_signals = [
        {'ticker': '0700.HK', 'ev': 1.5, 'confidence': 0.7, 'momentum': 0.6, 'rsi': 45, 'volume_ratio': 1.3, 'beta': 1.2},
        {'ticker': '9988.HK', 'ev': 1.2, 'confidence': 0.6, 'momentum': 0.5, 'rsi': 38, 'volume_ratio': 1.5, 'beta': 1.4},
        {'ticker': '1810.HK', 'ev': 0.8, 'confidence': 0.5, 'momentum': 0.4, 'rsi': 52, 'volume_ratio': 1.1, 'beta': 1.1},
        {'ticker': '2318.HK', 'ev': 2.0, 'confidence': 0.8, 'momentum': 0.7, 'rsi': 42, 'volume_ratio': 1.4, 'beta': 0.9},
        {'ticker': '0005.HK', 'ev': 0.6, 'confidence': 0.4, 'momentum': 0.3, 'rsi': 55, 'volume_ratio': 0.9, 'beta': 0.7},
    ]

    # Initialize maximizer
    maximizer = ChinaAdaptiveProfitMaximizer(base_capital=100000)

    print("\nRunning strategy at different market phases...\n")

    # Test at different phases
    checkpoints = [25, 45, 75]  # Bear, transition, bull

    for day in checkpoints:
        window = market_data.iloc[:day+1]

        # Calculate returns
        hsi_return_20d = (prices[day] / prices[max(0, day-20)] - 1) if day >= 20 else 0
        hsi_return_5d = (prices[day] / prices[max(0, day-5)] - 1) if day >= 5 else 0
        volatility = np.std([prices[i]/prices[i-1]-1 for i in range(max(1,day-20), day+1)]) if day >= 2 else 0.02

        result = maximizer.execute_strategy(
            signals=sample_signals,
            market_data=window,
            volatility_20d=volatility,
            hsi_return_20d=hsi_return_20d,
            hsi_return_5d=hsi_return_5d
        )

        print(f"Day {day+1} Analysis:")
        print(f"  Regime:              {result['regime']}")
        print(f"  Transition State:    {result['transition_state']}")
        print(f"  Confidence:          {result['confidence']:.2%}")
        print(f"  Signals Detected:    {', '.join(result['signals_detected']) or 'None'}")
        print(f"  Total Allocation:    {result['total_allocation']:.1%}")
        print(f"  Num Positions:       {result['num_positions']}")
        print(f"  Expected Return:     {result['expected_return']:.2%}")
        print(f"  Positions:")
        for pos in result['positions'][:3]:
            print(f"    - {pos['ticker']}: {pos['allocation']:.1%} (SL: {pos['stop_loss']:.1%})")
        print()

    print("=" * 70)
    print("KEY FEATURES:")
    print("-" * 70)
    print("1. Lag-free transition detection catches regime changes 10-30 days earlier")
    print("2. Gradual parameter blending prevents sudden position changes")
    print("3. Regime-specific optimization (BULL/BEAR/HIGH_VOL/NEUTRAL)")
    print("4. Dynamic position sizing based on regime + signal quality")
    print("5. Confirmation periods prevent whipsaws")
    print("=" * 70)


if __name__ == "__main__":
    main()
