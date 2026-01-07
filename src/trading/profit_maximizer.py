"""
Profit Maximization System - EV-Based Position Sizing and Portfolio Concentration
==================================================================================
From 'us model fixing6.pdf':
- Dynamic EV-based position sizing
- Kelly criterion integration
- Portfolio concentration on high-EV opportunities
- Complete profit maximization strategy

This module maximizes profits by:
1. Concentrating capital on highest EV opportunities
2. Using Kelly criterion for optimal position sizing
3. Filtering out negative and low EV signals
4. Dynamically adjusting positions based on EV tiers

Date: 2025-12-16
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EVTier(Enum):
    """Expected Value tiers for position sizing."""
    EXCEPTIONAL = "exceptional"      # EV > 10
    EXCELLENT = "excellent"          # EV 5-10
    GOOD = "good"                    # EV 2-5
    MODERATE = "moderate"            # EV 1-2
    LOW = "low"                      # EV 0-1
    NEGATIVE = "negative"            # EV < 0


@dataclass
class EVSizedPosition:
    """Result from EV-based position sizing."""
    ticker: str
    signal_type: str
    expected_value: float
    ev_tier: str
    confidence: float
    potential_gain: float

    # Position sizing
    position_pct: float
    position_amount: float
    risk_multiplier: float
    kelly_fraction: float

    # Risk management
    stop_loss_pct: float
    take_profit_pct: float
    max_loss_amount: float
    expected_profit: float

    # Metadata
    should_trade: bool = True
    skip_reason: Optional[str] = None
    priority_rank: int = 0


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation result."""
    total_capital: float
    allocated_capital: float
    cash_reserve: float
    positions: List[EVSizedPosition]

    # Summary metrics
    total_expected_profit: float
    portfolio_ev: float
    max_drawdown_estimate: float
    risk_adjusted_return: float

    # Counts
    buy_count: int = 0
    sell_count: int = 0
    skipped_count: int = 0


class DynamicEVPositionSizer:
    """
    Dynamic position sizing based on Expected Value tiers.

    Key Features:
    - EV-based position allocation rules
    - Kelly criterion integration
    - Maximum position limits to prevent over-concentration
    - RSI adjustment compatibility
    """

    def __init__(self, capital: float = 10000):
        self.capital = capital

        # EV-based position sizing rules from fixing6.pdf
        self.EV_POSITION_RULES = {
            EVTier.EXCEPTIONAL: {   # EV > 10
                'min_pct': 0.20,
                'max_pct': 0.35,
                'risk_mult': 1.5,
                'kelly_cap': 0.40,
            },
            EVTier.EXCELLENT: {     # EV 5-10
                'min_pct': 0.15,
                'max_pct': 0.25,
                'risk_mult': 1.2,
                'kelly_cap': 0.30,
            },
            EVTier.GOOD: {          # EV 2-5
                'min_pct': 0.10,
                'max_pct': 0.20,
                'risk_mult': 1.0,
                'kelly_cap': 0.25,
            },
            EVTier.MODERATE: {      # EV 1-2
                'min_pct': 0.05,
                'max_pct': 0.15,
                'risk_mult': 0.8,
                'kelly_cap': 0.15,
            },
            EVTier.LOW: {           # EV 0-1
                'min_pct': 0.02,
                'max_pct': 0.10,
                'risk_mult': 0.5,
                'kelly_cap': 0.10,
            },
            EVTier.NEGATIVE: {      # EV < 0
                'min_pct': 0.00,
                'max_pct': 0.00,
                'risk_mult': 0.0,
                'kelly_cap': 0.00,
            },
        }

        # Maximum position constraints
        self.MAX_SINGLE_POSITION = 0.35       # 35% max in any single position
        self.MAX_CONCENTRATED_POSITIONS = 3   # Top 3 can use enhanced sizing
        self.MIN_DIVERSIFICATION_POSITIONS = 5  # Try to have at least 5 positions

        # Risk parameters
        self.BASE_STOP_LOSS = 0.08           # 8% base stop loss
        self.CASH_RESERVE_PCT = 0.10         # Keep 10% in cash

        # Statistics
        self.stats = {
            'signals_processed': 0,
            'signals_skipped_negative_ev': 0,
            'signals_skipped_low_ev': 0,
            'exceptional_ev_count': 0,
            'excellent_ev_count': 0,
            'total_allocated': 0,
        }

        logger.info(f"DynamicEVPositionSizer initialized with ${capital:,.2f} capital")

    def classify_ev_tier(self, ev: float) -> EVTier:
        """Classify EV into tiers."""
        if ev > 10:
            return EVTier.EXCEPTIONAL
        elif ev >= 5:
            return EVTier.EXCELLENT
        elif ev >= 2:
            return EVTier.GOOD
        elif ev >= 1:
            return EVTier.MODERATE
        elif ev >= 0:
            return EVTier.LOW
        else:
            return EVTier.NEGATIVE

    def calculate_kelly_fraction(
        self,
        confidence: float,
        potential_gain: float,
        potential_loss: float,
    ) -> float:
        """
        Calculate Kelly criterion fraction for position sizing.

        Kelly Formula: f* = (bp - q) / b
        where:
            b = odds received on bet (potential_gain / potential_loss)
            p = probability of winning (confidence)
            q = probability of losing (1 - confidence)
        """
        if potential_loss <= 0 or potential_gain <= 0:
            return 0.0

        b = potential_gain / potential_loss
        p = confidence
        q = 1 - confidence

        kelly = (b * p - q) / b

        # Never bet more than half Kelly (more conservative)
        kelly = max(0, min(kelly * 0.5, 0.5))

        return kelly

    def size_position(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        expected_value: float,
        potential_gain_pct: float,
        rsi_multiplier: float = 1.0,
        priority_rank: int = 0,
    ) -> EVSizedPosition:
        """
        Calculate optimal position size based on EV.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            expected_value: Pre-calculated EV
            potential_gain_pct: Expected gain percentage
            rsi_multiplier: RSI-based position adjustment
            priority_rank: Priority among all signals (1 = highest)

        Returns:
            EVSizedPosition with complete sizing information
        """
        self.stats['signals_processed'] += 1

        # Classify EV tier
        ev_tier = self.classify_ev_tier(expected_value)
        rules = self.EV_POSITION_RULES[ev_tier]

        # Skip negative EV signals
        if ev_tier == EVTier.NEGATIVE:
            self.stats['signals_skipped_negative_ev'] += 1
            return EVSizedPosition(
                ticker=ticker,
                signal_type=signal_type,
                expected_value=expected_value,
                ev_tier=ev_tier.value,
                confidence=confidence,
                potential_gain=potential_gain_pct,
                position_pct=0,
                position_amount=0,
                risk_multiplier=0,
                kelly_fraction=0,
                stop_loss_pct=0,
                take_profit_pct=0,
                max_loss_amount=0,
                expected_profit=0,
                should_trade=False,
                skip_reason=f"Negative EV ({expected_value:.2f})",
                priority_rank=priority_rank,
            )

        # Track exceptional/excellent EVs
        if ev_tier == EVTier.EXCEPTIONAL:
            self.stats['exceptional_ev_count'] += 1
        elif ev_tier == EVTier.EXCELLENT:
            self.stats['excellent_ev_count'] += 1

        # Calculate Kelly fraction
        potential_loss = self.BASE_STOP_LOSS
        kelly = self.calculate_kelly_fraction(
            confidence,
            potential_gain_pct / 100,
            potential_loss,
        )

        # Cap Kelly to tier limit
        kelly = min(kelly, rules['kelly_cap'])

        # Calculate base position percentage
        # Start with midpoint of tier range
        base_pct = (rules['min_pct'] + rules['max_pct']) / 2

        # Scale by EV within tier
        if ev_tier == EVTier.EXCEPTIONAL:
            # Scale from 20% to 35% based on how high above 10
            ev_scale = min((expected_value - 10) / 20, 1.0)  # 0-1 scale
            base_pct = rules['min_pct'] + ev_scale * (rules['max_pct'] - rules['min_pct'])
        elif ev_tier == EVTier.EXCELLENT:
            ev_scale = (expected_value - 5) / 5  # 0-1 within tier
            base_pct = rules['min_pct'] + ev_scale * (rules['max_pct'] - rules['min_pct'])

        # Apply risk multiplier
        risk_mult = rules['risk_mult']

        # Apply RSI adjustment
        adjusted_pct = base_pct * risk_mult * rsi_multiplier

        # Boost for top priority positions
        if priority_rank <= 3:
            priority_boost = 1.0 + (0.1 * (4 - priority_rank))  # +30%, +20%, +10%
            adjusted_pct *= priority_boost

        # Apply constraints
        final_pct = min(adjusted_pct, self.MAX_SINGLE_POSITION)
        final_pct = max(final_pct, rules['min_pct'])

        # Calculate position amount
        available_capital = self.capital * (1 - self.CASH_RESERVE_PCT)
        position_amount = available_capital * final_pct

        # Calculate stop-loss adjusted for signal type and volatility
        stop_loss_pct = self.BASE_STOP_LOSS * (1.5 if signal_type == 'SELL' else 1.0)

        # Calculate take-profit target
        take_profit_pct = potential_gain_pct / 100

        # Calculate expected outcomes
        max_loss = position_amount * stop_loss_pct
        expected_profit = position_amount * (confidence * take_profit_pct - (1 - confidence) * stop_loss_pct)

        return EVSizedPosition(
            ticker=ticker,
            signal_type=signal_type,
            expected_value=expected_value,
            ev_tier=ev_tier.value,
            confidence=confidence,
            potential_gain=potential_gain_pct,
            position_pct=final_pct,
            position_amount=position_amount,
            risk_multiplier=risk_mult,
            kelly_fraction=kelly,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_loss_amount=max_loss,
            expected_profit=expected_profit,
            should_trade=True,
            priority_rank=priority_rank,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get sizing statistics."""
        return self.stats.copy()


class ProfitMaximizer:
    """
    Portfolio concentration strategy for maximum profits.

    Key Features:
    - Concentrates capital on highest EV opportunities
    - Filters out low-quality signals
    - Manages portfolio-level risk
    - Integrates with RSI risk adapter
    """

    def __init__(self, capital: float = 10000, max_positions: int = 10):
        self.capital = capital
        self.max_positions = max_positions
        self.position_sizer = DynamicEVPositionSizer(capital)

        # Filtering thresholds
        self.MIN_EV_THRESHOLD = 0.5           # Skip EV < 0.5
        self.MIN_CONFIDENCE = 0.55            # Skip confidence < 55%
        self.MIN_PROFIT_PCT = 1.0             # Skip potential gain < 1%

        # Portfolio constraints
        self.MAX_SECTOR_EXPOSURE = 0.40       # Max 40% in any sector
        self.MAX_CORRELATION_POSITIONS = 3   # Max 3 highly correlated positions

        logger.info(f"ProfitMaximizer initialized: ${capital:,.2f}, max {max_positions} positions")

    def filter_signals(
        self,
        signals: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter signals to keep only tradeable ones.

        Returns:
            Tuple of (tradeable_signals, filtered_out_signals)
        """
        tradeable = []
        filtered = []

        for signal in signals:
            ev = signal.get('expected_value', 0)
            confidence = signal.get('confidence', 0)
            potential_gain = abs(signal.get('potential_gain_5d', 0))

            # Apply filters
            if ev < self.MIN_EV_THRESHOLD:
                signal['filter_reason'] = f"EV too low ({ev:.2f} < {self.MIN_EV_THRESHOLD})"
                filtered.append(signal)
            elif confidence < self.MIN_CONFIDENCE:
                signal['filter_reason'] = f"Confidence too low ({confidence:.1%} < {self.MIN_CONFIDENCE:.0%})"
                filtered.append(signal)
            elif potential_gain < self.MIN_PROFIT_PCT:
                signal['filter_reason'] = f"Profit potential too low ({potential_gain:.1f}% < {self.MIN_PROFIT_PCT}%)"
                filtered.append(signal)
            else:
                tradeable.append(signal)

        return tradeable, filtered

    def rank_by_ev(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank signals by Expected Value (highest first)."""
        return sorted(signals, key=lambda x: x.get('expected_value', 0), reverse=True)

    def allocate_portfolio(
        self,
        buy_signals: List[Dict[str, Any]],
        sell_signals: List[Dict[str, Any]],
        include_sells: bool = True,
    ) -> PortfolioAllocation:
        """
        Allocate capital across buy and sell signals.

        Args:
            buy_signals: List of BUY signal dictionaries
            sell_signals: List of SELL signal dictionaries
            include_sells: Whether to include short positions

        Returns:
            PortfolioAllocation with complete portfolio
        """
        # Filter signals
        tradeable_buys, filtered_buys = self.filter_signals(buy_signals)
        tradeable_sells, filtered_sells = self.filter_signals(sell_signals) if include_sells else ([], sell_signals)

        # Combine and rank by EV
        all_signals = []
        for sig in tradeable_buys:
            sig['signal_type'] = 'BUY'
            all_signals.append(sig)
        for sig in tradeable_sells:
            sig['signal_type'] = 'SELL'
            all_signals.append(sig)

        ranked_signals = self.rank_by_ev(all_signals)

        # Allocate positions to top signals
        positions = []
        total_allocated = 0
        available_capital = self.capital * 0.90  # Keep 10% cash reserve

        for rank, signal in enumerate(ranked_signals[:self.max_positions], 1):
            ticker = signal.get('ticker', 'UNKNOWN')
            signal_type = signal.get('signal_type', 'BUY')
            confidence = signal.get('confidence', 0.5)
            ev = signal.get('expected_value', 0)
            potential_gain = abs(signal.get('potential_gain_5d', signal.get('potential_gain_10d', 5)))
            rsi_mult = signal.get('rsi_position_multiplier', 1.0)

            # Size the position
            position = self.position_sizer.size_position(
                ticker=ticker,
                signal_type=signal_type,
                confidence=confidence,
                expected_value=ev,
                potential_gain_pct=potential_gain,
                rsi_multiplier=rsi_mult,
                priority_rank=rank,
            )

            if position.should_trade:
                # Check we don't exceed available capital
                if total_allocated + position.position_amount <= available_capital:
                    positions.append(position)
                    total_allocated += position.position_amount
                else:
                    # Reduce position to fit remaining capital
                    remaining = available_capital - total_allocated
                    if remaining > 500:  # Min $500 position
                        position.position_amount = remaining
                        position.position_pct = remaining / self.capital
                        positions.append(position)
                        total_allocated += remaining

        # Calculate portfolio metrics
        total_expected_profit = sum(p.expected_profit for p in positions)
        portfolio_ev = sum(p.expected_value * p.position_pct for p in positions if p.should_trade)
        max_drawdown = sum(p.max_loss_amount for p in positions)

        risk_adjusted = total_expected_profit / max_drawdown if max_drawdown > 0 else 0

        buy_count = len([p for p in positions if p.signal_type == 'BUY'])
        sell_count = len([p for p in positions if p.signal_type == 'SELL'])
        skipped_count = len(filtered_buys) + len(filtered_sells)

        return PortfolioAllocation(
            total_capital=self.capital,
            allocated_capital=total_allocated,
            cash_reserve=self.capital - total_allocated,
            positions=positions,
            total_expected_profit=total_expected_profit,
            portfolio_ev=portfolio_ev,
            max_drawdown_estimate=max_drawdown,
            risk_adjusted_return=risk_adjusted,
            buy_count=buy_count,
            sell_count=sell_count,
            skipped_count=skipped_count,
        )

    def get_top_opportunities(
        self,
        buy_signals: List[Dict[str, Any]],
        sell_signals: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get top N opportunities sorted by EV.

        Returns list of dicts with opportunity details.
        """
        # Filter and rank
        tradeable_buys, _ = self.filter_signals(buy_signals)
        tradeable_sells, _ = self.filter_signals(sell_signals)

        all_signals = []
        for sig in tradeable_buys:
            all_signals.append({
                'ticker': sig.get('ticker'),
                'type': 'BUY',
                'ev': sig.get('expected_value', 0),
                'confidence': sig.get('confidence', 0),
                'profit_5d': sig.get('potential_gain_5d', 0),
                'profit_10d': sig.get('potential_gain_10d', 0),
                'rsi': sig.get('rsi', 50),
                'rsi_risk_level': sig.get('rsi_risk_level', 'N/A'),
            })
        for sig in tradeable_sells:
            all_signals.append({
                'ticker': sig.get('ticker'),
                'type': 'SELL',
                'ev': sig.get('expected_value', 0),
                'confidence': sig.get('confidence', 0),
                'profit_5d': abs(sig.get('potential_gain_5d', 0)),
                'profit_10d': abs(sig.get('potential_gain_10d', 0)),
                'rsi': sig.get('rsi', 50),
                'rsi_risk_level': sig.get('rsi_risk_level', 'N/A'),
            })

        # Sort by EV
        ranked = sorted(all_signals, key=lambda x: x['ev'], reverse=True)

        return ranked[:top_n]


class CompleteProfitMaximizationStrategy:
    """
    Complete profit maximization system combining all components.

    Features:
    - EV-based signal filtering
    - Dynamic position sizing
    - Portfolio concentration
    - Risk management integration
    - Performance tracking
    """

    def __init__(self, capital: float = 10000):
        self.capital = capital
        self.maximizer = ProfitMaximizer(capital)

        # Performance tracking
        self.execution_history = []
        self.total_pnl = 0
        self.win_count = 0
        self.loss_count = 0

        logger.info("CompleteProfitMaximizationStrategy initialized")

    def analyze_opportunities(
        self,
        buy_signals: List[Dict[str, Any]],
        sell_signals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Full analysis of trading opportunities.

        Returns comprehensive analysis including:
        - Portfolio allocation
        - Top opportunities
        - Risk metrics
        - Execution recommendations
        """
        # Get portfolio allocation
        allocation = self.maximizer.allocate_portfolio(
            buy_signals,
            sell_signals,
            include_sells=True,
        )

        # Get top opportunities
        top_opps = self.maximizer.get_top_opportunities(
            buy_signals,
            sell_signals,
            top_n=10,
        )

        # Categorize by EV tier
        exceptional = [o for o in top_opps if o['ev'] > 10]
        excellent = [o for o in top_opps if 5 <= o['ev'] <= 10]
        good = [o for o in top_opps if 2 <= o['ev'] < 5]

        # Build execution plan
        execution_plan = []
        for pos in allocation.positions[:5]:  # Top 5 positions
            execution_plan.append({
                'ticker': pos.ticker,
                'action': pos.signal_type,
                'amount': pos.position_amount,
                'pct_of_portfolio': pos.position_pct * 100,
                'ev': pos.expected_value,
                'stop_loss': pos.stop_loss_pct * 100,
                'take_profit': pos.take_profit_pct * 100,
                'expected_profit': pos.expected_profit,
            })

        return {
            'allocation': {
                'total_capital': allocation.total_capital,
                'allocated': allocation.allocated_capital,
                'cash_reserve': allocation.cash_reserve,
                'buy_positions': allocation.buy_count,
                'sell_positions': allocation.sell_count,
                'skipped_signals': allocation.skipped_count,
            },
            'metrics': {
                'expected_profit': allocation.total_expected_profit,
                'expected_roi': (allocation.total_expected_profit / allocation.total_capital) * 100,
                'portfolio_ev': allocation.portfolio_ev,
                'max_drawdown': allocation.max_drawdown_estimate,
                'risk_adjusted_return': allocation.risk_adjusted_return,
            },
            'opportunity_breakdown': {
                'exceptional_ev': len(exceptional),
                'excellent_ev': len(excellent),
                'good_ev': len(good),
                'total_tradeable': len(top_opps),
            },
            'top_opportunities': top_opps[:5],
            'execution_plan': execution_plan,
            'positions': [
                {
                    'ticker': p.ticker,
                    'type': p.signal_type,
                    'amount': p.position_amount,
                    'pct': p.position_pct * 100,
                    'ev': p.expected_value,
                    'ev_tier': p.ev_tier,
                    'confidence': p.confidence,
                    'expected_profit': p.expected_profit,
                }
                for p in allocation.positions
            ],
        }

    def get_concentrated_portfolio(
        self,
        buy_signals: List[Dict[str, Any]],
        sell_signals: List[Dict[str, Any]],
        max_positions: int = 5,
    ) -> Dict[str, Any]:
        """
        Get a concentrated portfolio of top positions only.

        This is for users who want maximum concentration on
        highest-EV opportunities.
        """
        # Create concentrated maximizer
        concentrated = ProfitMaximizer(
            capital=self.capital,
            max_positions=max_positions,
        )

        # Higher thresholds for concentrated portfolio
        concentrated.MIN_EV_THRESHOLD = 2.0
        concentrated.MIN_CONFIDENCE = 0.60
        concentrated.MIN_PROFIT_PCT = 3.0

        allocation = concentrated.allocate_portfolio(
            buy_signals,
            sell_signals,
            include_sells=True,
        )

        return {
            'strategy': 'concentrated',
            'positions': max_positions,
            'allocation': allocation,
            'summary': {
                'total_invested': allocation.allocated_capital,
                'expected_return': allocation.total_expected_profit,
                'expected_roi': (allocation.total_expected_profit / self.capital) * 100,
                'avg_position_size': allocation.allocated_capital / len(allocation.positions) if allocation.positions else 0,
            },
            'holdings': [
                {
                    'ticker': p.ticker,
                    'type': p.signal_type,
                    'amount': p.position_amount,
                    'ev': p.expected_value,
                    'expected_profit': p.expected_profit,
                }
                for p in allocation.positions
            ],
        }

    def record_execution(
        self,
        ticker: str,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
    ):
        """Record trade execution for performance tracking."""
        if signal_type == 'BUY':
            pnl = (exit_price - entry_price) / entry_price * position_size
        else:  # SELL (short)
            pnl = (entry_price - exit_price) / entry_price * position_size

        self.total_pnl += pnl
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.execution_history.append({
            'ticker': ticker,
            'type': signal_type,
            'entry': entry_price,
            'exit': exit_price,
            'size': position_size,
            'pnl': pnl,
        })

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': self.total_pnl / total_trades if total_trades > 0 else 0,
        }


# Factory functions
def create_profit_maximizer(capital: float = 10000) -> ProfitMaximizer:
    """Create a profit maximizer instance."""
    return ProfitMaximizer(capital)


def create_ev_position_sizer(capital: float = 10000) -> DynamicEVPositionSizer:
    """Create an EV position sizer instance."""
    return DynamicEVPositionSizer(capital)


def create_complete_strategy(capital: float = 10000) -> CompleteProfitMaximizationStrategy:
    """Create a complete profit maximization strategy."""
    return CompleteProfitMaximizationStrategy(capital)


# Utility function for quick EV analysis
def analyze_signal_ev(
    confidence: float,
    potential_gain: float,
    potential_loss: float = 8.0,
) -> Dict[str, Any]:
    """
    Quick EV analysis for a signal.

    Args:
        confidence: Win probability (0-1)
        potential_gain: Expected gain if win (%)
        potential_loss: Expected loss if lose (%)

    Returns:
        Dict with EV analysis
    """
    ev = (confidence * potential_gain) - ((1 - confidence) * potential_loss)

    if ev > 10:
        tier = "EXCEPTIONAL"
        recommendation = "Strong BUY - allocate 20-35%"
    elif ev >= 5:
        tier = "EXCELLENT"
        recommendation = "BUY - allocate 15-25%"
    elif ev >= 2:
        tier = "GOOD"
        recommendation = "BUY - allocate 10-20%"
    elif ev >= 1:
        tier = "MODERATE"
        recommendation = "Consider - allocate 5-15%"
    elif ev >= 0:
        tier = "LOW"
        recommendation = "Skip or minimal position"
    else:
        tier = "NEGATIVE"
        recommendation = "DO NOT TRADE"

    return {
        'expected_value': ev,
        'ev_tier': tier,
        'recommendation': recommendation,
        'confidence': confidence,
        'potential_gain': potential_gain,
        'potential_loss': potential_loss,
    }
