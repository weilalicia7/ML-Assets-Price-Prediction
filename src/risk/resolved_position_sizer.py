"""
Resolved Position Sizer - Phase 2 Conflict Resolution

Resolves conflicts between Phase 1 and Phase 2 position sizing approaches.
Phase 2 quarter-Kelly takes precedence over Phase 1 confidence-based sizing.

Kelly Criterion: f = p - (1-p)/b
where:
- f = optimal fraction of capital
- p = win probability
- b = win/loss ratio

Quarter-Kelly (0.25 fraction) used for conservative risk management.

Based on: phase2 fixing on C model_conflict resolutions.pdf
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ResolvedPositionSizer:
    """
    Resolved position sizer using Phase 2 quarter-Kelly approach.

    Combines:
    - Kelly Criterion for optimal sizing
    - Confidence calibration
    - Drawdown-aware adjustments
    - Volatility scaling

    Phase 2 quarter-Kelly (0.25) takes precedence over Phase 1 confidence-based.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,       # Quarter-Kelly (Phase 2)
        max_position: float = 0.30,          # Maximum position size
        min_position: float = 0.02,          # Minimum position size
        confidence_weight: float = 0.3,      # How much confidence affects sizing
        lookback_trades: int = 20,           # Trades for win rate calculation
        use_bayesian: bool = True            # Use Bayesian win rate estimation
    ):
        """
        Initialize resolved position sizer.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
            max_position: Maximum position as fraction of portfolio
            min_position: Minimum position as fraction of portfolio
            confidence_weight: Weight of confidence in final sizing
            lookback_trades: Number of recent trades for statistics
            use_bayesian: Use Bayesian estimation for win rate
        """
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
        self.confidence_weight = confidence_weight
        self.lookback_trades = lookback_trades
        self.use_bayesian = use_bayesian

        # Trade history for win rate calculation
        self.trade_history: deque = deque(maxlen=lookback_trades)

        # Bayesian prior (Beta distribution parameters)
        # Start with uninformative prior: alpha=1, beta=1 (uniform)
        self.bayesian_alpha = 1.0
        self.bayesian_beta = 1.0

        # Per-ticker tracking
        self.ticker_stats: Dict[str, Dict] = {}

        logger.info(f"Initialized ResolvedPositionSizer (Phase 2 quarter-Kelly):")
        logger.info(f"  Kelly fraction: {kelly_fraction:.2f}, Max position: {max_position:.1%}")

    def record_trade(
        self,
        ticker: str,
        was_profitable: bool,
        profit_pct: float = 0.0,
        loss_pct: float = 0.0
    ):
        """
        Record a completed trade for statistics.

        Args:
            ticker: Stock ticker
            was_profitable: Whether trade was profitable
            profit_pct: Profit percentage (if profitable)
            loss_pct: Loss percentage (if loss)
        """
        trade = {
            'ticker': ticker,
            'profitable': was_profitable,
            'profit_pct': profit_pct if was_profitable else 0,
            'loss_pct': loss_pct if not was_profitable else 0
        }
        self.trade_history.append(trade)

        # Update Bayesian prior (Beta-Bernoulli update)
        if self.use_bayesian:
            if was_profitable:
                self.bayesian_alpha += 1
            else:
                self.bayesian_beta += 1

        # Update ticker-specific stats
        if ticker not in self.ticker_stats:
            self.ticker_stats[ticker] = {
                'wins': 0, 'losses': 0,
                'total_profit': 0, 'total_loss': 0,
                'alpha': 1.0, 'beta': 1.0  # Per-ticker Bayesian prior
            }

        stats = self.ticker_stats[ticker]
        if was_profitable:
            stats['wins'] += 1
            stats['total_profit'] += profit_pct
            stats['alpha'] += 1
        else:
            stats['losses'] += 1
            stats['total_loss'] += abs(loss_pct)
            stats['beta'] += 1

    def get_win_rate(self, ticker: Optional[str] = None) -> float:
        """
        Get win rate using Bayesian estimation.

        Bayesian mean of Beta distribution: alpha / (alpha + beta)

        Args:
            ticker: Optional ticker for ticker-specific rate

        Returns:
            Estimated win rate
        """
        if ticker and ticker in self.ticker_stats:
            stats = self.ticker_stats[ticker]
            alpha = stats['alpha']
            beta = stats['beta']
        else:
            alpha = self.bayesian_alpha
            beta = self.bayesian_beta

        # Bayesian mean of Beta distribution
        return alpha / (alpha + beta)

    def get_win_loss_ratio(self, ticker: Optional[str] = None) -> float:
        """
        Get average win/loss ratio.

        Args:
            ticker: Optional ticker for ticker-specific ratio

        Returns:
            Win/loss ratio (average win / average loss)
        """
        if ticker and ticker in self.ticker_stats:
            stats = self.ticker_stats[ticker]
            avg_win = stats['total_profit'] / max(1, stats['wins'])
            avg_loss = stats['total_loss'] / max(1, stats['losses'])
        else:
            # Calculate from trade history
            wins = [t for t in self.trade_history if t['profitable']]
            losses = [t for t in self.trade_history if not t['profitable']]

            avg_win = np.mean([t['profit_pct'] for t in wins]) if wins else 0.02
            avg_loss = np.mean([t['loss_pct'] for t in losses]) if losses else 0.02

        # Ensure positive values
        avg_win = max(0.01, avg_win)
        avg_loss = max(0.01, avg_loss)

        return avg_win / avg_loss

    def calculate_kelly(
        self,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
        ticker: Optional[str] = None
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Kelly formula: f = p - (1-p)/b
        where:
        - f = optimal fraction
        - p = win probability
        - b = win/loss ratio

        We use quarter-Kelly (0.25 * f) for conservative sizing.

        Args:
            win_rate: Override win rate (uses historical if None)
            win_loss_ratio: Override win/loss ratio
            ticker: Ticker for ticker-specific calculation

        Returns:
            Kelly-based position size (as fraction of portfolio)
        """
        # Get parameters
        p = win_rate if win_rate is not None else self.get_win_rate(ticker)
        b = win_loss_ratio if win_loss_ratio is not None else self.get_win_loss_ratio(ticker)

        # Kelly formula: f = p - (1-p)/b
        if b > 0:
            kelly = p - (1 - p) / b
        else:
            kelly = 0

        # Apply Kelly fraction (quarter-Kelly by default)
        fractional_kelly = kelly * self.kelly_fraction

        # Clamp to valid range
        fractional_kelly = max(0, min(fractional_kelly, self.max_position))

        logger.debug(f"Kelly calculation: p={p:.2f}, b={b:.2f}, kelly={kelly:.3f}, "
                    f"quarter-kelly={fractional_kelly:.3f}")

        return fractional_kelly

    def calculate_position_size(
        self,
        confidence: float,
        ticker: Optional[str] = None,
        volatility: Optional[float] = None,
        drawdown_multiplier: float = 1.0
    ) -> Dict:
        """
        Calculate final position size combining all factors.

        Combines:
        1. Kelly-based sizing (primary - Phase 2)
        2. Confidence adjustment (secondary)
        3. Volatility scaling (optional)
        4. Drawdown adjustment (from UnifiedDrawdownManager)

        Args:
            confidence: Signal confidence (0-1)
            ticker: Stock ticker for ticker-specific stats
            volatility: Current volatility for scaling
            drawdown_multiplier: Multiplier from drawdown manager

        Returns:
            Dict with position size and breakdown
        """
        # 1. Calculate Kelly-based size
        kelly_size = self.calculate_kelly(ticker=ticker)

        # 2. Confidence adjustment
        # Blend Kelly with confidence-scaled max position
        confidence_size = self.max_position * confidence
        blended_size = (
            kelly_size * (1 - self.confidence_weight) +
            confidence_size * self.confidence_weight
        )

        # 3. Volatility scaling (reduce position in high volatility)
        vol_multiplier = 1.0
        if volatility is not None:
            # Higher volatility = smaller position
            # Assuming typical volatility around 0.02 (2%)
            vol_multiplier = min(1.0, 0.02 / max(0.005, volatility))
            vol_multiplier = max(0.5, vol_multiplier)  # Don't go below 50%

        # 4. Apply all multipliers
        final_size = blended_size * vol_multiplier * drawdown_multiplier

        # Enforce min/max bounds
        if final_size < self.min_position:
            final_size = 0  # Too small to trade
        final_size = min(final_size, self.max_position)

        # Get win rate for reporting
        win_rate = self.get_win_rate(ticker)

        return {
            'position_size': final_size,
            'kelly_size': kelly_size,
            'confidence_size': confidence_size,
            'blended_size': blended_size,
            'vol_multiplier': vol_multiplier,
            'drawdown_multiplier': drawdown_multiplier,
            'win_rate': win_rate,
            'kelly_fraction': self.kelly_fraction,
            'can_trade': final_size >= self.min_position
        }

    def get_position_value(
        self,
        portfolio_value: float,
        confidence: float,
        ticker: Optional[str] = None,
        volatility: Optional[float] = None,
        drawdown_multiplier: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Get position value in currency terms.

        Args:
            portfolio_value: Total portfolio value
            confidence: Signal confidence
            ticker: Stock ticker
            volatility: Current volatility
            drawdown_multiplier: From drawdown manager

        Returns:
            Tuple of (position_value, details_dict)
        """
        sizing = self.calculate_position_size(
            confidence=confidence,
            ticker=ticker,
            volatility=volatility,
            drawdown_multiplier=drawdown_multiplier
        )

        position_value = portfolio_value * sizing['position_size']
        sizing['position_value'] = position_value

        return position_value, sizing

    def reset(self):
        """Reset all statistics and history."""
        self.trade_history.clear()
        self.bayesian_alpha = 1.0
        self.bayesian_beta = 1.0
        self.ticker_stats.clear()
        logger.info("Position sizer reset")


# Convenience function
def get_resolved_position_sizer(**kwargs) -> ResolvedPositionSizer:
    """Get a configured ResolvedPositionSizer instance."""
    return ResolvedPositionSizer(**kwargs)
