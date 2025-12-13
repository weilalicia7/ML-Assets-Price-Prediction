"""
Advanced Risk Management Module for China Model

Implements:
1. Dynamic stock filtering (auto-ban poor performers)
2. Smart position sizing based on stock performance
3. Drawdown circuit breaker
4. Real-time performance monitoring

Based on: phase1 fixing on C model 2.pdf
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DynamicStockFilter:
    """
    Automatically filter out poor-performing stocks based on trading history.

    Auto-bans stocks with:
    - Win rate < 40% after 10+ trades
    - Negative average P&L
    """

    def __init__(self, min_win_rate: float = 0.40, min_trades: int = 10):
        """
        Initialize dynamic stock filter.

        Args:
            min_win_rate: Minimum win rate to remain tradeable (0.40 = 40%)
            min_trades: Minimum trades before filtering applies
        """
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.trading_history: Dict[str, List[float]] = defaultdict(list)
        self.banned_stocks: Set[str] = set()
        self.stock_performance: Dict[str, Dict] = {}

        logger.info(f"Initialized DynamicStockFilter: min_win_rate={min_win_rate:.0%}, min_trades={min_trades}")

    def record_trade(self, ticker: str, pnl: float) -> None:
        """Record a trade outcome for a stock."""
        self.trading_history[ticker].append(pnl)
        self._update_stock_performance(ticker)

    def _update_stock_performance(self, ticker: str) -> None:
        """Update performance metrics for a stock."""
        trades = self.trading_history[ticker]

        if len(trades) >= self.min_trades:
            win_rate = sum(1 for pnl in trades if pnl > 0) / len(trades)
            avg_pnl = np.mean(trades)
            total_pnl = sum(trades)

            self.stock_performance[ticker] = {
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'total_trades': len(trades),
                'approved': win_rate >= self.min_win_rate and avg_pnl > 0
            }

            # Auto-ban if criteria not met
            if not self.stock_performance[ticker]['approved']:
                if ticker not in self.banned_stocks:
                    self.banned_stocks.add(ticker)
                    logger.warning(f"AUTO-BANNED {ticker}: win_rate={win_rate:.1%}, avg_pnl=${avg_pnl:.2f}")
            elif ticker in self.banned_stocks:
                # Rehabilitate if performance improves
                self.banned_stocks.remove(ticker)
                logger.info(f"REHABILITATED {ticker}: win_rate={win_rate:.1%}, avg_pnl=${avg_pnl:.2f}")

    def is_tradeable(self, ticker: str) -> bool:
        """Check if a stock is allowed to be traded."""
        return ticker not in self.banned_stocks

    def get_banned_stocks(self) -> List[str]:
        """Get list of currently banned stocks."""
        return list(self.banned_stocks)

    def get_stock_metrics(self, ticker: str) -> Optional[Dict]:
        """Get performance metrics for a stock."""
        return self.stock_performance.get(ticker)


class SmartPositionSizer:
    """
    Size positions based on confidence AND historical stock performance.

    Reduces position sizes for:
    - Low win rate stocks
    - Negative average P&L stocks
    - High drawdown stocks
    """

    def __init__(self, max_position_size: float = 0.15,
                 win_rate_threshold: float = 0.45):
        """
        Initialize smart position sizer.

        Args:
            max_position_size: Maximum position as fraction of capital (0.15 = 15%)
            win_rate_threshold: Threshold below which to reduce position size
        """
        self.max_position_size = max_position_size
        self.win_rate_threshold = win_rate_threshold

        logger.info(f"Initialized SmartPositionSizer: max_size={max_position_size:.0%}")

    def calculate_position_size(
        self,
        confidence: float,
        stock_win_rate: Optional[float] = None,
        avg_pnl: Optional[float] = None,
        stock_drawdown: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on multiple factors.

        Args:
            confidence: Signal confidence (0-1)
            stock_win_rate: Historical win rate for this stock (0-1)
            avg_pnl: Average P&L per trade for this stock
            stock_drawdown: Current drawdown for this stock (0-1)

        Returns:
            Position size as fraction of capital
        """
        # Base size from confidence
        base_size = confidence * self.max_position_size

        # Adjust for stock-specific performance
        if stock_win_rate is not None and stock_win_rate < self.win_rate_threshold:
            # Linear penalty: reduce to 0 at 0% win rate
            performance_penalty = stock_win_rate / self.win_rate_threshold
            base_size *= performance_penalty
            logger.debug(f"Win rate penalty: {performance_penalty:.2f}")

        # Further reduce for negative average P&L
        if avg_pnl is not None and avg_pnl < 0:
            base_size *= 0.5  # Halve position size for losing stocks
            logger.debug(f"Negative P&L penalty: 0.5")

        # Reduce for high stock-level drawdown
        if stock_drawdown is not None and stock_drawdown > 0.15:
            drawdown_penalty = 1 - (stock_drawdown - 0.15) / 0.25  # 0-25% above 15%
            drawdown_penalty = max(0.3, drawdown_penalty)  # Minimum 30%
            base_size *= drawdown_penalty
            logger.debug(f"Drawdown penalty: {drawdown_penalty:.2f}")

        return min(base_size, self.max_position_size)


class DrawdownCircuitBreaker:
    """
    Circuit breaker to halt trading when drawdowns exceed thresholds.

    Implements:
    - Portfolio-level halt at 15% drawdown
    - Stock-level halt at 25% drawdown
    """

    def __init__(self, max_portfolio_drawdown: float = 0.15,
                 max_stock_drawdown: float = 0.25,
                 cooldown_days: int = 5):
        """
        Initialize circuit breaker.

        Args:
            max_portfolio_drawdown: Halt all trading above this level (0.15 = 15%)
            max_stock_drawdown: Halt specific stock above this level (0.25 = 25%)
            cooldown_days: Days to wait before re-enabling halted stock
        """
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_stock_drawdown = max_stock_drawdown
        self.cooldown_days = cooldown_days

        self.halted_stocks: Dict[str, datetime] = {}  # stock -> halt_time
        self.portfolio_halted = False
        self.portfolio_halt_time: Optional[datetime] = None

        logger.info(f"Initialized DrawdownCircuitBreaker: portfolio={max_portfolio_drawdown:.0%}, stock={max_stock_drawdown:.0%}")

    def check_trading_halts(
        self,
        portfolio_drawdown: float,
        stock_drawdowns: Dict[str, float]
    ) -> Dict:
        """
        Check if trading should be halted.

        Args:
            portfolio_drawdown: Current portfolio drawdown (0-1)
            stock_drawdowns: Dict of {ticker: drawdown}

        Returns:
            Dict with halt status and details
        """
        now = datetime.now()

        # Check portfolio-level halt
        if portfolio_drawdown > self.max_portfolio_drawdown:
            if not self.portfolio_halted:
                self.portfolio_halted = True
                self.portfolio_halt_time = now
                logger.warning(f"PORTFOLIO HALT: drawdown {portfolio_drawdown:.1%} > {self.max_portfolio_drawdown:.0%}")
        else:
            self.portfolio_halted = False
            self.portfolio_halt_time = None

        # Check stock-level halts
        stock_halts = {}
        for stock, drawdown in stock_drawdowns.items():
            if drawdown > self.max_stock_drawdown:
                if stock not in self.halted_stocks:
                    self.halted_stocks[stock] = now
                    logger.warning(f"STOCK HALT {stock}: drawdown {drawdown:.1%} > {self.max_stock_drawdown:.0%}")
                stock_halts[stock] = 'HALTED'

        # Check cooldown expiration
        expired = []
        for stock, halt_time in self.halted_stocks.items():
            if (now - halt_time).days >= self.cooldown_days:
                if stock not in stock_drawdowns or stock_drawdowns[stock] <= self.max_stock_drawdown * 0.8:
                    expired.append(stock)
                    logger.info(f"STOCK COOLDOWN EXPIRED {stock}: trading re-enabled")

        for stock in expired:
            del self.halted_stocks[stock]

        return {
            'portfolio_halt': self.portfolio_halted,
            'portfolio_drawdown': portfolio_drawdown,
            'stock_halts': stock_halts,
            'halted_stocks': list(self.halted_stocks.keys())
        }

    def should_trade_stock(self, ticker: str) -> bool:
        """Check if a specific stock can be traded."""
        if self.portfolio_halted:
            return False
        return ticker not in self.halted_stocks

    def should_trade_portfolio(self) -> bool:
        """Check if portfolio-level trading is allowed."""
        return not self.portfolio_halted


class RealTimePerformanceMonitor:
    """
    Monitor real-time performance and calculate health scores for stocks.

    Tracks recent trades and provides health scores to guide position sizing.
    """

    def __init__(self, lookback_trades: int = 30):
        """
        Initialize performance monitor.

        Args:
            lookback_trades: Number of recent trades to consider
        """
        self.lookback_trades = lookback_trades
        self.performance_history: Dict[str, List[float]] = defaultdict(list)

        logger.info(f"Initialized RealTimePerformanceMonitor: lookback={lookback_trades} trades")

    def update_performance(self, ticker: str, pnl: float) -> None:
        """Record a trade result."""
        self.performance_history[ticker].append(pnl)

        # Keep only recent history
        if len(self.performance_history[ticker]) > self.lookback_trades:
            self.performance_history[ticker].pop(0)

    def get_stock_health(self, ticker: str) -> float:
        """
        Get current health score for a stock.

        Returns:
            Health score from 0.0 (very unhealthy) to 1.0 (very healthy)
        """
        if ticker not in self.performance_history:
            return 1.0  # Default healthy for new stocks

        recent_trades = self.performance_history[ticker]
        if len(recent_trades) < 5:
            return 1.0  # Not enough data

        win_rate = sum(1 for pnl in recent_trades if pnl > 0) / len(recent_trades)
        avg_pnl = np.mean(recent_trades)

        # Health score weighted by win rate and P&L
        health_score = (win_rate * 0.6 + min(avg_pnl / 1000 + 0.5, 1.0) * 0.4)

        return max(0.0, min(1.0, health_score))

    def get_all_health_scores(self) -> Dict[str, float]:
        """Get health scores for all tracked stocks."""
        return {ticker: self.get_stock_health(ticker)
                for ticker in self.performance_history.keys()}

    def get_stock_stats(self, ticker: str) -> Optional[Dict]:
        """Get detailed statistics for a stock."""
        if ticker not in self.performance_history:
            return None

        trades = self.performance_history[ticker]
        if len(trades) < 3:
            return None

        return {
            'win_rate': sum(1 for pnl in trades if pnl > 0) / len(trades),
            'avg_pnl': np.mean(trades),
            'total_pnl': sum(trades),
            'trades': len(trades),
            'health_score': self.get_stock_health(ticker),
            'recent_streak': self._get_recent_streak(trades)
        }

    def _get_recent_streak(self, trades: List[float]) -> int:
        """Get the current winning/losing streak (positive = wins, negative = losses)."""
        if not trades:
            return 0

        streak = 0
        is_winning = trades[-1] > 0

        for pnl in reversed(trades):
            if (pnl > 0) == is_winning:
                streak += 1 if is_winning else -1
            else:
                break

        return streak


class IntegratedRiskManager:
    """
    Integrated risk management combining all components.

    Provides a single interface for:
    - Stock filtering
    - Position sizing
    - Circuit breakers
    - Performance monitoring
    """

    def __init__(
        self,
        min_win_rate: float = 0.40,
        min_trades_for_filter: int = 10,
        max_position_size: float = 0.15,
        max_portfolio_drawdown: float = 0.15,
        max_stock_drawdown: float = 0.25
    ):
        """Initialize integrated risk manager."""
        self.stock_filter = DynamicStockFilter(
            min_win_rate=min_win_rate,
            min_trades=min_trades_for_filter
        )
        self.position_sizer = SmartPositionSizer(
            max_position_size=max_position_size
        )
        self.circuit_breaker = DrawdownCircuitBreaker(
            max_portfolio_drawdown=max_portfolio_drawdown,
            max_stock_drawdown=max_stock_drawdown
        )
        self.performance_monitor = RealTimePerformanceMonitor()

        logger.info("Initialized IntegratedRiskManager")

    def should_trade(
        self,
        ticker: str,
        portfolio_drawdown: float = 0.0,
        stock_drawdown: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be allowed.

        Returns:
            Tuple of (should_trade, reason)
        """
        # Check circuit breaker
        if not self.circuit_breaker.should_trade_portfolio():
            return False, f"Portfolio halted (drawdown > {self.circuit_breaker.max_portfolio_drawdown:.0%})"

        if not self.circuit_breaker.should_trade_stock(ticker):
            return False, f"Stock halted due to high drawdown"

        # Check stock filter
        if not self.stock_filter.is_tradeable(ticker):
            return False, f"Stock auto-banned (poor performance)"

        # Check health score
        health = self.performance_monitor.get_stock_health(ticker)
        if health < 0.3:
            return False, f"Stock health too low ({health:.2f})"

        return True, "Approved"

    def get_position_size(
        self,
        ticker: str,
        confidence: float,
        portfolio_drawdown: float = 0.0
    ) -> float:
        """
        Get recommended position size for a trade.

        Args:
            ticker: Stock ticker
            confidence: Signal confidence (0-1)
            portfolio_drawdown: Current portfolio drawdown

        Returns:
            Position size as fraction of capital
        """
        # Get stock metrics
        stock_metrics = self.stock_filter.get_stock_metrics(ticker)
        stock_stats = self.performance_monitor.get_stock_stats(ticker)

        win_rate = None
        avg_pnl = None

        if stock_metrics:
            win_rate = stock_metrics.get('win_rate')
            avg_pnl = stock_metrics.get('avg_pnl')
        elif stock_stats:
            win_rate = stock_stats.get('win_rate')
            avg_pnl = stock_stats.get('avg_pnl')

        # Calculate position size with all adjustments
        position_size = self.position_sizer.calculate_position_size(
            confidence=confidence,
            stock_win_rate=win_rate,
            avg_pnl=avg_pnl
        )

        # Additional portfolio drawdown adjustment
        if portfolio_drawdown > 0.08:
            dd_penalty = 1 - (portfolio_drawdown - 0.08) / 0.12
            dd_penalty = max(0.3, dd_penalty)
            position_size *= dd_penalty

        return position_size

    def record_trade_result(self, ticker: str, pnl: float) -> None:
        """Record a trade result to all components."""
        self.stock_filter.record_trade(ticker, pnl)
        self.performance_monitor.update_performance(ticker, pnl)

    def update_drawdowns(
        self,
        portfolio_drawdown: float,
        stock_drawdowns: Dict[str, float]
    ) -> Dict:
        """Update circuit breaker with current drawdowns."""
        return self.circuit_breaker.check_trading_halts(
            portfolio_drawdown=portfolio_drawdown,
            stock_drawdowns=stock_drawdowns
        )

    def get_portfolio_status(self) -> Dict:
        """Get overall portfolio risk status."""
        return {
            'banned_stocks': self.stock_filter.get_banned_stocks(),
            'halted_stocks': list(self.circuit_breaker.halted_stocks.keys()),
            'portfolio_halted': self.circuit_breaker.portfolio_halted,
            'health_scores': self.performance_monitor.get_all_health_scores()
        }


if __name__ == "__main__":
    # Test the risk management module
    print("Testing Risk Management Module...")

    # Create integrated manager
    rm = IntegratedRiskManager()

    # Simulate some trades
    test_trades = [
        ('0700.HK', 500),
        ('0700.HK', -200),
        ('0700.HK', 300),
        ('3690.HK', -100),
        ('3690.HK', -150),
        ('3690.HK', -80),
        ('3690.HK', 50),
        ('3690.HK', -120),
        ('3690.HK', -90),
        ('3690.HK', -60),
        ('3690.HK', -110),
        ('3690.HK', -70),
        ('3690.HK', -130),  # 10 trades, should trigger ban
    ]

    for ticker, pnl in test_trades:
        rm.record_trade_result(ticker, pnl)
        print(f"Recorded: {ticker} P&L=${pnl}")

    print("\n--- Portfolio Status ---")
    status = rm.get_portfolio_status()
    print(f"Banned stocks: {status['banned_stocks']}")
    print(f"Health scores: {status['health_scores']}")

    print("\n--- Trade Approval Test ---")
    for ticker in ['0700.HK', '3690.HK', '2269.HK']:
        can_trade, reason = rm.should_trade(ticker)
        print(f"{ticker}: {'APPROVED' if can_trade else 'BLOCKED'} - {reason}")

    print("\n--- Position Size Test ---")
    for ticker in ['0700.HK', '2269.HK']:
        size = rm.get_position_size(ticker, confidence=0.7)
        print(f"{ticker}: Position size = {size:.1%}")
