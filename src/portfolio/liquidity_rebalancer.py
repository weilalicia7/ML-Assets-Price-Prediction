"""
Phase 6: Liquidity-Aware Rebalancing

This module provides intelligent portfolio rebalancing that minimizes
market impact and transaction costs.

Components:
- LiquidityAwareRebalancer: Main rebalancing engine
- RebalanceTrigger: Determines when to rebalance
- ExecutionOptimizer: Optimizes trade execution
- SlippageTracker: Tracks and predicts slippage

Expected Impact: +0.5-1% profit rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque


class TriggerType(Enum):
    """Types of rebalance triggers."""
    DRIFT = "drift"
    TIME = "time"
    REGIME = "regime"
    RISK = "risk"
    OPPORTUNITY = "opportunity"


class ExecutionStrategy(Enum):
    """Trade execution strategies."""
    IMMEDIATE = "immediate"
    TWAP = "twap"           # Time-Weighted Average Price
    VWAP = "vwap"           # Volume-Weighted Average Price
    ADAPTIVE = "adaptive"


class UrgencyLevel(Enum):
    """Trade urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing."""
    drift_threshold: float = 0.05               # 5% position drift
    min_rebalance_interval_hours: int = 24      # Minimum 24h between rebalances
    max_daily_turnover: float = 0.20            # 20% max daily turnover
    market_impact_coefficient: float = 0.3      # Impact model k parameter
    max_single_trade_volume_pct: float = 0.05   # 5% of daily volume per trade
    min_trade_value: float = 1000               # Minimum trade value
    slippage_lookback_days: int = 30            # Days for slippage estimation


@dataclass
class TradeOrder:
    """Represents a trade order."""
    ticker: str
    direction: str                    # 'buy' or 'sell'
    quantity: float                   # Shares or value
    target_price: float
    urgency: UrgencyLevel
    strategy: ExecutionStrategy
    max_slippage: float               # Maximum acceptable slippage
    execution_window_minutes: int      # Time to execute


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    ticker: str
    expected_price: float
    actual_price: float
    quantity: float
    slippage_bps: float              # Slippage in basis points
    market_impact_bps: float
    execution_time_seconds: float
    strategy_used: str
    success: bool


@dataclass
class RebalanceDecision:
    """Decision on whether and how to rebalance."""
    should_rebalance: bool
    trigger_type: Optional[TriggerType]
    trigger_value: float              # e.g., drift amount, time since last
    trades: List[TradeOrder]
    estimated_cost: float             # Total estimated cost
    estimated_impact: float           # Estimated market impact
    rationale: str


# =============================================================================
# 1. Slippage Tracker
# =============================================================================

class SlippageTracker:
    """
    Tracks and analyzes execution slippage for better predictions.
    """

    def __init__(self, lookback_days: int = 30):
        """
        Initialize tracker.

        Args:
            lookback_days: Days of history to maintain
        """
        self.lookback_days = lookback_days
        self.execution_history: Dict[str, deque] = {}
        self.impact_model_params: Dict[str, Dict] = {}

    def record_execution(
        self,
        ticker: str,
        expected_price: float,
        actual_price: float,
        quantity: float,
        daily_volume: float,
        volatility: float
    ) -> ExecutionResult:
        """
        Record a trade execution for analysis.

        Args:
            ticker: Ticker symbol
            expected_price: Decision price
            actual_price: Execution price
            quantity: Quantity traded
            daily_volume: Average daily volume
            volatility: Asset volatility

        Returns:
            ExecutionResult with slippage analysis
        """
        slippage = (actual_price - expected_price) / expected_price
        slippage_bps = slippage * 10000

        # Estimate market impact component
        volume_ratio = abs(quantity) / daily_volume if daily_volume > 0 else 0
        expected_impact = volatility * np.sqrt(volume_ratio) * self.impact_model_params.get(
            ticker, {'k': 0.3}
        ).get('k', 0.3)
        market_impact_bps = expected_impact * 10000

        result = ExecutionResult(
            ticker=ticker,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            execution_time_seconds=0,
            strategy_used='unknown',
            success=True
        )

        # Store in history
        if ticker not in self.execution_history:
            self.execution_history[ticker] = deque(maxlen=100)
        self.execution_history[ticker].append(result)

        # Update impact model
        self._update_impact_model(ticker)

        return result

    def _update_impact_model(self, ticker: str):
        """Update market impact model parameters from history."""
        if ticker not in self.execution_history:
            return

        history = list(self.execution_history[ticker])
        if len(history) < 5:
            return

        # Simple average of realized impacts
        avg_slippage = np.mean([h.slippage_bps for h in history])
        avg_impact = np.mean([h.market_impact_bps for h in history])

        self.impact_model_params[ticker] = {
            'k': max(0.1, min(0.5, avg_impact / 10000)),
            'avg_slippage': avg_slippage,
            'last_update': datetime.now()
        }

    def get_slippage_forecast(
        self,
        ticker: str,
        trade_size: float,
        daily_volume: float,
        volatility: float
    ) -> float:
        """
        Forecast expected slippage for a trade.

        Args:
            ticker: Ticker symbol
            trade_size: Trade size (value or shares)
            daily_volume: Average daily volume
            volatility: Asset volatility

        Returns:
            Expected slippage as decimal (0.001 = 10 bps)
        """
        volume_ratio = trade_size / daily_volume if daily_volume > 0 else 0.1

        # Get asset-specific parameters or defaults
        params = self.impact_model_params.get(ticker, {'k': 0.3, 'avg_slippage': 10})

        # Square-root market impact model
        impact = params['k'] * volatility * np.sqrt(volume_ratio)

        # Add historical average slippage
        historical_component = params.get('avg_slippage', 10) / 10000

        return impact + historical_component

    def calculate_slippage_stats(
        self,
        ticker: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate slippage statistics.

        Args:
            ticker: Optional specific ticker (all if None)
            lookback_days: Optional lookback override

        Returns:
            Dictionary of slippage statistics
        """
        lookback = lookback_days or self.lookback_days

        if ticker:
            history = list(self.execution_history.get(ticker, []))
        else:
            history = []
            for h in self.execution_history.values():
                history.extend(list(h))

        if not history:
            return {
                'mean_slippage_bps': 0.0,
                'std_slippage_bps': 0.0,
                'max_slippage_bps': 0.0,
                'execution_count': 0
            }

        slippages = [h.slippage_bps for h in history]

        return {
            'mean_slippage_bps': np.mean(slippages),
            'std_slippage_bps': np.std(slippages),
            'max_slippage_bps': np.max(np.abs(slippages)),
            'execution_count': len(slippages)
        }


# =============================================================================
# 2. Execution Optimizer
# =============================================================================

class ExecutionOptimizer:
    """
    Optimizes trade execution to minimize costs and market impact.
    """

    def __init__(self, config: RebalanceConfig):
        """
        Initialize optimizer.

        Args:
            config: Rebalancing configuration
        """
        self.config = config

        # Urgency parameters
        self.urgency_params = {
            UrgencyLevel.LOW: {'patience': 1.0, 'max_impact': 0.001},
            UrgencyLevel.MEDIUM: {'patience': 0.5, 'max_impact': 0.002},
            UrgencyLevel.HIGH: {'patience': 0.2, 'max_impact': 0.005},
            UrgencyLevel.CRITICAL: {'patience': 0.0, 'max_impact': 0.01}
        }

    def calculate_optimal_trade_size(
        self,
        total_quantity: float,
        daily_volume: float,
        volatility: float,
        urgency: UrgencyLevel
    ) -> Tuple[float, int]:
        """
        Calculate optimal trade size and number of slices.

        Args:
            total_quantity: Total quantity to trade
            daily_volume: Average daily volume
            volatility: Asset volatility
            urgency: Trade urgency

        Returns:
            Tuple of (optimal_slice_size, num_slices)
        """
        max_volume_pct = self.config.max_single_trade_volume_pct
        patience = self.urgency_params[urgency]['patience']

        # Adjust max participation based on urgency
        adjusted_max = max_volume_pct * (1 + (1 - patience))

        # Calculate max size per slice
        max_slice = daily_volume * adjusted_max

        if total_quantity <= max_slice:
            return total_quantity, 1

        # Calculate number of slices needed
        num_slices = int(np.ceil(total_quantity / max_slice))

        # Limit slices based on urgency
        max_slices = int(10 * patience) + 1
        num_slices = min(num_slices, max_slices)

        optimal_slice = total_quantity / num_slices

        return optimal_slice, num_slices

    def select_execution_strategy(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        urgency: UrgencyLevel
    ) -> ExecutionStrategy:
        """
        Select optimal execution strategy.

        Args:
            trade_size: Trade size
            daily_volume: Daily volume
            volatility: Asset volatility
            urgency: Trade urgency

        Returns:
            Recommended execution strategy
        """
        volume_ratio = trade_size / daily_volume if daily_volume > 0 else 1.0

        if urgency == UrgencyLevel.CRITICAL:
            return ExecutionStrategy.IMMEDIATE
        elif volume_ratio < 0.01:
            return ExecutionStrategy.IMMEDIATE
        elif volume_ratio < 0.05:
            return ExecutionStrategy.TWAP
        elif volatility > 0.03:  # High volatility
            return ExecutionStrategy.ADAPTIVE
        else:
            return ExecutionStrategy.VWAP

    def schedule_trades_twap(
        self,
        total_quantity: float,
        duration_minutes: int,
        num_intervals: int
    ) -> List[Tuple[int, float]]:
        """
        Create TWAP execution schedule.

        Args:
            total_quantity: Total quantity to trade
            duration_minutes: Total execution duration
            num_intervals: Number of time intervals

        Returns:
            List of (minute_offset, quantity) tuples
        """
        interval_duration = duration_minutes / num_intervals
        quantity_per_interval = total_quantity / num_intervals

        schedule = []
        for i in range(num_intervals):
            minute_offset = int(i * interval_duration)
            schedule.append((minute_offset, quantity_per_interval))

        return schedule

    def schedule_trades_vwap(
        self,
        total_quantity: float,
        volume_profile: List[float]
    ) -> List[Tuple[int, float]]:
        """
        Create VWAP execution schedule based on volume profile.

        Args:
            total_quantity: Total quantity to trade
            volume_profile: Relative volume by time bucket

        Returns:
            List of (bucket_index, quantity) tuples
        """
        # Normalize volume profile
        total_volume = sum(volume_profile)
        if total_volume == 0:
            # Equal distribution
            n = len(volume_profile)
            return [(i, total_quantity / n) for i in range(n)]

        schedule = []
        for i, vol in enumerate(volume_profile):
            proportion = vol / total_volume
            quantity = total_quantity * proportion
            if quantity > 0:
                schedule.append((i, quantity))

        return schedule

    def estimate_execution_cost(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        spread: float,
        strategy: ExecutionStrategy
    ) -> Dict[str, float]:
        """
        Estimate total execution cost.

        Args:
            trade_size: Trade size
            daily_volume: Daily volume
            volatility: Asset volatility
            spread: Bid-ask spread
            strategy: Execution strategy

        Returns:
            Dictionary of cost components
        """
        volume_ratio = trade_size / daily_volume if daily_volume > 0 else 0.1

        # Spread cost (always pay half spread)
        spread_cost = spread / 2

        # Market impact (square-root model)
        k = self.config.market_impact_coefficient
        market_impact = k * volatility * np.sqrt(volume_ratio)

        # Strategy-specific adjustments
        strategy_multiplier = {
            ExecutionStrategy.IMMEDIATE: 1.5,   # Higher impact
            ExecutionStrategy.TWAP: 0.8,        # Reduced impact
            ExecutionStrategy.VWAP: 0.7,        # Lower impact
            ExecutionStrategy.ADAPTIVE: 0.75    # Optimized
        }.get(strategy, 1.0)

        adjusted_impact = market_impact * strategy_multiplier

        # Timing risk (for slower strategies)
        timing_risk = 0.0
        if strategy in [ExecutionStrategy.TWAP, ExecutionStrategy.VWAP]:
            timing_risk = volatility * 0.1  # 10% of daily vol

        total_cost = spread_cost + adjusted_impact + timing_risk

        return {
            'spread_cost': spread_cost,
            'market_impact': adjusted_impact,
            'timing_risk': timing_risk,
            'total_cost': total_cost,
            'total_cost_bps': total_cost * 10000
        }


# =============================================================================
# 3. Rebalance Trigger
# =============================================================================

class RebalanceTrigger:
    """
    Determines when to trigger portfolio rebalancing.
    """

    def __init__(self, config: RebalanceConfig):
        """
        Initialize trigger.

        Args:
            config: Rebalancing configuration
        """
        self.config = config
        self.last_rebalance_time: Optional[datetime] = None
        self.last_regime: Optional[str] = None

    def check_drift_threshold(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Check if position drift exceeds threshold.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Tuple of (should_trigger, max_drift)
        """
        max_drift = 0.0

        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

        should_trigger = max_drift > self.config.drift_threshold

        return should_trigger, max_drift

    def check_time_based(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """
        Check if enough time has passed since last rebalance.

        Args:
            current_time: Current time (default: now)

        Returns:
            Tuple of (should_trigger, hours_since_last)
        """
        current_time = current_time or datetime.now()

        if self.last_rebalance_time is None:
            return True, float('inf')

        hours_since = (current_time - self.last_rebalance_time).total_seconds() / 3600

        # Only suggest time-based if minimum interval passed
        should_trigger = hours_since >= self.config.min_rebalance_interval_hours

        return should_trigger, hours_since

    def check_regime_change(
        self,
        current_regime: str
    ) -> Tuple[bool, str]:
        """
        Check if market regime has changed.

        Args:
            current_regime: Current market regime

        Returns:
            Tuple of (should_trigger, regime_change_type)
        """
        if self.last_regime is None:
            self.last_regime = current_regime
            return False, "initial"

        if current_regime != self.last_regime:
            change_type = f"{self.last_regime}_to_{current_regime}"
            return True, change_type

        return False, "none"

    def check_risk_breach(
        self,
        portfolio_risk: float,
        risk_limit: float
    ) -> Tuple[bool, float]:
        """
        Check if risk limit has been breached.

        Args:
            portfolio_risk: Current portfolio risk
            risk_limit: Maximum allowed risk

        Returns:
            Tuple of (should_trigger, breach_amount)
        """
        if portfolio_risk > risk_limit:
            breach = portfolio_risk - risk_limit
            return True, breach

        return False, 0.0

    def combined_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_regime: str,
        portfolio_risk: float,
        risk_limit: float,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, TriggerType, float]:
        """
        Combined trigger check with priority.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights
            current_regime: Current market regime
            portfolio_risk: Current portfolio risk
            risk_limit: Risk limit
            current_time: Current time

        Returns:
            Tuple of (should_trigger, trigger_type, trigger_value)
        """
        # Priority 1: Risk breach (critical)
        risk_trigger, risk_value = self.check_risk_breach(portfolio_risk, risk_limit)
        if risk_trigger:
            return True, TriggerType.RISK, risk_value

        # Priority 2: Regime change (high)
        regime_trigger, regime_value = self.check_regime_change(current_regime)
        if regime_trigger:
            return True, TriggerType.REGIME, 1.0

        # Check time constraint for lower priority triggers
        time_ok, hours_since = self.check_time_based(current_time)

        if not time_ok:
            return False, TriggerType.TIME, hours_since

        # Priority 3: Drift (medium)
        drift_trigger, drift_value = self.check_drift_threshold(
            current_weights, target_weights
        )
        if drift_trigger:
            return True, TriggerType.DRIFT, drift_value

        return False, TriggerType.DRIFT, drift_value

    def record_rebalance(self, time: Optional[datetime] = None, regime: Optional[str] = None):
        """Record that a rebalance occurred."""
        self.last_rebalance_time = time or datetime.now()
        if regime:
            self.last_regime = regime


# =============================================================================
# 4. Liquidity-Aware Rebalancer
# =============================================================================

class LiquidityAwareRebalancer:
    """
    Main rebalancing engine with liquidity awareness.

    Combines trigger logic, execution optimization, and slippage tracking
    to minimize rebalancing costs.
    """

    def __init__(self, config: Optional[RebalanceConfig] = None):
        """
        Initialize rebalancer.

        Args:
            config: Rebalancing configuration
        """
        self.config = config or RebalanceConfig()
        self.trigger = RebalanceTrigger(self.config)
        self.executor = ExecutionOptimizer(self.config)
        self.slippage_tracker = SlippageTracker(self.config.slippage_lookback_days)

    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        liquidity_data: Dict[str, Dict[str, float]]
    ) -> List[TradeOrder]:
        """
        Calculate trades needed for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            liquidity_data: Liquidity info per ticker
                {ticker: {'daily_volume': x, 'volatility': y, 'spread': z}}

        Returns:
            List of TradeOrder objects
        """
        trades = []
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            diff = target - current

            if abs(diff) < 0.001:  # Less than 0.1% change
                continue

            trade_value = abs(diff) * portfolio_value

            if trade_value < self.config.min_trade_value:
                continue

            # Get liquidity data
            liq = liquidity_data.get(ticker, {
                'daily_volume': 1000000,
                'volatility': 0.02,
                'spread': 0.001
            })

            daily_volume = liq.get('daily_volume', 1000000)
            volatility = liq.get('volatility', 0.02)

            # Determine urgency based on drift magnitude
            if abs(diff) > 0.15:
                urgency = UrgencyLevel.HIGH
            elif abs(diff) > 0.10:
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW

            # Select execution strategy
            strategy = self.executor.select_execution_strategy(
                trade_value, daily_volume, volatility, urgency
            )

            # Estimate max acceptable slippage
            cost_est = self.executor.estimate_execution_cost(
                trade_value, daily_volume, volatility,
                liq.get('spread', 0.001), strategy
            )
            max_slippage = cost_est['total_cost'] * 2  # Allow 2x estimated cost

            # Execution window based on urgency
            window_minutes = {
                UrgencyLevel.LOW: 480,      # 8 hours
                UrgencyLevel.MEDIUM: 120,   # 2 hours
                UrgencyLevel.HIGH: 30,      # 30 minutes
                UrgencyLevel.CRITICAL: 5    # 5 minutes
            }.get(urgency, 60)

            trades.append(TradeOrder(
                ticker=ticker,
                direction='buy' if diff > 0 else 'sell',
                quantity=trade_value,
                target_price=0.0,  # To be filled with current price
                urgency=urgency,
                strategy=strategy,
                max_slippage=max_slippage,
                execution_window_minutes=window_minutes
            ))

        # Sort by urgency (most urgent first)
        urgency_order = {
            UrgencyLevel.CRITICAL: 0,
            UrgencyLevel.HIGH: 1,
            UrgencyLevel.MEDIUM: 2,
            UrgencyLevel.LOW: 3
        }
        trades.sort(key=lambda t: urgency_order.get(t.urgency, 3))

        return trades

    def estimate_market_impact(
        self,
        trades: List[TradeOrder],
        liquidity_data: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Estimate total market impact of trades.

        Args:
            trades: List of trade orders
            liquidity_data: Liquidity info per ticker

        Returns:
            Estimated total market impact as decimal
        """
        total_impact = 0.0
        total_value = sum(t.quantity for t in trades)

        if total_value == 0:
            return 0.0

        for trade in trades:
            liq = liquidity_data.get(trade.ticker, {
                'daily_volume': 1000000,
                'volatility': 0.02,
                'spread': 0.001
            })

            cost_est = self.executor.estimate_execution_cost(
                trade.quantity,
                liq.get('daily_volume', 1000000),
                liq.get('volatility', 0.02),
                liq.get('spread', 0.001),
                trade.strategy
            )

            # Weight by trade value
            weight = trade.quantity / total_value
            total_impact += cost_est['market_impact'] * weight

        return total_impact

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_regime: str,
        portfolio_risk: float,
        risk_limit: float,
        liquidity_data: Dict[str, Dict[str, float]],
        portfolio_value: float
    ) -> RebalanceDecision:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_regime: Current market regime
            portfolio_risk: Current portfolio risk
            risk_limit: Risk limit
            liquidity_data: Liquidity info per ticker
            portfolio_value: Total portfolio value

        Returns:
            RebalanceDecision with full analysis
        """
        # Check triggers
        should_trigger, trigger_type, trigger_value = self.trigger.combined_trigger(
            current_weights, target_weights, current_regime,
            portfolio_risk, risk_limit
        )

        if not should_trigger:
            return RebalanceDecision(
                should_rebalance=False,
                trigger_type=trigger_type,
                trigger_value=trigger_value,
                trades=[],
                estimated_cost=0.0,
                estimated_impact=0.0,
                rationale=f"No trigger met. Max drift: {trigger_value:.2%}"
            )

        # Calculate trades
        trades = self.calculate_rebalance_trades(
            current_weights, target_weights, portfolio_value, liquidity_data
        )

        if not trades:
            return RebalanceDecision(
                should_rebalance=False,
                trigger_type=trigger_type,
                trigger_value=trigger_value,
                trades=[],
                estimated_cost=0.0,
                estimated_impact=0.0,
                rationale="No significant trades needed"
            )

        # Estimate costs
        total_cost = 0.0
        for trade in trades:
            liq = liquidity_data.get(trade.ticker, {
                'daily_volume': 1000000,
                'volatility': 0.02,
                'spread': 0.001
            })
            cost_est = self.executor.estimate_execution_cost(
                trade.quantity,
                liq.get('daily_volume', 1000000),
                liq.get('volatility', 0.02),
                liq.get('spread', 0.001),
                trade.strategy
            )
            total_cost += cost_est['total_cost'] * trade.quantity

        estimated_impact = self.estimate_market_impact(trades, liquidity_data)

        # Check if turnover limit exceeded
        total_turnover = sum(t.quantity for t in trades) / portfolio_value
        if total_turnover > self.config.max_daily_turnover:
            # Reduce trades to meet limit
            scale_factor = self.config.max_daily_turnover / total_turnover
            for trade in trades:
                trade.quantity *= scale_factor
            total_cost *= scale_factor

        rationale = (
            f"Trigger: {trigger_type.value} ({trigger_value:.2%}). "
            f"{len(trades)} trades, est. cost: {total_cost:.2f}, "
            f"impact: {estimated_impact:.4%}"
        )

        return RebalanceDecision(
            should_rebalance=True,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            trades=trades,
            estimated_cost=total_cost,
            estimated_impact=estimated_impact,
            rationale=rationale
        )

    def execute_rebalance(
        self,
        decision: RebalanceDecision,
        current_regime: Optional[str] = None
    ) -> List[ExecutionResult]:
        """
        Execute rebalancing trades (simulation).

        In production, this would interface with broker API.

        Args:
            decision: Rebalance decision with trades
            current_regime: Current regime for recording

        Returns:
            List of execution results
        """
        results = []

        for trade in decision.trades:
            # Simulate execution with some slippage
            slippage = np.random.uniform(0, trade.max_slippage)
            actual_price = trade.target_price * (1 + slippage if trade.direction == 'buy' else 1 - slippage)

            result = ExecutionResult(
                ticker=trade.ticker,
                expected_price=trade.target_price,
                actual_price=actual_price,
                quantity=trade.quantity,
                slippage_bps=slippage * 10000,
                market_impact_bps=slippage * 5000,  # Assume half is impact
                execution_time_seconds=trade.execution_window_minutes * 60 / 2,
                strategy_used=trade.strategy.value,
                success=True
            )
            results.append(result)

        # Record rebalance
        self.trigger.record_rebalance(regime=current_regime)

        return results

    def get_rebalancing_stats(self) -> Dict[str, float]:
        """Get rebalancing performance statistics."""
        slippage_stats = self.slippage_tracker.calculate_slippage_stats()

        return {
            'mean_slippage_bps': slippage_stats['mean_slippage_bps'],
            'max_slippage_bps': slippage_stats['max_slippage_bps'],
            'execution_count': slippage_stats['execution_count'],
            'hours_since_last_rebalance': (
                (datetime.now() - self.trigger.last_rebalance_time).total_seconds() / 3600
                if self.trigger.last_rebalance_time else float('inf')
            )
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_liquidity_aware_rebalancer(
    config: Optional[Dict] = None
) -> LiquidityAwareRebalancer:
    """
    Create configured liquidity-aware rebalancer.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured LiquidityAwareRebalancer
    """
    if config:
        rebal_config = RebalanceConfig(**config)
    else:
        rebal_config = RebalanceConfig()

    return LiquidityAwareRebalancer(config=rebal_config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'TriggerType',
    'ExecutionStrategy',
    'UrgencyLevel',

    # Data Classes
    'RebalanceConfig',
    'TradeOrder',
    'ExecutionResult',
    'RebalanceDecision',

    # Core Classes
    'SlippageTracker',
    'ExecutionOptimizer',
    'RebalanceTrigger',
    'LiquidityAwareRebalancer',

    # Factory
    'create_liquidity_aware_rebalancer',
]
