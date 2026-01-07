"""
Performance Tracker Module
===========================

Comprehensive performance tracking system for recording trade history and calculating metrics.
Integrates with the existing risk management framework (Fixes 16-19).

Key Features:
    - TradeRecord: Data class for individual trade records
    - PerformanceTracker: Main class for tracking and analyzing trade performance
    - update_performance(): Records trade outcomes with full metrics
    - performance_history: Maintains historical performance data by period

Integration Points:
    - MarketRegimeDetector: Tracks performance by market regime
    - PositionAdjuster: Provides adaptive position sizing based on performance
    - MockTrade/Portfolio: SQLAlchemy models for persistence

Last Updated: 2025-12-03
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import pickle
from pathlib import Path
from collections import defaultdict

# Import existing risk management components
try:
    from ..risk_management.market_regime_detector import MarketRegime
except ImportError:
    # Define locally if import fails
    class MarketRegime(Enum):
        BULL = 'bull'
        BEAR = 'bear'
        RISK_ON = 'risk_on'
        RISK_OFF = 'risk_off'
        VOLATILE = 'volatile'
        SIDEWAYS = 'sideways'
        INFLATION = 'inflation'
        DEFLATION = 'deflation'
        CRISIS = 'crisis'


class TradeResult(Enum):
    """Trade outcome classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


@dataclass
class TradeRecord:
    """
    Data class for individual trade records.

    Tracks all relevant trade information including:
    - Entry/exit prices and P&L
    - Market regime context
    - Position sizing and adjustments
    - Tags for categorization
    """
    trade_id: str
    timestamp: datetime
    asset: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    regime: str  # Market regime at entry
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_regime: Optional[str] = None  # Market regime at exit
    result: Optional[TradeResult] = None
    position_size: float = 1.0
    holding_period: Optional[timedelta] = None
    asset_type: str = "stock"  # stock, forex, crypto, commodity, jpy_pair, crude_oil
    tags: List[str] = field(default_factory=list)

    # Additional tracking fields
    signal_confidence: Optional[float] = None
    regime_confidence: Optional[float] = None
    adjustments_applied: List[str] = field(default_factory=list)
    blocked: bool = False
    blocking_reason: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.tags is None:
            self.tags = []
        if self.adjustments_applied is None:
            self.adjustments_applied = []
        if self.exit_price is not None and self.entry_price is not None:
            self._calculate_pnl()

    def _calculate_pnl(self):
        """Calculate P&L based on direction."""
        if self.entry_price is None or self.exit_price is None:
            return

        if self.direction.upper() == "BUY":
            self.pnl = self.exit_price - self.entry_price
        else:  # SELL
            self.pnl = self.entry_price - self.exit_price

        if self.entry_price > 0:
            self.pnl_percent = (self.pnl / self.entry_price) * 100
        else:
            self.pnl_percent = 0.0

        # Determine result
        if self.pnl is None:
            self.result = TradeResult.PENDING
        elif abs(self.pnl) < 0.0001:  # Essentially zero
            self.result = TradeResult.BREAKEVEN
        elif self.pnl > 0:
            self.result = TradeResult.WIN
        else:
            self.result = TradeResult.LOSS

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Handle non-serializable types
        data['timestamp'] = self.timestamp.isoformat()
        data['result'] = self.result.value if self.result else None
        if self.holding_period:
            data['holding_period'] = self.holding_period.total_seconds()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create TradeRecord from dictionary."""
        # Restore non-serializable types
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('result'):
            data['result'] = TradeResult(data['result'])
        if data.get('holding_period'):
            data['holding_period'] = timedelta(seconds=data['holding_period'])
        return cls(**data)


class PerformanceTracker:
    """
    Main class for tracking and analyzing trade performance.

    Key Features:
        - update_performance(): Records trade outcomes with full metrics
        - performance_history: Maintains historical performance data by period
        - Regime-based performance analysis
        - Asset-specific tracking
        - Adaptive position multiplier calculation
        - Persistence (JSON and pickle formats)
        - Comprehensive reporting
    """

    def __init__(
        self,
        max_history_days: int = 365,
        rolling_window: int = 50,
        storage_path: Optional[str] = None
    ):
        """
        Initialize performance tracker.

        Args:
            max_history_days: Maximum days to keep in history
            rolling_window: Window size for rolling statistics
            storage_path: Path to save/load performance data
        """
        self.max_history_days = max_history_days
        self.rolling_window = rolling_window
        self.storage_path = Path(storage_path) if storage_path else None

        # In-memory storage
        self.trade_history: List[TradeRecord] = []
        self.performance_history: List[Dict] = []  # Aggregated performance by period
        self._next_trade_id = 1

        # Performance tracking by category
        self._regime_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        )
        self._asset_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        )
        self._direction_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        )

        # Load existing data if available
        if self.storage_path and self.storage_path.exists():
            self.load_performance_data()

    def update_performance(
        self,
        trade_record: TradeRecord,
        calculate_metrics: bool = True
    ) -> None:
        """
        Update performance history with a new trade record.

        Args:
            trade_record: TradeRecord object with trade details
            calculate_metrics: Whether to recalculate performance metrics
        """
        # Generate trade ID if not provided
        if not trade_record.trade_id:
            trade_record.trade_id = self._generate_trade_id(trade_record.timestamp)

        # Add to history
        self.trade_history.append(trade_record)

        # Update categorical performance tracking
        self._update_categorical_performance(trade_record)

        # Clean old records
        self._clean_old_records()

        # Update aggregated performance history
        if calculate_metrics:
            self._update_performance_history()

        # Save to disk if storage path is set
        if self.storage_path:
            self.save_performance_data()

    def record_trade_entry(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        regime: str,
        position_size: float = 1.0,
        asset_type: str = "stock",
        signal_confidence: Optional[float] = None,
        regime_confidence: Optional[float] = None,
        tags: List[str] = None,
        adjustments_applied: List[str] = None
    ) -> str:
        """
        Record a new trade entry (when trade is opened).

        Args:
            asset: Asset ticker symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            regime: Market regime at entry
            position_size: Position size (0-1)
            asset_type: Type of asset
            signal_confidence: Signal confidence score
            regime_confidence: Regime detection confidence
            tags: List of tags for categorization
            adjustments_applied: List of adjustments applied

        Returns:
            trade_id: Unique identifier for the trade
        """
        trade_id = self._generate_trade_id(datetime.now())

        record = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.now(),
            asset=asset,
            direction=direction.upper(),
            entry_price=entry_price,
            regime=regime,
            position_size=position_size,
            asset_type=asset_type,
            signal_confidence=signal_confidence,
            regime_confidence=regime_confidence,
            tags=tags or [],
            adjustments_applied=adjustments_applied or []
        )

        self.trade_history.append(record)
        return trade_id

    def update_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_regime: Optional[str] = None
    ) -> bool:
        """
        Update an existing trade with exit information.

        Args:
            trade_id: Trade ID from record_trade_entry
            exit_price: Exit price
            exit_regime: Market regime at exit (optional)

        Returns:
            success: Whether the update was successful
        """
        for record in self.trade_history:
            if record.trade_id == trade_id and record.exit_price is None:
                record.exit_price = exit_price
                record.exit_regime = exit_regime or record.regime
                record.holding_period = datetime.now() - record.timestamp

                # Trigger P&L calculation
                record._calculate_pnl()

                # Update categorical performance
                self._update_categorical_performance(record)

                # Update performance history
                self._update_performance_history()

                if self.storage_path:
                    self.save_performance_data()

                return True

        return False

    def get_performance_metrics(
        self,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        direction: Optional[str] = None,
        asset_type: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            asset: Filter by asset symbol
            regime: Filter by market regime
            direction: Filter by trade direction
            asset_type: Filter by asset type
            lookback_days: Only consider trades within this many days

        Returns:
            Dictionary of performance metrics
        """
        # Filter trades
        filtered_trades = self._filter_trades(
            asset=asset,
            regime=regime,
            direction=direction,
            asset_type=asset_type,
            lookback_days=lookback_days
        )

        if not filtered_trades:
            return self._get_empty_metrics()

        # Calculate metrics
        completed_trades = [t for t in filtered_trades if t.result is not None and t.result != TradeResult.PENDING]
        if not completed_trades:
            return self._get_empty_metrics()

        wins = [t for t in completed_trades if t.result == TradeResult.WIN]
        losses = [t for t in completed_trades if t.result == TradeResult.LOSS]

        total_trades = len(completed_trades)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total_trades if total_trades > 0 else 0

        # P&L metrics
        pnl_values = [t.pnl for t in completed_trades if t.pnl is not None]
        pnl_percent_values = [t.pnl_percent for t in completed_trades if t.pnl_percent is not None]

        total_pnl = sum(pnl_values) if pnl_values else 0
        total_pnl_percent = sum(pnl_percent_values) if pnl_percent_values else 0

        win_pnls = [t.pnl for t in wins if t.pnl is not None]
        loss_pnls = [t.pnl for t in losses if t.pnl is not None]

        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0

        win_pnl_percents = [t.pnl_percent for t in wins if t.pnl_percent is not None]
        loss_pnl_percents = [t.pnl_percent for t in losses if t.pnl_percent is not None]

        avg_win_percent = np.mean(win_pnl_percents) if win_pnl_percents else 0
        avg_loss_percent = np.mean(loss_pnl_percents) if loss_pnl_percents else 0

        # Risk metrics
        total_wins_pnl = sum(win_pnls) if win_pnls else 0
        total_losses_pnl = abs(sum(loss_pnls)) if loss_pnls else 0
        profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else float('inf')

        # Sharpe-like ratio (assuming risk-free rate = 0 for simplicity)
        if len(pnl_percent_values) > 1:
            sharpe_ratio = np.mean(pnl_percent_values) / np.std(pnl_percent_values) if np.std(pnl_percent_values) > 0 else 0
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        if pnl_percent_values:
            cumulative_returns = np.cumsum(pnl_percent_values)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0

        # Kelly Criterion
        if win_rate > 0 and avg_win_percent > 0 and avg_loss_percent < 0:
            b = abs(avg_win_percent / avg_loss_percent)  # win/loss ratio
            kelly_fraction = win_rate - (1 - win_rate) / b
            kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp between 0 and 1
        else:
            kelly_fraction = 0

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_percent,
            "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0,
            "avg_pnl_percent": total_pnl_percent / total_trades if total_trades > 0 else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_percent": avg_win_percent,
            "avg_loss_percent": avg_loss_percent,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "kelly_fraction": kelly_fraction,
            "largest_win": max(win_pnls, default=0),
            "largest_loss": min(loss_pnls, default=0),
            "current_streak": self._calculate_current_streak(completed_trades)
        }

    def get_regime_performance(
        self,
        lookback_days: Optional[int] = 30
    ) -> Dict[str, Dict]:
        """
        Get performance metrics for each market regime.

        Returns:
            Dict mapping regime names to performance metrics
        """
        regimes = set(t.regime for t in self.trade_history if t.regime)
        regime_performance = {}

        for regime in regimes:
            metrics = self.get_performance_metrics(regime=regime, lookback_days=lookback_days)
            regime_performance[regime] = metrics

        return regime_performance

    def get_asset_performance(
        self,
        lookback_days: Optional[int] = 30
    ) -> Dict[str, Dict]:
        """
        Get performance metrics for each asset.

        Returns:
            Dict mapping asset symbols to performance metrics
        """
        assets = set(t.asset for t in self.trade_history)
        asset_performance = {}

        for asset in assets:
            metrics = self.get_performance_metrics(asset=asset, lookback_days=lookback_days)
            asset_performance[asset] = metrics

        return asset_performance

    def get_asset_type_performance(
        self,
        lookback_days: Optional[int] = 30
    ) -> Dict[str, Dict]:
        """
        Get performance metrics for each asset type.

        Returns:
            Dict mapping asset types to performance metrics
        """
        asset_types = set(t.asset_type for t in self.trade_history if t.asset_type)
        asset_type_performance = {}

        for asset_type in asset_types:
            metrics = self.get_performance_metrics(asset_type=asset_type, lookback_days=lookback_days)
            asset_type_performance[asset_type] = metrics

        return asset_type_performance

    def get_direction_performance(
        self,
        lookback_days: Optional[int] = 30
    ) -> Dict[str, Dict]:
        """
        Get performance metrics for BUY vs SELL directions.

        Returns:
            Dict mapping directions to performance metrics
        """
        direction_performance = {}
        for direction in ['BUY', 'SELL']:
            metrics = self.get_performance_metrics(direction=direction, lookback_days=lookback_days)
            direction_performance[direction] = metrics

        return direction_performance

    def get_adaptive_multiplier(
        self,
        asset: str,
        regime: str,
        direction: str,
        base_multiplier: float = 1.0,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate adaptive position multiplier based on historical performance.

        Args:
            asset: Asset symbol
            regime: Current market regime
            direction: Trade direction
            base_multiplier: Base multiplier to adjust
            lookback_days: Lookback period for performance

        Returns:
            Adjusted multiplier based on performance
        """
        metrics = self.get_performance_metrics(
            asset=asset,
            regime=regime,
            direction=direction,
            lookback_days=lookback_days
        )

        win_rate = metrics["win_rate"]
        profit_factor = metrics["profit_factor"]
        kelly_fraction = metrics["kelly_fraction"]

        # Adjust multiplier based on performance
        if win_rate > 0.6 and profit_factor > 1.5:
            # Strong performance - increase position
            multiplier = base_multiplier * 1.5
        elif win_rate < 0.4 or profit_factor < 0.8:
            # Poor performance - reduce position
            multiplier = base_multiplier * 0.5
        else:
            # Neutral performance - use Kelly fraction if available
            multiplier = base_multiplier * max(0.5, min(kelly_fraction * 2, 1.5))

        # Apply additional regime-specific adjustments
        regime_upper = regime.upper() if regime else ""
        if regime_upper in ["VOLATILE", "CRISIS"]:
            multiplier *= 0.7  # Reduce in volatile regimes
        elif regime_upper in ["BULL", "RISK_ON"] and direction.upper() == "BUY":
            multiplier *= 1.2  # Increase in favorable regimes

        return max(0.1, min(multiplier, 2.0))  # Clamp between 0.1 and 2.0

    def save_performance_data(self, filepath: Optional[str] = None) -> bool:
        """Save performance data to disk."""
        try:
            if filepath:
                path = Path(filepath)
            elif self.storage_path:
                path = self.storage_path
            else:
                return False

            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                "trade_history": [t.to_dict() for t in self.trade_history],
                "performance_history": self.performance_history,
                "next_trade_id": self._next_trade_id
            }

            # Save as JSON
            json_path = path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Also save as pickle for faster loading
            pkl_path = path.with_suffix('.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)

            return True

        except Exception as e:
            print(f"Error saving performance data: {e}")
            return False

    def load_performance_data(self, filepath: Optional[str] = None) -> bool:
        """Load performance data from disk."""
        try:
            if filepath:
                path = Path(filepath)
            elif self.storage_path:
                path = self.storage_path
            else:
                return False

            # Try pickle first (faster)
            pickle_path = path.with_suffix('.pkl')
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Fall back to JSON
                json_path = path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                else:
                    return False

            # Restore data
            self.trade_history = [TradeRecord.from_dict(t) for t in data.get("trade_history", [])]
            self.performance_history = data.get("performance_history", [])
            self._next_trade_id = data.get("next_trade_id", len(self.trade_history) + 1)

            # Rebuild categorical performance
            for trade in self.trade_history:
                if trade.result and trade.result != TradeResult.PENDING:
                    self._update_categorical_performance(trade)

            return True

        except Exception as e:
            print(f"Error loading performance data: {e}")
            return False

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export trade history to pandas DataFrame."""
        data = [t.to_dict() for t in self.trade_history]
        return pd.DataFrame(data)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        report_lines = []

        # Header
        report_lines.append("=" * 60)
        report_lines.append("PERFORMANCE REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Trades: {len(self.trade_history)}")
        report_lines.append("=" * 60)

        # Overall Performance
        overall_metrics = self.get_performance_metrics()
        report_lines.append("\nOVERALL PERFORMANCE:")
        report_lines.append(f"  Win Rate: {overall_metrics['win_rate']:.2%}")
        report_lines.append(f"  Total P&L: ${overall_metrics['total_pnl']:.2f}")
        report_lines.append(f"  Profit Factor: {overall_metrics['profit_factor']:.2f}")
        report_lines.append(f"  Sharpe Ratio: {overall_metrics['sharpe_ratio']:.2f}")
        report_lines.append(f"  Max Drawdown: {overall_metrics['max_drawdown']:.2f}%")
        report_lines.append(f"  Kelly Fraction: {overall_metrics['kelly_fraction']:.2%}")

        # Regime Performance
        report_lines.append("\nPERFORMANCE BY REGIME:")
        regime_perf = self.get_regime_performance()
        for regime, metrics in sorted(regime_perf.items()):
            if metrics['total_trades'] > 0:
                report_lines.append(
                    f"  {regime}: {metrics['win_rate']:.2%} "
                    f"({metrics['total_trades']} trades, P&L: ${metrics['total_pnl']:.2f})"
                )

        # Direction Performance
        report_lines.append("\nPERFORMANCE BY DIRECTION:")
        dir_perf = self.get_direction_performance()
        for direction, metrics in dir_perf.items():
            if metrics['total_trades'] > 0:
                report_lines.append(
                    f"  {direction}: {metrics['win_rate']:.2%} "
                    f"({metrics['total_trades']} trades, P&L: ${metrics['total_pnl']:.2f})"
                )

        # Asset Type Performance
        report_lines.append("\nPERFORMANCE BY ASSET TYPE:")
        asset_type_perf = self.get_asset_type_performance()
        for asset_type, metrics in sorted(asset_type_perf.items()):
            if metrics['total_trades'] > 0:
                report_lines.append(
                    f"  {asset_type}: {metrics['win_rate']:.2%} "
                    f"({metrics['total_trades']} trades)"
                )

        # Asset Performance (top 10)
        report_lines.append("\nTOP PERFORMING ASSETS:")
        asset_perf = self.get_asset_performance()
        sorted_assets = sorted(
            asset_perf.items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )[:10]
        for asset, metrics in sorted_assets:
            if metrics['total_trades'] > 0:
                report_lines.append(
                    f"  {asset}: {metrics['win_rate']:.2%} "
                    f"(P&L: ${metrics['total_pnl']:.2f})"
                )

        # Recent Trades
        report_lines.append("\nRECENT TRADES (last 10):")
        recent_trades = sorted(
            self.trade_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        for trade in recent_trades:
            result_str = trade.result.value if trade.result else "Pending"
            pnl_str = f"${trade.pnl:.2f}" if trade.pnl is not None else "N/A"
            report_lines.append(
                f"  {trade.timestamp.date()} {trade.asset} {trade.direction}: "
                f"{result_str} (P&L: {pnl_str})"
            )

        report = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)

        return report

    # ==================== Private Methods ====================

    def _generate_trade_id(self, timestamp: datetime) -> str:
        """Generate a unique trade ID."""
        trade_id = f"TRADE_{self._next_trade_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        self._next_trade_id += 1
        return trade_id

    def _update_categorical_performance(self, trade: TradeRecord) -> None:
        """Update performance tracking by category."""
        if trade.result is None or trade.result == TradeResult.PENDING:
            return

        is_win = trade.result == TradeResult.WIN
        pnl = trade.pnl or 0

        # Update regime performance
        if trade.regime:
            regime_data = self._regime_performance[trade.regime]
            regime_data['wins'] += 1 if is_win else 0
            regime_data['losses'] += 0 if is_win else 1
            regime_data['total_pnl'] += pnl
            regime_data['trades'].append(trade.trade_id)

        # Update asset performance
        if trade.asset:
            asset_data = self._asset_performance[trade.asset]
            asset_data['wins'] += 1 if is_win else 0
            asset_data['losses'] += 0 if is_win else 1
            asset_data['total_pnl'] += pnl
            asset_data['trades'].append(trade.trade_id)

        # Update direction performance
        direction_data = self._direction_performance[trade.direction]
        direction_data['wins'] += 1 if is_win else 0
        direction_data['losses'] += 0 if is_win else 1
        direction_data['total_pnl'] += pnl
        direction_data['trades'].append(trade.trade_id)

    def _filter_trades(
        self,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        direction: Optional[str] = None,
        asset_type: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> List[TradeRecord]:
        """Filter trades based on criteria."""
        filtered = self.trade_history

        if asset:
            filtered = [t for t in filtered if t.asset == asset]

        if regime:
            filtered = [t for t in filtered if t.regime == regime]

        if direction:
            filtered = [t for t in filtered if t.direction.upper() == direction.upper()]

        if asset_type:
            filtered = [t for t in filtered if t.asset_type == asset_type]

        if lookback_days:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            filtered = [t for t in filtered if t.timestamp >= cutoff_date]

        return filtered

    def _clean_old_records(self) -> None:
        """Remove records older than max_history_days."""
        if not self.max_history_days:
            return

        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        self.trade_history = [
            t for t in self.trade_history
            if t.timestamp >= cutoff_date or t.exit_price is None  # Keep pending trades
        ]

    def _update_performance_history(self) -> None:
        """Update aggregated performance history by day."""
        if not self.trade_history:
            return

        # Group by day
        daily_performance = {}
        for trade in self.trade_history:
            if trade.exit_price is None:  # Skip pending trades
                continue

            date_str = trade.timestamp.strftime('%Y-%m-%d')
            if date_str not in daily_performance:
                daily_performance[date_str] = {
                    'date': date_str,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'total_pnl_percent': 0
                }

            daily = daily_performance[date_str]
            daily['trades'] += 1
            if trade.result == TradeResult.WIN:
                daily['wins'] += 1
            elif trade.result == TradeResult.LOSS:
                daily['losses'] += 1

            daily['total_pnl'] += trade.pnl or 0
            daily['total_pnl_percent'] += trade.pnl_percent or 0

        # Convert to list and update
        self.performance_history = sorted(
            daily_performance.values(),
            key=lambda x: x['date']
        )

    def _calculate_current_streak(self, trades: List[TradeRecord]) -> int:
        """
        Calculate current win/loss streak.

        Returns:
            Positive for winning streak, negative for losing streak
        """
        if not trades:
            return 0

        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)

        streak = 0
        current_result = sorted_trades[-1].result

        for trade in reversed(sorted_trades):
            if trade.result == current_result:
                streak += 1 if current_result == TradeResult.WIN else -1
            else:
                break

        return streak

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_pnl_percent": 0,
            "avg_pnl": 0,
            "avg_pnl_percent": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "avg_win_percent": 0,
            "avg_loss_percent": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "kelly_fraction": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "current_streak": 0
        }


# ==================== Convenience Functions ====================

def create_performance_tracker(
    storage_path: Optional[str] = None,
    max_history_days: int = 365,
    rolling_window: int = 50
) -> PerformanceTracker:
    """
    Create a configured performance tracker.

    Args:
        storage_path: Path for data persistence
        max_history_days: Maximum days to keep
        rolling_window: Window for rolling statistics

    Returns:
        Configured PerformanceTracker instance
    """
    return PerformanceTracker(
        storage_path=storage_path,
        max_history_days=max_history_days,
        rolling_window=rolling_window
    )


if __name__ == '__main__':
    # Test the performance tracker
    print("=" * 80)
    print("PERFORMANCE TRACKER - TEST")
    print("=" * 80)

    tracker = PerformanceTracker()

    # Add sample trades
    sample_trades = [
        TradeRecord(
            trade_id="TEST_001",
            timestamp=datetime.now() - timedelta(days=5),
            asset="AAPL",
            direction="BUY",
            entry_price=150.0,
            exit_price=155.0,
            regime="BULL",
            asset_type="stock"
        ),
        TradeRecord(
            trade_id="TEST_002",
            timestamp=datetime.now() - timedelta(days=4),
            asset="USDJPY=X",
            direction="SELL",
            entry_price=150.0,
            exit_price=148.0,
            regime="BEAR",
            asset_type="jpy_pair"
        ),
        TradeRecord(
            trade_id="TEST_003",
            timestamp=datetime.now() - timedelta(days=3),
            asset="CL=F",
            direction="BUY",
            entry_price=75.0,
            exit_price=73.0,
            regime="VOLATILE",
            asset_type="crude_oil"
        ),
        TradeRecord(
            trade_id="TEST_004",
            timestamp=datetime.now() - timedelta(days=2),
            asset="BTC-USD",
            direction="BUY",
            entry_price=50000.0,
            exit_price=52000.0,
            regime="RISK_ON",
            asset_type="cryptocurrency"
        ),
        TradeRecord(
            trade_id="TEST_005",
            timestamp=datetime.now() - timedelta(days=1),
            asset="GOOGL",
            direction="SELL",
            entry_price=140.0,
            exit_price=138.0,
            regime="BULL",
            asset_type="stock"
        ),
    ]

    for trade in sample_trades:
        tracker.update_performance(trade, calculate_metrics=False)

    # Update metrics once at end
    tracker._update_performance_history()

    # Print report
    print(tracker.generate_report())

    # Test adaptive multiplier
    print("\n" + "=" * 80)
    print("ADAPTIVE MULTIPLIER TEST")
    print("=" * 80)

    test_cases = [
        ("AAPL", "BULL", "BUY"),
        ("USDJPY=X", "BEAR", "SELL"),
        ("CL=F", "VOLATILE", "BUY"),
    ]

    for asset, regime, direction in test_cases:
        mult = tracker.get_adaptive_multiplier(asset, regime, direction)
        print(f"  {asset} {direction} in {regime}: {mult:.2f}x")

    print("\n" + "=" * 80)
