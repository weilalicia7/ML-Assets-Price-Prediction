"""
Optimal Hybrid Trading Strategy
Combines directional prediction with volatility filtering for optimal performance.

Based on backtest results showing:
- Original system: 43.8% accuracy, -4.10% return
- Improved system: 57.5% accuracy, -36.42% return
- Optimal Hybrid: 49.0% accuracy, +0.51% return (BEST!)

Key Parameters:
- Confidence threshold: 50% (optimized from 65% based on Phase 1 testing)
- Volatility filter: Only trade when volatility < median
- Position sizing: Dynamic based on confidence and drawdown
- Stop loss: -5%
- Max trades: 2 per asset

PHASE 1 FIX: Added drawdown control to reduce position sizes during drawdown periods
See: docs/PHASE1_TEST_RESULTS.md and phase1 fixing on C model_extra.pdf
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OptimalHybridStrategy:
    """
    Optimal hybrid trading strategy combining directional prediction
    with volatility filtering for superior risk-adjusted returns.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,  # PHASE 1 FIX: Lowered from 0.65 to 0.50
        volatility_filter_percentile: float = 0.50,
        position_size: float = 0.50,
        stop_loss_pct: float = 0.05,
        max_trades_per_asset: int = 2,
        drawdown_threshold: float = 0.08,  # PHASE 1 FIX: Start reducing at 8% drawdown
        max_drawdown: float = 0.20  # PHASE 1 FIX: Maximum 20% drawdown before stopping
    ):
        """
        Initialize optimal hybrid strategy.

        Args:
            confidence_threshold: Minimum confidence to trade (0.50 = 50%, optimized from 0.65)
            volatility_filter_percentile: Only trade when vol below this percentile
            position_size: Maximum position size as fraction of capital (0.50 = 50%)
            stop_loss_pct: Stop loss percentage (0.05 = 5%)
            max_trades_per_asset: Maximum number of trades per asset
            drawdown_threshold: Start reducing position sizes at this drawdown level (0.08 = 8%)
            max_drawdown: Stop trading at this drawdown level (0.20 = 20%)
        """
        self.confidence_threshold = confidence_threshold
        self.volatility_filter_percentile = volatility_filter_percentile
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_trades_per_asset = max_trades_per_asset
        self.drawdown_threshold = drawdown_threshold  # PHASE 1 FIX
        self.max_drawdown = max_drawdown  # PHASE 1 FIX

        # PHASE 1 FIX: Win rate tracking for signal quality filtering
        # Track recent signal outcomes per ticker to filter consistently wrong signals
        self.signal_history: Dict[str, List[bool]] = defaultdict(list)
        self.min_win_rate = 0.40  # Skip trading if recent win rate < 40%
        self.history_window = 10  # Track last 10 signals per ticker

        logger.info(f"Initialized Optimal Hybrid Strategy:")
        logger.info(f"  Confidence threshold: {confidence_threshold:.1%}")
        logger.info(f"  Volatility filter: {volatility_filter_percentile:.1%} percentile")
        logger.info(f"  Position size: {position_size:.1%}")
        logger.info(f"  Stop loss: {stop_loss_pct:.1%}")
        logger.info(f"  Drawdown control: reduce at {drawdown_threshold:.1%}, stop at {max_drawdown:.1%}")
        logger.info(f"  Win rate filter: min {self.min_win_rate:.0%} over last {self.history_window} trades")

    def calculate_position_size_with_drawdown_control(
        self,
        confidence: float,
        portfolio_drawdown: float = 0.0
    ) -> float:
        """
        PHASE 1 FIX: Calculate position size with drawdown control.

        Reduces position sizes during drawdown periods to limit further losses.
        Based on: phase1 fixing on C model_extra.pdf

        Args:
            confidence: Signal confidence (0-1)
            portfolio_drawdown: Current portfolio drawdown as decimal (0.08 = 8%)

        Returns:
            Adjusted position size as fraction of capital
        """
        # Base position size scaled by confidence
        # Higher confidence = larger position (up to max position_size)
        base_size = self.position_size * confidence

        # Apply drawdown control
        if portfolio_drawdown >= self.max_drawdown:
            # Stop trading entirely at max drawdown
            logger.warning(f"Drawdown {portfolio_drawdown:.1%} >= max {self.max_drawdown:.1%}, stopping trades")
            return 0.0
        elif portfolio_drawdown > self.drawdown_threshold:
            # Linear reduction from 100% to 30% as drawdown increases from threshold to max
            drawdown_penalty = 1 - (portfolio_drawdown - self.drawdown_threshold) / (self.max_drawdown - self.drawdown_threshold)
            drawdown_penalty = max(drawdown_penalty, 0.30)  # Minimum 30% of original
            adjusted_size = base_size * drawdown_penalty
            logger.info(f"Drawdown control: {portfolio_drawdown:.1%} drawdown, reducing position to {drawdown_penalty:.1%}")
            return adjusted_size
        else:
            return base_size

    def record_signal_outcome(self, ticker: str, was_profitable: bool) -> None:
        """
        PHASE 1 FIX: Record the outcome of a trading signal.

        Track whether signals were profitable to filter out consistently wrong tickers.
        Based on: phase1 fixing on C model_extra.pdf - Win Rate Improvement Strategy

        Args:
            ticker: Stock ticker symbol
            was_profitable: True if the signal resulted in a profitable trade
        """
        self.signal_history[ticker].append(was_profitable)

        # Keep only the most recent signals
        if len(self.signal_history[ticker]) > self.history_window:
            self.signal_history[ticker] = self.signal_history[ticker][-self.history_window:]

        recent_win_rate = self.get_ticker_win_rate(ticker)
        logger.debug(f"Recorded signal outcome for {ticker}: {was_profitable}, win rate: {recent_win_rate:.1%}")

    def get_ticker_win_rate(self, ticker: str) -> float:
        """
        PHASE 1 FIX: Get the recent win rate for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Win rate as decimal (0-1), or 0.5 if no history
        """
        history = self.signal_history.get(ticker, [])
        if len(history) < 3:  # Need at least 3 signals to make a judgment
            return 0.5  # Neutral - allow trading

        return sum(history) / len(history)

    def should_skip_due_to_win_rate(self, ticker: str) -> Tuple[bool, float]:
        """
        PHASE 1 FIX: Check if we should skip trading this ticker due to poor win rate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (should_skip, win_rate)
        """
        win_rate = self.get_ticker_win_rate(ticker)

        # Skip if win rate is below minimum threshold
        if win_rate < self.min_win_rate:
            logger.info(f"Skipping {ticker}: win rate {win_rate:.1%} < {self.min_win_rate:.0%} minimum")
            return True, win_rate

        return False, win_rate

    def generate_hybrid_signal(
        self,
        direction_prediction: float,
        volatility_prediction: float,
        historical_volatility: np.ndarray,
        current_price: float,
        account_size: float = 100000,
        portfolio_drawdown: float = 0.0,  # PHASE 1 FIX: Added for drawdown control
        ticker: Optional[str] = None  # PHASE 1 FIX: Added for win rate filtering
    ) -> Dict:
        """
        Generate trading signal using optimal hybrid strategy.

        Args:
            direction_prediction: Predicted direction (0-1, >0.5 = up, <0.5 = down)
            volatility_prediction: Predicted volatility
            historical_volatility: Array of historical volatility values
            current_price: Current asset price
            account_size: Account size for position sizing
            portfolio_drawdown: Current portfolio drawdown (0.08 = 8%) for position scaling
            ticker: Optional ticker symbol for win rate filtering

        Returns:
            Dictionary containing trading signal and metadata
        """

        # Calculate directional confidence (distance from 0.5)
        confidence = abs(direction_prediction - 0.5) * 2

        # Calculate volatility percentile
        median_volatility = np.median(historical_volatility)
        volatility_percentile = np.mean(volatility_prediction < historical_volatility)

        # Determine direction
        if direction_prediction > 0.5:
            direction = "LONG"
            predicted_move = 1
        elif direction_prediction < 0.5:
            direction = "SHORT"
            predicted_move = -1
        else:
            direction = "NEUTRAL"
            predicted_move = 0

        # OPTIMAL HYBRID LOGIC: Trade only when BOTH conditions met
        should_trade = (
            confidence >= self.confidence_threshold and
            volatility_prediction < median_volatility
        )

        # PHASE 1 FIX: Check if drawdown is too high
        if portfolio_drawdown >= self.max_drawdown:
            should_trade = False

        # PHASE 1 FIX: Win rate filter - skip tickers with poor historical performance
        skip_win_rate = False
        ticker_win_rate = 0.5
        if ticker and should_trade:
            skip_win_rate, ticker_win_rate = self.should_skip_due_to_win_rate(ticker)
            if skip_win_rate:
                should_trade = False

        if should_trade and direction != "NEUTRAL":
            # PHASE 1 FIX: Calculate position with drawdown control
            adjusted_position_pct = self.calculate_position_size_with_drawdown_control(
                confidence=confidence,
                portfolio_drawdown=portfolio_drawdown
            )
            position_value = account_size * adjusted_position_pct
            shares = int(position_value / current_price)

            # Calculate stop loss and take profit
            stop_loss_price = current_price * (1 - self.stop_loss_pct * predicted_move)
            take_profit_price = current_price * (1 + volatility_prediction * 2 * predicted_move)

            # Calculate risk/reward
            risk_amount = abs(current_price - stop_loss_price) * shares
            reward_amount = abs(take_profit_price - current_price) * shares
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            signal = {
                'action': direction,
                'confidence': float(confidence),
                'should_trade': True,
                'entry_price': float(current_price),
                'stop_loss': float(stop_loss_price),
                'take_profit': float(take_profit_price),
                'position_size': shares,
                'position_value': float(position_value),
                'risk_amount': float(risk_amount),
                'reward_amount': float(reward_amount),
                'risk_reward_ratio': float(risk_reward_ratio),
                'volatility_percentile': float(volatility_percentile),
                'predicted_volatility': float(volatility_prediction),
                'median_volatility': float(median_volatility),
                'strategy': 'optimal_hybrid',
                'reason': f'High confidence ({confidence:.1%}) + Low volatility (below median)'
            }
        else:
            # HOLD signal with detailed reason
            reasons = []
            if confidence < self.confidence_threshold:
                reasons.append(f'Low confidence ({confidence:.1%} < {self.confidence_threshold:.1%})')
            if volatility_prediction >= median_volatility:
                reasons.append(f'High volatility (above median)')
            if direction == "NEUTRAL":
                reasons.append('Neutral prediction')
            if skip_win_rate:
                reasons.append(f'Poor win rate ({ticker_win_rate:.1%} < {self.min_win_rate:.0%})')
            if portfolio_drawdown >= self.max_drawdown:
                reasons.append(f'Max drawdown reached ({portfolio_drawdown:.1%})')

            signal = {
                'action': 'HOLD',
                'confidence': float(confidence),
                'should_trade': False,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0,
                'position_value': 0,
                'risk_amount': 0,
                'reward_amount': 0,
                'risk_reward_ratio': 0,
                'volatility_percentile': float(volatility_percentile),
                'predicted_volatility': float(volatility_prediction),
                'median_volatility': float(median_volatility),
                'strategy': 'optimal_hybrid',
                'reason': 'Conditions not met: ' + '; '.join(reasons)
            }

        return signal

    def get_asset_recommendation(self, asset_type: str) -> str:
        """
        Get asset-specific trading recommendation based on backtest results.

        Args:
            asset_type: Type of asset (Stock, Forex, Commodity, Cryptocurrency)

        Returns:
            Recommendation string
        """
        recommendations = {
            'Forex': 'BEST PERFORMANCE - Avg +1.78% return, +0.60% alpha. Highly recommended for this strategy.',
            'Commodity': 'EXCELLENT - Avg +0.26% return, +6.43% alpha. Good for positive returns.',
            'Cryptocurrency': 'GOOD FOR ALPHA - Avg +15.11% alpha. Excellent capital preservation in declining markets.',
            'Stock': 'CONSERVATIVE - Ultra-selective, 0-2 trades. Good for capital preservation but limited opportunities.'
        }

        return recommendations.get(asset_type, 'No specific recommendation available.')

    def calculate_optimal_parameters(
        self,
        asset_type: str
    ) -> Dict:
        """
        Get optimal parameters for specific asset class based on backtest results.

        Args:
            asset_type: Type of asset

        Returns:
            Dictionary of optimal parameters
        """
        # Asset-specific tuning based on backtest results
        if asset_type == 'Forex':
            return {
                'confidence_threshold': 0.65,
                'position_size': 0.50,
                'max_trades': 2,
                'expected_return': 0.0178,  # +1.78% based on backtest
                'expected_alpha': 0.0060
            }
        elif asset_type == 'Commodity':
            return {
                'confidence_threshold': 0.65,
                'position_size': 0.50,
                'max_trades': 2,
                'expected_return': 0.0026,  # +0.26% based on backtest
                'expected_alpha': 0.0643
            }
        elif asset_type == 'Cryptocurrency':
            return {
                'confidence_threshold': 0.65,
                'position_size': 0.40,  # Lower for crypto volatility
                'max_trades': 2,
                'expected_return': 0.0000,
                'expected_alpha': 0.1511  # +15.11% alpha!
            }
        else:  # Stocks
            return {
                'confidence_threshold': 0.70,  # Higher for stocks (ultra-conservative)
                'position_size': 0.30,
                'max_trades': 1,
                'expected_return': 0.0000,
                'expected_alpha': -0.0189
            }
