"""
Risk Management & Position Sizing for Trading
Implements professional risk management rules for volatility-based trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class RiskManager:
    """
    Professional risk management for volatility trading.

    Features:
    - Position sizing based on volatility
    - Stop loss calculation
    - Portfolio heat limits
    - Kelly Criterion
    - Risk-adjusted returns
    """

    def __init__(
        self,
        account_size: float = 100000,
        max_position_risk: float = 0.02,  # 2% max risk per trade
        max_portfolio_risk: float = 0.06,  # 6% max total portfolio risk
        max_leverage: float = 1.0,  # No leverage by default
        max_correlation_exposure: float = 0.3  # Max 30% in correlated assets
    ):
        """
        Initialize risk manager.

        Args:
            account_size: Total trading capital
            max_position_risk: Max % of capital to risk per trade (default: 2%)
            max_portfolio_risk: Max total portfolio risk (default: 6%)
            max_leverage: Maximum leverage allowed (default: 1.0 = no leverage)
            max_correlation_exposure: Max exposure to correlated assets
        """
        self.account_size = account_size
        self.max_position_risk = max_position_risk
        self.max_portfolio_risk = max_portfolio_risk
        self.max_leverage = max_leverage
        self.max_correlation_exposure = max_correlation_exposure

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        predicted_volatility: float,
        confidence: float = 1.0
    ) -> Dict:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            predicted_volatility: Predicted volatility (used for dynamic sizing)
            confidence: Confidence in prediction (0-1)

        Returns:
            Dictionary with position sizing details
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        # Maximum capital to risk
        max_risk_capital = self.account_size * self.max_position_risk

        # Adjust for confidence (lower confidence = smaller position)
        adjusted_risk_capital = max_risk_capital * confidence

        # Calculate shares (basic)
        shares_basic = adjusted_risk_capital / risk_per_share

        # Adjust for volatility (higher vol = smaller position)
        volatility_adjustment = 1.0 / (1.0 + predicted_volatility * 10)
        shares_adjusted = shares_basic * volatility_adjustment

        # Calculate position value
        position_value = shares_adjusted * entry_price

        # Apply leverage limit
        max_position_value = self.account_size * self.max_leverage
        if position_value > max_position_value:
            shares_adjusted = max_position_value / entry_price
            position_value = max_position_value

        # Calculate percentage of account
        position_pct = position_value / self.account_size

        return {
            'shares': int(shares_adjusted),
            'position_value': position_value,
            'position_pct': position_pct,
            'risk_per_share': risk_per_share,
            'total_risk': shares_adjusted * risk_per_share,
            'risk_pct': (shares_adjusted * risk_per_share) / self.account_size,
            'volatility_adjustment': volatility_adjustment,
            'entry_price': entry_price,
            'stop_loss': stop_loss
        }

    def calculate_stop_loss(
        self,
        entry_price: float,
        predicted_volatility: float,
        direction: str = 'long',
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss based on predicted volatility.

        Args:
            entry_price: Entry price
            predicted_volatility: Predicted daily volatility
            direction: 'long' or 'short'
            atr_multiplier: Multiplier for stop distance (default: 2x volatility)

        Returns:
            Stop loss price
        """
        # Stop distance based on predicted volatility
        stop_distance = entry_price * predicted_volatility * atr_multiplier

        if direction.lower() == 'long':
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_loss = entry_price + stop_distance

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0,
        direction: str = 'long'
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward_ratio: Desired risk/reward (default: 2.0)
            direction: 'long' or 'short'

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if direction.lower() == 'long':
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward

        return take_profit

    def check_portfolio_risk(
        self,
        current_positions: Dict[str, float],
        new_position_risk: float
    ) -> Tuple[bool, str]:
        """
        Check if adding new position exceeds portfolio risk limit.

        Args:
            current_positions: Dict of {ticker: risk_amount}
            new_position_risk: Risk amount for new position

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        total_current_risk = sum(current_positions.values())
        total_risk_pct = (total_current_risk + new_position_risk) / self.account_size

        if total_risk_pct > self.max_portfolio_risk:
            return False, f"Exceeds max portfolio risk: {total_risk_pct:.2%} > {self.max_portfolio_risk:.2%}"

        return True, "Within risk limits"

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (%)
            avg_loss: Average losing trade (%)

        Returns:
            Optimal position size as % of capital
        """
        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Use fractional Kelly (0.25 - 0.5 of full Kelly for safety)
        fractional_kelly = kelly_pct * 0.25

        # Cap at max position risk
        return min(fractional_kelly, self.max_position_risk)

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio for returns.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize returns
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)

        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe

    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Dict:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            equity_curve: Array of portfolio values over time

        Returns:
            Dict with drawdown metrics
        """
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax

        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # Find recovery
        recovery_idx = None
        if max_dd_idx < len(equity_curve) - 1:
            for i in range(max_dd_idx + 1, len(equity_curve)):
                if equity_curve[i] >= cummax[max_dd_idx]:
                    recovery_idx = i
                    break

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_dd_index': max_dd_idx,
            'recovery_index': recovery_idx,
            'recovery_days': recovery_idx - max_dd_idx if recovery_idx else None,
            'current_drawdown': drawdown[-1],
            'current_drawdown_pct': drawdown[-1] * 100
        }


class TradingSignalGenerator:
    """
    Generate trading signals based on volatility predictions.
    """

    def __init__(
        self,
        vol_percentile_long_threshold: float = 0.3,  # Enter long if vol < 30th percentile
        vol_percentile_short_threshold: float = 0.7,  # Enter short if vol > 70th percentile
        direction_confidence_threshold: float = 0.65,  # Need 65% confidence
        regime_filter: bool = True  # Only trade in favorable regimes
    ):
        """
        Initialize signal generator.

        Args:
            vol_percentile_long_threshold: Volatility percentile for long entries
            vol_percentile_short_threshold: Volatility percentile for short entries
            direction_confidence_threshold: Min confidence for directional trades
            regime_filter: Whether to filter trades by regime
        """
        self.vol_long_thresh = vol_percentile_long_threshold
        self.vol_short_thresh = vol_percentile_short_threshold
        self.direction_confidence = direction_confidence_threshold
        self.regime_filter = regime_filter

    def generate_signal(
        self,
        current_price: float,
        predicted_volatility: float,
        volatility_percentile: float,
        predicted_direction: int,  # 1 = up, -1 = down, 0 = neutral
        direction_confidence: float,
        current_regime: str,  # 'low', 'medium', 'high'
        historical_volatility: float
    ) -> Dict:
        """
        Generate trading signal based on predictions.

        Args:
            current_price: Current asset price
            predicted_volatility: Predicted next-day volatility
            volatility_percentile: Where predicted vol sits (0-1)
            predicted_direction: Predicted price direction
            direction_confidence: Confidence in direction prediction (0-1)
            current_regime: Current volatility regime
            historical_volatility: Recent historical volatility

        Returns:
            Trading signal with entry, stop, target
        """
        signal = {
            'action': 'HOLD',  # LONG, SHORT, HOLD
            'reason': '',
            'confidence': 0.0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'position_size_multiplier': 1.0
        }

        # Regime filter
        if self.regime_filter:
            if current_regime == 'high':
                signal['reason'] = 'High volatility regime - avoid new entries'
                return signal

        # Strategy 1: Mean Reversion (when vol is predicted to decrease)
        if predicted_volatility < historical_volatility * 0.8:
            if volatility_percentile < self.vol_long_thresh:
                # Volatility likely to mean-revert higher
                if predicted_direction == 1 and direction_confidence >= self.direction_confidence:
                    signal['action'] = 'LONG'
                    signal['reason'] = 'Low vol + upward direction = Long'
                    signal['confidence'] = direction_confidence
                    signal['position_size_multiplier'] = 1.2  # Larger position in low vol

        # Strategy 2: Volatility Breakout (when vol is predicted to increase)
        if predicted_volatility > historical_volatility * 1.2:
            if volatility_percentile > self.vol_short_thresh:
                # High volatility expected
                if abs(predicted_direction) == 1 and direction_confidence >= self.direction_confidence:
                    if predicted_direction == 1:
                        signal['action'] = 'LONG'
                        signal['reason'] = 'Vol breakout + upward momentum'
                    else:
                        signal['action'] = 'SHORT'
                        signal['reason'] = 'Vol breakout + downward momentum'

                    signal['confidence'] = direction_confidence
                    signal['position_size_multiplier'] = 0.7  # Smaller position in high vol

        # Calculate stop loss and take profit
        if signal['action'] != 'HOLD':
            risk_manager = RiskManager()

            stop_loss = risk_manager.calculate_stop_loss(
                current_price,
                predicted_volatility,
                direction='long' if signal['action'] == 'LONG' else 'short'
            )

            take_profit = risk_manager.calculate_take_profit(
                current_price,
                stop_loss,
                risk_reward_ratio=2.0,
                direction='long' if signal['action'] == 'LONG' else 'short'
            )

            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit

        return signal


def main():
    """Example usage of risk management and signal generation."""
    print("="*60)
    print("RISK MANAGEMENT & TRADING SIGNALS - EXAMPLE")
    print("="*60)

    # Initialize risk manager
    risk_mgr = RiskManager(
        account_size=100000,
        max_position_risk=0.02,
        max_portfolio_risk=0.06
    )

    print("\n" + "="*60)
    print("POSITION SIZING EXAMPLE")
    print("="*60)

    # Example: Calculate position size for AAPL
    entry_price = 180.00
    predicted_volatility = 0.025  # 2.5% predicted volatility

    # Calculate stop loss
    stop_loss = risk_mgr.calculate_stop_loss(
        entry_price,
        predicted_volatility,
        direction='long',
        atr_multiplier=2.0
    )

    print(f"\nEntry Price:        ${entry_price:.2f}")
    print(f"Predicted Vol:      {predicted_volatility:.2%}")
    print(f"Stop Loss:          ${stop_loss:.2f}")
    print(f"Risk per share:     ${entry_price - stop_loss:.2f}")

    # Calculate position size
    position = risk_mgr.calculate_position_size(
        entry_price,
        stop_loss,
        predicted_volatility,
        confidence=0.8
    )

    print(f"\nPosition Sizing:")
    print(f"  Shares:           {position['shares']}")
    print(f"  Position Value:   ${position['position_value']:,.2f}")
    print(f"  Position %:       {position['position_pct']:.2%}")
    print(f"  Total Risk:       ${position['total_risk']:,.2f}")
    print(f"  Risk %:           {position['risk_pct']:.2%}")

    # Calculate take profit
    take_profit = risk_mgr.calculate_take_profit(
        entry_price,
        stop_loss,
        risk_reward_ratio=2.0
    )

    print(f"\nTake Profit:        ${take_profit:.2f}")
    print(f"Risk/Reward:        1:{(take_profit - entry_price) / (entry_price - stop_loss):.1f}")

    # Trading signal example
    print("\n" + "="*60)
    print("TRADING SIGNAL EXAMPLE")
    print("="*60)

    signal_gen = TradingSignalGenerator()

    signal = signal_gen.generate_signal(
        current_price=180.00,
        predicted_volatility=0.020,
        volatility_percentile=0.25,
        predicted_direction=1,
        direction_confidence=0.75,
        current_regime='low',
        historical_volatility=0.025
    )

    print(f"\nSignal:             {signal['action']}")
    print(f"Reason:             {signal['reason']}")
    print(f"Confidence:         {signal['confidence']:.2%}")
    if signal['stop_loss']:
        print(f"Entry:              ${signal['entry_price']:.2f}")
        print(f"Stop Loss:          ${signal['stop_loss']:.2f}")
        print(f"Take Profit:        ${signal['take_profit']:.2f}")
        print(f"Position Multiplier: {signal['position_size_multiplier']:.1f}x")

    # Risk metrics example
    print("\n" + "="*60)
    print("RISK METRICS EXAMPLE")
    print("="*60)

    # Sample returns
    returns = np.random.randn(252) * 0.01 + 0.0005  # ~12.5% annual return, 16% vol

    sharpe = risk_mgr.calculate_sharpe_ratio(returns)
    print(f"\nSharpe Ratio:       {sharpe:.2f}")

    # Sample equity curve
    equity = 100000 * (1 + returns).cumprod()
    dd_metrics = risk_mgr.calculate_max_drawdown(equity)

    print(f"\nMax Drawdown:       {dd_metrics['max_drawdown_pct']:.2f}%")
    if dd_metrics['recovery_days']:
        print(f"Recovery Days:      {dd_metrics['recovery_days']}")
    print(f"Current Drawdown:   {dd_metrics['current_drawdown_pct']:.2f}%")

    print("\n[SUCCESS] Risk management examples complete!")


if __name__ == "__main__":
    main()
