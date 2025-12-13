"""
Kelly Criterion Backtester with Prediction Market Logic

Based on prediction market theory:
- Only bet when E[profit] = p - x > 0 (you have an edge)
- Size bets according to Kelly Criterion
- Track information decay and market efficiency

Key insight from paper:
E[profit_buyer] = p·(1-x) + (1-p)·(-x) = p - x

Where:
- p = true probability (model's estimate)
- x = market-implied probability (from price)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: pd.Timestamp
    ticker: str
    direction: str  # 'long' or 'short'
    entry_price: float
    size: float
    model_prob: float
    market_prob: float
    edge: float
    kelly_fraction: float
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    regime: Optional[str] = None


class KellyBacktester:
    """
    Backtest trading strategies using Kelly Criterion position sizing.

    Following prediction market logic:
    1. Calculate model's implied probability (p)
    2. Calculate market's implied probability (x) from recent price action
    3. Only trade when edge exists: p > x (for long) or p < x (for short)
    4. Size position using Kelly Criterion
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        kelly_fraction: float = 0.25,  # Quarter-Kelly for safety
        max_position_size: float = 0.10,  # Max 10% of capital per trade
        min_edge: float = 0.05,  # Minimum 5% edge to trade
        transaction_cost: float = 0.001  # 0.1% transaction cost
    ):
        """
        Initialize Kelly Criterion backtester.

        Args:
            initial_capital: Starting capital
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
            max_position_size: Maximum position as fraction of capital
            min_edge: Minimum edge required to place trade
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        self.min_edge = min_edge
        self.transaction_cost = transaction_cost

        # State tracking
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []

        print(f"[KellyBacktester] Initialized")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Kelly Fraction: {kelly_fraction:.2f} (conservative)")
        print(f"  Min Edge Required: {min_edge:.1%}")

    def calculate_market_implied_probability(
        self,
        prices: np.ndarray,
        window: int = 20
    ) -> float:
        """
        Calculate market-implied probability from recent price action.

        Uses momentum and volatility to infer what "the market thinks".

        Args:
            prices: Recent price history
            window: Lookback window

        Returns:
            Market-implied probability of upward movement
        """
        if len(prices) < window:
            return 0.5  # Neutral if insufficient history

        recent_prices = prices[-window:]

        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Momentum component (what direction has market been moving?)
        momentum = np.mean(returns)

        # Volatility component (how certain is the market?)
        volatility = np.std(returns)

        # Convert to probability using sigmoid-like transformation
        # High positive momentum + low volatility = high probability
        # Low momentum or high volatility = probability near 0.5

        if volatility > 0:
            # Sharpe-like ratio
            signal = momentum / volatility

            # Convert to probability (sigmoid)
            prob = 1 / (1 + np.exp(-5 * signal))  # Scale by 5 for sensitivity
        else:
            prob = 0.5

        # Clamp to reasonable range
        prob = np.clip(prob, 0.1, 0.9)

        return prob

    def kelly_criterion(
        self,
        win_prob: float,
        odds: float,
        edge: float
    ) -> float:
        """
        Calculate optimal Kelly Criterion position size.

        Kelly formula: f* = (p·b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing = 1 - p
        - b = odds (profit/loss ratio)

        Args:
            win_prob: Probability of winning the bet
            odds: Odds ratio (potential_profit / potential_loss)
            edge: Estimated edge (model_prob - market_prob)

        Returns:
            Kelly fraction (0 to 1)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        p = win_prob
        q = 1 - p
        b = odds

        # Standard Kelly formula
        kelly_f = (p * b - q) / b

        # Apply fractional Kelly (quarter-Kelly is common in practice)
        kelly_f = kelly_f * self.kelly_fraction

        # Ensure non-negative and apply maximum position size
        kelly_f = max(0, min(kelly_f, self.max_position_size))

        return kelly_f

    def calculate_edge(
        self,
        model_prob: float,
        market_prob: float
    ) -> float:
        """
        Calculate edge (expected profit opportunity).

        From paper: E[profit] = p - x
        Where p = model probability, x = market probability

        Args:
            model_prob: Model's estimated probability
            market_prob: Market-implied probability

        Returns:
            Edge (positive means profitable expected value)
        """
        return model_prob - market_prob

    def should_trade(
        self,
        model_prob: float,
        market_prob: float,
        current_position: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine if we should trade based on edge.

        Args:
            model_prob: Model's probability estimate
            market_prob: Market's implied probability
            current_position: Current position ('long', 'short', or None)

        Returns:
            (should_trade, direction)
        """
        edge = self.calculate_edge(model_prob, market_prob)

        # Long opportunity: model thinks upward probability is underpriced
        if edge >= self.min_edge and (current_position != 'long'):
            return True, 'long'

        # Short opportunity: model thinks upward probability is overpriced
        elif edge <= -self.min_edge and (current_position != 'short'):
            return True, 'short'

        # No trade opportunity
        return False, None

    def calculate_position_size(
        self,
        model_prob: float,
        market_prob: float,
        direction: str,
        current_capital: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            model_prob: Model's probability
            market_prob: Market probability
            direction: 'long' or 'short'
            current_capital: Available capital

        Returns:
            Position size in dollars
        """
        # Calculate win probability based on direction
        if direction == 'long':
            win_prob = model_prob
        else:  # short
            win_prob = 1 - model_prob

        # Estimate odds (assume 1:1 for simplicity, can be refined)
        odds = 1.0

        # Calculate edge
        edge = abs(self.calculate_edge(model_prob, market_prob))

        # Get Kelly fraction
        kelly_f = self.kelly_criterion(win_prob, odds, edge)

        # Calculate position size
        position_size = current_capital * kelly_f

        return position_size

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        price: float,
        model_prob: float,
        market_prob: float,
        direction: str,
        regime: Optional[str] = None
    ) -> Trade:
        """
        Execute a trade with Kelly sizing.

        Args:
            timestamp: Trade timestamp
            ticker: Asset ticker
            price: Current price
            model_prob: Model's probability estimate
            market_prob: Market-implied probability
            direction: 'long' or 'short'
            regime: Current market regime

        Returns:
            Trade object
        """
        # Calculate position size
        position_size = self.calculate_position_size(
            model_prob, market_prob, direction, self.capital
        )

        # Apply transaction costs
        cost = position_size * self.transaction_cost
        position_size -= cost

        # Calculate edge and Kelly fraction for record-keeping
        edge = self.calculate_edge(model_prob, market_prob)
        win_prob = model_prob if direction == 'long' else (1 - model_prob)
        kelly_f = self.kelly_criterion(win_prob, 1.0, abs(edge))

        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            direction=direction,
            entry_price=price,
            size=position_size,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            kelly_fraction=kelly_f,
            regime=regime
        )

        self.trades.append(trade)

        return trade

    def close_trade(
        self,
        trade: Trade,
        exit_price: float,
        timestamp: pd.Timestamp
    ):
        """
        Close an open trade and update capital.

        Args:
            trade: Trade to close
            exit_price: Exit price
            timestamp: Exit timestamp
        """
        # Calculate profit/loss
        if trade.direction == 'long':
            price_change = (exit_price - trade.entry_price) / trade.entry_price
        else:  # short
            price_change = (trade.entry_price - exit_price) / trade.entry_price

        # Profit = position_size * price_change
        profit = trade.size * price_change

        # Apply transaction costs
        exit_cost = trade.size * self.transaction_cost
        profit -= exit_cost

        # Update trade record
        trade.exit_price = exit_price
        trade.profit = profit

        # Update capital
        self.capital += profit

        # Record equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'trade_profit': profit,
            'ticker': trade.ticker,
            'direction': trade.direction,
            'edge': trade.edge
        })

    def backtest(
        self,
        data: pd.DataFrame,
        model_probabilities: np.ndarray,
        price_col: str = 'Close',
        regime_col: Optional[str] = None,
        hold_periods: int = 5
    ) -> Dict:
        """
        Run backtest using Kelly Criterion position sizing.

        Args:
            data: DataFrame with price data
            model_probabilities: Model's probability predictions
            price_col: Column name for prices
            regime_col: Column name for regime (optional)
            hold_periods: Number of periods to hold each position

        Returns:
            Backtest results dictionary
        """
        print(f"\n[Backtest] Running Kelly Criterion backtest...")
        print(f"  Data points: {len(data)}")
        print(f"  Hold period: {hold_periods} bars")

        open_trades = []
        prices = data[price_col].values

        for i in range(len(data) - hold_periods):
            timestamp = data.index[i]
            price = prices[i]
            model_prob = model_probabilities[i]

            # Calculate market-implied probability from recent price action
            market_prob = self.calculate_market_implied_probability(
                prices[:i+1], window=20
            )

            # Get regime if available
            regime = data[regime_col].iloc[i] if regime_col else None

            # Check if we should trade
            should_trade, direction = self.should_trade(model_prob, market_prob)

            if should_trade:
                # Execute trade
                trade = self.execute_trade(
                    timestamp, data.get('Ticker', ['UNKNOWN'])[0] if hasattr(data, 'get') else 'UNKNOWN',
                    price, model_prob, market_prob, direction, regime
                )
                open_trades.append((trade, i + hold_periods))  # Hold for N periods

            # Close trades that have reached their hold period
            trades_to_close = [(t, idx) for t, idx in open_trades if i >= idx]
            for trade, _ in trades_to_close:
                exit_price = prices[i]
                self.close_trade(trade, exit_price, data.index[i])
                open_trades.remove((trade, _))

        # Close any remaining open trades
        for trade, _ in open_trades:
            exit_price = prices[-1]
            self.close_trade(trade, exit_price, data.index[-1])

        # Calculate performance metrics
        results = self.calculate_performance_metrics()

        return results

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary with performance statistics
        """
        if not self.trades:
            return {'error': 'No trades executed'}

        df_trades = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'ticker': t.ticker,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'profit': t.profit,
                'edge': t.edge,
                'kelly_fraction': t.kelly_fraction,
                'model_prob': t.model_prob,
                'market_prob': t.market_prob,
                'regime': t.regime
            }
            for t in self.trades
        ])

        # Filter completed trades
        completed = df_trades[df_trades['profit'].notna()]

        if len(completed) == 0:
            return {'error': 'No completed trades'}

        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        n_trades = len(completed)
        n_winners = len(completed[completed['profit'] > 0])
        n_losers = len(completed[completed['profit'] <= 0])
        win_rate = n_winners / n_trades if n_trades > 0 else 0

        avg_win = completed[completed['profit'] > 0]['profit'].mean() if n_winners > 0 else 0
        avg_loss = abs(completed[completed['profit'] <= 0]['profit'].mean()) if n_losers > 0 else 0
        profit_factor = (avg_win * n_winners) / (avg_loss * n_losers) if (avg_loss * n_losers) > 0 else np.inf

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df['capital'].pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity_values = [e['capital'] for e in self.equity_curve]
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Edge realization
        avg_edge = completed['edge'].mean()
        edge_realization = win_rate / (0.5 + avg_edge) if avg_edge > -0.5 else 0

        results = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': self.capital,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_edge': avg_edge,
            'edge_realization': edge_realization,
            'avg_kelly_fraction': completed['kelly_fraction'].mean(),
            'avg_trade_size': completed['size'].mean(),
            'total_profit': completed['profit'].sum(),
            'trades_df': completed
        }

        return results

    def print_results(self, results: Dict):
        """Print backtest results in formatted way."""
        if 'error' in results:
            print(f"\n[Results] {results['error']}")
            return

        print(f"\n{'='*60}")
        print(f"KELLY CRITERION BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\nCapital:")
        print(f"  Initial:       ${self.initial_capital:>12,.2f}")
        print(f"  Final:         ${results['final_capital']:>12,.2f}")
        print(f"  Total Return:  {results['total_return_pct']:>12.2f}%")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:  {results['n_trades']:>12}")
        print(f"  Win Rate:      {results['win_rate']:>12.1%}")
        print(f"  Profit Factor: {results['profit_factor']:>12.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:  {results['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:  {results['max_drawdown']:>12.1%}")
        print(f"\nPrediction Market Metrics:")
        print(f"  Avg Edge:      {results['avg_edge']:>12.1%}")
        print(f"  Edge Real.:    {results['edge_realization']:>12.2f}")
        print(f"  Avg Kelly:     {results['avg_kelly_fraction']:>12.1%}")
        print(f"{'='*60}")


def run_kelly_backtest_example():
    """Example usage of Kelly Criterion backtester."""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Simulate price data with trend
    returns = np.random.randn(100) * 0.02 + 0.001  # Small positive drift
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Close': prices,
        'Ticker': 'BTC-USD'
    }, index=dates)

    # Simulate model probabilities (with some predictive power)
    future_returns = np.diff(prices) / prices[:-1]
    model_probs = 0.5 + 0.3 * np.sign(future_returns)  # 80% directional accuracy
    model_probs = np.append(model_probs, 0.5)  # Last value

    # Run backtest
    backtester = KellyBacktester(initial_capital=10000)
    results = backtester.backtest(data, model_probs, hold_periods=5)
    backtester.print_results(results)

    return backtester, results


if __name__ == "__main__":
    run_kelly_backtest_example()
