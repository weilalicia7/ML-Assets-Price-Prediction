"""
Walk-Forward Validation Test

Tests the trading strategy on OUT-OF-SAMPLE data to ensure we're not overfitting.

Walk-Forward Method:
1. Split data into multiple periods (e.g., 6 periods of 2 months each)
2. Train/optimize on first N periods
3. Test on period N+1 (out-of-sample)
4. Roll forward and repeat

This prevents look-ahead bias and ensures strategy robustness.

Based on: phase1 fixing on C model 2.pdf - "Testing without overfitting"
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.hybrid_strategy import OptimalHybridStrategy
from src.trading.risk_management import IntegratedRiskManager


class WalkForwardValidator:
    """
    Walk-forward validation framework to test strategy without overfitting.
    """

    def __init__(
        self,
        train_periods: int = 3,
        test_periods: int = 1,
        period_days: int = 60
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_periods: Number of periods to train on
            test_periods: Number of periods to test on (out-of-sample)
            period_days: Days per period (60 = ~2 months)
        """
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.period_days = period_days

    def split_data_into_periods(
        self,
        df: pd.DataFrame,
        total_periods: int
    ) -> List[pd.DataFrame]:
        """Split dataframe into equal periods."""
        periods = []
        period_length = len(df) // total_periods

        for i in range(total_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < total_periods - 1 else len(df)
            periods.append(df.iloc[start_idx:end_idx].copy())

        return periods

    def run_walk_forward_test(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: OptimalHybridStrategy,
        total_periods: int = 6
    ) -> Dict:
        """
        Run walk-forward validation across multiple stocks.

        Args:
            data: Dict of {ticker: DataFrame} with OHLC data
            strategy: Trading strategy to test
            total_periods: Total number of periods to split data into

        Returns:
            Dict with in-sample and out-of-sample results
        """
        results = {
            'in_sample': [],
            'out_of_sample': [],
            'walk_forward_windows': []
        }

        # Calculate number of walk-forward windows
        num_windows = total_periods - self.train_periods - self.test_periods + 1

        print(f"\n{'='*70}")
        print(f"WALK-FORWARD VALIDATION")
        print(f"{'='*70}")
        print(f"Total periods: {total_periods}")
        print(f"Train periods: {self.train_periods}")
        print(f"Test periods: {self.test_periods}")
        print(f"Walk-forward windows: {num_windows}")
        print(f"{'='*70}\n")

        for window in range(num_windows):
            train_start = window
            train_end = window + self.train_periods
            test_start = train_end
            test_end = test_start + self.test_periods

            print(f"\n--- Window {window + 1}/{num_windows} ---")
            print(f"Train: Periods {train_start + 1}-{train_end}")
            print(f"Test:  Periods {test_start + 1}-{test_end} (OUT-OF-SAMPLE)")

            # Initialize fresh risk manager for each window
            risk_manager = IntegratedRiskManager()

            # Track results for this window
            in_sample_trades = []
            out_of_sample_trades = []

            for ticker, df in data.items():
                if len(df) < 60:
                    continue

                periods = self.split_data_into_periods(df, total_periods)

                # TRAINING PHASE (in-sample)
                for period_idx in range(train_start, train_end):
                    if period_idx >= len(periods):
                        continue
                    period_df = periods[period_idx]
                    trades = self._simulate_period(period_df, strategy, ticker, risk_manager)
                    in_sample_trades.extend(trades)

                    # Record trades to risk manager (learn from in-sample)
                    for trade in trades:
                        risk_manager.record_trade_result(ticker, trade['pnl'])

                # TESTING PHASE (out-of-sample) - NO learning during test
                for period_idx in range(test_start, test_end):
                    if period_idx >= len(periods):
                        continue
                    period_df = periods[period_idx]

                    # Check if trading is allowed based on training performance
                    can_trade, reason = risk_manager.should_trade(ticker)

                    if can_trade:
                        trades = self._simulate_period(
                            period_df, strategy, ticker, risk_manager,
                            apply_risk_rules=True
                        )
                        out_of_sample_trades.extend(trades)
                    else:
                        print(f"    {ticker}: Blocked for test period - {reason}")

            # Calculate window results
            window_result = self._calculate_window_results(
                in_sample_trades, out_of_sample_trades, window + 1
            )
            results['walk_forward_windows'].append(window_result)
            results['in_sample'].extend(in_sample_trades)
            results['out_of_sample'].extend(out_of_sample_trades)

        # Calculate aggregate results
        results['summary'] = self._calculate_summary(results)

        return results

    def _simulate_period(
        self,
        df: pd.DataFrame,
        strategy: OptimalHybridStrategy,
        ticker: str,
        risk_manager: IntegratedRiskManager,
        apply_risk_rules: bool = False
    ) -> List[Dict]:
        """Simulate trading for one period."""
        trades = []

        # Calculate technical indicators
        df = df.copy()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['momentum'] = df['Close'].pct_change(5)
        df['volatility'] = df['Close'].pct_change().rolling(20).std()
        df = df.dropna()

        if len(df) < 10:
            return trades

        for i in range(5, len(df) - 5):
            row = df.iloc[i]
            hist_vol = df['volatility'].iloc[max(0, i-30):i].values

            if len(hist_vol) < 5:
                continue

            # Generate direction prediction
            sma_signal = 1 if row['SMA_5'] > row['SMA_20'] else 0
            momentum_signal = 1 if row['momentum'] > 0 else 0
            direction_pred = sma_signal * 0.6 + momentum_signal * 0.4

            # Get signal from strategy
            signal = strategy.generate_hybrid_signal(
                direction_prediction=direction_pred,
                volatility_prediction=row['volatility'],
                historical_volatility=hist_vol,
                current_price=row['Close'],
                ticker=ticker
            )

            if signal['should_trade'] and signal['action'] != 'HOLD':
                # Apply risk rules if testing out-of-sample
                position_size = signal['position_value']

                if apply_risk_rules:
                    can_trade, _ = risk_manager.should_trade(ticker)
                    if not can_trade:
                        continue

                    # Adjust position size based on risk
                    size_pct = risk_manager.get_position_size(ticker, signal['confidence'])
                    position_size = 100000 * size_pct

                # Calculate actual return (5-day forward)
                future_price = df['Close'].iloc[min(i + 5, len(df) - 1)]
                actual_return = (future_price - row['Close']) / row['Close']

                if signal['action'] == 'SHORT':
                    actual_return = -actual_return

                pnl = position_size * actual_return

                trades.append({
                    'date': df.index[i],
                    'ticker': ticker,
                    'action': signal['action'],
                    'confidence': signal['confidence'],
                    'position_value': position_size,
                    'actual_return': actual_return,
                    'pnl': pnl,
                    'was_profitable': pnl > 0
                })

        return trades

    def _calculate_window_results(
        self,
        in_sample_trades: List[Dict],
        out_of_sample_trades: List[Dict],
        window_num: int
    ) -> Dict:
        """Calculate results for one walk-forward window."""
        result = {
            'window': window_num,
            'in_sample': {},
            'out_of_sample': {}
        }

        for name, trades in [('in_sample', in_sample_trades), ('out_of_sample', out_of_sample_trades)]:
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0
                avg_pnl = total_pnl / len(trades) if trades else 0

                result[name] = {
                    'num_trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl
                }
            else:
                result[name] = {
                    'num_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_pnl': 0
                }

        # Print window results
        print(f"\n  IN-SAMPLE:     {result['in_sample']['num_trades']:3d} trades, "
              f"Win Rate: {result['in_sample']['win_rate']:.1%}, "
              f"Total P&L: ${result['in_sample']['total_pnl']:,.0f}")
        print(f"  OUT-OF-SAMPLE: {result['out_of_sample']['num_trades']:3d} trades, "
              f"Win Rate: {result['out_of_sample']['win_rate']:.1%}, "
              f"Total P&L: ${result['out_of_sample']['total_pnl']:,.0f}")

        # Check for overfitting (large gap between in-sample and out-of-sample)
        if result['in_sample']['num_trades'] > 0 and result['out_of_sample']['num_trades'] > 0:
            win_rate_gap = result['in_sample']['win_rate'] - result['out_of_sample']['win_rate']
            if win_rate_gap > 0.15:
                print(f"  WARNING: Possible overfitting (win rate gap: {win_rate_gap:.1%})")

        return result

    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics across all windows."""
        in_sample_trades = results['in_sample']
        oos_trades = results['out_of_sample']

        summary = {
            'in_sample': {
                'total_trades': len(in_sample_trades),
                'total_pnl': sum(t['pnl'] for t in in_sample_trades) if in_sample_trades else 0,
                'win_rate': sum(1 for t in in_sample_trades if t['pnl'] > 0) / len(in_sample_trades) if in_sample_trades else 0
            },
            'out_of_sample': {
                'total_trades': len(oos_trades),
                'total_pnl': sum(t['pnl'] for t in oos_trades) if oos_trades else 0,
                'win_rate': sum(1 for t in oos_trades if t['pnl'] > 0) / len(oos_trades) if oos_trades else 0
            }
        }

        # Calculate robustness metrics
        if summary['in_sample']['total_trades'] > 0 and summary['out_of_sample']['total_trades'] > 0:
            summary['robustness'] = {
                'win_rate_degradation': summary['in_sample']['win_rate'] - summary['out_of_sample']['win_rate'],
                'pnl_ratio': summary['out_of_sample']['total_pnl'] / summary['in_sample']['total_pnl'] if summary['in_sample']['total_pnl'] != 0 else 0,
                'is_robust': abs(summary['in_sample']['win_rate'] - summary['out_of_sample']['win_rate']) < 0.10
            }

        return summary


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION TEST")
    print("Testing Strategy Without Overfitting")
    print("=" * 70)

    # Test stocks
    test_stocks = [
        '0700.HK',  # Tencent
        '9988.HK',  # Alibaba
        '1810.HK',  # Xiaomi
        '2269.HK',  # WuXi Bio
        '1398.HK',  # ICBC
    ]

    # Date range - 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"\nTest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Test Stocks: {len(test_stocks)}")

    # Download data
    print("\nDownloading data...")
    data = {}

    for ticker in test_stocks:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if df.empty:
                print(f"  {ticker}: No data")
                continue

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")

        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    if not data:
        print("No data available for testing")
        return

    # Initialize strategy with Phase 1 parameters
    strategy = OptimalHybridStrategy(
        confidence_threshold=0.50,
        volatility_filter_percentile=0.50,
        position_size=0.50,
        stop_loss_pct=0.05,
        drawdown_threshold=0.08,
        max_drawdown=0.20
    )

    # Run walk-forward validation
    validator = WalkForwardValidator(
        train_periods=3,
        test_periods=1,
        period_days=60
    )

    results = validator.run_walk_forward_test(
        data=data,
        strategy=strategy,
        total_periods=6
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    summary = results['summary']

    print(f"\n--- In-Sample Results (Training) ---")
    print(f"Total Trades: {summary['in_sample']['total_trades']}")
    print(f"Total P&L: ${summary['in_sample']['total_pnl']:,.2f}")
    print(f"Win Rate: {summary['in_sample']['win_rate']:.1%}")

    print(f"\n--- Out-of-Sample Results (Testing) ---")
    print(f"Total Trades: {summary['out_of_sample']['total_trades']}")
    print(f"Total P&L: ${summary['out_of_sample']['total_pnl']:,.2f}")
    print(f"Win Rate: {summary['out_of_sample']['win_rate']:.1%}")

    if 'robustness' in summary:
        print(f"\n--- Robustness Check ---")
        print(f"Win Rate Degradation: {summary['robustness']['win_rate_degradation']:.1%}")
        print(f"P&L Ratio (OOS/IS): {summary['robustness']['pnl_ratio']:.2f}")

        if summary['robustness']['is_robust']:
            print("PASS: Strategy appears robust (win rate degradation < 10%)")
        else:
            print("WARNING: Strategy may be overfit (win rate degradation >= 10%)")

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
