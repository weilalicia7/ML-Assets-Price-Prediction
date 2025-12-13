"""
Enhanced Testing - Drawdown Analysis
Identifies which stocks/signals cause the most drawdown.

Based on: phase1 fixing on C model_extra.pdf
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.hybrid_strategy import OptimalHybridStrategy


def calculate_max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative P&L series."""
    if len(cumulative_pnl) == 0:
        return 0.0

    # Calculate running maximum
    running_max = cumulative_pnl.cummax()

    # Calculate drawdown at each point
    drawdown = cumulative_pnl - running_max

    # Return the maximum (most negative) drawdown
    return drawdown.min()


def generate_signals_for_stock(df: pd.DataFrame, strategy: OptimalHybridStrategy, ticker: str) -> pd.DataFrame:
    """Generate trading signals for a stock using the hybrid strategy."""
    signals = []

    # Calculate technical indicators
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['momentum'] = df['Close'].pct_change(5)
    df['volatility'] = df['Close'].pct_change().rolling(20).std()

    # Need at least 20 days for indicators
    df = df.dropna()

    if len(df) < 30:
        return pd.DataFrame()

    for i in range(20, len(df)):
        row = df.iloc[i]
        hist_vol = df['volatility'].iloc[max(0, i-60):i].values

        if len(hist_vol) < 10:
            continue

        # Generate direction prediction based on SMA crossover and momentum
        sma_signal = 1 if row['SMA_5'] > row['SMA_20'] else 0
        momentum_signal = 1 if row['momentum'] > 0 else 0
        direction_pred = (sma_signal * 0.6 + momentum_signal * 0.4)

        # Add some noise to simulate real predictions
        direction_pred = np.clip(direction_pred + np.random.normal(0, 0.1), 0, 1)

        signal = strategy.generate_hybrid_signal(
            direction_prediction=direction_pred,
            volatility_prediction=row['volatility'],
            historical_volatility=hist_vol,
            current_price=row['Close'],
            account_size=100000,
            portfolio_drawdown=0.0,  # Will track separately
            ticker=ticker
        )

        signal['date'] = df.index[i]
        signal['ticker'] = ticker
        signal['actual_return'] = df['Close'].iloc[min(i+5, len(df)-1)] / row['Close'] - 1
        signal['direction_pred'] = direction_pred
        signals.append(signal)

    return pd.DataFrame(signals)


def simulate_trades(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate trades from signals and calculate P&L."""
    if signals_df.empty:
        return pd.DataFrame()

    trades = []

    for _, signal in signals_df.iterrows():
        if signal['should_trade'] and signal['action'] != 'HOLD':
            # Determine if trade was profitable
            actual_return = signal['actual_return']

            if signal['action'] == 'LONG':
                trade_return = actual_return
            else:  # SHORT
                trade_return = -actual_return

            # Calculate P&L based on position size
            position_value = signal['position_value']
            pnl = position_value * trade_return

            trades.append({
                'date': signal['date'],
                'ticker': signal['ticker'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'position_value': position_value,
                'actual_return': actual_return,
                'trade_return': trade_return,
                'pnl': pnl,
                'was_profitable': pnl > 0
            })

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('date')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    return trades_df


def analyze_drawdown_sources(trades: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Identify which stocks/signals cause the most drawdown."""
    if trades.empty or predictions.empty:
        return {}

    drawdown_analysis = {}

    for stock in set(trades['ticker']):
        stock_trades = trades[trades['ticker'] == stock]
        stock_predictions = predictions[predictions['ticker'] == stock]

        if stock_trades.empty:
            continue

        # Calculate drawdown contribution
        total_pnl = stock_trades['pnl'].sum()
        max_drawdown = calculate_max_drawdown(stock_trades['cumulative_pnl'])

        drawdown_analysis[stock] = {
            'total_return': total_pnl,
            'max_drawdown': max_drawdown,
            'drawdown_ratio': abs(max_drawdown) / (abs(total_pnl) + 1e-10),
            'avg_confidence': stock_predictions['confidence'].mean() if 'confidence' in stock_predictions else 0,
            'win_rate': (stock_trades['pnl'] > 0).mean(),
            'num_trades': len(stock_trades),
            'avg_pnl_per_trade': total_pnl / len(stock_trades) if len(stock_trades) > 0 else 0
        }

    # Sort by worst drawdown ratio
    worst_offenders = sorted(
        drawdown_analysis.items(),
        key=lambda x: x[1]['drawdown_ratio'],
        reverse=True
    )

    return dict(worst_offenders)


def main():
    print("=" * 70)
    print("ENHANCED TESTING - DRAWDOWN ANALYSIS")
    print("Based on: phase1 fixing on C model_extra.pdf")
    print("=" * 70)

    # Test stocks - mix of China and HK stocks
    test_stocks = [
        '0700.HK',  # Tencent
        '9988.HK',  # Alibaba
        '2318.HK',  # Ping An
        '1810.HK',  # Xiaomi
        '3690.HK',  # Meituan
        '0939.HK',  # CCB
        '1398.HK',  # ICBC
        '2269.HK',  # WuXi Bio
    ]

    # Date range - last 1 year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"\nTest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Test Stocks: {len(test_stocks)}")

    # Initialize strategy with Phase 1 fixes
    strategy = OptimalHybridStrategy(
        confidence_threshold=0.50,  # Phase 1 fix
        volatility_filter_percentile=0.50,
        position_size=0.50,
        stop_loss_pct=0.05,
        drawdown_threshold=0.08,  # Phase 1 fix
        max_drawdown=0.20  # Phase 1 fix
    )

    print(f"\nStrategy Parameters:")
    print(f"  Confidence threshold: {strategy.confidence_threshold:.0%}")
    print(f"  Drawdown threshold: {strategy.drawdown_threshold:.0%}")
    print(f"  Max drawdown: {strategy.max_drawdown:.0%}")
    print(f"  Min win rate: {strategy.min_win_rate:.0%}")

    # Collect all signals and trades
    all_signals = []
    all_trades = []

    print("\n" + "-" * 70)
    print("Downloading data and generating signals...")
    print("-" * 70)

    for ticker in test_stocks:
        try:
            print(f"\n  Processing {ticker}...", end=" ")

            # Download data
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if df.empty or len(df) < 50:
                print("SKIP (insufficient data)")
                continue

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Generate signals
            signals_df = generate_signals_for_stock(df, strategy, ticker)

            if signals_df.empty:
                print("SKIP (no signals)")
                continue

            # Simulate trades
            trades_df = simulate_trades(signals_df)

            if trades_df.empty:
                print(f"OK ({len(signals_df)} signals, 0 trades)")
                all_signals.append(signals_df)
                continue

            all_signals.append(signals_df)
            all_trades.append(trades_df)

            print(f"OK ({len(signals_df)} signals, {len(trades_df)} trades)")

        except Exception as e:
            print(f"ERROR: {e}")

    if not all_trades:
        print("\nNo trades generated. Cannot perform drawdown analysis.")
        return

    # Combine all data
    all_signals_df = pd.concat(all_signals, ignore_index=True)
    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # Sort by date and recalculate cumulative P&L
    all_trades_df = all_trades_df.sort_values('date')
    all_trades_df['cumulative_pnl'] = all_trades_df['pnl'].cumsum()

    print("\n" + "=" * 70)
    print("DRAWDOWN ANALYSIS RESULTS")
    print("=" * 70)

    # Overall portfolio stats
    total_pnl = all_trades_df['pnl'].sum()
    max_dd = calculate_max_drawdown(all_trades_df['cumulative_pnl'])
    overall_win_rate = (all_trades_df['pnl'] > 0).mean()

    print(f"\n--- Portfolio Summary ---")
    print(f"Total Trades: {len(all_trades_df)}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    print(f"Overall Win Rate: {overall_win_rate:.1%}")
    print(f"Average P&L per Trade: ${total_pnl/len(all_trades_df):,.2f}")

    # Analyze drawdown sources
    drawdown_analysis = analyze_drawdown_sources(all_trades_df, all_signals_df)

    print(f"\n--- Drawdown Analysis by Stock ---")
    print(f"{'Stock':<12} {'Total P&L':>12} {'Max DD':>12} {'DD Ratio':>10} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 70)

    for stock, metrics in drawdown_analysis.items():
        print(f"{stock:<12} ${metrics['total_return']:>10,.2f} ${metrics['max_drawdown']:>10,.2f} "
              f"{metrics['drawdown_ratio']:>9.2f} {metrics['win_rate']:>9.1%} {metrics['num_trades']:>8}")

    # Identify worst offenders
    print(f"\n--- Worst Drawdown Contributors ---")
    worst_stocks = [s for s, m in drawdown_analysis.items() if m['drawdown_ratio'] > 2.0]

    if worst_stocks:
        print(f"Stocks with Drawdown Ratio > 2.0: {', '.join(worst_stocks)}")
        print("Recommendation: Consider excluding these stocks or reducing position sizes")
    else:
        print("No stocks with excessive drawdown ratio (> 2.0)")

    # Identify best performers
    print(f"\n--- Best Performers ---")
    best_stocks = [s for s, m in drawdown_analysis.items()
                   if m['win_rate'] > 0.55 and m['total_return'] > 0]

    if best_stocks:
        print(f"Stocks with Win Rate > 55% and Positive Returns: {', '.join(best_stocks)}")
    else:
        print("No stocks meeting best performer criteria")

    # Confidence analysis
    print(f"\n--- Confidence Level Analysis ---")

    for conf_level in [0.5, 0.6, 0.7, 0.8]:
        high_conf_trades = all_trades_df[all_trades_df['confidence'] >= conf_level]
        if len(high_conf_trades) > 0:
            win_rate = (high_conf_trades['pnl'] > 0).mean()
            avg_pnl = high_conf_trades['pnl'].mean()
            print(f"Confidence >= {conf_level:.0%}: {len(high_conf_trades)} trades, "
                  f"Win Rate: {win_rate:.1%}, Avg P&L: ${avg_pnl:,.2f}")

    # Save results to CSV
    output_file = os.path.join(os.path.dirname(__file__), 'drawdown_analysis_results.csv')
    results_df = pd.DataFrame.from_dict(drawdown_analysis, orient='index')
    results_df.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return drawdown_analysis


if __name__ == "__main__":
    main()
