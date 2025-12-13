"""
Portfolio Optimization for 18 Profitable Assets

Current Status: 18/22 = 81.8% profitability
Goal: Optimize portfolio allocation for maximum risk-adjusted returns

Approach:
1. Fetch historical returns for all 18 profitable assets
2. Calculate correlation matrix
3. Optimize weights using:
   - Equal weight baseline
   - Sharpe-weighted allocation
   - Minimum variance portfolio
   - Maximum Sharpe portfolio (Markowitz)
4. Compare portfolios and recommend production allocation

Assets (18):
- EUR/USD (1d), Alibaba (1d), Crude Oil (1d), Sony (4h), GBP/USD (1d)
- Nestle (1h), Iron Ore (1h), JPM (1d), Exxon (1h), J&J (1h)
- USD/JPY (4h), SAP (1h), Bitcoin (1h), Toyota (1h), Apple (1h)
- Ethereum (1h), ASML (1h), Microsoft (1h)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy.optimize import minimize


# Define 18 profitable assets with their optimal intervals
PROFITABLE_ASSETS = [
    {'name': 'EUR/USD', 'symbol': 'EURUSD=X', 'interval': '1d', 'period': '6mo', 'sharpe': 89.02, 'return': 0.0028, 'regime': 'Forex'},
    {'name': 'Alibaba', 'symbol': 'BABA', 'interval': '1d', 'period': '6mo', 'sharpe': 84.37, 'return': 0.0211, 'regime': 'Asia Stock'},
    {'name': 'Crude Oil', 'symbol': 'CL=F', 'interval': '1d', 'period': '6mo', 'sharpe': 75.54, 'return': 0.0151, 'regime': 'Commodity'},
    {'name': 'Sony', 'symbol': 'SONY', 'interval': '4h', 'period': '3mo', 'sharpe': 35.53, 'return': 0.0058, 'regime': 'Asia Stock'},
    {'name': 'GBP/USD', 'symbol': 'GBPUSD=X', 'interval': '1d', 'period': '6mo', 'sharpe': 27.62, 'return': 0.0015, 'regime': 'Forex'},
    {'name': 'Nestle', 'symbol': 'NSRGY', 'interval': '1h', 'period': '3mo', 'sharpe': 20.78, 'return': 0.0073, 'regime': 'Europe Stock'},
    {'name': 'Iron Ore', 'symbol': 'VALE', 'interval': '1h', 'period': '3mo', 'sharpe': 20.63, 'return': 0.0021, 'regime': 'Commodity'},
    {'name': 'JPM', 'symbol': 'JPM', 'interval': '1d', 'period': '6mo', 'sharpe': 18.62, 'return': 0.0017, 'regime': 'US Stock'},
    {'name': 'Exxon', 'symbol': 'XOM', 'interval': '1h', 'period': '3mo', 'sharpe': 15.51, 'return': 0.0033, 'regime': 'US Stock'},
    {'name': 'J&J', 'symbol': 'JNJ', 'interval': '1h', 'period': '3mo', 'sharpe': 10.76, 'return': 0.0035, 'regime': 'US Stock'},
    {'name': 'USD/JPY', 'symbol': 'JPY=X', 'interval': '4h', 'period': '3mo', 'sharpe': 2.97, 'return': 0.0046, 'regime': 'Forex'},
    {'name': 'SAP', 'symbol': 'SAP', 'interval': '1h', 'period': '3mo', 'sharpe': 2.80, 'return': 0.0061, 'regime': 'Europe Stock'},
    {'name': 'Bitcoin', 'symbol': 'BTC-USD', 'interval': '1h', 'period': '3mo', 'sharpe': 2.11, 'return': 0.0007, 'regime': 'Crypto'},
    {'name': 'Toyota', 'symbol': 'TM', 'interval': '1h', 'period': '3mo', 'sharpe': 1.94, 'return': 0.0030, 'regime': 'Asia Stock'},
    {'name': 'Apple', 'symbol': 'AAPL', 'interval': '1h', 'period': '3mo', 'sharpe': 1.88, 'return': 0.0025, 'regime': 'US Stock'},
    {'name': 'Ethereum', 'symbol': 'ETH-USD', 'interval': '1h', 'period': '3mo', 'sharpe': 1.12, 'return': 0.0185, 'regime': 'Crypto'},
    {'name': 'ASML', 'symbol': 'ASML', 'interval': '1h', 'period': '3mo', 'sharpe': 0.84, 'return': 0.0011, 'regime': 'Europe Stock'},
    {'name': 'Microsoft', 'symbol': 'MSFT', 'interval': '1h', 'period': '3mo', 'sharpe': 0.36, 'return': 0.0015, 'regime': 'US Stock'},
]


def fetch_returns(symbol: str, interval: str, period: str) -> pd.Series:
    """Fetch historical returns for an asset."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if len(data) < 20:
            return None

        returns = data['Close'].pct_change().dropna()
        return returns

    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def normalize_returns_to_daily(returns: pd.Series, interval: str) -> pd.Series:
    """Normalize returns to daily frequency for comparison."""
    if interval == '1h':
        # ~24 trading hours per day (crypto 24/7, stocks ~6.5h)
        return returns * np.sqrt(24)
    elif interval == '4h':
        # ~6 periods per day
        return returns * np.sqrt(6)
    elif interval == '1d':
        # Already daily
        return returns
    else:
        return returns


def calculate_portfolio_metrics(weights: np.ndarray, mean_returns: np.ndarray,
                                cov_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Negative Sharpe ratio for minimization."""
    _, _, sharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
    return -sharpe


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Portfolio variance for minimum variance optimization."""
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def optimize_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                      objective: str = 'max_sharpe') -> np.ndarray:
    """Optimize portfolio weights."""
    n_assets = len(mean_returns)

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Bounds: 0 <= weight <= 0.25 (max 25% per asset for diversification)
    bounds = tuple((0, 0.25) for _ in range(n_assets))

    # Initial guess: equal weights
    init_weights = np.array([1.0 / n_assets] * n_assets)

    if objective == 'max_sharpe':
        result = minimize(
            negative_sharpe,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    elif objective == 'min_variance':
        result = minimize(
            portfolio_variance,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return result.x if result.success else init_weights


def main():
    print("="*80)
    print(" "*20 + "PORTFOLIO OPTIMIZATION")
    print(" "*15 + "18 Profitable Assets - Optimal Allocation")
    print("="*80)

    start_time = datetime.now()

    print("\nCurrent Status:")
    print("  Profitable Assets: 18/22 = 81.8%")
    print("  Goal: Optimize portfolio allocation for production deployment")

    print(f"\n{'='*80}")
    print("STEP 1: FETCH HISTORICAL RETURNS")
    print(f"{'='*80}\n")

    returns_dict = {}
    valid_assets = []

    for asset in PROFITABLE_ASSETS:
        print(f"Fetching {asset['name']} ({asset['symbol']}, {asset['interval']})...", end=' ')
        returns = fetch_returns(asset['symbol'], asset['interval'], asset['period'])

        if returns is not None and len(returns) >= 20:
            # Normalize to daily frequency
            normalized_returns = normalize_returns_to_daily(returns, asset['interval'])
            returns_dict[asset['name']] = normalized_returns
            valid_assets.append(asset)
            print(f"✅ {len(returns)} bars")
        else:
            print(f"❌ Insufficient data")

    print(f"\nValid Assets: {len(valid_assets)}/18")

    if len(valid_assets) < 10:
        print("❌ Insufficient valid assets for portfolio optimization")
        return

    # Align returns to common dates
    print(f"\n{'='*80}")
    print("STEP 2: ALIGN RETURNS AND CALCULATE STATISTICS")
    print(f"{'='*80}\n")

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.fillna(0)  # Forward fill missing values

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    corr_matrix = returns_df.corr()

    print(f"Aligned Returns Shape: {returns_df.shape}")
    print(f"Date Range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"\nAnnualized Statistics:")
    print(f"  Mean Return: {mean_returns.mean()*252:.2%}")
    print(f"  Mean Volatility: {returns_df.std().mean()*np.sqrt(252):.2%}")

    # Correlation analysis
    print(f"\n{'='*80}")
    print("STEP 3: CORRELATION ANALYSIS")
    print(f"{'='*80}\n")

    print("Top 5 Correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Asset 1': corr_matrix.columns[i],
                'Asset 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
    print(corr_pairs_df.head(5).to_string(index=False))

    print("\nBottom 5 Correlations (Most Diversifying):")
    print(corr_pairs_df.tail(5).to_string(index=False))

    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"\nAverage Correlation: {avg_corr:.3f}")

    # Regime correlations
    print(f"\n{'='*80}")
    print("REGIME CORRELATIONS")
    print(f"{'='*80}\n")

    regime_assets = {}
    for asset in valid_assets:
        regime = asset['regime']
        if regime not in regime_assets:
            regime_assets[regime] = []
        regime_assets[regime].append(asset['name'])

    for regime, assets in regime_assets.items():
        if len(assets) >= 2:
            regime_corr = corr_matrix.loc[assets, assets]
            avg_regime_corr = regime_corr.values[np.triu_indices_from(regime_corr.values, k=1)].mean()
            print(f"{regime}: {len(assets)} assets, avg correlation: {avg_regime_corr:.3f}")

    # Portfolio optimizations
    print(f"\n{'='*80}")
    print("STEP 4: PORTFOLIO OPTIMIZATIONS")
    print(f"{'='*80}\n")

    portfolios = {}

    # 1. Equal Weight
    print("1. Equal Weight Portfolio...")
    n_assets = len(valid_assets)
    equal_weights = np.array([1.0 / n_assets] * n_assets)
    eq_return, eq_vol, eq_sharpe = calculate_portfolio_metrics(
        equal_weights, mean_returns.values, cov_matrix.values
    )
    portfolios['Equal Weight'] = {
        'weights': equal_weights,
        'return': eq_return * 252,
        'volatility': eq_vol * np.sqrt(252),
        'sharpe': eq_sharpe * np.sqrt(252)
    }
    print(f"   Return: {eq_return*252:.2%}, Vol: {eq_vol*np.sqrt(252):.2%}, Sharpe: {eq_sharpe*np.sqrt(252):.2f}")

    # 2. Sharpe-Weighted
    print("\n2. Sharpe-Weighted Portfolio...")
    sharpe_values = np.array([a['sharpe'] for a in valid_assets])
    sharpe_weights = sharpe_values / sharpe_values.sum()
    sw_return, sw_vol, sw_sharpe = calculate_portfolio_metrics(
        sharpe_weights, mean_returns.values, cov_matrix.values
    )
    portfolios['Sharpe-Weighted'] = {
        'weights': sharpe_weights,
        'return': sw_return * 252,
        'volatility': sw_vol * np.sqrt(252),
        'sharpe': sw_sharpe * np.sqrt(252)
    }
    print(f"   Return: {sw_return*252:.2%}, Vol: {sw_vol*np.sqrt(252):.2%}, Sharpe: {sw_sharpe*np.sqrt(252):.2f}")

    # 3. Minimum Variance
    print("\n3. Minimum Variance Portfolio...")
    min_var_weights = optimize_portfolio(mean_returns.values, cov_matrix.values, objective='min_variance')
    mv_return, mv_vol, mv_sharpe = calculate_portfolio_metrics(
        min_var_weights, mean_returns.values, cov_matrix.values
    )
    portfolios['Min Variance'] = {
        'weights': min_var_weights,
        'return': mv_return * 252,
        'volatility': mv_vol * np.sqrt(252),
        'sharpe': mv_sharpe * np.sqrt(252)
    }
    print(f"   Return: {mv_return*252:.2%}, Vol: {mv_vol*np.sqrt(252):.2%}, Sharpe: {mv_sharpe*np.sqrt(252):.2f}")

    # 4. Maximum Sharpe (Markowitz)
    print("\n4. Maximum Sharpe Portfolio (Markowitz)...")
    max_sharpe_weights = optimize_portfolio(mean_returns.values, cov_matrix.values, objective='max_sharpe')
    ms_return, ms_vol, ms_sharpe = calculate_portfolio_metrics(
        max_sharpe_weights, mean_returns.values, cov_matrix.values
    )
    portfolios['Max Sharpe'] = {
        'weights': max_sharpe_weights,
        'return': ms_return * 252,
        'volatility': ms_vol * np.sqrt(252),
        'sharpe': ms_sharpe * np.sqrt(252)
    }
    print(f"   Return: {ms_return*252:.2%}, Vol: {ms_vol*np.sqrt(252):.2%}, Sharpe: {ms_sharpe*np.sqrt(252):.2f}")

    # Compare portfolios
    print(f"\n{'='*80}")
    print("PORTFOLIO COMPARISON")
    print(f"{'='*80}\n")

    comparison_df = pd.DataFrame({
        'Portfolio': list(portfolios.keys()),
        'Return': [p['return'] for p in portfolios.values()],
        'Volatility': [p['volatility'] for p in portfolios.values()],
        'Sharpe': [p['sharpe'] for p in portfolios.values()]
    })

    print(comparison_df.to_string(index=False))

    # Best portfolio
    best_portfolio = max(portfolios.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n✅ Best Portfolio: {best_portfolio[0]} (Sharpe: {best_portfolio[1]['sharpe']:.2f})")

    # Detailed allocation
    print(f"\n{'='*80}")
    print(f"RECOMMENDED ALLOCATION: {best_portfolio[0]}")
    print(f"{'='*80}\n")

    allocation_df = pd.DataFrame({
        'Asset': [a['name'] for a in valid_assets],
        'Symbol': [a['symbol'] for a in valid_assets],
        'Interval': [a['interval'] for a in valid_assets],
        'Regime': [a['regime'] for a in valid_assets],
        'Sharpe': [a['sharpe'] for a in valid_assets],
        'Weight': best_portfolio[1]['weights']
    })

    allocation_df = allocation_df[allocation_df['Weight'] > 0.001]  # Filter out negligible weights
    allocation_df = allocation_df.sort_values('Weight', ascending=False)
    allocation_df['Weight %'] = allocation_df['Weight'] * 100

    print(allocation_df[['Asset', 'Regime', 'Interval', 'Sharpe', 'Weight %']].to_string(index=False))

    # Regime allocation
    print(f"\n{'='*80}")
    print("REGIME ALLOCATION")
    print(f"{'='*80}\n")

    regime_allocation = allocation_df.groupby('Regime')['Weight'].sum().sort_values(ascending=False)
    for regime, weight in regime_allocation.items():
        print(f"{regime:15s}: {weight*100:5.1f}%")

    # Production recommendations
    print(f"\n{'='*80}")
    print("PRODUCTION DEPLOYMENT RECOMMENDATIONS")
    print(f"{'='*80}\n")

    print("1. Portfolio Configuration:")
    print(f"   Expected Annual Return: {best_portfolio[1]['return']:.2%}")
    print(f"   Expected Annual Volatility: {best_portfolio[1]['volatility']:.2%}")
    print(f"   Expected Sharpe Ratio: {best_portfolio[1]['sharpe']:.2f}")

    print("\n2. Conservative Adjustments (50% safety margin):")
    print(f"   Production Return Estimate: {best_portfolio[1]['return']*0.5:.2%}")
    print(f"   Production Sharpe Estimate: {best_portfolio[1]['sharpe']*0.5:.2f}")

    print("\n3. Risk Controls:")
    print("   - Maximum position size per asset: 25%")
    print("   - Rebalance frequency: Weekly")
    print("   - Stop-loss per asset: -5%")
    print("   - Portfolio stop-loss: -10%")

    print("\n4. Top 5 Holdings:")
    top5 = allocation_df.head(5)
    for _, row in top5.iterrows():
        print(f"   {row['Asset']:12s} ({row['Interval']:3s}): {row['Weight %']:5.1f}%")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    # Save allocation
    allocation_df.to_csv('PORTFOLIO_ALLOCATION.csv', index=False)
    print("✅ Saved: PORTFOLIO_ALLOCATION.csv")

    # Save correlation matrix
    corr_matrix.to_csv('CORRELATION_MATRIX.csv')
    print("✅ Saved: CORRELATION_MATRIX.csv")

    # Save portfolio comparison
    comparison_df.to_csv('PORTFOLIO_COMPARISON.csv', index=False)
    print("✅ Saved: PORTFOLIO_COMPARISON.csv")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n\nOptimization Duration: {duration:.1f} seconds")

    print(f"\n{'='*80}")
    print("PORTFOLIO OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    print(f"\n🎉 RECOMMENDED PORTFOLIO: {best_portfolio[0]}")
    print(f"📊 Expected Sharpe Ratio: {best_portfolio[1]['sharpe']:.2f}")
    print(f"📈 Expected Annual Return: {best_portfolio[1]['return']:.2%}")
    print(f"⚠️  Production Estimate (50% margin): {best_portfolio[1]['return']*0.5:.2%} return, {best_portfolio[1]['sharpe']*0.5:.2f} Sharpe")


if __name__ == "__main__":
    main()
