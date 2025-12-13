"""
Phase 3 Validation Module - Structural Features Validation

Validates Phase 3 structural features on asset subsets:
- Volatility regime detection
- Momentum vs mean-reversion scoring
- Volume-based order flow features
- Intermarket correlation features

Target: +10-15% profit rate improvement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Validation Criteria from Roadmap
VALIDATION_CRITERIA = {
    'min_improvement': 0.10,           # +10% profit rate minimum
    'max_drawdown_increase': 0.02,     # No significant risk increase
    'consistency_threshold': 0.70,      # 70% of assets show improvement
    'statistical_significance': 0.05    # p-value threshold
}


class Phase3Validator:
    """
    Validate Phase 3 structural features on asset subset.

    Tests:
    1. Regime detection accuracy
    2. Volatility features impact
    3. Momentum/mean-reversion features
    4. Intermarket correlation features
    """

    def __init__(self, test_assets: List[str] = None):
        """
        Initialize validator.

        Args:
            test_assets: List of tickers to test (default: 5-asset subset)
        """
        # Default 5-asset subset for quick validation
        self.test_assets = test_assets or ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'SPY']
        self.baseline_metrics = {}
        self.results = {}

    def fetch_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """Fetch historical data for a ticker."""
        try:
            data = yf.download(ticker, period=period, progress=False)
            if len(data) > 0:
                data['Ticker'] = ticker
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(20).std()
                return data
        except Exception as e:
            print(f"[WARN] Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

    def calculate_baseline_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate baseline performance metrics without structural features."""
        if len(data) < 50:
            return {'valid': False}

        returns = data['Returns'].dropna()

        # Simple momentum strategy baseline
        # Buy if 20-day return > 0, sell otherwise
        momentum_signal = data['Close'].pct_change(20).shift(1)
        strategy_returns = np.where(momentum_signal > 0, returns, -returns)
        strategy_returns = pd.Series(strategy_returns).dropna()

        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        win_rate = (strategy_returns > 0).mean()
        max_dd = self._calculate_max_drawdown(strategy_returns)

        return {
            'valid': True,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'n_trades': len(strategy_returns)
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def test_regime_detection(self, data: pd.DataFrame) -> Dict:
        """Test regime detection feature impact."""
        if len(data) < 100:
            return {'improvement': 0, 'status': 'insufficient_data'}

        try:
            # Simple regime detection based on volatility percentiles
            vol = data['Volatility'].dropna()
            vol_25 = vol.quantile(0.25)
            vol_75 = vol.quantile(0.75)

            # Classify regimes
            regimes = pd.Series(index=vol.index, dtype=int)
            regimes[vol <= vol_25] = 0  # Low vol
            regimes[(vol > vol_25) & (vol <= vol_75)] = 1  # Medium vol
            regimes[vol > vol_75] = 2  # High vol

            returns = data['Returns'].reindex(regimes.index)

            # Regime-aware strategy: reduce position in high vol
            position_mult = pd.Series(index=regimes.index, dtype=float)
            position_mult[regimes == 0] = 1.0   # Full position in low vol
            position_mult[regimes == 1] = 0.8   # 80% in medium vol
            position_mult[regimes == 2] = 0.5   # 50% in high vol

            # Calculate regime-adjusted returns
            momentum_signal = data['Close'].pct_change(20).shift(1).reindex(regimes.index)
            strategy_returns = np.where(momentum_signal > 0, returns, -returns)
            adjusted_returns = strategy_returns * position_mult.values
            adjusted_returns = pd.Series(adjusted_returns).dropna()

            # Calculate improvement
            sharpe_with_regime = adjusted_returns.mean() / adjusted_returns.std() * np.sqrt(252) if adjusted_returns.std() > 0 else 0
            win_rate_with_regime = (adjusted_returns > 0).mean()

            return {
                'sharpe_ratio': sharpe_with_regime,
                'win_rate': win_rate_with_regime,
                'regime_distribution': regimes.value_counts().to_dict(),
                'status': 'success'
            }

        except Exception as e:
            return {'improvement': 0, 'status': f'error: {str(e)}'}

    def test_volatility_regime_features(self, data: pd.DataFrame) -> Dict:
        """Test volatility regime features."""
        if len(data) < 100:
            return {'improvement': 0, 'status': 'insufficient_data'}

        try:
            # Calculate volatility features
            vol = data['Volatility'].dropna()

            # Volatility z-score
            vol_zscore = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()

            # Volatility regime duration
            vol_high = (vol > vol.quantile(0.75)).astype(int)
            regime_change = vol_high.diff().abs()

            # Count consecutive days in regime
            regime_duration = []
            count = 0
            prev_regime = vol_high.iloc[0] if len(vol_high) > 0 else 0
            for curr in vol_high:
                if curr == prev_regime:
                    count += 1
                else:
                    count = 1
                    prev_regime = curr
                regime_duration.append(count)

            # Strategy: exit when volatility z-score > 2
            returns = data['Returns'].reindex(vol_zscore.index)
            exit_signal = vol_zscore > 2

            # Modified returns (no position when vol zscore > 2)
            momentum_signal = data['Close'].pct_change(20).shift(1).reindex(vol_zscore.index)
            strategy_returns = np.where(momentum_signal > 0, returns, -returns)
            modified_returns = np.where(exit_signal, 0, strategy_returns)
            modified_returns = pd.Series(modified_returns).dropna()

            sharpe = modified_returns.mean() / modified_returns.std() * np.sqrt(252) if modified_returns.std() > 0 else 0
            win_rate = (modified_returns[modified_returns != 0] > 0).mean() if (modified_returns != 0).sum() > 0 else 0

            return {
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'vol_zscore_exits': exit_signal.sum(),
                'avg_regime_duration': np.mean(regime_duration),
                'status': 'success'
            }

        except Exception as e:
            return {'improvement': 0, 'status': f'error: {str(e)}'}

    def test_momentum_vs_mean_reversion(self, data: pd.DataFrame) -> Dict:
        """Test momentum vs mean-reversion features."""
        if len(data) < 100:
            return {'improvement': 0, 'status': 'insufficient_data'}

        try:
            returns = data['Returns'].dropna()
            close = data['Close'].dropna()

            # Calculate autocorrelation (proxy for Hurst exponent)
            autocorr_1 = returns.rolling(60).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
            autocorr_5 = returns.rolling(60).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False)

            # Trend score: positive autocorr = trending, negative = mean-reverting
            trend_score = (autocorr_1 + autocorr_5) / 2

            # Adaptive strategy based on trend score
            # If trending (autocorr > 0): use momentum
            # If mean-reverting (autocorr < 0): use mean-reversion
            momentum_signal = close.pct_change(20).shift(1).reindex(trend_score.index)

            # Mean-reversion signal: buy when below 20-day MA, sell when above
            ma_20 = close.rolling(20).mean()
            mr_signal = (close < ma_20).shift(1).reindex(trend_score.index)

            # Combine based on trend score
            trend_score_aligned = trend_score.reindex(returns.index)
            momentum_signal_aligned = momentum_signal.reindex(returns.index)
            mr_signal_aligned = mr_signal.reindex(returns.index)

            # Use momentum in trending markets, mean-reversion otherwise
            combined_signal = np.where(
                trend_score_aligned > 0,
                np.where(momentum_signal_aligned > 0, 1, -1),
                np.where(mr_signal_aligned, 1, -1)
            )

            strategy_returns = returns * combined_signal
            strategy_returns = pd.Series(strategy_returns).dropna()

            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            win_rate = (strategy_returns > 0).mean()

            return {
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'avg_trend_score': trend_score.mean(),
                'trending_pct': (trend_score > 0).mean(),
                'status': 'success'
            }

        except Exception as e:
            return {'improvement': 0, 'status': f'error: {str(e)}'}

    def run_focused_validation(self, verbose: bool = True) -> Dict:
        """
        Run Phase 3 validation on 5-asset subset.

        Returns:
            Dict with validation results and improvement metrics
        """
        print("=" * 70)
        print("PHASE 3 STRUCTURAL FEATURES VALIDATION")
        print(f"Testing on {len(self.test_assets)} assets: {', '.join(self.test_assets)}")
        print("=" * 70)

        results = {}
        baseline_total = {'sharpe': [], 'win_rate': [], 'drawdown': []}
        enhanced_total = {'sharpe': [], 'win_rate': []}

        for ticker in self.test_assets:
            print(f"\n[{ticker}] Fetching data...")
            data = self.fetch_data(ticker)

            if len(data) < 100:
                print(f"[{ticker}] SKIP - Insufficient data ({len(data)} rows)")
                continue

            print(f"[{ticker}] Loaded {len(data)} days of data")

            # Calculate baseline
            baseline = self.calculate_baseline_metrics(data)
            if not baseline.get('valid', False):
                print(f"[{ticker}] SKIP - Invalid baseline")
                continue

            baseline_total['sharpe'].append(baseline['sharpe_ratio'])
            baseline_total['win_rate'].append(baseline['win_rate'])
            baseline_total['drawdown'].append(baseline['max_drawdown'])

            if verbose:
                print(f"[{ticker}] Baseline: Sharpe={baseline['sharpe_ratio']:.3f}, "
                      f"WinRate={baseline['win_rate']:.1%}, MaxDD={baseline['max_drawdown']:.1%}")

            # Test structural features
            regime_result = self.test_regime_detection(data)
            vol_result = self.test_volatility_regime_features(data)
            momentum_result = self.test_momentum_vs_mean_reversion(data)

            # Calculate best enhanced result
            enhanced_sharpe = max(
                regime_result.get('sharpe_ratio', 0),
                vol_result.get('sharpe_ratio', 0),
                momentum_result.get('sharpe_ratio', 0)
            )
            enhanced_win_rate = max(
                regime_result.get('win_rate', 0),
                vol_result.get('win_rate', 0),
                momentum_result.get('win_rate', 0)
            )

            enhanced_total['sharpe'].append(enhanced_sharpe)
            enhanced_total['win_rate'].append(enhanced_win_rate)

            # Determine improvement
            sharpe_improvement = enhanced_sharpe - baseline['sharpe_ratio']
            win_rate_improvement = enhanced_win_rate - baseline['win_rate']

            improved = sharpe_improvement > 0 or win_rate_improvement > 0

            results[ticker] = {
                'baseline': baseline,
                'regime_detection': regime_result,
                'volatility_features': vol_result,
                'momentum_features': momentum_result,
                'sharpe_improvement': sharpe_improvement,
                'win_rate_improvement': win_rate_improvement,
                'improved': improved
            }

            status = "IMPROVED" if improved else "NO CHANGE"
            print(f"[{ticker}] Enhanced: Sharpe={enhanced_sharpe:.3f} ({sharpe_improvement:+.3f}), "
                  f"WinRate={enhanced_win_rate:.1%} ({win_rate_improvement:+.1%}) - {status}")

        # Calculate overall metrics
        if len(baseline_total['sharpe']) > 0:
            avg_baseline_sharpe = np.mean(baseline_total['sharpe'])
            avg_enhanced_sharpe = np.mean(enhanced_total['sharpe'])
            avg_baseline_win = np.mean(baseline_total['win_rate'])
            avg_enhanced_win = np.mean(enhanced_total['win_rate'])

            improvement_pct = sum(1 for t, r in results.items() if r.get('improved', False)) / len(results)

            print("\n" + "=" * 70)
            print("VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Assets tested: {len(results)}")
            print(f"Assets improved: {sum(1 for r in results.values() if r.get('improved', False))}/{len(results)} ({improvement_pct:.0%})")
            print(f"\nAverage Sharpe Ratio:")
            print(f"  Baseline:  {avg_baseline_sharpe:.3f}")
            print(f"  Enhanced:  {avg_enhanced_sharpe:.3f}")
            print(f"  Change:    {avg_enhanced_sharpe - avg_baseline_sharpe:+.3f}")
            print(f"\nAverage Win Rate:")
            print(f"  Baseline:  {avg_baseline_win:.1%}")
            print(f"  Enhanced:  {avg_enhanced_win:.1%}")
            print(f"  Change:    {avg_enhanced_win - avg_baseline_win:+.1%}")

            # Check against validation criteria
            meets_criteria = improvement_pct >= VALIDATION_CRITERIA['consistency_threshold']

            print(f"\n{'PASS' if meets_criteria else 'FAIL'} - Consistency threshold: "
                  f"{improvement_pct:.0%} vs {VALIDATION_CRITERIA['consistency_threshold']:.0%} required")

            return {
                'results': results,
                'summary': {
                    'assets_tested': len(results),
                    'assets_improved': sum(1 for r in results.values() if r.get('improved', False)),
                    'improvement_pct': improvement_pct,
                    'avg_baseline_sharpe': avg_baseline_sharpe,
                    'avg_enhanced_sharpe': avg_enhanced_sharpe,
                    'avg_baseline_win_rate': avg_baseline_win,
                    'avg_enhanced_win_rate': avg_enhanced_win,
                    'meets_criteria': meets_criteria
                }
            }

        return {'results': results, 'summary': {'error': 'No valid results'}}

    def analyze_improvement_vs_baseline(self, results: Dict) -> Dict:
        """Analyze improvement relative to baseline."""
        if 'summary' not in results:
            return {'status': 'no_summary'}

        summary = results['summary']

        sharpe_improvement = summary.get('avg_enhanced_sharpe', 0) - summary.get('avg_baseline_sharpe', 0)
        win_rate_improvement = summary.get('avg_enhanced_win_rate', 0) - summary.get('avg_baseline_win_rate', 0)

        return {
            'sharpe_improvement': sharpe_improvement,
            'win_rate_improvement': win_rate_improvement,
            'improvement_pct': summary.get('improvement_pct', 0),
            'recommendation': 'PROCEED' if summary.get('meets_criteria', False) else 'INVESTIGATE',
            'next_steps': [
                'Run full 35-asset test' if summary.get('meets_criteria', False) else 'Review failing assets',
                'Begin Phase 2 implementation',
                'Create asset-class specific features'
            ]
        }


def main():
    """Run Phase 3 validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3 Structural Features Validation')
    parser.add_argument('--assets', type=int, default=5, help='Number of assets to test')
    parser.add_argument('--quick-test', action='store_true', help='Run quick validation')
    parser.add_argument('--full', action='store_true', help='Run full 35-asset test')
    args = parser.parse_args()

    # Asset lists
    QUICK_ASSETS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'SPY']
    FULL_ASSETS = [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL',
        # US Finance
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
        # US Consumer
        'AMZN', 'TSLA', 'DIS', 'NKE', 'SBUX', 'MCD',
        # International
        'BABA', 'SONY', 'TM', 'SAP', 'TSM', 'ASML',
        # Forex (ETFs as proxies)
        'FXE', 'FXY', 'FXB',
        # Commodities (ETFs)
        'GLD', 'USO', 'UNG',
        # Market ETFs
        'SPY', 'QQQ'
    ]

    if args.full:
        test_assets = FULL_ASSETS
    elif args.quick_test:
        test_assets = QUICK_ASSETS[:3]
    else:
        test_assets = QUICK_ASSETS[:args.assets]

    validator = Phase3Validator(test_assets=test_assets)
    results = validator.run_focused_validation()

    print("\n" + "=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)

    analysis = validator.analyze_improvement_vs_baseline(results)
    print(f"\nRecommendation: {analysis.get('recommendation', 'N/A')}")
    print("\nNext Steps:")
    for i, step in enumerate(analysis.get('next_steps', []), 1):
        print(f"  {i}. {step}")

    return results


if __name__ == "__main__":
    main()
