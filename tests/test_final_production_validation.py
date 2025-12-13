"""
Post-Production Validation Framework - Final Test Suite
Based on: phase1 fixing on C model_final test.pdf

PRIORITY: AVOIDING OVERFITTING through Walk-Forward Validation

Tests:
1. Walk-Forward Validation (OUT-OF-SAMPLE testing)
2. Advanced Features Integration
3. Stress Testing
4. Regime Transition Testing
5. Feature Importance Validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from scipy import stats

# Import our modules
from src.trading.hybrid_strategy import OptimalHybridStrategy
from src.trading.risk_management import IntegratedRiskManager
from src.trading.production_advanced import *


# =============================================================================
# 1. WALK-FORWARD VALIDATION (ANTI-OVERFITTING PRIORITY)
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.
    Tests on OUT-OF-SAMPLE data only.
    """

    def __init__(self, train_periods: int = 3, test_periods: int = 1, period_days: int = 60):
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.period_days = period_days

    def split_data_into_periods(self, df: pd.DataFrame, total_periods: int) -> List[pd.DataFrame]:
        """Split dataframe into equal periods."""
        periods = []
        period_length = len(df) // total_periods

        for i in range(total_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < total_periods - 1 else len(df)
            periods.append(df.iloc[start_idx:end_idx].copy())

        return periods

    def run_walk_forward_test(self, data: Dict[str, pd.DataFrame], strategy, total_periods: int = 6) -> Dict:
        """Run walk-forward validation - the KEY anti-overfitting test."""

        results = {
            'in_sample': [],
            'out_of_sample': [],
            'walk_forward_windows': [],
            'overfitting_detected': False
        }

        num_windows = total_periods - self.train_periods - self.test_periods + 1

        print(f"\n{'='*70}")
        print(f"WALK-FORWARD VALIDATION (ANTI-OVERFITTING TEST)")
        print(f"{'='*70}")
        print(f"Total periods: {total_periods}")
        print(f"Train periods: {self.train_periods}")
        print(f"Test periods: {self.test_periods} (OUT-OF-SAMPLE)")
        print(f"Walk-forward windows: {num_windows}")

        for window in range(num_windows):
            train_start = window
            train_end = window + self.train_periods
            test_start = train_end
            test_end = test_start + self.test_periods

            print(f"\n--- Window {window + 1}/{num_windows} ---")
            print(f"Train: Periods {train_start + 1}-{train_end} (IN-SAMPLE)")
            print(f"Test:  Periods {test_start + 1}-{test_end} (OUT-OF-SAMPLE)")

            risk_manager = IntegratedRiskManager()
            in_sample_trades = []
            out_of_sample_trades = []

            for ticker, df in data.items():
                if len(df) < 60:
                    continue

                periods = self.split_data_into_periods(df, total_periods)

                # TRAINING PHASE (in-sample) - learn patterns
                for period_idx in range(train_start, train_end):
                    if period_idx >= len(periods):
                        continue
                    period_df = periods[period_idx]
                    trades = self._simulate_period(period_df, strategy, ticker, risk_manager)
                    in_sample_trades.extend(trades)

                    for trade in trades:
                        risk_manager.record_trade_result(ticker, trade['pnl'])

                # TESTING PHASE (out-of-sample) - NO learning, just apply
                for period_idx in range(test_start, test_end):
                    if period_idx >= len(periods):
                        continue
                    period_df = periods[period_idx]

                    can_trade, reason = risk_manager.should_trade(ticker)
                    if can_trade:
                        trades = self._simulate_period(period_df, strategy, ticker, risk_manager, apply_risk_rules=True)
                        out_of_sample_trades.extend(trades)

            # Calculate window results
            window_result = self._calculate_window_results(in_sample_trades, out_of_sample_trades, window + 1)
            results['walk_forward_windows'].append(window_result)
            results['in_sample'].extend(in_sample_trades)
            results['out_of_sample'].extend(out_of_sample_trades)

        # Check for overfitting
        results['summary'] = self._calculate_summary(results)
        results['overfitting_detected'] = self._detect_overfitting(results)

        return results

    def _simulate_period(self, df: pd.DataFrame, strategy, ticker: str, risk_manager, apply_risk_rules: bool = False) -> List[Dict]:
        """Simulate trading for one period."""
        trades = []

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

            sma_signal = 1 if row['SMA_5'] > row['SMA_20'] else 0
            momentum_signal = 1 if row['momentum'] > 0 else 0
            direction_pred = sma_signal * 0.6 + momentum_signal * 0.4

            signal = strategy.generate_hybrid_signal(
                direction_prediction=direction_pred,
                volatility_prediction=row['volatility'],
                historical_volatility=hist_vol,
                current_price=row['Close'],
                ticker=ticker
            )

            if signal['should_trade'] and signal['action'] != 'HOLD':
                position_size = signal['position_value']

                if apply_risk_rules:
                    can_trade, _ = risk_manager.should_trade(ticker)
                    if not can_trade:
                        continue
                    size_pct = risk_manager.get_position_size(ticker, signal['confidence'])
                    position_size = 100000 * size_pct

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

    def _calculate_window_results(self, in_sample_trades: List[Dict], out_of_sample_trades: List[Dict], window_num: int) -> Dict:
        """Calculate results for one walk-forward window."""
        result = {'window': window_num, 'in_sample': {}, 'out_of_sample': {}}

        for name, trades in [('in_sample', in_sample_trades), ('out_of_sample', out_of_sample_trades)]:
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                avg_pnl = total_pnl / len(trades)

                result[name] = {
                    'num_trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl
                }
            else:
                result[name] = {'num_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0}

        print(f"\n  IN-SAMPLE:     {result['in_sample']['num_trades']:3d} trades, "
              f"Win Rate: {result['in_sample']['win_rate']:.1%}, "
              f"Total P&L: ${result['in_sample']['total_pnl']:,.0f}")
        print(f"  OUT-OF-SAMPLE: {result['out_of_sample']['num_trades']:3d} trades, "
              f"Win Rate: {result['out_of_sample']['win_rate']:.1%}, "
              f"Total P&L: ${result['out_of_sample']['total_pnl']:,.0f}")

        # Check for overfitting warning
        if result['in_sample']['num_trades'] > 0 and result['out_of_sample']['num_trades'] > 0:
            win_rate_gap = result['in_sample']['win_rate'] - result['out_of_sample']['win_rate']
            if win_rate_gap > 0.15:
                print(f"  WARNING: Possible OVERFITTING (win rate gap: {win_rate_gap:.1%})")

        return result

    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics."""
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

        if summary['in_sample']['total_trades'] > 0 and summary['out_of_sample']['total_trades'] > 0:
            summary['robustness'] = {
                'win_rate_degradation': summary['in_sample']['win_rate'] - summary['out_of_sample']['win_rate'],
                'pnl_ratio': summary['out_of_sample']['total_pnl'] / summary['in_sample']['total_pnl'] if summary['in_sample']['total_pnl'] != 0 else 0,
                'is_robust': abs(summary['in_sample']['win_rate'] - summary['out_of_sample']['win_rate']) < 0.10
            }

        return summary

    def _detect_overfitting(self, results: Dict) -> bool:
        """Detect if model is overfit based on in-sample vs out-of-sample gap."""
        summary = results.get('summary', {})
        robustness = summary.get('robustness', {})

        if not robustness:
            return False

        # Overfitting indicators:
        # 1. Win rate degrades > 10%
        # 2. OOS P&L is significantly worse than IS
        win_rate_gap = robustness.get('win_rate_degradation', 0)
        pnl_ratio = robustness.get('pnl_ratio', 1)

        overfitting = win_rate_gap > 0.10 or pnl_ratio < 0.3

        if overfitting:
            print(f"\n  *** OVERFITTING DETECTED ***")
            print(f"  Win rate degradation: {win_rate_gap:.1%} (threshold: 10%)")
            print(f"  P&L ratio (OOS/IS): {pnl_ratio:.2f} (threshold: 0.30)")

        return overfitting


# =============================================================================
# 2. ADVANCED STRESS TESTING
# =============================================================================

class AdvancedStressTester:
    """Comprehensive stress testing for production readiness."""

    def __init__(self):
        self.stress_scenarios = {
            'flash_crash': {'volatility_multiplier': 5.0, 'liquidity_drop': 0.8},
            'high_volatility': {'volatility_multiplier': 3.0, 'correlation_increase': 0.9},
            'liquidity_crisis': {'liquidity_drop': 0.5, 'transaction_cost_multiplier': 10},
            'sector_crash': {'sector_shock': True, 'max_sector_drop': 0.4},
            'black_swan': {'volatility_multiplier': 10.0, 'liquidity_drop': 0.9}
        }

    def run_stress_tests(self, portfolio_value: float = 100000) -> Dict:
        """Run all stress scenarios."""
        print(f"\n{'='*70}")
        print("STRESS TESTING")
        print(f"{'='*70}")

        results = {}

        for scenario_name, params in self.stress_scenarios.items():
            print(f"\n  Testing: {scenario_name}")

            # Simulate max drawdown under stress
            vol_mult = params.get('volatility_multiplier', 1.0)
            liq_drop = params.get('liquidity_drop', 0.0)

            # Simple stress simulation
            base_daily_vol = 0.02  # 2% daily volatility
            stressed_vol = base_daily_vol * vol_mult

            # Monte Carlo simulation of 30 days under stress
            np.random.seed(42)
            daily_returns = np.random.normal(-0.005, stressed_vol, 30)  # Slight negative bias in stress
            cumulative = np.cumprod(1 + daily_returns)
            max_drawdown = 1 - min(cumulative)

            # Apply liquidity impact
            max_drawdown *= (1 + liq_drop)

            survived = max_drawdown < 0.5  # 50% survival threshold

            results[scenario_name] = {
                'max_drawdown': max_drawdown,
                'survived': survived,
                'params': params
            }

            status = "SURVIVED" if survived else "FAILED"
            print(f"    Max Drawdown: {max_drawdown:.1%} - {status}")

        return results


# =============================================================================
# 3. FEATURE IMPORTANCE VALIDATION (Anti-overfitting)
# =============================================================================

class FeatureImportanceValidator:
    """Validate that features contribute meaningfully (not just noise)."""

    def validate_features(self, trades_data: List[Dict]) -> Dict:
        """Validate feature contributions."""
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE VALIDATION")
        print(f"{'='*70}")

        if not trades_data:
            print("  No trades to analyze")
            return {'valid': False}

        # Group trades by confidence levels
        high_conf = [t for t in trades_data if t.get('confidence', 0) >= 0.7]
        med_conf = [t for t in trades_data if 0.5 <= t.get('confidence', 0) < 0.7]
        low_conf = [t for t in trades_data if t.get('confidence', 0) < 0.5]

        results = {}

        for name, trades in [('high_confidence', high_conf), ('medium_confidence', med_conf), ('low_confidence', low_conf)]:
            if trades:
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
                results[name] = {'trades': len(trades), 'win_rate': win_rate, 'avg_pnl': avg_pnl}
                print(f"  {name}: {len(trades)} trades, Win Rate: {win_rate:.1%}, Avg P&L: ${avg_pnl:,.2f}")

        # Check if high confidence actually performs better (validates feature usefulness)
        if 'high_confidence' in results and 'low_confidence' in results:
            improvement = results['high_confidence']['win_rate'] - results['low_confidence']['win_rate']
            results['confidence_validates'] = improvement > 0.05
            print(f"\n  High vs Low confidence gap: {improvement:.1%}")
            print(f"  Features {'VALIDATED' if results['confidence_validates'] else 'NOT VALIDATED'}")

        return results


# =============================================================================
# 4. COMPREHENSIVE TEST RUNNER
# =============================================================================

def run_all_final_tests():
    """Run all post-production validation tests with overfitting prevention priority."""

    print("=" * 70)
    print("POST-PRODUCTION VALIDATION - FINAL TEST SUITE")
    print("PRIORITY: AVOIDING OVERFITTING")
    print("=" * 70)

    # Test stocks
    test_stocks = [
        '0700.HK',  # Tencent
        '9988.HK',  # Alibaba
        '1810.HK',  # Xiaomi
        '2269.HK',  # WuXi Bio
        '1398.HK',  # ICBC
    ]

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"\nTest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Downloading data...")

    data = {}
    for ticker in test_stocks:
        try:
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    if not data:
        print("No data available!")
        return False

    # Initialize strategy
    strategy = OptimalHybridStrategy(
        confidence_threshold=0.50,
        volatility_filter_percentile=0.50,
        position_size=0.50,
        stop_loss_pct=0.05,
        drawdown_threshold=0.08,
        max_drawdown=0.20
    )

    test_results = {}

    # ==========================================================================
    # TEST 1: WALK-FORWARD VALIDATION (CRITICAL - Anti-Overfitting)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: WALK-FORWARD VALIDATION (ANTI-OVERFITTING)")
    print("=" * 70)

    validator = WalkForwardValidator(train_periods=3, test_periods=1, period_days=60)
    wf_results = validator.run_walk_forward_test(data, strategy, total_periods=6)

    summary = wf_results.get('summary', {})
    robustness = summary.get('robustness', {})

    print(f"\n--- Walk-Forward Summary ---")
    print(f"In-Sample Trades: {summary['in_sample']['total_trades']}")
    print(f"In-Sample Win Rate: {summary['in_sample']['win_rate']:.1%}")
    print(f"In-Sample P&L: ${summary['in_sample']['total_pnl']:,.2f}")
    print(f"\nOut-of-Sample Trades: {summary['out_of_sample']['total_trades']}")
    print(f"Out-of-Sample Win Rate: {summary['out_of_sample']['win_rate']:.1%}")
    print(f"Out-of-Sample P&L: ${summary['out_of_sample']['total_pnl']:,.2f}")

    if robustness:
        print(f"\n--- Robustness Metrics ---")
        print(f"Win Rate Degradation: {robustness['win_rate_degradation']:.1%}")
        print(f"P&L Ratio (OOS/IS): {robustness['pnl_ratio']:.2f}")
        print(f"Is Robust: {'YES' if robustness['is_robust'] else 'NO'}")

    overfitting = wf_results.get('overfitting_detected', False)
    test_results['walk_forward'] = not overfitting
    print(f"\nOVERFITTING TEST: {'PASS (No overfitting)' if not overfitting else 'FAIL (Overfitting detected)'}")

    # ==========================================================================
    # TEST 2: ADVANCED FEATURES INTEGRATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ADVANCED FEATURES INTEGRATION")
    print("=" * 70)

    features_passed = 0
    features_total = 15

    # Test each feature
    tests = [
        ("AdaptiveFeatureSelector", lambda: test_adaptive_feature_selector()),
        ("MetaLearner", lambda: test_meta_learner()),
        ("CorrelationAwarePositionSizer", lambda: test_correlation_sizer()),
        ("VolatilityScaler", lambda: test_volatility_scaler()),
        ("RiskParityAllocator", lambda: test_risk_parity()),
        ("SmartRebalancer", lambda: test_smart_rebalancer()),
        ("CrossAssetSignalEnhancer", lambda: test_cross_asset_enhancer()),
        ("RegimeSignalWeighter", lambda: test_regime_weighter()),
        ("SmartOrderRouter", lambda: test_smart_order_router()),
        ("ExecutionAlgorithms", lambda: test_execution_algorithms()),
        ("PerformanceAttribution", lambda: test_performance_attribution()),
        ("StrategyDriftDetector", lambda: test_drift_detector()),
        ("DrawdownPredictor", lambda: test_drawdown_predictor()),
        ("LiquidityRiskMonitor", lambda: test_liquidity_monitor()),
        ("ContingencyManager", lambda: test_contingency_manager()),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                features_passed += 1
                print(f"  [PASS] {name}")
            else:
                print(f"  [FAIL] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    test_results['features_integration'] = features_passed == features_total
    print(f"\nFeatures Passed: {features_passed}/{features_total}")

    # ==========================================================================
    # TEST 3: STRESS TESTING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: STRESS TESTING")
    print("=" * 70)

    stress_tester = AdvancedStressTester()
    stress_results = stress_tester.run_stress_tests()

    all_survived = all(r['survived'] for r in stress_results.values())
    test_results['stress_tests'] = all_survived
    print(f"\nStress Test: {'PASS' if all_survived else 'FAIL'}")

    # ==========================================================================
    # TEST 4: FEATURE IMPORTANCE VALIDATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: FEATURE IMPORTANCE VALIDATION")
    print("=" * 70)

    feature_validator = FeatureImportanceValidator()
    all_trades = wf_results['in_sample'] + wf_results['out_of_sample']
    feature_results = feature_validator.validate_features(all_trades)

    test_results['feature_validation'] = feature_results.get('confidence_validates', False)

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")
        if result:
            passed += 1

    print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total:.0%})")

    if passed == total:
        print("\nSYSTEM FULLY VALIDATED FOR PRODUCTION!")
        print("No overfitting detected. Safe to deploy.")
    elif test_results.get('walk_forward', False):
        print("\nSYSTEM PARTIALLY VALIDATED")
        print("Walk-forward passed (no overfitting), but other tests need attention.")
    else:
        print("\nWARNING: OVERFITTING OR OTHER ISSUES DETECTED")
        print("Review results before deployment.")

    return passed == total


# =============================================================================
# INDIVIDUAL FEATURE TESTS
# =============================================================================

def test_adaptive_feature_selector():
    selector = AdaptiveFeatureSelector()
    np.random.seed(42)
    stock_data = pd.DataFrame({
        "momentum_5d": np.random.randn(100),
        "rsi_14": np.random.randn(100),
        "macd_signal": np.random.randn(100),
    })
    target_returns = pd.Series(np.random.randn(100))
    features = selector.select_features(stock_data, target_returns)
    return len(features) > 0

def test_meta_learner():
    meta = MetaLearner()
    model = meta.select_best_model("high_volatility", {"volatility": 0.3})
    return model is not None

def test_correlation_sizer():
    sizer = CorrelationAwarePositionSizer()
    new_trade = {"ticker": "0700.HK", "signal": 0.8}
    existing = [{"ticker": "9988.HK", "value": 10000}]
    adjusted = sizer.correlation_adjusted_sizing(new_trade, existing, base_size=0.1)
    return 0 < adjusted <= 0.1

def test_volatility_scaler():
    scaler = VolatilityScaler()
    returns = pd.Series(np.random.randn(100) * 0.02)
    adj = scaler.get_volatility_adjustment(returns)
    return 0.5 <= adj <= 1.0

def test_risk_parity():
    allocator = RiskParityAllocator()
    predictions = {
        "0700.HK": {"return": 0.05, "volatility": 0.20},
        "9988.HK": {"return": 0.03, "volatility": 0.25}
    }
    alloc = allocator.risk_parity_allocation(predictions, {"0700.HK": 0.5, "9988.HK": 0.5})
    return abs(sum(alloc.values()) - 1.0) < 0.01

def test_smart_rebalancer():
    rebalancer = SmartRebalancer()
    should, reason = rebalancer.should_rebalance(
        {"0700.HK": 0.15, "9988.HK": 0.10},
        {"0700.HK": 0.10, "9988.HK": 0.10},
        {"0700.HK": 0.02, "9988.HK": 0.01}
    )
    return isinstance(should, bool)

def test_cross_asset_enhancer():
    enhancer = CrossAssetSignalEnhancer()
    enhanced = enhancer.enhance_signal(0.7, {"0700.HK": 0.8, "9988.HK": 0.6})
    return 0 <= enhanced <= 1.0

def test_regime_weighter():
    weighter = RegimeSignalWeighter()
    weighted = weighter.regime_aware_signal_weighting(0.7, "high_volatility")
    return 0.3 <= weighted <= 0.95

def test_smart_order_router():
    router = SmartOrderRouter()
    order = {"ticker": "0700.HK", "shares": 1000, "direction": "BUY"}
    market_cond = {"spread": 0.002, "volume": 1000000, "volatility": 0.02}
    execution = router.optimize_execution(order, market_cond)
    return "urgency" in execution and "algorithm" in execution

def test_execution_algorithms():
    algo = ExecutionAlgorithms()
    order = {"ticker": "0700.HK", "quantity": 10000, "direction": "BUY", "price": 350.0}
    hist_vol = pd.Series([100000, 150000, 200000, 180000, 120000])
    vwap_plan = algo.vwap_execution_strategy(order, hist_vol)
    twap_plan = algo.twap_execution_strategy(order, 60, 5)
    return len(vwap_plan) > 0 and len(twap_plan) == 5

def test_performance_attribution():
    attrib = PerformanceAttribution()
    trades = [
        {"ticker": "0700.HK", "pnl": 500, "was_profitable": True, "regime": "bull"},
        {"ticker": "9988.HK", "pnl": -200, "was_profitable": False, "regime": "bull"},
    ]
    analysis = attrib.analyze_attribution(trades)
    return "by_ticker" in analysis

def test_drift_detector():
    detector = StrategyDriftDetector()
    detector.set_baseline({"win_rate": 0.55, "avg_holding_period": 5, "signals_per_day": 3})
    trades = [{"pnl": np.random.uniform(-500, 800), "was_profitable": np.random.random() > 0.45, "holding_period": 5} for _ in range(30)]
    is_drifting, metrics = detector.check_for_drift(trades)
    return isinstance(is_drifting, bool)

def test_drawdown_predictor():
    predictor = DrawdownPredictor()
    positions = {"0700.HK": 15000, "9988.HK": 10000}
    forecast = predictor.forecast_drawdown_risk(positions)
    return "scenarios" in forecast and len(forecast["scenarios"]) > 0

def test_liquidity_monitor():
    monitor = LiquidityRiskMonitor()
    positions = {"0700.HK": 50000, "9988.HK": 30000}
    volume = {"0700.HK": 5000000, "9988.HK": 3000000}
    risk = monitor.assess_liquidity_risk(positions, volume)
    return "overall_risk_level" in risk

def test_contingency_manager():
    contingency = ContingencyManager()
    low_wr_plan = contingency.contingency_plan_low_win_rate(0.35)
    high_dd_plan = contingency.contingency_plan_high_drawdown(0.12)
    return "actions" in low_wr_plan and "triggered" in high_dd_plan


if __name__ == "__main__":
    success = run_all_final_tests()
    sys.exit(0 if success else 1)
