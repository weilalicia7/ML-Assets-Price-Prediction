"""
Test script for all 20 advanced features (15 core + 5 stress protection).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.production_advanced import *
import numpy as np
import pandas as pd

def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST OF ALL 20 ADVANCED FEATURES")
    print("(15 Core + 5 Stress Protection)")
    print("=" * 70)

    passed = 0
    failed = 0

    # Test Feature 1: AdaptiveFeatureSelector
    print("\n[1] AdaptiveFeatureSelector...")
    try:
        selector = AdaptiveFeatureSelector()
        np.random.seed(42)
        stock_data = pd.DataFrame({
            "momentum_5d": np.random.randn(100),
            "rsi_14": np.random.randn(100),
            "macd_signal": np.random.randn(100),
            "volume_ratio": np.random.randn(100),
            "volatility_20d": np.random.randn(100)
        })
        target_returns = pd.Series(np.random.randn(100))
        features = selector.select_features(stock_data, target_returns)
        assert len(features) > 0
        print(f"   PASS - Selected {len(features)} features")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 2: MetaLearner
    print("\n[2] MetaLearner...")
    try:
        meta = MetaLearner()
        model = meta.select_best_model("high_volatility", {"volatility": 0.3})
        assert model in ["LSTM", "LightGBM", "XGBoost", "CatBoost", "HybridEnsemble", "Conservative", "Momentum"]
        print(f"   PASS - Selected model: {model}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 3: CorrelationAwarePositionSizer
    print("\n[3] CorrelationAwarePositionSizer...")
    try:
        sizer = CorrelationAwarePositionSizer()
        new_trade = {"ticker": "0700.HK", "signal": 0.8}
        existing = [{"ticker": "9988.HK", "value": 10000}]
        adjusted = sizer.correlation_adjusted_sizing(new_trade, existing, base_size=0.1)
        assert 0 < adjusted <= 0.1
        print(f"   PASS - Adjusted size: {adjusted:.4f}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 4: VolatilityScaler
    print("\n[4] VolatilityScaler...")
    try:
        scaler = VolatilityScaler()
        returns = pd.Series(np.random.randn(100) * 0.02)
        adj = scaler.get_volatility_adjustment(returns)
        assert 0.5 <= adj <= 1.0
        print(f"   PASS - Volatility adjustment: {adj:.4f}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 5: RiskParityAllocator
    print("\n[5] RiskParityAllocator...")
    try:
        allocator = RiskParityAllocator()
        predictions = {
            "0700.HK": {"return": 0.05, "volatility": 0.20},
            "9988.HK": {"return": 0.03, "volatility": 0.25}
        }
        alloc = allocator.risk_parity_allocation(predictions, {"0700.HK": 0.5, "9988.HK": 0.5})
        assert abs(sum(alloc.values()) - 1.0) < 0.01
        print(f"   PASS - Allocation sums to 1.0")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 6: SmartRebalancer
    print("\n[6] SmartRebalancer...")
    try:
        rebalancer = SmartRebalancer()
        should, reason = rebalancer.should_rebalance(
            {"0700.HK": 0.15, "9988.HK": 0.10},
            {"0700.HK": 0.10, "9988.HK": 0.10},
            {"0700.HK": 0.02, "9988.HK": 0.01}
        )
        assert isinstance(should, bool)
        print(f"   PASS - Rebalance decision: {should}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 7: CrossAssetSignalEnhancer
    print("\n[7] CrossAssetSignalEnhancer...")
    try:
        enhancer = CrossAssetSignalEnhancer()
        enhanced = enhancer.enhance_signal(0.7, {"0700.HK": 0.8, "9988.HK": 0.6})
        assert 0 <= enhanced <= 1.0
        print(f"   PASS - Enhanced signal: {enhanced:.4f}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 8: RegimeSignalWeighter
    print("\n[8] RegimeSignalWeighter...")
    try:
        weighter = RegimeSignalWeighter()
        weighted = weighter.regime_aware_signal_weighting(0.7, "high_volatility")
        assert 0.3 <= weighted <= 0.95
        print(f"   PASS - Regime-weighted signal: {weighted:.4f}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 9: SmartOrderRouter
    print("\n[9] SmartOrderRouter...")
    try:
        router = SmartOrderRouter()
        order = {"ticker": "0700.HK", "shares": 1000, "direction": "BUY"}
        market_cond = {"spread": 0.002, "volume": 1000000, "volatility": 0.02}
        execution = router.optimize_execution(order, market_cond)
        assert "urgency" in execution and "algorithm" in execution
        print(f"   PASS - urgency={execution['urgency']}, algo={execution['algorithm']}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 10: ExecutionAlgorithms
    print("\n[10] ExecutionAlgorithms...")
    try:
        algo = ExecutionAlgorithms()
        order = {"ticker": "0700.HK", "quantity": 10000, "direction": "BUY", "price": 350.0}
        hist_vol = pd.Series([100000, 150000, 200000, 180000, 120000])
        vwap_plan = algo.vwap_execution_strategy(order, hist_vol)
        twap_plan = algo.twap_execution_strategy(order, 60, 5)
        assert len(vwap_plan) > 0 and len(twap_plan) == 5
        print(f"   PASS - VWAP slices: {len(vwap_plan)}, TWAP slices: {len(twap_plan)}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 11: PerformanceAttribution
    print("\n[11] PerformanceAttribution...")
    try:
        attrib = PerformanceAttribution()
        trades = [
            {"ticker": "0700.HK", "pnl": 500, "was_profitable": True, "regime": "bull"},
            {"ticker": "9988.HK", "pnl": -200, "was_profitable": False, "regime": "bull"},
            {"ticker": "0700.HK", "pnl": 300, "was_profitable": True, "regime": "bear"},
        ]
        analysis = attrib.analyze_attribution(trades)
        assert "by_ticker" in analysis
        print(f"   PASS - Attribution by ticker: {list(analysis['by_ticker'].keys())}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 12: StrategyDriftDetector
    print("\n[12] StrategyDriftDetector...")
    try:
        detector = StrategyDriftDetector()
        detector.set_baseline({"win_rate": 0.55, "avg_holding_period": 5, "signals_per_day": 3})
        trades = [{"pnl": np.random.uniform(-500, 800), "was_profitable": np.random.random() > 0.45, "holding_period": 5} for _ in range(30)]
        is_drifting, metrics = detector.check_for_drift(trades)
        assert isinstance(is_drifting, bool)
        print(f"   PASS - Drift detected: {is_drifting}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 13: DrawdownPredictor
    print("\n[13] DrawdownPredictor...")
    try:
        predictor = DrawdownPredictor()
        positions = {"0700.HK": 15000, "9988.HK": 10000}
        forecast = predictor.forecast_drawdown_risk(positions)
        assert "scenarios" in forecast and len(forecast["scenarios"]) > 0
        print(f"   PASS - Scenarios tested: {len(forecast['scenarios'])}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 14: LiquidityRiskMonitor
    print("\n[14] LiquidityRiskMonitor...")
    try:
        monitor = LiquidityRiskMonitor()
        positions = {"0700.HK": 50000, "9988.HK": 30000}
        volume = {"0700.HK": 5000000, "9988.HK": 3000000}
        risk = monitor.assess_liquidity_risk(positions, volume)
        assert "overall_risk_level" in risk
        print(f"   PASS - Liquidity risk level: {risk['overall_risk_level']}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 15: ContingencyManager
    print("\n[15] ContingencyManager...")
    try:
        contingency = ContingencyManager()
        low_wr_plan = contingency.contingency_plan_low_win_rate(0.35)
        high_dd_plan = contingency.contingency_plan_high_drawdown(0.12)
        assert "actions" in low_wr_plan and "triggered" in high_dd_plan
        print(f"   PASS - Low WR actions: {len(low_wr_plan['actions'])}, High DD triggered: {high_dd_plan['triggered']}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # =========================================================================
    # STRESS PROTECTION FEATURES (16-20)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STRESS PROTECTION FEATURES (16-20)")
    print("=" * 70)

    # Test Feature 16: StressScenarioProtection
    print("\n[16] StressScenarioProtection...")
    try:
        stress = StressScenarioProtection()
        # Test VIX levels
        normal = stress.update_vix(15)
        assert normal['stress_level'] == 'NORMAL'
        assert normal['position_multiplier'] == 1.0

        warning = stress.update_vix(27)
        assert warning['stress_level'] == 'WARNING'
        assert warning['position_multiplier'] == 0.50

        halt = stress.update_vix(40)
        assert halt['stress_level'] == 'HALT'
        assert stress.circuit_breaker_active == True

        print(f"   PASS - VIX levels: NORMAL=100%, WARNING=50%, HALT=0%")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 17: FlashCrashDetector
    print("\n[17] FlashCrashDetector...")
    try:
        detector = FlashCrashDetector()
        # Simulate flash crash: 10% drop with volume spike
        prices = pd.Series([100, 95, 92, 90, 88])  # 12% drop
        volumes = pd.Series([100000, 200000, 500000, 800000, 1000000])  # 10x spike
        result = detector.detect_flash_crash("0700.HK", prices, volumes, 0.005, 0.001)
        assert result['flash_crash'] == True
        assert 'response' in result
        print(f"   PASS - Flash crash detected: drop={result['price_drop']:.1%}, response={result['response']['action']}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 18: BlackSwanPreparer
    print("\n[18] BlackSwanPreparer...")
    try:
        preparer = BlackSwanPreparer()
        # Test black swan detection (15% drop + VIX > 40)
        result = preparer.detect_black_swan(market_return=-0.18, current_vix=45)
        assert result['black_swan'] == True
        assert 'response' in result
        assert result['response']['action'] == 'BLACK_SWAN_DEFENSE'

        # Test exposure score
        positions = {"0700.HK": 50000, "9988.HK": 30000}
        score = preparer.calculate_exposure_score(positions)
        assert 0 <= score <= 1.0

        print(f"   PASS - Black swan detected, defense activated, exposure score: {score:.2f}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 19: EmergencyLiquidation
    print("\n[19] EmergencyLiquidation...")
    try:
        liquidator = EmergencyLiquidation()
        positions = {
            "0700.HK": {"value": 50000, "quantity": 100},
            "9988.HK": {"value": 30000, "quantity": 200}
        }
        volumes = {"0700.HK": 5000000, "9988.HK": 3000000}
        plan = liquidator.create_liquidation_plan(positions, volumes, urgency='HIGH')
        assert 'orders' in plan
        assert len(plan['orders']) == 2
        assert plan['total_value'] == 80000

        # Test emergency close
        emergency = liquidator.execute_emergency_close({"ticker": "0700.HK", "quantity": 100, "value": 50000})
        assert emergency['action'] == 'EMERGENCY_SELL'
        assert emergency['order_type'] == 'MARKET'

        print(f"   PASS - Liquidation plan: {len(plan['orders'])} orders, total ${plan['total_value']:,}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    # Test Feature 20: StressHardenedTradingSystem
    print("\n[20] StressHardenedTradingSystem...")
    try:
        system = StressHardenedTradingSystem()

        # Test normal conditions
        status = system.update_market_conditions(vix=15, market_return=0.01)
        assert status['system_status'] == 'NORMAL'
        assert status['position_multiplier'] == 1.0

        # Test stressed conditions
        status = system.update_market_conditions(vix=32, market_return=-0.05)
        assert status['stress_level'] == 'DANGER'
        assert status['position_multiplier'] == 0.25

        # Test trade allowed check
        allowed, reason = system.check_trade_allowed()
        assert isinstance(allowed, bool)

        # Test position sizing
        adjusted = system.get_adjusted_position_size(0.10, portfolio_drawdown=0.05)
        assert adjusted < 0.10  # Should be reduced

        # Get status report
        report = system.get_status_report()
        assert 'system_status' in report
        assert 'stress_level' in report

        print(f"   PASS - System status: {report['system_status']}, stress: {report['stress_level']}")
        passed += 1
    except Exception as e:
        print(f"   FAIL - {e}")
        failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/20 FEATURES PASSED")
    print("=" * 70)

    if passed == 20:
        print("\nALL 20 FEATURES WORKING CORRECTLY!")
        print("Production system with stress protection ready for deployment.")
    else:
        print(f"\n{failed} FEATURES NEED ATTENTION")

    return passed == 20


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
