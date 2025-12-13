"""
Test Runner for Fix 16-19 Tests
===============================

Run all tests for Fixes 16-19:
- Fix 16: China Stock Blocking
- Fix 17: JPY SELL Adaptive Blocking
- Fix 18: Crude Oil Adaptive Blocking
- Fix 19: Market Regime Detection

Usage:
    python run_all_tests.py           # Run all tests
    python run_all_tests.py -v        # Verbose mode
    python run_all_tests.py --quick   # Quick smoke tests only
    python run_all_tests.py --fix 17  # Run only Fix 17 tests

Author: Claude Code
Last Updated: 2025-12-03
"""

import unittest
import sys
import os
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_tests(fix_number=None, pattern=None):
    """Discover tests based on criteria."""
    loader = unittest.TestLoader()
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    if fix_number:
        # Load specific fix tests
        pattern = f'test_fix{fix_number}*.py'
    elif pattern:
        pattern = pattern
    else:
        # Load all fix 16-19 tests
        suite = unittest.TestSuite()
        patterns = [
            'test_fix16*.py',
            'test_fix17*.py',
            'test_fix18*.py',
            'test_fix19*.py',
            'test_integration*.py',
        ]
        for p in patterns:
            discovered = loader.discover(tests_dir, pattern=p)
            suite.addTests(discovered)
        return suite

    return loader.discover(tests_dir, pattern=pattern)


def run_quick_tests():
    """Run quick smoke tests only."""
    print("=" * 70)
    print("QUICK SMOKE TESTS FOR FIX 16-19")
    print("=" * 70)
    print()

    results = []

    # Test 1: Import all modules
    print("[1] Testing module imports...")
    try:
        from src.risk_management.market_regime_detector import MarketRegimeDetector, MarketRegime
        from src.risk_management.adaptive_blocker import AdaptiveBlocker, BlockingLevel
        from src.risk_management.position_adjuster import PositionAdjuster
        print("    [OK] All modules imported successfully")
        results.append(('Module Imports', True, None))
    except ImportError as e:
        print(f"    [FAIL] Import error: {e}")
        results.append(('Module Imports', False, str(e)))
        return results

    # Test 2: Create instances
    print("[2] Testing instance creation...")
    try:
        detector = MarketRegimeDetector()
        blocker = AdaptiveBlocker(detector)
        adjuster = PositionAdjuster(regime_detector=detector)
        print("    [OK] All instances created")
        results.append(('Instance Creation', True, None))
    except Exception as e:
        print(f"    [FAIL] Creation error: {e}")
        results.append(('Instance Creation', False, str(e)))
        return results

    # Test 3: Basic regime detection
    print("[3] Testing regime detection...")
    try:
        from src.risk_management.market_regime_detector import create_indicators_from_data

        # Bull market
        bull = create_indicators_from_data(vix=14, spy_return_20d=0.08, spy_above_200ma=True)
        bull_detection = detector.detect_regime(bull)
        assert bull_detection.primary_regime == MarketRegime.BULL, f"Expected BULL, got {bull_detection.primary_regime}"

        # Bear market
        bear = create_indicators_from_data(vix=30, spy_return_20d=-0.10, spy_above_200ma=False, gold_return_20d=0.05)
        bear_detection = detector.detect_regime(bear)
        assert bear_detection.primary_regime == MarketRegime.BEAR, f"Expected BEAR, got {bear_detection.primary_regime}"

        print("    [OK] Regime detection working")
        results.append(('Regime Detection', True, None))
    except Exception as e:
        print(f"    [FAIL] Detection error: {e}")
        results.append(('Regime Detection', False, str(e)))

    # Test 4: JPY SELL blocking (Fix 17)
    print("[4] Testing JPY SELL adaptive blocking...")
    try:
        # In bull market, JPY SELL should be reduced
        bull_result = blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=bull_detection,
        )
        assert not bull_result.blocked, "JPY SELL should NOT be blocked (adaptive)"
        assert bull_result.position_reduction < 0.5, f"Position should be reduced, got {bull_result.position_reduction}"

        # In bear market, JPY SELL should be favorable
        bear_result = blocker.evaluate_signal(
            ticker='USDJPY=X',
            signal_type='SELL',
            asset_type='jpy_pair',
            detection=bear_detection,
        )
        assert not bear_result.blocked, "JPY SELL should not be blocked"
        assert bear_result.position_reduction >= 0.8, f"Position should be high, got {bear_result.position_reduction}"

        print("    [OK] JPY SELL adaptive blocking working")
        results.append(('JPY SELL Blocking', True, None))
    except Exception as e:
        print(f"    [FAIL] JPY blocking error: {e}")
        results.append(('JPY SELL Blocking', False, str(e)))

    # Test 5: Crude Oil blocking (Fix 18)
    print("[5] Testing Crude Oil adaptive blocking...")
    try:
        # Crude BUY in bull should be favorable
        crude_buy_bull = blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='BUY',
            asset_type='crude_oil',
            detection=bull_detection,
        )
        assert not crude_buy_bull.blocked, "Crude BUY should not be blocked"
        assert crude_buy_bull.position_reduction >= 0.8, f"Position should be high, got {crude_buy_bull.position_reduction}"

        # Crude SELL in bear should be favorable
        crude_sell_bear = blocker.evaluate_signal(
            ticker='CL=F',
            signal_type='SELL',
            asset_type='crude_oil',
            detection=bear_detection,
        )
        assert not crude_sell_bear.blocked, "Crude SELL should not be blocked"
        assert crude_sell_bear.position_reduction >= 0.8, f"Position should be high, got {crude_sell_bear.position_reduction}"

        print("    [OK] Crude Oil adaptive blocking working")
        results.append(('Crude Oil Blocking', True, None))
    except Exception as e:
        print(f"    [FAIL] Crude blocking error: {e}")
        results.append(('Crude Oil Blocking', False, str(e)))

    # Test 6: Position adjustment (Fix 19)
    print("[6] Testing position adjustment...")
    try:
        result = adjuster.adjust_position(
            ticker='AAPL',
            signal_type='BUY',
            asset_type='stock',
            base_position=0.10,
            detection=bull_detection,
        )
        assert result.adjusted_position > 0, "Adjusted position should be > 0"
        assert result.adjusted_position <= 0.15, f"Position should be clamped, got {result.adjusted_position}"

        print("    [OK] Position adjustment working")
        results.append(('Position Adjustment', True, None))
    except Exception as e:
        print(f"    [FAIL] Position error: {e}")
        results.append(('Position Adjustment', False, str(e)))

    return results


def print_summary(results):
    """Print test summary."""
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status)
    failed = sum(1 for _, status, _ in results if not status)
    total = len(results)

    for name, status, error in results:
        status_str = "[OK]" if status else "[FAIL]"
        print(f"  {status_str} {name}")
        if error:
            print(f"        Error: {error}")

    print()
    print("-" * 70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)

    return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Fix 16-19 Tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke tests only')
    parser.add_argument('--fix', type=int, choices=[16, 17, 18, 19], help='Run only specific fix tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')

    args = parser.parse_args()

    print("=" * 70)
    print("FIX 16-19 TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    start_time = time.time()

    if args.quick:
        # Run quick smoke tests
        results = run_quick_tests()
        success = print_summary(results)
    else:
        # Run full unittest suite
        verbosity = 2 if args.verbose else 1

        if args.fix:
            print(f"Running Fix {args.fix} tests only...")
            suite = discover_tests(fix_number=args.fix)
        elif args.integration:
            print("Running integration tests only...")
            suite = discover_tests(pattern='test_integration*.py')
        else:
            print("Running all Fix 16-19 tests...")
            suite = discover_tests()

        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        success = result.wasSuccessful()

    elapsed = time.time() - start_time
    print()
    print(f"Completed in {elapsed:.2f} seconds")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
