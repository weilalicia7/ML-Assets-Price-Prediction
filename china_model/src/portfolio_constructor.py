"""
Portfolio Construction System for China Models

Builds diversified portfolios from robust performers with:
    - Risk-based position sizing (inverse volatility)
    - Correlation-aware diversification
    - Sector/market exposure limits
    - Signal strength weighting
    - Daily rebalancing signals
    - Phase 1-6 portfolio optimization calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent path to import Phase 1-6 calculations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from model_factory import ModelFactory, add_comprehensive_features
from china_stock_screener import detect_market_type

# Import Phase 1-6 calculation utilities
try:
    from src.utils.calculation_utils import (
        # Phase 1: Core Features & Trading
        calculate_volatility_scaling,
        correlation_aware_sizing,
        risk_parity_allocation,
        calculate_turnover,
        # Phase 2: Asset Class Ensembles
        safe_data_extraction,
        momentum_signal,
        volatility_signal,
        mean_reversion_signal,
        combine_signals,
        # Phase 5: Dynamic Weighting & Bayesian
        decay_weighted_sharpe,
        calmar_ratio,
        composite_performance_score,
        BayesianSignalUpdater,
        # Phase 6: Portfolio Optimization
        risk_contribution_analysis,
        expected_shortfall,
        _calculate_parametric_es,
        regime_aware_risk_budget,
        portfolio_optimization,
        stress_test_portfolio,
    )
    PHASE_1_6_AVAILABLE = True
    print("[INFO] Phase 1-6 calculations loaded successfully")
except ImportError as e:
    print(f"[WARNING] Phase 1-6 calculations not available: {e}")
    PHASE_1_6_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Portfolio constraints
MAX_POSITION_SIZE = 0.20       # Max 20% per position
MIN_POSITION_SIZE = 0.02       # Min 2% per position
MAX_SECTOR_EXPOSURE = 0.40    # Max 40% per sector
MAX_CORRELATION = 0.70        # Avoid highly correlated pairs
CASH_BUFFER = 0.05            # Keep 5% in cash

# Risk parameters
TARGET_VOLATILITY = 0.15      # 15% annualized target volatility
LOOKBACK_VOL = 60             # 60 days for volatility calculation
REBALANCE_THRESHOLD = 0.05   # Rebalance if drift > 5%

# China-specific Phase 6 stress scenarios
CHINA_STRESS_SCENARIOS = {
    '2015_china_crash': {'vol_mult': 3.5, 'corr_boost': 0.5},
    '2020_covid': {'vol_mult': 2.5, 'corr_boost': 0.3},
    'hk_political': {'vol_mult': 2.0, 'corr_boost': 0.25},
    'cny_devaluation': {'vol_mult': 2.2, 'corr_boost': 0.3},
    'normal': {'vol_mult': 1.0, 'corr_boost': 0.0}
}

# China-specific regime multipliers
CHINA_REGIME_MULTIPLIERS = {
    'crisis': 0.4,      # More defensive in crisis
    'high_vol': 0.65,
    'normal': 1.0,
    'low_vol': 1.1
}

# Sector mappings (expanded for new performers)
SECTOR_MAP = {
    # Tech
    '9999.HK': 'Tech',       # NetEase
    '9888.HK': 'Tech',       # Baidu
    '0981.HK': 'Tech',       # SMIC

    # Materials/Mining
    '3993.HK': 'Materials',  # CMOC
    '2600.HK': 'Materials',  # Aluminum Corp
    '0358.HK': 'Materials',  # Jiangxi Copper - NEW

    # Healthcare/Biotech
    '6160.HK': 'Healthcare', # Beigene
    '1093.HK': 'Healthcare', # CSPC Pharma
    '1801.HK': 'Healthcare', # Innovent Biologics - NEW

    # Consumer/Auto
    '0175.HK': 'Consumer',   # Geely

    # Financials
    '1299.HK': 'Financials', # AIA
    '2628.HK': 'Financials', # China Life

    # Industrials
    '300274.SZ': 'Industrials', # Sungrow Power
    '1088.HK': 'Energy',     # China Shenhua

    # ETF
    '2828.HK': 'ETF',        # HS China ETF
}


# ============================================================================
# PORTFOLIO CONSTRUCTOR
# ============================================================================

class PortfolioConstructor:
    """
    Constructs and manages portfolios from model signals
    """

    def __init__(self, model_factory=None):
        """
        Args:
            model_factory: Initialized ModelFactory with loaded models
        """
        self.factory = model_factory
        self.portfolio = {}
        self.signals = {}
        self.risk_metrics = {}

    def load_models(self, models_path=None):
        """Load models from disk"""
        if self.factory is None:
            self.factory = ModelFactory()
        self.factory.load_models(models_path)

    def calculate_volatility(self, symbols, lookback=LOOKBACK_VOL):
        """Calculate realized volatility for symbols"""
        volatilities = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback + 30)

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if len(df) >= lookback:
                    returns = df['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)  # Annualized
                    volatilities[symbol] = float(vol)
                else:
                    volatilities[symbol] = 0.30  # Default 30%
            except:
                volatilities[symbol] = 0.30

        return volatilities

    def calculate_correlations(self, symbols, lookback=LOOKBACK_VOL):
        """Calculate correlation matrix between symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback + 30)

        # Download price data
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if len(df) > 0:
                    prices[symbol] = df['Close']
            except:
                pass

        if len(prices) < 2:
            return pd.DataFrame()

        # Create returns DataFrame
        returns_df = pd.DataFrame(prices).pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        return corr_matrix

    def get_signals(self, symbols):
        """Get current signals for all symbols"""
        signals = {}

        for symbol in symbols:
            if symbol in self.factory.models:
                pred = self.factory.predict(symbol)
                if pred:
                    signals[symbol] = {
                        'signal': pred['signal'],
                        'probability': pred['probability'],
                        'strategy': pred['strategy'],
                        'confidence': 1 - pred['confidence_std'],  # Higher = more confident
                    }

        self.signals = signals
        return signals

    def calculate_position_sizes(self, symbols, total_capital=1.0, include_hold=True, target_cash=0.40):
        """
        Calculate position sizes using Phase 1-6 enhanced calculations.

        Uses:
        - Phase 1: Volatility scaling and correlation-aware sizing
        - Phase 6: Risk parity allocation

        Args:
            symbols: List of symbols to include
            total_capital: Total capital (1.0 = 100%)
            include_hold: Include HOLD signals with reduced weight
            target_cash: Target cash allocation (default 40%)

        Returns:
            dict of symbol -> position size (0-1)
        """
        # Get volatilities
        volatilities = self.calculate_volatility(symbols)

        # Get signals
        signals = self.get_signals(symbols)

        # Filter for investable signals
        buy_symbols = [s for s in symbols if s in signals and signals[s]['signal'] == 'BUY']
        hold_symbols = [s for s in symbols if s in signals and signals[s]['signal'] == 'HOLD'] if include_hold else []

        if not buy_symbols and not hold_symbols:
            print("No BUY/HOLD signals - returning all cash position")
            return {'CASH': 1.0}

        # Combine all investable symbols
        all_investable = buy_symbols + hold_symbols

        # Use Phase 6 risk parity if available
        if PHASE_1_6_AVAILABLE and len(all_investable) >= 2:
            try:
                # Build covariance matrix from volatilities and correlations
                corr_matrix = self.calculate_correlations(all_investable)
                vol_array = np.array([volatilities.get(s, 0.30) for s in all_investable])

                if len(corr_matrix) >= 2:
                    # Convert correlation to covariance
                    cov_matrix = np.outer(vol_array, vol_array) * corr_matrix.values

                    # Use Phase 6 risk parity allocation
                    rp_weights = risk_parity_allocation(cov_matrix)
                    weights = {s: float(rp_weights[i]) for i, s in enumerate(all_investable)}
                    print(f"  [Phase 6] Using risk parity allocation")
                else:
                    # Fallback to inverse volatility
                    inv_vol = {s: 1.0 / volatilities[s] for s in all_investable}
                    total_inv_vol = sum(inv_vol.values())
                    weights = {s: inv_vol[s] / total_inv_vol for s in all_investable}
            except Exception as e:
                print(f"  [WARNING] Phase 6 risk parity failed: {e}, using inverse volatility")
                inv_vol = {s: 1.0 / volatilities[s] for s in all_investable}
                total_inv_vol = sum(inv_vol.values())
                weights = {s: inv_vol[s] / total_inv_vol for s in all_investable}
        else:
            # Inverse volatility weights (fallback)
            inv_vol = {s: 1.0 / volatilities[s] for s in all_investable}
            total_inv_vol = sum(inv_vol.values())
            weights = {s: inv_vol[s] / total_inv_vol for s in all_investable}

        # Adjust by signal strength (probability) and signal type
        for s in all_investable:
            signal_strength = signals[s]['probability']
            # BUY signals get full weight, HOLD signals get 60% weight
            signal_multiplier = 1.0 if signals[s]['signal'] == 'BUY' else 0.6
            weights[s] *= signal_strength * signal_multiplier

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        # Apply Phase 1 correlation-aware sizing
        if PHASE_1_6_AVAILABLE:
            for s in list(weights.keys()):
                sector = SECTOR_MAP.get(s, 'Other')
                sector_value = sum(weights.get(sym, 0) for sym in weights if SECTOR_MAP.get(sym, 'Other') == sector)
                num_sector = sum(1 for sym in weights if SECTOR_MAP.get(sym, 'Other') == sector)

                try:
                    adjusted = correlation_aware_sizing(
                        base_size=weights[s],
                        sector_value=sector_value,
                        total_portfolio_value=1.0,
                        num_same_sector_positions=num_sector
                    )
                    weights[s] = adjusted
                except:
                    pass

        # Apply constraints with target cash allocation
        available_capital = total_capital * (1 - target_cash)

        for s in weights:
            weights[s] = weights[s] * available_capital
            weights[s] = min(weights[s], MAX_POSITION_SIZE)
            weights[s] = max(weights[s], MIN_POSITION_SIZE)

        # Re-normalize after constraints
        total_weight = sum(weights.values())
        if total_weight > available_capital:
            scale = available_capital / total_weight
            weights = {s: w * scale for s, w in weights.items()}

        # Check sector exposure
        sector_exposure = {}
        for s, w in weights.items():
            sector = SECTOR_MAP.get(s, 'Other')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + w

        for sector, exposure in sector_exposure.items():
            if exposure > MAX_SECTOR_EXPOSURE:
                scale = MAX_SECTOR_EXPOSURE / exposure
                for s in weights:
                    if SECTOR_MAP.get(s, 'Other') == sector:
                        weights[s] *= scale

        # Add cash position
        total_invested = sum(weights.values())
        weights['CASH'] = 1 - total_invested

        return weights

    def construct_portfolio(self, symbols, total_capital=100000, verbose=True):
        """
        Construct optimal portfolio from symbols

        Args:
            symbols: List of candidate symbols
            total_capital: Total capital in currency units
            verbose: Print details

        Returns:
            dict with portfolio allocation and metrics
        """
        if verbose:
            print("=" * 70)
            print("PORTFOLIO CONSTRUCTION")
            print("=" * 70)
            print(f"Candidates: {len(symbols)}")
            print(f"Total Capital: ${total_capital:,.0f}")
            print()

        # Get signals
        signals = self.get_signals(symbols)

        if verbose:
            print("Current Signals:")
            print("-" * 50)
            for s, sig in signals.items():
                print(f"  {s:<12} {sig['signal']:<6} prob={sig['probability']*100:.1f}%")
            print()

        # Calculate position sizes
        weights = self.calculate_position_sizes(symbols)

        # Get volatilities for risk calculation
        volatilities = self.calculate_volatility([s for s in weights if s != 'CASH'])

        # Calculate portfolio metrics
        portfolio_vol = 0
        for s, w in weights.items():
            if s != 'CASH' and s in volatilities:
                portfolio_vol += (w ** 2) * (volatilities[s] ** 2)
        portfolio_vol = np.sqrt(portfolio_vol)

        # Build portfolio
        portfolio = {
            'timestamp': datetime.now().isoformat(),
            'total_capital': total_capital,
            'positions': {},
            'metrics': {
                'portfolio_volatility': float(portfolio_vol),
                'num_positions': len([w for w in weights if w != 'CASH' and weights[w] > 0]),
                'total_invested': float(1 - weights.get('CASH', 0)),
            }
        }

        # Create positions
        for symbol, weight in weights.items():
            if weight > 0:
                allocation = weight * total_capital

                if symbol == 'CASH':
                    portfolio['positions']['CASH'] = {
                        'weight': float(weight),
                        'allocation': float(allocation),
                    }
                else:
                    signal_data = signals.get(symbol, {})
                    portfolio['positions'][symbol] = {
                        'weight': float(weight),
                        'allocation': float(allocation),
                        'signal': signal_data.get('signal', 'N/A'),
                        'probability': float(signal_data.get('probability', 0)),
                        'strategy': signal_data.get('strategy', 'N/A'),
                        'sector': SECTOR_MAP.get(symbol, 'Other'),
                        'volatility': float(volatilities.get(symbol, 0)),
                    }

        self.portfolio = portfolio

        if verbose:
            print("Portfolio Allocation:")
            print("-" * 70)
            print(f"{'Symbol':<12} {'Sector':<12} {'Weight':>8} {'Allocation':>12} {'Signal':<6}")
            print("-" * 70)

            for symbol, pos in sorted(portfolio['positions'].items(), key=lambda x: x[1]['weight'], reverse=True):
                if symbol == 'CASH':
                    print(f"{'CASH':<12} {'-':<12} {pos['weight']*100:>7.1f}% ${pos['allocation']:>10,.0f} {'-':<6}")
                else:
                    print(f"{symbol:<12} {pos['sector']:<12} {pos['weight']*100:>7.1f}% ${pos['allocation']:>10,.0f} {pos['signal']:<6}")

            print("-" * 70)
            print(f"Portfolio Volatility: {portfolio['metrics']['portfolio_volatility']*100:.1f}% (annualized)")
            print(f"Number of Positions: {portfolio['metrics']['num_positions']}")
            print(f"Total Invested: {portfolio['metrics']['total_invested']*100:.1f}%")

        return portfolio

    def generate_trades(self, current_holdings=None, verbose=True):
        """
        Generate trades to rebalance to target portfolio

        Args:
            current_holdings: dict of symbol -> current value
            verbose: Print details

        Returns:
            list of trade orders
        """
        if not self.portfolio:
            raise ValueError("No portfolio constructed. Call construct_portfolio first.")

        if current_holdings is None:
            current_holdings = {}

        total_capital = self.portfolio['total_capital']
        target_positions = self.portfolio['positions']

        trades = []

        if verbose:
            print("\n" + "=" * 70)
            print("REBALANCING TRADES")
            print("=" * 70)

        # Calculate required trades
        all_symbols = set(list(target_positions.keys()) + list(current_holdings.keys()))

        for symbol in all_symbols:
            current_value = current_holdings.get(symbol, 0)
            target_value = target_positions.get(symbol, {}).get('allocation', 0)

            diff = target_value - current_value
            diff_pct = abs(diff) / total_capital

            if diff_pct > REBALANCE_THRESHOLD:
                if diff > 0:
                    action = 'BUY'
                else:
                    action = 'SELL'
                    diff = abs(diff)

                trades.append({
                    'symbol': symbol,
                    'action': action,
                    'amount': float(diff),
                    'target_weight': float(target_positions.get(symbol, {}).get('weight', 0)),
                })

        if verbose:
            if trades:
                print(f"{'Symbol':<12} {'Action':<6} {'Amount':>12}")
                print("-" * 35)
                for t in trades:
                    print(f"{t['symbol']:<12} {t['action']:<6} ${t['amount']:>10,.0f}")
            else:
                print("No rebalancing required")

        return trades

    def save_portfolio(self, filepath=None):
        """Save portfolio to JSON"""
        if filepath is None:
            save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, 'current_portfolio.json')

        with open(filepath, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

        print(f"\nPortfolio saved to: {filepath}")

    def daily_update(self, verbose=True):
        """
        Run daily portfolio update

        Returns:
            dict with new signals and recommended trades
        """
        if verbose:
            print("=" * 70)
            print(f"DAILY PORTFOLIO UPDATE - {datetime.now().strftime('%Y-%m-%d')}")
            print("=" * 70)

        if not self.factory or not self.factory.models:
            self.load_models()

        symbols = list(self.factory.models.keys())

        # Get fresh signals
        signals = self.get_signals(symbols)

        # Reconstruct portfolio
        portfolio = self.construct_portfolio(symbols, verbose=verbose)

        # Save updated portfolio
        self.save_portfolio()

        return {
            'date': datetime.now().isoformat(),
            'signals': signals,
            'portfolio': portfolio,
        }


# ============================================================================
# RISK ANALYSIS
# ============================================================================

class RiskAnalyzer:
    """
    Analyze portfolio risk metrics using Phase 6 calculations.

    Includes:
    - VaR and Expected Shortfall (CVaR)
    - China-specific stress testing
    - Risk contribution analysis
    """

    def __init__(self, portfolio_constructor):
        self.constructor = portfolio_constructor

    def calculate_var(self, confidence=0.95, horizon_days=1):
        """Calculate Value at Risk"""
        if not self.constructor.portfolio:
            return None

        portfolio = self.constructor.portfolio
        positions = {s: p for s, p in portfolio['positions'].items() if s != 'CASH'}

        if not positions:
            return 0

        # Get returns
        end_date = datetime.now()
        start_date = end_date - timedelta(days=252)

        portfolio_returns = []

        for symbol, pos in positions.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                returns = df['Close'].pct_change().dropna()
                weighted_returns = returns * pos['weight']
                portfolio_returns.append(weighted_returns)
            except:
                pass

        if not portfolio_returns:
            return None

        # Combine returns
        combined = pd.concat(portfolio_returns, axis=1).sum(axis=1)

        # Calculate VaR
        var = np.percentile(combined, (1 - confidence) * 100) * np.sqrt(horizon_days)

        return float(var)

    def calculate_max_drawdown(self, lookback_days=252):
        """Calculate maximum drawdown"""
        if not self.constructor.portfolio:
            return None

        portfolio = self.constructor.portfolio
        positions = {s: p for s, p in portfolio['positions'].items() if s != 'CASH'}

        if not positions:
            return 0

        # Get cumulative returns
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        portfolio_value = None

        for symbol, pos in positions.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                price_series = df['Close'] / df['Close'].iloc[0] * pos['weight']

                if portfolio_value is None:
                    portfolio_value = price_series
                else:
                    portfolio_value = portfolio_value + price_series
            except:
                pass

        if portfolio_value is None or len(portfolio_value) == 0:
            return None

        # Calculate drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return float(max_drawdown)

    def calculate_expected_shortfall(self, confidence=0.95):
        """
        Calculate Expected Shortfall (CVaR) using Phase 6 calculations.

        Returns:
            Expected Shortfall value or None
        """
        if not PHASE_1_6_AVAILABLE:
            return None

        if not self.constructor.portfolio:
            return None

        portfolio = self.constructor.portfolio
        positions = {s: p for s, p in portfolio['positions'].items() if s != 'CASH'}

        if not positions:
            return 0

        try:
            # Get returns
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)

            returns_list = []
            weights_list = []

            for symbol, pos in positions.items():
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    returns = df['Close'].pct_change().dropna().values
                    if len(returns) > 0:
                        returns_list.append(returns[:min(len(returns), 250)])
                        weights_list.append(pos['weight'])
                except:
                    pass

            if not returns_list:
                return None

            # Align returns and calculate ES
            min_len = min(len(r) for r in returns_list)
            returns_matrix = np.array([r[:min_len] for r in returns_list]).T
            weights = np.array(weights_list)
            weights = weights / weights.sum()

            es = expected_shortfall(returns_matrix, weights, confidence)
            return float(es)

        except Exception as e:
            print(f"  [WARNING] ES calculation failed: {e}")
            return None

    def run_stress_tests(self, verbose=True):
        """
        Run China-specific stress tests using Phase 6 calculations.

        Returns:
            Dict with stress test results for each scenario
        """
        if not PHASE_1_6_AVAILABLE:
            if verbose:
                print("  [WARNING] Phase 6 not available for stress testing")
            return None

        if not self.constructor.portfolio:
            return None

        portfolio = self.constructor.portfolio
        positions = {s: p for s, p in portfolio['positions'].items() if s != 'CASH'}

        if not positions:
            return None

        try:
            # Get covariance matrix
            symbols = list(positions.keys())
            volatilities = self.constructor.calculate_volatility(symbols)
            corr_matrix = self.constructor.calculate_correlations(symbols)

            if len(corr_matrix) < 2:
                return None

            vol_array = np.array([volatilities.get(s, 0.30) for s in symbols])
            cov_matrix = np.outer(vol_array, vol_array) * corr_matrix.values
            weights = np.array([positions[s]['weight'] for s in symbols])
            weights = weights / weights.sum()

            # Run stress tests with China-specific scenarios
            results = stress_test_portfolio(
                weights=weights,
                covariance_matrix=cov_matrix,
                scenarios=CHINA_STRESS_SCENARIOS
            )

            if verbose:
                print("\n  China Stress Test Results:")
                print("  " + "-" * 50)
                for scenario, data in results.items():
                    print(f"    {scenario}:")
                    print(f"      Portfolio Vol: {data['portfolio_volatility']*100:.1f}%")
                    print(f"      Expected Shortfall: {data['expected_shortfall']*100:.2f}%")

            return results

        except Exception as e:
            if verbose:
                print(f"  [WARNING] Stress testing failed: {e}")
            return None

    def generate_risk_report(self, verbose=True):
        """Generate comprehensive risk report with Phase 6 enhancements"""
        if verbose:
            print("\n" + "=" * 70)
            print("RISK ANALYSIS (Phase 6 Enhanced)")
            print("=" * 70)

        var_95 = self.calculate_var(0.95, 1)
        var_99 = self.calculate_var(0.99, 1)
        max_dd = self.calculate_max_drawdown()

        # Phase 6 additions
        es_95 = self.calculate_expected_shortfall(0.95)
        stress_results = self.run_stress_tests(verbose=False)

        report = {
            'var_95_1day': var_95,
            'var_99_1day': var_99,
            'max_drawdown': max_dd,
            'expected_shortfall_95': es_95,
            'stress_tests': stress_results,
            'phase_6_enabled': PHASE_1_6_AVAILABLE,
            'generated_at': datetime.now().isoformat(),
        }

        if verbose:
            print(f"Value at Risk (95%, 1-day): {var_95*100:.2f}%" if var_95 else "VaR: N/A")
            print(f"Value at Risk (99%, 1-day): {var_99*100:.2f}%" if var_99 else "VaR: N/A")
            print(f"Expected Shortfall (95%): {es_95*100:.2f}%" if es_95 else "ES: N/A")
            print(f"Max Drawdown (1-year): {max_dd*100:.2f}%" if max_dd else "Max DD: N/A")

            if stress_results:
                print("\nStress Test Summary:")
                worst_scenario = max(stress_results.items(),
                                    key=lambda x: abs(x[1]['expected_shortfall']))
                print(f"  Worst case ({worst_scenario[0]}): "
                      f"ES = {worst_scenario[1]['expected_shortfall']*100:.2f}%")

        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run portfolio construction pipeline"""
    print("=" * 70)
    print("CHINA PORTFOLIO CONSTRUCTION SYSTEM")
    print("=" * 70)
    print()

    # Load models
    factory = ModelFactory()
    models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'production_models.pkl')

    if os.path.exists(models_path):
        factory.load_models(models_path)
    else:
        print("No models found. Running model factory first...")
        from model_factory import main as build_models
        build_models()
        factory.load_models(models_path)

    # Get symbols from loaded models
    symbols = list(factory.models.keys())
    print(f"\nLoaded {len(symbols)} models: {symbols}")

    # Construct portfolio
    constructor = PortfolioConstructor(model_factory=factory)
    portfolio = constructor.construct_portfolio(
        symbols=symbols,
        total_capital=100000,
        verbose=True
    )

    # Generate trades (assuming starting from scratch)
    trades = constructor.generate_trades(current_holdings={}, verbose=True)

    # Risk analysis
    risk_analyzer = RiskAnalyzer(constructor)
    risk_report = risk_analyzer.generate_risk_report(verbose=True)

    # Save portfolio
    constructor.save_portfolio()

    print("\n" + "=" * 70)
    print("PORTFOLIO CONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
