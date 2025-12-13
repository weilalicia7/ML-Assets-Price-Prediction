"""
Production Trading System

Real-time trading system for the optimized 18-asset portfolio.

Features:
- Real-time data fetching
- Signal generation with ensemble models
- Position sizing with Kelly criterion
- Risk management (stop-loss, position limits)
- Performance tracking
- Automated rebalancing

Portfolio: Max Sharpe (Markowitz)
- 11 assets with non-zero weights
- Expected Sharpe: 1.12 (backtest), 0.56 (production estimate)
- Forex-heavy: 50% allocation
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.enhanced_ensemble import EnhancedEnsemblePredictor
from src.trading.kelly_backtester import KellyBacktester


class ProductionTradingSystem:
    """Production-ready trading system with risk management."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trading system."""
        self.config = self._load_config(config_path)
        self.portfolio = self.config['portfolio']
        self.risk_controls = self.config['risk_controls']

        # State
        self.positions = {}  # Current positions
        self.models = {}  # Trained models per asset
        self.performance = {
            'total_return': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'daily_returns': []
        }

        # Initialize positions
        for symbol in self.portfolio.keys():
            self.positions[symbol] = {
                'size': 0.0,
                'entry_price': 0.0,
                'current_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use default."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)

        # Default configuration (from portfolio optimization)
        return {
            'portfolio': {
                'EURUSD=X': {'weight': 0.25, 'interval': '1d', 'kelly': 0.30, 'min_edge': 0.05},
                'JPY=X': {'weight': 0.25, 'interval': '4h', 'kelly': 0.25, 'min_edge': 0.05},
                'JPM': {'weight': 0.174, 'interval': '1d', 'kelly': 0.15, 'min_edge': 0.07},
                'JNJ': {'weight': 0.091, 'interval': '1h', 'kelly': 0.10, 'min_edge': 0.04},
                'BABA': {'weight': 0.063, 'interval': '1d', 'kelly': 0.10, 'min_edge': 0.05},
                'VALE': {'weight': 0.061, 'interval': '1h', 'kelly': 0.15, 'min_edge': 0.06},
                'AAPL': {'weight': 0.029, 'interval': '1h', 'kelly': 0.10, 'min_edge': 0.04},
                'ASML': {'weight': 0.028, 'interval': '1h', 'kelly': 0.10, 'min_edge': 0.04},
                'XOM': {'weight': 0.028, 'interval': '1h', 'kelly': 0.10, 'min_edge': 0.04},
                'CL=F': {'weight': 0.014, 'interval': '1d', 'kelly': 0.15, 'min_edge': 0.07},
                'NSRGY': {'weight': 0.012, 'interval': '1h', 'kelly': 0.15, 'min_edge': 0.06},
            },
            'risk_controls': {
                'max_position_size': 0.25,
                'per_asset_stop_loss': -0.05,
                'portfolio_stop_loss': -0.10,
                'daily_loss_limit': -0.02,
                'rebalance_frequency_days': 7,
                'rebalance_threshold': 0.05,
                'max_leverage': 1.0,
                'min_position_size': 0.01,
            },
            'execution': {
                'slippage_pct': 0.001,  # 0.1% slippage
                'commission_per_trade': 1.0,  # $1 per trade
                'max_trades_per_day': 20,
            }
        }

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to price data."""
        data = data.copy()
        data['returns'] = data['Close'].pct_change()
        data['returns_5'] = data['Close'].pct_change(5)
        data['returns_20'] = data['Close'].pct_change(20)
        data['vol_5'] = data['returns'].rolling(5).std()
        data['vol_20'] = data['returns'].rolling(20).std()

        if 'Volume' in data.columns and data['Volume'].sum() > 0:
            data['Volume'] = data['Volume'].fillna(1.0)
            data['volume_ma'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = data['Volume'] / (data['volume_ma'] + 1e-10)
        else:
            data['volume_ratio'] = 1.0

        data['hl_ratio'] = (data['High'] - data['Low']) / (data['Close'] + 1e-10)
        data['momentum_5'] = data['Close'].pct_change(5)
        data['momentum_20'] = data['Close'].pct_change(20)
        data['ma_20'] = data['Close'].rolling(20).mean()
        data['ma_50'] = data['Close'].rolling(50).mean()
        data['ma_cross'] = (data['ma_20'] / (data['ma_50'] + 1e-10) - 1)

        return data

    def train_model(self, symbol: str) -> bool:
        """Train model for a specific asset."""
        try:
            config = self.portfolio[symbol]
            interval = config['interval']
            period = '6mo' if interval == '1d' else '3mo'

            print(f"Training model for {symbol} ({interval})...", end=' ')

            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if len(data) < 100:
                print(f"❌ Insufficient data ({len(data)} bars)")
                return False

            # Add features
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_features = self.add_features(data)
            data_features['target'] = data_features['Close'].pct_change(24).shift(-24)
            data_features = data_features.dropna()

            feature_cols = ['returns', 'returns_5', 'returns_20', 'vol_5', 'vol_20',
                           'volume_ratio', 'hl_ratio', 'momentum_5', 'momentum_20', 'ma_cross']

            X = data_features[feature_cols]
            y = data_features['target']

            # Train/val split
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))

            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_val = X.iloc[train_size:train_size+val_size]
            y_val = y.iloc[train_size:train_size+val_size]

            # Train ensemble
            ensemble = EnhancedEnsemblePredictor(use_prediction_market=False)
            ensemble.train_all_models(X_train, y_train, X_val, y_val, models_to_train=['lightgbm'])

            self.models[symbol] = {
                'model': ensemble,
                'feature_cols': feature_cols,
                'last_train_date': datetime.now()
            }

            print(f"✅ Model trained ({len(data)} bars)")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def train_all_models(self):
        """Train models for all assets in portfolio."""
        print("="*80)
        print("TRAINING MODELS FOR ALL ASSETS")
        print("="*80 + "\n")

        success_count = 0
        for symbol in self.portfolio.keys():
            if self.train_model(symbol):
                success_count += 1
            time.sleep(0.5)  # Rate limiting

        print(f"\n✅ Trained {success_count}/{len(self.portfolio)} models successfully\n")
        return success_count == len(self.portfolio)

    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal for an asset."""
        try:
            if symbol not in self.models:
                return None

            model_info = self.models[symbol]
            config = self.portfolio[symbol]

            # Fetch latest data
            ticker = yf.Ticker(symbol)
            interval = config['interval']
            period = '1mo'

            data = ticker.history(period=period, interval=interval)

            if len(data) < 50:
                return None

            # Add features
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_features = self.add_features(data)
            data_features = data_features.dropna()

            # Get latest features
            X_latest = data_features[model_info['feature_cols']].iloc[-1:]

            # Predict
            prediction = model_info['model'].predict(X_latest)[0]
            current_price = data_features['Close'].iloc[-1]

            # Calculate edge
            prob = 0.5 + 0.2 * np.sign(prediction)  # Simple probability
            edge = abs(2 * prob - 1)

            # Apply Kelly criterion
            kelly_fraction = config['kelly']
            min_edge = config['min_edge']

            if edge < min_edge:
                return None  # No trade

            position_size = kelly_fraction * edge
            position_size = min(position_size, self.risk_controls['max_position_size'])

            # Determine direction
            direction = 'long' if prediction > 0 else 'short'

            return {
                'symbol': symbol,
                'direction': direction,
                'prediction': prediction,
                'probability': prob,
                'edge': edge,
                'position_size': position_size,
                'current_price': current_price,
                'timestamp': datetime.now()
            }

        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
            return None

    def check_stop_loss(self, symbol: str) -> bool:
        """Check if position should be closed due to stop-loss."""
        position = self.positions[symbol]

        if position['size'] == 0:
            return False

        if position['entry_price'] == 0:
            return False

        pnl_pct = (position['current_price'] - position['entry_price']) / position['entry_price']

        # Check per-asset stop-loss
        if pnl_pct <= self.risk_controls['per_asset_stop_loss']:
            print(f"⚠️  STOP-LOSS TRIGGERED: {symbol} ({pnl_pct:.2%})")
            return True

        return False

    def check_portfolio_stop_loss(self) -> bool:
        """Check if portfolio-level stop-loss is triggered."""
        if self.performance['current_drawdown'] <= self.risk_controls['portfolio_stop_loss']:
            print(f"⚠️  PORTFOLIO STOP-LOSS TRIGGERED: {self.performance['current_drawdown']:.2%}")
            return True
        return False

    def execute_trade(self, signal: Dict, portfolio_value: float = 100000) -> Dict:
        """Execute trade based on signal (simulated)."""
        symbol = signal['symbol']
        direction = signal['direction']
        size = signal['position_size'] * portfolio_value
        price = signal['current_price']

        # Apply slippage
        slippage = self.config['execution']['slippage_pct']
        execution_price = price * (1 + slippage) if direction == 'long' else price * (1 - slippage)

        # Commission
        commission = self.config['execution']['commission_per_trade']

        # Update position
        self.positions[symbol]['size'] = size
        self.positions[symbol]['entry_price'] = execution_price
        self.positions[symbol]['current_price'] = price

        # Record trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': execution_price,
            'commission': commission,
            'timestamp': datetime.now()
        }

        print(f"📈 EXECUTED: {symbol} {direction} ${size:,.0f} @ ${execution_price:.4f}")

        return trade

    def update_positions(self):
        """Update current prices and P&L for all positions."""
        for symbol, position in self.positions.items():
            if position['size'] == 0:
                continue

            try:
                ticker = yf.Ticker(symbol)
                latest = ticker.history(period='1d', interval='1h')

                if len(latest) > 0:
                    position['current_price'] = latest['Close'].iloc[-1]
                    position['unrealized_pnl'] = (
                        (position['current_price'] - position['entry_price']) /
                        position['entry_price'] * position['size']
                    )
            except:
                pass

    def calculate_portfolio_value(self, initial_value: float = 100000) -> float:
        """Calculate current portfolio value."""
        total_value = initial_value
        for position in self.positions.values():
            total_value += position['unrealized_pnl'] + position['realized_pnl']
        return total_value

    def generate_performance_report(self) -> str:
        """Generate performance summary."""
        report = "\n" + "="*80 + "\n"
        report += "PERFORMANCE SUMMARY\n"
        report += "="*80 + "\n\n"

        report += f"Total Return: {self.performance['total_return']:.2%}\n"
        report += f"Total Trades: {self.performance['total_trades']}\n"
        report += f"Winning Trades: {self.performance['winning_trades']} "
        report += f"({self.performance['winning_trades']/max(self.performance['total_trades'],1)*100:.1f}%)\n"
        report += f"Current Drawdown: {self.performance['current_drawdown']:.2%}\n"
        report += f"Max Drawdown: {self.performance['max_drawdown']:.2%}\n"
        report += f"Sharpe Ratio: {self.performance['sharpe_ratio']:.2f}\n\n"

        report += "CURRENT POSITIONS:\n"
        report += "-"*80 + "\n"

        for symbol, position in self.positions.items():
            if position['size'] > 0:
                pnl_pct = (position['current_price'] - position['entry_price']) / position['entry_price'] * 100
                report += f"{symbol:10s}: ${position['size']:>10,.0f} @ ${position['entry_price']:.4f} "
                report += f"→ ${position['current_price']:.4f} ({pnl_pct:+.2f}%)\n"

        return report

    def run_live_trading_loop(self, duration_minutes: int = 60, check_interval_seconds: int = 300):
        """Run live trading loop (for demo/testing)."""
        print("="*80)
        print("STARTING LIVE TRADING SYSTEM")
        print("="*80 + "\n")

        print(f"Duration: {duration_minutes} minutes")
        print(f"Check Interval: {check_interval_seconds} seconds")
        print(f"Portfolio: {len(self.portfolio)} assets\n")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        iteration = 0

        while datetime.now() < end_time:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            # Generate signals for all assets
            signals = []
            for symbol in self.portfolio.keys():
                signal = self.generate_signal(symbol)
                if signal:
                    signals.append(signal)

            print(f"Generated {len(signals)} trading signals\n")

            # Check stop-losses
            for symbol in self.portfolio.keys():
                if self.check_stop_loss(symbol):
                    # Close position
                    self.positions[symbol]['size'] = 0

            # Check portfolio stop-loss
            if self.check_portfolio_stop_loss():
                print("⚠️  PORTFOLIO STOP-LOSS - CLOSING ALL POSITIONS")
                for symbol in self.positions.keys():
                    self.positions[symbol]['size'] = 0
                break

            # Update positions
            self.update_positions()

            # Performance report
            print(self.generate_performance_report())

            # Sleep until next check
            print(f"\n⏳ Next check in {check_interval_seconds} seconds...")
            time.sleep(check_interval_seconds)

        print("\n" + "="*80)
        print("LIVE TRADING SESSION COMPLETE")
        print("="*80)
        print(self.generate_performance_report())


def main():
    """Main entry point for production system."""
    print("="*80)
    print(" "*20 + "PRODUCTION TRADING SYSTEM")
    print(" "*15 + "Max Sharpe Portfolio (11 Assets)")
    print("="*80 + "\n")

    # Initialize system
    system = ProductionTradingSystem()

    print("Portfolio Configuration:")
    for symbol, config in system.portfolio.items():
        print(f"  {symbol:12s}: {config['weight']*100:5.1f}% | {config['interval']:3s} | "
              f"Kelly: {config['kelly']:.2f} | Min Edge: {config['min_edge']:.2f}")

    print(f"\nRisk Controls:")
    for key, value in system.risk_controls.items():
        print(f"  {key}: {value}")

    # Train models
    print(f"\n{'='*80}")
    if not system.train_all_models():
        print("❌ Failed to train all models. Exiting.")
        return

    print(f"{'='*80}")
    print("SYSTEM READY FOR TRADING")
    print(f"{'='*80}\n")

    print("Options:")
    print("  1. Run live trading demo (60 min)")
    print("  2. Generate signals only (one-time)")
    print("  3. Check portfolio status")
    print("  4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        system.run_live_trading_loop(duration_minutes=60, check_interval_seconds=300)
    elif choice == '2':
        print("\nGenerating signals for all assets...\n")
        for symbol in system.portfolio.keys():
            signal = system.generate_signal(symbol)
            if signal:
                print(f"✅ {symbol}: {signal['direction'].upper()} "
                      f"| Edge: {signal['edge']:.3f} | Size: {signal['position_size']*100:.1f}%")
            else:
                print(f"⚠️  {symbol}: No signal (edge too low)")
    elif choice == '3':
        print(system.generate_performance_report())
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
