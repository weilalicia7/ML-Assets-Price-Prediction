"""
Daily Trading Workflow
Automated script for daily volatility prediction and trading signal generation.

Usage:
    python daily_trading.py --portfolio watchlist.txt --account-size 100000
"""

import sys
sys.path.insert(0, '.')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

from src.data.fetch_data import DataFetcher
from src.features.technical_features import TechnicalFeatureEngineer
from src.features.volatility_features import VolatilityFeatureEngineer
from src.models.ensemble_model import EnsemblePredictor
from src.models.regime_detector import RegimeDetector
from src.trading.risk_manager import RiskManager, TradingSignalGenerator
from src.evaluation.metrics import VolatilityMetrics


class DailyTradingWorkflow:
    """
    Automated daily trading workflow.

    Steps:
    1. Fetch latest data
    2. Engineer features
    3. Load trained model (or train new one)
    4. Generate predictions
    5. Detect regimes
    6. Generate trading signals
    7. Calculate position sizes
    8. Output trade recommendations
    9. Save results
    """

    def __init__(
        self,
        portfolio: list,
        account_size: float = 100000,
        model_path: str = None,
        lookback_days: int = 500,
        risk_per_trade: float = 0.02,
        use_regime_detection: bool = True
    ):
        """
        Initialize daily trading workflow.

        Args:
            portfolio: List of tickers to trade
            account_size: Total account size
            model_path: Path to pre-trained model (optional)
            lookback_days: Days of historical data
            risk_per_trade: Max risk per trade
            use_regime_detection: Use regime-based filtering
        """
        self.portfolio = portfolio
        self.account_size = account_size
        self.model_path = model_path
        self.lookback_days = lookback_days

        # Initialize components
        self.risk_manager = RiskManager(
            account_size=account_size,
            max_position_risk=risk_per_trade
        )

        self.signal_generator = TradingSignalGenerator(
            regime_filter=use_regime_detection
        )

        self.regime_detector = RegimeDetector(method='gmm')

        self.model = None
        self.results = []

    def run_daily_workflow(self):
        """Execute complete daily workflow."""
        print("="*70)
        print(f"DAILY TRADING WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"\nPortfolio: {', '.join(self.portfolio)}")
        print(f"Account Size: ${self.account_size:,.2f}")

        # Step 1: Fetch latest data
        print("\n" + "="*70)
        print("STEP 1: FETCHING LATEST DATA")
        print("="*70)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        fetcher = DataFetcher(
            self.portfolio,
            start_date=start_date.strftime('%Y-%m-%d')
        )

        data = fetcher.fetch_all()
        print(f"[OK] Fetched {len(data)} rows")

        # Step 2: Engineer features
        print("\n" + "="*70)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*70)

        all_processed = []
        for ticker in self.portfolio:
            ticker_data = data[data['Ticker'] == ticker].copy()

            if len(ticker_data) < 200:
                print(f"[WARNING] {ticker}: Insufficient data ({len(ticker_data)} rows)")
                continue

            # Technical features
            tech_eng = TechnicalFeatureEngineer()
            ticker_data = tech_eng.add_all_features(ticker_data)

            # Volatility features
            vol_eng = VolatilityFeatureEngineer()
            ticker_data = vol_eng.add_all_features(ticker_data)

            all_processed.append(ticker_data)

        processed_data = pd.concat(all_processed)
        print(f"[OK] Engineered {len(processed_data.columns)} features")

        # Step 3: Load or train model
        print("\n" + "="*70)
        print("STEP 3: LOADING/TRAINING MODEL")
        print("="*70)

        if self.model_path and os.path.exists(self.model_path):
            print(f"[INFO] Loading model from {self.model_path}")
            self.model = EnsemblePredictor()
            self.model.load_ensemble(self.model_path)
        else:
            print("[INFO] Training new model...")
            self._train_model(processed_data)

        # Step 4: Generate predictions for TODAY
        print("\n" + "="*70)
        print("STEP 4: GENERATING PREDICTIONS")
        print("="*70)

        trade_recommendations = []

        for ticker in self.portfolio:
            ticker_data = processed_data[processed_data['Ticker'] == ticker].copy()

            if len(ticker_data) == 0:
                continue

            # Get latest data point
            latest = ticker_data.iloc[-1]
            current_price = latest['Close']

            # Prepare features for prediction
            exclude_cols = ['Ticker', 'AssetType', 'Open', 'High', 'Low', 'Close', 'Volume',
                           'target_volatility', 'volatility_regime']
            feature_cols = [col for col in ticker_data.columns if col not in exclude_cols]

            X_latest = ticker_data[feature_cols].iloc[-1:].copy()

            # Make prediction
            predicted_vol = self.model.predict(X_latest)[0]
            pred_with_uncertainty, lower_bound, upper_bound = \
                self.model.predict_with_uncertainty(X_latest)

            # Get historical volatility
            hist_vol = latest['hist_vol_20']

            # Detect regime
            historical_vols = ticker_data['hist_vol_20'].dropna().values
            regimes, regime_info = self.regime_detector.detect_regime(historical_vols)
            current_regime_id = self.regime_detector.predict_regime(hist_vol)
            regime_names = ['low', 'medium', 'high']
            current_regime = regime_names[current_regime_id]

            # Calculate volatility percentile
            vol_percentile = (hist_vol < historical_vols).mean()

            # Predict direction (based on recent trend)
            recent_returns = ticker_data['returns_1d'].iloc[-10:].mean()
            predicted_direction = 1 if recent_returns > 0 else -1
            direction_confidence = min(abs(recent_returns) * 100, 0.9)  # Simple confidence

            # Generate trading signal
            signal = self.signal_generator.generate_signal(
                current_price=current_price,
                predicted_volatility=predicted_vol,
                volatility_percentile=vol_percentile,
                predicted_direction=predicted_direction,
                direction_confidence=direction_confidence,
                current_regime=current_regime,
                historical_volatility=hist_vol
            )

            # Calculate position size if signal is not HOLD
            if signal['action'] != 'HOLD' and signal['stop_loss']:
                position = self.risk_manager.calculate_position_size(
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    predicted_volatility=predicted_vol,
                    confidence=signal['confidence']
                )

                # Adjust for signal multiplier
                position['shares'] = int(position['shares'] * signal['position_size_multiplier'])
                position['position_value'] = position['shares'] * signal['entry_price']

                trade_recommendations.append({
                    'ticker': ticker,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'current_price': current_price,
                    'action': signal['action'],
                    'reason': signal['reason'],
                    'confidence': signal['confidence'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'shares': position['shares'],
                    'position_value': position['position_value'],
                    'risk_amount': position['total_risk'],
                    'risk_pct': position['risk_pct'],
                    'predicted_volatility': predicted_vol,
                    'historical_volatility': hist_vol,
                    'regime': current_regime,
                    'lower_bound': lower_bound[0],
                    'upper_bound': upper_bound[0]
                })

            # Log prediction even if no trade
            print(f"\n{ticker}:")
            print(f"  Current Price:      ${current_price:.2f}")
            print(f"  Predicted Vol:      {predicted_vol:.2%} (80% CI: {lower_bound[0]:.2%} - {upper_bound[0]:.2%})")
            print(f"  Historical Vol:     {hist_vol:.2%}")
            print(f"  Regime:             {current_regime.upper()}")
            print(f"  Signal:             {signal['action']}")
            if signal['reason']:
                print(f"  Reason:             {signal['reason']}")

        # Step 5: Output recommendations
        print("\n" + "="*70)
        print("STEP 5: TRADE RECOMMENDATIONS")
        print("="*70)

        if len(trade_recommendations) == 0:
            print("\n[INFO] No trades recommended today - all signals are HOLD")
        else:
            print(f"\n[OK] {len(trade_recommendations)} trade(s) recommended:\n")

            for i, trade in enumerate(trade_recommendations, 1):
                print(f"\nTRADE #{i}: {trade['action']} {trade['ticker']}")
                print(f"  Confidence:       {trade['confidence']:.1%}")
                print(f"  Reason:           {trade['reason']}")
                print(f"  Entry:            ${trade['entry_price']:.2f}")
                print(f"  Stop Loss:        ${trade['stop_loss']:.2f} ({abs(trade['stop_loss']-trade['entry_price'])/trade['entry_price']:.1%})")
                print(f"  Take Profit:      ${trade['take_profit']:.2f} ({abs(trade['take_profit']-trade['entry_price'])/trade['entry_price']:.1%})")
                print(f"  Shares:           {trade['shares']}")
                print(f"  Position Value:   ${trade['position_value']:,.2f}")
                print(f"  Risk Amount:      ${trade['risk_amount']:,.2f}")
                print(f"  Risk %:           {trade['risk_pct']:.2%}")
                print(f"  Regime:           {trade['regime'].upper()}")

        # Step 6: Save results
        self._save_results(trade_recommendations)

        print("\n" + "="*70)
        print("[SUCCESS] Daily workflow complete!")
        print("="*70)

        return trade_recommendations

    def _train_model(self, data):
        """Train new model on historical data."""
        from src.models.base_models import VolatilityPredictor

        # Create target
        temp_predictor = VolatilityPredictor()
        data = temp_predictor.create_target(data, target_type='next_day_volatility')

        # Split data
        train_df, val_df, _ = temp_predictor.prepare_data(data)

        # Select features
        exclude_cols = ['Ticker', 'AssetType', 'target_volatility', 'volatility_regime',
                        'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        X_train = train_df[feature_cols]
        y_train = train_df['target_volatility']
        X_val = val_df[feature_cols]
        y_val = val_df['target_volatility']

        # Train ensemble
        self.model = EnsemblePredictor(random_state=42)
        self.model.train_all_models(
            X_train, y_train, X_val, y_val,
            models_to_train=['lightgbm', 'xgboost']
        )

        # Save model
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        model_path = f"models/ensemble_daily_{timestamp}.pkl"
        self.model.save_ensemble(model_path)
        self.model_path = model_path

    def _save_results(self, recommendations):
        """Save trade recommendations to file."""
        os.makedirs('data/daily_trades', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        if recommendations:
            df = pd.DataFrame(recommendations)
            csv_path = f"data/daily_trades/trades_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n[OK] Recommendations saved to {csv_path}")

            # Also save as JSON for programmatic access
            json_path = f"data/daily_trades/trades_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            print(f"[OK] JSON saved to {json_path}")
        else:
            # Save empty file to mark that workflow ran
            marker_path = f"data/daily_trades/no_trades_{timestamp}.txt"
            with open(marker_path, 'w') as f:
                f.write(f"No trades recommended on {datetime.now()}\n")
            print(f"\n[OK] Marker saved to {marker_path}")


def main():
    """Main entry point for daily trading script."""
    parser = argparse.ArgumentParser(description='Daily Volatility Trading Workflow')

    parser.add_argument('--portfolio', nargs='+',
                       help='List of tickers (or path to file with tickers)')
    parser.add_argument('--portfolio-file', type=str,
                       help='Path to file with tickers (one per line)')
    parser.add_argument('--account-size', type=float, default=100000,
                       help='Account size (default: 100000)')
    parser.add_argument('--model-path', type=str,
                       help='Path to pre-trained model')
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                       help='Max risk per trade as decimal (default: 0.02 = 2%%)')
    parser.add_argument('--no-regime-filter', action='store_true',
                       help='Disable regime-based filtering')

    args = parser.parse_args()

    # Get portfolio
    if args.portfolio_file:
        with open(args.portfolio_file, 'r') as f:
            portfolio = [line.strip() for line in f if line.strip()]
    elif args.portfolio:
        portfolio = args.portfolio
    else:
        # Default portfolio
        portfolio = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
        print(f"[INFO] Using default portfolio: {portfolio}")

    # Initialize and run workflow
    workflow = DailyTradingWorkflow(
        portfolio=portfolio,
        account_size=args.account_size,
        model_path=args.model_path,
        risk_per_trade=args.risk_per_trade,
        use_regime_detection=not args.no_regime_filter
    )

    recommendations = workflow.run_daily_workflow()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Portfolio Size:     {len(portfolio)} assets")
    print(f"  Trades Recommended: {len(recommendations)}")
    if recommendations:
        total_capital = sum(r['position_value'] for r in recommendations)
        total_risk = sum(r['risk_amount'] for r in recommendations)
        print(f"  Total Capital:      ${total_capital:,.2f} ({total_capital/args.account_size:.1%})")
        print(f"  Total Risk:         ${total_risk:,.2f} ({total_risk/args.account_size:.1%})")


if __name__ == "__main__":
    main()
