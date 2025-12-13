"""
Main Execution Script for Stock/Crypto Volatility Prediction
Complete end-to-end pipeline.
"""

import sys
sys.path.insert(0, '.')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os

from src.data.fetch_data import DataFetcher
from src.features.technical_features import TechnicalFeatureEngineer
from src.features.volatility_features import VolatilityFeatureEngineer
from src.models.base_models import VolatilityPredictor
from src.models.ensemble_model import EnsemblePredictor
from src.evaluation.metrics import VolatilityMetrics


def main(
    tickers: list,
    start_date: str = '2022-01-01',
    model_type: str = 'ensemble',
    save_model: bool = True,
    save_predictions: bool = True
):
    """
    Main execution pipeline.

    Args:
        tickers: List of tickers to predict
        start_date: Start date for historical data
        model_type: 'lightgbm', 'xgboost', or 'ensemble'
        save_model: Whether to save trained model
        save_predictions: Whether to save predictions
    """
    print("="*70)
    print("STOCK/CRYPTO VOLATILITY PREDICTION SYSTEM")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Tickers:     {', '.join(tickers)}")
    print(f"  Start Date:  {start_date}")
    print(f"  Model Type:  {model_type}")
    print(f"  Save Model:  {save_model}")

    # ========================================
    # STEP 1: FETCH DATA
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 1: FETCHING DATA")
    print(f"{'='*70}")

    fetcher = DataFetcher(tickers, start_date=start_date)
    data = fetcher.fetch_all()

    print(f"\n[OK] Fetched {len(data)} total rows")
    print(f"[OK] Date range: {data.index.min()} to {data.index.max()}")
    print(f"[OK] Assets: {data['Ticker'].unique().tolist()}")

    # ========================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 2: FEATURE ENGINEERING")
    print(f"{'='*70}")

    # Process each ticker separately
    all_processed_data = []

    for ticker in data['Ticker'].unique():
        print(f"\n[INFO] Processing {ticker}...")
        ticker_data = data[data['Ticker'] == ticker].copy()

        # Technical features
        tech_eng = TechnicalFeatureEngineer()
        ticker_data = tech_eng.add_all_features(ticker_data)

        # Volatility features
        vol_eng = VolatilityFeatureEngineer()
        ticker_data = vol_eng.add_all_features(ticker_data)

        print(f"[OK] {ticker}: {len(ticker_data)} rows, {len(ticker_data.columns)} columns")

        all_processed_data.append(ticker_data)

    # Combine
    processed_data = pd.concat(all_processed_data)
    print(f"\n[OK] Total processed: {len(processed_data)} rows with {len(processed_data.columns)} columns")

    # ========================================
    # STEP 3: PREPARE FOR TRAINING
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 3: PREPARING DATA")
    print(f"{'='*70}")

    # Create target
    temp_predictor = VolatilityPredictor()
    processed_data = temp_predictor.create_target(processed_data, target_type='next_day_volatility')

    # Split data
    train_df, val_df, test_df = temp_predictor.prepare_data(processed_data)

    # Select features
    exclude_cols = ['Ticker', 'AssetType', 'target_volatility', 'volatility_regime',
                    'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['target_volatility']
    X_val = val_df[feature_cols]
    y_val = val_df['target_volatility']
    X_test = test_df[feature_cols]
    y_test = test_df['target_volatility']

    print(f"\n[OK] Features: {len(feature_cols)}")
    print(f"[OK] Train: {len(X_train)} rows")
    print(f"[OK] Val:   {len(X_val)} rows")
    print(f"[OK] Test:  {len(X_test)} rows")

    # ========================================
    # STEP 4: TRAIN MODEL
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 4: TRAINING MODEL")
    print(f"{'='*70}")

    if model_type == 'ensemble':
        print("\n[INFO] Training ensemble (LightGBM + XGBoost)...")
        model = EnsemblePredictor(random_state=42)
        model.train_all_models(X_train, y_train, X_val, y_val,
                               models_to_train=['lightgbm', 'xgboost'])
    else:
        print(f"\n[INFO] Training {model_type.upper()}...")
        model = VolatilityPredictor(model_type=model_type, random_state=42)

        if model_type == 'lightgbm':
            model.train_lightgbm(X_train, y_train, X_val, y_val)
        elif model_type == 'xgboost':
            model.train_xgboost(X_train, y_train, X_val, y_val)

    # ========================================
    # STEP 5: EVALUATE
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 5: EVALUATION")
    print(f"{'='*70}")

    # Predictions
    if model_type == 'ensemble':
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # With uncertainty
        y_pred_test_full, lower_bound, upper_bound = model.predict_with_uncertainty(X_test)
    else:
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        lower_bound = upper_bound = None

    # Metrics
    metrics_evaluator = VolatilityMetrics()

    # Train metrics
    metrics_evaluator.print_metrics_report(y_train.values, y_pred_train, "Train")

    # Validation metrics
    metrics_evaluator.print_metrics_report(y_val.values, y_pred_val, "Validation")

    # Test metrics (most important)
    test_metrics, vol_metrics = metrics_evaluator.print_metrics_report(y_test.values, y_pred_test, "Test")

    # Coverage if ensemble
    if lower_bound is not None:
        print(f"\nPrediction Interval Coverage (Test Set):")
        coverage = metrics_evaluator.calculate_coverage(
            y_test.values, y_pred_test, lower_bound, upper_bound
        )
        print(f"  Coverage Rate:      {coverage['coverage_rate']:.2f}%")
        print(f"  Avg Interval Width: {coverage['avg_interval_width']:.6f}")

    # ========================================
    # STEP 6: FEATURE IMPORTANCE
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 6: FEATURE IMPORTANCE")
    print(f"{'='*70}")

    print(f"\nTop 15 Most Important Features:")
    if model_type == 'ensemble':
        importance = model.get_feature_importance(top_n=15, method='weighted')
    else:
        importance = model.get_feature_importance(top_n=15)

    for idx, row in importance.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:>10.0f}")

    # ========================================
    # STEP 7: SAVE RESULTS
    # ========================================
    print(f"\n{'='*70}")
    print("STEP 7: SAVING RESULTS")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    if save_model:
        os.makedirs('models', exist_ok=True)
        model_path = f"models/{model_type}_model_{timestamp}.pkl"

        if model_type == 'ensemble':
            model.save_ensemble(model_path)
        else:
            model.save_model(model_path)

    # Save predictions
    if save_predictions:
        os.makedirs('data/predictions', exist_ok=True)

        predictions_df = pd.DataFrame({
            'date': test_df.index,
            'ticker': test_df['Ticker'].values,
            'actual_volatility': y_test.values,
            'predicted_volatility': y_pred_test
        })

        if lower_bound is not None:
            predictions_df['lower_bound_80%'] = lower_bound
            predictions_df['upper_bound_80%'] = upper_bound

        pred_path = f"data/predictions/predictions_{timestamp}.csv"
        predictions_df.to_csv(pred_path, index=False)
        print(f"[OK] Predictions saved to {pred_path}")

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}")

    print(f"\nModel Performance:")
    print(f"  Test MAE:               {test_metrics['mae']:.6f}")
    print(f"  Test RMSE:              {test_metrics['rmse']:.6f}")
    print(f"  Test R²:                {test_metrics['r2']:.4f}")
    print(f"  Test MAPE:              {test_metrics['mape']:.2f}%")
    print(f"  Directional Accuracy:   {test_metrics['directional_accuracy']:.2f}%")

    if 'vol_regime_accuracy' in vol_metrics:
        print(f"  Vol Regime Accuracy:    {vol_metrics['vol_regime_accuracy']:.2f}%")

    print(f"\n[SUCCESS] Pipeline execution complete!")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock/Crypto Volatility Prediction')

    parser.add_argument('--tickers', nargs='+', default=['AAPL'],
                       help='List of tickers to predict (default: AAPL)')
    parser.add_argument('--start-date', default='2022-01-01',
                       help='Start date for historical data (default: 2022-01-01)')
    parser.add_argument('--model', choices=['lightgbm', 'xgboost', 'ensemble'],
                       default='ensemble',
                       help='Model type (default: ensemble)')
    parser.add_argument('--no-save-model', action='store_true',
                       help='Don\'t save trained model')
    parser.add_argument('--no-save-predictions', action='store_true',
                       help='Don\'t save predictions')

    args = parser.parse_args()

    # Run pipeline
    model, metrics = main(
        tickers=args.tickers,
        start_date=args.start_date,
        model_type=args.model,
        save_model=not args.no_save_model,
        save_predictions=not args.no_save_predictions
    )
