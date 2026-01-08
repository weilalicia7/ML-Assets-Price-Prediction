# examples/complete_workflow.py
"""
Complete ML workflow demonstrating:
1. Data fetching and preprocessing
2. Descriptive statistics
3. Feature engineering
4. Model training
5. Evaluation on test set
"""

from src.data.fetch_data import DataFetcher
from src.features.feature_engineering import create_features
from src.features.technical_features import TechnicalFeatureEngineer
from src.models.base_models import VolatilityPredictor
import pandas as pd

def main():
    print("=" * 60)
    print("COMPLETE ML WORKFLOW")
    print("=" * 60)

    # ========================================
    # PART 1: DATA PREPROCESSING
    # ========================================
    print("\n[1/5] DATA FETCHING AND PREPROCESSING")
    print("-" * 60)

    # Fetch data
    fetcher = DataFetcher(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2020-01-01'
    )
    raw_data = fetcher.fetch_all()
    print(f"✓ Fetched {len(raw_data)} rows for {raw_data['Ticker'].nunique()} tickers")

    # ========================================
    # PART 2: DESCRIPTIVE STATISTICS
    # ========================================
    print("\n[2/5] DESCRIPTIVE STATISTICS")
    print("-" * 60)

    print("\nDataset Summary:")
    print(raw_data.describe())

    print("\nData by Ticker:")
    print(raw_data.groupby('Ticker')['Close'].agg(['count', 'mean', 'std']))

    # ========================================
    # PART 3: FEATURE ENGINEERING
    # ========================================
    print("\n[3/5] FEATURE ENGINEERING")
    print("-" * 60)

    # Base features
    features_df = create_features(raw_data)
    print(f"✓ Created {len(features_df.columns)} base features")

    # Technical features
    tech_engineer = TechnicalFeatureEngineer()
    full_features = tech_engineer.add_all_features(features_df)
    full_features = full_features.dropna()
    print(f"✓ Added technical indicators. Total features: {len(full_features.columns)}")

    # ========================================
    # PART 4: TRAIN/TEST SPLIT & TRAINING
    # ========================================
    print("\n[4/5] MODEL TRAINING")
    print("-" * 60)

    # Initialize predictor
    predictor = VolatilityPredictor(model_type='lightgbm')

    # Split data
    train_df, val_df, test_df = predictor.prepare_data(full_features)
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Prepare features
    feature_cols = [col for col in full_features.columns
                   if col not in ['next_day_direction', 'Date', 'Ticker']]

    X_train = train_df[feature_cols]
    y_train = predictor.create_target(train_df)
    X_val = val_df[feature_cols]
    y_val = predictor.create_target(val_df)
    X_test = test_df[feature_cols]
    y_test = predictor.create_target(test_df)

    # Train
    print("✓ Training model...")
    model = predictor.train_lightgbm(X_train, y_train, X_val, y_val)

    # ========================================
    # PART 5: EVALUATION
    # ========================================
    print("\n[5/5] TEST SET EVALUATION")
    print("-" * 60)

    metrics = predictor.evaluate(X_test, y_test)

    print(f"\nTest Set Metrics:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
