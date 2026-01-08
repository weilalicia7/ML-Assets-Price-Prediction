# examples/train_and_evaluate.py
"""
Train machine learning models and evaluate on test set.

This script demonstrates:
1. Data preprocessing and cleaning
2. Feature engineering (90+ features)
3. Train/validation/test split
4. Model training with LightGBM
5. Evaluation on test set
6. Feature importance analysis
"""

from src.data.fetch_data import DataFetcher
from src.features.feature_engineering import create_features
from src.features.technical_features import TechnicalFeatureEngineer
from src.models.base_models import VolatilityPredictor
from src.evaluation.metrics import VolatilityMetrics
import pandas as pd

print("=== MODEL TRAINING AND EVALUATION ===\n")

# Step 1: Data Preprocessing
print("Step 1: Fetching and preprocessing data...")
fetcher = DataFetcher(tickers=['AAPL'], start_date='2020-01-01')
raw_data = fetcher.fetch_all()

# Step 2: Feature Engineering
print("Step 2: Engineering features...")
base_features = create_features(raw_data)

# Add advanced technical features
tech_engineer = TechnicalFeatureEngineer()
full_features = tech_engineer.add_all_features(base_features)

# Remove NaN from rolling calculations
full_features = full_features.dropna()
print(f"Final feature set shape: {full_features.shape}")
print(f"Number of features: {len(full_features.columns)}")

# Step 3: Train/Test Split
print("\nStep 3: Splitting data (Train 70%, Val 15%, Test 15%)...")
predictor = VolatilityPredictor(model_type='lightgbm')
train_df, val_df, test_df = predictor.prepare_data(full_features)

print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Step 4: Model Training
print("\nStep 4: Training LightGBM model...")

# Create target variable
train_df = predictor.create_target(train_df, target_type='next_day_volatility')
val_df = predictor.create_target(val_df, target_type='next_day_volatility')
test_df = predictor.create_target(test_df, target_type='next_day_volatility')

# Prepare features and target
feature_cols = [col for col in full_features.columns if col not in ['next_day_direction', 'Date', 'Ticker', 'target_volatility']]
X_train = train_df[feature_cols]
y_train = train_df['target_volatility']

X_val = val_df[feature_cols]
y_val = val_df['target_volatility']

X_test = test_df[feature_cols]
y_test = test_df['target_volatility']

# Train the model
model = predictor.train_lightgbm(X_train, y_train, X_val, y_val)
print("Training complete!")

# Step 5: Evaluation on Test Set
print("\nStep 5: Evaluating on test set...")
test_predictions = predictor.predict(X_test)
test_metrics = predictor.evaluate(X_test, y_test)

print("\n=== TEST SET RESULTS ===")
print(f"MAE: {test_metrics['mae']:.6f}")
print(f"RMSE: {test_metrics['rmse']:.6f}")
print(f"R²: {test_metrics['r2']:.6f}")
print(f"MAPE: {test_metrics['mape']:.2f}%")

# Feature importance
print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
importance = predictor.get_feature_importance(top_n=10)
for i, (idx, row) in enumerate(importance.iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f}")

print("\n=== Training and evaluation complete ===")
