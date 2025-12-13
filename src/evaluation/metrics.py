"""
Evaluation Metrics for Volatility Prediction
Comprehensive metrics including standard regression metrics and volatility-specific ones.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class VolatilityMetrics:
    """
    Calculate comprehensive evaluation metrics for volatility prediction.
    """

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate all metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan

        # Directional accuracy (did we predict up/down correctly?)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
        else:
            metrics['directional_accuracy'] = np.nan

        # Max error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))

        # Median absolute error
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))

        return metrics

    @staticmethod
    def calculate_coverage(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> Dict:
        """
        Calculate coverage metrics for prediction intervals.

        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bound: Lower prediction bound
            upper_bound: Upper prediction bound

        Returns:
            Dictionary with coverage metrics
        """
        # Check if true values fall within bounds
        within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)

        coverage = {
            'coverage_rate': np.mean(within_bounds) * 100,
            'avg_interval_width': np.mean(upper_bound - lower_bound),
            'interval_width_std': np.std(upper_bound - lower_bound),
            'underestimate_rate': np.mean(y_true < lower_bound) * 100,
            'overestimate_rate': np.mean(y_true > upper_bound) * 100
        }

        return coverage

    @staticmethod
    def calculate_volatility_specific_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate volatility-specific metrics.

        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values

        Returns:
            Dictionary with volatility-specific metrics
        """
        metrics = {}

        # Bias in high vs low volatility periods
        median_vol = np.median(y_true)
        high_vol_mask = y_true > median_vol
        low_vol_mask = y_true <= median_vol

        if high_vol_mask.sum() > 0:
            metrics['mae_high_vol'] = mean_absolute_error(
                y_true[high_vol_mask], y_pred[high_vol_mask]
            )

        if low_vol_mask.sum() > 0:
            metrics['mae_low_vol'] = mean_absolute_error(
                y_true[low_vol_mask], y_pred[low_vol_mask]
            )

        # Volatility clustering prediction
        # Can we predict when volatility will be high?
        true_high_vol = (y_true > median_vol).astype(int)
        pred_high_vol = (y_pred > np.median(y_pred)).astype(int)
        metrics['vol_regime_accuracy'] = np.mean(true_high_vol == pred_high_vol) * 100

        # Under/over prediction bias
        residuals = y_pred - y_true
        metrics['mean_bias'] = np.mean(residuals)
        metrics['bias_std'] = np.std(residuals)

        return metrics

    @staticmethod
    def print_metrics_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test"
    ):
        """
        Print formatted metrics report.

        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset (for display)
        """
        metrics = VolatilityMetrics.calculate_all_metrics(y_true, y_pred)
        vol_metrics = VolatilityMetrics.calculate_volatility_specific_metrics(y_true, y_pred)

        print(f"\n{'='*60}")
        print(f"{dataset_name} SET METRICS")
        print(f"{'='*60}")

        print(f"\nStandard Metrics:")
        print(f"  MAE:                    {metrics['mae']:.6f}")
        print(f"  RMSE:                   {metrics['rmse']:.6f}")
        print(f"  R²:                     {metrics['r2']:.4f}")
        print(f"  MAPE:                   {metrics['mape']:.2f}%")
        print(f"  Median AE:              {metrics['median_ae']:.6f}")
        print(f"  Max Error:              {metrics['max_error']:.6f}")
        print(f"  Directional Accuracy:   {metrics['directional_accuracy']:.2f}%")

        print(f"\nVolatility-Specific Metrics:")
        if 'mae_high_vol' in vol_metrics:
            print(f"  MAE (High Vol Periods): {vol_metrics['mae_high_vol']:.6f}")
        if 'mae_low_vol' in vol_metrics:
            print(f"  MAE (Low Vol Periods):  {vol_metrics['mae_low_vol']:.6f}")
        print(f"  Vol Regime Accuracy:    {vol_metrics['vol_regime_accuracy']:.2f}%")
        print(f"  Mean Bias:              {vol_metrics['mean_bias']:.6f}")
        print(f"  Bias Std Dev:           {vol_metrics['bias_std']:.6f}")

        return metrics, vol_metrics

    @staticmethod
    def plot_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex = None,
        title: str = "Volatility Predictions",
        save_path: str = None
    ):
        """
        Plot predictions vs actuals.

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date index (optional)
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Time series
        ax = axes[0, 0]
        if dates is not None:
            ax.plot(dates, y_true, label='Actual', alpha=0.7)
            ax.plot(dates, y_pred, label='Predicted', alpha=0.7)
        else:
            ax.plot(y_true, label='Actual', alpha=0.7)
            ax.plot(y_pred, label='Predicted', alpha=0.7)
        ax.set_title('Volatility Over Time')
        ax.set_xlabel('Date' if dates is not None else 'Index')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        ax = axes[0, 1]
        ax.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax.set_title('Predicted vs Actual')
        ax.set_xlabel('Actual Volatility')
        ax.set_ylabel('Predicted Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax = axes[1, 0]
        residuals = y_pred - y_true
        if dates is not None:
            ax.scatter(dates, residuals, alpha=0.5)
        else:
            ax.scatter(range(len(residuals)), residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Prediction Residuals')
        ax.set_xlabel('Date' if dates is not None else 'Index')
        ax.set_ylabel('Residual (Pred - Actual)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Error distribution
        ax = axes[1, 1]
        errors = np.abs(y_true - y_pred)
        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
        ax.axvline(x=np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.4f}')
        ax.set_title('Error Distribution')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Plot saved to {save_path}")

        return fig


def main():
    """
    Example usage of VolatilityMetrics.
    """
    import sys
    sys.path.insert(0, '.')

    from src.data.fetch_data import DataFetcher
    from src.features.technical_features import TechnicalFeatureEngineer
    from src.features.volatility_features import VolatilityFeatureEngineer
    from src.models.ensemble_model import EnsemblePredictor
    from src.models.base_models import VolatilityPredictor

    print("="*60)
    print("EVALUATION METRICS - EXAMPLE")
    print("="*60)

    # Get data and train model (abbreviated)
    print("\nPreparing data and training model...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    tech_eng = TechnicalFeatureEngineer()
    aapl = tech_eng.add_all_features(aapl)
    vol_eng = VolatilityFeatureEngineer()
    aapl = vol_eng.add_all_features(aapl)

    temp_pred = VolatilityPredictor()
    aapl = temp_pred.create_target(aapl)
    train_df, val_df, test_df = temp_pred.prepare_data(aapl)

    exclude_cols = ['Ticker', 'AssetType', 'target_volatility', 'volatility_regime',
                    'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['target_volatility']
    X_test = test_df[feature_cols]
    y_test = test_df['target_volatility']

    ensemble = EnsemblePredictor()
    ensemble.train_all_models(X_train, y_train, X_test, y_test, models_to_train=['lightgbm'])

    # Make predictions
    print("\nMaking predictions...")
    y_pred = ensemble.predict(X_test)
    y_pred_lower, y_pred_upper = None, None

    if len(ensemble.models) > 1:
        y_pred, y_pred_lower, y_pred_upper = ensemble.predict_with_uncertainty(X_test)

    # Calculate metrics
    print("\nCalculating comprehensive metrics...")
    metrics_evaluator = VolatilityMetrics()
    metrics_evaluator.print_metrics_report(y_test.values, y_pred, dataset_name="Test")

    # Coverage metrics if we have bounds
    if y_pred_lower is not None and y_pred_upper is not None:
        print(f"\nPrediction Interval Coverage:")
        coverage = metrics_evaluator.calculate_coverage(
            y_test.values, y_pred, y_pred_lower, y_pred_upper
        )
        print(f"  Coverage Rate:          {coverage['coverage_rate']:.2f}%")
        print(f"  Avg Interval Width:     {coverage['avg_interval_width']:.6f}")
        print(f"  Underestimate Rate:     {coverage['underestimate_rate']:.2f}%")
        print(f"  Overestimate Rate:      {coverage['overestimate_rate']:.2f}%")

    print("\n[SUCCESS] Metrics calculation complete!")


if __name__ == "__main__":
    main()
