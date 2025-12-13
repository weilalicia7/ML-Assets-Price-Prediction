"""
Visualization Tools for Volatility Predictions
Creates publication-quality plots for model performance and predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class VolatilityPlotter:
    """
    Create visualizations for volatility predictions.
    """

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize plotter.

        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'bmh')
        """
        self.style = style
        if style != 'seaborn':
            plt.style.use(style)

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        tickers: Optional[np.ndarray] = None,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        title: str = "Volatility Predictions",
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs actuals with confidence intervals.

        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            dates: Datetime index
            tickers: Asset tickers (for multi-asset plots)
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: Time series
        ax = axes[0, 0]
        if dates is not None:
            ax.plot(dates, y_true, label='Actual', alpha=0.7, linewidth=1.5)
            ax.plot(dates, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)

            if lower_bound is not None and upper_bound is not None:
                ax.fill_between(dates, lower_bound, upper_bound,
                               alpha=0.2, label='80% Confidence Interval')
        else:
            ax.plot(y_true, label='Actual', alpha=0.7)
            ax.plot(y_pred, label='Predicted', alpha=0.7)

        ax.set_title('Volatility Over Time', fontweight='bold')
        ax.set_xlabel('Date' if dates is not None else 'Index')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        ax = axes[0, 1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        ax.set_title('Predicted vs Actual', fontweight='bold')
        ax.set_xlabel('Actual Volatility')
        ax.set_ylabel('Predicted Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3: Residuals
        ax = axes[1, 0]
        residuals = y_pred - y_true
        if dates is not None:
            ax.scatter(dates, residuals, alpha=0.5, s=20)
        else:
            ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_title('Prediction Residuals', fontweight='bold')
        ax.set_xlabel('Date' if dates is not None else 'Index')
        ax.set_ylabel('Residual (Pred - Actual)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Error distribution
        ax = axes[1, 1]
        errors = np.abs(y_true - y_pred)
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(x=np.mean(errors), color='r', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(errors):.4f}')
        ax.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(errors):.4f}')
        ax.set_title('Error Distribution', fontweight='bold')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Plot saved to {save_path}")

        return fig

    def plot_regime_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regimes: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        regime_names: List[str] = ['Low', 'Medium', 'High'],
        save_path: Optional[str] = None
    ):
        """
        Plot performance by volatility regime.

        Args:
            y_true: True values
            y_pred: Predictions
            regimes: Regime labels
            dates: Datetime index
            regime_names: Names for regimes
            save_path: Save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: Time series with regime colors
        ax = axes[0, 0]
        regime_colors = ['green', 'orange', 'red']

        if dates is not None:
            for regime_id, (regime_name, color) in enumerate(zip(regime_names, regime_colors)):
                mask = regimes == regime_id
                if mask.sum() > 0:
                    ax.scatter(dates[mask], y_true[mask], c=color, alpha=0.3,
                             s=30, label=f'{regime_name} Vol Regime')

            ax.plot(dates, y_pred, 'b-', alpha=0.7, linewidth=1, label='Predicted')

        ax.set_title('Actual Volatility by Regime', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error by regime
        ax = axes[0, 1]
        errors_by_regime = []
        for regime_id in range(len(regime_names)):
            mask = regimes == regime_id
            if mask.sum() > 0:
                errors = np.abs(y_true[mask] - y_pred[mask])
                errors_by_regime.append(errors)

        ax.boxplot(errors_by_regime, labels=regime_names)
        ax.set_title('Prediction Error by Regime', fontweight='bold')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)

        # Plot 3: Regime distribution
        ax = axes[1, 0]
        regime_counts = [np.sum(regimes == i) for i in range(len(regime_names))]
        ax.bar(regime_names, regime_counts, color=regime_colors, alpha=0.7, edgecolor='black')
        ax.set_title('Regime Distribution', fontweight='bold')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        total = sum(regime_counts)
        for i, (name, count) in enumerate(zip(regime_names, regime_counts)):
            pct = count / total * 100
            ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Performance metrics by regime
        ax = axes[1, 1]
        from sklearn.metrics import mean_absolute_error, r2_score

        metrics_data = []
        for regime_id, regime_name in enumerate(regime_names):
            mask = regimes == regime_id
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                r2 = r2_score(y_true[mask], y_pred[mask])
                metrics_data.append([mae, r2])

        metrics_df = pd.DataFrame(metrics_data, columns=['MAE', 'R²'], index=regime_names)

        x = np.arange(len(regime_names))
        width = 0.35

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, metrics_df['MAE'], width, label='MAE', alpha=0.7)
        bars2 = ax2.bar(x + width/2, metrics_df['R²'], width, label='R²',
                       alpha=0.7, color='orange')

        ax.set_xlabel('Regime')
        ax.set_ylabel('MAE', color='tab:blue')
        ax2.set_ylabel('R²', color='tab:orange')
        ax.set_title('Performance Metrics by Regime', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regime_names)
        ax.grid(True, alpha=0.3, axis='y')

        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.suptitle('Regime-Based Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Regime analysis plot saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Plot title
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get top N features
        top_features = importance_df.head(top_n).sort_values('importance', ascending=True)

        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                       color='steelblue', alpha=0.7, edgecolor='black')

        # Customize
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'], i, f" {row['importance']:.0f}",
                   va='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Feature importance plot saved to {save_path}")

        return fig

    def plot_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Compare multiple models.

        Args:
            metrics_dict: Dictionary of {model_name: {metric: value}}
            title: Plot title
            save_path: Save path
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['mae', 'rmse', 'r2']
        metric_names = ['MAE (lower is better)', 'RMSE (lower is better)', 'R² (higher is better)']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]

            model_names = list(metrics_dict.keys())
            values = [metrics_dict[name].get(metric, 0) for name in model_names]

            bars = ax.bar(model_names, values, alpha=0.7, edgecolor='black')

            # Color best bar
            if metric == 'r2':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.9)

            ax.set_title(metric_name, fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.4f}', ha='center',
                       va='bottom' if metric == 'r2' else 'top',
                       fontweight='bold')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Model comparison plot saved to {save_path}")

        return fig

    def plot_multi_asset(
        self,
        predictions_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot predictions for multiple assets.

        Args:
            predictions_df: DataFrame with columns: date, ticker, actual_volatility, predicted_volatility
            save_path: Save path
        """
        tickers = predictions_df['ticker'].unique()
        n_tickers = len(tickers)

        fig, axes = plt.subplots(n_tickers, 1, figsize=(15, 4*n_tickers))

        if n_tickers == 1:
            axes = [axes]

        for idx, ticker in enumerate(tickers):
            ax = axes[idx]
            ticker_data = predictions_df[predictions_df['ticker'] == ticker]

            ax.plot(ticker_data['date'], ticker_data['actual_volatility'],
                   label='Actual', alpha=0.7, linewidth=2)
            ax.plot(ticker_data['date'], ticker_data['predicted_volatility'],
                   label='Predicted', alpha=0.7, linewidth=2)

            # Add confidence intervals if available
            if 'lower_bound_80%' in ticker_data.columns:
                ax.fill_between(ticker_data['date'],
                               ticker_data['lower_bound_80%'],
                               ticker_data['upper_bound_80%'],
                               alpha=0.2, label='80% CI')

            ax.set_title(f'{ticker} - Volatility Predictions', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Calculate and display metrics
            mae = np.mean(np.abs(ticker_data['actual_volatility'] - ticker_data['predicted_volatility']))
            ax.text(0.02, 0.98, f'MAE: {mae:.6f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.suptitle('Multi-Asset Volatility Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Multi-asset plot saved to {save_path}")

        return fig


def main():
    """Example usage."""
    print("="*60)
    print("VISUALIZATION MODULE - EXAMPLE")
    print("="*60)

    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    y_true = np.abs(np.random.randn(n) * 0.02 + 0.02)
    y_pred = y_true + np.random.randn(n) * 0.005
    regimes = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    lower = y_pred - 0.005
    upper = y_pred + 0.005

    # Create plotter
    plotter = VolatilityPlotter()

    # Create output directory
    os.makedirs('plots', exist_ok=True)

    # Plot predictions
    print("\n[INFO] Creating prediction plot...")
    plotter.plot_predictions(
        y_true, y_pred, dates,
        lower_bound=lower, upper_bound=upper,
        title="Example Volatility Predictions",
        save_path="plots/example_predictions.png"
    )

    # Plot regime analysis
    print("[INFO] Creating regime analysis plot...")
    plotter.plot_regime_analysis(
        y_true, y_pred, regimes, dates,
        save_path="plots/example_regime_analysis.png"
    )

    # Plot feature importance
    print("[INFO] Creating feature importance plot...")
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.randint(100, 1000, 20)
    }).sort_values('importance', ascending=False)

    plotter.plot_feature_importance(
        importance_df,
        save_path="plots/example_feature_importance.png"
    )

    # Plot model comparison
    print("[INFO] Creating model comparison plot...")
    metrics_dict = {
        'LightGBM': {'mae': 0.008, 'rmse': 0.012, 'r2': 0.25},
        'XGBoost': {'mae': 0.009, 'rmse': 0.013, 'r2': 0.20},
        'Ensemble': {'mae': 0.007, 'rmse': 0.011, 'r2': 0.30}
    }

    plotter.plot_comparison(
        metrics_dict,
        save_path="plots/example_model_comparison.png"
    )

    print("\n[SUCCESS] All example plots created in 'plots/' directory!")


if __name__ == "__main__":
    main()
