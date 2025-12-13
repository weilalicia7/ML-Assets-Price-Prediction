"""
Data Quality Monitoring for Commodity Futures
Tracks interpolation frequency, data retention, and quality metrics
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor data quality metrics for model training"""

    def __init__(self, log_dir: str = "logs/data_quality"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Dict] = {}

    def log_data_quality(self, ticker: str, metrics: Dict):
        """
        Log data quality metrics for a ticker

        Args:
            ticker: Stock/commodity ticker symbol
            metrics: Dictionary containing quality metrics
        """
        timestamp = datetime.now().isoformat()

        self.metrics[ticker] = {
            'timestamp': timestamp,
            'ticker': ticker,
            **metrics
        }

        # Log to file
        log_file = self.log_dir / f"data_quality_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(self.metrics[ticker]) + '\n')

        # Check thresholds and alert if needed
        self._check_thresholds(ticker, metrics)

    def _check_thresholds(self, ticker: str, metrics: Dict):
        """Check if metrics exceed alert thresholds"""

        # Alert if >40% of data is interpolated
        interpolation_pct = metrics.get('interpolation_percentage', 0)
        if interpolation_pct > 40:
            logger.warning(
                f"⚠️  HIGH INTERPOLATION: {ticker} has {interpolation_pct:.1f}% interpolated data (threshold: 40%)"
            )

        # Alert if data retention is very low
        retention_pct = metrics.get('retention_percentage', 100)
        if retention_pct < 50:
            logger.warning(
                f"⚠️  LOW DATA RETENTION: {ticker} retained only {retention_pct:.1f}% of data after cleaning"
            )

        # Alert if fallback to extended lookback was needed
        if metrics.get('used_extended_lookback', False):
            logger.info(
                f"ℹ️  EXTENDED LOOKBACK: {ticker} required 3000-day lookback for sufficient data"
            )

        # Alert if categorical columns were dropped
        categorical_drops = metrics.get('categorical_columns_dropped', 0)
        if categorical_drops > 0:
            logger.info(
                f"ℹ️  CATEGORICAL DROPS: {ticker} dropped {categorical_drops} categorical columns with NaN"
            )

    def get_summary(self, ticker: Optional[str] = None) -> Dict:
        """
        Get quality metrics summary

        Args:
            ticker: Optional ticker to filter. If None, returns all.

        Returns:
            Summary dictionary with quality metrics
        """
        if ticker:
            return self.metrics.get(ticker, {})

        # Aggregate summary across all tickers
        total_tickers = len(self.metrics)
        high_interpolation = sum(
            1 for m in self.metrics.values()
            if m.get('interpolation_percentage', 0) > 40
        )
        used_fallback = sum(
            1 for m in self.metrics.values()
            if m.get('used_extended_lookback', False)
        )

        return {
            'total_tickers_monitored': total_tickers,
            'high_interpolation_count': high_interpolation,
            'fallback_usage_count': used_fallback,
            'average_retention': sum(
                m.get('retention_percentage', 0)
                for m in self.metrics.values()
            ) / max(total_tickers, 1)
        }

    def export_report(self, filepath: str):
        """Export detailed quality report to JSON file"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'details': self.metrics
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Data quality report exported to {filepath}")


class ModelPerformanceMonitor:
    """Monitor model prediction performance over time"""

    def __init__(self, log_dir: str = "logs/model_performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: List[Dict] = []

    def log_prediction(self, ticker: str, prediction: Dict):
        """
        Log a model prediction for future validation

        Args:
            ticker: Ticker symbol
            prediction: Prediction details including forecast, confidence, etc.
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            **prediction
        }

        self.predictions.append(record)

        # Log to daily file
        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_actual_outcome(self, ticker: str, prediction_date: str, actual_return: float):
        """
        Log actual outcome for a prediction to calculate accuracy

        Args:
            ticker: Ticker symbol
            prediction_date: Date when prediction was made
            actual_return: Actual observed return
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'prediction_date': prediction_date,
            'actual_return': actual_return,
            'validation_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Log to validation file
        log_file = self.log_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def calculate_accuracy_metrics(self, ticker: Optional[str] = None) -> Dict:
        """
        Calculate prediction accuracy metrics

        Args:
            ticker: Optional ticker to filter

        Returns:
            Dictionary with MAE, RMSE, directional accuracy
        """
        # This would compare predictions vs actual outcomes
        # For now, return placeholder
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'directional_accuracy': 0.0,
            'note': 'Requires validation data collection over time'
        }


class CacheMonitor:
    """Monitor model cache performance"""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.training_times: List[float] = []

    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1

    def record_cache_miss(self, training_time: float):
        """
        Record a cache miss with training time

        Args:
            training_time: Time taken to train model in seconds
        """
        self.cache_misses += 1
        self.training_times.append(training_time)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def get_avg_training_time(self) -> float:
        """Get average model training time"""
        if not self.training_times:
            return 0.0
        return sum(self.training_times) / len(self.training_times)

    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': self.get_hit_rate(),
            'avg_training_time_seconds': self.get_avg_training_time(),
            'total_training_time_seconds': sum(self.training_times)
        }


# Global monitors (singleton pattern)
_data_quality_monitor = None
_performance_monitor = None
_cache_monitor = None


def get_data_quality_monitor() -> DataQualityMonitor:
    """Get or create the global data quality monitor"""
    global _data_quality_monitor
    if _data_quality_monitor is None:
        _data_quality_monitor = DataQualityMonitor()
    return _data_quality_monitor


def get_performance_monitor() -> ModelPerformanceMonitor:
    """Get or create the global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = ModelPerformanceMonitor()
    return _performance_monitor


def get_cache_monitor() -> CacheMonitor:
    """Get or create the global cache monitor"""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CacheMonitor()
    return _cache_monitor
