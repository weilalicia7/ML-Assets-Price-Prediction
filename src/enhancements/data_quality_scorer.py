"""
Data Quality Scoring System
Calculates and displays quality metrics for predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class DataQualityScore:
    """Container for data quality metrics"""
    overall_score: float  # 0-100
    completeness_score: float  # % of non-interpolated data
    reliability_score: float  # based on data source quality
    freshness_score: float  # based on data recency
    confidence_interval: Tuple[float, float]  # (lower, upper) for predictions
    quality_grade: str  # A, B, C, D, F
    warnings: list  # List of quality warnings


class DataQualityScorer:
    """
    Calculate comprehensive data quality scores

    Scoring Components:
    1. Completeness (40%): % of original vs interpolated data
    2. Reliability (30%): Source quality and consistency
    3. Freshness (20%): How recent is the data
    4. Volatility (10%): Data stability/noise level

    Final Score = Σ (Component × Weight)
    """

    def __init__(self):
        self.weights = {
            'completeness': 0.40,
            'reliability': 0.30,
            'freshness': 0.20,
            'volatility': 0.10
        }

    def calculate_score(self,
                       original_rows: int,
                       final_rows: int,
                       interpolated_values: int,
                       total_values: int,
                       data_age_days: int,
                       volatility: float) -> DataQualityScore:
        """
        Calculate comprehensive data quality score

        Args:
            original_rows: Number of rows before cleaning
            final_rows: Number of rows after cleaning
            interpolated_values: Count of interpolated data points
            total_values: Total data points in final dataset
            data_age_days: Days since most recent data point
            volatility: Historical volatility of the asset

        Returns:
            DataQualityScore object with all metrics
        """
        warnings = []

        # 1. Completeness Score (0-100)
        if total_values > 0:
            interpolation_pct = (interpolated_values / total_values) * 100
            completeness = max(0, 100 - interpolation_pct)
        else:
            completeness = 0
            warnings.append("No data available")

        if interpolation_pct > 40:
            warnings.append(f"High interpolation: {interpolation_pct:.1f}% of data is estimated")

        # 2. Reliability Score (0-100)
        retention_rate = (final_rows / max(original_rows, 1)) * 100
        reliability = retention_rate  # Higher retention = more reliable

        if retention_rate < 50:
            warnings.append(f"Low data retention: only {retention_rate:.1f}% of original data retained")

        # 3. Freshness Score (0-100)
        # Penalize stale data
        if data_age_days <= 1:
            freshness = 100
        elif data_age_days <= 7:
            freshness = 90
        elif data_age_days <= 30:
            freshness = 70
        elif data_age_days <= 90:
            freshness = 50
        else:
            freshness = max(0, 50 - (data_age_days - 90) / 10)

        if data_age_days > 7:
            warnings.append(f"Stale data: {data_age_days} days old")

        # 4. Volatility Score (0-100)
        # Lower volatility = higher score (more stable predictions)
        # Typical stock volatility: 15-30% annual
        if volatility < 0.15:
            vol_score = 100
        elif volatility < 0.30:
            vol_score = 80
        elif volatility < 0.50:
            vol_score = 60
        else:
            vol_score = max(0, 60 - (volatility - 0.50) * 100)

        if volatility > 0.50:
            warnings.append(f"High volatility: {volatility*100:.1f}% (predictions less reliable)")

        # Calculate weighted overall score
        overall = (
            completeness * self.weights['completeness'] +
            reliability * self.weights['reliability'] +
            freshness * self.weights['freshness'] +
            vol_score * self.weights['volatility']
        )

        # Assign letter grade
        if overall >= 90:
            grade = 'A'
        elif overall >= 80:
            grade = 'B'
        elif overall >= 70:
            grade = 'C'
        elif overall >= 60:
            grade = 'D'
        else:
            grade = 'F'
            warnings.append("Poor data quality - predictions may be unreliable")

        # Calculate confidence interval based on quality
        # Higher quality = narrower confidence interval
        base_interval_width = 0.10  # ±10% base
        quality_factor = (100 - overall) / 100  # 0 to 1
        interval_width = base_interval_width * (1 + quality_factor)

        confidence_interval = (
            1.0 - interval_width,
            1.0 + interval_width
        )

        return DataQualityScore(
            overall_score=round(overall, 2),
            completeness_score=round(completeness, 2),
            reliability_score=round(reliability, 2),
            freshness_score=round(freshness, 2),
            confidence_interval=confidence_interval,
            quality_grade=grade,
            warnings=warnings
        )

    def get_display_dict(self, score: DataQualityScore) -> Dict:
        """
        Convert DataQualityScore to displayable dictionary

        Returns:
            Dictionary suitable for JSON serialization and UI display
        """
        return {
            'overall_score': score.overall_score,
            'quality_grade': score.quality_grade,
            'metrics': {
                'completeness': score.completeness_score,
                'reliability': score.reliability_score,
                'freshness': score.freshness_score
            },
            'confidence_interval': {
                'lower': round(score.confidence_interval[0], 3),
                'upper': round(score.confidence_interval[1], 3),
                'width_pct': round((score.confidence_interval[1] - score.confidence_interval[0]) * 100, 1)
            },
            'warnings': score.warnings,
            'interpretation': self._get_interpretation(score.quality_grade)
        }

    def _get_interpretation(self, grade: str) -> str:
        """Get human-readable interpretation of quality grade"""
        interpretations = {
            'A': 'Excellent data quality. Predictions highly reliable.',
            'B': 'Good data quality. Predictions generally reliable.',
            'C': 'Acceptable data quality. Predictions moderately reliable.',
            'D': 'Poor data quality. Use predictions with caution.',
            'F': 'Very poor data quality. Predictions unreliable.'
        }
        return interpretations.get(grade, 'Unknown quality')


def calculate_prediction_confidence(quality_score: DataQualityScore,
                                    model_mae: float,
                                    predicted_return: float) -> Dict:
    """
    Calculate confidence bounds for a prediction based on data quality

    Args:
        quality_score: DataQualityScore object
        model_mae: Model's Mean Absolute Error
        predicted_return: Predicted return percentage

    Returns:
        Dictionary with confidence bounds and probability estimates
    """
    # Adjust MAE based on quality score
    # Lower quality = higher error margin
    quality_factor = quality_score.overall_score / 100
    adjusted_mae = model_mae / quality_factor

    # Calculate confidence bounds (±2 MAE for ~95% confidence)
    lower_bound = predicted_return - (2 * adjusted_mae)
    upper_bound = predicted_return + (2 * adjusted_mae)

    # Estimate probability of positive return
    if predicted_return > 0:
        # Higher quality and further from 0 = higher probability
        prob_positive = min(95, 50 + (predicted_return / adjusted_mae) * 10 * quality_factor)
    else:
        prob_positive = max(5, 50 + (predicted_return / adjusted_mae) * 10 * quality_factor)

    return {
        'predicted_return': round(predicted_return, 4),
        'confidence_bounds': {
            'lower': round(lower_bound, 4),
            'upper': round(upper_bound, 4)
        },
        'probability_positive': round(prob_positive, 1),
        'quality_adjusted_error': round(adjusted_mae, 4),
        'quality_score': quality_score.overall_score
    }
