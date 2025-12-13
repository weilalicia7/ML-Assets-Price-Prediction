"""
Advanced Interpolation Methods for Future Enhancement
Includes cubic spline, seasonal decomposition, and ARIMA-based imputation
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CubicSplineInterpolator:
    """
    Cubic spline interpolation for smoother gap filling

    Mathematical Foundation:
    A cubic spline S(x) is a piecewise cubic polynomial that:
    - Passes through all data points
    - Has continuous first and second derivatives
    - Minimizes curvature (smoothest curve)

    For each interval [x_i, x_{i+1}], the spline is:
    S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)² + d_i(x - x_i)³
    """

    def __init__(self, kind: str = 'cubic'):
        """
        Args:
            kind: Type of spline ('cubic', 'quadratic', 'linear')
        """
        self.kind = kind

    def interpolate(self, series: pd.Series) -> pd.Series:
        """
        Apply cubic spline interpolation to fill NaN values

        Args:
            series: Pandas series with potential NaN values

        Returns:
            Interpolated series
        """
        # Get indices of non-NaN values
        valid_idx = series.notna()

        if valid_idx.sum() < 4:
            # Need at least 4 points for cubic spline
            logger.warning(f"Insufficient points for cubic spline, falling back to linear")
            return series.interpolate(method='linear')

        # Extract valid data points
        x_valid = np.arange(len(series))[valid_idx]
        y_valid = series[valid_idx].values

        # Create spline interpolator
        try:
            spline = interpolate.interp1d(
                x_valid,
                y_valid,
                kind=self.kind,
                fill_value='extrapolate'
            )

            # Apply to all indices
            x_all = np.arange(len(series))
            interpolated_values = spline(x_all)

            return pd.Series(interpolated_values, index=series.index)

        except Exception as e:
            logger.error(f"Cubic spline failed: {e}, falling back to linear")
            return series.interpolate(method='linear')


class SeasonalDecompositionImputer:
    """
    Seasonal decomposition for commodity data with cyclical patterns

    Decomposes time series into:
    - Trend component (long-term movement)
    - Seasonal component (regular cycles)
    - Residual component (noise)

    Missing values are filled using reconstructed components
    """

    def __init__(self, period: int = 30):
        """
        Args:
            period: Seasonal period (e.g., 30 for monthly patterns)
        """
        self.period = period

    def impute(self, series: pd.Series) -> pd.Series:
        """
        Fill NaN using seasonal decomposition

        Mathematical Approach:
        1. Decompose: Y(t) = T(t) + S(t) + R(t)
           where T=trend, S=seasonal, R=residual
        2. Interpolate each component separately
        3. Reconstruct: Y_filled(t) = T_interp(t) + S_interp(t) + R_interp(t)

        Args:
            series: Time series with NaN values

        Returns:
            Imputed series
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Need sufficient data for decomposition
            if len(series.dropna()) < 2 * self.period:
                logger.warning("Insufficient data for seasonal decomposition")
                return series.interpolate(method='linear')

            # First, do basic interpolation to prepare for decomposition
            series_interp = series.interpolate(method='linear')

            # Decompose
            decomposition = seasonal_decompose(
                series_interp,
                model='additive',
                period=self.period,
                extrapolate_trend='freq'
            )

            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Reconstruct (components are already interpolated)
            reconstructed = trend + seasonal + residual

            # Only fill NaN values from original series
            result = series.copy()
            result[series.isna()] = reconstructed[series.isna()]

            return result

        except ImportError:
            logger.warning("statsmodels not installed, using linear interpolation")
            return series.interpolate(method='linear')
        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            return series.interpolate(method='linear')


class ARIMAImputer:
    """
    ARIMA-based imputation for time-aware gap filling

    ARIMA(p,d,q) model:
    - AR(p): Auto-Regressive - use p past values
    - I(d): Integrated - differencing to achieve stationarity
    - MA(q): Moving Average - use q past errors

    Mathematical Model:
    φ(B)(1-B)^d Y_t = θ(B)ε_t

    where:
    φ(B) = 1 - φ_1*B - φ_2*B² - ... - φ_p*B^p  (AR polynomial)
    θ(B) = 1 + θ_1*B + θ_2*B² + ... + θ_q*B^q  (MA polynomial)
    B = backshift operator (B*Y_t = Y_{t-1})
    ε_t = white noise
    """

    def __init__(self, order: tuple = (1, 1, 1)):
        """
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order

    def impute(self, series: pd.Series, max_gap: int = 5) -> pd.Series:
        """
        Fill NaN using ARIMA forecasting

        Args:
            series: Time series with NaN
            max_gap: Maximum gap size to fill with ARIMA

        Returns:
            Imputed series
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            result = series.copy()

            # Find NaN gaps
            is_null = series.isna()

            # Identify contiguous NaN blocks
            null_blocks = []
            start = None
            for i, is_na in enumerate(is_null):
                if is_na and start is None:
                    start = i
                elif not is_na and start is not None:
                    null_blocks.append((start, i))
                    start = None

            # Fill each gap
            for gap_start, gap_end in null_blocks:
                gap_size = gap_end - gap_start

                if gap_size > max_gap:
                    # Gap too large for ARIMA, use linear interpolation
                    continue

                # Get data before gap
                if gap_start < 20:
                    # Need enough history for ARIMA
                    continue

                train_data = series[:gap_start].dropna()

                if len(train_data) < 10:
                    continue

                try:
                    # Fit ARIMA model
                    model = ARIMA(train_data, order=self.order)
                    fitted = model.fit()

                    # Forecast the gap
                    forecast = fitted.forecast(steps=gap_size)

                    # Fill the gap
                    result.iloc[gap_start:gap_end] = forecast.values

                except Exception as e:
                    logger.debug(f"ARIMA failed for gap {gap_start}-{gap_end}: {e}")
                    continue

            # Fill any remaining with linear
            result = result.interpolate(method='linear')

            return result

        except ImportError:
            logger.warning("statsmodels not installed for ARIMA")
            return series.interpolate(method='linear')
        except Exception as e:
            logger.error(f"ARIMA imputation failed: {e}")
            return series.interpolate(method='linear')


def choose_best_interpolation(series: pd.Series, method: str = 'auto') -> pd.Series:
    """
    Intelligently choose the best interpolation method based on data characteristics

    Args:
        series: Time series to interpolate
        method: 'auto', 'cubic', 'seasonal', 'arima', or 'linear'

    Returns:
        Interpolated series
    """
    if method == 'auto':
        # Analyze data to choose best method
        nan_pct = series.isna().sum() / len(series)
        data_length = len(series.dropna())

        if nan_pct > 0.5:
            # Too much missing data, use simple linear
            logger.info("High NaN percentage, using linear interpolation")
            return series.interpolate(method='linear')
        elif data_length < 30:
            # Not enough data for advanced methods
            logger.info("Limited data, using linear interpolation")
            return series.interpolate(method='linear')
        elif nan_pct < 0.1 and data_length > 100:
            # Good data, use cubic spline for smoothness
            logger.info("Using cubic spline interpolation")
            interpolator = CubicSplineInterpolator(kind='cubic')
            return interpolator.interpolate(series)
        else:
            # Default to linear - robust and fast
            return series.interpolate(method='linear')

    elif method == 'cubic':
        interpolator = CubicSplineInterpolator(kind='cubic')
        return interpolator.interpolate(series)

    elif method == 'seasonal':
        imputer = SeasonalDecompositionImputer(period=30)
        return imputer.impute(series)

    elif method == 'arima':
        imputer = ARIMAImputer(order=(1, 1, 1))
        return imputer.impute(series)

    else:  # linear
        return series.interpolate(method='linear')
