"""
Phase 4 Enhancements - Missing Components

Based on: phase4 math fixing on C model.pdf

Implements the 4 missing components to bring Phase 4 to 100%:
1. Real-Time Macro Data Fetcher
2. Macro Feature Importance Analysis
3. Asset-Specific Macro Multipliers
4. Regime Persistence Analysis

Plus:
- Data Quality Validation
- Critical Formula Verification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. REAL-TIME MACRO DATA FETCHER
# =============================================================================

class MacroDataFetcher:
    """
    Real-time macro data fetcher for Phase 4.

    Fetches current values for:
    - VIX: Market volatility/fear gauge
    - UUP/DXY: US Dollar strength
    - SPY: S&P 500 market benchmark
    - TLT: Treasury bonds (interest rates)
    - GLD: Gold (safe haven)
    """

    def __init__(self):
        self.symbols = {
            'VIX': '^VIX',      # CBOE Volatility Index
            'DXY': 'UUP',       # Dollar ETF (proxy for DXY)
            'SPY': 'SPY',       # S&P 500 ETF
            'TLT': 'TLT',       # 20+ Year Treasury ETF
            'GLD': 'GLD'        # Gold ETF
        }
        self.cache = {}
        self.cache_time = None
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

    def fetch_real_time_macro(self, use_cache: bool = True) -> Dict:
        """
        Fetch real-time macro data from Yahoo Finance.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Dict with macro indicator values
        """
        # Check cache
        if use_cache and self.cache and self.cache_time:
            if datetime.now() - self.cache_time < self.cache_duration:
                logger.debug("Using cached macro data")
                return self.cache

        macro_data = {}

        for name, symbol in self.symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                # Get the most recent price
                hist = ticker.history(period='5d')
                if len(hist) > 0:
                    macro_data[name] = {
                        'current': float(hist['Close'].iloc[-1]),
                        'prev_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 else None,
                        'change_pct': float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100) if len(hist) > 1 else 0,
                        'high_5d': float(hist['High'].max()),
                        'low_5d': float(hist['Low'].min()),
                        'timestamp': str(hist.index[-1])
                    }
                    logger.debug(f"Fetched {name}: {macro_data[name]['current']:.2f}")
                else:
                    logger.warning(f"No data for {symbol}")
                    macro_data[name] = None

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                macro_data[name] = None

        # Update cache
        self.cache = macro_data
        self.cache_time = datetime.now()

        return macro_data

    def fetch_macro_history(self, period: str = '3mo') -> pd.DataFrame:
        """
        Fetch historical macro data for feature engineering.

        Args:
            period: History period ('1mo', '3mo', '6mo', '1y', etc.)

        Returns:
            DataFrame with historical macro data
        """
        all_data = {}

        for name, symbol in self.symbols.items():
            try:
                data = yf.download(symbol, period=period, progress=False)
                if len(data) > 0:
                    # Handle MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    all_data[name] = data['Close']
                    logger.info(f"Fetched {len(data)} days of {name} history")
            except Exception as e:
                logger.error(f"Error fetching {symbol} history: {e}")

        if not all_data:
            raise ValueError("Failed to fetch any macro data!")

        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)

        return df

    def get_current_macro_context(self) -> Dict:
        """
        Get current macro context for trading decisions.

        Returns:
            Dict with macro context summary
        """
        macro_data = self.fetch_real_time_macro()

        context = {
            'timestamp': str(datetime.now()),
            'vix_level': macro_data.get('VIX', {}).get('current'),
            'vix_regime': self._classify_vix(macro_data.get('VIX', {}).get('current')),
            'spy_trend': 'up' if macro_data.get('SPY', {}).get('change_pct', 0) > 0 else 'down',
            'gld_trend': 'up' if macro_data.get('GLD', {}).get('change_pct', 0) > 0 else 'down',
            'dxy_trend': 'up' if macro_data.get('DXY', {}).get('change_pct', 0) > 0 else 'down',
            'tlt_trend': 'up' if macro_data.get('TLT', {}).get('change_pct', 0) > 0 else 'down',
        }

        # Quick risk assessment
        risk_on_signals = 0
        if context['spy_trend'] == 'up':
            risk_on_signals += 1
        if context['vix_regime'] in ['low_vol', 'normal']:
            risk_on_signals += 1
        if context['gld_trend'] == 'down':
            risk_on_signals += 1
        if context['tlt_trend'] == 'down':
            risk_on_signals += 1

        context['risk_assessment'] = 'risk_on' if risk_on_signals >= 3 else (
            'risk_off' if risk_on_signals <= 1 else 'neutral'
        )

        return context

    def _classify_vix(self, vix_level: Optional[float]) -> str:
        """Classify VIX into regime."""
        if vix_level is None:
            return 'unknown'
        if vix_level < 15:
            return 'low_vol'
        elif vix_level < 20:
            return 'normal'
        elif vix_level < 30:
            return 'elevated'
        else:
            return 'crisis'

    def classify_current_regime(self, macro_data: Dict) -> Dict:
        """
        Classify the current regime based on macro data.

        Args:
            macro_data: Dict with macro indicator values (VIX, SPY, etc.)

        Returns:
            Dict with regime classification
        """
        # Handle both nested dict format and flat format
        vix_val = None
        if 'VIX' in macro_data:
            if isinstance(macro_data['VIX'], dict):
                vix_val = macro_data['VIX'].get('current')
            else:
                vix_val = macro_data['VIX']

        vix_regime = self._classify_vix(vix_val)

        # Determine risk regime based on macro indicators
        risk_signals = 0

        # SPY trend
        spy_data = macro_data.get('SPY', {})
        if isinstance(spy_data, dict):
            if spy_data.get('change_pct', 0) > 0:
                risk_signals += 1

        # GLD inverse (gold up = risk off)
        gld_data = macro_data.get('GLD', {})
        if isinstance(gld_data, dict):
            if gld_data.get('change_pct', 0) < 0:
                risk_signals += 1

        # VIX (low VIX = risk on)
        if vix_regime in ['low_vol', 'normal']:
            risk_signals += 1

        # TLT inverse (bonds up = risk off)
        tlt_data = macro_data.get('TLT', {})
        if isinstance(tlt_data, dict):
            if tlt_data.get('change_pct', 0) < 0:
                risk_signals += 1

        # Classify risk regime
        if risk_signals >= 3:
            risk_regime = 'risk_on'
        elif risk_signals <= 1:
            risk_regime = 'risk_off'
        else:
            risk_regime = 'neutral'

        return {
            'vix_regime': vix_regime,
            'vix_level': vix_val,
            'risk_regime': risk_regime,
            'risk_score': (risk_signals - 2) / 2,  # -1 to +1 scale
            'correlation_breakdown': False  # Would need more data to determine
        }


# =============================================================================
# 2. MACRO FEATURE IMPORTANCE ANALYSIS
# =============================================================================

class MacroFeatureAnalyzer:
    """
    Analyze which macro features are most predictive of returns.
    """

    def __init__(self):
        self.feature_importance = {}
        self.feature_stability = {}

    def analyze_macro_feature_importance(
        self,
        features_df: pd.DataFrame,
        target_returns: pd.Series,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Analyze which macro features are most predictive.

        Args:
            features_df: DataFrame with macro features
            target_returns: Series of forward returns
            top_n: Number of top features to return

        Returns:
            Dict mapping feature name to importance score
        """
        feature_importance = {}

        # Identify macro features
        macro_prefixes = ('vix_', 'spy_', 'dxy_', 'tlt_', 'gld_', 'VIX', 'SPY',
                         'DXY', 'TLT', 'GLD', 'risk_', 'corr_', 'beta_', 'rel_strength')

        for feature in features_df.columns:
            if any(feature.startswith(prefix) or feature == prefix for prefix in macro_prefixes):
                try:
                    # Calculate correlation with forward returns
                    valid_mask = ~(features_df[feature].isna() | target_returns.isna())
                    if valid_mask.sum() > 30:  # Minimum samples
                        correlation = np.corrcoef(
                            features_df[feature][valid_mask].values,
                            target_returns[valid_mask].values
                        )[0, 1]

                        if not np.isnan(correlation):
                            feature_importance[feature] = abs(correlation)
                except Exception as e:
                    logger.debug(f"Error analyzing {feature}: {e}")

        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        self.feature_importance = sorted_importance
        return sorted_importance

    def analyze_feature_stability(
        self,
        features_df: pd.DataFrame,
        target_returns: pd.Series,
        window: int = 60
    ) -> Dict[str, Dict]:
        """
        Analyze how stable feature importance is over time.

        Args:
            features_df: DataFrame with macro features
            target_returns: Series of forward returns
            window: Rolling window for stability analysis

        Returns:
            Dict with stability metrics per feature
        """
        stability_results = {}

        for feature in self.feature_importance.keys():
            if feature not in features_df.columns:
                continue

            # Calculate rolling correlations
            rolling_corr = features_df[feature].rolling(window).corr(target_returns)

            stability_results[feature] = {
                'mean_correlation': rolling_corr.mean(),
                'std_correlation': rolling_corr.std(),
                'stability_score': 1 - rolling_corr.std(),  # Higher = more stable
                'sign_consistency': (rolling_corr > 0).mean(),  # Fraction positive
                'current_correlation': rolling_corr.iloc[-1] if len(rolling_corr) > 0 else None
            }

        self.feature_stability = stability_results
        return stability_results

    def get_recommended_features(self, min_importance: float = 0.05) -> List[str]:
        """
        Get list of recommended features based on importance and stability.

        Args:
            min_importance: Minimum importance threshold

        Returns:
            List of recommended feature names
        """
        recommended = []

        for feature, importance in self.feature_importance.items():
            if importance >= min_importance:
                stability = self.feature_stability.get(feature, {})
                stability_score = stability.get('stability_score', 0)

                # Recommend if important AND stable
                if stability_score > 0.5:
                    recommended.append(feature)

        return recommended


# =============================================================================
# 3. ASSET-SPECIFIC MACRO MULTIPLIERS
# =============================================================================

class AssetSpecificMacroMultiplier:
    """
    Different assets respond differently to macro conditions.

    This class provides asset-class-specific adjustments to the
    base macro multiplier.
    """

    def __init__(self):
        # Asset class sensitivity to macro conditions
        # Higher = more sensitive to risk-on/off
        self.asset_sensitivities = {
            'equity': 1.0,          # Standard response
            'tech': 1.2,            # More volatile
            'crypto': 1.5,          # Most sensitive to risk-on/off
            'bonds': 0.5,           # Inverse, safe haven
            'gold': 0.6,            # Safe haven, inverse sometimes
            'commodity': 1.1,       # Moderate sensitivity
            'forex': 0.8,           # Lower sensitivity
            'international': 1.1,   # Slightly more sensitive (FX exposure)
            'etf': 0.9,             # Diversified, less sensitive
            'reit': 1.0,            # Standard
        }

        # VIX sensitivity by asset class
        self.vix_sensitivity = {
            'equity': 1.0,
            'tech': 1.3,
            'crypto': 1.5,
            'bonds': -0.5,   # Negative = inverse relationship
            'gold': -0.3,
            'commodity': 0.8,
            'forex': 0.5,
            'international': 1.2,
            'etf': 0.9,
            'reit': 1.1,
        }

        # DXY sensitivity (dollar strength impact)
        self.dxy_sensitivity = {
            'equity': -0.3,         # Strong dollar hurts earnings
            'tech': -0.4,           # Global revenue exposure
            'crypto': -0.5,         # Inverse to dollar
            'bonds': 0.2,
            'gold': -0.8,           # Strong inverse to dollar
            'commodity': -0.6,      # Dollar-denominated
            'forex': 1.0,           # Direct impact
            'international': -0.7,  # FX translation effects
            'etf': -0.2,
            'reit': -0.2,
        }

    def get_asset_specific_multiplier(
        self,
        base_multiplier: float,
        asset_class: str,
        macro_state: Dict
    ) -> float:
        """
        Get asset-specific macro multiplier.

        Args:
            base_multiplier: Base macro multiplier from Phase 4
            asset_class: Asset class identifier
            macro_state: Current macro state dict

        Returns:
            Adjusted multiplier for this asset class
        """
        # Get base sensitivity
        sensitivity = self.asset_sensitivities.get(asset_class.lower(), 1.0)

        # Start with base multiplier
        adjusted_mult = base_multiplier

        # Apply sensitivity scaling
        # If base_mult < 1 (risk reduction), higher sensitivity = more reduction
        # If base_mult > 1 (risk increase), higher sensitivity = more increase
        if base_multiplier < 1.0:
            # Risk reduction scenario
            reduction = 1.0 - base_multiplier
            adjusted_reduction = reduction * sensitivity
            adjusted_mult = 1.0 - adjusted_reduction
        else:
            # Risk increase scenario
            increase = base_multiplier - 1.0
            adjusted_increase = increase * sensitivity
            adjusted_mult = 1.0 + adjusted_increase

        # Apply VIX-specific adjustment
        vix_level = macro_state.get('vix_level')
        if vix_level is not None:
            vix_sens = self.vix_sensitivity.get(asset_class.lower(), 1.0)

            # High VIX impact
            if vix_level > 25:
                vix_impact = (vix_level - 25) / 100 * vix_sens
                adjusted_mult *= (1 - vix_impact)

        # Clamp to valid range
        adjusted_mult = max(0.0, min(1.5, adjusted_mult))

        return adjusted_mult

    def get_all_asset_multipliers(
        self,
        base_multiplier: float,
        macro_state: Dict
    ) -> Dict[str, float]:
        """
        Get multipliers for all asset classes.

        Args:
            base_multiplier: Base macro multiplier
            macro_state: Current macro state

        Returns:
            Dict mapping asset class to multiplier
        """
        return {
            asset_class: self.get_asset_specific_multiplier(
                base_multiplier, asset_class, macro_state
            )
            for asset_class in self.asset_sensitivities.keys()
        }

    def get_asset_specific_macro_multiplier(
        self,
        asset_class: str,
        macro_state: Dict,
        base_multiplier: float = 0.7
    ) -> float:
        """
        Convenience method matching PDF test signature.

        Args:
            asset_class: Asset class identifier
            macro_state: Current macro state
            base_multiplier: Base multiplier (default 0.7 for risk-off scenario)

        Returns:
            Adjusted multiplier for this asset class
        """
        return self.get_asset_specific_multiplier(base_multiplier, asset_class, macro_state)


# =============================================================================
# 4. REGIME PERSISTENCE ANALYSIS
# =============================================================================

class RegimePersistenceAnalyzer:
    """
    Analyze how long risk regimes typically last.

    This helps in:
    - Predicting regime transitions
    - Setting appropriate position holding periods
    - Calibrating regime-based strategies
    """

    def __init__(self):
        self.regime_statistics = {}

    def calculate_regime_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how long risk regimes typically last.

        Args:
            df: DataFrame with 'risk_regime' column

        Returns:
            DataFrame with regime duration statistics
        """
        if 'risk_regime' not in df.columns:
            raise ValueError("DataFrame must have 'risk_regime' column")

        # Identify regime changes
        regime_changes = df['risk_regime'].ne(df['risk_regime'].shift())
        regime_periods = regime_changes.cumsum()

        # Calculate durations and statistics per period
        regime_durations = df.groupby(regime_periods).agg({
            'risk_regime': 'first',
            'risk_regime_score': ['mean', 'std', 'min', 'max'] if 'risk_regime_score' in df.columns else 'first'
        })

        # Flatten column names if MultiIndex
        if isinstance(regime_durations.columns, pd.MultiIndex):
            regime_durations.columns = ['_'.join(col).strip('_') for col in regime_durations.columns]

        # Add duration
        regime_durations['duration'] = df.groupby(regime_periods).size()

        # Add start and end dates
        regime_durations['start_date'] = df.groupby(regime_periods).apply(lambda x: x.index[0])
        regime_durations['end_date'] = df.groupby(regime_periods).apply(lambda x: x.index[-1])

        return regime_durations

    def get_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for each regime type.

        Args:
            df: DataFrame with risk_regime column

        Returns:
            Dict with statistics per regime
        """
        regime_durations = self.calculate_regime_persistence(df)

        # Handle different column name possibilities
        regime_col = 'risk_regime_first' if 'risk_regime_first' in regime_durations.columns else 'risk_regime'

        statistics = {}
        for regime in regime_durations[regime_col].unique():
            regime_data = regime_durations[regime_durations[regime_col] == regime]

            statistics[regime] = {
                'count': len(regime_data),
                'avg_duration': regime_data['duration'].mean(),
                'std_duration': regime_data['duration'].std(),
                'min_duration': regime_data['duration'].min(),
                'max_duration': regime_data['duration'].max(),
                'median_duration': regime_data['duration'].median(),
                'total_days': regime_data['duration'].sum(),
                'pct_of_time': regime_data['duration'].sum() / df.shape[0] * 100
            }

        self.regime_statistics = statistics
        return statistics

    def predict_regime_transition_probability(
        self,
        df: pd.DataFrame,
        current_regime: str,
        days_in_regime: int
    ) -> Dict[str, float]:
        """
        Estimate probability of regime transition based on historical patterns.

        Args:
            df: Historical data with risk_regime
            current_regime: Current regime name
            days_in_regime: How long we've been in current regime

        Returns:
            Dict with transition probabilities
        """
        if not self.regime_statistics:
            self.get_regime_statistics(df)

        regime_stats = self.regime_statistics.get(current_regime, {})
        avg_duration = regime_stats.get('avg_duration', 20)
        std_duration = regime_stats.get('std_duration', 10)

        # Simple probability model based on duration
        # Probability increases as we exceed average duration
        if days_in_regime < avg_duration - std_duration:
            transition_prob = 0.1  # Low probability
        elif days_in_regime < avg_duration:
            transition_prob = 0.3  # Moderate
        elif days_in_regime < avg_duration + std_duration:
            transition_prob = 0.5  # Higher
        else:
            transition_prob = 0.7  # Very high

        return {
            'transition_probability': transition_prob,
            'days_in_regime': days_in_regime,
            'avg_regime_duration': avg_duration,
            'regime_remaining_estimate': max(0, avg_duration - days_in_regime)
        }

    def predict_regime_transition(
        self,
        df: pd.DataFrame,
        current_regime: str
    ) -> float:
        """
        Simplified method that returns just the transition probability.

        Args:
            df: Historical data with risk_regime
            current_regime: Current regime name

        Returns:
            Transition probability (0-1)
        """
        # Calculate current days in regime
        if 'risk_regime' not in df.columns:
            return 0.5

        # Find how long we've been in the current regime
        regime_changes = df['risk_regime'].ne(df['risk_regime'].shift())
        regime_periods = regime_changes.cumsum()

        # Get the duration of the current regime period
        current_period = regime_periods.iloc[-1]
        days_in_regime = (regime_periods == current_period).sum()

        result = self.predict_regime_transition_probability(df, current_regime, days_in_regime)
        return result['transition_probability']


# =============================================================================
# 5. DATA QUALITY VALIDATION
# =============================================================================

def validate_macro_data_quality(df: pd.DataFrame) -> Dict:
    """
    Ensure macro data is valid before calculations.

    Args:
        df: DataFrame with macro data

    Returns:
        Dict with validation results
    """
    checks = {}
    required_cols = ['VIX', 'SPY']  # Minimum required
    optional_cols = ['GLD', 'TLT', 'DXY']

    # Check required columns
    for col in required_cols:
        if col not in df.columns:
            checks[col] = 'MISSING'
        elif df[col].isnull().all():
            checks[col] = 'ALL_NULL'
        elif df[col].isnull().sum() / len(df) > 0.1:
            checks[col] = f'HIGH_NULL ({df[col].isnull().sum() / len(df) * 100:.1f}%)'
        else:
            checks[col] = 'OK'

    # Check optional columns
    for col in optional_cols:
        if col not in df.columns:
            checks[col] = 'MISSING (optional)'
        elif df[col].isnull().all():
            checks[col] = 'ALL_NULL'
        else:
            checks[col] = 'OK'

    # Check data ranges
    if 'VIX' in df.columns and checks.get('VIX') == 'OK':
        if df['VIX'].min() < 0 or df['VIX'].max() > 100:
            checks['VIX_range'] = f'SUSPICIOUS ({df["VIX"].min():.1f}-{df["VIX"].max():.1f})'
        else:
            checks['VIX_range'] = 'OK'

    # Overall status
    checks['overall'] = 'PASS' if all(
        v in ['OK', 'MISSING (optional)'] for v in checks.values()
    ) else 'FAIL'

    return checks


# =============================================================================
# 6. CRITICAL FORMULA VERIFICATION
# =============================================================================

def verify_critical_formulas(df: pd.DataFrame) -> Dict:
    """
    Verify key Phase 4 calculations are correct.

    Args:
        df: DataFrame with Phase 4 features

    Returns:
        Dict with verification results
    """
    checks = {}

    # 1. Beta calculation check
    if 'Close' in df.columns and 'SPY' in df.columns:
        try:
            stock_vol = df['Close'].pct_change().rolling(20).std()
            spy_vol = df['SPY'].pct_change().rolling(20).std()
            corr = df['Close'].pct_change().rolling(20).corr(df['SPY'].pct_change())
            calculated_beta = corr * (stock_vol / spy_vol)

            checks['beta_calculation'] = 'OK' if not calculated_beta.isnull().all() else 'FAIL'
            checks['beta_range'] = f'{calculated_beta.min():.2f} to {calculated_beta.max():.2f}'
        except Exception as e:
            checks['beta_calculation'] = f'ERROR: {e}'
    else:
        checks['beta_calculation'] = 'SKIP (missing columns)'

    # 2. Risk score range check
    if 'risk_regime_score' in df.columns:
        min_score = df['risk_regime_score'].min()
        max_score = df['risk_regime_score'].max()
        checks['risk_score_range'] = 'OK' if (min_score >= -1.0 and max_score <= 1.0) else f'FAIL ({min_score:.2f} to {max_score:.2f})'
    else:
        checks['risk_score_range'] = 'SKIP (missing column)'

    # 3. Multiplier bounds check
    if 'regime_position_mult' in df.columns:
        min_mult = df['regime_position_mult'].min()
        max_mult = df['regime_position_mult'].max()
        checks['multiplier_bounds'] = 'OK' if (min_mult >= 0.0 and max_mult <= 1.5) else f'FAIL ({min_mult:.2f} to {max_mult:.2f})'
    else:
        checks['multiplier_bounds'] = 'SKIP (missing column)'

    # 4. VIX regime classification
    if 'vix_low' in df.columns:
        # Check one-hot encoding sums to 1 (or 0 if all NaN)
        vix_cols = ['vix_low', 'vix_normal', 'vix_elevated', 'vix_crisis']
        existing_cols = [c for c in vix_cols if c in df.columns]
        if len(existing_cols) == 4:
            vix_sum = df[existing_cols].sum(axis=1)
            checks['vix_regimes'] = 'OK' if (vix_sum.dropna() == 1).all() else 'FAIL (not one-hot)'
        else:
            checks['vix_regimes'] = f'PARTIAL ({len(existing_cols)}/4 columns)'
    else:
        checks['vix_regimes'] = 'SKIP (missing columns)'

    # 5. Correlation features check
    corr_cols = [c for c in df.columns if 'corr_' in c.lower()]
    if corr_cols:
        # Correlations should be between -1 and 1
        corr_valid = all(
            df[col].dropna().between(-1.01, 1.01).all()
            for col in corr_cols
        )
        checks['correlation_bounds'] = 'OK' if corr_valid else 'FAIL (out of range)'
    else:
        checks['correlation_bounds'] = 'SKIP (no correlation columns)'

    # Overall status
    failures = [k for k, v in checks.items() if isinstance(v, str) and 'FAIL' in v]
    checks['overall'] = 'PASS' if not failures else f'FAIL ({len(failures)} issues)'

    return checks


# =============================================================================
# 7. PHASE 4 IMPROVEMENT VALIDATOR
# =============================================================================

def validate_phase4_improvement(
    phase3_results: Dict,
    phase4_results: Dict
) -> Dict:
    """
    Validate Phase 4 delivers expected +5-8% improvement.

    Args:
        phase3_results: Dict with Phase 3 backtest results
        phase4_results: Dict with Phase 4 backtest results

    Returns:
        Dict with improvement metrics
    """
    improvement_metrics = {}

    # Profit rate improvement
    profit_improvement = phase4_results.get('profit_rate', 0) - phase3_results.get('profit_rate', 0)
    improvement_metrics['profit_rate_improvement'] = profit_improvement
    improvement_metrics['profit_rate_improvement_pct'] = f"{profit_improvement * 100:.1f}%"

    # Risk-adjusted improvement (Sharpe)
    sharpe_improvement = phase4_results.get('sharpe_ratio', 0) - phase3_results.get('sharpe_ratio', 0)
    improvement_metrics['sharpe_improvement'] = sharpe_improvement

    # Drawdown improvement (positive = better)
    drawdown_improvement = phase3_results.get('max_drawdown', 0) - phase4_results.get('max_drawdown', 0)
    improvement_metrics['drawdown_improvement'] = drawdown_improvement

    # Success criteria
    improvement_metrics['meets_profit_target'] = profit_improvement >= 0.05  # +5% minimum
    improvement_metrics['meets_drawdown_target'] = drawdown_improvement >= 0  # No worse drawdown
    improvement_metrics['meets_sharpe_target'] = sharpe_improvement >= 0  # Better risk-adjusted

    improvement_metrics['overall_success'] = (
        improvement_metrics['meets_profit_target'] and
        improvement_metrics['meets_drawdown_target'] and
        improvement_metrics['meets_sharpe_target']
    )

    # Alias for PDF test compatibility
    improvement_metrics['meets_target'] = improvement_metrics['overall_success']

    return improvement_metrics


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Test Phase 4 enhancements."""
    print("="*70)
    print("PHASE 4 ENHANCEMENTS TEST")
    print("="*70)

    # Test 1: Real-Time Macro Data Fetcher
    print("\n[1/6] Testing MacroDataFetcher...")
    try:
        fetcher = MacroDataFetcher()
        context = fetcher.get_current_macro_context()
        print(f"  VIX Level: {context['vix_level']}")
        print(f"  VIX Regime: {context['vix_regime']}")
        print(f"  Risk Assessment: {context['risk_assessment']}")
        print("  [OK] MacroDataFetcher works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 2: Asset-Specific Multipliers
    print("\n[2/6] Testing AssetSpecificMacroMultiplier...")
    try:
        asset_mult = AssetSpecificMacroMultiplier()
        macro_state = {'vix_level': 22}

        print("  Base multiplier: 0.7")
        for asset_class in ['equity', 'crypto', 'bonds', 'gold']:
            mult = asset_mult.get_asset_specific_multiplier(0.7, asset_class, macro_state)
            print(f"    {asset_class}: {mult:.3f}")
        print("  [OK] AssetSpecificMacroMultiplier works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 3: Data Quality Validation
    print("\n[3/6] Testing validate_macro_data_quality...")
    try:
        test_df = pd.DataFrame({
            'VIX': [18, 19, 20, 21, 22],
            'SPY': [450, 451, 452, 453, 454],
            'GLD': [180, 181, 182, 183, 184]
        })
        checks = validate_macro_data_quality(test_df)
        print(f"  VIX: {checks['VIX']}")
        print(f"  SPY: {checks['SPY']}")
        print(f"  Overall: {checks['overall']}")
        print("  [OK] Data quality validation works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 4: Formula Verification
    print("\n[4/6] Testing verify_critical_formulas...")
    try:
        test_df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'SPY': np.random.randn(100).cumsum() + 450,
            'risk_regime_score': np.random.uniform(-0.5, 0.5, 100),
            'regime_position_mult': np.random.uniform(0.3, 1.2, 100)
        })
        checks = verify_critical_formulas(test_df)
        print(f"  Beta calculation: {checks['beta_calculation']}")
        print(f"  Risk score range: {checks['risk_score_range']}")
        print(f"  Overall: {checks['overall']}")
        print("  [OK] Formula verification works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 5: Phase 4 Improvement Validator
    print("\n[5/6] Testing validate_phase4_improvement...")
    try:
        phase3 = {'profit_rate': 0.45, 'sharpe_ratio': 1.2, 'max_drawdown': 0.12}
        phase4 = {'profit_rate': 0.52, 'sharpe_ratio': 1.4, 'max_drawdown': 0.10}

        improvement = validate_phase4_improvement(phase3, phase4)
        print(f"  Profit improvement: {improvement['profit_rate_improvement_pct']}")
        print(f"  Sharpe improvement: {improvement['sharpe_improvement']:.2f}")
        print(f"  Meets target: {improvement['overall_success']}")
        print("  [OK] Improvement validator works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 6: Feature Analyzer
    print("\n[6/6] Testing MacroFeatureAnalyzer...")
    try:
        analyzer = MacroFeatureAnalyzer()
        test_features = pd.DataFrame({
            'vix_momentum_20d': np.random.randn(100),
            'spy_momentum_20d': np.random.randn(100),
            'beta_spy_20d': np.random.randn(100),
            'other_feature': np.random.randn(100)
        })
        target = pd.Series(np.random.randn(100))

        importance = analyzer.analyze_macro_feature_importance(test_features, target)
        print(f"  Top features: {list(importance.keys())[:3]}")
        print("  [OK] Feature analyzer works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "="*70)
    print("PHASE 4 ENHANCEMENTS TEST COMPLETE")
    print("="*70)


# =============================================================================
# IMPORT ELITE ENHANCEMENTS INTO THIS MODULE
# =============================================================================
# Make elite classes available from this module to match PDF test reference

from features.phase4_elite_enhancements import (
    AdaptiveMacroFeatureSelector,
    MacroAwareEnsembleWeighter,
    DynamicMacroSensitivity,
    MacroRegimeForecaster,
    MacroFeatureCompressor,
    MacroImpactMonitor,
    MacroAwarePositionSizer,
    MultiTimeframeMacroAnalyzer,
    Phase4EliteSystem
)

# Add compress_macro_features method to MacroFeatureCompressor for PDF compatibility
# This is handled in the elite enhancements file via fit_transform

__all__ = [
    # Base Phase 4 classes
    'MacroDataFetcher',
    'MacroFeatureAnalyzer',
    'AssetSpecificMacroMultiplier',
    'RegimePersistenceAnalyzer',
    'validate_macro_data_quality',
    'verify_critical_formulas',
    'validate_phase4_improvement',
    # Elite classes (re-exported)
    'AdaptiveMacroFeatureSelector',
    'MacroAwareEnsembleWeighter',
    'DynamicMacroSensitivity',
    'MacroRegimeForecaster',
    'MacroFeatureCompressor',
    'MacroImpactMonitor',
    'MacroAwarePositionSizer',
    'MultiTimeframeMacroAnalyzer',
    'Phase4EliteSystem'
]


if __name__ == "__main__":
    main()
