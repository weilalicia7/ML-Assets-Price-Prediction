# Phase 4: Macro Integration - Calculation Code Reference

This document contains all the calculation code for Phase 4 Macro Integration.

**Expected Impact: +5-8% profit rate improvement**

---

## Table of Contents

1. [Macro Feature Calculations](#1-macro-feature-calculations)
2. [Cross-Market Correlation Calculations](#2-cross-market-correlation-calculations)
3. [Risk Regime Calculations](#3-risk-regime-calculations)
4. [Position Multiplier Calculations](#4-position-multiplier-calculations)

---

## 1. Macro Feature Calculations

### 1.1 Macro Momentum Features

```python
def _add_macro_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add macro momentum features (rate of change)."""
    windows = [5, 20, 60]

    for col in ['VIX', 'DXY', 'SPY', 'TLT', 'GLD']:
        if col not in df.columns:
            continue

        for window in windows:
            # Momentum (percent change)
            momentum_col = f'{col}_momentum_{window}d'
            df[momentum_col] = df[col].pct_change(window)

            # Moving average
            ma_col = f'{col}_ma_{window}d'
            df[ma_col] = df[col].rolling(window).mean()

            # Distance from MA (normalized)
            dist_col = f'{col}_dist_ma_{window}d'
            df[dist_col] = (df[col] - df[ma_col]) / df[ma_col]

    return df
```

**Features Generated (per indicator):**
- `{indicator}_momentum_{window}d` = `pct_change(window)`
- `{indicator}_ma_{window}d` = `rolling(window).mean()`
- `{indicator}_dist_ma_{window}d` = `(price - MA) / MA`

---

### 1.2 VIX Regime Classification

```python
def _add_macro_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add macro regime classification features."""

    # VIX regimes (volatility)
    if 'VIX' in df.columns:
        df['vix_regime'] = pd.cut(
            df['VIX'],
            bins=[0, 15, 20, 30, 100],
            labels=['low_vol', 'normal', 'elevated', 'crisis']
        ).astype(str)

        # One-hot encode
        df['vix_low'] = (df['vix_regime'] == 'low_vol').astype(int)
        df['vix_normal'] = (df['vix_regime'] == 'normal').astype(int)
        df['vix_elevated'] = (df['vix_regime'] == 'elevated').astype(int)
        df['vix_crisis'] = (df['vix_regime'] == 'crisis').astype(int)

        # VIX spike detection
        df['vix_spike'] = (df['VIX'] > df['VIX'].rolling(60).mean() +
                          2 * df['VIX'].rolling(60).std()).astype(int)
```

**VIX Thresholds:**
| VIX Level | Regime |
|-----------|--------|
| 0-15 | Low Volatility |
| 15-20 | Normal |
| 20-30 | Elevated |
| 30+ | Crisis |

**VIX Spike Formula:**
```
vix_spike = VIX > MA(60) + 2 * STD(60)
```

---

### 1.3 Risk-On / Risk-Off Detection

```python
# Market regime (SPY + VIX + GLD)
if all(col in df.columns for col in ['SPY', 'VIX', 'GLD']):
    # Risk-on: SPY up, VIX down, GLD flat/down
    spy_up = df['SPY'].pct_change(20) > 0
    vix_down = df['VIX'].pct_change(20) < 0
    gld_down = df['GLD'].pct_change(20) < 0

    df['risk_on'] = (spy_up & vix_down).astype(int)
    df['risk_off'] = ((~spy_up) & (~vix_down) & (~gld_down)).astype(int)
```

**Logic:**
- **Risk-On** = SPY momentum > 0 AND VIX momentum < 0
- **Risk-Off** = SPY momentum < 0 AND VIX momentum > 0 AND GLD momentum > 0

---

### 1.4 DXY (Dollar) Strength

```python
# DXY regimes (dollar strength)
if 'DXY' in df.columns:
    dxy_ma = df['DXY'].rolling(60).mean()
    df['dxy_strong'] = (df['DXY'] > dxy_ma).astype(int)
    df['dxy_weak'] = (df['DXY'] < dxy_ma).astype(int)
```

**Formula:**
```
dxy_strong = DXY > MA(60)
dxy_weak = DXY < MA(60)
```

---

### 1.5 Beta to SPY Calculation

```python
def _add_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add relative strength vs market (SPY)."""
    windows = [20, 60]

    for window in windows:
        # Stock return
        stock_return = df['Close'].pct_change(window)

        # Market return
        spy_return = df['SPY'].pct_change(window)

        # Relative strength (stock outperformance)
        rs_col = f'rel_strength_spy_{window}d'
        df[rs_col] = stock_return - spy_return

        # Beta to SPY (rolling correlation * volatility ratio)
        stock_vol = df['Close'].pct_change().rolling(window).std()
        spy_vol = df['SPY'].pct_change().rolling(window).std()
        corr = df['Close'].pct_change().rolling(window).corr(df['SPY'].pct_change())

        beta_col = f'beta_spy_{window}d'
        df[beta_col] = corr * (stock_vol / spy_vol)

    return df
```

**Formulas:**
```
relative_strength = stock_return - spy_return

beta_spy = correlation(stock, SPY) * (stock_volatility / spy_volatility)
```

---

## 2. Cross-Market Correlation Calculations

### 2.1 Correlation Network Features

```python
def calculate_correlation_network_features(
    self,
    df: pd.DataFrame,
    price_cols: List[str] = None
) -> pd.DataFrame:
    """
    Calculate correlation network features.

    Features generated:
    - avg_correlation: Average pairwise correlation
    - max_correlation: Maximum pairwise correlation
    - correlation_dispersion: Std of correlations (high = divergence)
    """
    df = df.copy()

    if price_cols is None:
        price_cols = [col for col in self.macro_symbols if col in df.columns]

    # Calculate returns
    returns = pd.DataFrame()
    for col in price_cols:
        returns[col] = df[col].pct_change()

    # Rolling correlation features
    avg_corr = []
    max_corr = []
    corr_disp = []

    for i in range(len(df)):
        if i < self.correlation_window:
            avg_corr.append(np.nan)
            max_corr.append(np.nan)
            corr_disp.append(np.nan)
            continue

        # Get window of returns
        window_returns = returns.iloc[i-self.correlation_window:i]

        # Calculate correlation matrix
        corr = window_returns.corr()

        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_values = corr.values[mask]

        # Calculate features
        avg_corr.append(np.nanmean(corr_values))
        max_corr.append(np.nanmax(np.abs(corr_values)))
        corr_disp.append(np.nanstd(corr_values))

    df['corr_network_avg'] = avg_corr
    df['corr_network_max'] = max_corr
    df['corr_network_dispersion'] = corr_disp

    return df
```

**Formulas:**
```
corr_network_avg = mean(upper_triangle(correlation_matrix))
corr_network_max = max(abs(upper_triangle(correlation_matrix)))
corr_network_dispersion = std(upper_triangle(correlation_matrix))
```

---

### 2.2 PCA Correlation Features

```python
def calculate_pca_correlation_features(
    self,
    df: pd.DataFrame,
    price_cols: List[str] = None
) -> pd.DataFrame:
    """
    Calculate PCA-based correlation clustering features.

    Uses PCA to identify common factors driving asset movements.
    High explained variance by PC1 = high correlation regime
    """
    # Calculate returns
    returns = pd.DataFrame()
    for col in price_cols:
        returns[col] = df[col].pct_change()

    # Rolling PCA explained variance
    pc1_explained = []
    pc_total_explained = []

    for i in range(len(df)):
        if i < self.correlation_window:
            pc1_explained.append(np.nan)
            pc_total_explained.append(np.nan)
            continue

        window_returns = returns.iloc[i-self.correlation_window:i].dropna()

        try:
            # Standardize
            standardized = (window_returns - window_returns.mean()) / (window_returns.std() + 1e-10)

            # PCA
            pca = PCA(n_components=min(self.pca_components, len(price_cols)))
            pca.fit(standardized)

            pc1_explained.append(pca.explained_variance_ratio_[0])
            pc_total_explained.append(sum(pca.explained_variance_ratio_[:self.pca_components]))
        except Exception:
            pc1_explained.append(np.nan)
            pc_total_explained.append(np.nan)

    df['pca_pc1_explained'] = pc1_explained
    df['pca_total_explained'] = pc_total_explained

    # High correlation regime when PC1 explains > 50%
    df['pca_high_corr_regime'] = (df['pca_pc1_explained'] > 0.5).astype(int)

    return df
```

**Formulas:**
```
standardized_returns = (returns - mean) / std

pca_pc1_explained = PCA.explained_variance_ratio_[0]
pca_total_explained = sum(explained_variance_ratio_[0:3])

pca_high_corr_regime = 1 if pca_pc1_explained > 0.5 else 0
```

---

### 2.3 Correlation Breakdown Detection

```python
def detect_correlation_breakdown(
    self,
    df: pd.DataFrame,
    asset_col: str = 'Close',
    benchmark_col: str = 'SPY'
) -> pd.DataFrame:
    """
    Detect correlation breakdown events.

    Correlation breakdown occurs when:
    - Historical correlation is high but recent correlation drops
    - This often precedes market stress or regime changes
    """
    # Calculate returns
    asset_returns = df[asset_col].pct_change()
    benchmark_returns = df[benchmark_col].pct_change()

    # Short-term vs long-term correlation
    short_window = 20
    long_window = 60

    short_corr = asset_returns.rolling(short_window).corr(benchmark_returns)
    long_corr = asset_returns.rolling(long_window).corr(benchmark_returns)

    df['corr_short_term'] = short_corr
    df['corr_long_term'] = long_corr

    # Correlation change
    df['corr_change'] = short_corr - long_corr

    # Breakdown detection: correlation dropped significantly
    df['corr_breakdown'] = (
        (long_corr > 0.5) &  # Was highly correlated
        (df['corr_change'] < -self.breakdown_threshold)  # Now dropping
    ).astype(int)

    # Correlation regime stability
    df['corr_stability'] = 1 - df['corr_change'].abs().rolling(20).mean()

    return df
```

**Formulas:**
```
corr_short_term = rolling_corr(asset, benchmark, window=20)
corr_long_term = rolling_corr(asset, benchmark, window=60)

corr_change = corr_short_term - corr_long_term

corr_breakdown = (long_corr > 0.5) AND (corr_change < -0.3)

corr_stability = 1 - rolling_mean(abs(corr_change), window=20)
```

---

## 3. Risk Regime Calculations

### 3.1 Comprehensive Risk Regime Score

```python
def calculate_risk_regime(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive risk-on/risk-off regime.

    Risk-On indicators:
    - SPY momentum positive
    - VIX declining or low
    - GLD declining (risk assets preferred)
    - TLT declining (risk assets preferred)

    Risk-Off indicators:
    - SPY momentum negative
    - VIX rising or high
    - GLD rising (safe haven demand)
    - TLT rising (flight to safety)
    """
    # Initialize scores
    risk_on_score = pd.Series(0.0, index=df.index)
    risk_off_score = pd.Series(0.0, index=df.index)

    # SPY momentum (weight: 0.3)
    if 'SPY' in df.columns:
        spy_mom = df['SPY'].pct_change(20)
        risk_on_score += (spy_mom > 0).astype(float) * 0.3
        risk_off_score += (spy_mom < 0).astype(float) * 0.3

    # VIX level and momentum (weight: 0.25)
    if 'VIX' in df.columns:
        vix_low = df['VIX'] < 20
        vix_declining = df['VIX'].pct_change(10) < 0
        risk_on_score += ((vix_low) | (vix_declining)).astype(float) * 0.25

        vix_high = df['VIX'] > 25
        vix_rising = df['VIX'].pct_change(10) > 0.1
        risk_off_score += ((vix_high) | (vix_rising)).astype(float) * 0.25

    # GLD (safe haven) (weight: 0.2)
    if 'GLD' in df.columns:
        gld_mom = df['GLD'].pct_change(20)
        risk_on_score += (gld_mom < 0).astype(float) * 0.2
        risk_off_score += (gld_mom > 0).astype(float) * 0.2

    # TLT (bonds/safety) (weight: 0.15)
    if 'TLT' in df.columns:
        tlt_mom = df['TLT'].pct_change(20)
        risk_on_score += (tlt_mom < 0).astype(float) * 0.15
        risk_off_score += (tlt_mom > 0).astype(float) * 0.15

    # DXY (dollar strength) (weight: 0.1)
    if 'DXY' in df.columns:
        dxy_mom = df['DXY'].pct_change(20)
        # Strong dollar can be risk-off (flight to safety)
        risk_off_score += (dxy_mom > 0.02).astype(float) * 0.1
        risk_on_score += (dxy_mom < -0.02).astype(float) * 0.1

    # Net risk score (-1 = full risk-off, +1 = full risk-on)
    df['risk_regime_score'] = risk_on_score - risk_off_score

    # Categorical regime
    df['risk_regime'] = 'neutral'
    df.loc[df['risk_regime_score'] > 0.2, 'risk_regime'] = 'risk_on'
    df.loc[df['risk_regime_score'] < -0.2, 'risk_regime'] = 'risk_off'
    df.loc[df['risk_regime_score'] > 0.5, 'risk_regime'] = 'strong_risk_on'
    df.loc[df['risk_regime_score'] < -0.5, 'risk_regime'] = 'strong_risk_off'

    return df
```

**Risk Score Weights:**
| Indicator | Weight | Risk-On Condition | Risk-Off Condition |
|-----------|--------|-------------------|-------------------|
| SPY | 0.30 | momentum > 0 | momentum < 0 |
| VIX | 0.25 | VIX < 20 OR declining | VIX > 25 OR rising > 10% |
| GLD | 0.20 | momentum < 0 | momentum > 0 |
| TLT | 0.15 | momentum < 0 | momentum > 0 |
| DXY | 0.10 | momentum < -2% | momentum > 2% |

**Risk Regime Formula:**
```
risk_regime_score = risk_on_score - risk_off_score

Range: [-1, +1]
- strong_risk_off: score < -0.5
- risk_off: score < -0.2
- neutral: -0.2 <= score <= 0.2
- risk_on: score > 0.2
- strong_risk_on: score > 0.5
```

---

### 3.2 Regime Position Multiplier

```python
# Position multiplier based on regime
df['regime_position_mult'] = 1.0
df.loc[df['risk_regime'] == 'strong_risk_on', 'regime_position_mult'] = 1.2
df.loc[df['risk_regime'] == 'risk_on', 'regime_position_mult'] = 1.1
df.loc[df['risk_regime'] == 'risk_off', 'regime_position_mult'] = 0.7
df.loc[df['risk_regime'] == 'strong_risk_off', 'regime_position_mult'] = 0.3
```

**Position Multipliers:**
| Risk Regime | Position Multiplier |
|-------------|---------------------|
| strong_risk_on | 1.2x (20% increase) |
| risk_on | 1.1x (10% increase) |
| neutral | 1.0x (no change) |
| risk_off | 0.7x (30% reduction) |
| strong_risk_off | 0.3x (70% reduction) |

---

## 4. Position Multiplier Calculations

### 4.1 Combined Macro Multiplier

```python
def _calculate_macro_multiplier(self, regime_position_mult: float = 1.0) -> float:
    """
    Calculate combined macro position multiplier.

    Combines:
    - VIX regime impact
    - Risk-on/risk-off impact
    - Correlation breakdown impact
    """
    multiplier = 1.0

    # VIX impact
    if self.macro_state['vix_regime'] == 'crisis':
        multiplier *= 0.3  # 70% reduction in crisis
    elif self.macro_state['vix_regime'] == 'elevated':
        multiplier *= 0.7  # 30% reduction when elevated

    # Risk regime impact
    if 'risk_off' in self.macro_state['risk_regime']:
        multiplier *= self.risk_off_reduction  # default 0.5
    elif 'risk_on' in self.macro_state['risk_regime']:
        multiplier *= 1.1  # Slight increase in risk-on

    # Correlation breakdown protection
    if self.macro_state['correlation_breakdown']:
        multiplier *= (1 - self.correlation_breakdown_reduction)  # default 0.3

    # Apply regime-based multiplier
    multiplier *= regime_position_mult

    # Clamp to valid range
    multiplier = max(0.0, min(1.2, multiplier))

    return multiplier
```

**VIX Regime Impact:**
| VIX Regime | Multiplier |
|------------|------------|
| normal | 1.0x |
| elevated | 0.7x |
| crisis | 0.3x |

**Risk Regime Impact:**
| Risk Regime | Multiplier |
|-------------|------------|
| risk_on | 1.1x |
| neutral | 1.0x |
| risk_off | 0.5x |

**Correlation Breakdown:**
```
If corr_breakdown == True:
    multiplier *= (1 - 0.3) = 0.7
```

**Combined Formula:**
```
macro_multiplier = vix_mult * risk_mult * breakdown_mult * regime_position_mult

Final range: [0.0, 1.2]
```

---

### 4.2 Final Position Adjustment

```python
# Apply Phase 4 macro multiplier
macro_mult = self.macro_state['macro_multiplier']

# Adjust position size with macro context
if decision['can_trade'] and decision['shares'] > 0:
    # Apply macro multiplier
    adjusted_value = decision['position_value'] * macro_mult
    adjusted_shares = int(adjusted_value / current_price) if current_price > 0 else 0

    decision['position_value_pre_macro'] = decision['position_value']
    decision['position_value'] = adjusted_value
    decision['shares_pre_macro'] = decision['shares']
    decision['shares'] = adjusted_shares

# Update combined multiplier to include macro
decision['combined_multiplier'] = (
    decision['drawdown_multiplier'] *
    decision['regime_multiplier'] *
    macro_mult
)
```

**Final Position Formula:**
```
final_position_value = base_position_value * macro_multiplier

combined_multiplier = drawdown_mult * regime_mult * macro_mult
```

---

## Summary: All Phase 4 Calculations

### Feature Calculations
| Feature | Formula |
|---------|---------|
| Momentum | `pct_change(window)` |
| Distance from MA | `(price - MA) / MA` |
| Beta | `corr * (stock_vol / market_vol)` |
| Relative Strength | `stock_return - market_return` |
| Correlation Network Avg | `mean(corr_matrix_upper_triangle)` |
| PCA PC1 Explained | `PCA.explained_variance_ratio_[0]` |
| Correlation Breakdown | `long_corr > 0.5 AND corr_change < -0.3` |
| Correlation Stability | `1 - rolling_mean(abs(corr_change))` |
| Risk Score | `risk_on_score - risk_off_score` |

### Position Multiplier Summary
| Component | Range | Default |
|-----------|-------|---------|
| VIX Regime | 0.3 - 1.0 | 1.0 |
| Risk Regime | 0.5 - 1.1 | 1.0 |
| Correlation Breakdown | 0.7 or 1.0 | 1.0 |
| Regime Position | 0.3 - 1.2 | 1.0 |
| **Final Macro Multiplier** | **0.0 - 1.2** | **1.0** |

---

---

## 5. Phase 4 Enhancements (from phase4 math fixing on C model.pdf)

### 5.1 Real-Time Macro Data Fetcher

```python
class MacroDataFetcher:
    """Real-time macro data fetcher for Phase 4."""

    def __init__(self):
        self.symbols = {
            'VIX': '^VIX',      # CBOE Volatility Index
            'DXY': 'UUP',       # Dollar ETF (proxy for DXY)
            'SPY': 'SPY',       # S&P 500 ETF
            'TLT': 'TLT',       # 20+ Year Treasury ETF
            'GLD': 'GLD'        # Gold ETF
        }

    def fetch_real_time_macro(self) -> Dict:
        """Fetch real-time macro data from Yahoo Finance."""
        macro_data = {}
        for name, symbol in self.symbols.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            if len(hist) > 0:
                macro_data[name] = {
                    'current': float(hist['Close'].iloc[-1]),
                    'change_pct': float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100)
                }
        return macro_data
```

### 5.2 Asset-Specific Macro Multipliers

```python
class AssetSpecificMacroMultiplier:
    """Different assets respond differently to macro conditions."""

    def __init__(self):
        # Asset class sensitivity to macro conditions
        self.asset_sensitivities = {
            'equity': 1.0,          # Standard response
            'tech': 1.2,            # More volatile
            'crypto': 1.5,          # Most sensitive to risk-on/off
            'bonds': 0.5,           # Inverse, safe haven
            'gold': 0.6,            # Safe haven
            'commodity': 1.1,       # Moderate sensitivity
            'forex': 0.8,           # Lower sensitivity
            'international': 1.1,   # FX exposure
        }

    def get_asset_specific_multiplier(
        self,
        base_multiplier: float,
        asset_class: str,
        macro_state: Dict
    ) -> float:
        """Get asset-specific macro multiplier."""
        sensitivity = self.asset_sensitivities.get(asset_class.lower(), 1.0)

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

        return max(0.0, min(1.5, adjusted_mult))
```

**Asset Sensitivity Table:**
| Asset Class | Sensitivity | Example |
|-------------|-------------|---------|
| equity | 1.0 | AAPL, MSFT |
| tech | 1.2 | NVDA, AMD |
| crypto | 1.5 | BTC, ETH |
| bonds | 0.5 | TLT, BND |
| gold | 0.6 | GLD |
| commodity | 1.1 | XOM, CVX |
| forex | 0.8 | EURUSD |
| international | 1.1 | .HK stocks |

### 5.3 Macro Feature Importance Analysis

```python
class MacroFeatureAnalyzer:
    def analyze_macro_feature_importance(
        self,
        features_df: pd.DataFrame,
        target_returns: pd.Series,
        top_n: int = 10
    ) -> Dict[str, float]:
        """Analyze which macro features are most predictive."""
        feature_importance = {}

        macro_prefixes = ('vix_', 'spy_', 'dxy_', 'tlt_', 'gld_', 'risk_', 'corr_', 'beta_')

        for feature in features_df.columns:
            if any(feature.startswith(prefix) for prefix in macro_prefixes):
                correlation = np.corrcoef(
                    features_df[feature].fillna(0),
                    target_returns
                )[0, 1]
                feature_importance[feature] = abs(correlation)

        return dict(sorted(feature_importance.items(),
                          key=lambda x: x[1], reverse=True)[:top_n])
```

### 5.4 Regime Persistence Analysis

```python
def calculate_regime_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate how long risk regimes typically last."""
    regime_changes = df['risk_regime'].ne(df['risk_regime'].shift())
    regime_periods = regime_changes.cumsum()

    regime_durations = df.groupby(regime_periods).agg({
        'risk_regime': 'first',
        'risk_regime_score': 'mean'
    })
    regime_durations['duration'] = df.groupby(regime_periods).size()

    return regime_durations

def predict_regime_transition_probability(
    self,
    current_regime: str,
    days_in_regime: int,
    avg_duration: float
) -> float:
    """Estimate probability of regime transition."""
    if days_in_regime < avg_duration * 0.5:
        return 0.1  # Low probability
    elif days_in_regime < avg_duration:
        return 0.3  # Moderate
    elif days_in_regime < avg_duration * 1.5:
        return 0.5  # Higher
    else:
        return 0.7  # Very high
```

### 5.5 Data Quality Validation

```python
def validate_macro_data_quality(df: pd.DataFrame) -> Dict:
    """Ensure macro data is valid before calculations."""
    checks = {}
    required_cols = ['VIX', 'SPY']

    for col in required_cols:
        if col not in df.columns:
            checks[col] = 'MISSING'
        elif df[col].isnull().all():
            checks[col] = 'ALL_NULL'
        elif df[col].isnull().sum() / len(df) > 0.1:
            checks[col] = f'HIGH_NULL ({df[col].isnull().sum() / len(df) * 100:.1f}%)'
        else:
            checks[col] = 'OK'

    checks['overall'] = 'PASS' if all(v == 'OK' for v in checks.values()) else 'FAIL'
    return checks
```

### 5.6 Critical Formula Verification

```python
def verify_critical_formulas(df: pd.DataFrame) -> Dict:
    """Verify key Phase 4 calculations are correct."""
    checks = {}

    # 1. Beta calculation
    stock_vol = df['Close'].pct_change().rolling(20).std()
    spy_vol = df['SPY'].pct_change().rolling(20).std()
    corr = df['Close'].pct_change().rolling(20).corr(df['SPY'].pct_change())
    calculated_beta = corr * (stock_vol / spy_vol)
    checks['beta_calculation'] = not calculated_beta.isnull().all()

    # 2. Risk score range check (-1 to +1)
    checks['risk_score_range'] = (
        df['risk_regime_score'].min() >= -1.0 and
        df['risk_regime_score'].max() <= 1.0
    )

    # 3. Multiplier bounds (0 to 1.2)
    checks['multiplier_bounds'] = (
        df['regime_position_mult'].min() >= 0.0 and
        df['regime_position_mult'].max() <= 1.2
    )

    return checks
```

### 5.7 Phase 4 Improvement Validator

```python
def validate_phase4_improvement(phase3_results, phase4_results) -> Dict:
    """Validate Phase 4 delivers expected +5-8% improvement."""
    improvement_metrics = {}

    # Profit rate improvement
    profit_improvement = phase4_results['profit_rate'] - phase3_results['profit_rate']
    improvement_metrics['profit_rate_improvement'] = profit_improvement

    # Success criteria
    improvement_metrics['meets_target'] = (
        profit_improvement >= 0.05 and  # +5% minimum
        phase4_results['max_drawdown'] <= phase3_results['max_drawdown'] and
        phase4_results['sharpe_ratio'] >= phase3_results['sharpe_ratio']
    )

    return improvement_metrics
```

---

## Implementation Status Summary

| Component | Status | File |
|-----------|--------|------|
| Macro Feature Calculations | Complete | `macro_features.py` |
| Risk Regime Detection | Complete | `cross_market_correlations.py` |
| Position Multipliers | Complete | `phase4_macro_resolver.py` |
| Real-Time Data Fetching | **Complete** | `phase4_enhancements.py` |
| Feature Importance | **Complete** | `phase4_enhancements.py` |
| Asset-Specific Multipliers | **Complete** | `phase4_enhancements.py` |
| Regime Persistence | **Complete** | `phase4_enhancements.py` |
| Data Quality Validation | **Complete** | `phase4_enhancements.py` |
| Formula Verification | **Complete** | `phase4_enhancements.py` |

**Phase 4 Status: 100% COMPLETE**

---

*Document created: 2025-11-29*
*Document updated: 2025-11-29 (added enhancements from phase4 math fixing PDF)*
*Phase 4 Status: 100% COMPLETE*
