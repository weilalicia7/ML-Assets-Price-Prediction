"""
Liquidity Feature Engineering for Chinese Stock Markets

PDF RECOMMENDATION: Liquidity emerged as crucial predictor in Chinese markets.
Including liquidity features prevents models from exploiting untradeable patterns.

Key Liquidity Indicators:
1. Volume-based: Relative volume, volume volatility, volume trends
2. Price-based: Bid-ask spread proxies, price impact estimates
3. Turnover: Share turnover rates, turnover volatility
4. Market depth: Volume at different price levels (approximated)

Reference:
- PDF: "Leverage Liquidity and Transaction Cost Analysis"
- Research shows liquidity crucial for Chinese market predictability
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LiquidityFeatureEngineer:
    """
    Engineer liquidity features for stock price prediction.

    PDF Insight: Liquidity crucial for Chinese markets to avoid untradeable patterns.
    """

    def __init__(self):
        """Initialize liquidity feature engineer."""
        self.required_columns = ['Close', 'Volume', 'High', 'Low']

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all liquidity features to the DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with liquidity features added
        """
        print(f"[INFO] Engineering liquidity features for {len(df)} days...")

        # Validate required columns
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            print(f"  [WARNING] Missing columns for liquidity features: {missing}")
            return df

        try:
            # Volume-based liquidity (6 features)
            df = self._add_volume_liquidity(df)

            # Price-based liquidity (4 features)
            df = self._add_price_liquidity(df)

            # Turnover features (3 features)
            df = self._add_turnover_features(df)

            # Advanced liquidity metrics (4 features)
            df = self._add_advanced_liquidity(df)

            print(f"[OK] Added 17 liquidity features")

        except Exception as e:
            logger.error(f"Error adding liquidity features: {e}")
            print(f"  [ERROR] Failed to add liquidity features: {e}")

        return df

    def _add_volume_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based liquidity indicators.

        PDF Insight: Volume patterns crucial for identifying tradeable vs untradeable signals.
        """
        # 1. Relative Volume (vs 20-day average)
        df['vol_ratio_20'] = df['Volume'] / df['Volume'].rolling(20, min_periods=5).mean()

        # 2. Volume Volatility (20-day std of volume)
        df['vol_volatility_20'] = df['Volume'].rolling(20, min_periods=5).std() / df['Volume'].rolling(20, min_periods=5).mean()

        # 3. Volume Trend (5-day vs 20-day average volume)
        vol_5d_avg = df['Volume'].rolling(5, min_periods=2).mean()
        vol_20d_avg = df['Volume'].rolling(20, min_periods=5).mean()
        df['vol_trend'] = (vol_5d_avg - vol_20d_avg) / vol_20d_avg

        # 4. Volume Acceleration (rate of change in volume)
        df['vol_acceleration'] = df['Volume'].pct_change(5)

        # 5. Average Dollar Volume (proxy for liquidity depth)
        df['dollar_volume'] = df['Close'] * df['Volume']
        df['dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()

        # 6. Volume Z-score (how unusual is current volume?)
        vol_mean = df['Volume'].rolling(60, min_periods=10).mean()
        vol_std = df['Volume'].rolling(60, min_periods=10).std()
        df['volume_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

        return df

    def _add_price_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based liquidity indicators.

        PDF Insight: Bid-ask spread proxies help identify liquidity constraints.
        """
        # 1. High-Low Spread (proxy for bid-ask spread)
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']

        # 2. Average High-Low Spread (20-day)
        df['hl_spread_20d'] = df['hl_spread'].rolling(20, min_periods=5).mean()

        # 3. Price Impact Estimate (Amihud illiquidity measure)
        # |return| / dollar_volume - higher = less liquid
        abs_return = np.abs(df['Close'].pct_change())
        df['amihud_illiq'] = abs_return / (df['dollar_volume'] + 1e-8)
        df['amihud_illiq_20d'] = df['amihud_illiq'].rolling(20, min_periods=5).mean()

        return df

    def _add_turnover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add turnover-based liquidity features.

        Note: We approximate turnover without shares outstanding data.
        """
        # 1. Volume Rate of Change (proxy for turnover changes)
        df['turnover_proxy'] = df['Volume'] / df['Volume'].rolling(60, min_periods=10).mean()

        # 2. Turnover Volatility
        df['turnover_vol'] = df['turnover_proxy'].rolling(20, min_periods=5).std()

        # 3. Turnover Trend
        df['turnover_trend'] = df['turnover_proxy'].rolling(10, min_periods=3).mean() / df['turnover_proxy'].rolling(30, min_periods=5).mean()

        return df

    def _add_advanced_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced liquidity metrics.

        PDF Insight: Combining multiple liquidity proxies improves predictive power.
        """
        # 1. Liquidity Score (composite of volume and spread)
        # Higher = more liquid (normalized 0-1)
        vol_norm = (df['vol_ratio_20'] - df['vol_ratio_20'].rolling(60, min_periods=10).min()) / \
                   (df['vol_ratio_20'].rolling(60, min_periods=10).max() - df['vol_ratio_20'].rolling(60, min_periods=10).min() + 1e-8)
        spread_norm = 1 - ((df['hl_spread'] - df['hl_spread'].rolling(60, min_periods=10).min()) / \
                          (df['hl_spread'].rolling(60, min_periods=10).max() - df['hl_spread'].rolling(60, min_periods=10).min() + 1e-8))
        df['liquidity_score'] = (vol_norm + spread_norm) / 2

        # 2. Liquidity Regime (0=illiquid, 1=normal, 2=very liquid)
        liquidity_rank = df['liquidity_score'].rolling(60, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['liquidity_regime'] = np.where(liquidity_rank < 0.33, 0,
                                         np.where(liquidity_rank < 0.67, 1, 2))

        # 3. Liquidity Stress (sudden drops in liquidity)
        df['liquidity_stress'] = np.where(
            df['liquidity_score'] < df['liquidity_score'].rolling(20, min_periods=5).mean() * 0.7,
            1, 0
        )

        # 4. Days Since Liquidity Event (how long since last high-volume day)
        high_vol_threshold = df['Volume'].rolling(60, min_periods=10).quantile(0.8)
        is_high_vol = (df['Volume'] > high_vol_threshold).astype(int)

        # Count days since last high volume day
        days_since = []
        counter = 0
        for val in is_high_vol:
            if val == 1:
                counter = 0
            else:
                counter += 1
            days_since.append(counter)

        df['days_since_liq_event'] = days_since

        return df

    def get_liquidity_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of liquidity features.

        Args:
            df: DataFrame with liquidity features

        Returns:
            Dictionary with liquidity summary
        """
        summary = {}

        if 'liquidity_score' in df.columns:
            summary['avg_liquidity_score'] = df['liquidity_score'].mean()
            summary['liquidity_volatility'] = df['liquidity_score'].std()

        if 'vol_ratio_20' in df.columns:
            summary['avg_volume_ratio'] = df['vol_ratio_20'].mean()

        if 'hl_spread_20d' in df.columns:
            summary['avg_hl_spread'] = df['hl_spread_20d'].mean()

        if 'amihud_illiq_20d' in df.columns:
            summary['avg_amihud_illiq'] = df['amihud_illiq_20d'].mean()

        if 'liquidity_stress' in df.columns:
            summary['pct_liquidity_stress_days'] = df['liquidity_stress'].mean()

        return summary


# Transaction Cost Model
class TransactionCostModel:
    """
    Model transaction costs for backtesting and predictions.

    PDF RECOMMENDATION: Including transaction costs prevents exploiting untradeable patterns.

    Costs modeled:
    1. Broker commissions
    2. Bid-ask spread
    3. Market impact (slippage)
    4. Liquidity-dependent costs
    """

    # Hong Kong market typical costs
    DEFAULT_COMMISSION_RATE = 0.0008  # 0.08% (HK broker commission)
    DEFAULT_STAMP_DUTY = 0.0013  # 0.13% (HK stamp duty on sales)
    DEFAULT_BASE_SPREAD = 0.0005  # 0.05% base bid-ask spread

    def __init__(self, commission_rate: Optional[float] = None,
                 stamp_duty: Optional[float] = None,
                 base_spread: Optional[float] = None):
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Broker commission as decimal (default: 0.08%)
            stamp_duty: Stamp duty for HK stocks (default: 0.13%)
            base_spread: Base bid-ask spread (default: 0.05%)
        """
        self.commission_rate = commission_rate or self.DEFAULT_COMMISSION_RATE
        self.stamp_duty = stamp_duty or self.DEFAULT_STAMP_DUTY
        self.base_spread = base_spread or self.DEFAULT_BASE_SPREAD

    def calculate_total_cost(self, df: pd.DataFrame, trade_size_pct: float = 0.1) -> pd.Series:
        """
        Calculate total transaction cost for each trade.

        Args:
            df: DataFrame with liquidity features
            trade_size_pct: Trade size as % of average volume (default: 10%)

        Returns:
            Series with total transaction cost per trade (as decimal)
        """
        # Base costs (commission + stamp duty)
        base_cost = self.commission_rate + self.stamp_duty

        # Spread cost (higher when liquidity is low)
        if 'hl_spread_20d' in df.columns:
            spread_cost = df['hl_spread_20d'].clip(lower=self.base_spread)
        else:
            spread_cost = self.base_spread

        # Market impact (increases with trade size and decreases with liquidity)
        if 'liquidity_score' in df.columns:
            # Higher impact when liquidity is low
            impact_multiplier = 1.0 / (df['liquidity_score'] + 0.1)  # Avoid division by zero
            market_impact = trade_size_pct * 0.01 * impact_multiplier  # Base 1% impact per 10% volume
        else:
            market_impact = trade_size_pct * 0.01

        # Total cost
        total_cost = base_cost + spread_cost + market_impact

        return total_cost

    def adjust_prediction_for_costs(self, prediction: float, transaction_cost: float) -> float:
        """
        Adjust predicted return to account for transaction costs.

        PDF Insight: Predictions must exceed transaction costs to be profitable.

        Args:
            prediction: Predicted return (as decimal)
            transaction_cost: Total transaction cost (as decimal)

        Returns:
            Net return after costs
        """
        # For a long trade: buy at cost, sell at cost
        # Total cost is 2x one-way cost
        total_round_trip_cost = transaction_cost * 2

        # Net return = gross return - costs
        net_return = prediction - total_round_trip_cost

        return net_return

    def get_min_profitable_return(self, transaction_cost: float) -> float:
        """
        Calculate minimum return needed to be profitable after costs.

        Args:
            transaction_cost: One-way transaction cost

        Returns:
            Minimum profitable return
        """
        return transaction_cost * 2  # Round-trip cost


if __name__ == '__main__':
    # Test liquidity features
    print("Liquidity Feature Engineering - Test\n")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'High': 100 + np.cumsum(np.random.randn(100) * 2) + 1,
        'Low': 100 + np.cumsum(np.random.randn(100) * 2) - 1,
        'Volume': np.random.randint(1000000, 5000000, 100),
    }, index=dates)

    # Add liquidity features
    engineer = LiquidityFeatureEngineer()
    df_with_liq = engineer.add_all_features(df.copy())

    print("\nLiquidity Features Added:")
    liq_cols = [col for col in df_with_liq.columns if col not in df.columns]
    print(f"  {len(liq_cols)} features: {liq_cols[:5]}...")

    print("\nLiquidity Summary:")
    summary = engineer.get_liquidity_summary(df_with_liq)
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    # Test transaction costs
    print("\n" + "=" * 70)
    print("Transaction Cost Model - Test\n")

    cost_model = TransactionCostModel()

    # Add liquidity score if not present
    if 'liquidity_score' not in df_with_liq.columns:
        df_with_liq['liquidity_score'] = 0.5

    df_with_liq['transaction_cost'] = cost_model.calculate_total_cost(df_with_liq)

    print(f"Average transaction cost: {df_with_liq['transaction_cost'].mean():.4%}")
    print(f"Min transaction cost: {df_with_liq['transaction_cost'].min():.4%}")
    print(f"Max transaction cost: {df_with_liq['transaction_cost'].max():.4%}")

    min_return = cost_model.get_min_profitable_return(df_with_liq['transaction_cost'].mean())
    print(f"\nMinimum profitable return: {min_return:.4%}")

    # Example: Adjust prediction for costs
    prediction = 0.02  # 2% predicted return
    avg_cost = df_with_liq['transaction_cost'].mean()
    net_return = cost_model.adjust_prediction_for_costs(prediction, avg_cost)

    print(f"\nExample Prediction Adjustment:")
    print(f"  Gross predicted return: {prediction:.2%}")
    print(f"  Transaction cost: {avg_cost:.2%}")
    print(f"  Net return after costs: {net_return:.2%}")
    print(f"  Worth trading: {net_return > 0}")
