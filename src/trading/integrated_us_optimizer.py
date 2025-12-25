"""
Integrated US Optimizer - Master Optimizer
==========================================
Combines all profit maximization fixes into a unified optimizer for US/Intl model.

From 'profit test fixing1 full code.pdf':
- Priority 1: EnergySpecificOptimizer (fixes -82% energy loss)
- Priority 2: CommodityDominanceClassifier (fixes -91% mining loss)
- Priority 3: DividendAwareOptimizer (improves financials)

From 'us model fixing4.pdf':
- Priority 4: DynamicTrendAdapter (fixes BUY/SELL asymmetry in downtrends)

Integration Logic:
1. Detect market trend (uptrend/downtrend/sideways)
2. Classify ticker into category (energy, commodity, financial, standard)
3. Route to appropriate specialized optimizer
4. Apply trend-based adjustments (reduces BUY in downtrend, increases SELL)
5. Return unified signal optimization

This is the main entry point for the profit maximization fixes.
US/INTL MODEL ONLY - China stocks are handled by DeepSeek model.

Author: Claude Code
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import logging

# Import specialized optimizers
from .energy_optimizer import (
    EnergySpecificOptimizer,
    EnergySubsector,
    EnergySignalAdjustment,
    create_energy_optimizer,
)
from .commodity_classifier import (
    CommodityDominanceClassifier,
    CommodityDominance,
    CommoditySignalAdjustment,
    create_commodity_classifier,
)
from .dividend_optimizer import (
    DividendAwareOptimizer,
    DividendProfile,
    DividendSignalAdjustment,
    create_dividend_optimizer,
)
from .market_trend_adapter import (
    DynamicTrendAdapter,
    TrendBasedSignalGenerator,
    MarketTrend,
    TrendAdjustedSignal,
    create_trend_adapter,
)
from .rsi_risk_adapter import (
    RSIRiskAdapter,
    RSIRiskLevel,
    RSIEnhancedSignal,
    create_rsi_adapter,
)
from .portfolio_risk_optimizer import (
    PortfolioRiskOptimizer,
    PortfolioPosition,
    PortfolioRiskMetrics,
    create_portfolio_optimizer,
)

logger = logging.getLogger(__name__)


class StockCategory:
    """Stock category classification."""
    ENERGY = "energy"
    COMMODITY = "commodity"  # Mining, metals, etc.
    FINANCIAL = "financial"
    STANDARD = "standard"    # Default equity handling


@dataclass
class IntegratedSignalOptimization:
    """Result from integrated optimizer."""
    ticker: str
    signal_type: str
    category: str
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    should_trade: bool
    adjustment_reason: str

    # Category-specific details
    energy_details: Optional[EnergySignalAdjustment] = None
    commodity_details: Optional[CommoditySignalAdjustment] = None
    dividend_details: Optional[DividendSignalAdjustment] = None
    trend_details: Optional[TrendAdjustedSignal] = None
    rsi_details: Optional[RSIEnhancedSignal] = None

    # Market trend info
    market_trend: str = "unknown"
    trend_multiplier: float = 1.0
    trend_win_rate: float = 0.5

    # RSI risk info (from fixing5.pdf)
    rsi_risk_level: str = "unknown"
    rsi_stop_loss: float = 0.08
    rsi_take_profit: float = 0.10

    # Combined metrics
    final_position_size: float = 1.0
    risk_adjusted_confidence: float = 0.0


class IntegratedUSOptimizer:
    """
    Master optimizer that routes signals to specialized sub-optimizers.

    Classification hierarchy:
    1. Energy stocks -> EnergySpecificOptimizer
    2. Commodity-dominant stocks -> CommodityDominanceClassifier
    3. Financial/dividend stocks -> DividendAwareOptimizer
    4. All others -> Standard processing

    Note: A stock can belong to multiple categories (e.g., energy + dividend).
    In such cases, we apply adjustments from all relevant optimizers.
    """

    def __init__(
        self,
        enable_energy_optimizer: bool = True,
        enable_commodity_classifier: bool = True,
        enable_dividend_optimizer: bool = True,
        enable_futures_analysis: bool = True,
        enable_trend_adapter: bool = True,
        enable_rsi_adapter: bool = True,
    ):
        """
        Initialize integrated optimizer.

        Args:
            enable_energy_optimizer: Enable energy subsector optimization
            enable_commodity_classifier: Enable commodity dominance classification
            enable_dividend_optimizer: Enable dividend-aware optimization
            enable_futures_analysis: Enable futures curve analysis (energy)
            enable_trend_adapter: Enable dynamic market trend adaptation
            enable_rsi_adapter: Enable RSI risk adapter (from fixing5.pdf)
        """
        self.energy_optimizer = None
        self.commodity_classifier = None
        self.dividend_optimizer = None
        self.trend_adapter = None
        self.rsi_adapter = None

        if enable_energy_optimizer:
            self.energy_optimizer = create_energy_optimizer(enable_futures_analysis)
            logger.info("IntegratedUSOptimizer: Energy optimizer enabled")

        if enable_commodity_classifier:
            self.commodity_classifier = create_commodity_classifier()
            logger.info("IntegratedUSOptimizer: Commodity classifier enabled")

        if enable_dividend_optimizer:
            self.dividend_optimizer = create_dividend_optimizer()
            logger.info("IntegratedUSOptimizer: Dividend optimizer enabled")

        if enable_trend_adapter:
            self.trend_adapter = create_trend_adapter()
            logger.info("IntegratedUSOptimizer: Market trend adapter enabled")

        if enable_rsi_adapter:
            self.rsi_adapter = create_rsi_adapter()
            logger.info("IntegratedUSOptimizer: RSI risk adapter enabled (fixing5)")

        # Track statistics
        self.stats = {
            'total_signals': 0,
            'energy_optimized': 0,
            'commodity_optimized': 0,
            'dividend_optimized': 0,
            'trend_adjusted': 0,
            'rsi_adjusted': 0,
            'standard_processed': 0,
            'blocked_signals': 0,
        }

        logger.info("IntegratedUSOptimizer initialized with all profit maximization fixes")

    def classify_stock(self, ticker: str) -> List[str]:
        """
        Classify stock into one or more categories.

        A stock can belong to multiple categories, e.g.:
        - XOM is both ENERGY and has dividend considerations
        - BHP is COMMODITY but may also have dividend

        Args:
            ticker: Stock ticker

        Returns:
            List of applicable categories
        """
        categories = []
        ticker_upper = ticker.upper()

        # Check energy
        if self.energy_optimizer and self.energy_optimizer.is_energy_stock(ticker_upper):
            categories.append(StockCategory.ENERGY)

        # Check commodity (mining, metals)
        if self.commodity_classifier and self.commodity_classifier.is_commodity_stock(ticker_upper):
            categories.append(StockCategory.COMMODITY)

        # Check financial/dividend
        if self.dividend_optimizer and self.dividend_optimizer.is_financial_stock(ticker_upper):
            categories.append(StockCategory.FINANCIAL)

        # Default to standard if no special categories
        if not categories:
            categories.append(StockCategory.STANDARD)

        return categories

    def optimize_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        rate_environment: str = 'neutral',
        rsi: float = 50.0,
    ) -> IntegratedSignalOptimization:
        """
        Optimize a trading signal using all applicable specialized optimizers.

        This is the main entry point for signal optimization.

        Args:
            ticker: Stock ticker (US/INTL only - no China stocks)
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility (annualized)
            momentum: Recent price momentum
            rate_environment: Interest rate trend ('rising', 'falling', 'neutral')
            rsi: RSI value (0-100) for RSI risk adapter (from fixing5.pdf)

        Returns:
            IntegratedSignalOptimization with combined adjustments
        """
        ticker_upper = ticker.upper()
        categories = self.classify_stock(ticker_upper)

        # Track statistics
        self.stats['total_signals'] += 1

        # Initialize with original values
        adjusted_confidence = confidence
        position_multiplier = 1.0
        should_trade = True
        reasons = []

        # Category-specific details
        energy_details = None
        commodity_details = None
        dividend_details = None

        # Primary category determines base adjustments
        primary_category = categories[0]

        # Apply energy optimization
        if StockCategory.ENERGY in categories and self.energy_optimizer:
            energy_result = self.energy_optimizer.optimize_energy_signal(
                ticker=ticker_upper,
                signal_type=signal_type,
                confidence=confidence,
                volatility=volatility,
                momentum=momentum,
            )
            energy_details = energy_result

            if primary_category == StockCategory.ENERGY:
                adjusted_confidence = energy_result.adjusted_confidence
                position_multiplier = energy_result.position_multiplier
                should_trade = energy_result.should_trade
                reasons.append(f"Energy({energy_result.subsector.value}): {energy_result.adjustment_reason}")
                self.stats['energy_optimized'] += 1

        # Apply commodity optimization
        if StockCategory.COMMODITY in categories and self.commodity_classifier:
            commodity_result = self.commodity_classifier.optimize_commodity_signal(
                ticker=ticker_upper,
                signal_type=signal_type,
                confidence=confidence,
                volatility=volatility,
                momentum=momentum,
            )
            commodity_details = commodity_result

            if primary_category == StockCategory.COMMODITY:
                adjusted_confidence = commodity_result.adjusted_confidence
                position_multiplier = commodity_result.position_multiplier
                should_trade = commodity_result.should_trade
                reasons.append(f"Commodity({commodity_result.dominance_class.value}): {commodity_result.adjustment_reason}")
                self.stats['commodity_optimized'] += 1
            elif StockCategory.COMMODITY in categories:
                # Secondary adjustment - blend
                adjusted_confidence = (adjusted_confidence + commodity_result.adjusted_confidence) / 2
                position_multiplier = min(position_multiplier, commodity_result.position_multiplier)
                should_trade = should_trade and commodity_result.should_trade
                reasons.append(f"Commodity overlay: {commodity_result.adjustment_reason}")

        # Apply dividend optimization
        if (StockCategory.FINANCIAL in categories or
            self._should_check_dividends(ticker_upper)) and self.dividend_optimizer:
            dividend_result = self.dividend_optimizer.optimize_dividend_signal(
                ticker=ticker_upper,
                signal_type=signal_type,
                confidence=adjusted_confidence,  # Use already-adjusted confidence
                volatility=volatility,
                momentum=momentum,
                rate_environment=rate_environment,
            )
            dividend_details = dividend_result

            if primary_category == StockCategory.FINANCIAL:
                adjusted_confidence = dividend_result.adjusted_confidence
                position_multiplier = dividend_result.position_multiplier
                should_trade = dividend_result.should_trade
                reasons.append(f"Financial({dividend_result.dividend_profile.value}): {dividend_result.adjustment_reason}")
                self.stats['dividend_optimized'] += 1
            elif StockCategory.FINANCIAL in categories:
                # Secondary adjustment for dividend considerations
                if dividend_result.near_ex_dividend:
                    adjusted_confidence *= 0.95  # Small penalty near ex-div
                    reasons.append(f"Dividend overlay: near ex-div")
                if dividend_result.dividend_profile == DividendProfile.ARISTOCRAT:
                    if signal_type == 'BUY':
                        adjusted_confidence *= 1.05
                        reasons.append("Dividend aristocrat bonus")

        # Standard processing if no special category
        if primary_category == StockCategory.STANDARD:
            self.stats['standard_processed'] += 1
            reasons.append("Standard equity processing")

        # Apply market trend adjustment (Priority 4 from fixing4.pdf)
        # This is the FINAL adjustment layer - adapts BUY/SELL based on market regime
        trend_details = None
        market_trend = "unknown"
        trend_multiplier = 1.0
        trend_win_rate = 0.5

        if self.trend_adapter and should_trade:
            trend_result = self.trend_adapter.adjust_signal_for_trend(
                ticker=ticker_upper,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                position_multiplier=position_multiplier,
            )
            trend_details = trend_result
            market_trend = trend_result.market_trend
            trend_multiplier = trend_result.trend_multiplier
            trend_win_rate = trend_result.trend_win_rate

            # Apply trend adjustments
            adjusted_confidence = trend_result.adjusted_confidence
            position_multiplier = trend_result.position_multiplier

            # Check if trend adapter blocked the signal
            if not trend_result.should_trade:
                should_trade = False
                if trend_result.block_reason:
                    reasons.append(f"Trend blocked: {trend_result.block_reason}")

            # Add trend info to reasons
            reasons.append(f"Trend({market_trend}): mult={trend_multiplier:.2f}, win={trend_win_rate:.0%}")
            self.stats['trend_adjusted'] += 1

        # Apply RSI risk adapter (Priority 5 from fixing5.pdf)
        # This applies RSI-based risk adjustments to confidence, position size, and stop-loss
        rsi_details = None
        rsi_risk_level = "unknown"
        rsi_stop_loss = 0.08
        rsi_take_profit = 0.10

        if self.rsi_adapter and should_trade:
            rsi_result = self.rsi_adapter.enhance_signal_with_rsi(
                ticker=ticker_upper,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                rsi=rsi,
                volatility=volatility,
                position_multiplier=position_multiplier,
                stop_loss_pct=None,  # Use default from adapter
                market_trend=market_trend,
                strict_blocking=False,  # Allow reduced positions instead of blocking
            )
            rsi_details = rsi_result
            rsi_risk_level = rsi_result.rsi_risk_level
            rsi_stop_loss = rsi_result.stop_loss_pct
            rsi_take_profit = rsi_result.take_profit_pct

            # Apply RSI adjustments
            adjusted_confidence = rsi_result.adjusted_confidence
            position_multiplier = rsi_result.position_multiplier

            # Check if RSI adapter blocked the signal
            if not rsi_result.should_trade:
                should_trade = False
                if rsi_result.block_reason:
                    reasons.append(f"RSI blocked: {rsi_result.block_reason}")

            # Add RSI info to reasons
            reasons.append(f"RSI({rsi:.0f}): {rsi_risk_level}, SL={rsi_stop_loss:.1%}, TP={rsi_take_profit:.1%}")
            self.stats['rsi_adjusted'] += 1

        # Final validation
        if not should_trade:
            self.stats['blocked_signals'] += 1

        # Calculate risk-adjusted confidence
        risk_adjusted_confidence = adjusted_confidence * position_multiplier

        # Calculate final position size (relative to base)
        final_position_size = position_multiplier

        # Combine all reasons
        adjustment_reason = " | ".join(reasons) if reasons else "No adjustments"

        return IntegratedSignalOptimization(
            ticker=ticker_upper,
            signal_type=signal_type,
            category=primary_category,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_multiplier,
            should_trade=should_trade,
            adjustment_reason=adjustment_reason,
            energy_details=energy_details,
            commodity_details=commodity_details,
            dividend_details=dividend_details,
            trend_details=trend_details,
            rsi_details=rsi_details,
            market_trend=market_trend,
            trend_multiplier=trend_multiplier,
            trend_win_rate=trend_win_rate,
            rsi_risk_level=rsi_risk_level,
            rsi_stop_loss=rsi_stop_loss,
            rsi_take_profit=rsi_take_profit,
            final_position_size=final_position_size,
            risk_adjusted_confidence=risk_adjusted_confidence,
        )

    def _should_check_dividends(self, ticker: str) -> bool:
        """
        Check if we should apply dividend optimizer even for non-financial stocks.
        High-dividend stocks in any sector benefit from dividend awareness.
        """
        # Could check dividend yield dynamically, but for efficiency,
        # only apply to known dividend-relevant stocks
        return False  # For now, only apply to explicitly financial stocks

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer usage statistics."""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats

        stats = {
            **self.stats,
            'energy_pct': self.stats['energy_optimized'] / total * 100 if total > 0 else 0,
            'commodity_pct': self.stats['commodity_optimized'] / total * 100 if total > 0 else 0,
            'dividend_pct': self.stats['dividend_optimized'] / total * 100 if total > 0 else 0,
            'trend_pct': self.stats['trend_adjusted'] / total * 100 if total > 0 else 0,
            'rsi_pct': self.stats['rsi_adjusted'] / total * 100 if total > 0 else 0,
            'blocked_pct': self.stats['blocked_signals'] / total * 100 if total > 0 else 0,
        }

        # Add current trend info if available
        if self.trend_adapter:
            trend_info = self.trend_adapter.get_current_trend_info()
            stats['current_market_trend'] = trend_info['trend']
            stats['trend_recommendation'] = trend_info['recommendation']
            stats['buy_win_rate'] = trend_info['buy_win_rate']
            stats['sell_win_rate'] = trend_info['sell_win_rate']

        return stats

    def get_category_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked stock categories."""
        summary = {}

        if self.energy_optimizer:
            summary['energy'] = self.energy_optimizer.get_subsector_stats()

        if self.commodity_classifier:
            summary['commodity'] = self.commodity_classifier.get_commodity_stock_summary()

        if self.dividend_optimizer:
            summary['financial'] = self.dividend_optimizer.get_financial_sector_summary()

        return summary

    def reset_statistics(self):
        """Reset usage statistics."""
        self.stats = {
            'total_signals': 0,
            'energy_optimized': 0,
            'commodity_optimized': 0,
            'dividend_optimized': 0,
            'trend_adjusted': 0,
            'rsi_adjusted': 0,
            'standard_processed': 0,
            'blocked_signals': 0,
        }

    def get_current_trend(self) -> Dict[str, Any]:
        """Get current market trend information."""
        if self.trend_adapter:
            return self.trend_adapter.get_current_trend_info()
        return {'trend': 'unknown', 'description': 'Trend adapter disabled'}


# Factory function
def create_integrated_optimizer(
    enable_energy: bool = True,
    enable_commodity: bool = True,
    enable_dividend: bool = True,
    enable_futures: bool = True,
    enable_trend: bool = True,
    enable_rsi: bool = True,
) -> IntegratedUSOptimizer:
    """
    Create an IntegratedUSOptimizer instance.

    Args:
        enable_energy: Enable energy subsector optimization
        enable_commodity: Enable commodity dominance classification
        enable_dividend: Enable dividend-aware optimization
        enable_futures: Enable futures curve analysis
        enable_trend: Enable dynamic market trend adaptation
        enable_rsi: Enable RSI risk adapter (from fixing5.pdf)

    Returns:
        Configured IntegratedUSOptimizer
    """
    return IntegratedUSOptimizer(
        enable_energy_optimizer=enable_energy,
        enable_commodity_classifier=enable_commodity,
        enable_dividend_optimizer=enable_dividend,
        enable_futures_analysis=enable_futures,
        enable_trend_adapter=enable_trend,
        enable_rsi_adapter=enable_rsi,
    )
