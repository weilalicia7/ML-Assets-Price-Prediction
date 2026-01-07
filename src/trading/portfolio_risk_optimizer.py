"""
Portfolio Risk Optimizer - Enhanced Portfolio Management with Risk Controls
============================================================================
From 'us model fixing5.pdf':
- Portfolio-level risk management
- Sector concentration limits
- Correlation-based diversification
- Maximum drawdown controls
- Position sizing based on portfolio risk

This module provides:
1. Portfolio risk assessment
2. Position sizing optimization
3. Sector/asset class limits
4. Correlation-aware position management
5. Drawdown-based position scaling

Date: 2025-12-16
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Portfolio risk category."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class SectorType(Enum):
    """Stock sector classification."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    ENERGY = "energy"
    CONSUMER = "consumer"
    INDUSTRIAL = "industrial"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FOREX = "forex"
    OTHER = "other"


@dataclass
class PortfolioPosition:
    """Represents a position in the portfolio."""
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    sector: SectorType
    position_type: str  # 'LONG' or 'SHORT'
    entry_date: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        if self.position_type == 'LONG':
            return (self.current_price - self.entry_price) * self.shares
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.shares

    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0
        if self.position_type == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics."""
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    positions_count: int
    long_count: int
    short_count: int
    sector_concentration: Dict[str, float]
    max_position_pct: float
    max_sector_pct: float
    risk_score: float  # 0-100
    risk_category: str
    warnings: List[str]
    recommendations: List[str]


class PortfolioRiskOptimizer:
    """
    Portfolio-level risk management and optimization.

    Key Features:
    1. Sector Concentration Limits - Max 30% in any single sector
    2. Position Size Limits - Max 15% in any single position
    3. Correlation-Aware Management - Reduce correlated positions
    4. Drawdown Controls - Scale positions based on recent drawdown
    5. Risk-Adjusted Position Sizing - Based on portfolio risk score
    """

    def __init__(
        self,
        max_position_pct: float = 0.15,       # Max 15% in single position
        max_sector_pct: float = 0.30,          # Max 30% in single sector
        max_correlated_exposure: float = 0.40,  # Max 40% in correlated assets
        drawdown_reduce_threshold: float = 0.10,  # Reduce at 10% drawdown
        drawdown_stop_threshold: float = 0.20,    # Stop trading at 20% drawdown
        risk_profile: RiskCategory = RiskCategory.MODERATE,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_correlated_exposure = max_correlated_exposure
        self.drawdown_reduce_threshold = drawdown_reduce_threshold
        self.drawdown_stop_threshold = drawdown_stop_threshold
        self.risk_profile = risk_profile

        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.initial_capital = 100000.0
        self.current_capital = 100000.0
        self.peak_value = 100000.0
        self.current_drawdown = 0.0

        # Sector mappings for common tickers
        self.sector_map = {
            # Technology
            'AAPL': SectorType.TECHNOLOGY, 'MSFT': SectorType.TECHNOLOGY,
            'GOOGL': SectorType.TECHNOLOGY, 'GOOG': SectorType.TECHNOLOGY,
            'META': SectorType.TECHNOLOGY, 'NVDA': SectorType.TECHNOLOGY,
            'AMD': SectorType.TECHNOLOGY, 'INTC': SectorType.TECHNOLOGY,
            'AFRM': SectorType.TECHNOLOGY, 'WIX': SectorType.TECHNOLOGY,
            'CRCL': SectorType.TECHNOLOGY, 'TSLA': SectorType.TECHNOLOGY,
            # Healthcare
            'PFE': SectorType.HEALTHCARE, 'MRK': SectorType.HEALTHCARE,
            'JNJ': SectorType.HEALTHCARE, 'UNH': SectorType.HEALTHCARE,
            'MIRM': SectorType.HEALTHCARE,
            # Financial
            'JPM': SectorType.FINANCIAL, 'BAC': SectorType.FINANCIAL,
            'GS': SectorType.FINANCIAL, 'MS': SectorType.FINANCIAL,
            # Energy
            'XOM': SectorType.ENERGY, 'CVX': SectorType.ENERGY,
            'CL=F': SectorType.COMMODITY, 'NG=F': SectorType.COMMODITY,
            # Consumer
            'AMZN': SectorType.CONSUMER, 'WMT': SectorType.CONSUMER,
            'HD': SectorType.CONSUMER, 'MCD': SectorType.CONSUMER,
            # Industrial
            'CAT': SectorType.INDUSTRIAL, 'BA': SectorType.INDUSTRIAL,
            'GE': SectorType.INDUSTRIAL, 'HON': SectorType.INDUSTRIAL,
            'AAL': SectorType.INDUSTRIAL, 'CCL': SectorType.CONSUMER,
            'UBER': SectorType.TECHNOLOGY, 'F': SectorType.CONSUMER,
            # Crypto
            'BTC-USD': SectorType.CRYPTO, 'ETH-USD': SectorType.CRYPTO,
            'BNB-USD': SectorType.CRYPTO, 'XRP-USD': SectorType.CRYPTO,
            'DOGE-USD': SectorType.CRYPTO, 'SOL-USD': SectorType.CRYPTO,
            'ADA-USD': SectorType.CRYPTO, 'AVAX-USD': SectorType.CRYPTO,
            # Commodities
            'GC=F': SectorType.COMMODITY, 'SI=F': SectorType.COMMODITY,
            # Forex
            'EUR=X': SectorType.FOREX, 'GBP=X': SectorType.FOREX,
            'JPY=X': SectorType.FOREX,
        }

        # Correlation groups (assets that tend to move together)
        self.correlation_groups = {
            'big_tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD'],
            'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD'],
            'financials': ['JPM', 'BAC', 'GS', 'MS'],
            'energy': ['XOM', 'CVX', 'CL=F'],
            'airlines_travel': ['AAL', 'CCL', 'UBER'],
        }

        # Risk profile adjustments
        self.risk_adjustments = {
            RiskCategory.CONSERVATIVE: {
                'max_position_mult': 0.7,
                'max_sector_mult': 0.8,
                'drawdown_threshold_mult': 0.8,
            },
            RiskCategory.MODERATE: {
                'max_position_mult': 1.0,
                'max_sector_mult': 1.0,
                'drawdown_threshold_mult': 1.0,
            },
            RiskCategory.AGGRESSIVE: {
                'max_position_mult': 1.3,
                'max_sector_mult': 1.2,
                'drawdown_threshold_mult': 1.3,
            },
        }

        # Statistics
        self.stats = {
            'signals_evaluated': 0,
            'signals_approved': 0,
            'signals_reduced': 0,
            'signals_blocked': 0,
            'sector_limit_hits': 0,
            'position_limit_hits': 0,
            'correlation_limit_hits': 0,
            'drawdown_reductions': 0,
        }

        logger.info(f"PortfolioRiskOptimizer initialized with {risk_profile.value} profile")

    def get_sector(self, ticker: str) -> SectorType:
        """Get sector for a ticker."""
        ticker_upper = ticker.upper()

        # Check direct mapping
        if ticker_upper in self.sector_map:
            return self.sector_map[ticker_upper]

        # Detect by pattern
        if '-USD' in ticker_upper:
            return SectorType.CRYPTO
        elif '=F' in ticker_upper:
            return SectorType.COMMODITY
        elif '=X' in ticker_upper:
            return SectorType.FOREX

        return SectorType.OTHER

    def get_correlation_group(self, ticker: str) -> Optional[str]:
        """Get correlation group for a ticker."""
        ticker_upper = ticker.upper()
        for group, tickers in self.correlation_groups.items():
            if ticker_upper in tickers:
                return group
        return None

    def set_capital(self, initial: float, current: float):
        """Set portfolio capital levels."""
        self.initial_capital = initial
        self.current_capital = current
        self.peak_value = max(self.peak_value, current)
        self.current_drawdown = (self.peak_value - current) / self.peak_value if self.peak_value > 0 else 0

    def add_position(self, position: PortfolioPosition):
        """Add a position to the portfolio."""
        self.positions[position.ticker] = position

    def remove_position(self, ticker: str):
        """Remove a position from the portfolio."""
        if ticker in self.positions:
            del self.positions[ticker]

    def clear_positions(self):
        """Clear all positions."""
        self.positions.clear()

    def get_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure as percentage of portfolio."""
        total_value = sum(abs(p.market_value) for p in self.positions.values())
        if total_value == 0:
            return {}

        sector_values = defaultdict(float)
        for p in self.positions.values():
            sector_values[p.sector.value] += abs(p.market_value)

        return {sector: value / total_value for sector, value in sector_values.items()}

    def get_correlation_exposure(self, ticker: str) -> float:
        """Calculate exposure to tickers correlated with the given ticker."""
        group = self.get_correlation_group(ticker)
        if not group:
            return 0

        total_value = sum(abs(p.market_value) for p in self.positions.values())
        if total_value == 0:
            return 0

        correlated_value = 0
        group_tickers = self.correlation_groups[group]
        for p in self.positions.values():
            if p.ticker.upper() in group_tickers:
                correlated_value += abs(p.market_value)

        return correlated_value / total_value

    def get_position_exposure(self, ticker: str) -> float:
        """Get current exposure to a specific ticker."""
        total_value = sum(abs(p.market_value) for p in self.positions.values())
        if total_value == 0:
            return 0

        if ticker in self.positions:
            return abs(self.positions[ticker].market_value) / total_value
        return 0

    def calculate_max_position_size(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        base_position_pct: float = 0.10,
    ) -> Tuple[float, List[str]]:
        """
        Calculate maximum allowed position size based on portfolio constraints.

        Returns:
            Tuple of (max_position_pct, list_of_reasons)
        """
        self.stats['signals_evaluated'] += 1
        warnings = []

        # Get risk-adjusted limits
        adj = self.risk_adjustments[self.risk_profile]
        effective_max_position = self.max_position_pct * adj['max_position_mult']
        effective_max_sector = self.max_sector_pct * adj['max_sector_mult']

        # Start with base position size
        max_position = base_position_pct

        # 1. Check sector concentration
        sector = self.get_sector(ticker)
        sector_exposure = self.get_sector_exposure()
        current_sector_pct = sector_exposure.get(sector.value, 0)

        if current_sector_pct >= effective_max_sector:
            max_position = 0
            warnings.append(f"Sector limit reached: {sector.value} at {current_sector_pct*100:.1f}%")
            self.stats['sector_limit_hits'] += 1
        elif current_sector_pct + base_position_pct > effective_max_sector:
            max_position = effective_max_sector - current_sector_pct
            warnings.append(f"Position reduced due to sector limit: {sector.value}")
            self.stats['signals_reduced'] += 1

        # 2. Check position concentration
        current_position_pct = self.get_position_exposure(ticker)
        if current_position_pct >= effective_max_position:
            max_position = 0
            warnings.append(f"Position limit reached for {ticker}")
            self.stats['position_limit_hits'] += 1
        elif current_position_pct + base_position_pct > effective_max_position:
            max_position = min(max_position, effective_max_position - current_position_pct)
            warnings.append(f"Position capped at {effective_max_position*100:.1f}%")

        # 3. Check correlation exposure
        corr_exposure = self.get_correlation_exposure(ticker)
        if corr_exposure >= self.max_correlated_exposure:
            max_position = min(max_position, 0.05)  # Cap at 5%
            warnings.append(f"High correlation exposure: {corr_exposure*100:.1f}%")
            self.stats['correlation_limit_hits'] += 1

        # 4. Apply drawdown-based scaling
        if self.current_drawdown >= self.drawdown_stop_threshold:
            max_position = 0
            warnings.append(f"Trading halted: drawdown at {self.current_drawdown*100:.1f}%")
            self.stats['drawdown_reductions'] += 1
        elif self.current_drawdown >= self.drawdown_reduce_threshold:
            reduction = 1 - (self.current_drawdown - self.drawdown_reduce_threshold) / \
                       (self.drawdown_stop_threshold - self.drawdown_reduce_threshold)
            max_position *= reduction
            warnings.append(f"Position reduced due to {self.current_drawdown*100:.1f}% drawdown")
            self.stats['drawdown_reductions'] += 1

        # 5. Confidence-based adjustment
        if confidence < 0.5:
            max_position *= 0.7
        elif confidence > 0.8:
            max_position *= 1.2

        # Ensure non-negative
        max_position = max(0, min(max_position, effective_max_position))

        # Track approvals
        if max_position > 0:
            self.stats['signals_approved'] += 1
        else:
            self.stats['signals_blocked'] += 1

        return max_position, warnings

    def evaluate_new_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        price: float,
        rsi: float = 50.0,
        volatility: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate a new trading signal against portfolio constraints.

        Returns:
            Dictionary with evaluation results and recommendations
        """
        max_position, warnings = self.calculate_max_position_size(
            ticker, signal_type, confidence
        )

        sector = self.get_sector(ticker)
        corr_group = self.get_correlation_group(ticker)

        # Build recommendations
        recommendations = []

        if max_position == 0:
            recommendations.append("Signal blocked by portfolio constraints")
        elif max_position < 0.05:
            recommendations.append("Consider smaller position due to concentration risk")

        if corr_group:
            recommendations.append(f"Correlated with {corr_group} positions")

        # RSI-based recommendation
        if signal_type == 'BUY' and rsi > 70:
            recommendations.append("Caution: RSI indicates overbought")
        elif signal_type == 'SELL' and rsi < 30:
            recommendations.append("Caution: RSI indicates oversold")

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'approved': max_position > 0,
            'max_position_pct': max_position,
            'max_position_value': max_position * self.current_capital,
            'sector': sector.value,
            'correlation_group': corr_group,
            'current_drawdown': self.current_drawdown,
            'warnings': warnings,
            'recommendations': recommendations,
            'portfolio_risk_score': self.calculate_risk_score(),
        }

    def calculate_risk_score(self) -> float:
        """Calculate overall portfolio risk score (0-100, higher = more risky)."""
        if not self.positions:
            return 0

        score = 0

        # Sector concentration risk (0-30 points)
        sector_exposure = self.get_sector_exposure()
        max_sector = max(sector_exposure.values()) if sector_exposure else 0
        score += min(30, max_sector * 100)

        # Position concentration risk (0-25 points)
        total_value = sum(abs(p.market_value) for p in self.positions.values())
        if total_value > 0:
            max_position = max(abs(p.market_value) / total_value for p in self.positions.values())
            score += min(25, max_position * 167)

        # Drawdown risk (0-25 points)
        score += min(25, self.current_drawdown * 125)

        # Correlation risk (0-20 points)
        corr_scores = []
        for group, tickers in self.correlation_groups.items():
            group_value = sum(abs(p.market_value) for p in self.positions.values()
                            if p.ticker.upper() in tickers)
            if total_value > 0:
                corr_scores.append(group_value / total_value)
        if corr_scores:
            score += min(20, max(corr_scores) * 50)

        return min(100, score)

    def get_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Get comprehensive portfolio metrics."""
        total_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.pnl for p in self.positions.values())
        total_cost = sum(p.entry_price * p.shares for p in self.positions.values())

        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        sector_exposure = self.get_sector_exposure()
        max_sector = max(sector_exposure.values()) if sector_exposure else 0

        total_abs_value = sum(abs(p.market_value) for p in self.positions.values())
        max_position = max(abs(p.market_value) / total_abs_value for p in self.positions.values()) \
                      if total_abs_value > 0 else 0

        risk_score = self.calculate_risk_score()

        # Determine risk category
        if risk_score < 30:
            risk_cat = "LOW"
        elif risk_score < 60:
            risk_cat = "MODERATE"
        else:
            risk_cat = "HIGH"

        # Generate warnings
        warnings = []
        if max_sector > self.max_sector_pct:
            warnings.append(f"Sector concentration exceeds limit: {max_sector*100:.1f}%")
        if max_position > self.max_position_pct:
            warnings.append(f"Position concentration exceeds limit: {max_position*100:.1f}%")
        if self.current_drawdown > self.drawdown_reduce_threshold:
            warnings.append(f"Drawdown warning: {self.current_drawdown*100:.1f}%")

        # Generate recommendations
        recommendations = []
        if risk_score > 70:
            recommendations.append("Consider reducing position sizes")
        if max_sector > 0.25:
            recommendations.append("Diversify across more sectors")
        if len(self.positions) < 5 and total_value > 10000:
            recommendations.append("Consider adding more positions for diversification")

        return PortfolioRiskMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            positions_count=len(self.positions),
            long_count=sum(1 for p in self.positions.values() if p.position_type == 'LONG'),
            short_count=sum(1 for p in self.positions.values() if p.position_type == 'SHORT'),
            sector_concentration=sector_exposure,
            max_position_pct=max_position,
            max_sector_pct=max_sector,
            risk_score=risk_score,
            risk_category=risk_cat,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total = self.stats['signals_evaluated']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'approval_rate': self.stats['signals_approved'] / total * 100,
            'reduction_rate': self.stats['signals_reduced'] / total * 100,
            'block_rate': self.stats['signals_blocked'] / total * 100,
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'signals_evaluated': 0,
            'signals_approved': 0,
            'signals_reduced': 0,
            'signals_blocked': 0,
            'sector_limit_hits': 0,
            'position_limit_hits': 0,
            'correlation_limit_hits': 0,
            'drawdown_reductions': 0,
        }


# Factory function
def create_portfolio_optimizer(
    risk_profile: str = "moderate",
    max_position_pct: float = 0.15,
    max_sector_pct: float = 0.30,
) -> PortfolioRiskOptimizer:
    """Create a portfolio risk optimizer."""
    profile_map = {
        'conservative': RiskCategory.CONSERVATIVE,
        'moderate': RiskCategory.MODERATE,
        'aggressive': RiskCategory.AGGRESSIVE,
    }
    profile = profile_map.get(risk_profile.lower(), RiskCategory.MODERATE)

    return PortfolioRiskOptimizer(
        max_position_pct=max_position_pct,
        max_sector_pct=max_sector_pct,
        risk_profile=profile,
    )
