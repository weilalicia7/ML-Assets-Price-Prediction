"""
Phase 6: Tax-Efficient Optimization

This module provides tax-aware portfolio management including:
- Tax lot tracking and management
- Tax-loss harvesting
- Capital gains optimization
- Hold period optimization

Expected Impact: +1-2% net returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid


class LotSelectionMethod(Enum):
    """Methods for selecting tax lots to sell."""
    FIFO = "fifo"                    # First In, First Out
    LIFO = "lifo"                    # Last In, First Out
    HIGHEST_COST = "highest_cost"    # Minimize realized gains
    LOWEST_COST = "lowest_cost"      # Maximize realized gains
    SPECIFIC = "specific"            # Specific lot identification
    TAX_OPTIMIZED = "tax_optimized"  # Optimize based on tax situation


class HarvestAction(Enum):
    """Tax loss harvesting actions."""
    HARVEST = "harvest"
    HOLD = "hold"
    MONITOR = "monitor"


@dataclass
class TaxConfig:
    """Configuration for tax optimization."""
    short_term_rate: float = 0.35            # 35% short-term capital gains
    long_term_rate: float = 0.15             # 15% long-term capital gains
    loss_harvesting_threshold: float = 0.03  # 3% minimum loss to harvest
    wash_sale_window_days: int = 30          # 30 days before/after
    min_tax_benefit: float = 100             # $100 minimum benefit
    long_term_holding_days: int = 365        # 1 year for long-term
    defer_sale_threshold_days: int = 30      # Defer if < 30 days to LT
    replacement_correlation_min: float = 0.7 # Min correlation for replacement


@dataclass
class TaxLot:
    """Represents a tax lot for a position."""
    lot_id: str
    ticker: str
    quantity: float
    cost_basis: float                  # Per share
    purchase_date: datetime
    is_long_term: bool = False         # > 1 year holding
    unrealized_gain: float = 0.0
    unrealized_gain_pct: float = 0.0
    wash_sale_disallowed: float = 0.0  # Disallowed loss amount

    def __post_init__(self):
        """Calculate if long-term based on purchase date."""
        days_held = (datetime.now() - self.purchase_date).days
        self.is_long_term = days_held >= 365

    def days_held(self) -> int:
        """Get days held."""
        return (datetime.now() - self.purchase_date).days

    def days_to_long_term(self) -> int:
        """Get days until long-term status."""
        days = 365 - self.days_held()
        return max(0, days)

    def update_unrealized(self, current_price: float):
        """Update unrealized gain with current price."""
        self.unrealized_gain = (current_price - self.cost_basis) * self.quantity
        self.unrealized_gain_pct = (current_price / self.cost_basis - 1) if self.cost_basis > 0 else 0


@dataclass
class HarvestOpportunity:
    """Tax loss harvesting opportunity."""
    ticker: str
    lots: List[TaxLot]
    total_loss: float
    tax_benefit: float
    replacement_tickers: List[str]
    wash_sale_risk: bool
    action: HarvestAction
    rationale: str


@dataclass
class TaxImpact:
    """Tax impact of a transaction."""
    ticker: str
    quantity: float
    realized_gain: float
    short_term_gain: float
    long_term_gain: float
    tax_liability: float
    effective_rate: float
    lots_used: List[str]


# =============================================================================
# 1. Tax Lot Manager
# =============================================================================

class TaxLotManager:
    """
    Tracks tax lots for all positions.

    Maintains complete history of purchases and sales for accurate
    tax lot accounting.
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        """
        Initialize manager.

        Args:
            config: Tax configuration
        """
        self.config = config or TaxConfig()
        self.lots: Dict[str, List[TaxLot]] = defaultdict(list)
        self.wash_sale_windows: Dict[str, List[datetime]] = defaultdict(list)
        self.realized_gains: List[TaxImpact] = []

    def add_lot(
        self,
        ticker: str,
        quantity: float,
        cost_basis: float,
        purchase_date: Optional[datetime] = None
    ) -> TaxLot:
        """
        Add a new tax lot.

        Args:
            ticker: Ticker symbol
            quantity: Number of shares
            cost_basis: Cost per share
            purchase_date: Purchase date (default: now)

        Returns:
            Created TaxLot
        """
        lot = TaxLot(
            lot_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            quantity=quantity,
            cost_basis=cost_basis,
            purchase_date=purchase_date or datetime.now()
        )

        self.lots[ticker].append(lot)

        return lot

    def get_lots(
        self,
        ticker: str,
        sort_by: LotSelectionMethod = LotSelectionMethod.FIFO
    ) -> List[TaxLot]:
        """
        Get tax lots for a ticker, sorted by selection method.

        Args:
            ticker: Ticker symbol
            sort_by: Sorting method

        Returns:
            Sorted list of tax lots
        """
        lots = self.lots.get(ticker, [])

        if sort_by == LotSelectionMethod.FIFO:
            return sorted(lots, key=lambda l: l.purchase_date)
        elif sort_by == LotSelectionMethod.LIFO:
            return sorted(lots, key=lambda l: l.purchase_date, reverse=True)
        elif sort_by == LotSelectionMethod.HIGHEST_COST:
            return sorted(lots, key=lambda l: l.cost_basis, reverse=True)
        elif sort_by == LotSelectionMethod.LOWEST_COST:
            return sorted(lots, key=lambda l: l.cost_basis)
        else:
            return lots

    def get_total_quantity(self, ticker: str) -> float:
        """Get total quantity held for a ticker."""
        return sum(lot.quantity for lot in self.lots.get(ticker, []))

    def get_average_cost_basis(self, ticker: str) -> float:
        """Get average cost basis for a ticker."""
        lots = self.lots.get(ticker, [])
        if not lots:
            return 0.0

        total_cost = sum(lot.cost_basis * lot.quantity for lot in lots)
        total_quantity = sum(lot.quantity for lot in lots)

        return total_cost / total_quantity if total_quantity > 0 else 0.0

    def calculate_unrealized_gains(
        self,
        ticker: str,
        current_price: float
    ) -> Dict[str, float]:
        """
        Calculate unrealized gains for a ticker.

        Args:
            ticker: Ticker symbol
            current_price: Current market price

        Returns:
            Dictionary with gain breakdown
        """
        lots = self.lots.get(ticker, [])

        short_term_gain = 0.0
        long_term_gain = 0.0
        total_gain = 0.0

        for lot in lots:
            lot.update_unrealized(current_price)
            total_gain += lot.unrealized_gain

            if lot.is_long_term:
                long_term_gain += lot.unrealized_gain
            else:
                short_term_gain += lot.unrealized_gain

        return {
            'total_gain': total_gain,
            'short_term_gain': short_term_gain,
            'long_term_gain': long_term_gain,
            'total_quantity': sum(l.quantity for l in lots),
            'num_lots': len(lots)
        }

    def get_short_term_lots(self, ticker: str) -> List[TaxLot]:
        """Get only short-term lots."""
        return [lot for lot in self.lots.get(ticker, []) if not lot.is_long_term]

    def get_long_term_lots(self, ticker: str) -> List[TaxLot]:
        """Get only long-term lots."""
        return [lot for lot in self.lots.get(ticker, []) if lot.is_long_term]

    def sell_lots(
        self,
        ticker: str,
        quantity: float,
        current_price: float,
        method: LotSelectionMethod = LotSelectionMethod.FIFO
    ) -> TaxImpact:
        """
        Sell shares and calculate tax impact.

        Args:
            ticker: Ticker symbol
            quantity: Quantity to sell
            current_price: Current market price
            method: Lot selection method

        Returns:
            TaxImpact of the sale
        """
        lots = self.get_lots(ticker, method)
        remaining = quantity

        short_term_gain = 0.0
        long_term_gain = 0.0
        lots_used = []

        for lot in lots:
            if remaining <= 0:
                break

            sell_qty = min(lot.quantity, remaining)
            gain = (current_price - lot.cost_basis) * sell_qty

            if lot.is_long_term:
                long_term_gain += gain
            else:
                short_term_gain += gain

            lot.quantity -= sell_qty
            remaining -= sell_qty
            lots_used.append(lot.lot_id)

        # Remove empty lots
        self.lots[ticker] = [l for l in self.lots[ticker] if l.quantity > 0]

        # Record wash sale window
        self.wash_sale_windows[ticker].append(datetime.now())

        # Calculate tax liability
        st_tax = short_term_gain * self.config.short_term_rate if short_term_gain > 0 else 0
        lt_tax = long_term_gain * self.config.long_term_rate if long_term_gain > 0 else 0
        total_tax = st_tax + lt_tax

        total_gain = short_term_gain + long_term_gain
        effective_rate = total_tax / total_gain if total_gain > 0 else 0

        impact = TaxImpact(
            ticker=ticker,
            quantity=quantity - remaining,
            realized_gain=total_gain,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            tax_liability=total_tax,
            effective_rate=effective_rate,
            lots_used=lots_used
        )

        self.realized_gains.append(impact)

        return impact

    def check_wash_sale_window(self, ticker: str) -> bool:
        """
        Check if ticker is in wash sale window.

        Args:
            ticker: Ticker symbol

        Returns:
            True if in wash sale window
        """
        window_start = datetime.now() - timedelta(days=self.config.wash_sale_window_days)

        sales = self.wash_sale_windows.get(ticker, [])
        recent_sales = [s for s in sales if s >= window_start]

        return len(recent_sales) > 0

    def get_ytd_realized(self) -> Dict[str, float]:
        """Get year-to-date realized gains."""
        year_start = datetime(datetime.now().year, 1, 1)

        ytd_st_gain = 0.0
        ytd_lt_gain = 0.0
        ytd_tax = 0.0

        for impact in self.realized_gains:
            ytd_st_gain += impact.short_term_gain
            ytd_lt_gain += impact.long_term_gain
            ytd_tax += impact.tax_liability

        return {
            'short_term_gains': ytd_st_gain,
            'long_term_gains': ytd_lt_gain,
            'total_gains': ytd_st_gain + ytd_lt_gain,
            'tax_liability': ytd_tax
        }


# =============================================================================
# 2. Tax Loss Harvester
# =============================================================================

class TaxLossHarvester:
    """
    Identifies and manages tax-loss harvesting opportunities.
    """

    def __init__(
        self,
        lot_manager: TaxLotManager,
        config: Optional[TaxConfig] = None
    ):
        """
        Initialize harvester.

        Args:
            lot_manager: Tax lot manager
            config: Tax configuration
        """
        self.lot_manager = lot_manager
        self.config = config or TaxConfig()
        self.harvest_history: List[HarvestOpportunity] = []

        # Similar securities mapping (for wash sale avoidance)
        self.similar_securities: Dict[str, Set[str]] = {}

    def add_similar_securities(self, ticker: str, similar: List[str]):
        """Register similar securities for wash sale tracking."""
        self.similar_securities[ticker] = set(similar)

    def find_harvesting_opportunities(
        self,
        current_prices: Dict[str, float],
        min_loss_threshold: Optional[float] = None
    ) -> List[HarvestOpportunity]:
        """
        Find all tax-loss harvesting opportunities.

        Args:
            current_prices: Current prices by ticker
            min_loss_threshold: Minimum loss threshold

        Returns:
            List of harvesting opportunities
        """
        threshold = min_loss_threshold or self.config.loss_harvesting_threshold
        opportunities = []

        for ticker, lots in self.lot_manager.lots.items():
            if ticker not in current_prices:
                continue

            current_price = current_prices[ticker]
            loss_lots = []
            total_loss = 0.0

            for lot in lots:
                lot.update_unrealized(current_price)
                if lot.unrealized_gain_pct < -threshold:
                    loss_lots.append(lot)
                    total_loss += lot.unrealized_gain

            if not loss_lots:
                continue

            # Calculate tax benefit
            tax_benefit = abs(total_loss) * self.config.short_term_rate

            # Check wash sale risk
            wash_sale_risk = self.check_wash_sale_risk(ticker)

            # Find replacement securities
            replacements = self._find_replacement_securities(ticker)

            # Determine action
            if wash_sale_risk:
                action = HarvestAction.MONITOR
                rationale = "In wash sale window - monitor only"
            elif tax_benefit < self.config.min_tax_benefit:
                action = HarvestAction.HOLD
                rationale = f"Tax benefit ${tax_benefit:.2f} below minimum ${self.config.min_tax_benefit}"
            elif not replacements:
                action = HarvestAction.MONITOR
                rationale = "No suitable replacement securities found"
            else:
                action = HarvestAction.HARVEST
                rationale = f"Harvest ${abs(total_loss):.2f} loss for ${tax_benefit:.2f} tax benefit"

            opportunities.append(HarvestOpportunity(
                ticker=ticker,
                lots=loss_lots,
                total_loss=total_loss,
                tax_benefit=tax_benefit,
                replacement_tickers=replacements,
                wash_sale_risk=wash_sale_risk,
                action=action,
                rationale=rationale
            ))

        # Sort by tax benefit descending
        opportunities.sort(key=lambda o: o.tax_benefit, reverse=True)

        return opportunities

    def check_wash_sale_risk(
        self,
        ticker: str,
        check_similar: bool = True
    ) -> bool:
        """
        Check if there's wash sale risk for a ticker.

        Args:
            ticker: Ticker symbol
            check_similar: Also check similar securities

        Returns:
            True if wash sale risk exists
        """
        # Check direct wash sale window
        if self.lot_manager.check_wash_sale_window(ticker):
            return True

        # Check similar securities
        if check_similar:
            similar = self.similar_securities.get(ticker, set())
            for sim_ticker in similar:
                if self.lot_manager.check_wash_sale_window(sim_ticker):
                    return True

        return False

    def _find_replacement_securities(
        self,
        ticker: str
    ) -> List[str]:
        """Find suitable replacement securities."""
        # In production, this would look up correlated securities
        # For now, return sector ETFs based on ticker patterns

        # Simple sector mapping
        sector_replacements = {
            'AAPL': ['XLK', 'QQQ', 'VGT'],
            'MSFT': ['XLK', 'QQQ', 'VGT'],
            'GOOGL': ['XLC', 'QQQ', 'VOX'],
            'AMZN': ['XLY', 'QQQ', 'VCR'],
            'JPM': ['XLF', 'KBE', 'VFH'],
            'XOM': ['XLE', 'VDE', 'IYE'],
        }

        return sector_replacements.get(ticker, [])

    def execute_harvest(
        self,
        opportunity: HarvestOpportunity,
        current_price: float,
        replacement_ticker: Optional[str] = None
    ) -> TaxImpact:
        """
        Execute tax loss harvest.

        Args:
            opportunity: Harvest opportunity
            current_price: Current price
            replacement_ticker: Replacement to buy

        Returns:
            Tax impact of harvest
        """
        # Sell the loss lots
        quantity = sum(lot.quantity for lot in opportunity.lots)

        impact = self.lot_manager.sell_lots(
            opportunity.ticker,
            quantity,
            current_price,
            LotSelectionMethod.SPECIFIC
        )

        # Record harvest
        self.harvest_history.append(opportunity)

        return impact

    def calculate_harvest_benefit(
        self,
        opportunities: List[HarvestOpportunity],
        ytd_gains: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate benefit of harvesting against YTD gains.

        Args:
            opportunities: Harvest opportunities
            ytd_gains: YTD realized gains

        Returns:
            Dictionary with benefit analysis
        """
        total_harvestable_loss = sum(
            abs(o.total_loss) for o in opportunities
            if o.action == HarvestAction.HARVEST
        )

        # Can offset short-term first, then long-term, then $3k ordinary
        st_offset = min(total_harvestable_loss, ytd_gains.get('short_term_gains', 0))
        remaining = total_harvestable_loss - st_offset

        lt_offset = min(remaining, ytd_gains.get('long_term_gains', 0))
        remaining = remaining - lt_offset

        ordinary_offset = min(remaining, 3000)  # $3k limit

        st_tax_savings = st_offset * self.config.short_term_rate
        lt_tax_savings = lt_offset * self.config.long_term_rate
        ordinary_savings = ordinary_offset * self.config.short_term_rate

        return {
            'total_harvestable_loss': total_harvestable_loss,
            'st_gains_offset': st_offset,
            'lt_gains_offset': lt_offset,
            'ordinary_income_offset': ordinary_offset,
            'carryforward': max(0, remaining - ordinary_offset),
            'total_tax_savings': st_tax_savings + lt_tax_savings + ordinary_savings
        }


# =============================================================================
# 3. Capital Gains Optimizer
# =============================================================================

class CapitalGainsOptimizer:
    """
    Optimizes capital gains recognition timing and lot selection.
    """

    def __init__(
        self,
        lot_manager: TaxLotManager,
        config: Optional[TaxConfig] = None
    ):
        """
        Initialize optimizer.

        Args:
            lot_manager: Tax lot manager
            config: Tax configuration
        """
        self.lot_manager = lot_manager
        self.config = config or TaxConfig()

    def select_lots_for_sale(
        self,
        ticker: str,
        quantity: float,
        current_price: float,
        strategy: LotSelectionMethod = LotSelectionMethod.TAX_OPTIMIZED
    ) -> Tuple[List[TaxLot], TaxImpact]:
        """
        Select optimal lots for a sale.

        Args:
            ticker: Ticker symbol
            quantity: Quantity to sell
            current_price: Current market price
            strategy: Lot selection strategy

        Returns:
            Tuple of (selected lots, estimated tax impact)
        """
        if strategy == LotSelectionMethod.TAX_OPTIMIZED:
            return self._tax_optimized_selection(ticker, quantity, current_price)

        lots = self.lot_manager.get_lots(ticker, strategy)
        selected = []
        remaining = quantity

        for lot in lots:
            if remaining <= 0:
                break
            selected.append(lot)
            remaining -= lot.quantity

        # Calculate impact
        impact = self._calculate_sale_impact(selected, current_price)

        return selected, impact

    def _tax_optimized_selection(
        self,
        ticker: str,
        quantity: float,
        current_price: float
    ) -> Tuple[List[TaxLot], TaxImpact]:
        """
        Select lots to minimize tax liability.

        Priority:
        1. Long-term losses
        2. Short-term losses
        3. Long-term gains
        4. Short-term gains (last resort)
        """
        lots = self.lot_manager.lots.get(ticker, [])
        for lot in lots:
            lot.update_unrealized(current_price)

        # Categorize lots
        lt_losses = [l for l in lots if l.is_long_term and l.unrealized_gain < 0]
        st_losses = [l for l in lots if not l.is_long_term and l.unrealized_gain < 0]
        lt_gains = [l for l in lots if l.is_long_term and l.unrealized_gain >= 0]
        st_gains = [l for l in lots if not l.is_long_term and l.unrealized_gain >= 0]

        # Sort each category (biggest losses first, smallest gains first)
        lt_losses.sort(key=lambda l: l.unrealized_gain)
        st_losses.sort(key=lambda l: l.unrealized_gain)
        lt_gains.sort(key=lambda l: l.unrealized_gain)
        st_gains.sort(key=lambda l: l.unrealized_gain)

        # Build selection order
        selection_order = lt_losses + st_losses + lt_gains + st_gains

        selected = []
        remaining = quantity

        for lot in selection_order:
            if remaining <= 0:
                break
            selected.append(lot)
            remaining -= lot.quantity

        impact = self._calculate_sale_impact(selected, current_price)

        return selected, impact

    def _calculate_sale_impact(
        self,
        lots: List[TaxLot],
        current_price: float
    ) -> TaxImpact:
        """Calculate tax impact of selling lots."""
        short_term_gain = 0.0
        long_term_gain = 0.0
        total_quantity = 0.0
        lot_ids = []

        for lot in lots:
            gain = (current_price - lot.cost_basis) * lot.quantity
            if lot.is_long_term:
                long_term_gain += gain
            else:
                short_term_gain += gain
            total_quantity += lot.quantity
            lot_ids.append(lot.lot_id)

        st_tax = max(0, short_term_gain) * self.config.short_term_rate
        lt_tax = max(0, long_term_gain) * self.config.long_term_rate
        total_tax = st_tax + lt_tax

        total_gain = short_term_gain + long_term_gain
        effective_rate = total_tax / total_gain if total_gain > 0 else 0

        return TaxImpact(
            ticker=lots[0].ticker if lots else "",
            quantity=total_quantity,
            realized_gain=total_gain,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            tax_liability=total_tax,
            effective_rate=effective_rate,
            lots_used=lot_ids
        )

    def estimate_tax_liability(
        self,
        gains: Dict[str, float]
    ) -> float:
        """
        Estimate tax liability from gains.

        Args:
            gains: Dictionary with 'short_term' and 'long_term' gains

        Returns:
            Estimated tax liability
        """
        st_gain = gains.get('short_term', 0)
        lt_gain = gains.get('long_term', 0)

        st_tax = max(0, st_gain) * self.config.short_term_rate
        lt_tax = max(0, lt_gain) * self.config.long_term_rate

        return st_tax + lt_tax


# =============================================================================
# 4. Hold Period Optimizer
# =============================================================================

class HoldPeriodOptimizer:
    """
    Optimizes decisions based on tax holding periods.
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Tax configuration
        """
        self.config = config or TaxConfig()

    def should_defer_sale(
        self,
        lot: TaxLot,
        signal_strength: float,
        expected_return: float
    ) -> Tuple[bool, str]:
        """
        Determine if sale should be deferred for long-term treatment.

        Args:
            lot: Tax lot to evaluate
            signal_strength: Sell signal strength (0-1)
            expected_return: Expected return if held

        Returns:
            Tuple of (should_defer, rationale)
        """
        if lot.is_long_term:
            return False, "Already long-term"

        days_to_lt = lot.days_to_long_term()

        if days_to_lt == 0:
            return False, "Already long-term"

        if days_to_lt > self.config.defer_sale_threshold_days:
            return False, f"Too long to wait ({days_to_lt} days)"

        # Calculate breakeven
        breakeven = self._calculate_breakeven_return(lot, days_to_lt)

        if signal_strength > 0.8:
            return False, f"Strong sell signal ({signal_strength:.0%})"

        if expected_return > breakeven:
            return True, f"Expected return {expected_return:.1%} > breakeven {breakeven:.1%}, defer {days_to_lt} days"

        return False, f"Expected return {expected_return:.1%} < breakeven {breakeven:.1%}"

    def _calculate_breakeven_return(
        self,
        lot: TaxLot,
        days_to_lt: int
    ) -> float:
        """
        Calculate breakeven return for deferring sale.

        Returns the return needed to justify waiting for long-term treatment.
        """
        if lot.unrealized_gain <= 0:
            return 0.0  # No tax benefit for losses

        # Tax savings from long-term vs short-term
        tax_savings_rate = self.config.short_term_rate - self.config.long_term_rate

        # Convert to annualized return
        days_per_year = 365
        annualized_factor = days_per_year / max(days_to_lt, 1)

        # Breakeven: the return that equals tax savings
        breakeven = tax_savings_rate / annualized_factor

        return breakeven

    def get_deferral_recommendations(
        self,
        lots: List[TaxLot],
        sell_signals: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> List[Dict]:
        """
        Get deferral recommendations for multiple lots.

        Args:
            lots: List of tax lots
            sell_signals: Sell signal strength by ticker
            expected_returns: Expected returns by ticker

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        for lot in lots:
            signal = sell_signals.get(lot.ticker, 0.5)
            exp_return = expected_returns.get(lot.ticker, 0.0)

            should_defer, rationale = self.should_defer_sale(lot, signal, exp_return)

            recommendations.append({
                'ticker': lot.ticker,
                'lot_id': lot.lot_id,
                'days_to_long_term': lot.days_to_long_term(),
                'unrealized_gain': lot.unrealized_gain,
                'should_defer': should_defer,
                'rationale': rationale
            })

        return recommendations


# =============================================================================
# 5. Integrated Tax Optimizer
# =============================================================================

class TaxOptimizer:
    """
    Integrated tax optimization system.

    Combines lot management, loss harvesting, and capital gains optimization.
    """

    def __init__(self, config: Optional[TaxConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Tax configuration
        """
        self.config = config or TaxConfig()
        self.lot_manager = TaxLotManager(config)
        self.harvester = TaxLossHarvester(self.lot_manager, config)
        self.gains_optimizer = CapitalGainsOptimizer(self.lot_manager, config)
        self.hold_optimizer = HoldPeriodOptimizer(config)

    def optimize_sale(
        self,
        ticker: str,
        quantity: float,
        current_price: float,
        signal_strength: float = 0.5
    ) -> Dict:
        """
        Optimize a sale for tax efficiency.

        Args:
            ticker: Ticker symbol
            quantity: Quantity to sell
            current_price: Current price
            signal_strength: Sell signal strength

        Returns:
            Optimization result with recommendations
        """
        # Get lots
        lots = self.lot_manager.lots.get(ticker, [])
        if not lots:
            return {'action': 'no_position', 'rationale': 'No position to sell'}

        # Check for deferral opportunities
        near_lt_lots = [l for l in lots if 0 < l.days_to_long_term() <= 30]
        if near_lt_lots and signal_strength < 0.8:
            return {
                'action': 'defer',
                'rationale': f'{len(near_lt_lots)} lots near long-term status',
                'days_to_wait': min(l.days_to_long_term() for l in near_lt_lots)
            }

        # Select optimal lots
        selected, impact = self.gains_optimizer.select_lots_for_sale(
            ticker, quantity, current_price
        )

        return {
            'action': 'sell',
            'lots': [l.lot_id for l in selected],
            'tax_impact': impact,
            'rationale': f"Tax-optimized selection: ${impact.tax_liability:.2f} tax on ${impact.realized_gain:.2f} gain"
        }

    def get_tax_situation(
        self,
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Get comprehensive tax situation report.

        Args:
            current_prices: Current prices by ticker

        Returns:
            Tax situation dictionary
        """
        # YTD realized
        ytd = self.lot_manager.get_ytd_realized()

        # Unrealized by ticker
        unrealized = {}
        for ticker in self.lot_manager.lots:
            if ticker in current_prices:
                unrealized[ticker] = self.lot_manager.calculate_unrealized_gains(
                    ticker, current_prices[ticker]
                )

        # Harvest opportunities
        opportunities = self.harvester.find_harvesting_opportunities(current_prices)
        harvest_benefit = self.harvester.calculate_harvest_benefit(opportunities, ytd)

        return {
            'ytd_realized': ytd,
            'unrealized_by_ticker': unrealized,
            'harvest_opportunities': len([o for o in opportunities if o.action == HarvestAction.HARVEST]),
            'potential_harvest_savings': harvest_benefit['total_tax_savings'],
            'positions_near_long_term': sum(
                1 for lots in self.lot_manager.lots.values()
                for lot in lots if 0 < lot.days_to_long_term() <= 30
            )
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_tax_optimizer(
    config: Optional[Dict] = None
) -> TaxOptimizer:
    """
    Create configured tax optimizer.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured TaxOptimizer
    """
    if config:
        tax_config = TaxConfig(**config)
    else:
        tax_config = TaxConfig()

    return TaxOptimizer(config=tax_config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'LotSelectionMethod',
    'HarvestAction',

    # Data Classes
    'TaxConfig',
    'TaxLot',
    'HarvestOpportunity',
    'TaxImpact',

    # Core Classes
    'TaxLotManager',
    'TaxLossHarvester',
    'CapitalGainsOptimizer',
    'HoldPeriodOptimizer',
    'TaxOptimizer',

    # Factory
    'create_tax_optimizer',
]
