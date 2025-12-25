"""
Profit-Maximizing Regime Classifier for US/International Markets

This module extends the LagFreeRegimeClassifier to maximize profit during
regime transitions rather than just detecting them early.

Key Improvements (US Model Fixing 8):
1. Dynamic Multiplier Scaling based on profit potential
2. Transition-Optimized Position Sizing by phase
3. Concentration vs Diversification based on confidence
4. Profit-Taking Strategy with tiered targets
5. Risk-Adjusted Transition Allocation

NOTE: This is for US/International model ONLY. China/DeepSeek model has
separate profit optimization in china_adaptive_profit_maximizer.py

Usage:
    from src.features.profit_maximizing_regime import ProfitMaximizingRegimeClassifier

    classifier = ProfitMaximizingRegimeClassifier()
    result = classifier.classify_regime_with_profit(spy_data, stock_data, vix_level)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import base lag-free classifier
from src.features.lag_free_regime import LagFreeRegimeClassifier, LagFreeRegimeOutput


class TransitionPhase(Enum):
    """Transition phases for position sizing optimization."""
    PRE_TRANSITION = "pre_transition"      # Day -5 to 0
    EARLY_TRANSITION = "early_transition"  # Day 0-3
    MID_TRANSITION = "mid_transition"      # Day 4-7
    LATE_TRANSITION = "late_transition"    # Day 8+
    NO_TRANSITION = "no_transition"


@dataclass
class ProfitMaximizingOutput:
    """Output from the profit-maximizing classifier."""
    # Base regime info
    regime: str
    transition_state: str
    transition_phase: TransitionPhase
    transition_day: int

    # Profit optimization
    profit_score: float
    dynamic_multiplier: float

    # Position sizing
    recommended_position_size: float
    small_cap_multiplier: float
    large_cap_multiplier: float
    max_position_size: float
    stop_loss: float

    # Concentration
    max_positions: int
    top_n_concentration: float
    position_sizing_method: str

    # Profit targets
    profit_target_1: float
    profit_target_2: float
    profit_target_3: float
    trailing_stop: float

    # Capital allocation
    capital_allocation: Dict[str, float]

    # Sector focus
    sector_focus: List[str]

    # Raw base output
    base_output: LagFreeRegimeOutput


class ProfitMaximizingRegimeClassifier(LagFreeRegimeClassifier):
    """
    Extends LagFreeRegimeClassifier to maximize profit during transitions.

    Key Features:
    - Dynamic profit-based multiplier scaling (up to +40% vs fixed +6%)
    - Phase-aware position sizing (early vs late transition)
    - Kelly Criterion-based position sizing
    - Tiered profit-taking targets
    - VIX-adjusted capital allocation
    """

    # =========================================================================
    # PROFIT SCORE WEIGHTS
    # =========================================================================
    PROFIT_SCORE_WEIGHTS = {
        'momentum': 0.30,      # 30% - 5D return strength
        'volume': 0.25,        # 25% - Volume confirmation
        'rsi': 0.20,           # 20% - RSI positioning
        'volatility': 0.25     # 25% - Low volatility = quality
    }

    # =========================================================================
    # TRANSITION PHASE PARAMETERS
    # =========================================================================
    PHASE_PARAMS = {
        TransitionPhase.EARLY_TRANSITION: {  # Day 0-3
            'small_cap_multiplier': 1.5,
            'large_cap_multiplier': 0.8,
            'sector_focus': ['technology', 'consumer_discretionary', 'financials'],
            'max_position_size': 0.10,
            'stop_loss': 0.08,
            'position_build_rate': 0.30
        },
        TransitionPhase.MID_TRANSITION: {  # Day 4-7
            'small_cap_multiplier': 1.2,
            'large_cap_multiplier': 1.0,
            'sector_focus': ['technology', 'communication', 'industrials'],
            'max_position_size': 0.08,
            'stop_loss': 0.10,
            'position_build_rate': 0.15
        },
        TransitionPhase.LATE_TRANSITION: {  # Day 8+
            'small_cap_multiplier': 0.9,
            'large_cap_multiplier': 1.3,
            'sector_focus': ['technology', 'healthcare', 'consumer_staples'],
            'max_position_size': 0.06,
            'stop_loss': 0.12,
            'position_build_rate': 0.00
        },
        TransitionPhase.NO_TRANSITION: {
            'small_cap_multiplier': 1.0,
            'large_cap_multiplier': 1.0,
            'sector_focus': ['all'],
            'max_position_size': 0.05,
            'stop_loss': 0.10,
            'position_build_rate': 0.10
        },
        TransitionPhase.PRE_TRANSITION: {  # Day -5 to 0
            'small_cap_multiplier': 1.2,
            'large_cap_multiplier': 0.9,
            'sector_focus': ['technology', 'consumer_discretionary', 'financials'],
            'max_position_size': 0.08,
            'stop_loss': 0.12,
            'position_build_rate': 0.20
        }
    }

    # =========================================================================
    # CONCENTRATION PARAMETERS
    # =========================================================================
    CONCENTRATION_PARAMS = {
        'high_confidence': {  # > 80%
            'max_positions': 8,
            'top_n_concentration': 0.60,
            'minimum_conviction': 0.85,
            'position_sizing_method': 'kelly_criterion'
        },
        'medium_confidence': {  # 60-80%
            'max_positions': 12,
            'top_n_concentration': 0.40,
            'minimum_conviction': 0.75,
            'position_sizing_method': 'half_kelly'
        },
        'low_confidence': {  # < 60%
            'max_positions': 20,
            'top_n_concentration': 0.25,
            'minimum_conviction': 0.65,
            'position_sizing_method': 'equal_weight'
        }
    }

    # =========================================================================
    # PROFIT TARGET CONFIGURATIONS
    # =========================================================================
    PROFIT_TARGETS = {
        'high_momentum_high_vol': {
            'profit_target_1': 0.15,
            'profit_target_2': 0.25,
            'profit_target_3': 0.40,
            'trailing_stop': 0.08,
            'time_exit_days': 10
        },
        'quality_large_cap': {
            'profit_target_1': 0.10,
            'profit_target_2': 0.18,
            'profit_target_3': 0.30,
            'trailing_stop': 0.12,
            'time_exit_days': 20
        },
        'default': {
            'profit_target_1': 0.12,
            'profit_target_2': 0.20,
            'profit_target_3': 0.35,
            'trailing_stop': 0.10,
            'time_exit_days': 15
        }
    }

    def __init__(self, initial_vix: float = 20.0):
        """Initialize the profit-maximizing classifier."""
        super().__init__(initial_vix)
        self.transition_start_date = None
        self.transition_day_count = 0
        self.last_regime = None

        print("[US PROFIT MAX] ProfitMaximizingRegimeClassifier initialized")
        print("  - Dynamic multiplier scaling: ENABLED")
        print("  - Transition phase detection: ENABLED")
        print("  - Profit-taking targets: ENABLED")

    # =========================================================================
    # MAIN CLASSIFICATION METHOD
    # =========================================================================

    def classify_regime_with_profit(
        self,
        spy_data: pd.DataFrame,
        stock_data: Optional[Dict[str, Any]] = None,
        vix_level: float = 20.0,
        is_fomc_week: bool = False,
        is_earnings_season: bool = False,
        is_opex_week: bool = False
    ) -> ProfitMaximizingOutput:
        """
        Classify regime with profit maximization.

        Args:
            spy_data: SPY/market OHLCV data
            stock_data: Optional individual stock data for profit scoring
            vix_level: Current VIX level
            is_fomc_week: FOMC week flag
            is_earnings_season: Earnings season flag
            is_opex_week: Options expiration week flag

        Returns:
            ProfitMaximizingOutput with complete profit optimization data
        """
        # Get base regime classification
        base_output = self.classify_regime(
            spy_data, vix_level, is_fomc_week, is_earnings_season, is_opex_week
        )

        # Detect transition phase
        transition_phase, transition_day = self._detect_transition_phase(base_output)

        # Calculate profit score if stock data provided
        profit_score = self._calculate_profit_score(stock_data) if stock_data else 70.0

        # Get dynamic multiplier based on profit score
        dynamic_multiplier = self._get_dynamic_multiplier(
            profit_score, base_output.buy_multiplier
        )

        # Get phase-specific parameters
        phase_params = self.PHASE_PARAMS.get(
            transition_phase, self.PHASE_PARAMS[TransitionPhase.NO_TRANSITION]
        )

        # Get concentration parameters based on confidence
        concentration = self._get_concentration_params(base_output.confidence)

        # Get profit targets based on stock type
        stock_type = self._classify_stock_type(stock_data) if stock_data else 'default'
        profit_targets = self.PROFIT_TARGETS.get(stock_type, self.PROFIT_TARGETS['default'])

        # Calculate capital allocation
        capital_allocation = self._calculate_capital_allocation(vix_level, transition_phase)

        # Calculate recommended position size
        recommended_size = self._calculate_position_size(
            profit_score,
            dynamic_multiplier,
            phase_params['max_position_size'],
            concentration['position_sizing_method']
        )

        return ProfitMaximizingOutput(
            regime=base_output.primary_regime,
            transition_state=base_output.transition_state,
            transition_phase=transition_phase,
            transition_day=transition_day,
            profit_score=profit_score,
            dynamic_multiplier=dynamic_multiplier,
            recommended_position_size=recommended_size,
            small_cap_multiplier=phase_params['small_cap_multiplier'],
            large_cap_multiplier=phase_params['large_cap_multiplier'],
            max_position_size=phase_params['max_position_size'],
            stop_loss=phase_params['stop_loss'],
            max_positions=concentration['max_positions'],
            top_n_concentration=concentration['top_n_concentration'],
            position_sizing_method=concentration['position_sizing_method'],
            profit_target_1=profit_targets['profit_target_1'],
            profit_target_2=profit_targets['profit_target_2'],
            profit_target_3=profit_targets['profit_target_3'],
            trailing_stop=profit_targets['trailing_stop'],
            capital_allocation=capital_allocation,
            sector_focus=phase_params['sector_focus'],
            base_output=base_output
        )

    # =========================================================================
    # PROFIT SCORE CALCULATION
    # =========================================================================

    def _calculate_profit_score(self, stock_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive profit potential score (0-100).

        Components:
        - Momentum (30%): 5D return strength
        - Volume (25%): Volume confirmation
        - RSI (20%): Sweet spot positioning
        - Volatility (25%): Low vol = quality
        """
        score = 0.0

        # 1. Momentum Score (30%)
        momentum_5d = stock_data.get('5d_return', stock_data.get('ret_5d', 0))
        if isinstance(momentum_5d, (int, float)):
            # Positive momentum scores higher
            momentum_score = min(abs(momentum_5d) * 10, 30)
            if momentum_5d > 0:
                score += momentum_score
            else:
                score += momentum_score * 0.3  # Negative momentum gets partial credit

        # 2. Volume Confirmation (25%)
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if isinstance(volume_ratio, (int, float)):
            if volume_ratio > 1.5:
                score += 25
            elif volume_ratio > 1.2:
                score += 15
            elif volume_ratio > 1.0:
                score += 10

        # 3. RSI Positioning (20%)
        rsi = stock_data.get('rsi', 50)
        if isinstance(rsi, (int, float)):
            if 40 < rsi < 70:  # Sweet spot
                score += 20
            elif 30 < rsi < 80:  # Acceptable
                score += 10
            elif rsi < 30:  # Oversold (potential opportunity)
                score += 15

        # 4. Volatility Quality (25%)
        volatility = stock_data.get('volatility', stock_data.get('vol', 50))
        if isinstance(volatility, (int, float)):
            # Convert to percentage if needed
            if volatility < 1:
                volatility = volatility * 100

            if volatility < 30:
                score += 25  # Low vol = high quality
            elif volatility < 50:
                score += 15
            elif volatility < 70:
                score += 8

        return min(score, 100)

    # =========================================================================
    # DYNAMIC MULTIPLIER
    # =========================================================================

    def _get_dynamic_multiplier(
        self,
        profit_score: float,
        base_multiplier: float
    ) -> float:
        """
        Calculate dynamic multiplier based on profit potential.

        - High profit potential (>80): Up to +40% scaling
        - Medium (60-80): 1.0x (no change)
        - Low (<60): Scale down
        """
        if profit_score > 80:
            # High profit potential: scale up to +40%
            scaling_factor = 1.0 + (profit_score - 80) * 0.02
        elif profit_score > 60:
            # Medium profit potential: standard
            scaling_factor = 1.0
        else:
            # Low profit potential: scale down
            scaling_factor = 0.5 + profit_score * 0.0083

        return base_multiplier * scaling_factor

    # =========================================================================
    # TRANSITION PHASE DETECTION
    # =========================================================================

    def _detect_transition_phase(
        self,
        base_output: LagFreeRegimeOutput
    ) -> Tuple[TransitionPhase, int]:
        """
        Detect which phase of transition we're in.

        Returns:
            Tuple of (TransitionPhase, day_count)
        """
        # Check if we're in a transition
        is_transition = base_output.is_transition or 'TRANSITION' in base_output.transition_state

        if not is_transition:
            self.transition_start_date = None
            self.transition_day_count = 0
            return TransitionPhase.NO_TRANSITION, 0

        # Update transition day count
        if self.last_regime != base_output.primary_regime:
            # New transition started
            self.transition_start_date = pd.Timestamp.now()
            self.transition_day_count = 0
        else:
            self.transition_day_count += 1

        self.last_regime = base_output.primary_regime
        day = self.transition_day_count

        # Determine phase based on day
        if 'EARLY' in base_output.transition_state:
            if day < 0:
                return TransitionPhase.PRE_TRANSITION, day
            elif day < 4:
                return TransitionPhase.EARLY_TRANSITION, day
            elif day < 8:
                return TransitionPhase.MID_TRANSITION, day
            else:
                return TransitionPhase.LATE_TRANSITION, day
        else:
            # Confirmed transition
            if day < 4:
                return TransitionPhase.EARLY_TRANSITION, day
            elif day < 8:
                return TransitionPhase.MID_TRANSITION, day
            else:
                return TransitionPhase.LATE_TRANSITION, day

    # =========================================================================
    # CONCENTRATION PARAMETERS
    # =========================================================================

    def _get_concentration_params(self, confidence: float) -> Dict[str, Any]:
        """Get concentration parameters based on transition confidence."""
        if confidence > 0.8:
            return self.CONCENTRATION_PARAMS['high_confidence']
        elif confidence > 0.6:
            return self.CONCENTRATION_PARAMS['medium_confidence']
        else:
            return self.CONCENTRATION_PARAMS['low_confidence']

    # =========================================================================
    # STOCK TYPE CLASSIFICATION
    # =========================================================================

    def _classify_stock_type(self, stock_data: Dict[str, Any]) -> str:
        """Classify stock type for profit target selection."""
        volatility = stock_data.get('volatility', stock_data.get('vol', 50))
        momentum = abs(stock_data.get('5d_return', stock_data.get('ret_5d', 0)))
        market_cap = stock_data.get('market_cap', 'large')

        # Convert vol to percentage if needed
        if isinstance(volatility, (int, float)) and volatility < 1:
            volatility = volatility * 100

        if isinstance(volatility, (int, float)) and isinstance(momentum, (int, float)):
            if volatility > 50 and momentum > 5:
                return 'high_momentum_high_vol'
            elif market_cap == 'large' or (isinstance(volatility, (int, float)) and volatility < 30):
                return 'quality_large_cap'

        return 'default'

    # =========================================================================
    # CAPITAL ALLOCATION
    # =========================================================================

    def _calculate_capital_allocation(
        self,
        vix_level: float,
        transition_phase: TransitionPhase
    ) -> Dict[str, float]:
        """
        Calculate optimal capital allocation during transition.

        Base allocation:
        - 40% early movers (day 0-5)
        - 30% confirmation plays (day 6-10)
        - 20% momentum continuation (day 11-15)
        - 10% cash reserve

        Adjusted by VIX level.
        """
        # Base allocation
        allocation = {
            'early_transition_tier': 0.40,
            'confirmation_tier': 0.30,
            'momentum_tier': 0.20,
            'cash_reserve': 0.10
        }

        # Adjust based on VIX
        if vix_level < 15:
            # Low volatility = more aggressive
            allocation['early_transition_tier'] = 0.50
            allocation['cash_reserve'] = 0.05
            allocation['confirmation_tier'] = 0.30
            allocation['momentum_tier'] = 0.15
        elif vix_level > 25:
            # High volatility = more conservative
            allocation['early_transition_tier'] = 0.25
            allocation['cash_reserve'] = 0.20
            allocation['confirmation_tier'] = 0.35
            allocation['momentum_tier'] = 0.20
        elif vix_level > 30:
            # Crisis mode = very conservative
            allocation['early_transition_tier'] = 0.15
            allocation['cash_reserve'] = 0.35
            allocation['confirmation_tier'] = 0.30
            allocation['momentum_tier'] = 0.20

        # Adjust based on transition phase
        if transition_phase == TransitionPhase.EARLY_TRANSITION:
            # Emphasize early tier
            allocation['early_transition_tier'] *= 1.2
            allocation['momentum_tier'] *= 0.8
        elif transition_phase == TransitionPhase.LATE_TRANSITION:
            # Emphasize confirmation
            allocation['confirmation_tier'] *= 1.2
            allocation['early_transition_tier'] *= 0.8

        # Normalize to 100%
        total = sum(allocation.values())
        allocation = {k: v / total for k, v in allocation.items()}

        return allocation

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def _calculate_position_size(
        self,
        profit_score: float,
        dynamic_multiplier: float,
        max_position: float,
        sizing_method: str
    ) -> float:
        """
        Calculate recommended position size.

        Methods:
        - kelly_criterion: Full Kelly formula
        - half_kelly: Conservative Kelly
        - equal_weight: Simple equal weighting
        """
        # Base win probability from profit score
        win_probability = min(0.95, 0.5 + (profit_score - 50) * 0.01)

        # Assumed win/loss ratio based on profit score
        win_loss_ratio = 1.5 + (profit_score - 50) * 0.02

        if sizing_method == 'kelly_criterion':
            # Full Kelly: f = (p * b - q) / b
            # where p = win prob, q = 1-p, b = win/loss ratio
            kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
            optimal_size = max(0.02, kelly * dynamic_multiplier)

        elif sizing_method == 'half_kelly':
            # Half Kelly for risk management
            kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
            optimal_size = max(0.02, (kelly * 0.5) * dynamic_multiplier)

        else:  # equal_weight
            optimal_size = 0.05 * dynamic_multiplier

        return min(max_position, optimal_size)

    # =========================================================================
    # PROFIT ASYMMETRY ANALYSIS
    # =========================================================================

    def find_profit_asymmetry(
        self,
        stocks_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find stocks with highest upside/downside asymmetry.

        Returns stocks with 2.5:1+ reward:risk ratio.
        """
        high_asymmetry_stocks = []

        for stock in stocks_data:
            ticker = stock.get('ticker', 'UNKNOWN')

            # Calculate upside potential
            momentum = stock.get('5d_return', stock.get('ret_5d', 0))
            volatility = stock.get('volatility', stock.get('vol', 30))
            confidence = stock.get('confidence', stock.get('conf', 70))

            # Convert if needed
            if isinstance(volatility, (int, float)) and volatility < 1:
                volatility = volatility * 100
            if isinstance(momentum, (int, float)) and abs(momentum) < 1:
                momentum = momentum * 100

            # Upside = momentum potential * confidence
            upside = abs(momentum) * (confidence / 100) * 1.5

            # Downside = volatility-based risk
            downside = volatility * 0.3

            # Asymmetry ratio
            asymmetry_ratio = upside / max(downside, 1.0)

            if asymmetry_ratio > 2.0:  # 2:1 minimum
                high_asymmetry_stocks.append({
                    'ticker': ticker,
                    'asymmetry_ratio': round(asymmetry_ratio, 2),
                    'upside_potential': round(upside, 2),
                    'downside_risk': round(downside, 2),
                    'optimal_position_size': min(0.15, 0.05 * asymmetry_ratio),
                    'profit_target': upside * 0.7,
                    'stop_loss': downside * 0.5,
                    'raw_data': stock
                })

        return sorted(high_asymmetry_stocks, key=lambda x: x['asymmetry_ratio'], reverse=True)

    # =========================================================================
    # GENERATE MAXIMUM PROFIT SIGNALS
    # =========================================================================

    def generate_profit_optimized_signals(
        self,
        spy_data: pd.DataFrame,
        stocks_data: List[Dict[str, Any]],
        vix_level: float = 20.0
    ) -> Dict[str, Any]:
        """
        Generate profit-optimized signals for a list of stocks.

        Returns:
            Dict with regime info and ranked stock signals
        """
        # Get base regime with profit optimization
        regime_output = self.classify_regime_with_profit(
            spy_data, None, vix_level
        )

        # Score and rank all stocks
        scored_stocks = []
        for stock in stocks_data:
            # Calculate profit score
            profit_score = self._calculate_profit_score(stock)

            # Get dynamic multiplier
            dynamic_mult = self._get_dynamic_multiplier(
                profit_score, regime_output.base_output.buy_multiplier
            )

            # Calculate position size
            position_size = self._calculate_position_size(
                profit_score,
                dynamic_mult,
                regime_output.max_position_size,
                regime_output.position_sizing_method
            )

            # Get profit targets
            stock_type = self._classify_stock_type(stock)
            targets = self.PROFIT_TARGETS.get(stock_type, self.PROFIT_TARGETS['default'])

            scored_stocks.append({
                'ticker': stock.get('ticker', 'UNKNOWN'),
                'profit_score': round(profit_score, 1),
                'dynamic_multiplier': round(dynamic_mult, 3),
                'position_size': round(position_size, 4),
                'stock_type': stock_type,
                'profit_targets': targets,
                'original_data': stock
            })

        # Sort by profit score
        scored_stocks.sort(key=lambda x: x['profit_score'], reverse=True)

        # Find high asymmetry opportunities
        asymmetry_stocks = self.find_profit_asymmetry(stocks_data)

        return {
            'regime': {
                'primary': regime_output.regime,
                'transition_state': regime_output.transition_state,
                'transition_phase': regime_output.transition_phase.value,
                'transition_day': regime_output.transition_day,
                'confidence': regime_output.base_output.confidence
            },
            'allocation': regime_output.capital_allocation,
            'sector_focus': regime_output.sector_focus,
            'position_params': {
                'max_positions': regime_output.max_positions,
                'top_n_concentration': regime_output.top_n_concentration,
                'sizing_method': regime_output.position_sizing_method
            },
            'ranked_stocks': scored_stocks,
            'high_asymmetry': asymmetry_stocks[:5],
            'phase_multipliers': {
                'small_cap': regime_output.small_cap_multiplier,
                'large_cap': regime_output.large_cap_multiplier
            }
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_profit_optimized_regime(
    spy_data: pd.DataFrame,
    stock_data: Optional[Dict[str, Any]] = None,
    vix_level: float = 20.0
) -> ProfitMaximizingOutput:
    """
    Convenience function to get profit-optimized regime classification.

    Usage:
        result = get_profit_optimized_regime(spy_data, stock_data, vix_level)
        print(f"Dynamic multiplier: {result.dynamic_multiplier}")
        print(f"Position size: {result.recommended_position_size}")
    """
    classifier = ProfitMaximizingRegimeClassifier(initial_vix=vix_level)
    return classifier.classify_regime_with_profit(spy_data, stock_data, vix_level)


# =============================================================================
# MAIN / EXAMPLE
# =============================================================================

def main():
    """Example usage of ProfitMaximizingRegimeClassifier."""
    import yfinance as yf

    print("=" * 70)
    print("PROFIT-MAXIMIZING REGIME CLASSIFIER - EXAMPLE")
    print("=" * 70)

    # Download SPY data
    print("\nDownloading SPY data...")
    spy = yf.download('SPY', period='6mo', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Download VIX
    vix = yf.download('^VIX', period='1mo', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    current_vix = float(vix['Close'].iloc[-1])

    print(f"SPY data: {len(spy)} days")
    print(f"Current VIX: {current_vix:.1f}")

    # Initialize classifier
    classifier = ProfitMaximizingRegimeClassifier(initial_vix=current_vix)

    # Example stock data
    example_stock = {
        'ticker': 'TSLA',
        '5d_return': 11.4,
        'volume_ratio': 1.3,
        'rsi': 65,
        'volatility': 41,
        'market_cap': 'large'
    }

    # Get profit-optimized regime
    result = classifier.classify_regime_with_profit(spy, example_stock, current_vix)

    print(f"\n{'='*50}")
    print("REGIME ANALYSIS")
    print(f"{'='*50}")
    print(f"Regime: {result.regime}")
    print(f"Transition State: {result.transition_state}")
    print(f"Transition Phase: {result.transition_phase.value}")
    print(f"Transition Day: {result.transition_day}")

    print(f"\n{'='*50}")
    print("PROFIT OPTIMIZATION")
    print(f"{'='*50}")
    print(f"Profit Score: {result.profit_score:.1f}/100")
    print(f"Dynamic Multiplier: {result.dynamic_multiplier:.3f}x")
    print(f"Recommended Position: {result.recommended_position_size:.1%}")

    print(f"\n{'='*50}")
    print("POSITION SIZING")
    print(f"{'='*50}")
    print(f"Small-Cap Multiplier: {result.small_cap_multiplier:.2f}x")
    print(f"Large-Cap Multiplier: {result.large_cap_multiplier:.2f}x")
    print(f"Max Position Size: {result.max_position_size:.1%}")
    print(f"Stop Loss: {result.stop_loss:.1%}")

    print(f"\n{'='*50}")
    print("PROFIT TARGETS")
    print(f"{'='*50}")
    print(f"Target 1: {result.profit_target_1:.1%}")
    print(f"Target 2: {result.profit_target_2:.1%}")
    print(f"Target 3: {result.profit_target_3:.1%}")
    print(f"Trailing Stop: {result.trailing_stop:.1%}")

    print(f"\n{'='*50}")
    print("CAPITAL ALLOCATION")
    print(f"{'='*50}")
    for tier, alloc in result.capital_allocation.items():
        print(f"  {tier}: {alloc:.1%}")

    print(f"\nSector Focus: {result.sector_focus}")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
