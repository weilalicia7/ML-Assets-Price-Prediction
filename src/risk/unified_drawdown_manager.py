"""
Unified Drawdown Manager - Phase 2 Conflict Resolution

Resolves conflicts between Phase 1 and Phase 2 drawdown control thresholds.
Phase 2 thresholds take precedence (more conservative):
- Phase 1: 8% warning, 20% max
- Phase 2: 5% warning, 10% danger, 15% max (USED)

Based on: phase2 fixing on C model_conflict resolutions.pdf
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class UnifiedDrawdownManager:
    """
    Unified drawdown management combining Phase 1 and Phase 2 approaches.

    Phase 2 conservative thresholds take precedence:
    - 5% warning (vs Phase 1's 8%)
    - 10% danger (new in Phase 2)
    - 15% max (vs Phase 1's 20%)

    Features:
    - Stepped position multipliers (Phase 2) instead of linear (Phase 1)
    - Velocity-based protection for rapid drawdowns
    - Circuit breaker for extreme conditions
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,    # Phase 2: 5% (vs Phase 1: 8%)
        danger_threshold: float = 0.10,     # Phase 2: 10% (new)
        max_threshold: float = 0.15,        # Phase 2: 15% (vs Phase 1: 20%)
        circuit_breaker: float = 0.12,      # Emergency stop
        velocity_window: int = 5            # Days to measure drawdown velocity
    ):
        """
        Initialize unified drawdown manager.

        Args:
            warning_threshold: Start reducing positions (Phase 2: 5%)
            danger_threshold: Significant reduction (Phase 2: 10%)
            max_threshold: Stop trading entirely (Phase 2: 15%)
            circuit_breaker: Emergency halt threshold
            velocity_window: Window for measuring drawdown velocity
        """
        # Phase 2 thresholds (conservative)
        self.drawdown_thresholds = {
            'warning': warning_threshold,     # 0.05
            'danger': danger_threshold,       # 0.10
            'max': max_threshold,             # 0.15
            'circuit_breaker': circuit_breaker  # 0.12
        }

        # Phase 2 stepped multipliers (vs Phase 1 linear reduction)
        self.position_multipliers = {
            'normal': 1.0,      # No drawdown
            'warning': 0.7,     # 5-10% drawdown
            'danger': 0.3,      # 10-15% drawdown
            'critical': 0.0     # >15% drawdown
        }

        # Velocity-based protection
        self.velocity_window = velocity_window
        self.daily_returns = deque(maxlen=velocity_window)
        self.velocity_multipliers = {
            'normal': 1.0,      # Normal conditions
            'fast': 0.7,        # >1% daily loss
            'rapid': 0.5        # >2% daily loss
        }

        # Tracking
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        self.consecutive_loss_days = 0

        logger.info(f"Initialized UnifiedDrawdownManager (Phase 2 thresholds):")
        logger.info(f"  Warning: {warning_threshold:.1%}, Danger: {danger_threshold:.1%}, Max: {max_threshold:.1%}")

    def update(self, portfolio_value: float, daily_return: float = 0.0) -> Dict:
        """
        Update drawdown state with new portfolio value.

        Args:
            portfolio_value: Current portfolio value
            daily_return: Today's return for velocity calculation

        Returns:
            Dict with drawdown state and recommended multiplier
        """
        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.consecutive_loss_days = 0
        else:
            if daily_return < 0:
                self.consecutive_loss_days += 1

        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        # Track daily returns for velocity
        self.daily_returns.append(daily_return)

        # Determine drawdown state using Phase 2 thresholds
        state = self._get_drawdown_state()

        # Calculate position multiplier
        position_multiplier = self._calculate_position_multiplier(state)

        # Apply velocity-based adjustment
        velocity_multiplier = self._get_velocity_multiplier()
        final_multiplier = position_multiplier * velocity_multiplier

        # Check circuit breaker
        if self.current_drawdown >= self.drawdown_thresholds['circuit_breaker']:
            self.circuit_breaker_triggered = True
            final_multiplier = 0.0
            logger.warning(f"CIRCUIT BREAKER TRIGGERED: {self.current_drawdown:.1%} drawdown")

        return {
            'drawdown': self.current_drawdown,
            'state': state,
            'position_multiplier': final_multiplier,
            'base_multiplier': position_multiplier,
            'velocity_multiplier': velocity_multiplier,
            'circuit_breaker': self.circuit_breaker_triggered,
            'consecutive_loss_days': self.consecutive_loss_days,
            'can_trade': final_multiplier > 0 and not self.circuit_breaker_triggered
        }

    def _get_drawdown_state(self) -> str:
        """Determine drawdown state based on Phase 2 thresholds."""
        dd = self.current_drawdown

        if dd >= self.drawdown_thresholds['max']:
            return 'critical'
        elif dd >= self.drawdown_thresholds['danger']:
            return 'danger'
        elif dd >= self.drawdown_thresholds['warning']:
            return 'warning'
        else:
            return 'normal'

    def _calculate_position_multiplier(self, state: str) -> float:
        """
        Calculate position multiplier using Phase 2 stepped approach.

        Phase 2 stepped multipliers replace Phase 1 linear reduction:
        - normal: 1.0 (full position)
        - warning: 0.7 (70% position)
        - danger: 0.3 (30% position)
        - critical: 0.0 (no trading)
        """
        return self.position_multipliers.get(state, 1.0)

    def _get_velocity_multiplier(self) -> float:
        """
        Calculate velocity-based multiplier for rapid drawdowns.

        Additional protection when losses are happening quickly.
        """
        if len(self.daily_returns) < 2:
            return 1.0

        # Calculate recent daily loss rate
        recent_loss = -min(0, sum(self.daily_returns) / len(self.daily_returns))

        if recent_loss >= 0.02:  # >2% average daily loss
            return self.velocity_multipliers['rapid']  # 0.5
        elif recent_loss >= 0.01:  # >1% average daily loss
            return self.velocity_multipliers['fast']   # 0.7
        else:
            return self.velocity_multipliers['normal']  # 1.0

    def get_position_size(
        self,
        base_position: float,
        portfolio_value: float,
        daily_return: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Get adjusted position size based on drawdown state.

        Args:
            base_position: Desired position size before drawdown adjustment
            portfolio_value: Current portfolio value
            daily_return: Today's return

        Returns:
            Tuple of (adjusted_position, state_dict)
        """
        state = self.update(portfolio_value, daily_return)
        adjusted_position = base_position * state['position_multiplier']

        if state['state'] != 'normal':
            logger.info(
                f"Drawdown adjustment: {self.current_drawdown:.1%} drawdown, "
                f"state={state['state']}, multiplier={state['position_multiplier']:.2f}"
            )

        return adjusted_position, state

    def reset_circuit_breaker(self):
        """Reset circuit breaker after recovery (manual action required)."""
        self.circuit_breaker_triggered = False
        logger.info("Circuit breaker reset")

    def reset(self):
        """Reset all tracking state."""
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        self.consecutive_loss_days = 0
        self.daily_returns.clear()
        logger.info("Drawdown manager reset")


# Convenience function for quick usage
def get_unified_drawdown_manager(**kwargs) -> UnifiedDrawdownManager:
    """Get a configured UnifiedDrawdownManager instance."""
    return UnifiedDrawdownManager(**kwargs)
