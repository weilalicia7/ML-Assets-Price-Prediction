"""
Phase 6: Regulatory Compliance Manager

This module ensures portfolio adherence to regulatory requirements,
position limits, and internal policies.

Components:
- PositionLimitManager: Manages position and concentration limits
- ConcentrationMonitor: Monitors portfolio concentration metrics
- ComplianceChecker: Pre/post-trade compliance validation
- AuditTrailManager: Maintains audit trail for compliance

Expected Impact: Risk reduction, regulatory safety
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json
import uuid


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    BLOCKED = "blocked"


class AlertSeverity(Enum):
    """Severity levels for compliance alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComplianceConfig:
    """Configuration for compliance management."""
    max_single_position_pct: float = 0.20      # 20% max single position
    max_sector_concentration_pct: float = 0.40 # 40% max sector
    max_hhi: float = 0.25                       # Maximum HHI allowed
    min_effective_n: int = 5                    # Minimum effective positions
    min_liquidity_score: float = 0.3           # Minimum liquidity for trading
    max_daily_turnover: float = 0.25           # 25% max daily turnover
    max_leverage: float = 1.0                   # No leverage by default
    audit_retention_days: int = 365 * 7        # 7 years retention


@dataclass
class ComplianceAlert:
    """A compliance alert."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    details: Dict
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class TradeComplianceResult:
    """Result of trade compliance check."""
    is_compliant: bool
    status: ComplianceStatus
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    blocking_issues: List[str]
    recommended_adjustments: Dict[str, float]


@dataclass
class AuditEntry:
    """An audit trail entry."""
    entry_id: str
    timestamp: datetime
    action_type: str              # 'trade', 'rebalance', 'harvest', 'check'
    ticker: Optional[str]
    quantity: Optional[float]
    rationale: str
    compliance_checks: Dict       # All checks performed
    approved_by: str              # 'system' or user ID
    risk_metrics: Dict            # Portfolio risk at time of action
    outcome: str                  # 'executed', 'blocked', 'modified'


# =============================================================================
# 1. Position Limit Manager
# =============================================================================

class PositionLimitManager:
    """
    Manages position limits and concentration constraints.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize manager.

        Args:
            config: Compliance configuration
        """
        self.config = config or ComplianceConfig()
        self.position_limits: Dict[str, float] = {}       # Ticker-specific limits
        self.sector_limits: Dict[str, float] = {}         # Sector limits
        self.ticker_sectors: Dict[str, str] = {}          # Ticker to sector mapping

    def set_position_limit(self, ticker: str, limit: float):
        """Set custom position limit for a ticker."""
        self.position_limits[ticker] = limit

    def set_sector_limit(self, sector: str, limit: float):
        """Set custom sector limit."""
        self.sector_limits[sector] = limit

    def assign_ticker_sector(self, ticker: str, sector: str):
        """Assign a ticker to a sector."""
        self.ticker_sectors[ticker] = sector

    def get_position_limit(self, ticker: str) -> float:
        """Get position limit for a ticker."""
        return self.position_limits.get(ticker, self.config.max_single_position_pct)

    def get_sector_limit(self, sector: str) -> float:
        """Get limit for a sector."""
        return self.sector_limits.get(sector, self.config.max_sector_concentration_pct)

    def check_position_limits(
        self,
        current_weights: Dict[str, float],
        proposed_trade: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if position limits are satisfied.

        Args:
            current_weights: Current portfolio weights
            proposed_trade: Optional proposed weight changes

        Returns:
            Tuple of (is_compliant, violations)
        """
        # Calculate post-trade weights
        if proposed_trade:
            weights = current_weights.copy()
            for ticker, change in proposed_trade.items():
                weights[ticker] = weights.get(ticker, 0.0) + change
        else:
            weights = current_weights

        violations = []

        for ticker, weight in weights.items():
            limit = self.get_position_limit(ticker)
            if weight > limit:
                violations.append(
                    f"{ticker}: {weight:.1%} exceeds limit {limit:.1%}"
                )

        return len(violations) == 0, violations

    def check_sector_limits(
        self,
        current_weights: Dict[str, float],
        proposed_trade: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if sector concentration limits are satisfied.

        Args:
            current_weights: Current portfolio weights
            proposed_trade: Optional proposed weight changes

        Returns:
            Tuple of (is_compliant, violations)
        """
        # Calculate post-trade weights
        if proposed_trade:
            weights = current_weights.copy()
            for ticker, change in proposed_trade.items():
                weights[ticker] = weights.get(ticker, 0.0) + change
        else:
            weights = current_weights

        # Aggregate by sector
        sector_weights: Dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = self.ticker_sectors.get(ticker, 'unknown')
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        violations = []

        for sector, weight in sector_weights.items():
            limit = self.get_sector_limit(sector)
            if weight > limit:
                violations.append(
                    f"Sector {sector}: {weight:.1%} exceeds limit {limit:.1%}"
                )

        return len(violations) == 0, violations

    def get_remaining_capacity(
        self,
        ticker: str,
        current_weight: float
    ) -> float:
        """
        Get remaining capacity for a position.

        Args:
            ticker: Ticker symbol
            current_weight: Current position weight

        Returns:
            Remaining capacity as weight
        """
        limit = self.get_position_limit(ticker)
        return max(0, limit - current_weight)

    def calculate_concentration_risk(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate concentration risk metrics.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary of concentration metrics
        """
        weight_values = list(weights.values())

        if not weight_values:
            return {
                'hhi': 0.0,
                'effective_n': 0,
                'max_position': 0.0,
                'top_3_concentration': 0.0,
                'top_5_concentration': 0.0
            }

        # HHI (Herfindahl-Hirschman Index)
        hhi = sum(w ** 2 for w in weight_values)

        # Effective N
        effective_n = 1 / hhi if hhi > 0 else len(weight_values)

        # Top concentrations
        sorted_weights = sorted(weight_values, reverse=True)
        top_3 = sum(sorted_weights[:3])
        top_5 = sum(sorted_weights[:5])

        return {
            'hhi': hhi,
            'effective_n': effective_n,
            'max_position': max(weight_values),
            'top_3_concentration': top_3,
            'top_5_concentration': top_5
        }


# =============================================================================
# 2. Concentration Monitor
# =============================================================================

class ConcentrationMonitor:
    """
    Monitors portfolio concentration metrics in real-time.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize monitor.

        Args:
            config: Compliance configuration
        """
        self.config = config or ComplianceConfig()
        self.concentration_history: deque = deque(maxlen=252)  # 1 year daily
        self.alerts: List[ComplianceAlert] = []

    def calculate_herfindahl_index(
        self,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index.

        Args:
            weights: Portfolio weights

        Returns:
            HHI value (0 to 1)
        """
        weight_values = list(weights.values())
        if not weight_values:
            return 0.0
        return sum(w ** 2 for w in weight_values)

    def calculate_effective_n(
        self,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate effective number of positions.

        Args:
            weights: Portfolio weights

        Returns:
            Effective N (inverse of HHI)
        """
        hhi = self.calculate_herfindahl_index(weights)
        return 1 / hhi if hhi > 0 else len(weights)

    def check_concentration_limits(
        self,
        weights: Dict[str, float]
    ) -> Tuple[ComplianceStatus, List[str]]:
        """
        Check concentration against limits.

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (status, messages)
        """
        hhi = self.calculate_herfindahl_index(weights)
        effective_n = self.calculate_effective_n(weights)
        max_position = max(weights.values()) if weights else 0

        messages = []
        status = ComplianceStatus.COMPLIANT

        # Check HHI
        if hhi > self.config.max_hhi:
            status = ComplianceStatus.VIOLATION
            messages.append(f"HHI {hhi:.3f} exceeds limit {self.config.max_hhi}")
        elif hhi > self.config.max_hhi * 0.8:
            if status == ComplianceStatus.COMPLIANT:
                status = ComplianceStatus.WARNING
            messages.append(f"HHI {hhi:.3f} approaching limit {self.config.max_hhi}")

        # Check effective N
        if effective_n < self.config.min_effective_n:
            status = ComplianceStatus.VIOLATION
            messages.append(
                f"Effective N {effective_n:.1f} below minimum {self.config.min_effective_n}"
            )

        # Check max position
        if max_position > self.config.max_single_position_pct:
            status = ComplianceStatus.VIOLATION
            messages.append(
                f"Max position {max_position:.1%} exceeds limit "
                f"{self.config.max_single_position_pct:.0%}"
            )

        # Record
        self.concentration_history.append({
            'timestamp': datetime.now(),
            'hhi': hhi,
            'effective_n': effective_n,
            'max_position': max_position,
            'status': status.value
        })

        return status, messages

    def suggest_diversification(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Suggest weight adjustments for better diversification.

        Args:
            weights: Current portfolio weights

        Returns:
            Suggested weight adjustments
        """
        if not weights:
            return {}

        adjustments = {}
        n = len(weights)
        equal_weight = 1.0 / n
        max_position = self.config.max_single_position_pct

        for ticker, weight in weights.items():
            if weight > max_position:
                # Suggest reducing to limit
                adjustments[ticker] = max_position - weight
            elif weight < equal_weight * 0.5:
                # Suggest increasing very small positions
                adjustments[ticker] = equal_weight - weight

        return adjustments

    def generate_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        details: Dict
    ) -> ComplianceAlert:
        """Generate and store a compliance alert."""
        alert = ComplianceAlert(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            details=details
        )
        self.alerts.append(alert)
        return alert

    def get_active_alerts(self) -> List[ComplianceAlert]:
        """Get unresolved alerts."""
        return [a for a in self.alerts if not a.resolved]


# =============================================================================
# 3. Compliance Checker
# =============================================================================

class ComplianceChecker:
    """
    Performs pre-trade and post-trade compliance validation.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize checker.

        Args:
            config: Compliance configuration
        """
        self.config = config or ComplianceConfig()
        self.position_manager = PositionLimitManager(config)
        self.concentration_monitor = ConcentrationMonitor(config)
        self.restricted_list: Set[str] = set()
        self.watchlist: Set[str] = set()

    def add_to_restricted_list(self, ticker: str):
        """Add ticker to restricted list (no trading allowed)."""
        self.restricted_list.add(ticker)

    def remove_from_restricted_list(self, ticker: str):
        """Remove ticker from restricted list."""
        self.restricted_list.discard(ticker)

    def add_to_watchlist(self, ticker: str):
        """Add ticker to watchlist (trading allowed with warning)."""
        self.watchlist.add(ticker)

    def pre_trade_check(
        self,
        ticker: str,
        trade_direction: str,         # 'buy' or 'sell'
        trade_weight: float,          # Weight change (positive for buy)
        current_weights: Dict[str, float],
        liquidity_score: Optional[float] = None
    ) -> TradeComplianceResult:
        """
        Perform pre-trade compliance check.

        Args:
            ticker: Ticker symbol
            trade_direction: 'buy' or 'sell'
            trade_weight: Weight change
            current_weights: Current portfolio weights
            liquidity_score: Optional liquidity score

        Returns:
            TradeComplianceResult
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        blocking_issues = []
        adjustments = {}

        # 1. Restricted list check
        if ticker in self.restricted_list:
            blocking_issues.append(f"{ticker} is on restricted list")
            checks_failed.append("restricted_list")
        else:
            checks_passed.append("restricted_list")

        # 2. Watchlist check
        if ticker in self.watchlist:
            warnings.append(f"{ticker} is on watchlist - proceed with caution")

        # 3. Position limit check
        proposed_trade = {ticker: trade_weight if trade_direction == 'buy' else -abs(trade_weight)}
        pos_ok, pos_violations = self.position_manager.check_position_limits(
            current_weights, proposed_trade
        )
        if pos_ok:
            checks_passed.append("position_limits")
        else:
            checks_failed.append("position_limits")
            for v in pos_violations:
                warnings.append(v)

            # Calculate adjustment
            current = current_weights.get(ticker, 0.0)
            limit = self.position_manager.get_position_limit(ticker)
            max_allowed = limit - current
            adjustments[ticker] = max_allowed

        # 4. Sector limit check
        sector_ok, sector_violations = self.position_manager.check_sector_limits(
            current_weights, proposed_trade
        )
        if sector_ok:
            checks_passed.append("sector_limits")
        else:
            checks_failed.append("sector_limits")
            for v in sector_violations:
                warnings.append(v)

        # 5. Concentration check (post-trade)
        post_trade_weights = current_weights.copy()
        post_trade_weights[ticker] = post_trade_weights.get(ticker, 0.0) + \
            (trade_weight if trade_direction == 'buy' else -abs(trade_weight))

        conc_status, conc_messages = self.concentration_monitor.check_concentration_limits(
            post_trade_weights
        )
        if conc_status == ComplianceStatus.COMPLIANT:
            checks_passed.append("concentration")
        elif conc_status == ComplianceStatus.WARNING:
            checks_passed.append("concentration")
            warnings.extend(conc_messages)
        else:
            checks_failed.append("concentration")
            warnings.extend(conc_messages)

        # 6. Liquidity check
        if liquidity_score is not None:
            if liquidity_score >= self.config.min_liquidity_score:
                checks_passed.append("liquidity")
            else:
                warnings.append(
                    f"Liquidity score {liquidity_score:.2f} below minimum "
                    f"{self.config.min_liquidity_score}"
                )
                if liquidity_score < self.config.min_liquidity_score * 0.5:
                    checks_failed.append("liquidity")
                    blocking_issues.append("Insufficient liquidity")

        # Determine overall status
        if blocking_issues:
            status = ComplianceStatus.BLOCKED
            is_compliant = False
        elif checks_failed:
            status = ComplianceStatus.VIOLATION
            is_compliant = False
        elif warnings:
            status = ComplianceStatus.WARNING
            is_compliant = True
        else:
            status = ComplianceStatus.COMPLIANT
            is_compliant = True

        return TradeComplianceResult(
            is_compliant=is_compliant,
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            blocking_issues=blocking_issues,
            recommended_adjustments=adjustments
        )

    def post_trade_validation(
        self,
        executed_trade: Dict,
        portfolio_weights: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio after trade execution.

        Args:
            executed_trade: Details of executed trade
            portfolio_weights: Current portfolio weights

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        # Check all limits
        pos_ok, pos_issues = self.position_manager.check_position_limits(portfolio_weights)
        if not pos_ok:
            issues.extend(pos_issues)

        sector_ok, sector_issues = self.position_manager.check_sector_limits(portfolio_weights)
        if not sector_ok:
            issues.extend(sector_issues)

        conc_status, conc_issues = self.concentration_monitor.check_concentration_limits(
            portfolio_weights
        )
        if conc_status == ComplianceStatus.VIOLATION:
            issues.extend(conc_issues)

        return len(issues) == 0, issues

    def generate_compliance_report(
        self,
        portfolio_weights: Dict[str, float],
        period_start: datetime,
        period_end: datetime
    ) -> Dict:
        """
        Generate compliance report for a period.

        Args:
            portfolio_weights: Current portfolio weights
            period_start: Report period start
            period_end: Report period end

        Returns:
            Compliance report dictionary
        """
        # Current state
        concentration = self.position_manager.calculate_concentration_risk(portfolio_weights)
        conc_status, conc_messages = self.concentration_monitor.check_concentration_limits(
            portfolio_weights
        )

        # Alerts during period
        period_alerts = [
            a for a in self.concentration_monitor.alerts
            if period_start <= a.timestamp <= period_end
        ]

        return {
            'report_period': {
                'start': period_start.isoformat(),
                'end': period_end.isoformat()
            },
            'current_status': conc_status.value,
            'concentration_metrics': concentration,
            'position_count': len(portfolio_weights),
            'restricted_list_size': len(self.restricted_list),
            'watchlist_size': len(self.watchlist),
            'alerts_during_period': len(period_alerts),
            'unresolved_alerts': len(self.concentration_monitor.get_active_alerts()),
            'messages': conc_messages
        }


# =============================================================================
# 4. Audit Trail Manager
# =============================================================================

class AuditTrailManager:
    """
    Maintains audit trail for compliance and regulatory purposes.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize manager.

        Args:
            config: Compliance configuration
        """
        self.config = config or ComplianceConfig()
        self.entries: List[AuditEntry] = []
        self.retention_cutoff = datetime.now() - timedelta(
            days=self.config.audit_retention_days
        )

    def log_decision(
        self,
        decision_type: str,
        details: Dict,
        rationale: str,
        risk_metrics: Optional[Dict] = None
    ) -> AuditEntry:
        """
        Log a decision for audit purposes.

        Args:
            decision_type: Type of decision
            details: Decision details
            rationale: Reason for decision
            risk_metrics: Current risk metrics

        Returns:
            Created AuditEntry
        """
        entry = AuditEntry(
            entry_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            action_type=decision_type,
            ticker=details.get('ticker'),
            quantity=details.get('quantity'),
            rationale=rationale,
            compliance_checks=details.get('compliance_checks', {}),
            approved_by=details.get('approved_by', 'system'),
            risk_metrics=risk_metrics or {},
            outcome=details.get('outcome', 'pending')
        )

        self.entries.append(entry)
        self._cleanup_old_entries()

        return entry

    def log_trade(
        self,
        ticker: str,
        quantity: float,
        direction: str,
        compliance_result: TradeComplianceResult,
        risk_metrics: Dict,
        outcome: str
    ) -> AuditEntry:
        """
        Log a trade for audit purposes.

        Args:
            ticker: Ticker symbol
            quantity: Trade quantity
            direction: 'buy' or 'sell'
            compliance_result: Compliance check result
            risk_metrics: Risk metrics at trade time
            outcome: Trade outcome

        Returns:
            Created AuditEntry
        """
        return self.log_decision(
            decision_type='trade',
            details={
                'ticker': ticker,
                'quantity': quantity if direction == 'buy' else -quantity,
                'direction': direction,
                'compliance_checks': {
                    'status': compliance_result.status.value,
                    'passed': compliance_result.checks_passed,
                    'failed': compliance_result.checks_failed,
                    'warnings': compliance_result.warnings
                },
                'outcome': outcome
            },
            rationale=f"{direction.upper()} {ticker}: {', '.join(compliance_result.checks_passed)}",
            risk_metrics=risk_metrics
        )

    def _cleanup_old_entries(self):
        """Remove entries older than retention period."""
        self.entries = [
            e for e in self.entries
            if e.timestamp >= self.retention_cutoff
        ]

    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate audit report for a period.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Audit report dictionary
        """
        period_entries = [
            e for e in self.entries
            if start_date <= e.timestamp <= end_date
        ]

        # Aggregate by action type
        by_type = {}
        for entry in period_entries:
            by_type[entry.action_type] = by_type.get(entry.action_type, 0) + 1

        # Aggregate by outcome
        by_outcome = {}
        for entry in period_entries:
            by_outcome[entry.outcome] = by_outcome.get(entry.outcome, 0) + 1

        # Blocked actions
        blocked = [e for e in period_entries if e.outcome == 'blocked']

        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_entries': len(period_entries),
            'by_action_type': by_type,
            'by_outcome': by_outcome,
            'blocked_actions': len(blocked),
            'blocked_details': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'ticker': e.ticker,
                    'rationale': e.rationale
                }
                for e in blocked[:10]  # Limit to 10
            ]
        }

    def export_for_regulatory(
        self,
        format: str = 'json',
        period_days: int = 365
    ) -> str:
        """
        Export audit trail for regulatory purposes.

        Args:
            format: Export format ('json' or 'csv')
            period_days: Number of days to export

        Returns:
            Exported data as string
        """
        cutoff = datetime.now() - timedelta(days=period_days)
        entries = [e for e in self.entries if e.timestamp >= cutoff]

        if format == 'json':
            data = [
                {
                    'entry_id': e.entry_id,
                    'timestamp': e.timestamp.isoformat(),
                    'action_type': e.action_type,
                    'ticker': e.ticker,
                    'quantity': e.quantity,
                    'rationale': e.rationale,
                    'outcome': e.outcome,
                    'approved_by': e.approved_by
                }
                for e in entries
            ]
            return json.dumps(data, indent=2)
        else:
            # CSV format
            lines = ['entry_id,timestamp,action_type,ticker,quantity,outcome']
            for e in entries:
                lines.append(
                    f"{e.entry_id},{e.timestamp.isoformat()},{e.action_type},"
                    f"{e.ticker or ''},{e.quantity or ''},{e.outcome}"
                )
            return '\n'.join(lines)


# =============================================================================
# 5. Integrated Compliance Manager
# =============================================================================

class ComplianceManager:
    """
    Integrated compliance management system.

    Combines position limits, concentration monitoring, compliance checking,
    and audit trail management.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        """
        Initialize manager.

        Args:
            config: Compliance configuration
        """
        self.config = config or ComplianceConfig()
        self.position_manager = PositionLimitManager(config)
        self.concentration_monitor = ConcentrationMonitor(config)
        self.checker = ComplianceChecker(config)
        self.audit_trail = AuditTrailManager(config)

    def validate_trade(
        self,
        ticker: str,
        direction: str,
        weight_change: float,
        current_weights: Dict[str, float],
        liquidity_score: Optional[float] = None
    ) -> TradeComplianceResult:
        """
        Validate a proposed trade.

        Args:
            ticker: Ticker symbol
            direction: 'buy' or 'sell'
            weight_change: Weight change
            current_weights: Current weights
            liquidity_score: Optional liquidity score

        Returns:
            TradeComplianceResult
        """
        result = self.checker.pre_trade_check(
            ticker, direction, weight_change, current_weights, liquidity_score
        )

        # Log the check
        self.audit_trail.log_decision(
            decision_type='compliance_check',
            details={
                'ticker': ticker,
                'direction': direction,
                'weight_change': weight_change,
                'compliance_checks': {
                    'passed': result.checks_passed,
                    'failed': result.checks_failed
                },
                'outcome': 'approved' if result.is_compliant else 'flagged'
            },
            rationale=f"Pre-trade check for {direction} {ticker}",
            risk_metrics=self.position_manager.calculate_concentration_risk(current_weights)
        )

        return result

    def record_trade(
        self,
        ticker: str,
        quantity: float,
        direction: str,
        compliance_result: TradeComplianceResult,
        portfolio_weights: Dict[str, float],
        executed: bool
    ):
        """
        Record a trade in audit trail.

        Args:
            ticker: Ticker symbol
            quantity: Trade quantity
            direction: 'buy' or 'sell'
            compliance_result: Compliance check result
            portfolio_weights: Post-trade weights
            executed: Whether trade was executed
        """
        risk_metrics = self.position_manager.calculate_concentration_risk(portfolio_weights)

        self.audit_trail.log_trade(
            ticker=ticker,
            quantity=quantity,
            direction=direction,
            compliance_result=compliance_result,
            risk_metrics=risk_metrics,
            outcome='executed' if executed else 'blocked'
        )

    def get_portfolio_status(
        self,
        weights: Dict[str, float]
    ) -> Dict:
        """
        Get comprehensive portfolio compliance status.

        Args:
            weights: Portfolio weights

        Returns:
            Status dictionary
        """
        concentration = self.position_manager.calculate_concentration_risk(weights)
        conc_status, messages = self.concentration_monitor.check_concentration_limits(weights)

        pos_ok, pos_issues = self.position_manager.check_position_limits(weights)
        sector_ok, sector_issues = self.position_manager.check_sector_limits(weights)

        return {
            'overall_status': conc_status.value,
            'concentration_metrics': concentration,
            'position_limits_ok': pos_ok,
            'position_issues': pos_issues,
            'sector_limits_ok': sector_ok,
            'sector_issues': sector_issues,
            'messages': messages,
            'active_alerts': len(self.concentration_monitor.get_active_alerts())
        }

    def generate_compliance_report(
        self,
        weights: Dict[str, float],
        period_days: int = 30
    ) -> Dict:
        """
        Generate comprehensive compliance report.

        Args:
            weights: Current portfolio weights
            period_days: Report period in days

        Returns:
            Compliance report
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        status = self.get_portfolio_status(weights)
        audit_report = self.audit_trail.generate_audit_report(start_date, end_date)
        compliance_report = self.checker.generate_compliance_report(
            weights, start_date, end_date
        )

        return {
            'current_status': status,
            'audit_summary': audit_report,
            'compliance_details': compliance_report,
            'generated_at': datetime.now().isoformat()
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_compliance_manager(
    config: Optional[Dict] = None
) -> ComplianceManager:
    """
    Create configured compliance manager.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured ComplianceManager
    """
    if config:
        comp_config = ComplianceConfig(**config)
    else:
        comp_config = ComplianceConfig()

    return ComplianceManager(config=comp_config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'ComplianceStatus',
    'AlertSeverity',

    # Data Classes
    'ComplianceConfig',
    'ComplianceAlert',
    'TradeComplianceResult',
    'AuditEntry',

    # Core Classes
    'PositionLimitManager',
    'ConcentrationMonitor',
    'ComplianceChecker',
    'AuditTrailManager',
    'ComplianceManager',

    # Factory
    'create_compliance_manager',
]
