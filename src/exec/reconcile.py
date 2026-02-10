"""Position and PnL reconciliation between strategy state and broker."""

from __future__ import annotations

from typing import Any

from src.core.logging_utils import AuditLogger, get_logger
from src.exec.broker_base import BrokerBase, Position

logger = get_logger("exec.reconcile")


class Reconciler:
    """Reconcile positions and PnL between local state and broker.

    Detects discrepancies in:
    - Position quantities
    - Cash balance
    - Net PnL
    """

    def __init__(self, audit_logger: AuditLogger | None = None):
        self.audit = audit_logger

    def reconcile_positions(
        self,
        local_positions: dict[str, int],
        broker: BrokerBase,
    ) -> list[dict[str, Any]]:
        """Compare local position state with broker positions.

        Args:
            local_positions: Dict of symbol -> quantity from strategy state.
            broker: Broker instance to query real positions.

        Returns:
            List of discrepancy dicts.
        """
        broker_positions = broker.get_positions()
        broker_map = {p.symbol: p for p in broker_positions}

        discrepancies = []

        # Check all local positions
        for symbol, local_qty in local_positions.items():
            broker_pos = broker_map.get(symbol)
            if broker_pos is None:
                disc = {
                    "symbol": symbol,
                    "type": "missing_at_broker",
                    "local_qty": local_qty,
                    "broker_qty": 0,
                }
                discrepancies.append(disc)
                logger.warning(f"Position discrepancy: {symbol} local={local_qty}, broker=0")
            elif broker_pos.quantity != local_qty:
                disc = {
                    "symbol": symbol,
                    "type": "quantity_mismatch",
                    "local_qty": local_qty,
                    "broker_qty": broker_pos.quantity,
                }
                discrepancies.append(disc)
                logger.warning(
                    f"Position discrepancy: {symbol} "
                    f"local={local_qty}, broker={broker_pos.quantity}"
                )

        # Check for positions at broker not tracked locally
        for symbol, broker_pos in broker_map.items():
            if symbol not in local_positions and broker_pos.quantity != 0:
                disc = {
                    "symbol": symbol,
                    "type": "missing_locally",
                    "local_qty": 0,
                    "broker_qty": broker_pos.quantity,
                }
                discrepancies.append(disc)
                logger.warning(
                    f"Position discrepancy: {symbol} "
                    f"local=0, broker={broker_pos.quantity}"
                )

        if discrepancies and self.audit:
            self.audit.log("reconciliation", {
                "n_discrepancies": len(discrepancies),
                "details": discrepancies,
            })

        if not discrepancies:
            logger.info("Reconciliation: all positions match")

        return discrepancies

    def reconcile_cash(
        self,
        local_cash: float,
        broker: BrokerBase,
        tolerance: float = 1.0,
    ) -> dict[str, Any] | None:
        """Compare local cash with broker balance.

        Args:
            local_cash: Local tracked cash.
            broker: Broker instance.
            tolerance: Acceptable difference in currency units.

        Returns:
            Discrepancy dict or None if within tolerance.
        """
        broker_cash = broker.get_balance()
        diff = abs(local_cash - broker_cash)

        if diff > tolerance:
            disc = {
                "type": "cash_mismatch",
                "local_cash": local_cash,
                "broker_cash": broker_cash,
                "difference": diff,
            }
            logger.warning(f"Cash discrepancy: local={local_cash:.2f}, broker={broker_cash:.2f}")
            if self.audit:
                self.audit.log("cash_reconciliation", disc)
            return disc

        return None
