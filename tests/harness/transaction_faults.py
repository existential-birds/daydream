"""Durable-boundary fault driver for transaction recovery tests."""

from pathlib import Path

from daydream.benchmark.storage import Transaction


class TransactionFaultDriver:
    """Drive a real transaction to one documented interruption boundary.

    Tests stage targets through :attr:`transaction`, then call
    :meth:`halt_at`. ``staged`` and ``backup`` mean immediately after that
    caller-owned staging work; ``manifest`` means complete-before-cleanup.
    The driver shares the transaction's actual durable transitions; it never
    manufactures a journal document or changes state directly.
    """

    def __init__(self, root: Path, *, op_id: str, kind: str) -> None:
        self.transaction = Transaction(root, op_id=op_id, kind=kind)

    def halt_at(self, boundary: str) -> None:
        """Leave the transaction at a documented durable crash boundary."""
        tx = self.transaction
        if boundary in ("staged", "backup"):
            return
        if boundary in ("journal", "target-0"):
            tx.prepare()
            return
        if boundary == "data":
            tx.prepare()
            tx.begin_commit()
            return
        if boundary == "manifest":
            tx.prepare()
            tx.begin_commit()
            tx._complete_commit()
            return
        if boundary.startswith("target-"):
            suffix = boundary[len("target-"):]
            try:
                count = int(suffix)
            except ValueError:
                raise ValueError(f"invalid target boundary {boundary!r}") from None
            if not (0 <= count <= len(tx._replacement_order)):
                raise ValueError(f"target boundary {count!r} out of range")
            tx.prepare()
            if count == 0:
                return
            tx._begin_committing()
            for rel in tx._replacement_order[:count]:
                tx._apply_replacement(rel)
            return
        raise ValueError(f"unknown crash boundary {boundary!r}")
