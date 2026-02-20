"""
Audit logging for query execution and user actions.
Maintains a complete audit trail for security and compliance.
"""
import logging
import json
from datetime import datetime
from pathlib import Path

from config import AUDIT_LOG_FILE, LOG_LEVEL, APP_NAME


class AuditLogger:
    """Logs all query executions, approvals, and rejections."""

    def __init__(self, log_file: str = AUDIT_LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(f"{APP_NAME}.audit")
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # File handler
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_query(
        self,
        status: str,
        query: str,
        reason: str = "",
        user: str = "unknown",
        duration: float = 0.0,
        rows_affected: int = 0,
    ):
        """
        Log a query execution event.
        Status: PENDING, APPROVED, REJECTED, EXECUTED, ERROR, TIMEOUT
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "user": user,
            "query": query[:200] + "..." if len(query) > 200 else query,
            "reason": reason,
            "duration": f"{duration:.2f}s",
            "rows_affected": rows_affected,
        }
        self.logger.info(json.dumps(event))

    def log_approval(self, query: str, user: str = "unknown", approved: bool = True):
        """Log user approval or rejection of a query."""
        status = "APPROVED" if approved else "REJECTED"
        self.log_query(status, query, user=user)

    def log_execution(self, query: str, duration: float, rows: int, error: str = ""):
        """Log query execution result."""
        status = "ERROR" if error else "EXECUTED"
        reason = error if error else f"Success"
        self.log_query(status, query, reason=reason, duration=duration, rows_affected=rows)


# Global instance
audit_logger = AuditLogger()
