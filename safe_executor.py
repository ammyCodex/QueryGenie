"""
Safe query validation and execution.
Ensures only SELECT queries, validates syntax, enforces limits and timeouts.
"""
from typing import Tuple, Optional
import re
import time
from sqlalchemy import text, event
from sqlalchemy.exc import TimeoutError as SQLTimeoutError

import sqlparse
from query_validator import QueryValidator


class SafeQueryExecutor:
    """Executes queries safely with validation, limits, timeouts, and audit logging."""

    # Query limits
    MAX_QUERY_LENGTH = 10000  # Max SQL length in chars
    MAX_ROWS = 10000  # Max rows to return
    QUERY_TIMEOUT = 30  # Timeout in seconds

    def __init__(self, session):
        self.session = session
        self.query_validator = QueryValidator()

    def validate_and_execute(
        self,
        sql: str,
        audit_log_func: Optional[callable] = None,
    ) -> Tuple[bool, str, Optional[list]]:
        """
        Validate and safely execute an SQL query.
        Returns: (success: bool, message: str, results: Optional[list])
        """
        # 1. Basic validation
        if not sql or not sql.strip():
            return False, "Query cannot be empty", None

        if len(sql) > self.MAX_QUERY_LENGTH:
            return False, f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters", None

        # 2. Parse and validate query
        is_safe, reason = self.query_validator.is_safe(sql)
        if not is_safe:
            if audit_log_func:
                audit_log_func("REJECTED", sql, reason)
            return False, f"Query validation failed: {reason}", None

        # 3. Check it's a SELECT statement
        if not self._is_select_only(sql):
            if audit_log_func:
                audit_log_func("REJECTED", sql, "Non-SELECT query not permitted")
            return False, "Only SELECT queries are permitted", None

        # 4. Execute with timeout and row limit
        try:
            if audit_log_func:
                audit_log_func("PENDING", sql, "Awaiting execution")

            start_time = time.time()
            stmt = text(sql)
            result = self.session.execute(stmt, execution_options={"timeout": self.QUERY_TIMEOUT})
            rows = result.fetchmany(self.MAX_ROWS + 1)  # Fetch one extra to detect overflow
            elapsed = time.time() - start_time

            # Check if we hit the row limit
            if len(rows) > self.MAX_ROWS:
                rows = rows[: self.MAX_ROWS]
                message = f"Query returned {len(rows)}+ results (limited to {self.MAX_ROWS})"
            else:
                message = f"Query executed successfully in {elapsed:.2f}s, returned {len(rows)} rows"

            if audit_log_func:
                audit_log_func("EXECUTED", sql, f"Success: {len(rows)} rows, {elapsed:.2f}s")

            return True, message, rows

        except SQLTimeoutError:
            if audit_log_func:
                audit_log_func("TIMEOUT", sql, f"Query exceeded {self.QUERY_TIMEOUT}s timeout")
            return False, f"Query timeout (max {self.QUERY_TIMEOUT}s)", None
        except Exception as e:
            if audit_log_func:
                audit_log_func("ERROR", sql, str(e))
            return False, f"Query execution error: {str(e)}", None

    @staticmethod
    def _is_select_only(sql: str) -> bool:
        """Check if query contains only SELECT statements."""
        parsed = sqlparse.parse(sql)
        for stmt in parsed:
            # Remove comments and whitespace
            stmt_str = str(stmt).strip()
            if not stmt_str:
                continue
            if stmt.get_type() != "SELECT":
                return False
        return True
