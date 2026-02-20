"""
Query validation module.
Validates SQL queries for safety, syntax, and injection attacks.
"""
import re
from typing import Tuple

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison


class QueryValidator:
    """Validates SQL queries for safety and correctness."""

    # Dangerous keywords
    DANGEROUS_KEYWORDS = {
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
    }

    # SQL injection patterns
    INJECTION_PATTERNS = [
        r"('['\"])\s*or\s*\1\s*=\s*\1",  # ' or '='
        r"--\s*|#\s*|;\s*DROP",  # Comment/drop combos
        r"union\s+select",  # Union-based injection
        r"/*.*?\*/",  # Multi-line comments
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def is_safe(self, sql: str) -> Tuple[bool, str]:
        """
        Validate query for safety.
        Returns: (is_safe: bool, reason: str)
        """
        if not sql or not sql.strip():
            return False, "Query cannot be empty"

        # Check for dangerous keywords
        tokens = sqlparse.parse(sql)[0].tokens
        for token in tokens:
            if token.ttype is None and token.value.upper() in self.DANGEROUS_KEYWORDS:
                return False, f"Dangerous keyword '{token.value.upper()}' not allowed"

        # Check for injection patterns
        sql_upper = sql.upper()
        for pattern in self.compiled_patterns:
            if pattern.search(sql):
                return False, "Potential SQL injection pattern detected"

        # Validate syntax
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Invalid SQL syntax"
        except Exception as e:
            return False, f"SQL parse error: {str(e)}"

        return True, "Query is safe"

    def extract_tables(self, sql: str) -> list:
        """Extract table names from a SELECT query."""
        try:
            parsed = sqlparse.parse(sql)[0]
            tables = []
            from_seen = False

            for token in parsed.tokens:
                if token.ttype is None and "FROM" in token.value.upper():
                    from_seen = True
                elif from_seen and isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.append(str(identifier).split()[0])
                elif from_seen and isinstance(token, Identifier):
                    tables.append(str(token).split()[0])
                elif from_seen and token.ttype is None and token.value.upper() in ("WHERE", "GROUP", "ORDER", "LIMIT"):
                    from_seen = False

            return tables
        except Exception:
            return []

    def extract_columns(self, sql: str) -> list:
        """Extract selected column names from a SELECT query."""
        try:
            parsed = sqlparse.parse(sql)[0]
            columns = []
            select_seen = False

            for token in parsed.tokens:
                if token.ttype is None and "SELECT" in token.value.upper():
                    select_seen = True
                elif select_seen and isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        col = str(identifier).split()[-1]
                        columns.append(col)
                elif select_seen and isinstance(token, Identifier):
                    col = str(token).split()[-1]
                    columns.append(col)
                elif select_seen and token.ttype is None and token.value.upper() in ("FROM", "WHERE"):
                    break

            return columns
        except Exception:
            return []
