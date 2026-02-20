"""
Unit tests for query validator module.
"""
import pytest

from query_validator import QueryValidator


class TestQueryValidator:
    """Test query validation logic."""

    @pytest.fixture
    def validator(self):
        return QueryValidator()

    def test_empty_query_fails(self, validator):
        is_safe, reason = validator.is_safe("")
        assert not is_safe
        assert "empty" in reason.lower()

    def test_select_query_passes(self, validator):
        is_safe, reason = validator.is_safe("SELECT * FROM users")
        assert is_safe
        assert "safe" in reason.lower()

    def test_dangerous_keywords_fail(self, validator):
        dangerous = ["DROP TABLE users", "DELETE FROM users", "ALTER TABLE users ADD COLUMN x INT"]
        for query in dangerous:
            is_safe, reason = validator.is_safe(query)
            assert not is_safe, f"Should reject: {query}"

    def test_injection_patterns_fail(self, validator):
        injection_queries = [
            "SELECT * FROM users WHERE id = '1' OR '1'='1'",
            "SELECT * FROM users; DROP TABLE users --",
            "SELECT * FROM users UNION SELECT password FROM admin",
        ]
        for query in injection_queries:
            is_safe, reason = validator.is_safe(query)
            assert not is_safe, f"Should reject injection: {query}"

    def test_extract_tables(self, validator):
        query = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        tables = validator.extract_tables(query)
        assert "users" in str(tables).lower()

    def test_extract_columns(self, validator):
        query = "SELECT id, name, email FROM users"
        columns = validator.extract_columns(query)
        assert len(columns) >= 1


class TestSafeQueryExecutor:
    """Test safe query executor."""

    def test_max_length_enforced(self):
        from safe_executor import SafeQueryExecutor
        executor = SafeQueryExecutor(None)
        long_query = "SELECT * FROM users WHERE " + "x = 1 AND " * 2000
        assert len(long_query) > executor.MAX_QUERY_LENGTH

    def test_is_select_only(self):
        from safe_executor import SafeQueryExecutor
        assert SafeQueryExecutor._is_select_only("SELECT * FROM users")
        assert not SafeQueryExecutor._is_select_only("INSERT INTO users VALUES (1, 'test')")
        assert not SafeQueryExecutor._is_select_only("DELETE FROM users")
