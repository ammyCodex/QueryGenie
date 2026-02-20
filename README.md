# QueryGenie 2.0 – AI-Powered Query Builder with Human-in-Loop Validation

A secure, production-ready SQL query builder powered by Cohere AI, featuring human-in-loop approval, query validation, and comprehensive audit logging.

## Features

### ✨ Core Features
- **AI-Powered SQL Generation**: Uses Cohere AI to generate SQLite queries from natural language
- **Human-in-Loop Approval**: All generated queries require user approval before execution
- **Iterative Improvement**: Users can request AI to improve/refine queries (up to 3 rounds)
- **Manual Editing**: Direct SQL editing with preview before execution
- **Query Explanation**: Built-in feature to explain what any query does

### 🔒 Security Features
- **SELECT-Only Enforcement**: Only read queries are permitted (no INSERT/UPDATE/DELETE/DDL)
- **SQL Injection Prevention**: Validates queries for common injection patterns
- **Query Limits**: Configurable row limits and execution timeouts
- **Comprehensive Audit Logging**: Full audit trail of all queries, approvals, and executions
- **Parameterized Queries**: Uses SQLAlchemy for safe database operations
- **Configuration Validation**: Required environment variables are validated at startup

### 🛠️ Developer Features
- **SQLAlchemy ORM**: Type-safe database layer with connection pooling
- **Modular Architecture**: Separate concerns (validation, execution, logging)
- **Type Hints**: Full type annotations for better IDE support
- **Testing Suite**: Pytest-based tests for validators and executor
- **Code Quality**: Black, isort, flake8, mypy, bandit checks configured
- **Pre-commit Hooks**: Automated code quality checks before commits

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/QueryGenie.git
cd QueryGenie
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Configure Environment
```bash
# Copy the example config
cp .env.example .env

# Edit .env with your settings
# IMPORTANT: Set your COHERE_API_KEY
```

### 5. Install Pre-commit Hooks (Optional)
```bash
pre-commit install
```

## Configuration

Create a `.env` file in the project root:

```env
# Cohere API
COHERE_API_KEY=your_api_key_here
COHERE_MODEL=command-r-plus

# Database (default: SQLite)
DATABASE_URL=sqlite:///querydb.sqlite

# Query Execution Limits
MAX_QUERY_LENGTH=10000  # Max SQL length in characters
MAX_ROWS=10000          # Max rows returned per query
QUERY_TIMEOUT=30        # Timeout in seconds

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_FILE=audit.log

# Development
DEBUG=false
```

### Supported Databases
- **SQLite**: `sqlite:///path/to/db.sqlite` (default)
- **PostgreSQL**: `postgresql://user:password@localhost:5432/dbname`
- **MySQL**: `mysql+pymysql://user:password@localhost:3306/dbname`

## Running the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Workflow

1. **Upload Database**: Select your SQLite database file in the sidebar
2. **Ask Question**: Describe what data you want in natural language
3. **Review**: The AI generates SQL—review it in the expandable panel
4. **Choose Action**:
   - ✅ **Approve** – Execute the query and get results
   - 🛠 **Improve** – Let AI refine the query (up to 3 times)
   - ✏️ **Edit** – Manually edit the SQL before execution
5. **View Results**: Results are displayed in a table

## Project Structure

```
QueryGenie/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── db.py                       # SQLAlchemy database layer
├── query_validator.py          # SQL validation and security checks
├── safe_executor.py            # Safe query execution with limits
├── logging_audit.py            # Audit logging system
├── pyproject.toml              # Python project config (black, isort, mypy, pytest)
├── .pre-commit-config.yaml     # Pre-commit hooks
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── .env.example                # Configuration template
├── audit.log                   # Query execution audit trail
└── tests/
    ├── __init__.py
    └── test_validator.py       # Validator unit tests
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html
```

## Code Quality

### Format Code
```bash
black .
isort .
```

### Lint Code
```bash
flake8 .
```

### Type Check
```bash
mypy .
```

### Security Check
```bash
bandit -r .
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Security Considerations

1. **Never expose COHERE_API_KEY** – Keep it in environment variables
2. **Audit Log Review** – Regularly review `audit.log` for suspicious queries
3. **Row Limits** – Adjust `MAX_ROWS` based on your database size
4. **Query Timeouts** – Set `QUERY_TIMEOUT` based on typical query complexity
5. **Database Backups** – Regular backups recommended before using with production data
6. **Connection Pooling** – Configured with limits to prevent resource exhaustion

## Audit Logging

Every query action is logged to `audit.log` with:
- Timestamp
- Query status (PENDING, APPROVED, EXECUTED, REJECTED, ERROR, TIMEOUT)
- SQL query (truncated to 200 chars)
- Reason/error message
- Execution duration
- Rows affected

Example:
```json
{"timestamp": "2024-02-20T10:30:45.123456", "status": "EXECUTED", "user": "unknown", "query": "SELECT * FROM users WHERE id > 100", "reason": "Success", "duration": "0.45s", "rows_affected": 42}
```

## Troubleshooting

### "COHERE_API_KEY not set"
- Verify `.env` file exists in project root
- Check `COHERE_API_KEY` is set and not empty
- Reload the Streamlit app

### "DATABASE_URL not set"
- Ensure `.env` file has `DATABASE_URL` defined
- Default is SQLite, but can be changed to PostgreSQL, MySQL, etc.

### Query Timeout
- Increase `QUERY_TIMEOUT` in `.env` for slower queries
- Optimize slow queries or use row limits

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Verify Python 3.9+: `python --version`

## Contributing

1. Install dev dependencies: `pip install -r requirements-dev.txt`
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and run: `pre-commit run --all-files`
4. Run tests: `pytest`
5. Commit and push your changes
6. Create a Pull Request

## License

MIT License – See LICENSE file

## Changelog

### Version 2.0.0
- ✨ Added human-in-loop query approval workflow
- 🔒 Added comprehensive query validation and SQL injection prevention
- 📊 Added query execution limits and timeouts
- 📝 Added audit logging for all queries and actions
- 🏗️ Refactored with SQLAlchemy ORM layer
- 🧪 Added comprehensive test suite
- 📋 Added pre-commit hooks and code quality tools

### Version 1.0.0
- Initial release with basic AI-powered SQL generation

License & Copyright

© 2025 Ammy

This project is provided as-is for personal and educational use.
Do not redistribute or use commercially without permission.

---

Contact

For questions or suggestions, please contact Ammy Sharma.

---

