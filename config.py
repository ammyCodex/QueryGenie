"""
Configuration management using environment variables and dotenv.
Centralizes all settings for the QueryGenie application.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///querydb.sqlite")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "30"))

# Query Execution
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "10000"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))

# Streamlit
STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "dark")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "audit.log")

# Application
APP_NAME = "QueryGenie"
APP_VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def validate_config():
    """Validate that all required configuration is set."""
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not set in environment")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set in environment")
