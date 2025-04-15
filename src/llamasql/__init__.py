"""
LlamaSQL - A powerful SQL query builder and database utilities library for Python.

This package provides:
- A fluent query builder API for SQL
- Database connection management
- Migrations system
- SQL validation and parameterization
"""

__version__ = "0.1.0"

from llamasql.connection import Connection, connect
from llamasql.migration import Migration, Migrator

# Import main classes for easier access
from llamasql.query import Query
from llamasql.schema import Column, Table

# Export all important classes
__all__ = ["Query", "Table", "Column", "connect", "Connection", "Migration", "Migrator"]
