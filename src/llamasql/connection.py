"""
Connection module for database connections.
"""

import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class Result:
    """
    Represents a database query result.

    This class provides a way to access rows as dictionaries or by attribute.
    """

    def __init__(self, cursor):
        """
        Initialize a result.

        Args:
            cursor: Database cursor with executed query
        """
        self.cursor = cursor
        self.description = cursor.description
        self.column_names = (
            [column[0] for column in cursor.description] if cursor.description else []
        )
        self._rows = []

        # Collect all rows
        if cursor.description:
            for row in cursor.fetchall():
                self._rows.append(Row(self.column_names, row))

    def __iter__(self):
        """Iterate over rows."""
        return iter(self._rows)

    def __len__(self):
        """Get the number of rows."""
        return len(self._rows)

    def __getitem__(self, index):
        """Get a row by index."""
        return self._rows[index]

    @property
    def rowcount(self):
        """Get the number of rows."""
        return len(self._rows)

    @property
    def lastrowid(self):
        """Get the last inserted row ID."""
        return self.cursor.lastrowid

    def first(self):
        """Get the first row or None if no rows."""
        return self._rows[0] if self._rows else None

    def all(self):
        """Get all rows."""
        return self._rows

    def to_dicts(self):
        """Convert all rows to dictionaries."""
        return [row.to_dict() for row in self._rows]


class Row:
    """
    Represents a database row.

    This class provides a way to access row values by column name or by attribute.
    """

    def __init__(self, column_names, values):
        """
        Initialize a row.

        Args:
            column_names: List of column names
            values: List of values
        """
        self._data = dict(zip(column_names, values))

    def __getattr__(self, name):
        """Get a column value by attribute."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'Row' object has no attribute '{name}'")

    def __getitem__(self, key):
        """Get a column value by key."""
        return self._data[key]

    def to_dict(self):
        """Convert the row to a dictionary."""
        return self._data


class Connection:
    """
    A database connection wrapper.

    This class provides a unified interface for different database backends.
    """

    def __init__(self, connection):
        """
        Initialize a connection.

        Args:
            connection: Database connection object
        """
        self.connection = connection
        self.closed = False

    def execute(self, query, params=None):
        """
        Execute a query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query result
        """
        if self.closed:
            raise ValueError("Connection is closed")

        cursor = self.connection.cursor()
        cursor.execute(query, params or [])

        return Result(cursor)

    def executemany(self, query, params_list):
        """
        Execute a query multiple times.

        Args:
            query: SQL query
            params_list: List of query parameters

        Returns:
            Query result
        """
        if self.closed:
            raise ValueError("Connection is closed")

        cursor = self.connection.cursor()
        cursor.executemany(query, params_list)

        return Result(cursor)

    def commit(self):
        """Commit the current transaction."""
        if self.closed:
            raise ValueError("Connection is closed")

        self.connection.commit()

    def rollback(self):
        """Rollback the current transaction."""
        if self.closed:
            raise ValueError("Connection is closed")

        self.connection.rollback()

    def close(self):
        """Close the connection."""
        if not self.closed:
            self.connection.close()
            self.closed = True

    def transaction(self):
        """
        Start a transaction.

        Returns:
            Transaction context manager
        """
        return Transaction(self)


class Transaction:
    """
    A database transaction context manager.

    Example:
        with connection.transaction():
            connection.execute("INSERT INTO users (name) VALUES (?)", ["John"])
            connection.execute("INSERT INTO profiles (user_id) VALUES (?)", [1])
    """

    def __init__(self, connection):
        """
        Initialize a transaction.

        Args:
            connection: Database connection
        """
        self.connection = connection

    def __enter__(self):
        """Start the transaction."""
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        End the transaction.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            Whether the exception was handled
        """
        if exc_type is not None:
            # An exception occurred, rollback
            self.connection.rollback()
        else:
            # No exception, commit
            self.connection.commit()

        return False  # Don't suppress exceptions


@contextmanager
def connect(dsn: str) -> Connection:
    """
    Connect to a database.

    Args:
        dsn: Database connection string
            - sqlite:///path/to/file.db
            - postgres://user:pass@host:port/dbname
            - mysql://user:pass@host:port/dbname

    Returns:
        Database connection context manager

    Example:
        with connect("sqlite:///example.db") as conn:
            result = conn.execute("SELECT * FROM users")
    """
    conn = None

    try:
        # Parse the DSN
        if dsn.startswith("sqlite:///"):
            # SQLite
            import sqlite3

            db_path = dsn[len("sqlite:///") :]
            conn_obj = sqlite3.connect(db_path)
            conn_obj.row_factory = sqlite3.Row
            conn = Connection(conn_obj)

        elif dsn.startswith(("postgres://", "postgresql://")):
            # PostgreSQL
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError:
                raise ImportError(
                    "PostgreSQL support requires psycopg2. "
                    "Install it with: pip install psycopg2-binary"
                )

            conn_obj = psycopg2.connect(dsn)
            conn_obj.cursor_factory = psycopg2.extras.DictCursor
            conn = Connection(conn_obj)

        elif dsn.startswith(("mysql://", "mariadb://")):
            # MySQL/MariaDB
            try:
                import MySQLdb
                import MySQLdb.cursors
            except ImportError:
                raise ImportError(
                    "MySQL/MariaDB support requires mysqlclient. "
                    "Install it with: pip install mysqlclient"
                )

            # Parse the DSN
            pattern = r"(?P<driver>mysql|mariadb)://(?P<user>[^:@]+)(?::(?P<password>[^@]+))?@(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>[^?]+)(?:\?(?P<options>.+))?"
            match = re.match(pattern, dsn)

            if not match:
                raise ValueError(f"Invalid MySQL/MariaDB DSN: {dsn}")

            conn_params = match.groupdict()

            conn_obj = MySQLdb.connect(
                host=conn_params["host"],
                user=conn_params["user"],
                passwd=conn_params["password"] or "",
                db=conn_params["database"],
                port=int(conn_params["port"]) if conn_params["port"] else 3306,
                cursorclass=MySQLdb.cursors.DictCursor,
            )

            conn = Connection(conn_obj)

        else:
            raise ValueError(f"Unsupported database type in DSN: {dsn}")

        yield conn

    finally:
        if conn:
            conn.close()
