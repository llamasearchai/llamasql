"""
Migration module for database migrations.
"""

import importlib.util
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .connection import Connection


class Migration:
    """
    Represents a database migration.

    A migration is a collection of SQL statements to execute against a database.
    """

    def __init__(self, name: str, order: int = None):
        """
        Initialize a migration.

        Args:
            name: Migration name
            order: Migration order, defaults to current timestamp
        """
        self.name = name
        self.order = order or int(time.time())
        self.up_operations = []
        self.down_operations = []

    def up(self, sql: str, params: List = None):
        """
        Add an up operation.

        Args:
            sql: SQL statement
            params: SQL parameters
        """
        self.up_operations.append((sql, params or []))
        return self

    def down(self, sql: str, params: List = None):
        """
        Add a down operation.

        Args:
            sql: SQL statement
            params: SQL parameters
        """
        self.down_operations.append((sql, params or []))
        return self

    def create_table(
        self, table_name: str, columns: Dict[str, str], if_not_exists: bool = False
    ):
        """
        Add a create table operation.

        Args:
            table_name: Table name
            columns: Column definitions
            if_not_exists: Whether to add IF NOT EXISTS
        """
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        column_defs = []

        for name, definition in columns.items():
            column_defs.append(f"{name} {definition}")

        sql = (
            f"CREATE TABLE {exists_clause}{table_name} (\n  "
            + ",\n  ".join(column_defs)
            + "\n)"
        )

        self.up(sql)
        self.down(f"DROP TABLE IF EXISTS {table_name}")

        return self

    def drop_table(self, table_name: str, if_exists: bool = False):
        """
        Add a drop table operation.

        Args:
            table_name: Table name
            if_exists: Whether to add IF EXISTS
        """
        exists_clause = "IF EXISTS " if if_exists else ""

        self.up(f"DROP TABLE {exists_clause}{table_name}")
        # No down operation as we can't recreate the table without knowing its structure

        return self

    def add_column(self, table_name: str, column_name: str, column_type: str):
        """
        Add a column.

        Args:
            table_name: Table name
            column_name: Column name
            column_type: Column type
        """
        self.up(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        self.down(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")

        return self

    def drop_column(self, table_name: str, column_name: str):
        """
        Drop a column.

        Args:
            table_name: Table name
            column_name: Column name
        """
        self.up(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")
        # No down operation as we can't recreate the column without knowing its type

        return self

    def rename_column(self, table_name: str, old_name: str, new_name: str):
        """
        Rename a column.

        Args:
            table_name: Table name
            old_name: Old column name
            new_name: New column name
        """
        self.up(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}")
        self.down(f"ALTER TABLE {table_name} RENAME COLUMN {new_name} TO {old_name}")

        return self

    def create_index(
        self,
        table_name: str,
        column_names: List[str],
        index_name: str = None,
        unique: bool = False,
    ):
        """
        Create an index.

        Args:
            table_name: Table name
            column_names: Column names
            index_name: Index name
            unique: Whether the index is unique
        """
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"

        unique_clause = "UNIQUE " if unique else ""

        self.up(
            f"CREATE {unique_clause}INDEX {index_name} ON {table_name} ({', '.join(column_names)})"
        )
        self.down(f"DROP INDEX IF EXISTS {index_name}")

        return self

    def drop_index(self, index_name: str):
        """
        Drop an index.

        Args:
            index_name: Index name
        """
        self.up(f"DROP INDEX IF EXISTS {index_name}")
        # No down operation as we can't recreate the index without knowing its structure

        return self


class Migrator:
    """
    Manages database migrations.

    The migrator keeps track of which migrations have been applied to a database.
    """

    def __init__(self, connection: Connection, migrations_table: str = "migrations"):
        """
        Initialize a migrator.

        Args:
            connection: Database connection
            migrations_table: Name of the migrations table
        """
        self.connection = connection
        self.migrations_table = migrations_table
        self.migrations = []

        # Ensure migrations table exists
        self._ensure_migrations_table()

    def _ensure_migrations_table(self):
        """Ensure the migrations table exists."""
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                batch INTEGER NOT NULL,
                executed_at TIMESTAMP NOT NULL
            )
        """
        )

    def add_migration(self, migration: Migration):
        """
        Add a migration to the migrator.

        Args:
            migration: Migration to add
        """
        self.migrations.append(migration)
        return self

    def load_migrations_from_directory(self, directory: str) -> List[Migration]:
        """
        Load migrations from a directory.

        Args:
            directory: Directory to load migrations from

        Returns:
            List of loaded migrations
        """
        migrations = []

        # Get all Python files in the directory
        try:
            files = sorted([f for f in os.listdir(directory) if f.endswith(".py")])
        except FileNotFoundError:
            raise ValueError(f"Migrations directory not found: {directory}")

        for file in files:
            # Load the migration module
            file_path = os.path.join(directory, file)
            module_name = file[:-3]  # Remove .py

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the migration
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, Migration):
                    migrations.append(attr)
                    self.add_migration(attr)

        return migrations

    def get_applied_migrations(self) -> List[Dict]:
        """
        Get all applied migrations.

        Returns:
            List of applied migrations
        """
        result = self.connection.execute(
            f"SELECT * FROM {self.migrations_table} ORDER BY id"
        )
        return result.to_dicts()

    def get_pending_migrations(self) -> List[Migration]:
        """
        Get all pending migrations.

        Returns:
            List of pending migrations
        """
        applied = self.get_applied_migrations()
        applied_names = [m["name"] for m in applied]

        return [m for m in self.migrations if m.name not in applied_names]

    def run_migration(self, migration: Migration, direction: str = "up"):
        """
        Run a migration.

        Args:
            migration: Migration to run
            direction: Migration direction (up or down)
        """
        operations = (
            migration.up_operations if direction == "up" else migration.down_operations
        )

        with self.connection.transaction():
            for sql, params in operations:
                self.connection.execute(sql, params)

            if direction == "up":
                # Record the migration
                batch = self._get_next_batch()

                self.connection.execute(
                    f"INSERT INTO {self.migrations_table} (name, batch, executed_at) VALUES (?, ?, ?)",
                    [migration.name, batch, datetime.now().isoformat()],
                )
            else:
                # Remove the migration record
                self.connection.execute(
                    f"DELETE FROM {self.migrations_table} WHERE name = ?",
                    [migration.name],
                )

    def _get_next_batch(self) -> int:
        """
        Get the next batch number.

        Returns:
            Next batch number
        """
        result = self.connection.execute(
            f"SELECT MAX(batch) as max_batch FROM {self.migrations_table}"
        )
        row = result.first()

        if row and row.max_batch is not None:
            return row.max_batch + 1

        return 1

    def migrate(self):
        """Run all pending migrations."""
        pending = self.get_pending_migrations()

        if not pending:
            return False

        for migration in sorted(pending, key=lambda m: m.order):
            self.run_migration(migration, "up")

        return True

    def rollback(self, steps: int = 1):
        """
        Rollback the last batch of migrations.

        Args:
            steps: Number of batches to rollback
        """
        applied = self.get_applied_migrations()

        if not applied:
            return False

        # Group by batch
        batches = {}
        for migration in applied:
            batch = migration["batch"]
            if batch not in batches:
                batches[batch] = []
            batches[batch].append(migration)

        # Get the last N batches
        batch_numbers = sorted(batches.keys(), reverse=True)[:steps]

        for batch in batch_numbers:
            batch_migrations = batches[batch]

            for migration_data in batch_migrations:
                migration_name = migration_data["name"]

                # Find the migration object
                migration = next(
                    (m for m in self.migrations if m.name == migration_name), None
                )

                if migration:
                    self.run_migration(migration, "down")

        return True
