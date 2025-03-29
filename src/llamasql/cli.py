"""
Command-line interface for LlamaSQL.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .connection import connect
from .migration import Migration, Migrator


def create_migration_command(args):
    """Create a new migration file."""
    migration_name = args.name.replace(" ", "_").lower()
    timestamp = int(time.time())
    filename = f"{timestamp}_{migration_name}.py"
    
    # Ensure migrations directory exists
    os.makedirs(args.directory, exist_ok=True)
    
    # Create the migration file
    file_path = os.path.join(args.directory, filename)
    
    with open(file_path, "w") as f:
        f.write(f"""from llamasql.migration import Migration

# Migration: {args.name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

migration = Migration(
    name="{migration_name}",
    order={timestamp}
)

# Define your migration steps here
# migration.create_table(
#     "users",
#     {
#         "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
#         "name": "TEXT NOT NULL",
#         "email": "TEXT UNIQUE NOT NULL",
#         "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
#     }
# )
""")
    
    print(f"Created migration file: {file_path}")


def migrate_command(args):
    """Run all pending migrations."""
    try:
        with connect(args.dsn) as conn:
            migrator = Migrator(conn)
            
            # Load migrations
            migrator.load_migrations_from_directory(args.directory)
            
            # Get pending migrations
            pending = migrator.get_pending_migrations()
            
            if not pending:
                print("No pending migrations.")
                return
            
            print(f"Running {len(pending)} migrations...")
            
            # Run migrations
            if migrator.migrate():
                print("Migrations completed successfully.")
            else:
                print("No migrations were applied.")
    
    except Exception as e:
        print(f"Error running migrations: {e}")
        sys.exit(1)


def rollback_command(args):
    """Rollback the last batch of migrations."""
    try:
        with connect(args.dsn) as conn:
            migrator = Migrator(conn)
            
            # Load migrations
            migrator.load_migrations_from_directory(args.directory)
            
            # Get applied migrations
            applied = migrator.get_applied_migrations()
            
            if not applied:
                print("No migrations to rollback.")
                return
            
            print(f"Rolling back last {args.steps} batch(es) of migrations...")
            
            # Run rollback
            if migrator.rollback(args.steps):
                print("Rollback completed successfully.")
            else:
                print("No migrations were rolled back.")
    
    except Exception as e:
        print(f"Error rolling back migrations: {e}")
        sys.exit(1)


def migration_status_command(args):
    """Show migration status."""
    try:
        with connect(args.dsn) as conn:
            migrator = Migrator(conn)
            
            # Load migrations
            migrator.load_migrations_from_directory(args.directory)
            
            # Get applied migrations
            applied = migrator.get_applied_migrations()
            
            # Get pending migrations
            pending = migrator.get_pending_migrations()
            
            print(f"Applied migrations: {len(applied)}")
            for migration in applied:
                print(f"  ✓ {migration['name']} (batch {migration['batch']}, {migration['executed_at']})")
            
            print(f"\nPending migrations: {len(pending)}")
            for migration in pending:
                print(f"  ✗ {migration.name}")
    
    except Exception as e:
        print(f"Error showing migration status: {e}")
        sys.exit(1)


def query_command(args):
    """Execute a SQL query."""
    try:
        with connect(args.dsn) as conn:
            # Execute the query
            if args.file:
                # Read query from file
                with open(args.file, "r") as f:
                    query = f.read()
            else:
                # Read query from stdin
                query = args.query or sys.stdin.read()
            
            # Execute the query
            result = conn.execute(query, args.params)
            
            # Print the results
            if result.description:
                # Print header
                headers = result.column_names
                header_line = " | ".join(headers)
                print(header_line)
                print("-" * len(header_line))
                
                # Print rows
                for row in result:
                    print(" | ".join(str(getattr(row, col)) for col in headers))
                
                print(f"\n{result.rowcount} rows returned")
            else:
                print("Query executed successfully")
                if result.lastrowid:
                    print(f"Last inserted ID: {result.lastrowid}")
    
    except Exception as e:
        print(f"Error executing query: {e}")
        sys.exit(1)


def main():
    """Main entry point for the LlamaSQL CLI."""
    parser = argparse.ArgumentParser(description="LlamaSQL command-line tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("name", help="Migration name")
    create_parser.add_argument("--directory", "-d", default="./migrations", help="Migrations directory")
    create_parser.set_defaults(func=create_migration_command)
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--dsn", "-c", required=True, help="Database connection string")
    migrate_parser.add_argument("--directory", "-d", default="./migrations", help="Migrations directory")
    migrate_parser.set_defaults(func=migrate_command)
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--dsn", "-c", required=True, help="Database connection string")
    rollback_parser.add_argument("--directory", "-d", default="./migrations", help="Migrations directory")
    rollback_parser.add_argument("--steps", "-s", type=int, default=1, help="Number of batches to rollback")
    rollback_parser.set_defaults(func=rollback_command)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    status_parser.add_argument("--dsn", "-c", required=True, help="Database connection string")
    status_parser.add_argument("--directory", "-d", default="./migrations", help="Migrations directory")
    status_parser.set_defaults(func=migration_status_command)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a SQL query")
    query_parser.add_argument("--dsn", "-c", required=True, help="Database connection string")
    query_parser.add_argument("--query", "-q", help="SQL query to execute")
    query_parser.add_argument("--file", "-f", help="SQL file to execute")
    query_parser.add_argument("--params", "-p", nargs="+", help="Query parameters")
    query_parser.set_defaults(func=query_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the command
    args.func(args)


if __name__ == "__main__":
    main() 