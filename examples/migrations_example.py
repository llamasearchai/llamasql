"""
Migrations example for LlamaSQL.

This example demonstrates how to:
1. Create database migrations
2. Apply migrations
3. Roll back migrations
"""
import os
import shutil
from llamasql.connection import connect
from llamasql.migration import Migration, Migrator


def main():
    """Run the LlamaSQL migrations example."""
    # Setup
    db_path = "migrations_example.db"
    migrations_dir = "example_migrations"
    
    # Clean up from previous runs
    if os.path.exists(db_path):
        os.remove(db_path)
    
    if os.path.exists(migrations_dir):
        shutil.rmtree(migrations_dir)
    
    os.makedirs(migrations_dir)
    
    # Connect to the database
    with connect(f"sqlite:///{db_path}") as conn:
        # Create a migrator
        migrator = Migrator(conn)
        
        # Create initial migration - Create users table
        print("Creating initial migration...")
        create_users_migration = Migration("create_users_table", 1)
        create_users_migration.create_table(
            "users",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "email": "TEXT UNIQUE NOT NULL",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        )
        
        # Save the migration file
        with open(os.path.join(migrations_dir, "001_create_users_table.py"), "w") as f:
            f.write(f"""from llamasql.migration import Migration

# Migration: Create users table
migration = Migration(
    name="create_users_table",
    order=1
)

migration.create_table(
    "users",
    {{
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
        "email": "TEXT UNIQUE NOT NULL",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }}
)
""")
        
        # Create second migration - Create posts table
        print("Creating second migration...")
        create_posts_migration = Migration("create_posts_table", 2)
        create_posts_migration.create_table(
            "posts",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "user_id": "INTEGER NOT NULL",
                "title": "TEXT NOT NULL",
                "content": "TEXT",
                "published": "BOOLEAN DEFAULT 0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "FOREIGN KEY (user_id)": "REFERENCES users(id) ON DELETE CASCADE"
            }
        )
        
        # Save the migration file
        with open(os.path.join(migrations_dir, "002_create_posts_table.py"), "w") as f:
            f.write(f"""from llamasql.migration import Migration

# Migration: Create posts table
migration = Migration(
    name="create_posts_table",
    order=2
)

migration.create_table(
    "posts",
    {{
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "user_id": "INTEGER NOT NULL",
        "title": "TEXT NOT NULL",
        "content": "TEXT",
        "published": "BOOLEAN DEFAULT 0",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "FOREIGN KEY (user_id)": "REFERENCES users(id) ON DELETE CASCADE"
    }}
)
""")
        
        # Create third migration - Add columns to users table
        print("Creating third migration...")
        add_user_columns_migration = Migration("add_user_columns", 3)
        add_user_columns_migration.add_column("users", "age", "INTEGER")
        add_user_columns_migration.add_column("users", "active", "BOOLEAN DEFAULT 1")
        
        # Save the migration file
        with open(os.path.join(migrations_dir, "003_add_user_columns.py"), "w") as f:
            f.write(f"""from llamasql.migration import Migration

# Migration: Add columns to users table
migration = Migration(
    name="add_user_columns",
    order=3
)

migration.add_column("users", "age", "INTEGER")
migration.add_column("users", "active", "BOOLEAN DEFAULT 1")
""")
        
        # Load migrations from directory
        migrator.load_migrations_from_directory(migrations_dir)
        
        # Show migration status before running
        print("\nMigration status before running:")
        applied = migrator.get_applied_migrations()
        pending = migrator.get_pending_migrations()
        
        print(f"Applied migrations: {len(applied)}")
        print(f"Pending migrations: {len(pending)}")
        for migration in pending:
            print(f"  - {migration.name}")
        
        # Run migrations
        print("\nRunning migrations...")
        migrator.migrate()
        
        # Show migration status after running
        print("\nMigration status after running:")
        applied = migrator.get_applied_migrations()
        pending = migrator.get_pending_migrations()
        
        print(f"Applied migrations: {len(applied)}")
        for migration in applied:
            print(f"  - {migration['name']} (batch {migration['batch']})")
        
        print(f"Pending migrations: {len(pending)}")
        
        # Insert sample data
        print("\nInserting sample data...")
        conn.execute(
            "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
            ["John Doe", "john@example.com", 30]
        )
        
        conn.execute(
            "INSERT INTO posts (user_id, title, content, published) VALUES (?, ?, ?, ?)",
            [1, "First Post", "This is the first post", 1]
        )
        
        # Query data
        print("\nQuerying data:")
        result = conn.execute("""
            SELECT u.name, u.email, u.age, p.title, p.content
            FROM users u
            JOIN posts p ON p.user_id = u.id
        """)
        
        for row in result:
            print(f"User: {row.name} ({row.email}, {row.age})")
            print(f"Post: {row.title} - {row.content}")
        
        # Rollback the last migration
        print("\nRolling back the last migration...")
        migrator.rollback(1)
        
        # Show migration status after rollback
        print("\nMigration status after rollback:")
        applied = migrator.get_applied_migrations()
        pending = migrator.get_pending_migrations()
        
        print(f"Applied migrations: {len(applied)}")
        for migration in applied:
            print(f"  - {migration['name']} (batch {migration['batch']})")
        
        print(f"Pending migrations: {len(pending)}")
        for migration in pending:
            print(f"  - {migration.name}")
        
        # Verify the columns were removed
        print("\nVerifying schema after rollback:")
        try:
            result = conn.execute("SELECT age FROM users")
            print("Age column still exists (unexpected)")
        except Exception as e:
            print("Age column was removed (expected)")
        
        # Re-run migrations
        print("\nRe-running migrations...")
        migrator.migrate()
        
        # Create a fourth migration - Create comments table
        print("\nCreating fourth migration...")
        create_comments_migration = Migration("create_comments_table", 4)
        create_comments_migration.create_table(
            "comments",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "post_id": "INTEGER NOT NULL",
                "user_id": "INTEGER NOT NULL",
                "content": "TEXT NOT NULL",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "FOREIGN KEY (post_id)": "REFERENCES posts(id) ON DELETE CASCADE",
                "FOREIGN KEY (user_id)": "REFERENCES users(id) ON DELETE CASCADE"
            }
        )
        
        # Save the migration file
        with open(os.path.join(migrations_dir, "004_create_comments_table.py"), "w") as f:
            f.write(f"""from llamasql.migration import Migration

# Migration: Create comments table
migration = Migration(
    name="create_comments_table",
    order=4
)

migration.create_table(
    "comments",
    {{
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "post_id": "INTEGER NOT NULL",
        "user_id": "INTEGER NOT NULL",
        "content": "TEXT NOT NULL",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "FOREIGN KEY (post_id)": "REFERENCES posts(id) ON DELETE CASCADE",
        "FOREIGN KEY (user_id)": "REFERENCES users(id) ON DELETE CASCADE"
    }}
)
""")
        
        # Load migrations again to include the new one
        migrator.load_migrations_from_directory(migrations_dir)
        
        # Run the new migration
        print("\nRunning new migration...")
        migrator.migrate()
        
        # Show final migration status
        print("\nFinal migration status:")
        applied = migrator.get_applied_migrations()
        pending = migrator.get_pending_migrations()
        
        print(f"Applied migrations: {len(applied)}")
        for migration in applied:
            print(f"  - {migration['name']} (batch {migration['batch']})")
        
        print(f"Pending migrations: {len(pending)}")


if __name__ == "__main__":
    main() 