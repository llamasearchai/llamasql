"""
Basic usage example for LlamaSQL.

This example demonstrates how to:
1. Define tables and columns
2. Build and execute simple queries
3. Work with database connections
"""
import os
from llamasql.schema import Table, Column
from llamasql.query import Query
from llamasql.connection import connect


def main():
    """Run the LlamaSQL basic usage example."""
    # Create a database file
    db_path = "example.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Connect to the database
    with connect(f"sqlite:///{db_path}") as conn:
        # Define tables
        users_table = Table("users", [
            Column("id", "INTEGER", primary_key=True),
            Column("name", "TEXT", nullable=False),
            Column("email", "TEXT", nullable=False, unique=True),
            Column("age", "INTEGER"),
        ])
        
        posts_table = Table("posts", [
            Column("id", "INTEGER", primary_key=True),
            Column("user_id", "INTEGER", nullable=False),
            Column("title", "TEXT", nullable=False),
            Column("content", "TEXT"),
            Column("published", "BOOLEAN", default=False),
            Column("created_at", "TIMESTAMP", default="CURRENT_TIMESTAMP"),
        ])
        
        # Create tables
        conn.execute(users_table.create_sql())
        conn.execute(posts_table.create_sql())
        
        print("Created tables:")
        print("  - users")
        print("  - posts")
        
        # Insert users
        insert_query = Query().insert_into(users_table.name).columns(
            "name", "email", "age"
        ).values(
            ["Alice", "alice@example.com", 30],
            ["Bob", "bob@example.com", 25],
            ["Charlie", "charlie@example.com", 35]
        )
        
        sql, params = insert_query.build()
        conn.execute(sql, params)
        
        print("\nInserted users:")
        print("  - Alice (30)")
        print("  - Bob (25)")
        print("  - Charlie (35)")
        
        # Insert posts
        insert_query = Query().insert_into(posts_table.name).columns(
            "user_id", "title", "content", "published"
        ).values(
            [1, "Alice's First Post", "This is my first post", True],
            [1, "Alice's Second Post", "This is my second post", False],
            [2, "Bob's First Post", "Hello, world!", True]
        )
        
        sql, params = insert_query.build()
        conn.execute(sql, params)
        
        print("\nInserted posts:")
        print("  - Alice's First Post (published)")
        print("  - Alice's Second Post (draft)")
        print("  - Bob's First Post (published)")
        
        # Query: Select all users
        select_query = Query().select(
            "id", "name", "email", "age"
        ).from_table(
            users_table.name
        ).order_by(
            "name", "ASC"
        )
        
        sql, params = select_query.build()
        result = conn.execute(sql, params)
        
        print("\nAll users:")
        for row in result:
            print(f"  - {row.id}: {row.name} ({row.email}, {row.age})")
        
        # Query: Select published posts with user information
        select_query = Query().select(
            "u.name AS author",
            "p.title",
            "p.content",
            "p.created_at"
        ).from_table(
            "posts AS p"
        ).join(
            "users AS u", "p.user_id = u.id"
        ).where(
            "p.published = ?", [True]
        ).order_by(
            "p.created_at", "DESC"
        )
        
        sql, params = select_query.build()
        result = conn.execute(sql, params)
        
        print("\nPublished posts:")
        for row in result:
            print(f"  - '{row.title}' by {row.author}")
            print(f"    {row.content}")
            print(f"    Published at: {row.created_at}")
        
        # Query: Count posts by user
        select_query = Query().select(
            "u.name",
            Query.func.count("p.id").as_("post_count")
        ).from_table(
            "users AS u"
        ).left_join(
            "posts AS p", "p.user_id = u.id"
        ).group_by(
            "u.id", "u.name"
        ).order_by(
            "post_count", "DESC"
        )
        
        sql, params = select_query.build()
        result = conn.execute(sql, params)
        
        print("\nPost count by user:")
        for row in result:
            print(f"  - {row.name}: {row.post_count} posts")
        
        # Query: Update a post
        update_query = Query().update(
            posts_table.name
        ).set(
            {"published": True}
        ).where(
            "title = ?", ["Alice's Second Post"]
        )
        
        sql, params = update_query.build()
        conn.execute(sql, params)
        
        print("\nUpdated Alice's Second Post to published")
        
        # Query: Delete a user
        delete_query = Query().delete_from(
            users_table.name
        ).where(
            "name = ?", ["Charlie"]
        )
        
        sql, params = delete_query.build()
        conn.execute(sql, params)
        
        print("\nDeleted user Charlie")
        
        # Check remaining users
        select_query = Query().select("name").from_table(users_table.name).order_by("name")
        sql, params = select_query.build()
        result = conn.execute(sql, params)
        
        print("\nRemaining users:")
        for row in result:
            print(f"  - {row.name}")


if __name__ == "__main__":
    main() 