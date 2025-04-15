"""
Tests for the Query module.
"""

import unittest

from llamasql.query import Query
from llamasql.schema import Column, Table


class TestQuery(unittest.TestCase):
    """Test the Query class."""

    def setUp(self):
        """Set up test case."""
        # Define test tables
        self.users = Table(
            "users",
            [
                Column("id", "INTEGER", primary_key=True),
                Column("name", "TEXT", nullable=False),
                Column("email", "TEXT", nullable=False, unique=True),
                Column("age", "INTEGER"),
            ],
        )

        self.posts = Table(
            "posts",
            [
                Column("id", "INTEGER", primary_key=True),
                Column("user_id", "INTEGER", nullable=False),
                Column("title", "TEXT", nullable=False),
                Column("content", "TEXT"),
                Column("published", "BOOLEAN", default=False),
            ],
        )

    def test_simple_select(self):
        """Test a simple SELECT query."""
        query = Query().select("id", "name").from_(self.users)
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users")
        self.assertEqual(params, [])

    def test_select_all(self):
        """Test a SELECT * query."""
        query = Query().select().from_(self.users)
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT * FROM users")
        self.assertEqual(params, [])

    def test_distinct(self):
        """Test a SELECT DISTINCT query."""
        query = Query().select("user_id", distinct=True).from_(self.posts)
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT DISTINCT user_id FROM posts")
        self.assertEqual(params, [])

    def test_where_condition(self):
        """Test a query with a WHERE condition."""
        query = (
            Query().select("id", "name").from_(self.users).where(self.users.c.age > 21)
        )
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users WHERE users.age > ?")
        self.assertEqual(params, [21])

    def test_multiple_where_conditions(self):
        """Test a query with multiple WHERE conditions."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(self.users.c.age > 21)
            .where(self.users.c.name.like("A%"))
        )
        sql, params = query.sql()

        self.assertEqual(
            sql, "SELECT id, name FROM users WHERE users.age > ? AND users.name LIKE ?"
        )
        self.assertEqual(params, [21, "A%"])

    def test_and_conditions(self):
        """Test a query with AND-combined conditions."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where((self.users.c.age > 21) & (self.users.c.name.like("A%")))
        )
        sql, params = query.sql()

        self.assertEqual(
            sql, "SELECT id, name FROM users WHERE users.age > ? AND users.name LIKE ?"
        )
        self.assertEqual(params, [21, "A%"])

    def test_or_conditions(self):
        """Test a query with OR-combined conditions."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where((self.users.c.age < 18) | (self.users.c.age > 65))
        )
        sql, params = query.sql()

        self.assertEqual(
            sql, "SELECT id, name FROM users WHERE users.age < ? OR users.age > ?"
        )
        self.assertEqual(params, [18, 65])

    def test_complex_conditions(self):
        """Test a query with complex conditions."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(
                (self.users.c.age > 21)
                & ((self.users.c.name.like("A%")) | (self.users.c.name.like("B%")))
            )
        )
        sql, params = query.sql()

        self.assertEqual(
            sql,
            "SELECT id, name FROM users WHERE users.age > ? AND (users.name LIKE ? OR users.name LIKE ?)",
        )
        self.assertEqual(params, [21, "A%", "B%"])

    def test_join(self):
        """Test a query with a JOIN."""
        query = (
            Query()
            .select(self.users.c.name, self.posts.c.title)
            .from_(self.users)
            .join(self.posts, self.users.c.id == self.posts.c.user_id)
        )
        sql, params = query.sql()

        self.assertEqual(
            sql,
            "SELECT users.name, posts.title FROM users INNER JOIN posts ON users.id = posts.user_id",
        )
        self.assertEqual(params, [])

    def test_left_join(self):
        """Test a query with a LEFT JOIN."""
        query = (
            Query()
            .select(self.users.c.name, self.posts.c.title)
            .from_(self.users)
            .left_join(self.posts, self.users.c.id == self.posts.c.user_id)
        )
        sql, params = query.sql()

        self.assertEqual(
            sql,
            "SELECT users.name, posts.title FROM users LEFT JOIN posts ON users.id = posts.user_id",
        )
        self.assertEqual(params, [])

    def test_group_by(self):
        """Test a query with GROUP BY."""
        query = (
            Query()
            .select(
                self.users.c.name, Query.func.count(self.posts.c.id).as_("post_count")
            )
            .from_(self.users)
            .left_join(self.posts, self.users.c.id == self.posts.c.user_id)
            .group_by(self.users.c.id, self.users.c.name)
        )
        sql, params = query.sql()

        self.assertEqual(
            sql,
            "SELECT users.name, COUNT(posts.id) AS post_count FROM users LEFT JOIN posts ON users.id = posts.user_id GROUP BY users.id, users.name",
        )
        self.assertEqual(params, [])

    def test_having(self):
        """Test a query with HAVING."""
        query = (
            Query()
            .select(
                self.users.c.name, Query.func.count(self.posts.c.id).as_("post_count")
            )
            .from_(self.users)
            .left_join(self.posts, self.users.c.id == self.posts.c.user_id)
            .group_by(self.users.c.id, self.users.c.name)
            .having(Query.func.count(self.posts.c.id) > 5)
        )
        sql, params = query.sql()

        self.assertEqual(
            sql,
            "SELECT users.name, COUNT(posts.id) AS post_count FROM users LEFT JOIN posts ON users.id = posts.user_id GROUP BY users.id, users.name HAVING COUNT(posts.id) > ?",
        )
        self.assertEqual(params, [5])

    def test_order_by(self):
        """Test a query with ORDER BY."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .order_by(self.users.c.name.asc())
        )
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users ORDER BY users.name ASC")
        self.assertEqual(params, [])

    def test_multiple_order_by(self):
        """Test a query with multiple ORDER BY clauses."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .order_by(self.users.c.age.desc(), self.users.c.name.asc())
        )
        sql, params = query.sql()

        self.assertEqual(
            sql, "SELECT id, name FROM users ORDER BY users.age DESC, users.name ASC"
        )
        self.assertEqual(params, [])

    def test_limit(self):
        """Test a query with LIMIT."""
        query = Query().select("id", "name").from_(self.users).limit(10)
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users LIMIT 10")
        self.assertEqual(params, [])

    def test_offset(self):
        """Test a query with OFFSET."""
        query = Query().select("id", "name").from_(self.users).limit(10).offset(20)
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users LIMIT 10 OFFSET 20")
        self.assertEqual(params, [])

    def test_in_condition(self):
        """Test a query with IN condition."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(self.users.c.id.in_([1, 2, 3]))
        )
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users WHERE users.id IN (?, ?, ?)")
        self.assertEqual(params, [1, 2, 3])

    def test_between_condition(self):
        """Test a query with BETWEEN condition."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(self.users.c.age.between(18, 65))
        )
        sql, params = query.sql()

        self.assertEqual(
            sql, "SELECT id, name FROM users WHERE users.age BETWEEN ? AND ?"
        )
        self.assertEqual(params, [18, 65])

    def test_is_null_condition(self):
        """Test a query with IS NULL condition."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(self.users.c.age.is_null())
        )
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users WHERE users.age IS NULL")
        self.assertEqual(params, [])

    def test_is_not_null_condition(self):
        """Test a query with IS NOT NULL condition."""
        query = (
            Query()
            .select("id", "name")
            .from_(self.users)
            .where(self.users.c.age.is_not_null())
        )
        sql, params = query.sql()

        self.assertEqual(sql, "SELECT id, name FROM users WHERE users.age IS NOT NULL")
        self.assertEqual(params, [])


if __name__ == "__main__":
    unittest.main()
