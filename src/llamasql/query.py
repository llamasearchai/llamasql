"""
Query module for building SQL queries.
"""

import json
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llamasql.schema import Column, ColumnReference, Table


class SQLFunction:
    """
    SQL function call (COUNT, SUM, AVG, etc.)
    """

    def __init__(self, name: str):
        self.name = name.upper()

    def __call__(self, *args) -> "FunctionCall":
        return FunctionCall(self.name, args)


class SQLFunctionFactory:
    """
    Factory for SQL functions.

    Allows for syntax like: Query.func.count(*)
    """

    def __getattr__(self, name: str) -> SQLFunction:
        return SQLFunction(name)


class FunctionCall:
    """
    Represents a SQL function call expression.
    """

    def __init__(self, function_name: str, args: tuple):
        self.function_name = function_name
        self.args = args

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.function_name}({args_str})"

    # Allow method chaining for aggregate functions
    def asc(self) -> "OrderBy":
        """Sort ascending by this function result."""
        return OrderBy(self, "ASC")

    def desc(self) -> "OrderBy":
        """Sort descending by this function result."""
        return OrderBy(self, "DESC")


class OrderBy:
    """
    Represents an ORDER BY clause in a SQL query.
    """

    def __init__(self, expression: Any, direction: str = "ASC"):
        self.expression = expression
        self.direction = direction

    def __str__(self) -> str:
        return f"{self.expression} {self.direction}"


class JoinType(Enum):
    """
    SQL join types
    """

    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()

    def __str__(self) -> str:
        if self == JoinType.INNER:
            return "INNER JOIN"
        elif self == JoinType.LEFT:
            return "LEFT JOIN"
        elif self == JoinType.RIGHT:
            return "RIGHT JOIN"
        elif self == JoinType.FULL:
            return "FULL JOIN"
        return ""


class Condition:
    """
    Represents a SQL condition (for WHERE, HAVING, etc.)
    """

    def __init__(
        self, left: Any, operator: str, right: Any, connector: Optional[str] = None
    ):
        self.left = left
        self.operator = operator
        self.right = right
        self.connector = connector

    def __str__(self) -> str:
        # Handle NULL conditions
        if self.operator in ("IS", "IS NOT") and self.right is None:
            return f"{self.left} {self.operator} NULL"

        # Handle IN conditions
        if self.operator == "IN" and isinstance(self.right, (list, tuple)):
            placeholders = ", ".join(["?"] * len(self.right))
            return f"{self.left} IN ({placeholders})"

        # Handle BETWEEN conditions
        if (
            self.operator == "BETWEEN"
            and isinstance(self.right, tuple)
            and len(self.right) == 2
        ):
            return f"{self.left} BETWEEN ? AND ?"

        return f"{self.left} {self.operator} ?"

    # Logical operators for combining conditions
    def __and__(self, other: "Condition") -> "Condition":
        if not isinstance(other, Condition):
            raise TypeError("AND operator only works with Condition objects")

        return CompoundCondition([self, other], "AND")

    def __or__(self, other: "Condition") -> "Condition":
        if not isinstance(other, Condition):
            raise TypeError("OR operator only works with Condition objects")

        return CompoundCondition([self, other], "OR")

    def get_parameters(self) -> list:
        """Get the parameters for this condition."""
        # Handle NULL conditions
        if self.operator in ("IS", "IS NOT") and self.right is None:
            return []

        # Handle IN conditions
        if self.operator == "IN" and isinstance(self.right, (list, tuple)):
            return list(self.right)

        # Handle BETWEEN conditions
        if (
            self.operator == "BETWEEN"
            and isinstance(self.right, tuple)
            and len(self.right) == 2
        ):
            return list(self.right)

        return [self.right]


class CompoundCondition:
    """
    Represents a compound condition joined by logical operators.
    """

    def __init__(
        self, conditions: List[Union[Condition, "CompoundCondition"]], operator: str
    ):
        self.conditions = conditions
        self.operator = operator

    def __str__(self) -> str:
        parts = []
        for condition in self.conditions:
            # Add parentheses around compound conditions
            if isinstance(condition, CompoundCondition):
                parts.append(f"({condition})")
            else:
                parts.append(str(condition))

        return f" {self.operator} ".join(parts)

    # Logical operators for combining conditions
    def __and__(
        self, other: Union[Condition, "CompoundCondition"]
    ) -> "CompoundCondition":
        if not isinstance(other, (Condition, CompoundCondition)):
            raise TypeError(
                "AND operator only works with Condition or CompoundCondition objects"
            )

        if self.operator == "AND":
            # Append to existing AND condition
            return CompoundCondition(self.conditions + [other], "AND")
        else:
            # Create new AND condition wrapping this OR condition
            return CompoundCondition([self, other], "AND")

    def __or__(
        self, other: Union[Condition, "CompoundCondition"]
    ) -> "CompoundCondition":
        if not isinstance(other, (Condition, CompoundCondition)):
            raise TypeError(
                "OR operator only works with Condition or CompoundCondition objects"
            )

        if self.operator == "OR":
            # Append to existing OR condition
            return CompoundCondition(self.conditions + [other], "OR")
        else:
            # Create new OR condition wrapping this AND condition
            return CompoundCondition([self, other], "OR")

    def get_parameters(self) -> list:
        """Get the parameters for this compound condition."""
        params = []
        for condition in self.conditions:
            if isinstance(condition, Condition):
                params.extend(condition.get_parameters())
            elif isinstance(condition, CompoundCondition):
                params.extend(condition.get_parameters())
        return params


class Query:
    """
    SQL query builder with fluent API.

    Example:
        query = (Query()
            .select("users.name", "users.email")
            .from_("users")
            .where(("users.active", "=", True))
            .order_by("users.name")
            .limit(10)
        )
    """

    # SQL function factory for use like Query.func.count(*)
    func = SQLFunctionFactory()

    def __init__(self):
        """Initialize an empty query."""
        self._select_columns = []
        self._from_table = None
        self._joins = []
        self._where_conditions = []
        self._group_by = []
        self._having_conditions = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._distinct = False
        self._parameters = []

    def select(self, *columns, distinct: bool = False) -> "Query":
        """
        Set the SELECT part of the query.

        Args:
            *columns: Column names, ColumnReferences, or expressions
            distinct: Whether to select distinct values

        Returns:
            This query for chaining
        """
        self._select_columns = list(columns)
        self._distinct = distinct
        return self

    def from_(self, table: Union[str, Table]) -> "Query":
        """
        Set the FROM part of the query.

        Args:
            table: Table name or Table object

        Returns:
            This query for chaining
        """
        self._from_table = table
        return self

    def join(
        self,
        table: Union[str, Table],
        on: Union[Condition, str],
        join_type: JoinType = JoinType.INNER,
    ) -> "Query":
        """
        Add a JOIN clause to the query.

        Args:
            table: Table name or Table object to join
            on: Join condition
            join_type: Type of join (INNER, LEFT, RIGHT, FULL)

        Returns:
            This query for chaining
        """
        self._joins.append((table, on, join_type))
        return self

    def left_join(self, table: Union[str, Table], on: Union[Condition, str]) -> "Query":
        """
        Add a LEFT JOIN clause to the query.

        Args:
            table: Table name or Table object to join
            on: Join condition

        Returns:
            This query for chaining
        """
        return self.join(table, on, JoinType.LEFT)

    def right_join(
        self, table: Union[str, Table], on: Union[Condition, str]
    ) -> "Query":
        """
        Add a RIGHT JOIN clause to the query.

        Args:
            table: Table name or Table object to join
            on: Join condition

        Returns:
            This query for chaining
        """
        return self.join(table, on, JoinType.RIGHT)

    def full_join(self, table: Union[str, Table], on: Union[Condition, str]) -> "Query":
        """
        Add a FULL JOIN clause to the query.

        Args:
            table: Table name or Table object to join
            on: Join condition

        Returns:
            This query for chaining
        """
        return self.join(table, on, JoinType.FULL)

    def where(self, condition: Union[Condition, CompoundCondition]) -> "Query":
        """
        Add a WHERE condition to the query.

        Args:
            condition: WHERE condition

        Returns:
            This query for chaining
        """
        self._where_conditions.append(condition)
        return self

    def group_by(self, *columns) -> "Query":
        """
        Add a GROUP BY clause to the query.

        Args:
            *columns: Columns to group by

        Returns:
            This query for chaining
        """
        self._group_by.extend(columns)
        return self

    def having(self, condition: Union[Condition, CompoundCondition]) -> "Query":
        """
        Add a HAVING condition to the query.

        Args:
            condition: HAVING condition

        Returns:
            This query for chaining
        """
        self._having_conditions.append(condition)
        return self

    def order_by(self, *expressions) -> "Query":
        """
        Add an ORDER BY clause to the query.

        Args:
            *expressions: Expressions to order by

        Returns:
            This query for chaining
        """
        self._order_by.extend(expressions)
        return self

    def limit(self, limit: int) -> "Query":
        """
        Add a LIMIT clause to the query.

        Args:
            limit: Maximum number of rows to return

        Returns:
            This query for chaining
        """
        self._limit = limit
        return self

    def offset(self, offset: int) -> "Query":
        """
        Add an OFFSET clause to the query.

        Args:
            offset: Number of rows to skip

        Returns:
            This query for chaining
        """
        self._offset = offset
        return self

    def sql(self) -> Tuple[str, list]:
        """
        Generate the SQL query string and parameters.

        Returns:
            Tuple of (sql_string, parameters)
        """
        # Build the query
        sql_parts = []
        parameters = []

        # SELECT clause
        select_clause = "SELECT"
        if self._distinct:
            select_clause += " DISTINCT"

        if not self._select_columns:
            select_clause += " *"
        else:
            select_clause += " " + ", ".join(str(c) for c in self._select_columns)

        sql_parts.append(select_clause)

        # FROM clause
        if self._from_table:
            sql_parts.append(f"FROM {self._from_table}")

        # JOIN clauses
        for table, on, join_type in self._joins:
            join_clause = f"{join_type} {table} ON {on}"
            sql_parts.append(join_clause)

            # Add parameters for the join condition
            if isinstance(on, (Condition, CompoundCondition)):
                parameters.extend(on.get_parameters())

        # WHERE clause
        if self._where_conditions:
            where_conditions_str = " AND ".join(str(c) for c in self._where_conditions)
            sql_parts.append(f"WHERE {where_conditions_str}")

            # Add parameters for the where conditions
            for condition in self._where_conditions:
                parameters.extend(condition.get_parameters())

        # GROUP BY clause
        if self._group_by:
            group_by_str = ", ".join(str(c) for c in self._group_by)
            sql_parts.append(f"GROUP BY {group_by_str}")

        # HAVING clause
        if self._having_conditions:
            having_conditions_str = " AND ".join(
                str(c) for c in self._having_conditions
            )
            sql_parts.append(f"HAVING {having_conditions_str}")

            # Add parameters for the having conditions
            for condition in self._having_conditions:
                parameters.extend(condition.get_parameters())

        # ORDER BY clause
        if self._order_by:
            order_by_str = ", ".join(str(o) for o in self._order_by)
            sql_parts.append(f"ORDER BY {order_by_str}")

        # LIMIT clause
        if self._limit is not None:
            sql_parts.append(f"LIMIT {self._limit}")

        # OFFSET clause
        if self._offset is not None:
            sql_parts.append(f"OFFSET {self._offset}")

        # Combine the parts
        sql = " ".join(sql_parts)

        return sql, parameters

    def __str__(self) -> str:
        """Return the SQL query string without parameters."""
        sql, _ = self.sql()
        return sql

    def execute(self, connection) -> Any:
        """
        Execute the query on the given connection.

        Args:
            connection: Database connection

        Returns:
            Query result (implementation-specific)
        """
        sql, params = self.sql()
        return connection.execute(sql, params)
