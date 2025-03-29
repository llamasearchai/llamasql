"""
Schema module for defining database tables and columns.
"""
from typing import List, Dict, Any, Optional, Union, TypeVar


class Column:
    """
    Represents a database column with type and constraints.
    
    Attributes:
        name (str): The column name
        type_name (str): The SQL type of the column
        primary_key (bool): Whether this column is a primary key
        nullable (bool): Whether this column allows NULL values
        unique (bool): Whether this column has a UNIQUE constraint
        default: The default value for the column
        references: Reference to another table (for foreign keys)
    """
    
    def __init__(
        self,
        name: str,
        type_name: str,
        primary_key: bool = False,
        nullable: bool = True,
        unique: bool = False,
        default: Any = None,
        references: Optional[str] = None,
    ):
        """
        Initialize a column.
        
        Args:
            name: The column name
            type_name: The SQL type of the column (e.g. "INTEGER", "TEXT", etc.)
            primary_key: Whether this column is a primary key
            nullable: Whether this column allows NULL values
            unique: Whether this column has a UNIQUE constraint
            default: The default value for the column
            references: Reference to another table (for foreign keys)
        """
        self.name = name
        self.type_name = type_name
        self.primary_key = primary_key
        self.nullable = nullable
        self.unique = unique
        self.default = default
        self.references = references
        self.table = None  # Will be set when added to a table
    
    def __str__(self) -> str:
        return f"{self.name} {self.type_name}"
    
    def __repr__(self) -> str:
        return f"Column({self.name!r}, {self.type_name!r}, primary_key={self.primary_key}, nullable={self.nullable})"
    
    @property
    def sql_definition(self) -> str:
        """Generate the SQL column definition for CREATE TABLE statements."""
        parts = [f'"{self.name}" {self.type_name}']
        
        if self.primary_key:
            parts.append("PRIMARY KEY")
        
        if not self.nullable and not self.primary_key:
            parts.append("NOT NULL")
        
        if self.unique and not self.primary_key:
            parts.append("UNIQUE")
        
        if self.default is not None:
            if isinstance(self.default, str) and not self.default.startswith(("'", '"')):
                # Handle special SQL expressions like CURRENT_TIMESTAMP
                parts.append(f"DEFAULT {self.default}")
            else:
                parts.append(f"DEFAULT {repr(self.default)}")
        
        if self.references:
            parts.append(f"REFERENCES {self.references}")
        
        return " ".join(parts)


class ColumnReference:
    """
    A reference to a column in a table. Used for query building.
    """
    
    def __init__(self, table: 'Table', column: Column):
        """
        Initialize a column reference.
        
        Args:
            table: The table this column belongs to
            column: The column being referenced
        """
        self.table = table
        self.column = column
    
    def __str__(self) -> str:
        """String representation as table_name.column_name."""
        return f'{self.table.name}.{self.column.name}'
    
    # Comparison operators for use in query conditions
    def __eq__(self, other: Any) -> 'Condition':
        """Equality comparison, e.g., column == value."""
        from llamasql.query import Condition
        return Condition(self, "=", other)
    
    def __ne__(self, other: Any) -> 'Condition':
        """Inequality comparison, e.g., column != value."""
        from llamasql.query import Condition
        return Condition(self, "<>", other)
    
    def __lt__(self, other: Any) -> 'Condition':
        """Less than comparison, e.g., column < value."""
        from llamasql.query import Condition
        return Condition(self, "<", other)
    
    def __le__(self, other: Any) -> 'Condition':
        """Less than or equal comparison, e.g., column <= value."""
        from llamasql.query import Condition
        return Condition(self, "<=", other)
    
    def __gt__(self, other: Any) -> 'Condition':
        """Greater than comparison, e.g., column > value."""
        from llamasql.query import Condition
        return Condition(self, ">", other)
    
    def __ge__(self, other: Any) -> 'Condition':
        """Greater than or equal comparison, e.g., column >= value."""
        from llamasql.query import Condition
        return Condition(self, ">=", other)
    
    # Sorting helpers
    def asc(self) -> 'OrderBy':
        """Sort ascending by this column."""
        from llamasql.query import OrderBy
        return OrderBy(self, "ASC")
    
    def desc(self) -> 'OrderBy':
        """Sort descending by this column."""
        from llamasql.query import OrderBy
        return OrderBy(self, "DESC")
    
    # Special SQL operators
    def like(self, pattern: str) -> 'Condition':
        """LIKE operator, e.g., column.like('%pattern%')."""
        from llamasql.query import Condition
        return Condition(self, "LIKE", pattern)
    
    def in_(self, values: List[Any]) -> 'Condition':
        """IN operator, e.g., column.in_([1, 2, 3])."""
        from llamasql.query import Condition
        return Condition(self, "IN", values)
    
    def between(self, lower: Any, upper: Any) -> 'Condition':
        """BETWEEN operator, e.g., column.between(1, 10)."""
        from llamasql.query import Condition
        return Condition(self, "BETWEEN", (lower, upper))
    
    def is_null(self) -> 'Condition':
        """IS NULL operator, e.g., column.is_null()."""
        from llamasql.query import Condition
        return Condition(self, "IS", None)
    
    def is_not_null(self) -> 'Condition':
        """IS NOT NULL operator, e.g., column.is_not_null()."""
        from llamasql.query import Condition
        return Condition(self, "IS NOT", None)


class ColumnCollection:
    """
    A collection of column references for a table.
    
    Allows for easy access to columns by attribute, e.g., table.c.column_name
    """
    
    def __init__(self, table: 'Table'):
        """
        Initialize a column collection for a table.
        
        Args:
            table: The table this collection belongs to
        """
        self.table = table
        
        # Create references to all columns
        for column in table.columns:
            setattr(self, column.name, ColumnReference(table, column))
    
    def __getattr__(self, name: str) -> ColumnReference:
        """Get a column reference by name."""
        raise AttributeError(f"Table '{self.table.name}' has no column '{name}'")


class Table:
    """
    Represents a database table.
    
    Attributes:
        name (str): The table name
        columns (List[Column]): The columns in this table
        c: A ColumnCollection for easy access to column references
    """
    
    def __init__(self, name: str, columns: Optional[List[Column]] = None):
        """
        Initialize a table.
        
        Args:
            name: The table name
            columns: The columns in this table (optional)
        """
        self.name = name
        self.columns = columns or []
        
        # Set the table reference on each column
        for column in self.columns:
            column.table = self
        
        # Create the column collection
        self.c = ColumnCollection(self)
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Table({self.name!r}, columns={len(self.columns)})"
    
    def add_column(self, column: Column) -> None:
        """
        Add a column to the table.
        
        Args:
            column: The column to add
        """
        column.table = self
        self.columns.append(column)
        
        # Update the column collection
        setattr(self.c, column.name, ColumnReference(self, column))
    
    @property
    def column_names(self) -> List[str]:
        """Get a list of column names."""
        return [column.name for column in self.columns]
    
    @property
    def sql_definition(self) -> str:
        """Generate the SQL table definition for CREATE TABLE statements."""
        columns_sql = ",\n  ".join(column.sql_definition for column in self.columns)
        return f'CREATE TABLE "{self.name}" (\n  {columns_sql}\n)' 