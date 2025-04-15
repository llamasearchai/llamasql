# Filtering

LlamaDB provides a powerful metadata filtering system that allows you to combine vector similarity search with metadata constraints. This guide covers the filtering capabilities and how to use them effectively.

## Basic Filtering

The simplest form of filtering is exact matching on metadata fields:

```python
from llamadb.core import VectorIndex
import numpy as np

# Create an index and add vectors with metadata
index = VectorIndex(dimension=128)

# Add vectors with rich metadata
for i in range(1000):
    vector = np.random.random(128).astype(np.float32)
    
    # Create metadata with various fields
    metadata = {
        "id": i,
        "category": "Technology" if i % 3 == 0 else "Science" if i % 3 == 1 else "Business",
        "rating": float(i % 5 + 1),
        "date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        "tags": ["sample", "vector", f"tag{i % 10}"],
        "is_active": i % 2 == 0,
        "nested": {
            "level1": {
                "level2": i % 10
            }
        }
    }
    
    index.add_item(vector, metadata)

# Search with a simple filter
results = index.search(
    query_vector=np.random.random(128).astype(np.float32),
    k=10,
    filter={"category": "Technology"}  # Only return items in the "Technology" category
)
```

## Filter Operators

LlamaDB supports various operators for more complex filtering:

### Comparison Operators

```python
# Greater than
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$gt": 3}}  # Rating > 3
)

# Greater than or equal
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$gte": 3}}  # Rating >= 3
)

# Less than
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$lt": 3}}  # Rating < 3
)

# Less than or equal
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$lte": 3}}  # Rating <= 3
)

# Equal
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$eq": 3}}  # Rating == 3
)

# Not equal
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$ne": 3}}  # Rating != 3
)
```

### Range Operators

```python
# In range (inclusive)
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$in": [3, 4, 5]}}  # Rating is 3, 4, or 5
)

# Not in range
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$nin": [1, 2]}}  # Rating is not 1 or 2
)

# Between (inclusive)
results = index.search(
    query_vector,
    k=10,
    filter={"rating": {"$between": [2, 4]}}  # 2 <= Rating <= 4
)
```

### String Operators

```python
# String starts with
results = index.search(
    query_vector,
    k=10,
    filter={"category": {"$startswith": "Tech"}}  # Category starts with "Tech"
)

# String ends with
results = index.search(
    query_vector,
    k=10,
    filter={"category": {"$endswith": "ogy"}}  # Category ends with "ogy"
)

# String contains
results = index.search(
    query_vector,
    k=10,
    filter={"category": {"$contains": "ech"}}  # Category contains "ech"
)

# Regular expression
results = index.search(
    query_vector,
    k=10,
    filter={"category": {"$regex": "^T.*y$"}}  # Category matches regex "^T.*y$"
)
```

### Array Operators

```python
# Array contains
results = index.search(
    query_vector,
    k=10,
    filter={"tags": {"$contains": "sample"}}  # Tags contains "sample"
)

# Array contains any
results = index.search(
    query_vector,
    k=10,
    filter={"tags": {"$containsAny": ["tag1", "tag2"]}}  # Tags contains "tag1" or "tag2"
)

# Array contains all
results = index.search(
    query_vector,
    k=10,
    filter={"tags": {"$containsAll": ["sample", "vector"]}}  # Tags contains both "sample" and "vector"
)

# Array length
results = index.search(
    query_vector,
    k=10,
    filter={"tags": {"$size": 3}}  # Tags array has exactly 3 elements
)
```

### Date Operators

```python
# Date before
results = index.search(
    query_vector,
    k=10,
    filter={"date": {"$lt": "2023-06-01"}}  # Date before June 1, 2023
)

# Date after
results = index.search(
    query_vector,
    k=10,
    filter={"date": {"$gt": "2023-01-01"}}  # Date after January 1, 2023
)

# Date between
results = index.search(
    query_vector,
    k=10,
    filter={"date": {"$between": ["2023-01-01", "2023-06-30"]}}  # Date in first half of 2023
)
```

### Logical Operators

```python
# Logical AND
results = index.search(
    query_vector,
    k=10,
    filter={
        "$and": [
            {"category": "Technology"},
            {"rating": {"$gte": 4}}
        ]
    }  # Category is Technology AND rating >= 4
)

# Logical OR
results = index.search(
    query_vector,
    k=10,
    filter={
        "$or": [
            {"category": "Technology"},
            {"category": "Science"}
        ]
    }  # Category is Technology OR Science
)

# Logical NOT
results = index.search(
    query_vector,
    k=10,
    filter={
        "$not": {"category": "Business"}
    }  # Category is NOT Business
)
```

### Nested Fields

```python
# Access nested fields with dot notation
results = index.search(
    query_vector,
    k=10,
    filter={"nested.level1.level2": {"$lt": 5}}  # nested.level1.level2 < 5
)
```

## Complex Queries

You can combine operators for complex filtering needs:

```python
# Complex query combining multiple conditions
results = index.search(
    query_vector,
    k=10,
    filter={
        "$and": [
            # Either Technology category with high rating
            {
                "$or": [
                    {"category": "Technology", "rating": {"$gte": 4}},
                    # Or Science category with specific tag
                    {"category": "Science", "tags": {"$contains": "tag7"}}
                ]
            },
            # Must be active and have a date in the first quarter
            {"is_active": True},
            {"date": {"$between": ["2023-01-01", "2023-03-31"]}}
        ]
    }
)
```

## Performance Considerations

Filtering can impact search performance. Here are some best practices:

1. **Simple filters first**: Start with the most restrictive simple equality filters
2. **Limit complex operations**: Regular expressions and complex logical operations are slower
3. **Indexable fields**: LlamaDB internally indexes some fields for faster filtering
4. **Pre-filtering**: For very restrictive filters, consider using `filter_items()` first, then searching
5. **Benchmark**: Test different filter strategies for your specific use case

Example of pre-filtering:

```python
# For very restrictive filters, pre-filter then search
filtered_ids = index.filter_items({"category": "Technology", "rating": 5})
if filtered_ids:
    results = index.search(
        query_vector,
        k=10,
        filter={"id": {"$in": filtered_ids}}
    )
else:
    results = []
```

## Filter Creation Helpers

LlamaDB provides helper functions to build filters programmatically:

```python
from llamadb.filters import and_filter, or_filter, not_filter, eq, gt, lt, contains

# Create a complex filter using helpers
filter = and_filter(
    eq("category", "Technology"),
    gt("rating", 3),
    or_filter(
        contains("tags", "sample"),
        not_filter(eq("is_active", False))
    )
)

# Use the filter in search
results = index.search(query_vector, k=10, filter=filter)
```

## Serializing and Storing Filters

You can serialize filters to JSON for storage and reuse:

```python
import json

# Create a complex filter
filter = {
    "$and": [
        {"category": "Technology"},
        {"rating": {"$gte": 4}}
    ]
}

# Serialize to JSON
filter_json = json.dumps(filter)

# Store or transmit the filter
# ...

# Deserialize and use later
stored_filter = json.loads(filter_json)
results = index.search(query_vector, k=10, filter=stored_filter)
```

## Next Steps

- [REST API](rest-api.md) - Use LlamaDB from any language or framework
- [API Reference](../api/core.md) - Explore the detailed API documentation
- [Examples](../examples/metadata-filtering.md) - See more filtering examples 