# Vector Indexes

This guide provides detailed information about the vector indexes available in LlamaDB, how to configure them, and best practices for different use cases.

## Index Types

LlamaDB offers multiple index types optimized for different scenarios:

### Flat Index

The flat index is the simplest index type, storing vectors in a contiguous array and performing brute-force search.

```python
from llamadb.core import VectorIndex

# Create a flat index (default)
index = VectorIndex(
    dimension=128,
    index_type="flat"
)
```

**Characteristics:**
- **Exact search**: Always returns the exact nearest neighbors
- **Fast indexing**: O(1) insertion time
- **Slower search**: O(n) search complexity (linear with dataset size)
- **Low memory overhead**: Minimal additional memory beyond the vectors themselves
- **Ideal for**: Small to medium datasets (up to ~100K vectors) or when exact results are required

### HNSW Index (Hierarchical Navigable Small World)

HNSW is an advanced graph-based index that offers excellent performance for approximate nearest neighbor search.

```python
from llamadb.core import VectorIndex

# Create an HNSW index with custom parameters
index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    M=16,  # Connections per node
    ef_construction=200,  # Build-time accuracy parameter
    ef_search=100  # Query-time accuracy parameter
)
```

**Characteristics:**
- **Approximate search**: Returns approximate nearest neighbors with controllable accuracy
- **Moderate indexing speed**: O(log n) insertion time
- **Fast search**: O(log n) search complexity (logarithmic with dataset size)
- **Higher memory overhead**: Requires additional memory for graph structure
- **Ideal for**: Medium to large datasets (100K to billions of vectors) where approximate results are acceptable

### Hybrid Index

LlamaDB also offers a hybrid index that combines multiple index types for different trade-offs.

```python
from llamadb.core import VectorIndex

# Create a hybrid index
index = VectorIndex(
    dimension=128,
    index_type="hybrid",
    primary_index="hnsw",
    secondary_index="flat",
    primary_threshold=10000  # Use flat index until 10K vectors, then switch to HNSW
)
```

**Characteristics:**
- **Adaptive performance**: Automatically switches between index types based on dataset size
- **Balanced trade-offs**: Combines advantages of multiple index types
- **Dynamic reconfiguration**: Can reindex as dataset grows
- **Ideal for**: Applications where dataset size will grow over time

## Index Configuration Parameters

### Common Parameters

Parameters available for all index types:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `dimension` | Vector dimension | Required | 2-10,000 |
| `metric` | Distance metric ("cosine", "l2", "dot") | "cosine" | - |
| `use_mlx` | Whether to use MLX acceleration | True | - |
| `allow_version_upgrade` | Allow automatic format upgrades | True | - |

### HNSW Parameters

Parameters specific to HNSW indexes:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `M` | Connections per node | 16 | 8-64 |
| `ef_construction` | Build-time accuracy | 200 | 100-500 |
| `ef_search` | Query-time accuracy | 100 | 50-400 |
| `max_elements` | Maximum capacity | 1,000,000 | Based on memory |

## Index Construction

### Basic Creation

The simplest way to create an index:

```python
from llamadb.core import VectorIndex

# Create an index
index = VectorIndex(dimension=128)
```

### Advanced Creation

For more control over the index:

```python
from llamadb.core import VectorIndex, IndexConfig

# Create a configuration
config = IndexConfig(
    dimension=128,
    metric="cosine",
    use_mlx=True,
    ef_construction=200,
    ef_search=100,
    M=16
)

# Create an index with the configuration
index = VectorIndex(config=config)
```

### From Existing Data

Creating an index from existing vectors:

```python
import numpy as np
from llamadb.core import VectorIndex

# Create 1000 random vectors with 128 dimensions
vectors = np.random.random((1000, 128)).astype(np.float32)

# Create metadata for each vector
metadata_list = [{"id": i, "category": "Technology"} for i in range(1000)]

# Create an index and bulk load the vectors
index = VectorIndex.from_vectors(
    vectors=vectors,
    metadata_list=metadata_list,
    dimension=128,
    index_type="hnsw"
)
```

## Index Operations

### Adding Vectors

Adding vectors individually:

```python
# Add a single vector
index.add_item(
    embedding=[0.1, 0.2, ...],  # 128-dim vector
    metadata={"id": 1, "text": "Document content"}
)
```

Adding vectors in bulk (much faster):

```python
# Add multiple vectors
index.add_items(
    embeddings=vectors,  # Shape (n, dimension)
    metadata_list=metadata_list  # List of n metadata dictionaries
)
```

### Updating Vectors

Updating vectors:

```python
# Update a vector by ID
index.update_item(
    id=42,
    embedding=new_vector,
    metadata={"id": 42, "text": "Updated content"}
)
```

### Deleting Vectors

Removing vectors:

```python
# Delete a vector by ID
index.delete_item(id=42)

# Delete multiple vectors by ID
index.delete_items(ids=[42, 43, 44])

# Delete vectors matching a filter
index.delete_by_filter(filter={"category": "Outdated"})
```

### Searching

Basic search:

```python
# Search for similar vectors
results = index.search(
    query_vector=[0.1, 0.2, ...],
    k=10
)
```

Advanced search with filters:

```python
# Search with filtering
results = index.search(
    query_vector=[0.1, 0.2, ...],
    k=10,
    filter={"category": "Technology", "date": {"$gt": "2023-01-01"}},
    include_vectors=True  # Include the actual vectors in results
)
```

## Index Persistence

### Saving an Index

```python
# Save the index to disk
index.save("my_index.llamadb")
```

### Loading an Index

```python
# Load the index from disk
loaded_index = VectorIndex.load("my_index.llamadb")
```

### Auto-saving

```python
# Create an index with auto-save
index = VectorIndex(
    dimension=128,
    auto_save=True,
    auto_save_path="my_index.llamadb",
    auto_save_interval=300  # Save every 5 minutes
)
```

## Performance Considerations

### Memory Usage

Memory usage can be estimated as:

- **Flat Index**: `vector_size * num_vectors * 4 bytes` (for float32) + `metadata_size`
- **HNSW Index**: `vector_size * num_vectors * 4 bytes` + `graph_overhead` + `metadata_size`

Where:
- `vector_size` is the dimension of your vectors
- `num_vectors` is the number of vectors in the index
- `graph_overhead` depends on `M` and is approximately `M * 8 bytes * num_vectors`
- `metadata_size` depends on your metadata and is typically 100-500 bytes per vector

### Building Efficiently

For optimal performance when building a large index:

```python
# Pre-allocate a large index for better performance
index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    max_elements=10_000_000,  # Allocate for 10M vectors upfront
    ef_construction=100,  # Lower for faster building
    M=12  # Lower for faster building
)

# Add vectors in large batches
batch_size = 100_000
for i in range(0, len(all_vectors), batch_size):
    index.add_items(
        embeddings=all_vectors[i:i+batch_size],
        metadata_list=all_metadata[i:i+batch_size]
    )
    print(f"Added {i + batch_size} vectors")

# Increase search quality after building
index.set_ef_search(200)
```

### Search Optimization

To optimize search performance:

1. **Use filters wisely**: More complex filters slow down search
2. **Adjust ef_search**: Higher for better accuracy, lower for speed
3. **Batch similar queries**: Use `search_batch` for multiple similar queries
4. **Limit result size**: Only request as many results as needed
5. **Use MLX acceleration**: Enable on Apple Silicon for best performance

## Next Steps

- [MLX Acceleration](mlx-acceleration.md) - Learn how to leverage Apple Silicon for 10x performance
- [Filtering](filtering.md) - Master the filtering capabilities
- [API Reference: Vector Index](../api/vector-index.md) - Detailed API documentation 