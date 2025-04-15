# Core API Reference

This page documents the core APIs of LlamaDB.

## Module: `llamadb.core`

The core module contains the fundamental classes and functions for working with vector data.

### `VectorIndex`

::: llamadb.core.VectorIndex
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### Vector Operations

#### `cosine_similarity`

```python
def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
```

#### `l2_distance`

```python
def l2_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate the L2 (Euclidean) distance between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        L2 distance value (non-negative)
    """
```

#### `dot_product`

```python
def dot_product(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate the dot product between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        Dot product value
    """
```

### Utility Functions

#### `is_mlx_available`

```python
def is_mlx_available() -> bool:
    """
    Check if MLX is available for acceleration.
    
    Returns:
        True if MLX is available, False otherwise
    """
```

#### `is_apple_silicon`

```python
def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon hardware.
    
    Returns:
        True if running on Apple Silicon, False otherwise
    """
```

## Module: `llamadb.models`

### Data Models

#### `SearchResult`

```python
class SearchResult:
    """
    A search result containing a vector, its metadata, and similarity score.
    
    Attributes:
        id (int): The unique identifier of the vector
        score (float): The similarity score
        metadata (Dict[str, Any]): The metadata associated with the vector
        vector (Optional[np.ndarray]): The vector itself (optional)
    """
    
    def __init__(
        self, 
        id: int, 
        score: float, 
        metadata: Dict[str, Any], 
        vector: Optional[np.ndarray] = None
    ):
        """
        Initialize a SearchResult.
        
        Args:
            id: Unique identifier
            score: Similarity score
            metadata: Associated metadata
            vector: The vector itself (optional)
        """
        self.id = id
        self.score = score
        self.metadata = metadata
        self.vector = vector
        
    def __repr__(self) -> str:
        """String representation of the search result."""
        return f"SearchResult(id={self.id}, score={self.score:.4f}, metadata={self.metadata})"
```

#### `IndexConfig`

```python
class IndexConfig:
    """
    Configuration for a vector index.
    
    Attributes:
        dimension (int): The dimensionality of vectors in the index
        metric (str): The distance metric to use ('cosine', 'l2', 'dot')
        use_mlx (bool): Whether to use MLX acceleration if available
        ef_construction (int): HNSW index parameter for construction
        ef_search (int): HNSW index parameter for search
        M (int): HNSW index parameter for connections per node
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        use_mlx: bool = True,
        ef_construction: int = 200,
        ef_search: int = 100,
        M: int = 16
    ):
        """
        Initialize an IndexConfig.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric ('cosine', 'l2', 'dot')
            use_mlx: Whether to use MLX acceleration if available
            ef_construction: HNSW parameter for construction
            ef_search: HNSW parameter for search
            M: HNSW parameter for connections per node
        """
        self.dimension = dimension
        self.metric = metric
        self.use_mlx = use_mlx
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        
    def __repr__(self) -> str:
        """String representation of the index config."""
        return (
            f"IndexConfig(dimension={self.dimension}, metric={self.metric}, "
            f"use_mlx={self.use_mlx}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, M={self.M})"
        )
```

## Module: `llamadb.errors`

### Exceptions

#### `DimensionMismatchError`

```python
class DimensionMismatchError(Exception):
    """Raised when vector dimensions don't match the index configuration."""
    pass
```

#### `EmptyIndexError`

```python
class EmptyIndexError(Exception):
    """Raised when attempting to search an empty index."""
    pass
```

#### `InvalidFilterError`

```python
class InvalidFilterError(Exception):
    """Raised when an invalid filter is provided."""
    pass
```

#### `MLXNotAvailableError`

```python
class MLXNotAvailableError(Exception):
    """Raised when attempting to use MLX on an unsupported platform."""
    pass
```

## Usage Examples

### Creating and Searching an Index

```python
from llamadb.core import VectorIndex
import numpy as np

# Create a new index
index = VectorIndex(dimension=128)

# Add vectors
for i in range(100):
    vector = np.random.random(128).astype(np.float32)
    index.add_item(vector, {"id": i, "category": "Technology"})
    
# Search
query = np.random.random(128).astype(np.float32)
results = index.search(query, k=10)

# Print results
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

### Computing Vector Similarities

```python
from llamadb.core import cosine_similarity, l2_distance, dot_product
import numpy as np

# Create two random vectors
a = np.random.random(128).astype(np.float32)
b = np.random.random(128).astype(np.float32)

# Calculate similarities
cos_sim = cosine_similarity(a, b)
l2_dist = l2_distance(a, b)
dot_prod = dot_product(a, b)

print(f"Cosine similarity: {cos_sim}")
print(f"L2 distance: {l2_dist}")
print(f"Dot product: {dot_prod}")
``` 