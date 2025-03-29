# Similarity Search in LlamaDB

This guide explains the similarity search capabilities in LlamaDB, covering different distance metrics, search algorithms, and optimization techniques.

## Understanding Similarity Search

Similarity search (also known as vector search or nearest neighbor search) finds items in a collection that are most similar to a query item based on vector representations. The "similarity" is measured using distance metrics in the vector space.

## Distance Metrics

LlamaDB supports several distance metrics, each with different characteristics:

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, ignoring their magnitude:

```python
from llamadb.core import VectorIndex

# Create an index with cosine similarity
index = VectorIndex(dimension=128, metric="cosine")
```

**Characteristics:**
- Range: -1 to 1 (higher values indicate greater similarity)
- Sensitive to orientation, not magnitude
- Best for text embeddings that benefit from normalization
- Works well with models like OpenAI's embeddings and sentence-transformers

**Note:** LlamaDB automatically normalizes vectors for cosine similarity to optimize performance.

### Euclidean Distance (L2)

Euclidean distance measures the straight-line distance between two points:

```python
# Create an index with L2 distance
index = VectorIndex(dimension=128, metric="l2")
```

**Characteristics:**
- Range: 0 to ∞ (lower values indicate greater similarity)
- Sensitive to both orientation and magnitude
- Good for embeddings where magnitude matters (e.g., certain image embeddings)
- LlamaDB returns negative distances to maintain the "higher is better" convention

### Dot Product

Dot product measures the product of vector magnitudes and the cosine of the angle between them:

```python
# Create an index with dot product
index = VectorIndex(dimension=128, metric="dot")
```

**Characteristics:**
- Range: -∞ to ∞ (higher values indicate greater similarity)
- Sensitive to both orientation and magnitude
- Good for embeddings specifically trained with dot product
- Useful for certain retrieval models trained with dot product as the objective

## Search Algorithms

LlamaDB offers different algorithms for vector search, balancing accuracy, speed, and memory usage:

### Flat Index (Exhaustive Search)

The simplest approach that compares the query vector against every vector in the collection:

```python
# Create a flat index
index = VectorIndex(dimension=128, index_type="flat")
```

**Characteristics:**
- **Accuracy**: 100% accurate (always finds true nearest neighbors)
- **Speed**: O(n) - scales linearly with collection size, slow for large collections
- **Memory Usage**: Minimal overhead beyond storing vectors
- **Best for**: Small to medium collections (up to ~100K vectors) with infrequent updates

### HNSW (Hierarchical Navigable Small World)

An approximate nearest neighbor algorithm that builds a graph-based index structure:

```python
# Create an HNSW index with custom parameters
index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    ef_construction=200,  # Controls index quality
    ef_search=50,         # Controls search accuracy
    M=16                  # Controls graph connectivity
)
```

**Characteristics:**
- **Accuracy**: Approximate (controllable via parameters)
- **Speed**: O(log n) - dramatically faster than flat search for large collections
- **Memory Usage**: Higher than flat (typically 1.5-2x the vector storage)
- **Best for**: Large collections (100K+ vectors) with high search throughput requirements

#### HNSW Parameters

- **ef_construction**: Higher values create a better index but take longer to build (default: 200)
- **ef_search**: Higher values give more accurate search results but increase search time (default: 50)
- **M**: Higher values create a more connected graph, improving accuracy but increasing memory usage (default: 16)

```python
# Example: Optimized for build speed (less accurate)
fast_build_index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    ef_construction=100,  # Lower for faster builds
    M=8                   # Lower for less memory usage
)

# Example: Optimized for search accuracy
accurate_index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    ef_construction=400,  # Higher for better index quality
    ef_search=200,        # Higher for more accurate searches
    M=32                  # Higher for better graph connectivity
)
```

## Search Parameters

### k - Number of Results

The `k` parameter controls how many nearest neighbors to return:

```python
# Get the 10 most similar vectors
results = index.search(query_vector, k=10)

# Get only the single most similar vector
nearest = index.search(query_vector, k=1)[0]
```

### include_vectors

Set `include_vectors=True` to include the actual vector values in the results:

```python
results = index.search(
    query_vector=query_vector,
    k=10,
    include_vectors=True
)

# Access the vector
first_vector = results[0].vector  # numpy array
```

### Include Metadata

LlamaDB always includes metadata in search results:

```python
results = index.search(query_vector, k=10)

# Access metadata
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Title: {result.metadata.get('title')}")
    print(f"Tags: {result.metadata.get('tags')}")
```

### Filtering

You can combine vector search with metadata filtering:

```python
# Search only within a specific category
results = index.search(
    query_vector=query_vector,
    k=10,
    filter={"category": "Technology"}
)

# Complex filtering with multiple conditions
results = index.search(
    query_vector=query_vector,
    k=10,
    filter={
        "$and": [
            {"category": "Technology"},
            {"rating": {"$gte": 4}},
            {"date": {"$gt": "2023-01-01"}}
        ]
    }
)
```

See the [Filtering Guide](filtering.md) for detailed information on filter options.

## Advanced Search Techniques

### Batched Search

For multiple queries, use batched search for better performance:

```python
# Create multiple query vectors
query_vectors = np.random.random((5, 128)).astype(np.float32)

# Search for each query vector in a single call
results_batch = index.search_batch(
    query_vectors=query_vectors,
    k=10
)

# Process results for each query
for i, results in enumerate(results_batch):
    print(f"Results for query {i+1}:")
    for result in results:
        print(f"  {result.id}: {result.score}")
```

### Multi-Query Expansion

Improve recall by searching with multiple query formulations:

```python
def multi_query_search(text, embedder, index, k=10):
    """Search with multiple query formulations to improve recall"""
    # Generate different query formulations
    queries = [
        text,  # Original query
        f"Information about {text}",  # Expanded version
        f"Tell me about {text}"  # Alternative formulation
    ]
    
    # Generate embeddings for each query
    query_vectors = [embedder.embed(q) for q in queries]
    
    # Search with each query vector
    all_results = []
    for qv in query_vectors:
        results = index.search(qv, k=k)
        all_results.extend(results)
    
    # Deduplicate by ID
    unique_results = {}
    for result in all_results:
        if result.id not in unique_results or result.score > unique_results[result.id].score:
            unique_results[result.id] = result
    
    # Sort by score and return top k
    sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
    return sorted_results[:k]
```

### Negative Queries

Improve relevance by pushing results away from certain concepts:

```python
def negative_query_search(positive_vector, negative_vector, index, k=10, alpha=0.3):
    """Search with a positive direction and a negative direction"""
    # Combine positive and negative vectors
    # Subtract a portion of the negative vector from the positive vector
    import numpy as np
    
    combined_vector = positive_vector - alpha * negative_vector
    
    # Normalize the result
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector = combined_vector / norm
    
    # Search with the combined vector
    return index.search(combined_vector, k=k)

# Example: Search for "programming" but not "javascript"
programming_vector = embedder.embed("programming")
javascript_vector = embedder.embed("javascript")

results = negative_query_search(programming_vector, javascript_vector, index)
```

## MLX Acceleration

On Apple Silicon devices, LlamaDB can leverage MLX for faster search:

```python
# Create an index with MLX acceleration
index = VectorIndex(
    dimension=128,
    use_mlx=True  # Enable MLX acceleration
)
```

MLX provides significant speedups for both flat and HNSW search operations. See the [MLX Acceleration Guide](mlx-acceleration.md) for details.

## Performance Optimization

### Index Building

For large collections, optimize index building:

```python
# For faster (but less optimal) HNSW index building
fast_index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    ef_construction=100,  # Lower value for faster building
    M=8                   # Lower for less memory usage
)

# For optimal index quality (slower build)
quality_index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    ef_construction=400,  # Higher for better quality
    M=32                  # Higher for better connectivity
)
```

### Search Speed vs. Accuracy

Control the accuracy-speed tradeoff at search time:

```python
# For maximum accuracy (slower)
results_accurate = index.search(
    query_vector=query_vector,
    k=10,
    ef_search=200  # Higher value = more accurate but slower
)

# For maximum speed (less accurate)
results_fast = index.search(
    query_vector=query_vector,
    k=10,
    ef_search=20  # Lower value = faster but less accurate
)
```

### Memory Optimization

For very large indices, consider quantization:

```python
# Create an index with vector quantization
index = VectorIndex(
    dimension=1536,
    quantize=True,
    quantization_bits=8  # Use 8-bit quantization (4x memory reduction)
)
```

## Benchmarking

To evaluate search performance in your specific environment:

```python
import time
import numpy as np
from llamadb.core import VectorIndex

# Create test data
dim = 128
num_vectors = 100_000
vectors = np.random.random((num_vectors, dim)).astype(np.float32)
metadata_list = [{"id": i} for i in range(num_vectors)]

# Test different configurations
configurations = [
    {"name": "Flat", "index_type": "flat", "use_mlx": False},
    {"name": "Flat+MLX", "index_type": "flat", "use_mlx": True},
    {"name": "HNSW", "index_type": "hnsw", "use_mlx": False},
    {"name": "HNSW+MLX", "index_type": "hnsw", "use_mlx": True}
]

for config in configurations:
    # Skip MLX tests if not available
    if config["use_mlx"] and not is_mlx_available():
        continue
        
    # Create and build index
    index = VectorIndex(
        dimension=dim,
        index_type=config["index_type"],
        use_mlx=config["use_mlx"]
    )
    
    # Add vectors
    start = time.time()
    index.add_items(vectors, metadata_list)
    build_time = time.time() - start
    
    # Measure search time
    query = np.random.random(dim).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        index.search(query, k=10)
    
    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        index.search(query, k=10)
    search_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"{config['name']}:")
    print(f"  Build time: {build_time:.2f} seconds")
    print(f"  Search time: {search_time:.2f} ms per query")
```

## Troubleshooting

### Poor Search Quality

If search results don't match expectations:

1. **Check the distance metric** - ensure it's appropriate for your embeddings
2. **Verify normalization** - for cosine similarity, ensure vectors are normalized
3. **Inspect HNSW parameters** - try increasing `ef_construction` and `M` for better index quality
4. **Adjust search parameters** - increase `ef_search` for more accurate results
5. **Check embedding quality** - verify that your embedding model produces good representations

### Slow Search

If search is too slow:

1. **Switch to HNSW** - if using flat search with large collections
2. **Reduce `ef_search`** - lower values give faster but less accurate results
3. **Enable MLX** - if on Apple Silicon, enable MLX acceleration
4. **Optimize filtering** - complex filters can slow down search
5. **Use batch operations** - for multiple queries, use `search_batch`

## Next Steps

- [Filtering Guide](filtering.md) - Learn how to use metadata filters effectively
- [Vector Indexes Guide](vector-indexes.md) - Explore different index types and configurations
- [MLX Acceleration Guide](mlx-acceleration.md) - Optimize for Apple Silicon devices
- [Embeddings Guide](embeddings.md) - Learn how to generate and work with embeddings 