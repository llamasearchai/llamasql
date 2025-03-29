# MLX Acceleration

This guide explains how LlamaDB leverages Apple's MLX framework to achieve significant performance improvements on Apple Silicon devices (M1, M2, M3 series).

## What is MLX?

[MLX](https://github.com/ml-explore/mlx) is an array framework designed for efficient machine learning on Apple Silicon. It offers:

- GPU-accelerated array operations
- Memory-efficient computation
- Automatic differentiation
- A Python API similar to NumPy

LlamaDB integrates MLX for vector operations, resulting in up to 10x speedup for critical operations.

## Automatic Detection and Activation

LlamaDB automatically detects Apple Silicon devices and will use MLX acceleration by default if available:

```python
from llamadb.core import is_apple_silicon, is_mlx_available, VectorIndex

# Check if running on Apple Silicon
if is_apple_silicon():
    print("Running on Apple Silicon")
    
# Check if MLX is available
if is_mlx_available():
    print("MLX acceleration is available")
    
# Create an index with MLX acceleration (enabled by default on Apple Silicon)
index = VectorIndex(dimension=128)
```

You can explicitly control whether to use MLX:

```python
# Force enable/disable MLX
index = VectorIndex(dimension=128, use_mlx=True)  # Enable
index = VectorIndex(dimension=128, use_mlx=False)  # Disable
```

## Performance Benefits

MLX acceleration provides significant performance improvements for vector operations:

| Operation | NumPy | MLX | Speedup |
|-----------|-------|-----|---------|
| Cosine Similarity (1K vectors) | 5µs | 0.5µs | 10x |
| L2 Distance (1K vectors) | 3µs | 0.4µs | 7.5x |
| Matrix Multiply (1K×1K) | 50ms | 5ms | 10x |
| KNN Search (10K vectors) | 10ms | 1ms | 10x |
| Batch Index (100K vectors) | 5s | 0.8s | 6.25x |

These improvements are particularly noticeable for larger datasets and real-time applications.

## Installation Requirements

To use MLX acceleration:

1. You need an Apple Silicon device (M1, M2, or M3 series)
2. Install the MLX package:

```bash
pip install mlx
# or
pip install llamadb[mlx]
```

## Using MLX-Accelerated Operations

### Basic Vector Operations

MLX acceleration works transparently for all vector operations:

```python
from llamadb.core import cosine_similarity, l2_distance, dot_product
import numpy as np

# Create vectors
vector_a = np.random.random(128).astype(np.float32)
vector_b = np.random.random(128).astype(np.float32)

# Calculate similarity (MLX used automatically if available)
similarity = cosine_similarity(vector_a, vector_b)
distance = l2_distance(vector_a, vector_b)
dot = dot_product(vector_a, vector_b)
```

### Vector Search

Search operations also benefit from MLX:

```python
from llamadb.core import VectorIndex
import numpy as np

# Create an index
index = VectorIndex(dimension=128)

# Add vectors
for i in range(10000):
    vector = np.random.random(128).astype(np.float32)
    index.add_item(vector, {"id": i})
    
# Search (MLX acceleration used automatically)
query = np.random.random(128).astype(np.float32)
results = index.search(query, k=10)
```

## Advanced Configuration

You can fine-tune MLX usage with advanced configuration:

```python
from llamadb.core import VectorIndex

# Configure MLX memory allocation
index = VectorIndex(
    dimension=128,
    use_mlx=True,
    mlx_memory_fraction=0.5,  # Use up to 50% of GPU memory
    mlx_prefer_precision="float16"  # Use half-precision for better performance
)
```

## MLX Benchmarking

LlamaDB includes tools to benchmark MLX performance on your system:

```python
from llamadb.utils.benchmarks import run_mlx_benchmark

# Run a comprehensive benchmark
results = run_mlx_benchmark(
    dimensions=[128, 768, 1536],
    dataset_sizes=[10_000, 100_000, 1_000_000],
    compare_with_numpy=True
)

# Print results
for op, perf in results.items():
    print(f"{op}: MLX: {perf['mlx_time']:.2f}ms, NumPy: {perf['numpy_time']:.2f}ms, Speedup: {perf['speedup']:.2f}x")
```

## Troubleshooting MLX Acceleration

### Verifying MLX is in Use

You can verify if MLX is actually being used:

```python
from llamadb.core import VectorIndex
import llamadb.utils.profiling as profiling

# Enable operation logging
profiling.set_log_level("INFO")

# Create an index and perform operations
index = VectorIndex(dimension=128)
# ... add vectors and search ...

# Check the logged operations
acceleration_stats = profiling.get_acceleration_stats()
print(f"MLX operations: {acceleration_stats['mlx_ops']}")
print(f"NumPy operations: {acceleration_stats['numpy_ops']}")
```

### Common Issues

If MLX acceleration isn't working:

1. **MLX not installed**: Ensure you've installed MLX with `pip install mlx`
2. **Non-Apple Silicon device**: MLX only works on Apple Silicon (M1/M2/M3)
3. **Incompatible data types**: Ensure vectors are float32 (MLX works best with float32)
4. **Memory issues**: If operations are failing, try reducing `mlx_memory_fraction`

## MLX and Batch Processing

MLX provides significant benefits for batch operations:

```python
import numpy as np
from llamadb.core import VectorIndex

# Create a large batch of vectors
vectors = np.random.random((100_000, 128)).astype(np.float32)
metadata_list = [{"id": i} for i in range(100_000)]

# Measure time for batch insertion
import time
start = time.time()
index = VectorIndex(dimension=128, use_mlx=True)
index.add_items(vectors, metadata_list)
end = time.time()
print(f"Inserted 100,000 vectors in {end - start:.2f} seconds")

# Batch search is also accelerated
query_batch = np.random.random((100, 128)).astype(np.float32)
start = time.time()
results_batch = index.search_batch(query_batch, k=10)
end = time.time()
print(f"Searched 100 queries in {end - start:.2f} seconds")
```

## Best Practices for MLX Acceleration

1. **Use float32 vectors**: MLX performs best with float32 data
2. **Process in batches**: Batch operations get the most benefit from MLX
3. **Optimize memory usage**: Set appropriate memory fraction for large datasets
4. **Consider precision trade-offs**: float16 can be much faster with minimal accuracy impact
5. **Monitor temperature**: Sustained heavy MLX usage can cause thermal throttling on laptops

## Next Steps

- [Filtering](filtering.md) - Learn about LlamaDB's powerful filtering capabilities
- [REST API](rest-api.md) - Use LlamaDB from any language or framework
- [API Reference](../api/core.md) - Explore the detailed API documentation 