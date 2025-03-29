# LlamaDB Documentation

![LlamaDB Logo](assets/llamadb_logo.png){ align=center }

## Overview

LlamaDB is a next-generation vector database designed for AI applications with a focus on performance, ease of use, and platform-specific optimizations, particularly for Apple Silicon devices through MLX acceleration.

Key features include:

- 🚀 **High-performance vector search** - Fast similarity search for embeddings
- 🍎 **MLX acceleration** - Up to 10x speedup on Apple Silicon devices
- 🦀 **Rust extensions** - Critical paths implemented in Rust for maximum performance
- 🐍 **Python-first API** - Simple, intuitive API with Python at its core
- 🔌 **REST API** - Easy integration with any language or framework

## Quick Start

### Installation

```bash
# Install from PyPI
pip install llamadb

# For development features
pip install llamadb[dev]

# For MLX acceleration on Apple Silicon
pip install llamadb[mlx]

# For all features
pip install llamadb[full]
```

### Basic Usage

```python
from llamadb.core import VectorIndex

# Create a new index with 128-dimensional vectors
index = VectorIndex(dimension=128)

# Add vectors with metadata
index.add_item(
    embedding=[0.1, 0.2, ...],  # 128-dim vector
    metadata={"id": 1, "text": "Document content", "category": "Technology"}
)

# Search for similar vectors
results = index.search(
    query_vector=[0.15, 0.25, ...],  # 128-dim vector
    k=10
)

# Print results
for result in results:
    print(f"Score: {result.score}, Metadata: {result.metadata}")
```

### Starting the API Server

```bash
# Start the server
llamadb server start

# Check status
llamadb server status

# Stop the server
llamadb server stop
```

## Why LlamaDB?

### Performance

LlamaDB is designed for performance, with critical paths implemented in Rust and optional MLX acceleration for Apple Silicon devices.

| Operation | NumPy | MLX (Apple Silicon) | Speedup |
|-----------|-------|---------------------|---------|
| Cosine Similarity | 5µs | 0.5µs | 10x |
| Matrix Multiply (1000x1000) | 50ms | 5ms | 10x |
| Search (10k vectors) | 10ms | 1ms | 10x |

### Simplicity

LlamaDB provides a simple, intuitive API that makes it easy to get started with vector search:

```python
# Simple API for common operations
from llamadb.core import cosine_similarity, l2_distance, dot_product

# Calculate similarity between vectors
similarity = cosine_similarity(vector_a, vector_b)
distance = l2_distance(vector_a, vector_b)
dot = dot_product(vector_a, vector_b)
```

### Flexibility

LlamaDB supports a wide range of use cases, from simple vector search to complex filtering and aggregation:

```python
# Search with metadata filtering
results = index.search(
    query_vector=query_embedding,
    k=10,
    filter={"category": "Technology", "date": {"$gt": "2023-01-01"}}
)
```

## Next Steps

- [Installation](installation.md) - Detailed installation instructions
- [Getting Started](user-guide/getting-started.md) - Learn the basics
- [Core Concepts](user-guide/core-concepts.md) - Understand key concepts
- [Examples](examples/basic-usage.md) - See LlamaDB in action
- [API Reference](api/core.md) - Detailed API documentation 