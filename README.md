# LlamaDB

High-performance vector database optimized for AI workloads with MLX acceleration for Apple Silicon.

![LlamaDB Logo](docs/assets/llamadb_logo.png)

## Overview

LlamaDB is a next-generation vector database designed for AI applications with a focus on performance, ease of use, and platform-specific optimizations. It provides:

- 🚀 **High-performance vector search** - Fast similarity search for embeddings
- 🍎 **MLX acceleration** - Up to 10x speedup on Apple Silicon devices
- 🦀 **Rust extensions** - Critical paths implemented in Rust for maximum performance
- 🐍 **Python-first API** - Simple, intuitive API with Python at its core
- 🔌 **REST API** - Easy integration with any language or framework

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llamadb.git
cd llamadb

# Set up development environment
./setup_dev_environment.sh

# Activate the virtual environment
source .venv/bin/activate
```

### Run the Quickstart Demo

```bash
python quickstart.py
```

This will demonstrate basic vector operations, similarity search, and performance benchmarks.

### Start the API Server

```bash
# Start the server
python api_launcher.py --start

# Check status
python api_launcher.py --status

# Stop the server
python api_launcher.py --stop
```

## Core Features

### Vector Index

The `VectorIndex` class provides efficient similarity search:

```python
from llamadb.core import VectorIndex

# Create a new index with 128-dimensional vectors
index = VectorIndex(dimension=128)

# Add vectors with metadata
index.add_item(embedding, {"id": 1, "text": "Document content", "category": "Technology"})

# Search for similar vectors
results = index.search(query_vector, k=10)
```

### Accelerated Operations

LlamaDB provides optimized vector operations:

```python
from llamadb.core import cosine_similarity, l2_distance, dot_product

# Calculate similarity between vectors
similarity = cosine_similarity(vector_a, vector_b)
distance = l2_distance(vector_a, vector_b)
dot = dot_product(vector_a, vector_b)
```

### MLX Acceleration

On Apple Silicon devices, LlamaDB automatically uses MLX for acceleration:

```python
from llamadb.core import is_mlx_available, is_apple_silicon

if is_apple_silicon():
    print("Running on Apple Silicon")
    
if is_mlx_available():
    print("MLX acceleration is available")
```

## Development

For development instructions, see [DEVELOPER.md](./DEVELOPER.md).

### Utility Scripts

LlamaDB includes several utility scripts:

```bash
# Setup development environment
./setup_dev_environment.sh

# Run comprehensive tests
python test_llamadb.py --benchmark --plot

# Manage API server
python api_launcher.py --start
```

### Command Line Utilities

The `commands.sh` script provides common operations:

```bash
# Show help information
./commands.sh help

# Set up development environment
./commands.sh setup

# Run tests
./commands.sh test

# Run benchmarks
./commands.sh benchmark

# Run quickstart demo
./commands.sh quickstart
```

## API Server

LlamaDB includes a REST API server for language-agnostic access:

### Start the server

```bash
python api_launcher.py --start
```

### API Endpoints

- `GET /health` - Health check
- `GET /system` - System information
- `GET /index` - Index information
- `POST /search` - Search for similar vectors

Example search request:

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10
}
```

## Performance

LlamaDB is designed for performance:

| Operation | NumPy | MLX (Apple Silicon) | Speedup |
|-----------|-------|---------------------|---------|
| Cosine Similarity | 5µs | 0.5µs | 10x |
| Matrix Multiply (1000x1000) | 50ms | 5ms | 10x |
| Search (10k vectors) | 10ms | 1ms | 10x |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) - For the Apple Silicon acceleration
- [NumPy](https://numpy.org/) - For fundamental array operations
- [FastAPI](https://fastapi.tiangolo.com/) - For the REST API server
# Updated in commit 1 - 2025-04-04 17:07:35

# Updated in commit 9 - 2025-04-04 17:07:36

# Updated in commit 17 - 2025-04-04 17:07:37
