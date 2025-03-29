# LlamaDB

High-performance vector database optimized for AI workloads with MLX acceleration for Apple Silicon.

![LlamaDB Logo](docs/assets/llamadb_logo.png)

[![PyPI version](https://badge.fury.io/py/llamadb.svg)](https://badge.fury.io/py/llamadb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/llamadb)](https://pypi.org/project/llamadb/)
[![Build Status](https://github.com/llamasearch-ai/llamadb/actions/workflows/ci.yml/badge.svg)](https://github.com/llamasearch-ai/llamadb/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/llamasearch-ai/llamadb/tree/main/docs)

## Overview

LlamaDB is a next-generation vector database designed for AI applications with a focus on performance, ease of use, and platform-specific optimizations. It provides:

- 🚀 **High-performance vector search** - Fast similarity search for embeddings
- 🍎 **MLX acceleration** - Up to 10x speedup on Apple Silicon devices
- 🦀 **Rust extensions** - Critical paths implemented in Rust for maximum performance
- 🐍 **Python-first API** - Simple, intuitive API with Python at its core
- 🔌 **REST API** - Easy integration with any language or framework

## Installation

```bash
# Basic installation
pip install llamadb

# With MLX acceleration for Apple Silicon
pip install llamadb[mlx]

# Full installation with all optional dependencies
pip install llamadb[full]
```

For detailed installation instructions, see the [Installation Guide](docs/installation.md).

## Quick Start

```python
from llamadb.core import VectorIndex
import numpy as np

# Create a new index with 128-dimensional vectors
index = VectorIndex(dimension=128)

# Create random vectors for demonstration
vectors = np.random.random((100, 128)).astype(np.float32)
metadata_list = [
    {"id": i, "text": f"Document {i}", "category": "Technology"}
    for i in range(100)
]

# Add vectors with metadata
index.add_items(vectors, metadata_list)

# Search for similar vectors
query_vector = np.random.random(128).astype(np.float32)
results = index.search(query_vector, k=10)

# Print results
for result in results:
    print(f"ID: {result.id}, Score: {result.score}, Metadata: {result.metadata}")
```

## REST API

LlamaDB includes a REST API server for language-agnostic access:

```bash
# Start the server
llamadb server start

# Search using HTTP requests
curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"vector": [0.1, 0.2, ...], "k": 10, "filter": {"category": "Technology"}}'
```

## Performance

LlamaDB is designed for performance, with critical paths implemented in Rust and optional MLX acceleration for Apple Silicon devices.

| Operation | NumPy | MLX (Apple Silicon) | Speedup |
|-----------|-------|---------------------|---------|
| Cosine Similarity | 5µs | 0.5µs | 10x |
| Matrix Multiply (1000x1000) | 50ms | 5ms | 10x |
| Search (10k vectors) | 10ms | 1ms | 10x |

## Documentation

- [Installation Guide](docs/installation.md)
- [Getting Started](docs/user-guide/getting-started.md)
- [User Guide](docs/user-guide/core-concepts.md)
- [API Reference](docs/api/core.md)
- [Examples](docs/examples/basic-usage.md)
- [Development Guide](docs/development/contributing.md)
- [Roadmap](docs/roadmap.md)

## Development

For development setup:

```bash
# Clone the repository
git clone https://github.com/llamasearch-ai/llamadb.git
cd llamadb

# Set up development environment
./setup_dev_environment.sh

# Run tests
python -m pytest
```

For more information, see our [Contributing Guide](docs/development/contributing.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) - For the Apple Silicon acceleration
- [NumPy](https://numpy.org/) - For fundamental array operations
- [FastAPI](https://fastapi.tiangolo.com/) - For the REST API server 