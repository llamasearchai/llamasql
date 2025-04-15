# Core Concepts

This guide explains the fundamental concepts and components of LlamaDB. Understanding these concepts will help you build effective applications with LlamaDB.

## Vector Databases

### What are Vector Databases?

Vector databases are specialized database systems designed to store, manage, and query vector embeddings - numerical representations of data in multi-dimensional space. They're optimized for similarity search operations, which are essential for many AI applications.

### Why Use a Vector Database?

Vector databases address several challenges in AI applications:

1. **Efficient Similarity Search**: Finding similar items in large datasets quickly
2. **Scalability**: Managing millions or billions of vectors
3. **Hybrid Filtering**: Combining vector search with metadata filtering
4. **Persistence**: Storing embeddings reliably for later retrieval
5. **Performance**: Optimizing for vector-specific operations

## Architecture Overview

LlamaDB consists of several key components:

### Vector Index

The `VectorIndex` is the core component of LlamaDB. It:

- Stores vector embeddings and associated metadata
- Provides efficient similarity search capabilities
- Supports filtering based on metadata
- Handles serialization and deserialization for persistence

### Distance Metrics

LlamaDB supports multiple distance metrics for similarity computation:

1. **Cosine Similarity**: Measures the cosine of the angle between vectors (range: -1 to 1, higher is more similar)
2. **L2 Distance (Euclidean)**: Measures the straight-line distance between vectors (lower is more similar)
3. **Dot Product**: Computes the dot product between vectors (higher is more similar)

The choice of distance metric depends on your specific use case and how your embeddings were generated.

### Metadata Management

Each vector in LlamaDB can have associated metadata, which:

- Provides additional context about the vector
- Enables filtering during search operations
- Supports complex data structures (nested objects, arrays)
- Is fully indexable for efficient querying

### Query System

LlamaDB's query system allows for:

- Pure vector similarity search
- Combined vector and metadata filtering
- Complex filtering expressions
- Pagination and result limits

### Acceleration

LlamaDB leverages hardware acceleration where available:

- **MLX**: Apple Silicon-specific acceleration for significant performance gains
- **NumPy**: Efficient array operations on all platforms
- **Rust Extensions**: Performance-critical paths implemented in Rust

## Data Flow

The typical data flow in a LlamaDB application follows this pattern:

1. **Data Preparation**: Convert raw data into vector embeddings (using external models)
2. **Indexing**: Add embeddings and metadata to a vector index
3. **Query Construction**: Create queries combining vector similarity and metadata filters
4. **Search**: Execute queries against the index to find similar items
5. **Result Processing**: Process the search results in your application logic

## Indexing Strategies

### In-Memory Index

The default index in LlamaDB is an in-memory index, which:

- Provides the fastest possible performance
- Is limited by available RAM
- Can be persisted to disk when needed
- Is ideal for datasets up to millions of vectors

### Hybrid HNSW Index

For larger datasets, LlamaDB uses a Hierarchical Navigable Small World (HNSW) index:

- Offers logarithmic search complexity
- Maintains high accuracy with configurable parameters
- Trades some build time for query performance
- Supports efficient updates and insertions

### Indexing Parameters

Key parameters for tuning index performance:

- **M**: Controls the number of connections per node (higher = more accuracy, more memory)
- **ef_construction**: Controls index build quality (higher = better quality, slower build)
- **ef_search**: Controls search quality (higher = better quality, slower search)

## Search Results

Search results in LlamaDB contain:

- **ID**: The unique identifier of the found item
- **Score**: The similarity score (interpretation depends on the distance metric)
- **Metadata**: The associated metadata of the item
- **Vector** (optional): The vector itself, if requested

## Filtering System

LlamaDB's filtering system supports:

- **Exact Match**: `{"category": "Technology"}`
- **Range Queries**: `{"date": {"$gt": "2023-01-01"}}`
- **Contains**: `{"tags": {"$contains": "sample"}}`
- **Logical Operators**: `{"$and": [{"category": "Technology"}, {"price": {"$lt": 100}}]}`
- **Nested Fields**: `{"user.preferences.theme": "dark"}`

## Persistence

LlamaDB provides methods to:

- Save indexes to disk: `index.save("my_index.llamadb")`
- Load indexes from disk: `VectorIndex.load("my_index.llamadb")`
- Auto-save on a schedule (for durability)
- Export/import in standardized formats

## Next Steps

Now that you understand the core concepts of LlamaDB, you can explore more specific topics:

- [Vector Indexes](vector-indexes.md) - Learn more about index types and configuration
- [MLX Acceleration](mlx-acceleration.md) - Understand how to leverage Apple Silicon
- [Filtering](filtering.md) - Master the filtering capabilities
- [REST API](rest-api.md) - Use LlamaDB in any language or framework 