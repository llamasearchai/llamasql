# Basic Usage Examples

This page provides examples of common operations with LlamaDB.

## Creating an Index

```python
from llamadb.core import VectorIndex

# Create a simple index with 128-dimensional vectors
index = VectorIndex(dimension=128)

# Create an index with custom settings
index = VectorIndex(
    dimension=128,
    index_type="hnsw",
    metric="cosine",
    use_mlx=True,
    ef_construction=200,
    ef_search=100,
    M=16
)
```

## Adding Vectors

### Adding a Single Vector

```python
import numpy as np

# Create a random vector
vector = np.random.random(128).astype(np.float32)

# Add the vector with metadata
index.add_item(
    embedding=vector,
    metadata={
        "id": 1,
        "title": "Example Document",
        "category": "Technology",
        "tags": ["example", "document", "technology"],
        "date": "2023-03-15",
        "rating": 4.5,
        "is_active": True
    }
)
```

### Adding Multiple Vectors

```python
# Create 100 random vectors
vectors = np.random.random((100, 128)).astype(np.float32)

# Create metadata for each vector
metadata_list = []
for i in range(100):
    metadata_list.append({
        "id": i + 1,
        "title": f"Document {i + 1}",
        "category": "Technology" if i % 3 == 0 else "Science" if i % 3 == 1 else "Business",
        "tags": ["document", f"tag{i % 10}"],
        "date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        "rating": (i % 5) + 1,
        "is_active": i % 2 == 0
    })

# Add all vectors at once
index.add_items(embeddings=vectors, metadata_list=metadata_list)
```

## Searching

### Basic Search

```python
# Create a query vector
query_vector = np.random.random(128).astype(np.float32)

# Search for the 10 most similar vectors
results = index.search(query_vector, k=10)

# Print results
for result in results:
    print(f"ID: {result.id}, Score: {result.score}, Title: {result.metadata['title']}")
```

### Search with Filtering

```python
# Search with a simple filter
results = index.search(
    query_vector=query_vector,
    k=10,
    filter={"category": "Technology"}
)

# Search with a complex filter
results = index.search(
    query_vector=query_vector,
    k=10,
    filter={
        "$and": [
            {"category": "Technology"},
            {"rating": {"$gte": 4}},
            {"tags": {"$contains": "document"}}
        ]
    }
)
```

### Including Vectors in Results

```python
# Search and include the actual vectors
results = index.search(
    query_vector=query_vector,
    k=10,
    include_vectors=True
)

# Print results with vectors
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Vector shape: {result.vector.shape}, Type: {result.vector.dtype}")
```

## Updating Vectors

```python
# Update a vector by ID
new_vector = np.random.random(128).astype(np.float32)
index.update_item(
    id=42,
    embedding=new_vector,
    metadata={
        "id": 42,
        "title": "Updated Document",
        "category": "Technology",
        "tags": ["updated", "document", "technology"],
        "date": "2023-03-16",
        "rating": 5.0,
        "is_active": True
    }
)
```

## Deleting Vectors

```python
# Delete a single vector by ID
index.delete_item(id=42)

# Delete multiple vectors by ID
index.delete_items(ids=[1, 2, 3, 4, 5])

# Delete vectors matching a filter
index.delete_by_filter(filter={"category": "Outdated"})
```

## Saving and Loading

```python
# Save the index to disk
index.save("my_index.llamadb")

# Load the index from disk
loaded_index = VectorIndex.load("my_index.llamadb")
```

## Querying by ID

```python
# Get a vector by ID
item = index.get_item(id=42)
if item:
    print(f"ID: {item.id}")
    print(f"Metadata: {item.metadata}")
    print(f"Vector: {item.vector}")
else:
    print("Item not found")
```

## Counting Items

```python
# Count all items
total_count = index.count()
print(f"Total items: {total_count}")

# Count items matching a filter
tech_count = index.count(filter={"category": "Technology"})
print(f"Technology items: {tech_count}")
```

## Batch Operations

### Batch Search

```python
# Create multiple query vectors
query_vectors = np.random.random((5, 128)).astype(np.float32)

# Search for each query vector
results_batch = index.search_batch(query_vectors, k=10)

# Print results for each query
for i, results in enumerate(results_batch):
    print(f"Results for query {i + 1}:")
    for result in results:
        print(f"  ID: {result.id}, Score: {result.score}")
```

### Batch Update

```python
# Create new vectors and metadata
new_vectors = np.random.random((3, 128)).astype(np.float32)
ids = [10, 20, 30]
new_metadata_list = [
    {"id": 10, "title": "Updated 10", "category": "Technology"},
    {"id": 20, "title": "Updated 20", "category": "Science"},
    {"id": 30, "title": "Updated 30", "category": "Business"}
]

# Update multiple items
index.update_items(ids=ids, embeddings=new_vectors, metadata_list=new_metadata_list)
```

## Using Distance Metrics

```python
# Create indexes with different distance metrics
cosine_index = VectorIndex(dimension=128, metric="cosine")
l2_index = VectorIndex(dimension=128, metric="l2")
dot_index = VectorIndex(dimension=128, metric="dot")

# Add the same vectors to each index
vectors = np.random.random((100, 128)).astype(np.float32)
metadata_list = [{"id": i} for i in range(100)]

cosine_index.add_items(vectors, metadata_list)
l2_index.add_items(vectors, metadata_list)
dot_index.add_items(vectors, metadata_list)

# Search with different metrics
query = np.random.random(128).astype(np.float32)

cosine_results = cosine_index.search(query, k=5)
l2_results = l2_index.search(query, k=5)
dot_results = dot_index.search(query, k=5)

print("Cosine similarity results:")
for r in cosine_results:
    print(f"  ID: {r.id}, Score: {r.score}")

print("L2 distance results:")
for r in l2_results:
    print(f"  ID: {r.id}, Score: {r.score}")

print("Dot product results:")
for r in dot_results:
    print(f"  ID: {r.id}, Score: {r.score}")
```

## Using MLX Acceleration

```python
from llamadb.core import is_mlx_available, is_apple_silicon

# Check if MLX acceleration is available
if is_apple_silicon():
    print("Running on Apple Silicon")
    
if is_mlx_available():
    print("MLX acceleration is available")
    
    # Create an index with MLX acceleration
    index = VectorIndex(dimension=128, use_mlx=True)
    
    # Add and search vectors (MLX will be used automatically)
    # ...
```

## Performance Optimization

```python
import time

# Create a large dataset
num_vectors = 100_000
dim = 128
vectors = np.random.random((num_vectors, dim)).astype(np.float32)
metadata_list = [{"id": i} for i in range(num_vectors)]

# Measure performance with different configurations
for index_type in ["flat", "hnsw"]:
    for use_mlx in [False, True]:
        if use_mlx and not is_mlx_available():
            continue
            
        print(f"Testing {index_type} index with MLX={use_mlx}")
        
        # Create index
        index = VectorIndex(
            dimension=dim,
            index_type=index_type,
            use_mlx=use_mlx
        )
        
        # Measure insertion time
        start = time.time()
        index.add_items(vectors, metadata_list)
        insert_time = time.time() - start
        print(f"  Insertion time: {insert_time:.2f} seconds")
        
        # Measure search time
        query = np.random.random(dim).astype(np.float32)
        start = time.time()
        for _ in range(100):
            index.search(query, k=10)
        search_time = time.time() - start
        print(f"  Avg search time: {search_time / 100 * 1000:.2f} ms")
```

## Next Steps

Now that you've seen the basic usage of LlamaDB, you can explore more advanced topics:

- [Similarity Search](similarity-search.md) - Advanced similarity search techniques
- [Metadata Filtering](metadata-filtering.md) - Complex filtering operations
- [MLX Acceleration](mlx-acceleration.md) - Getting the most from Apple Silicon
- [API Server](api-server.md) - Using the REST API 