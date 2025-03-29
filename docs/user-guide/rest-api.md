# REST API

LlamaDB provides a REST API that allows you to interact with the database from any language or framework. This guide explains how to use the API server and the available endpoints.

## Starting the API Server

### Command Line

The simplest way to start the API server is using the command line:

```bash
# Start the server with default settings
llamadb server start

# Start the server with custom settings
llamadb server start --host 0.0.0.0 --port 8080 --index-path my_index.llamadb
```

### Python API

You can also start the server programmatically:

```python
from llamadb.server import APIServer
from llamadb.core import VectorIndex

# Create an index
index = VectorIndex(dimension=128)

# Add some vectors
# ... add your vectors here ...

# Start the server with the index
server = APIServer(index=index, host="127.0.0.1", port=8000)
server.start()

# The server runs in a background thread, so your application can continue
# ...

# Stop the server when done
server.stop()
```

### Server Management

The LlamaDB CLI provides commands to manage the server:

```bash
# Check server status
llamadb server status

# Stop the server
llamadb server stop

# Restart the server
llamadb server restart

# Start server in debug mode (more verbose logs)
llamadb server start --debug
```

## API Endpoints

### Health Check

Check if the API server is running:

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### System Information

Get information about the system:

```http
GET /system
```

Example response:

```json
{
  "version": "0.1.0",
  "platform": "darwin",
  "python_version": "3.11.2",
  "mlx_available": true,
  "apple_silicon": true,
  "cpu_count": 10,
  "memory_total_gb": 32.0,
  "started_at": "2023-03-15T12:34:56.789Z",
  "uptime_seconds": 3600
}
```

### Index Information

Get information about the vector index:

```http
GET /index
```

Example response:

```json
{
  "dimension": 128,
  "index_type": "hnsw",
  "metric": "cosine",
  "item_count": 10000,
  "metadata_fields": ["id", "category", "rating", "date", "tags", "is_active"],
  "created_at": "2023-03-15T10:00:00.000Z",
  "last_modified": "2023-03-15T11:30:00.000Z",
  "using_mlx": true
}
```

### Search

Search for similar vectors:

```http
POST /search
```

Request body:

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "filter": {"category": "Technology"},
  "include_vectors": false,
  "include_metadata": true
}
```

Example response:

```json
[
  {
    "id": 42,
    "score": 0.95,
    "metadata": {
      "id": 42,
      "category": "Technology",
      "rating": 4.5,
      "date": "2023-03-15",
      "tags": ["sample", "vector", "tech"],
      "is_active": true
    }
  },
  {
    "id": 17,
    "score": 0.92,
    "metadata": {
      "id": 17,
      "category": "Technology",
      "rating": 5.0,
      "date": "2023-02-10",
      "tags": ["sample", "vector", "tech"],
      "is_active": true
    }
  },
  ...
]
```

### Batch Search

Search for multiple query vectors in a single request:

```http
POST /search_batch
```

Request body:

```json
{
  "vectors": [
    [0.1, 0.2, 0.3, ...],
    [0.2, 0.3, 0.4, ...],
    [0.3, 0.4, 0.5, ...]
  ],
  "k": 5,
  "filter": {"category": "Technology"},
  "include_vectors": false,
  "include_metadata": true
}
```

Example response:

```json
[
  [
    {"id": 42, "score": 0.95, "metadata": {...}},
    {"id": 17, "score": 0.92, "metadata": {...}},
    ...
  ],
  [
    {"id": 28, "score": 0.97, "metadata": {...}},
    {"id": 35, "score": 0.94, "metadata": {...}},
    ...
  ],
  [
    {"id": 56, "score": 0.96, "metadata": {...}},
    {"id": 72, "score": 0.93, "metadata": {...}},
    ...
  ]
]
```

### Add Item

Add a single vector to the index:

```http
POST /add
```

Request body:

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "category": "Technology",
    "rating": 4.5,
    "date": "2023-03-15",
    "tags": ["sample", "vector", "tech"],
    "is_active": true
  }
}
```

Example response:

```json
{
  "id": 10001,
  "status": "added"
}
```

### Add Items (Batch)

Add multiple vectors to the index:

```http
POST /add_batch
```

Request body:

```json
{
  "vectors": [
    [0.1, 0.2, 0.3, ...],
    [0.2, 0.3, 0.4, ...],
    [0.3, 0.4, 0.5, ...]
  ],
  "metadata_list": [
    {"category": "Technology", "rating": 4.5},
    {"category": "Science", "rating": 3.5},
    {"category": "Business", "rating": 5.0}
  ]
}
```

Example response:

```json
{
  "ids": [10001, 10002, 10003],
  "status": "added",
  "count": 3
}
```

### Update Item

Update an existing vector:

```http
PUT /items/{id}
```

Request body:

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "category": "Technology",
    "rating": 5.0,
    "date": "2023-03-15",
    "tags": ["sample", "vector", "tech", "updated"],
    "is_active": true
  }
}
```

Example response:

```json
{
  "id": 42,
  "status": "updated"
}
```

### Delete Item

Delete a vector by ID:

```http
DELETE /items/{id}
```

Example response:

```json
{
  "id": 42,
  "status": "deleted"
}
```

### Delete by Filter

Delete vectors matching a filter:

```http
POST /delete_by_filter
```

Request body:

```json
{
  "filter": {"category": "Outdated"}
}
```

Example response:

```json
{
  "status": "deleted",
  "count": 15
}
```

### Get Item

Get a specific vector by ID:

```http
GET /items/{id}
```

Example response:

```json
{
  "id": 42,
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "id": 42,
    "category": "Technology",
    "rating": 4.5,
    "date": "2023-03-15",
    "tags": ["sample", "vector", "tech"],
    "is_active": true
  }
}
```

### Count Items

Count vectors matching a filter:

```http
POST /count
```

Request body:

```json
{
  "filter": {"category": "Technology"}
}
```

Example response:

```json
{
  "count": 333
}
```

### Save Index

Save the current index to disk:

```http
POST /save
```

Request body:

```json
{
  "path": "my_index.llamadb"
}
```

Example response:

```json
{
  "status": "saved",
  "path": "my_index.llamadb",
  "item_count": 10000
}
```

## Client Libraries

### Python Client

LlamaDB provides a Python client for the REST API:

```python
from llamadb.client import LlamaDBClient
import numpy as np

# Create a client
client = LlamaDBClient(url="http://localhost:8000")

# Check health
health = client.health()
print(f"Server status: {health['status']}")

# Get index info
index_info = client.get_index_info()
print(f"Index dimension: {index_info['dimension']}")

# Add a vector
vector = np.random.random(128).astype(np.float32)
metadata = {"category": "Technology", "rating": 4.5}
response = client.add_item(vector, metadata)
item_id = response["id"]

# Search
results = client.search(
    query_vector=vector,
    k=10,
    filter={"category": "Technology"}
)

# Print results
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

### Other Languages

You can interact with the API using any HTTP client in any language:

#### JavaScript/TypeScript

```javascript
// Using fetch API
async function searchVectors(queryVector, filter) {
  const response = await fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      vector: queryVector,
      k: 10,
      filter: filter,
    }),
  });
  
  return await response.json();
}

// Example usage
const results = await searchVectors(
  [0.1, 0.2, 0.3, /* ... */],
  { category: "Technology" }
);
```

#### Curl

```bash
# Search for similar vectors
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "filter": {"category": "Technology"}
  }'
```

## Authentication and Security

### Basic Authentication

The API server supports basic authentication:

```bash
# Start server with authentication
llamadb server start --auth-file auth.json
```

The auth.json file should contain user credentials:

```json
{
  "users": [
    {
      "username": "admin",
      "password": "password123"
    },
    {
      "username": "readonly",
      "password": "password456"
    }
  ]
}
```

### API Keys

You can also use API keys for authentication:

```bash
# Start server with API key authentication
llamadb server start --api-keys-file keys.json
```

The keys.json file should contain API keys and permissions:

```json
{
  "keys": [
    {
      "key": "sk_abcdef123456",
      "name": "Production Key",
      "permissions": ["read", "write"]
    },
    {
      "key": "sk_xyz789012345",
      "name": "Read-only Key",
      "permissions": ["read"]
    }
  ]
}
```

### HTTPS

For production, you should use HTTPS:

```bash
# Start server with HTTPS
llamadb server start --ssl-cert cert.pem --ssl-key key.pem
```

## Next Steps

- [REST API Reference](../api/rest.md) - Detailed API reference
- [Examples: API Server](../examples/api-server.md) - More API usage examples
- [API Client Reference](../api/client.md) - Python client API reference 