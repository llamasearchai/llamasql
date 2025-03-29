# Working with Embeddings in LlamaDB

This guide explains how to create and work with embeddings in LlamaDB. Embeddings are the foundation of vector search, and understanding how to generate and use them effectively is crucial for building powerful applications.

## What are Embeddings?

Embeddings are dense vector representations of data objects (text, images, audio, etc.) that capture semantic meaning in a high-dimensional space. In this space, similar items are positioned close to each other, enabling efficient similarity search.

## Embedding Generation

LlamaDB does not generate embeddings directly - it focuses on storing and searching them. You'll need to generate embeddings using external libraries or services.

### Popular Embedding Sources

#### Text Embeddings

```python
# Using OpenAI's embedding models
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text to embed"
)
embedding = response.data[0].embedding  # This is a list of floats

# Convert to numpy array for LlamaDB
import numpy as np
embedding_array = np.array(embedding, dtype=np.float32)
```

```python
# Using HuggingFace Transformers
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize and generate embeddings
def get_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use CLS token embedding as sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding[0].astype(np.float32)  # Convert to float32 for LlamaDB

text_embedding = get_embedding("Your text to embed")
```

#### Image Embeddings

```python
# Using CLIP for image embeddings
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Generate image embedding
def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Get image embedding
    embedding = outputs.numpy()
    return embedding[0].astype(np.float32)  # Convert to float32 for LlamaDB

image_embedding = get_image_embedding("path/to/your/image.jpg")
```

## Working with Different Dimensionalities

Different embedding models produce vectors of different dimensions. When creating a LlamaDB index, you must specify the correct dimension:

```python
from llamadb.core import VectorIndex

# For OpenAI text-embedding-3-small (1536 dimensions)
openai_index = VectorIndex(dimension=1536)

# For all-MiniLM-L6-v2 (384 dimensions)
minilm_index = VectorIndex(dimension=384)

# For CLIP embeddings (512 dimensions)
clip_index = VectorIndex(dimension=512)
```

## Normalizing Embeddings

Some distance metrics (like cosine similarity) work better with normalized vectors. You can normalize your embeddings before adding them to LlamaDB:

```python
def normalize_vector(vector):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

# Normalize before adding to index
normalized_embedding = normalize_vector(embedding)
index.add_item(embedding=normalized_embedding, metadata={"id": 1, "text": "Original text"})
```

## Multi-Model Strategy

You can maintain multiple indices for different embedding models:

```python
# Create separate indices for different models
openai_index = VectorIndex(dimension=1536, metric="cosine")
minilm_index = VectorIndex(dimension=384, metric="cosine")

# Add the same item to both indices with different embeddings
document = "This is a sample document."

# Get embeddings from different models
openai_embedding = get_openai_embedding(document)
minilm_embedding = get_minilm_embedding(document)

# Add to respective indices with the same metadata
metadata = {"id": 1, "text": document, "source": "blog"}
openai_index.add_item(embedding=openai_embedding, metadata=metadata)
minilm_index.add_item(embedding=minilm_embedding, metadata=metadata)
```

## Batch Processing

For large datasets, batch processing is more efficient:

```python
import pandas as pd

# Load data
df = pd.read_csv("documents.csv")

# Generate embeddings in batches
batch_size = 32
embeddings = []
metadata_list = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    # Generate embeddings for batch (using your preferred method)
    batch_embeddings = generate_embeddings_for_texts(batch["text"].tolist())
    
    # Create metadata for each item
    for j, row in batch.iterrows():
        metadata_list.append({
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
            "category": row["category"]
        })
    
    embeddings.extend(batch_embeddings)

# Convert to numpy array
embeddings_array = np.array(embeddings, dtype=np.float32)

# Add to index in one batch operation
index.add_items(embeddings=embeddings_array, metadata_list=metadata_list)
```

## Chunking Strategies

When working with long documents, it's often beneficial to chunk them into smaller pieces before embedding:

```python
def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

# Process a document
document = "Very long document text..."
chunks = chunk_text(document)

# Generate embeddings for each chunk
chunk_embeddings = []
metadata_list = []

for i, chunk in enumerate(chunks):
    embedding = generate_embedding(chunk)  # Using your preferred method
    chunk_embeddings.append(embedding)
    
    metadata_list.append({
        "id": f"doc1_chunk{i}",
        "document_id": "doc1",
        "chunk_index": i,
        "text": chunk,
        "is_chunk": True
    })

# Add chunks to index
index.add_items(embeddings=np.array(chunk_embeddings), metadata_list=metadata_list)
```

## Hybrid Search

For optimal search results, you can combine vector search with traditional keyword search:

```python
def hybrid_search(query, vector_index, keyword_index, alpha=0.5):
    """Combine vector search and keyword search with a weighted approach"""
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    # Vector search
    vector_results = vector_index.search(query_embedding, k=100)
    
    # Keyword search (implementation depends on your keyword search engine)
    keyword_results = keyword_index.search(query, k=100)
    
    # Combine results
    result_map = {}
    
    # Process vector results
    for result in vector_results:
        result_map[result.id] = {
            "score": alpha * result.score,
            "metadata": result.metadata
        }
    
    # Process keyword results
    for result in keyword_results:
        if result.id in result_map:
            result_map[result.id]["score"] += (1 - alpha) * result.score
        else:
            result_map[result.id] = {
                "score": (1 - alpha) * result.score,
                "metadata": result.metadata
            }
    
    # Sort by combined score
    combined_results = sorted(
        result_map.items(), 
        key=lambda x: x[1]["score"], 
        reverse=True
    )
    
    return combined_results[:10]  # Return top 10
```

## Performance Considerations

### Memory Usage

Embedding dimensionality affects memory usage:

| Dimensions | Items | Approx. Memory Usage |
|------------|-------|---------------------|
| 384        | 10,000 | ~15 MB              |
| 768        | 10,000 | ~30 MB              |
| 1536       | 10,000 | ~60 MB              |

### Storage Optimization

For large datasets, consider using quantization to reduce memory usage:

```python
# Create an index with quantization
from llamadb.core import VectorIndex

index = VectorIndex(
    dimension=1536,
    quantize=True,  # Enable quantization
    quantization_bits=8  # Use 8-bit quantization (reduces memory by 4x compared to float32)
)
```

## Best Practices

1. **Choose the right embedding model** for your use case - different models have different strengths.
2. **Match metric to model** - some models work better with cosine similarity, others with dot product.
3. **Normalize vectors** when using cosine similarity.
4. **Use chunking** for long documents to improve retrieval quality.
5. **Batch operations** for better performance when working with large datasets.
6. **Store source data** in metadata for context retrieval.
7. **Consider hybrid search** for complex queries.

## Troubleshooting

### Embedding Quality Issues

If search results don't match expectations:

1. Check if you're using the right distance metric for your embeddings
2. Ensure your vectors are normalized if using cosine similarity
3. Try different embedding models to see which works best for your data
4. Experiment with chunking strategies for long documents

### Performance Issues

If search is slow:

1. Use HNSW index type instead of flat for larger datasets
2. Reduce the `ef_search` parameter for faster (but potentially less accurate) search
3. Enable MLX acceleration on Apple Silicon devices
4. Consider quantization for large collections

## Next Steps

- [Similarity Search Guide](similarity-search.md) - Learn advanced similarity search techniques
- [Vector Indexes Guide](vector-indexes.md) - Understand different index types and configurations
- [MLX Acceleration Guide](mlx-acceleration.md) - Optimize for Apple Silicon devices 