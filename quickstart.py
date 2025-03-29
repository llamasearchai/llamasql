#!/usr/bin/env python
"""
LlamaDB Quickstart Demo
This script demonstrates the basic functionality of LlamaDB
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any

# Ensure LlamaDB can be imported regardless of where this script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
python_dir = os.path.join(project_root, "python")
if os.path.exists(python_dir):
    sys.path.insert(0, python_dir)

try:
    from llamadb.core import (
        VectorIndex,
        cosine_similarity,
        l2_distance,
        dot_product,
        is_mlx_available,
        is_apple_silicon
    )
except ImportError:
    # Try alternative import paths
    print("Trying alternative import paths...")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from llamadb.core import (
            VectorIndex,
            cosine_similarity,
            l2_distance,
            dot_product,
            is_mlx_available,
            is_apple_silicon
        )
    except ImportError:
        print("Failed to import LlamaDB modules.")
        print("Make sure LlamaDB is properly installed or run this from the project root directory.")
        sys.exit(1)

def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def generate_sample_data(count: int = 1000, dim: int = 128) -> List[Dict[str, Any]]:
    """Generate sample document data with embeddings."""
    documents = []
    categories = ["Technology", "Science", "Health", "Business", "Entertainment"]
    
    for i in range(count):
        # Generate random embedding and normalize it
        embedding = np.random.rand(dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Generate sample document
        category = categories[i % len(categories)]
        doc = {
            "id": i,
            "title": f"Document {i}",
            "category": category,
            "importance": np.random.randint(1, 6),
            "created_at": time.strftime("%Y-%m-%d", time.localtime(time.time() - np.random.randint(0, 365*24*3600))),
            "embedding": embedding
        }
        documents.append(doc)
    
    return documents

def run_quickstart_demo() -> None:
    """Run the quickstart demo."""
    print_header("LlamaDB Quickstart Demo")
    
    # System information
    print("System Information:")
    print(f"- Running on Apple Silicon: {'Yes' if is_apple_silicon() else 'No'}")
    print(f"- MLX Acceleration: {'Available' if is_mlx_available() else 'Not available'}")
    print(f"- Python version: {sys.version.split()[0]}")
    print(f"- NumPy version: {np.__version__}")
    
    # Configuration
    dim = 128
    doc_count = 1000
    
    print(f"\nCreating a vector database with {doc_count} documents, each with {dim}-dimensional embeddings...")
    
    # Generate sample data
    start_time = time.time()
    documents = generate_sample_data(doc_count, dim)
    gen_time = time.time() - start_time
    print(f"Generated {len(documents)} sample documents in {gen_time:.2f} seconds")
    
    # Create vector index
    print("\nBuilding vector index...")
    start_time = time.time()
    index = VectorIndex(dim)
    
    # Track how many items we've added
    items_added = 0
    
    for doc in documents:
        # Extract embedding and metadata
        embedding = doc["embedding"]
        metadata = {k: v for k, v in doc.items() if k != "embedding"}
        index.add_item(embedding, metadata)
        items_added += 1
    
    build_time = time.time() - start_time
    print(f"Built vector index with {items_added} items in {build_time:.2f} seconds")
    
    # Run sample queries
    print_header("Sample Queries")
    
    # Generate a random query vector
    query_vector = np.random.rand(dim).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Search by vector similarity
    print("\n1. Basic vector similarity search:")
    start_time = time.time()
    results = index.search(query_vector, k=5)
    search_time = time.time() - start_time
    
    print(f"Found top 5 most similar documents in {search_time*1000:.2f} ms")
    for i, result in enumerate(results):
        print(f"  {i+1}. Document {result['id']} (Score: {result.get('score', 'N/A'):.4f})")
    
    # Filter results manually for category-specific results
    print("\n2. Category-filtered results (Technology only):")
    filter_category = "Technology"
    
    # First search for more results than we need
    start_time = time.time()
    all_results = index.search(query_vector, k=50)
    
    # Then filter the results manually
    filtered_results = [
        result for result in all_results 
        if result.get("category") == filter_category
    ][:5]  # Take the top 5 after filtering
    
    filtered_time = time.time() - start_time
    
    print(f"Found top 5 Technology documents in {filtered_time*1000:.2f} ms")
    for i, result in enumerate(filtered_results):
        print(f"  {i+1}. {result.get('title', 'Untitled')} - {result.get('category', 'Uncategorized')}")
    
    # Performance comparison
    print_header("Vector Operations Performance")
    
    # Use NumPy directly for vector operations benchmark to avoid any issues
    print("Benchmarking vector operations with NumPy...")
    
    # Create sample vectors
    v1 = np.random.rand(dim).astype(np.float32)
    v2 = np.random.rand(dim).astype(np.float32)
    
    # NumPy cosine similarity
    iterations = 1000
    start_time = time.time()
    for _ in range(iterations):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        cos_sim = dot / (norm1 * norm2)
    cos_time = (time.time() - start_time) / iterations
    
    # NumPy L2 distance
    start_time = time.time()
    for _ in range(iterations):
        l2 = np.linalg.norm(v1 - v2)
    l2_time = (time.time() - start_time) / iterations
    
    # NumPy dot product
    start_time = time.time()
    for _ in range(iterations):
        dot = np.dot(v1, v2)
    dot_time = (time.time() - start_time) / iterations
    
    print(f"Average time per operation over {iterations} iterations:")
    print(f"- Cosine similarity: {cos_time*1000000:.2f} µs")
    print(f"- L2 distance:       {l2_time*1000000:.2f} µs")
    print(f"- Dot product:       {dot_time*1000000:.2f} µs")
    
    print_header("Next Steps")
    print("""
1. Explore more advanced features:
   - Create custom distance functions
   - Try different indexing strategies
   - Implement batch operations for better performance

2. Integrate with your application:
   - Store your vectors and metadata
   - Implement similarity search
   - Build a semantic search engine

3. Optimize for your use case:
   - Tune parameters for your data
   - Implement caching for frequent queries
   - Use batched operations for bulk processing

For more information, check out the documentation and examples in the repository.
""")

if __name__ == "__main__":
    run_quickstart_demo() 