"""
Tests for the accelerated operations module.

These tests verify the functionality of the accelerated operations module,
including the VectorIndex and BatchProcessor classes.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.llamadb.core.accelerated_ops import (
    VectorIndex,
    BatchProcessor,
    demonstrate_acceleration,
)

# Test data
@pytest.fixture
def vector_data():
    """Create test vector data."""
    # Create vectors
    dimension = 128
    num_vectors = 100
    
    # Generate random vectors
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    
    # Create metadata
    metadata = [{"id": i, "text": f"Document {i}"} for i in range(num_vectors)]
    
    # Create a query vector
    query = np.random.random(dimension).astype(np.float32)
    
    return {
        "dimension": dimension,
        "num_vectors": num_vectors,
        "vectors": vectors,
        "metadata": metadata,
        "query": query,
    }

def test_vector_index_creation(vector_data):
    """Test creating a VectorIndex."""
    # Create a vector index
    index = VectorIndex(dimension=vector_data["dimension"], metric="cosine")
    
    # Check that the index was created with the correct parameters
    assert index.dimension == vector_data["dimension"]
    assert index.metric == "cosine"
    assert len(index) == 0

def test_vector_index_add(vector_data):
    """Test adding vectors to a VectorIndex."""
    # Create a vector index
    index = VectorIndex(dimension=vector_data["dimension"], metric="cosine")
    
    # Add vectors to the index
    index.add(vector_data["vectors"], vector_data["metadata"])
    
    # Check that the vectors were added
    assert len(index) == vector_data["num_vectors"]
    
    # Check that adding vectors with incorrect dimensions raises an error
    with pytest.raises(ValueError):
        index.add(np.random.random((10, vector_data["dimension"] + 1)).astype(np.float32))
    
    # Check that adding vectors with incorrect metadata raises an error
    with pytest.raises(ValueError):
        index.add(
            np.random.random((10, vector_data["dimension"])).astype(np.float32),
            [{"id": i} for i in range(5)]  # Only 5 metadata items for 10 vectors
        )

def test_vector_index_search(vector_data):
    """Test searching a VectorIndex."""
    # Create a vector index
    index = VectorIndex(dimension=vector_data["dimension"], metric="cosine")
    
    # Add vectors to the index
    index.add(vector_data["vectors"], vector_data["metadata"])
    
    # Search the index
    results = index.search(vector_data["query"], k=5)
    
    # Check that the correct number of results was returned
    assert len(results) == 5
    
    # Check that the results have the expected structure
    for result in results:
        assert "id" in result
        assert "score" in result
        assert "metadata" in result
        assert "vector" in result
        
        assert isinstance(result["id"], int)
        assert isinstance(result["score"], float)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["vector"], np.ndarray)
    
    # Check that searching with a filter works
    def filter_fn(metadata):
        return metadata["id"] % 2 == 0
    
    filtered_results = index.search(vector_data["query"], k=5, filter_fn=filter_fn)
    
    # Check that all results match the filter
    for result in filtered_results:
        assert result["metadata"]["id"] % 2 == 0

def test_vector_index_batch_search(vector_data):
    """Test batch searching a VectorIndex."""
    # Create a vector index
    index = VectorIndex(dimension=vector_data["dimension"], metric="cosine")
    
    # Add vectors to the index
    index.add(vector_data["vectors"], vector_data["metadata"])
    
    # Create a batch of queries
    queries = np.random.random((3, vector_data["dimension"])).astype(np.float32)
    
    # Batch search the index
    results = index.batch_search(queries, k=5)
    
    # Check that the correct number of result sets was returned
    assert len(results) == 3
    
    # Check that each result set has the expected structure
    for result_set in results:
        assert len(result_set) == 5
        
        for result in result_set:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert "vector" in result
            
            assert isinstance(result["id"], int)
            assert isinstance(result["score"], float)
            assert isinstance(result["metadata"], dict)
            assert isinstance(result["vector"], list)

def test_vector_index_save_load(vector_data):
    """Test saving and loading a VectorIndex."""
    # Create a vector index
    index = VectorIndex(dimension=vector_data["dimension"], metric="cosine")
    
    # Add vectors to the index
    index.add(vector_data["vectors"], vector_data["metadata"])
    
    # Create a temporary directory for saving the index
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the index
        save_path = Path(temp_dir) / "test_index"
        index.save(save_path)
        
        # Check that the files were created
        assert (save_path.with_suffix(".vectors.npy")).exists()
        assert (save_path.with_suffix(".metadata.npy")).exists()
        assert (save_path.with_suffix(".config.npy")).exists()
        
        # Load the index
        loaded_index = VectorIndex.load(save_path)
        
        # Check that the loaded index has the same properties
        assert loaded_index.dimension == index.dimension
        assert loaded_index.metric == index.metric
        assert len(loaded_index) == len(index)
        
        # Check that the loaded index returns the same search results
        original_results = index.search(vector_data["query"], k=5)
        loaded_results = loaded_index.search(vector_data["query"], k=5)
        
        assert len(original_results) == len(loaded_results)
        
        for i in range(len(original_results)):
            assert original_results[i]["id"] == loaded_results[i]["id"]
            assert np.isclose(original_results[i]["score"], loaded_results[i]["score"])

def test_batch_processor():
    """Test the BatchProcessor class."""
    # Create a batch processor
    processor = BatchProcessor(batch_size=10)
    
    # Create test data
    data = np.random.random((100, 10)).astype(np.float32)
    matrix = np.random.random((10, 5)).astype(np.float32)
    
    # Define a processing function
    def process_fn(batch):
        return batch * 2
    
    # Process the data
    processed = processor.process_batches(data, process_fn)
    
    # Check that the processed data has the expected shape
    assert processed.shape == data.shape
    
    # Check that the processing was applied correctly
    assert np.allclose(processed, data * 2)
    
    # Test map_vectors
    mapped = processor.map_vectors(data, matrix)
    
    # Check that the mapped data has the expected shape
    assert mapped.shape == (100, 5)
    
    # Check that the mapping was applied correctly
    expected = np.matmul(data, matrix)
    assert np.allclose(mapped, expected)

def test_demonstrate_acceleration():
    """Test the demonstrate_acceleration function."""
    # Run the demonstration
    results = demonstrate_acceleration(dimension=32, num_vectors=100)
    
    # Check that the results have the expected structure
    assert "acceleration" in results
    assert "index_time" in results
    assert "search_time" in results
    assert "vectors_per_second" in results
    
    # Check that the values are reasonable
    assert results["index_time"] > 0
    assert results["search_time"] > 0
    assert results["vectors_per_second"] > 0 