"""
FastAPI application for LlamaDB.

This module implements the REST API for LlamaDB, providing endpoints
for vector search, system information, and other operations.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llamadb.core.mlx_acceleration import (
    is_mlx_available,
    is_apple_silicon,
    get_system_info,
)
from llamadb.core.vector_index import VectorIndex

# Configure logger
logger = logging.getLogger(__name__)

# API Models
class SystemInfo(BaseModel):
    """System information response model."""
    platform: str
    processor: str
    python_version: str
    is_apple_silicon: bool
    mlx_available: bool
    mlx_version: Optional[str] = None

class VectorSearchRequest(BaseModel):
    """Vector search request model."""
    vector: List[float] = Field(..., description="Query vector")
    k: int = Field(10, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")

class VectorSearchResult(BaseModel):
    """Vector search result model."""
    id: int
    score: float
    metadata: Dict[str, Any]

class VectorSearchResponse(BaseModel):
    """Vector search response model."""
    results: List[VectorSearchResult]
    took_ms: float
    backend: str
    error: Optional[str] = None

class VectorIndexInfo(BaseModel):
    """Vector index information model."""
    dimension: int
    metric: str
    count: int
    backend: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime: float
    acceleration: str

# Global variables
_start_time = time.time()
_vector_index: Optional[VectorIndex] = None

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="LlamaDB API",
        description="REST API for LlamaDB, a next-gen hybrid Python/Rust data platform with MLX acceleration",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        logger.info("Starting LlamaDB API server")
        
        # Initialize vector index with demo data
        global _vector_index
        _vector_index = create_demo_index()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        logger.info("Shutting down LlamaDB API server")
    
    # Register routes
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "uptime": time.time() - _start_time,
            "acceleration": "MLX" if is_mlx_available() else "NumPy"
        }
    
    @app.get("/system", response_model=SystemInfo)
    async def get_system_information():
        """Get system information."""
        info = get_system_info()
        return {
            "platform": info["platform"],
            "processor": info["processor"],
            "python_version": info["python_version"],
            "is_apple_silicon": info["is_apple_silicon"] == "True",
            "mlx_available": info["mlx_available"] == "True",
            "mlx_version": info.get("mlx_version")
        }
    
    @app.get("/index", response_model=VectorIndexInfo)
    async def get_index_info():
        """Get information about the vector index."""
        if _vector_index is None:
            raise HTTPException(status_code=404, detail="Vector index not initialized")
        
        return {
            "dimension": _vector_index.dimension,
            "metric": _vector_index.metric,
            "count": len(_vector_index),
            "backend": "MLX" if _vector_index.use_mlx else "NumPy"
        }
    
    @app.post("/search", response_model=VectorSearchResponse)
    async def search_vectors(request: VectorSearchRequest):
        """
        Search for similar vectors.
        
        Args:
            request: Search request containing query vector and parameters
            
        Returns:
            Search results with similarity scores
        """
        if _vector_index is None:
            raise HTTPException(status_code=404, detail="Vector index not initialized")
        
        try:
            # Convert query vector to numpy array
            query = np.array(request.vector, dtype=np.float32)
            
            # Check dimension and handle padding/truncation if needed
            if query.shape[0] != _vector_index.dimension:
                logger.warning(f"Query vector dimension ({query.shape[0]}) does not match index dimension ({_vector_index.dimension})")
                if query.shape[0] < _vector_index.dimension:
                    # Pad vector with zeros to match dimension
                    padding = np.zeros(_vector_index.dimension - query.shape[0], dtype=np.float32)
                    query = np.concatenate([query, padding])
                    logger.info(f"Vector padded from {len(request.vector)} to {_vector_index.dimension} dimensions")
                else:
                    # Truncate vector to match dimension
                    query = query[:_vector_index.dimension]
                    logger.info(f"Vector truncated from {len(request.vector)} to {_vector_index.dimension} dimensions")
            
            # Create filter function if filter is provided
            filter_metadata = None
            if request.filter:
                filter_metadata = request.filter
            
            # Perform search
            start = time.time()
            results = _vector_index.search(query, k=request.k, filter_metadata=filter_metadata)
            took = time.time() - start
            
            # Format results
            formatted_results = [
                {
                    "id": int(r["id"]),
                    "score": float(r["score"]),
                    "metadata": r["metadata"]
                }
                for r in results
            ]
            
            return {
                "results": formatted_results,
                "took_ms": took * 1000,
                "backend": "MLX" if _vector_index.use_mlx else "NumPy"
            }
            
        except Exception as e:
            logger.exception("Error in vector search")
            # Check if this is a dimension mismatch error and handle it gracefully
            error_str = str(e)
            if "dimension" in error_str and "does not match" in error_str:
                # This is a dimension mismatch error, return empty results
                logger.warning(f"Dimension mismatch error handled gracefully: {error_str}")
                return {
                    "results": [],
                    "took_ms": 0,
                    "backend": "MLX" if _vector_index and _vector_index.use_mlx else "NumPy",
                    "error": f"Vector dimension mismatch: {error_str}"
                }
            # For other errors, still raise the exception
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def create_demo_index(dimension: int = 128, num_vectors: int = 10000) -> VectorIndex:
    """
    Create a demo vector index with random data.
    
    Args:
        dimension: Dimensionality of vectors
        num_vectors: Number of vectors to generate
        
    Returns:
        VectorIndex: Initialized vector index
    """
    logger.info(f"Creating demo vector index with {num_vectors} vectors of dimension {dimension}")
    
    # Create vector index
    index = VectorIndex(dimension=dimension, metric="cosine", use_mlx=True)
    
    # Generate random vectors
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    
    # Create metadata
    metadata = [
        {
            "id": i,
            "text": f"Document {i}",
            "category": f"Category {i % 10}",
            "score": float(np.random.random()),
            "tags": [f"tag_{j}" for j in range(3)]
        }
        for i in range(num_vectors)
    ]
    
    # Add vectors to index
    index.add(vectors, metadata)
    
    logger.info(f"Demo vector index created with {len(index)} vectors")
    return index 