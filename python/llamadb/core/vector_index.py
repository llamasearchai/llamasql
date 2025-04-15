"""
Vector Index

This module provides a high-performance vector index for similarity search,
with optional MLX acceleration on Apple Silicon.
"""

import json
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from llamadb.core.accelerated_ops import cosine_similarity, dot_product, l2_distance
from llamadb.utils.logging import get_logger

logger = get_logger(__name__)


class VectorIndex:
    """
    High-performance vector index for similarity search.

    This class provides methods for storing and searching vectors,
    with optional MLX acceleration on Apple Silicon.
    """

    def __init__(
        self,
        dimension: int = 1536,
        metric: str = "cosine",
        use_mlx: bool = True,
        index_path: Optional[str] = None,
    ):
        """
        Initialize the vector index.

        Args:
            dimension: Dimension of vectors to store.
            metric: Distance metric to use ('cosine', 'l2', or 'dot').
            use_mlx: Whether to use MLX acceleration if available.
            index_path: Path to load index from.
        """
        self.dimension = dimension
        self.metric = metric.lower()
        self.use_mlx = use_mlx

        # Initialize empty index
        self.vectors = []
        self.metadata = []
        self.ids = []

        # Load index if path provided
        if index_path:
            self.load(index_path)

        # Set similarity function based on metric
        if self.metric == "cosine":
            self.similarity_fn = cosine_similarity
        elif self.metric == "l2":
            self.similarity_fn = l2_distance
        elif self.metric == "dot":
            self.similarity_fn = dot_product
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. Use 'cosine', 'l2', or 'dot'."
            )

        logger.info(
            f"Initialized VectorIndex with dimension={dimension}, metric={metric}"
        )
        if use_mlx:
            try:
                import mlx.core as mx

                logger.info(
                    "🚀 VectorIndex using MLX acceleration for similarity calculations"
                )
            except ImportError:
                logger.info(
                    "MLX not available, falling back to NumPy for similarity calculations"
                )

    def add_item(
        self,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """
        Add a single vector to the index.

        Args:
            vector: Vector to add.
            metadata: Optional metadata to associate with the vector.
            item_id: Optional ID for the vector. If not provided, a sequential ID will be used.

        Returns:
            ID of the added vector.
        """
        # Convert vector to numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        # Check dimension
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vector.shape[0]}) does not match index dimension ({self.dimension})"
            )

        # Generate ID if not provided
        if item_id is None:
            item_id = str(len(self.vectors))

        # Add to index
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
        self.ids.append(item_id)

        return item_id

    def add(
        self,
        vectors: Union[List[List[float]], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple vectors to the index.

        Args:
            vectors: List of vectors or numpy array of shape (n_vectors, dimension).
            metadata: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the vectors.

        Returns:
            List of IDs of the added vectors.
        """
        # Convert vectors to numpy array
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # Check dimensions
        if len(vectors.shape) != 2:
            raise ValueError(f"Expected 2D array of vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.vectors)
            ids = [str(i) for i in range(start_id, start_id + vectors.shape[0])]
        elif len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) does not match number of vectors ({vectors.shape[0]})"
            )

        # Generate metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(vectors.shape[0])]
        elif len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"Number of metadata items ({len(metadata)}) does not match number of vectors ({vectors.shape[0]})"
            )

        # Add to index
        for i in range(vectors.shape[0]):
            self.vectors.append(vectors[i])
            self.metadata.append(metadata[i])
            self.ids.append(ids[i])

        return ids

    def search(
        self,
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        query_text: Optional[str] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Vector to search for.
            query_text: Text to search for (requires embedding_fn to be set).
            k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of dictionaries with 'id', 'score', and 'metadata' keys.
        """
        if not self.vectors:
            logger.warning("Search called on empty index")
            return []

        # Convert vectors to numpy array for batch processing
        vectors_array = np.array(self.vectors, dtype=np.float32)

        # Get query vector
        if query_vector is not None:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
        elif query_text is not None:
            raise ValueError("Text search requires embedding_fn to be set")
        else:
            raise ValueError("Either query_vector or query_text must be provided")

        # Calculate similarities
        similarities = self.similarity_fn(
            query_vector, vectors_array, use_mlx=self.use_mlx
        )

        # Create result items
        results = []
        for i, score in enumerate(similarities):
            # Apply metadata filter if provided
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key not in self.metadata[i] or self.metadata[i][key] != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append(
                {"id": self.ids[i], "score": float(score), "metadata": self.metadata[i]}
            )

        # Sort by score (higher is better)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top k
        return results[:k]

    def save(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path: Path to save the index to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save index
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                    "ids": self.ids,
                },
                f,
            )

        logger.info(f"Saved index with {len(self.vectors)} vectors to {path}")

    @classmethod
    def load(cls, path: str, use_mlx: bool = True) -> "VectorIndex":
        """
        Load an index from disk.

        Args:
            path: Path to load the index from.
            use_mlx: Whether to use MLX acceleration if available.

        Returns:
            Loaded VectorIndex.
        """
        # Load index
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create index
        index = cls(dimension=data["dimension"], metric=data["metric"], use_mlx=use_mlx)

        # Set data
        index.vectors = data["vectors"]
        index.metadata = data["metadata"]
        index.ids = data["ids"]

        logger.info(f"Loaded index with {len(index.vectors)} vectors from {path}")
        return index

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self.vectors)
