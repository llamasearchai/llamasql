"""
Claude Embeddings

This module provides a wrapper for Anthropic's Claude embeddings API.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from llamadb.integrations.anthropic.client import AnthropicClient
from llamadb.utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeEmbeddings:
    """
    Wrapper for Anthropic's Claude embeddings API.

    This class provides methods for generating embeddings from text using
    Claude's embedding models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        batch_size: int = 10,
        normalize: bool = True,
    ):
        """
        Initialize the Claude embeddings wrapper.

        Args:
            api_key: API key for Anthropic. If not provided, will look for
                ANTHROPIC_API_KEY environment variable.
            model: Model to use for embeddings.
            batch_size: Number of texts to embed in a single API call.
            normalize: Whether to normalize embeddings to unit length.
        """
        self.client = AnthropicClient(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given texts.

        Args:
            texts: Text or list of texts to embed.

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Get embeddings from API
            try:
                embeddings = self.client.get_embeddings(batch, model=self.model)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error getting embeddings for batch {i}: {e}")
                # Return empty embeddings for failed batch
                empty_embeddings = [[0.0] * 1536] * len(
                    batch
                )  # Claude embeddings are 1536-dimensional
                all_embeddings.extend(empty_embeddings)

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1.0, norms)
            embeddings_array = embeddings_array / norms

        return embeddings_array

    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document dictionaries.
            text_key: Key in document dictionaries that contains the text to embed.

        Returns:
            Dictionary with document IDs as keys and embeddings as values.
        """
        # Extract texts and IDs
        texts = []
        ids = []

        for doc in documents:
            if text_key in doc:
                texts.append(doc[text_key])
                ids.append(doc.get("id", str(len(ids))))
            else:
                logger.warning(f"Document missing '{text_key}' key: {doc}")

        # Generate embeddings
        embeddings = self.embed(texts)

        # Create result dictionary
        result = {
            "model": self.model,
            "embeddings": {
                doc_id: embedding.tolist() for doc_id, embedding in zip(ids, embeddings)
            },
        }

        return result

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        # Normalize if not already normalized
        if self.normalize:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 > 0:
                embedding1 = embedding1 / norm1
            if norm2 > 0:
                embedding2 = embedding2 / norm2

        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2))
