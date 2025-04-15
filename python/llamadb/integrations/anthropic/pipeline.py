"""
Claude RAG Pipeline

This module provides a retrieval-augmented generation pipeline using Anthropic's Claude models.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llamadb.core.vector_index import VectorIndex
from llamadb.integrations.anthropic.client import AnthropicClient
from llamadb.utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeRAGResponse:
    """Container for Claude RAG response data."""

    def __init__(
        self,
        text: str,
        model: str,
        retrieved_documents: List[Dict[str, Any]],
        raw_response: Dict[str, Any],
    ):
        self.text = text
        self.model = model
        self.retrieved_documents = retrieved_documents
        self.raw_response = raw_response

    def __repr__(self) -> str:
        return f"ClaudeRAGResponse(model={self.model}, text_length={len(self.text)}, num_docs={len(self.retrieved_documents)})"


class ClaudeRAGPipeline:
    """
    Retrieval-augmented generation pipeline using Claude.

    This pipeline combines vector search with Claude generation to provide
    context-aware responses based on retrieved documents.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        document_formatter: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """
        Initialize the Claude RAG pipeline.

        Args:
            vector_index: Vector index containing documents to search
            model: Claude model to use for generation
            api_key: Anthropic API key
            system_prompt: Optional system prompt for Claude
            document_formatter: Optional function to format documents for context
        """
        self.vector_index = vector_index
        self.client = AnthropicClient(api_key=api_key, default_model=model)
        self.model = model

        # Default system prompt if none provided
        self.system_prompt = system_prompt or (
            "You are Claude, an AI assistant built by Anthropic to be helpful, harmless, and honest. "
            "You have been provided with relevant documents to help answer the user's question. "
            "Base your response on the content of these documents when possible. "
            "If the documents don't contain the information needed, acknowledge this and provide "
            "your best response based on your general knowledge, making clear what is from the "
            "documents and what is not."
        )

        # Default document formatter if none provided
        self.document_formatter = document_formatter or self._default_document_formatter

    def _default_document_formatter(self, doc: Dict[str, Any]) -> str:
        """Default function to format a document for context."""
        metadata = doc.get("metadata", {})
        text = metadata.get("text", "")
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown source")

        return f"Document: {title}\nSource: {source}\nContent: {text}\n"

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context for Claude."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            formatted_doc = self.document_formatter(doc)
            context_parts.append(f"[DOCUMENT {i}]\n{formatted_doc}\n")

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Build the prompt for Claude with retrieved documents as context."""
        context = self._format_context(documents)

        prompt = (
            "I'll answer your question based on the following documents:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        return prompt

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to the query.

        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant documents
        """
        results = self.vector_index.search(
            query_text=query,
            k=k,
            filter_metadata=filter_metadata,
        )

        return results

    def generate(
        self,
        query: str,
        num_documents: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        filter_metadata: Optional[Dict[str, Any]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> ClaudeRAGResponse:
        """
        Generate a response to a query using retrieval-augmented generation.

        Args:
            query: User query to answer
            num_documents: Number of documents to retrieve
            temperature: Sampling temperature for Claude
            max_tokens: Maximum number of tokens to generate
            filter_metadata: Optional metadata filters for document retrieval
            custom_system_prompt: Optional custom system prompt to override default

        Returns:
            ClaudeRAGResponse object containing the response and retrieved documents
        """
        # Search for relevant documents
        documents = self.search(query, k=num_documents, filter_metadata=filter_metadata)

        if not documents:
            logger.warning(f"No documents found for query: {query}")

        # Build prompt with retrieved documents
        prompt = self._build_prompt(query, documents)

        # Use custom system prompt if provided
        system_prompt = custom_system_prompt or self.system_prompt

        # Generate response with Claude
        response = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Extract the generated text
        text = response["content"][0]["text"]

        # Return structured response
        return ClaudeRAGResponse(
            text=text,
            model=self.model,
            retrieved_documents=documents,
            raw_response=response,
        )
