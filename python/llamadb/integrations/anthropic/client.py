"""
Anthropic Claude Client

Provides a client for interacting with Anthropic's Claude API.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Literal

import httpx

from llamadb.utils.logging import get_logger

logger = get_logger(__name__)

class AnthropicClient:
    """
    Client for interacting with Anthropic's Claude API.
    
    This client provides methods for text generation, embeddings, and 
    other Claude-specific functionality.
    """
    
    API_URL = "https://api.anthropic.com/v1"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = "claude-3-opus-20240229",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: API key for Anthropic. If not provided, will look for
                ANTHROPIC_API_KEY environment variable.
            default_model: Default model to use for requests.
            timeout: Timeout for API requests in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Anthropic API key provided. Please set ANTHROPIC_API_KEY environment "
                "variable or provide api_key parameter."
            )
        
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize HTTP client with retries
        self.client = httpx.Client(timeout=timeout)
    
    def _headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from the API."""
        if response.status_code == 401:
            raise ValueError("Invalid API key. Please check your Anthropic API key.")
        elif response.status_code == 400:
            error_data = response.json()
            raise ValueError(f"Bad request: {error_data.get('error', {}).get('message', 'Unknown error')}")
        elif response.status_code == 429:
            raise ValueError("Rate limit exceeded. Please try again later.")
        else:
            response.raise_for_status()
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text with Claude.
        
        Args:
            prompt: The user prompt to send to Claude
            model: Model to use (defaults to instance default_model)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            top_p: Optional nucleus sampling parameter
            top_k: Optional top-k sampling parameter
            
        Returns:
            Response from Claude API
        """
        model = model or self.default_model
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
            
        if top_p is not None:
            payload["top_p"] = top_p
            
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = self.client.post(
            f"{self.API_URL}/messages",
            headers=self._headers(),
            json=payload,
        )
        
        if response.status_code != 200:
            self._handle_error(response)
            
        return response.json()
    
    def get_embeddings(
        self,
        texts: List[str],
        model: str = "claude-3-sonnet-20240229",
    ) -> List[List[float]]:
        """
        Get embeddings for texts using Claude's embedding models.
        
        Args:
            texts: List of texts to get embeddings for
            model: Model to use for embeddings
            
        Returns:
            List of embeddings as float arrays
        """
        # Check for API key
        if not self.api_key:
            raise ValueError("Anthropic API key is required for embeddings")
        
        # Prepare payload
        payload = {
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }
        
        # Make request
        response = self.client.post(
            f"{self.API_URL}/embeddings",
            headers=self._headers(),
            json=payload,
        )
        
        if response.status_code != 200:
            self._handle_error(response)
        
        data = response.json()
        return [item["embedding"] for item in data["embeddings"]] 