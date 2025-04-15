#!/usr/bin/env python
"""
LlamaDB API Client Demonstration

This script demonstrates how to use the LlamaDB API client to interact with
the LlamaDB API server, handling vector dimensions and filtering results correctly.
"""

import json
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Set up console for rich output
console = Console()

# Default API endpoint
API_BASE_URL = "http://127.0.0.1:8000"


def print_header():
    """Print a stylized header."""
    console.print()
    console.print(Panel.fit("LlamaDB API Client Demonstration", border_style="blue"))
    console.print()


class LlamaDBClient:
    """A simple client for the LlamaDB API."""

    def __init__(self, base_url: str = API_BASE_URL):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            console.print(f"[bold red]Error checking API health:[/bold red] {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information from the API."""
        try:
            response = self.session.get(f"{self.base_url}/system")
            if response.status_code == 200:
                return response.json()
            else:
                console.print(
                    f"[bold red]Error {response.status_code}:[/bold red] {response.text}"
                )
                return {}
        except Exception as e:
            console.print(f"[bold red]Error getting system info:[/bold red] {e}")
            return {}

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the vector index."""
        try:
            response = self.session.get(f"{self.base_url}/index")
            if response.status_code == 200:
                return response.json()
            else:
                console.print(
                    f"[bold red]Error {response.status_code}:[/bold red] {response.text}"
                )
                return {}
        except Exception as e:
            console.print(f"[bold red]Error getting index info:[/bold red] {e}")
            return {}

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector index for similar vectors.

        Args:
            vector: The query vector
            k: Number of results to return
            filter_dict: Dictionary of metadata filter conditions

        Returns:
            List of search results
        """
        # Prepare the vector
        if vector.ndim > 1:
            # Ensure the vector is 1D
            vector = vector.flatten()

        # Get index information to determine the target dimension
        index_info = self.get_index_info()
        target_dim = index_info.get("dimension", 128)

        # Handle dimension mismatch
        if len(vector) != target_dim:
            console.print(
                f"[yellow]Warning: Vector dimension ({len(vector)}) doesn't match index dimension ({target_dim})[/yellow]"
            )

            if len(vector) < target_dim:
                # Pad with zeros
                padded = np.zeros(target_dim, dtype=vector.dtype)
                padded[: len(vector)] = vector
                vector = padded
                console.print(
                    f"[green]Padded vector to {target_dim} dimensions[/green]"
                )
            else:
                # Truncate
                vector = vector[:target_dim]
                console.print(
                    f"[yellow]Truncated vector to {target_dim} dimensions[/yellow]"
                )

        # Convert vector to list for JSON serialization
        vector_list = vector.tolist()

        # Prepare request payload
        payload = {"vector": vector_list, "k": k}

        # Add filter if provided
        if filter_dict:
            payload["filter"] = filter_dict

        # Send the request
        try:
            response = self.session.post(f"{self.base_url}/search", json=payload)

            if response.status_code == 200:
                return response.json()
            else:
                console.print(
                    f"[bold red]Error {response.status_code}:[/bold red] {response.text}"
                )
                return []
        except Exception as e:
            console.print(f"[bold red]Error during search:[/bold red] {e}")
            return []


def demo_api_client():
    """Demonstrate using the LlamaDB API client."""
    print_header()

    # Initialize client
    console.print("[bold]Initializing LlamaDB API client[/bold]")
    client = LlamaDBClient()

    # Check API health
    console.print("\n[bold cyan]Checking API Health[/bold cyan]")
    if client.health_check():
        console.print("[bold green]API is healthy! ✅[/bold green]")
    else:
        console.print("[bold red]API is not responding! ❌[/bold red]")
        console.print("Make sure the API server is running with:")
        console.print("  python api_launcher.py --start")
        return

    # Get system info
    console.print("\n[bold cyan]System Information[/bold cyan]")
    system_info = client.get_system_info()

    if system_info:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Property")
        table.add_column("Value")

        for key, value in system_info.items():
            table.add_row(key, str(value))

        console.print(table)
    else:
        console.print("[yellow]Could not retrieve system information[/yellow]")

    # Get index info
    console.print("\n[bold cyan]Vector Index Information[/bold cyan]")
    index_info = client.get_index_info()

    if index_info:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Property")
        table.add_column("Value")

        for key, value in index_info.items():
            table.add_row(key, str(value))

        console.print(table)

        dimension = index_info.get("dimension", 128)
        item_count = index_info.get("item_count", 0)

        console.print(
            f"Vector index has {item_count} items with dimension {dimension}."
        )
    else:
        console.print("[yellow]Could not retrieve index information[/yellow]")

    # Demo search with different vector dimensions
    console.print("\n[bold cyan]Vector Search Demo[/bold cyan]")

    # Get the correct dimension from index info
    dimension = index_info.get("dimension", 128)

    # Demo 1: Search with correctly sized vector
    console.print("\n[cyan]1. Search with correct dimensions:[/cyan]")
    correct_vector = np.random.random(dimension).astype(np.float32)
    results = client.search(correct_vector, k=3)

    if results:
        console.print(f"Found {len(results)} results")
        display_search_results(results)
    else:
        console.print("[yellow]No results found[/yellow]")

    # Demo 2: Search with undersized vector (should be padded)
    console.print("\n[cyan]2. Search with smaller vector (automatic padding):[/cyan]")
    small_vector = np.random.random(5).astype(np.float32)
    results = client.search(small_vector, k=3)

    if results:
        console.print(f"Found {len(results)} results after padding")
        display_search_results(results)
    else:
        console.print("[yellow]No results found[/yellow]")

    # Demo 3: Search with oversized vector (should be truncated)
    console.print("\n[cyan]3. Search with larger vector (automatic truncation):[/cyan]")
    large_vector = np.random.random(dimension * 2).astype(np.float32)
    results = client.search(large_vector, k=3)

    if results:
        console.print(f"Found {len(results)} results after truncation")
        display_search_results(results)
    else:
        console.print("[yellow]No results found[/yellow]")

    # Demo 4: Search with filter
    console.print("\n[cyan]4. Search with category filter:[/cyan]")
    # Using the categories from our demo script
    filter_dict = {"category": "Technology"}
    results = client.search(correct_vector, k=3, filter_dict=filter_dict)

    if results:
        console.print(f"Found {len(results)} Technology results")
        display_search_results(results)
    else:
        console.print("[yellow]No filtered results found[/yellow]")
        console.print(
            "[yellow]Note: If filtering doesn't work, the API may not support the filter_fn parameter yet[/yellow]"
        )

    console.print("\n[bold green]API client demo completed![/bold green]")


def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a table."""
    if not results:
        return

    # Check if results is a dictionary with a 'results' key (API response format)
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    # If still no results after extraction, return
    if not results or len(results) == 0:
        console.print("[yellow]No results to display[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")

    # Determine available columns based on first result
    first_result = results[0]
    columns = []

    # Always include these if available
    for key in ["id", "score"]:
        if key in first_result:
            columns.append(key)

    # Add metadata columns if present
    if "metadata" in first_result:
        for key in first_result["metadata"]:
            if key not in columns:
                columns.append(f"metadata.{key}")
    else:
        # Add other keys that might be flattened metadata
        for key in first_result:
            if key not in ["id", "score", "metadata"] and key not in columns:
                columns.append(key)

    # Add columns to table
    for col in columns:
        table.add_column(col.replace("metadata.", ""))

    # Add rows
    for result in results:
        row_values = []

        for col in columns:
            if "metadata." in col:
                # Extract from nested metadata
                meta_key = col.replace("metadata.", "")
                if "metadata" in result and meta_key in result["metadata"]:
                    row_values.append(str(result["metadata"][meta_key]))
                else:
                    row_values.append("N/A")
            else:
                # Direct attribute
                row_values.append(str(result.get(col, "N/A")))

        table.add_row(*row_values)

    console.print(table)


if __name__ == "__main__":
    demo_api_client()
