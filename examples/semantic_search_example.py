#!/usr/bin/env python
"""
Semantic Search Example with LlamaDB

This example demonstrates how to use LlamaDB to build a semantic search system
for a collection of articles. It covers:

1. Loading and preprocessing articles
2. Generating embeddings using sentence-transformers
3. Creating and configuring a LlamaDB vector index
4. Adding articles with metadata to the index
5. Performing semantic searches with metadata filtering
6. Saving and loading the index for persistence
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from llamadb.core import VectorIndex

# Check if running on Apple Silicon for MLX acceleration
try:
    from llamadb.core import is_apple_silicon, is_mlx_available
    HAS_MLX = is_apple_silicon() and is_mlx_available()
except ImportError:
    HAS_MLX = False


class SemanticSearchEngine:
    """A semantic search engine built with LlamaDB."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        use_mlx: bool = HAS_MLX,
        index_type: str = "hnsw",
        metric: str = "cosine"
    ):
        """
        Initialize the search engine.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            use_mlx: Whether to use MLX acceleration on Apple Silicon
            index_type: Index type to use ("flat" or "hnsw")
            metric: Distance metric to use ("cosine", "l2", or "dot")
        """
        # Load the embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Get the embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
        
        # Create the vector index
        print(f"Creating {index_type} index with {metric} metric")
        self.index = VectorIndex(
            dimension=self.dimension,
            index_type=index_type,
            metric=metric,
            use_mlx=use_mlx,
            ef_construction=200 if index_type == "hnsw" else None,
            ef_search=50 if index_type == "hnsw" else None,
            M=16 if index_type == "hnsw" else None
        )
        
        self.articles_added = 0
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text."""
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.astype(np.float32)
    
    def add_article(self, article: Dict[str, Any]) -> None:
        """
        Add a single article to the index.
        
        Args:
            article: Dictionary containing article data with at least 'title' and 'content' fields
        """
        # Create text to embed (title + content)
        text_to_embed = f"{article['title']}. {article['content']}"
        
        # Generate embedding
        embedding = self._generate_embedding(text_to_embed)
        
        # Add to index
        self.index.add_item(
            embedding=embedding,
            metadata={
                "id": article.get("id", self.articles_added),
                "title": article["title"],
                "content": article["content"],
                "category": article.get("category", "Unknown"),
                "date": article.get("date", "Unknown"),
                "author": article.get("author", "Unknown"),
                "url": article.get("url", ""),
                "tags": article.get("tags", [])
            }
        )
        
        self.articles_added += 1
    
    def add_articles_batch(self, articles: List[Dict[str, Any]]) -> None:
        """
        Add multiple articles to the index in a batch.
        
        Args:
            articles: List of article dictionaries
        """
        # Prepare embeddings and metadata
        texts_to_embed = [f"{article['title']}. {article['content']}" for article in articles]
        
        # Generate embeddings in batch
        print(f"Generating embeddings for {len(texts_to_embed)} articles...")
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)
        
        # Prepare metadata list
        metadata_list = []
        for i, article in enumerate(articles):
            metadata_list.append({
                "id": article.get("id", self.articles_added + i),
                "title": article["title"],
                "content": article["content"],
                "category": article.get("category", "Unknown"),
                "date": article.get("date", "Unknown"),
                "author": article.get("author", "Unknown"),
                "url": article.get("url", ""),
                "tags": article.get("tags", [])
            })
        
        # Add to index in batch
        print("Adding articles to index...")
        self.index.add_items(embeddings=embeddings, metadata_list=metadata_list)
        
        self.articles_added += len(articles)
        print(f"Added {len(articles)} articles. Total: {self.articles_added}")
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for articles similar to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of search results with metadata and relevance scores
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        # Search the index
        results = self.index.search(
            query_vector=query_embedding,
            k=k,
            filter=filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "title": result.metadata["title"],
                "content": result.metadata["content"][:200] + "...",  # Show snippet
                "category": result.metadata["category"],
                "date": result.metadata["date"],
                "author": result.metadata["author"],
                "url": result.metadata["url"],
                "tags": result.metadata["tags"]
            })
        
        return formatted_results
    
    def save(self, file_path: str) -> None:
        """Save the index to disk."""
        print(f"Saving index to {file_path}...")
        self.index.save(file_path)
        print("Index saved successfully.")
    
    @classmethod
    def load(
        cls, 
        file_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        use_mlx: bool = HAS_MLX
    ) -> "SemanticSearchEngine":
        """Load a saved index from disk."""
        print(f"Loading index from {file_path}...")
        
        # Create an instance
        engine = cls(model_name=model_name, use_mlx=use_mlx)
        
        # Load the index
        engine.index = VectorIndex.load(file_path)
        
        # Update the dimension
        engine.dimension = engine.index.dimension
        
        print(f"Index loaded successfully with {engine.index.count()} articles.")
        engine.articles_added = engine.index.count()
        
        return engine


def load_sample_articles() -> List[Dict[str, Any]]:
    """Load sample articles for demonstration."""
    articles = [
        {
            "title": "Understanding Vector Databases",
            "content": "Vector databases are specialized database systems designed to store and query high-dimensional vector embeddings. They are fundamental for building semantic search, recommendation systems, and other AI applications. Unlike traditional databases, vector databases optimize for similarity search in vector spaces.",
            "category": "Technology",
            "date": "2023-06-15",
            "author": "Jane Smith",
            "tags": ["vector database", "embeddings", "similarity search"]
        },
        {
            "title": "Introduction to Neural Networks",
            "content": "Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of interconnected nodes (neurons) that process and transmit signals. Deep learning involves neural networks with multiple layers that can learn hierarchical representations of data. They excel at tasks like image recognition and natural language processing.",
            "category": "Machine Learning",
            "date": "2023-05-20",
            "author": "John Doe",
            "tags": ["neural networks", "deep learning", "AI"]
        },
        {
            "title": "Python for Data Science",
            "content": "Python has become the de facto language for data science due to its simplicity and powerful libraries. Libraries like NumPy, Pandas, and Matplotlib provide essential tools for data manipulation and visualization. More specialized libraries like TensorFlow and PyTorch enable building complex machine learning models with ease.",
            "category": "Programming",
            "date": "2023-07-10",
            "author": "Alex Johnson",
            "tags": ["python", "data science", "programming"]
        },
        {
            "title": "Climate Change: Global Impact",
            "content": "Climate change refers to long-term shifts in temperature and weather patterns. Human activities, particularly the burning of fossil fuels, have been the main driver of climate change since the 1800s. The effects include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems worldwide.",
            "category": "Environment",
            "date": "2023-04-22",
            "author": "Maria Garcia",
            "tags": ["climate change", "environment", "global warming"]
        },
        {
            "title": "The Future of Remote Work",
            "content": "The COVID-19 pandemic accelerated the adoption of remote work across industries. Many companies have discovered benefits like reduced overhead costs and access to a global talent pool. Employees often report improved work-life balance and productivity. However, challenges remain, including maintaining company culture and addressing digital inequality.",
            "category": "Business",
            "date": "2023-08-05",
            "author": "David Wilson",
            "tags": ["remote work", "future of work", "business"]
        },
        {
            "title": "Quantum Computing Explained",
            "content": "Quantum computing leverages quantum mechanics to process information in fundamentally different ways than classical computers. Instead of using bits that are either 0 or 1, quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously. This enables them to solve certain problems exponentially faster than classical computers.",
            "category": "Technology",
            "date": "2023-03-18",
            "author": "Sarah Chen",
            "tags": ["quantum computing", "technology", "computing"]
        },
        {
            "title": "Sustainable Agriculture Practices",
            "content": "Sustainable agriculture focuses on meeting society's food needs while preserving the environment for future generations. Practices include crop rotation, reduced tillage, precision farming, and organic methods. By minimizing inputs like water and chemicals, sustainable agriculture can reduce environmental impact while maintaining productivity.",
            "category": "Environment",
            "date": "2023-05-30",
            "author": "James Miller",
            "tags": ["agriculture", "sustainability", "environment"]
        },
        {
            "title": "Introduction to Blockchain Technology",
            "content": "Blockchain is a distributed ledger technology that enables secure, transparent, and tamper-resistant record-keeping. Each block contains a timestamp and a link to the previous block, forming a chain of information. While best known for cryptocurrencies like Bitcoin, blockchain has applications in supply chain management, voting systems, and more.",
            "category": "Technology",
            "date": "2023-07-25",
            "author": "Robert Kim",
            "tags": ["blockchain", "cryptocurrency", "technology"]
        },
        {
            "title": "The Science of Sleep",
            "content": "Sleep is a natural, periodic state of rest characterized by altered consciousness and reduced sensory activity. Scientists have identified several stages of sleep, each serving different functions for physical and mental health. Quality sleep is essential for memory consolidation, immune function, and emotional regulation.",
            "category": "Health",
            "date": "2023-06-10",
            "author": "Emily Johnson",
            "tags": ["sleep", "health", "science"]
        },
        {
            "title": "Exploring Mars: Recent Discoveries",
            "content": "Recent Mars missions have revolutionized our understanding of the Red Planet. Rovers like Perseverance and orbiters have found evidence of ancient lakes and rivers, suggesting Mars once had conditions suitable for life. These missions use sophisticated instruments to analyze soil samples, capture high-resolution images, and search for signs of past or present microbial life.",
            "category": "Space",
            "date": "2023-08-20",
            "author": "Thomas Brown",
            "tags": ["Mars", "space exploration", "astronomy"]
        }
    ]
    
    # Add IDs
    for i, article in enumerate(articles):
        article["id"] = i + 1
    
    return articles


def main():
    # Create the search engine
    print("Initializing semantic search engine...")
    search_engine = SemanticSearchEngine(
        model_name="all-MiniLM-L6-v2",
        use_mlx=HAS_MLX,
        index_type="hnsw",
        metric="cosine"
    )
    
    # Display MLX status
    if HAS_MLX:
        print("MLX acceleration is enabled")
    else:
        print("MLX acceleration is not available")
    
    # Load sample articles
    articles = load_sample_articles()
    print(f"Loaded {len(articles)} sample articles")
    
    # Add articles to the index
    search_engine.add_articles_batch(articles)
    
    # Perform searches
    print("\n--- Basic Search ---")
    results = search_engine.search("climate impact on environment", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Category: {result['category']}, Date: {result['date']}")
        print(f"   {result['content']}\n")
    
    # Search with category filter
    print("\n--- Search with Category Filter ---")
    results = search_engine.search(
        "advanced technology trends", 
        k=3,
        filter={"category": "Technology"}
    )
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Category: {result['category']}, Date: {result['date']}")
        print(f"   {result['content']}\n")
    
    # Search with complex filter
    print("\n--- Search with Complex Filter ---")
    results = search_engine.search(
        "modern computing systems", 
        k=3,
        filter={
            "$or": [
                {"category": "Technology"},
                {"category": "Machine Learning"}
            ],
            "date": {"$gte": "2023-06-01"}
        }
    )
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Category: {result['category']}, Date: {result['date']}")
        print(f"   {result['content']}\n")
    
    # Save the index
    index_path = "article_index.llamadb"
    search_engine.save(index_path)
    
    # Load the index
    loaded_engine = SemanticSearchEngine.load(index_path)
    
    # Verify the loaded index works
    print("\n--- Search with Loaded Index ---")
    results = loaded_engine.search("sustainable environmental practices", k=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Category: {result['category']}, Date: {result['date']}")
        print(f"   {result['content']}\n")
    
    print(f"Index file saved to {os.path.abspath(index_path)}")
    print("Semantic search example completed successfully!")


if __name__ == "__main__":
    main() 