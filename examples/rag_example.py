#!/usr/bin/env python
"""
Retrieval Augmented Generation (RAG) Example with LlamaDB

This example demonstrates how to build a simple RAG system using LlamaDB for
vector storage and retrieval, and OpenAI for embedding generation and text completion.

The example covers:
1. Loading and preprocessing documents
2. Generating embeddings with OpenAI
3. Chunking documents into smaller sections
4. Storing document chunks in LlamaDB
5. Retrieving relevant context based on user queries
6. Generating responses with OpenAI using retrieved context

Requirements:
- LlamaDB
- OpenAI Python package
- dotenv (for API key management)
"""

import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the OpenAI package: pip install openai")

# Import LlamaDB
try:
    from llamadb.core import VectorIndex, is_apple_silicon, is_mlx_available

    HAS_MLX = is_apple_silicon() and is_mlx_available()
except ImportError:
    print("Warning: MLX acceleration not available")
    HAS_MLX = False


# Load environment variables (for OpenAI API key)
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not openai_client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


class Document:
    """Simple document class to store text content and metadata."""

    def __init__(
        self,
        id: str,
        content: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.content = content
        self.title = title or f"Document {id}"
        self.source = source
        self.metadata = metadata or {}


class Chunk:
    """A chunk of a document, used for embedding and retrieval."""

    def __init__(
        self,
        id: str,
        content: str,
        document_id: str,
        document_title: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.content = content
        self.document_id = document_id
        self.document_title = document_title
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class RAGSystem:
    """Retrieval Augmented Generation system using LlamaDB and OpenAI."""

    def __init__(
        self,
        use_mlx: bool = HAS_MLX,
        embedding_model: str = "text-embedding-3-small",
        completion_model: str = "gpt-3.5-turbo",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_tokens: int = 600,
    ):
        """
        Initialize the RAG system.

        Args:
            use_mlx: Whether to use MLX acceleration for vector operations
            embedding_model: OpenAI embedding model to use
            completion_model: OpenAI completion model to use
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            max_tokens: Maximum number of tokens for completions
        """
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens

        # Create the vector index (dimension 1536 for OpenAI embedding-3-small)
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        print(f"Creating vector index with dimension {self.dimension}...")
        self.index = VectorIndex(
            dimension=self.dimension,
            index_type="hnsw",
            metric="cosine",
            use_mlx=use_mlx,
        )

        self.documents = {}  # Store document objects by ID
        self.chunks = []  # Store all chunks

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        response = openai_client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into overlapping chunks."""
        content = document.content
        chunks = []

        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0

        while start < len(content):
            # Calculate end position with overlap
            end = min(start + self.chunk_size, len(content))

            # If this is not the first chunk and not the last chunk of the document,
            # try to break at a sentence boundary
            if start > 0 and end < len(content):
                # Look for sentence boundaries (., !, ?) followed by space or newline
                sentences_end = [
                    m.end() for m in re.finditer(r"[.!?]\s", content[start:end])
                ]
                if sentences_end:
                    # Use the last sentence boundary found
                    end = start + sentences_end[-1]

            # Create chunk with this portion of text
            chunk_content = content[start:end]
            chunk_id = f"{document.id}_chunk{chunk_index}"

            chunk = Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                document_title=document.title,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "is_chunk": True,
                },
            )

            chunks.append(chunk)

            # Move start position for next chunk, considering overlap
            if end == len(content):
                break

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks

    def add_document(self, document: Document) -> None:
        """
        Add a document to the RAG system.

        The document will be chunked, embedded, and stored in the vector index.
        """
        print(f"Processing document: {document.title} (ID: {document.id})")

        # Store the document
        self.documents[document.id] = document

        # Chunk the document
        chunks = self._chunk_document(document)
        print(f"  Created {len(chunks)} chunks")

        # Generate embeddings and add to index
        for chunk in chunks:
            embedding = self._generate_embedding(chunk.content)

            # Add to index
            self.index.add_item(
                embedding=np.array(embedding, dtype=np.float32),
                metadata={
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )

            # Store the chunk
            self.chunks.append(chunk)

        print(f"  Indexed {len(chunks)} chunks for document {document.id}")

    def retrieve_context(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The user query
            k: Number of chunks to retrieve
            filter: Optional metadata filter

        Returns:
            List of context chunks with metadata
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)

        # Search for relevant chunks
        results = self.index.search(
            query_vector=np.array(query_embedding, dtype=np.float32), k=k, filter=filter
        )

        # Format results
        context = []
        for result in results:
            context.append(
                {
                    "id": result.metadata["id"],
                    "content": result.metadata["content"],
                    "document_title": result.metadata["document_title"],
                    "document_id": result.metadata["document_id"],
                    "score": result.score,
                }
            )

        return context

    def generate_response(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response to a query using RAG.

        Args:
            query: The user query
            k: Number of context chunks to retrieve
            filter: Optional metadata filter for retrieval
            system_prompt: Optional system prompt for the LLM

        Returns:
            Tuple of (generated response, retrieved context)
        """
        # Retrieve context
        context = self.retrieve_context(query, k=k, filter=filter)

        # Format context for the prompt
        context_text = ""
        for i, ctx in enumerate(context):
            context_text += (
                f"\n[Document {i+1}: {ctx['document_title']}]\n{ctx['content']}\n"
            )

        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that provides accurate information based on the context provided. "
                "If the answer cannot be determined from the context, politely say so. "
                "Do not make up information. Use the context to provide precise answers."
            )

        # Generate completion with context
        prompt = f"Context information is below.\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        response = openai_client.chat.completions.create(
            model=self.completion_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )

        return response.choices[0].message.content, context

    def save(self, file_path: str) -> None:
        """Save the vector index to disk."""
        print(f"Saving index to {file_path}...")
        self.index.save(file_path)
        print(
            f"Index saved successfully. Current index has {self.index.count()} chunks."
        )


def load_sample_knowledge_base() -> List[Document]:
    """Load a sample knowledge base about AI and machine learning."""
    documents = [
        Document(
            id="doc1",
            title="Introduction to Neural Networks",
            content="""
            Neural networks are a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. They can adapt to changing input, so the network generates the best possible result without needing to redesign the output criteria.

            The concept of neural networks is inspired by the human brain. The human brain contains billions of neurons that are connected by synapses. These neurons and synapses work together to process and transmit information. Similarly, artificial neural networks consist of nodes (artificial neurons) and connections (artificial synapses) that work together to learn from data and make decisions.

            Neural networks have three main components: an input layer, hidden layers, and an output layer. The input layer receives data from outside sources. The hidden layers process this data using mathematical operations, and the output layer produces the final result. Each connection between nodes has a weight that adjusts as the network learns.

            Training a neural network involves feeding it large amounts of data. The network makes predictions on the data, compares these predictions to the actual outcomes, and adjusts its weights to reduce the error in its predictions. This process, known as backpropagation, is repeated many times until the network's predictions are accurate enough.

            Neural networks have many applications, including image and speech recognition, natural language processing, and autonomous vehicles. They're particularly good at handling complex, high-dimensional data and finding patterns that may not be obvious to humans.
            """,
            source="AI Textbook",
            metadata={"category": "Machine Learning", "difficulty": "beginner"},
        ),
        Document(
            id="doc2",
            title="Deep Learning vs. Traditional Machine Learning",
            content="""
            Deep learning and traditional machine learning are both subfields of artificial intelligence, but they differ in several key ways.

            Traditional machine learning algorithms require feature extraction as a separate step. This means that domain experts must identify and extract relevant features from the raw data before feeding it into the algorithm. For example, in image recognition, features might include edges, shapes, or textures. The algorithm then learns to use these pre-defined features to make predictions or decisions.

            In contrast, deep learning algorithms can learn to extract features automatically from raw data. They do this through multiple layers of processing (hence "deep"), where each layer transforms the data and extracts increasingly abstract features. This eliminates the need for manual feature extraction, which can be time-consuming and may miss important patterns in the data.

            Traditional machine learning algorithms generally perform well with small to medium-sized datasets. They can achieve good results with less data because they rely on pre-defined features. However, their performance typically plateaus after reaching a certain amount of data.

            Deep learning algorithms, on the other hand, often require large amounts of data to perform well. They need more data to effectively learn the features from the raw input. However, their performance continues to improve as more data is added, making them well-suited for scenarios where large datasets are available.

            Traditional machine learning algorithms are typically less computationally intensive and can be trained on standard CPUs. This makes them more accessible for many applications. Deep learning algorithms, especially deep neural networks, are more computationally intensive and often require specialized hardware like GPUs for efficient training. This can make them more resource-intensive and potentially more costly to deploy.

            In terms of interpretability, traditional machine learning algorithms like decision trees or linear regression often provide clear insights into how they make decisions. This transparency can be crucial in fields like healthcare or finance, where understanding the reasoning behind a decision is important.

            Deep learning algorithms, especially complex neural networks, are often considered "black boxes" because it can be difficult to understand how they arrive at a particular output. The multiple layers of transformations make it challenging to trace the decision-making process, which can be a drawback in applications where interpretability is important.

            The choice between deep learning and traditional machine learning depends on the specific application, the amount of available data, the computational resources, and the importance of interpretability. Traditional machine learning might be preferred for smaller datasets or when interpretability is crucial, while deep learning might be the better choice for large, complex datasets or when automatic feature extraction is beneficial.
            """,
            source="AI Research Paper",
            metadata={"category": "Machine Learning", "difficulty": "intermediate"},
        ),
        Document(
            id="doc3",
            title="Transformer Architecture in NLP",
            content="""
            The Transformer architecture, introduced in the 2017 paper "Attention is All You Need" by Vaswani et al., has revolutionized natural language processing (NLP). Unlike previous sequence-to-sequence models that used recurrent or convolutional neural networks, Transformers rely entirely on a mechanism called "attention" to draw global dependencies between input and output.

            The key innovation of the Transformer is the self-attention mechanism. Self-attention allows the model to weigh the importance of different words in a sentence when processing a specific word. For example, when processing the word "it" in a sentence, the model might give high attention to a noun that appeared earlier, understanding that "it" refers to that noun.

            A Transformer consists of an encoder and a decoder, each made up of multiple identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The encoder processes the input sequence, while the decoder generates the output sequence.

            The multi-head attention mechanism runs the attention function in parallel, with different learned linear projections of the same input. This allows the model to jointly attend to information from different representation subspaces at different positions, improving the model's ability to capture complex patterns in the data.

            Position encoding is another important aspect of Transformers. Since the model doesn't use recurrence or convolution, it would otherwise have no way to account for the order of words. Position encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks, providing information about the position of each word in the sequence.

            The Transformer architecture has several advantages over previous models. First, it's more parallelizable, meaning it can be trained more efficiently on modern hardware like GPUs. Second, it can capture long-range dependencies in text more effectively. This is particularly important in tasks like translation, where the meaning of a word might depend on words that appeared much earlier in the sentence.

            The success of the Transformer architecture has led to the development of many Transformer-based models, such as BERT, GPT, T5, and many others. These models have set new state-of-the-art results on a wide range of NLP tasks, from text classification and named entity recognition to machine translation and text generation.
            """,
            source="NLP Conference Proceedings",
            metadata={
                "category": "Natural Language Processing",
                "difficulty": "advanced",
            },
        ),
        Document(
            id="doc4",
            title="Reinforcement Learning Fundamentals",
            content="""
            Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It's different from supervised learning, where the training data includes the correct answers, and from unsupervised learning, where the goal is to find structure in unlabeled data.

            The core components of a reinforcement learning system are the agent and the environment. The agent is the learner or decision-maker, while the environment is everything the agent interacts with. The agent and environment interact continually: the agent selecting actions and the environment responding to these actions and presenting new situations to the agent.

            A key concept in RL is the reward signal. After each action, the environment sends a single number—the reward—to the agent. The agent's objective is to maximize the total reward it receives over the long run. This reward can be immediate or delayed, which is what makes RL challenging.

            The policy is the agent's behavior function—it maps from states to actions. Policies can be deterministic (always selecting the same action in a given state) or stochastic (selecting actions according to a probability distribution).

            The value function gives the expected return (cumulative future reward) if the agent starts in a particular state (or state-action pair) and follows a particular policy. It helps the agent decide which actions to take by evaluating how good different states are.

            The model is the agent's representation of the environment, which predicts how the environment will respond to actions. Some RL algorithms use models (model-based methods), while others learn directly from experience without a model (model-free methods).

            Exploration vs. exploitation is a fundamental trade-off in RL. The agent must exploit its knowledge to get high rewards, but it also needs to explore to find better actions. Many RL algorithms include mechanisms to balance exploration and exploitation.

            RL has been successfully applied to a variety of problems, including game playing (like AlphaGo), robotics, resource management, and recommendation systems. However, it can be computationally intensive and may require a lot of data to learn effectively.
            """,
            source="Reinforcement Learning Textbook",
            metadata={
                "category": "Reinforcement Learning",
                "difficulty": "intermediate",
            },
        ),
        Document(
            id="doc5",
            title="Ethics in Artificial Intelligence",
            content="""
            As artificial intelligence (AI) becomes more integrated into our daily lives, ethical considerations surrounding its development and use have become increasingly important. AI ethics encompasses a range of concerns, from data privacy and algorithmic bias to autonomous weapons and the long-term impact of AI on society.

            One of the central ethical concerns in AI is privacy. AI systems often require vast amounts of data to function effectively, raising questions about how this data is collected, stored, and used. There's a tension between the need for data to improve AI systems and the right of individuals to control their personal information. This has led to regulations like the GDPR in Europe, which gives individuals more control over their data.

            Algorithmic bias is another significant concern. AI systems learn from historical data, which may contain biases based on race, gender, or other factors. If not addressed, these biases can be perpetuated and even amplified by AI systems, leading to unfair outcomes. For example, a biased hiring algorithm might disproportionately reject candidates from certain demographic groups.

            Transparency and explainability are also important ethical considerations. Many AI systems, particularly deep learning models, function as "black boxes," making decisions that even their creators may not fully understand. This lack of transparency can erode trust in AI systems and make it difficult to identify and address issues like bias.

            The potential impact of AI on employment is another ethical concern. As AI systems become more capable, they may automate tasks currently performed by humans, potentially leading to job displacement. While AI may also create new jobs, there's a concern about whether the transition will be equitable and how to ensure that the benefits of AI are widely shared.

            In the longer term, there are ethical questions about the development of artificial general intelligence (AGI) and the potential risks of creating systems that could surpass human intelligence. These concerns have led to calls for responsible AI development and governance frameworks to ensure that AI is aligned with human values and goals.

            Addressing these ethical concerns requires a multidisciplinary approach, involving not just AI researchers and engineers, but also ethicists, social scientists, policymakers, and representatives from affected communities. It's increasingly recognized that ethical considerations should be integrated into AI development from the outset, rather than addressed as an afterthought.
            """,
            source="AI Ethics Journal",
            metadata={"category": "AI Ethics", "difficulty": "intermediate"},
        ),
    ]

    return documents


def main():
    """Run the RAG example."""
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(
        use_mlx=HAS_MLX,
        embedding_model="text-embedding-3-small",
        completion_model="gpt-3.5-turbo",
        chunk_size=600,
        chunk_overlap=100,
    )

    # Display MLX status
    if HAS_MLX:
        print("MLX acceleration is enabled")
    else:
        print("MLX acceleration is not available")

    # Load sample documents
    documents = load_sample_knowledge_base()
    print(f"Loaded {len(documents)} sample documents")

    # Add documents to the RAG system
    for doc in documents:
        rag.add_document(doc)
        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Demonstration queries
    queries = [
        "Explain how neural networks work and their components.",
        "What are the main differences between deep learning and traditional machine learning?",
        "How does the Transformer architecture work in NLP?",
        "What are the ethical concerns related to AI development?",
        "How do reinforcement learning agents balance exploration and exploitation?",
    ]

    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n\n--- Query {i}: {query} ---\n")

        # Generate response
        response, contexts = rag.generate_response(
            query=query, k=3  # Number of context chunks to retrieve
        )

        # Print retrieved contexts
        print("Retrieved context:")
        for j, ctx in enumerate(contexts, 1):
            print(f"Context {j} - {ctx['document_title']} (Score: {ctx['score']:.4f})")
            print(textwrap.fill(ctx["content"][:150] + "...", width=80))
            print()

        # Print the generated response
        print("Generated response:")
        print(textwrap.fill(response, width=80))
        print("\n" + "-" * 80)

        # Small delay to avoid rate limits
        time.sleep(1)

    # Custom query with filtering
    print("\n\n--- Custom Query with Category Filter ---\n")

    query = "Explain the concept of attention in natural language processing."
    response, contexts = rag.generate_response(
        query=query, k=2, filter={"category": "Natural Language Processing"}
    )

    # Print retrieved contexts
    print("Retrieved context (filtered by NLP category):")
    for j, ctx in enumerate(contexts, 1):
        print(f"Context {j} - {ctx['document_title']} (Score: {ctx['score']:.4f})")
        print(textwrap.fill(ctx["content"][:150] + "...", width=80))
        print()

    # Print the generated response
    print("Generated response:")
    print(textwrap.fill(response, width=80))

    # Save the index for future use
    rag.save("rag_knowledge_base.llamadb")
    print("\nRAG example completed successfully!")


if __name__ == "__main__":
    main()
