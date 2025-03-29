#!/usr/bin/env python
"""
Web Interface for RAG with LlamaDB

This script creates a simple web interface for the RAG example,
allowing users to ask questions and see responses with source context.

Requirements:
- Flask
- All requirements from rag_example.py
"""

import os
import time
import json
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
import numpy as np

# Import RAG components
from rag_example import RAGSystem, Document, load_sample_knowledge_base

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize the RAG system
print("Initializing RAG system...")
try:
    rag = RAGSystem(
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        completion_model=os.getenv("COMPLETION_MODEL", "gpt-3.5-turbo"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        max_tokens=int(os.getenv("MAX_TOKENS", "600"))
    )
    print("RAG system initialized successfully.")
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    rag = None

# Check if knowledge base exists and load it, otherwise build it
KNOWLEDGE_BASE_PATH = "knowledge_base.llamadb"

if os.path.exists(KNOWLEDGE_BASE_PATH):
    try:
        print(f"Loading existing knowledge base from {KNOWLEDGE_BASE_PATH}...")
        from llamadb.core import VectorIndex
        rag.index = VectorIndex.load(KNOWLEDGE_BASE_PATH)
        print(f"Knowledge base loaded with {rag.index.count()} chunks.")
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        # Will build a new knowledge base below
else:
    print("No existing knowledge base found. Creating a new one...")


# Load documents if needed
def initialize_knowledge_base():
    if rag.index.count() == 0:
        print("Loading sample documents...")
        documents = load_sample_knowledge_base()
        print(f"Loaded {len(documents)} sample documents.")
        
        print("Processing and indexing documents...")
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}: {doc.title}")
            rag.add_document(doc)
            time.sleep(0.5)  # Small delay to avoid rate limits
        
        print(f"Indexing complete. Saving knowledge base to {KNOWLEDGE_BASE_PATH}...")
        rag.save(KNOWLEDGE_BASE_PATH)
        print("Knowledge base saved successfully.")


# Routes
@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Process a query and return results."""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    filter_data = data.get('filter', None)
    
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        # Generate response
        response, contexts = rag.generate_response(
            query=query_text,
            k=data.get('k', 3),
            filter=filter_data
        )
        
        # Format contexts
        formatted_contexts = []
        for ctx in contexts:
            formatted_contexts.append({
                'document_title': ctx['document_title'],
                'content': ctx['content'][:300] + '...' if len(ctx['content']) > 300 else ctx['content'],
                'score': round(ctx['score'], 4)
            })
        
        return jsonify({
            'response': response,
            'contexts': formatted_contexts
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/categories')
def get_categories():
    """Get the list of available document categories."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        # Query for all documents and collect unique categories
        results = rag.index.search(
            query_vector=np.zeros(rag.dimension, dtype=np.float32),
            k=rag.index.count(),
            include_vectors=False
        )
        
        categories = set()
        for result in results:
            if 'category' in result.metadata:
                categories.add(result.metadata['category'])
        
        return jsonify(list(categories))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get statistics about the knowledge base."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        total_chunks = rag.index.count()
        
        # Get document counts by document ID
        results = rag.index.search(
            query_vector=np.zeros(rag.dimension, dtype=np.float32),
            k=total_chunks,
            include_vectors=False
        )
        
        document_ids = set()
        categories = {}
        
        for result in results:
            if 'document_id' in result.metadata:
                document_ids.add(result.metadata['document_id'])
            
            if 'category' in result.metadata:
                category = result.metadata['category']
                if category in categories:
                    categories[category] += 1
                else:
                    categories[category] = 1
        
        return jsonify({
            'total_chunks': total_chunks,
            'total_documents': len(document_ids),
            'categories': categories
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Create templates directory and HTML template
@app.route('/create_templates')
def create_templates():
    """Create the templates directory and index.html file."""
    os.makedirs('templates', exist_ok=True)
    
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaDB RAG Interface</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eee;
        }
        h1 {
            color: #166534;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 2rem;
        }
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        .main-content {
            display: flex;
            flex-direction: column;
        }
        .query-box {
            display: flex;
            margin-bottom: 1rem;
        }
        #query-input {
            flex-grow: 1;
            padding: 0.8rem;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 4px 0 0 4px;
        }
        #submit-btn {
            padding: 0.8rem 1.5rem;
            background-color: #15803d;
            color: white;
            border: none;
            border-radius: 0 4px 4px 0;
            cursor: pointer;
            font-size: 1rem;
        }
        #submit-btn:hover {
            background-color: #166534;
        }
        .response-area {
            margin-top: 1rem;
            min-height: 200px;
        }
        .response {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid #15803d;
            margin-bottom: 1.5rem;
        }
        .sources {
            margin-top: 1.5rem;
        }
        .source-item {
            background-color: #f1f5f9;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .source-title {
            font-weight: bold;
            color: #334155;
            margin-bottom: 0.25rem;
        }
        .source-content {
            color: #64748b;
        }
        .sidebar {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 4px;
        }
        .stats {
            margin-bottom: 2rem;
        }
        .filter-section h3 {
            margin-bottom: 0.5rem;
        }
        select {
            width: 100%;
            padding: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        .loading {
            display: none;
            margin: 1rem 0;
            color: #666;
        }
        .spinner {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid rgba(0, 0, 0, 0.1);
            border-left-color: #15803d;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <h1>LlamaDB RAG Interface</h1>
        <p>Ask questions about AI, machine learning, and related topics</p>
    </header>
    
    <div class="container">
        <div class="main-content">
            <div class="query-box">
                <input type="text" id="query-input" placeholder="Ask a question about AI and machine learning...">
                <button id="submit-btn">Ask</button>
            </div>
            
            <div class="loading">
                <div class="spinner"></div> Generating response...
            </div>
            
            <div class="response-area" id="response-area">
                <!-- Response will be inserted here -->
            </div>
        </div>
        
        <div class="sidebar">
            <div class="stats">
                <h3>Knowledge Base Stats</h3>
                <div id="stats-content">Loading...</div>
            </div>
            
            <div class="filter-section">
                <h3>Filter by Category</h3>
                <select id="category-filter">
                    <option value="">All Categories</option>
                    <!-- Categories will be inserted here -->
                </select>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const queryInput = document.getElementById('query-input');
            const submitBtn = document.getElementById('submit-btn');
            const responseArea = document.getElementById('response-area');
            const loadingDiv = document.querySelector('.loading');
            const statsContent = document.getElementById('stats-content');
            const categoryFilter = document.getElementById('category-filter');
            
            // Load stats
            fetchStats();
            
            // Load categories
            fetchCategories();
            
            // Event listeners
            submitBtn.addEventListener('click', submitQuery);
            queryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    submitQuery();
                }
            });
            
            categoryFilter.addEventListener('change', function() {
                // Store the selected category in session storage
                sessionStorage.setItem('selectedCategory', categoryFilter.value);
            });
            
            // Check for stored category
            const storedCategory = sessionStorage.getItem('selectedCategory');
            if (storedCategory) {
                categoryFilter.value = storedCategory;
            }
            
            function submitQuery() {
                const query = queryInput.value.trim();
                if (!query) return;
                
                // Show loading indicator
                loadingDiv.style.display = 'block';
                responseArea.innerHTML = '';
                
                // Prepare request data
                const requestData = {
                    query: query,
                    k: 3
                };
                
                // Add filter if category is selected
                const selectedCategory = categoryFilter.value;
                if (selectedCategory) {
                    requestData.filter = { "category": selectedCategory };
                }
                
                // Send the query
                fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData),
                })
                .then(response => response.json())
                .then(data => {
                    loadingDiv.style.display = 'none';
                    
                    if (data.error) {
                        responseArea.innerHTML = `<div class="response"><p>Error: ${data.error}</p></div>`;
                        return;
                    }
                    
                    // Format and display the response
                    const responseHtml = `
                        <div class="response">
                            <p>${formatResponse(data.response)}</p>
                        </div>
                        
                        <div class="sources">
                            <h3>Sources:</h3>
                            ${data.contexts.map(context => `
                                <div class="source-item">
                                    <div class="source-title">${context.document_title} (Score: ${context.score})</div>
                                    <div class="source-content">${context.content}</div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                    
                    responseArea.innerHTML = responseHtml;
                })
                .catch(error => {
                    loadingDiv.style.display = 'none';
                    responseArea.innerHTML = `<div class="response"><p>Error: ${error.message}</p></div>`;
                });
            }
            
            function fetchStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            statsContent.textContent = `Error: ${data.error}`;
                            return;
                        }
                        
                        statsContent.innerHTML = `
                            <p><strong>Documents:</strong> ${data.total_documents}</p>
                            <p><strong>Chunks:</strong> ${data.total_chunks}</p>
                            <p><strong>Categories:</strong></p>
                            <ul>
                                ${Object.entries(data.categories).map(([category, count]) => 
                                    `<li>${category}: ${count}</li>`).join('')}
                            </ul>
                        `;
                    })
                    .catch(error => {
                        statsContent.textContent = `Error loading stats: ${error.message}`;
                    });
            }
            
            function fetchCategories() {
                fetch('/api/categories')
                    .then(response => response.json())
                    .then(categories => {
                        if (Array.isArray(categories)) {
                            const options = categories.map(category => 
                                `<option value="${category}">${category}</option>`
                            ).join('');
                            
                            categoryFilter.innerHTML = `<option value="">All Categories</option>${options}`;
                            
                            // Restore selected category if it exists
                            if (storedCategory) {
                                categoryFilter.value = storedCategory;
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error loading categories:', error);
                    });
            }
            
            function formatResponse(text) {
                // Add paragraph breaks
                return text.replace(/\n\n/g, '</p><p>');
            }
        });
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    return "Templates created successfully!"


if __name__ == '__main__':
    # Create templates if not exists
    if not os.path.exists('templates'):
        create_templates()
    
    # Initialize the knowledge base if needed
    if rag and rag.index.count() == 0:
        initialize_knowledge_base()
    
    # Start the server
    app.run(host='0.0.0.0', port=8000, debug=os.getenv('DEBUG', 'False').lower() == 'true') 