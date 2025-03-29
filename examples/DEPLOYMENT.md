# Deploying LlamaDB RAG in Production

This guide provides instructions for deploying the LlamaDB RAG example and web application in production environments. We'll cover different deployment options and best practices for ensuring your application is reliable, secure, and performant.

## Deployment Options

### 1. Docker Deployment

Docker is recommended for most production deployments due to its isolation, portability, and reproducibility.

#### Prerequisites

- Docker and Docker Compose installed
- Git installed (to clone the repository)
- A server with sufficient resources (2+ CPU cores, 4GB+ RAM)

#### Deployment Steps

1. Clone the repository and navigate to the examples directory:

```bash
git clone https://github.com/llamasql/llamadb.git
cd llamadb/examples
```

2. Create a production `.env` file:

```bash
cp .env.example .env
```

3. Edit the `.env` file with your production configuration:

```
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
COMPLETION_MODEL=gpt-3.5-turbo
CHUNK_SIZE=600
CHUNK_OVERLAP=100
MAX_TOKENS=600
DEBUG=False
```

4. (Optional) Edit the `docker-compose.web.yml` file to customize the deployment:

```yaml
version: '3'

services:
  rag-web-app:
    build:
      context: .
      dockerfile: Dockerfile.web
    volumes:
      - ./data:/app/data
    env_file:
      - .env
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
    restart: unless-stopped  # Add for automatic restarts
    # Add healthcheck for monitoring
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
```

5. Start the application in detached mode:

```bash
docker-compose -f docker-compose.web.yml up -d --build
```

6. Monitor the logs:

```bash
docker-compose -f docker-compose.web.yml logs -f
```

### 2. Kubernetes Deployment

For larger, scalable deployments, Kubernetes offers more advanced orchestration capabilities.

#### Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or self-hosted)
- kubectl CLI tool
- Helm (optional, for package management)

#### Basic Kubernetes Deployment

1. Create a Kubernetes deployment YAML file `llamadb-rag-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamadb-rag
  labels:
    app: llamadb-rag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llamadb-rag
  template:
    metadata:
      labels:
        app: llamadb-rag
    spec:
      containers:
      - name: llamadb-rag
        image: llamadb-rag:latest  # Replace with your actual image
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: api-key
        - name: DEBUG
          value: "False"
        - name: EMBEDDING_MODEL
          value: "text-embedding-3-small"
        - name: COMPLETION_MODEL
          value: "gpt-3.5-turbo"
        resources:
          limits:
            memory: "4Gi"
            cpu: "1"
          requests:
            memory: "2Gi"
            cpu: "0.5"
        volumeMounts:
        - name: llamadb-data
          mountPath: /app/data
      volumes:
      - name: llamadb-data
        persistentVolumeClaim:
          claimName: llamadb-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llamadb-rag-service
spec:
  selector:
    app: llamadb-rag
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamadb-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

2. Create a Secret for the OpenAI API key:

```bash
kubectl create secret generic openai-credentials --from-literal=api-key=your-openai-api-key
```

3. Apply the deployment:

```bash
kubectl apply -f llamadb-rag-deployment.yaml
```

4. Create an Ingress or expose the service as needed for your Kubernetes setup.

### 3. Cloud Platform-Specific Deployments

#### AWS Elastic Beanstalk

1. Install the EB CLI:

```bash
pip install awsebcli
```

2. Initialize the EB application:

```bash
cd llamadb/examples
eb init -p docker llamadb-rag
```

3. Deploy the application:

```bash
eb create llamadb-rag-env
```

#### Google Cloud Run

1. Build and push your Docker image:

```bash
gcloud builds submit --tag gcr.io/your-project/llamadb-rag
```

2. Deploy the image to Cloud Run:

```bash
gcloud run deploy llamadb-rag \
  --image gcr.io/your-project/llamadb-rag \
  --platform managed \
  --memory 4Gi \
  --set-env-vars="OPENAI_API_KEY=your-api-key,DEBUG=False"
```

## Production Best Practices

### Security Considerations

1. **API Key Management**: Never hardcode API keys in your application code or Docker images. Use environment variables or secret management solutions.

2. **Network Security**: Limit access to your application by using firewalls, VPCs, or network policies.

3. **Authentication**: Consider adding authentication to your web application:

```python
# Example using Flask-Login
from flask_login import LoginManager, login_required

@app.route('/')
@login_required
def index():
    return render_template('index.html')
```

4. **HTTPS**: Always use HTTPS in production. You can use services like Cloudflare, Let's Encrypt, or your cloud provider's SSL options.

### Performance Optimization

1. **Caching Responses**: Implement caching for frequent queries:

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/query', methods=['POST'])
@cache.memoize(timeout=3600)  # Cache for 1 hour
def cached_query():
    # Process query
    return response
```

2. **Precomputing Embeddings**: For known content, precompute and store embeddings to avoid API calls:

```python
# Script to precompute embeddings
def precompute_embeddings(documents):
    for doc in documents:
        doc.embedding = generate_embedding(doc.content)
        save_to_database(doc)
```

3. **Horizontal Scaling**: For high-traffic applications, deploy multiple instances behind a load balancer.

### Monitoring and Logging

1. **Structured Logging**: Implement structured logging for easier analysis:

```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger("llamadb-rag")

# Log in structured format
logger.info(json.dumps({
    "event": "query_processed",
    "query": query_text,
    "response_time_ms": processing_time,
    "num_contexts": len(contexts)
}))
```

2. **Health Checks**: Implement a health check endpoint:

```python
@app.route('/health')
def health_check():
    # Check if rag system is initialized
    if not rag:
        return jsonify({"status": "error", "message": "RAG system not initialized"}), 500
    
    return jsonify({"status": "healthy", "documents": rag.index.count()})
```

3. **Metrics Collection**: Consider adding Prometheus metrics:

```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info("llamadb_rag_app_info", "Application info", version="1.0.0")

# Custom metrics
query_requests = metrics.counter(
    'llamadb_query_requests_total', 'Number of query requests'
)
```

### Cost Optimization

1. **Embedding Model Selection**: Choose the most cost-effective embedding model for your needs.

2. **Batching Requests**: Batch embedding generation to reduce API costs:

```python
def batch_generate_embeddings(texts, batch_size=20):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai_client.embeddings.create(
            model=embedding_model,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
```

3. **Caching Strategy**: Implement a tiered caching strategy to reduce redundant API calls.

## Handling Production Data

### Data Loading Strategies

1. **Incremental Loading**: Load and index data incrementally:

```python
def incremental_load(new_documents):
    for doc in new_documents:
        rag.add_document(doc)
    rag.save("knowledge_base.llamadb")
```

2. **Scheduled Updates**: Set up scheduled jobs to update the knowledge base:

```python
# In a cron job or scheduler
def update_knowledge_base():
    new_docs = fetch_new_documents_since_last_update()
    incremental_load(new_docs)
```

### Data Persistence

1. **Backup Strategy**: Regularly back up your vector index:

```bash
# In a cron job
cp /app/data/knowledge_base.llamadb /backups/knowledge_base_$(date +%Y%m%d).llamadb
```

2. **Versioning**: Implement versioning for your knowledge base:

```python
# Save with version info
rag.save(f"knowledge_base_v{version}.llamadb")

# Load specific version
rag.index = VectorIndex.load(f"knowledge_base_v{version}.llamadb")
```

## Example Production Stack

A complete production stack might include:

- **Application**: LlamaDB RAG web application in Docker
- **API Gateway**: Nginx or Traefik for routing and SSL termination
- **Authentication**: OAuth2 provider or JWT-based auth
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Monitoring**: Prometheus and Grafana
- **CDN**: Cloudflare or similar for edge caching
- **Scheduled Jobs**: Airflow or cron for data updates
- **Backup**: Automated backup to object storage (S3, GCS)

## Troubleshooting Production Issues

### Common Issues

1. **Memory Leaks**: Monitor memory usage and implement proper cleanup.

2. **API Rate Limits**: Implement rate limiting and retries:

```python
def generate_embedding_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return openai_client.embeddings.create(
                model=embedding_model,
                input=text
            ).data[0].embedding
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
```

3. **Database Locks**: Implement proper locking mechanisms for concurrent access.

## Conclusion

This deployment guide should help you successfully deploy the LlamaDB RAG application in a production environment. Adapt these recommendations to your specific needs and infrastructure requirements. Remember to test thoroughly in a staging environment before deploying to production. 