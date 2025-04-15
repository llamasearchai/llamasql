FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy example code
COPY rag_example.py .

# Copy any other necessary files
COPY .env.example .env

# Command to run the example
CMD ["python", "rag_example.py"] 