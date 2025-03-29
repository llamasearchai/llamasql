FROM python:3.11-slim AS builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
WORKDIR /app
COPY . .

# Install build dependencies
RUN pip install --upgrade pip wheel setuptools-rust maturin

# Build and install the package
RUN pip install -e ".[api]"

# Build the Rust extensions
RUN cd rust_extensions/llamadb_core && maturin develop

# Second stage: runtime
FROM python:3.11-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy necessary files
COPY --from=builder /app/python /app/python
COPY --from=builder /app/README.md /app/
COPY --from=builder /app/LICENSE /app/

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV LLAMADB_ENV="production"

# Expose API port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["llamadb"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"] 