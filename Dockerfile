# Multi-stage Docker build for MCP Forge
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install MCP Forge
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_DATA_ROOT=/var/lib/mcp/data \
    MCP_SCHEMA_PATH=/var/lib/mcp/schemas \
    MCP_UI_BIND=0.0.0.0:8788

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r mcp && useradd -r -g mcp mcp

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app /app
WORKDIR /app

# Create required directories
RUN mkdir -p /var/lib/mcp/data \
             /var/lib/mcp/schemas \
             /var/lib/mcp/handoffs \
             /var/log/mcp \
    && chown -R mcp:mcp /var/lib/mcp /var/log/mcp /app

# Copy schemas
COPY schemas/ /var/lib/mcp/schemas/
COPY templates/ /var/lib/mcp/templates/

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8788/api/health || exit 1

# Expose ports
EXPOSE 8788 8787 8443

# Default command
CMD ["mcp-forge", "ui", "--host", "0.0.0.0", "--port", "8788"]
