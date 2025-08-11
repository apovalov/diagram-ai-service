# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /build

# Copy dependency files for layer caching
COPY pyproject.toml ./

# Install dependencies WITH cache mount for faster rebuilds
# This uses Docker BuildKit cache mounts to persist pip cache between builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Test stage - includes dev dependencies
FROM builder as test

# Install dev dependencies for testing
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[dev]"

# Install runtime dependencies for test stage too
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the application code
COPY ./app /app

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set the working directory in the container
WORKDIR /project

# Copy the entire project structure for proper package installation
COPY pyproject.toml ./
COPY ./app ./app

# Install the package in the production environment (this makes 'app' module discoverable)
RUN pip install -e .

# Change back to app directory for runtime
WORKDIR /app

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /project /app
USER appuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]