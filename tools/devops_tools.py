"""
DevOps tools for the MLOps pipeline.
Implements Dockerfile generation and container utilities.
"""

from langchain_core.tools import tool


@tool
def generate_dockerfile(
    app_name: str = "mlops-app",
    entry_point: str = "serve.py",
    port: int = 8000,
    python_version: str = "3.11",
    use_uvicorn: bool = True,
) -> str:
    """
    Generate an optimized Dockerfile for a Python ML application with FastAPI.
    
    Args:
        app_name: Name of the application (used for labeling).
        entry_point: Main Python file to execute (default: serve.py).
        port: Port to expose for the application.
        python_version: Python version to use (default 3.11).
        use_uvicorn: Whether to use uvicorn for serving (default True for FastAPI).
        
    Returns:
        String containing the complete Dockerfile content.
    """
    # Extract module name from entry point (serve.py -> serve)
    module_name = entry_point.replace(".py", "")
    
    # Command based on whether using uvicorn or plain python
    if use_uvicorn:
        cmd = f'["uvicorn", "{module_name}:app", "--host", "0.0.0.0", "--port", "{port}"]'
        healthcheck_cmd = f'curl --fail http://localhost:{port}/health || exit 1'
    else:
        cmd = f'["python", "{entry_point}"]'
        healthcheck_cmd = 'python -c "import sys; sys.exit(0)" || exit 1'
    
    dockerfile = f'''# =============================================================================
# Dockerfile for {app_name}
# Optimized multi-stage build for Python {python_version} ML applications
# FastAPI + Uvicorn production setup
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:{python_version}-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:{python_version}-slim AS runtime

# Labels
LABEL maintainer="MLOps Pipeline" \\
      app.name="{app_name}" \\
      app.version="1.0.0" \\
      app.framework="FastAPI"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    PATH="/home/appuser/.local/bin:$PATH" \\
    # FastAPI/Uvicorn settings
    UVICORN_HOST=0.0.0.0 \\
    UVICORN_PORT={port}

# Install curl for healthcheck (minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser \\
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Create models directory (for model files)
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE {port}

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD {healthcheck_cmd}

# Default command - run with uvicorn
CMD {cmd}
'''
    
    return dockerfile

