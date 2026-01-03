"""
DevOps tools for the MLOps pipeline.
Implements Dockerfile generation and container utilities.
"""

from langchain_core.tools import tool


@tool
def generate_dockerfile(
    app_name: str = "mlops-app",
    entry_point: str = "main.py",
    port: int = 8000,
    python_version: str = "3.9",
) -> str:
    """
    Generate an optimized Dockerfile for a Python ML application.
    
    Args:
        app_name: Name of the application (used for labeling).
        entry_point: Main Python file to execute.
        port: Port to expose for the application.
        python_version: Python version to use (default 3.9).
        
    Returns:
        String containing the complete Dockerfile content.
    """
    dockerfile = f'''# =============================================================================
# Dockerfile for {app_name}
# Optimized multi-stage build for Python {python_version} ML applications
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
      app.version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 appuser \\
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "{entry_point}"]
'''
    
    return dockerfile

