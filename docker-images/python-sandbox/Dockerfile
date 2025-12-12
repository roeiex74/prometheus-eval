# Prometheus-Eval Python Sandbox
# Minimal, secure Python execution environment for code evaluation
# No network access, resource-limited, non-root execution

FROM python:3.11-slim

# Install minimal essential packages
# Note: timeout utility is part of coreutils which is already installed in base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash sandbox && \
    mkdir -p /workspace && \
    chown -R sandbox:sandbox /workspace

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER sandbox

# Disable network access at runtime (enforced via docker run --network none)
# Resource limits enforced via docker run flags:
# --memory: Memory limit
# --cpus: CPU quota
# --pids-limit: Process limit

# Default command: execute Python code from stdin
CMD ["python3", "-u"]
