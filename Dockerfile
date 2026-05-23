# ============================================================
# FILE     : Dockerfile
# PROJECT  : Customer Segmentation — Docker Container
# ============================================================
#
# WHAT IS DOCKER? (explain in interviews)
# ───────────────────────────────────────
# Docker packages your entire project — code, dependencies,
# Python version, system libraries — into a single "container"
# that runs identically on ANY machine.
#
# The classic problem Docker solves:
#   "It works on my laptop but not on the server."
#
# With Docker:
#   → One command builds the environment: docker build
#   → One command runs the project:       docker compose up
#   → Works the same on Windows, Mac, Linux, cloud servers
#
# WHY THIS MATTERS FOR RECRUITERS:
# ───────────────────────────────────
# Docker is listed in ~60% of data engineering job descriptions
# and ~30% of data analyst JDs. It shows you understand
# DEPLOYMENT — not just analysis.
#
# It also shows you think about your COLLEAGUES:
#   "Anyone can clone this repo and run it with one command.
#    No 'install this, configure that, it might work' instructions."
#
# WHAT THIS DOCKERFILE DOES:
# ───────────────────────────────────
# 1. Starts from a slim Python 3.11 image (~120MB vs ~1GB for full)
# 2. Installs only necessary system libraries
# 3. Copies requirements.txt and installs Python packages
# 4. Copies project code
# 5. Creates non-root user (security best practice)
# 6. Exposes port 8501 for Streamlit
# 7. Default command: run the Streamlit dashboard
#
# RECRUITER ONE-LINER:
# "I containerised the entire analytics pipeline using Docker,
#  including the Streamlit dashboard, ML pipeline, and SQL database.
#  Anyone can run the full project with one command — no environment
#  setup required."
# ============================================================

# ── Base image ───────────────────────────────────────────────
# python:3.11-slim = official Python image, minimal OS (~120MB)
# Using a pinned version ensures reproducibility
FROM python:3.11-slim

# ── Build-time metadata ──────────────────────────────────────
LABEL project="rfm-customer-segmentation"
LABEL description="Customer Segmentation: RFM + Clustering + A/B Testing"
LABEL author="Your Name"

# ── Environment variables ────────────────────────────────────
# PYTHONUNBUFFERED=1 : Python logs appear immediately (not buffered)
# PYTHONDONTWRITEBYTECODE=1 : Don't create .pyc cache files in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Working directory inside the container ───────────────────
WORKDIR /app

# ── System dependencies ──────────────────────────────────────
# These are OS-level packages needed by Python libraries:
#   build-essential : needed to compile some Python C extensions
#   libgomp1        : needed by scikit-learn (OpenMP parallelism)
#   sqlite3         : our database engine
#   curl            : for health checks
# We clean up the apt cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────
# Copy requirements first (separate layer = Docker cache optimisation)
# If requirements.txt hasn't changed, Docker reuses the cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project code ────────────────────────────────────────
COPY . .

# ── Create necessary directories ─────────────────────────────
RUN mkdir -p data/raw data/processed reports logs \
             data/processed/.gitkeep

# ── Create non-root user (security best practice) ────────────
# Running as root inside containers is a security risk
RUN useradd --create-home --shell /bin/bash rfmuser && \
    chown -R rfmuser:rfmuser /app
USER rfmuser

# ── Expose Streamlit port ────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────
# Docker will ping this endpoint every 30s to verify the app is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command ──────────────────────────────────────────
# What runs when you do: docker run rfm-segmentation
# --server.address=0.0.0.0 : accept connections from outside container
# --server.port=8501        : standard Streamlit port
# --server.headless=true    : no browser auto-open in container
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
