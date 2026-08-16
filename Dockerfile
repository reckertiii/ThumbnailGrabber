# ------------------------------------------------------------------------------
# Production Headless Dockerfile for Thumbnail Grabber
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TG_OUTPUT_DIR=/app/captures \
    PORT=8000 \
    HOST=0.0.0.0

# Install required system runtime libraries for OpenCV, MediaPipe, and FFMPEG
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Filter out desktop GUI dependencies (PyQt6) in container for minimal image size
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v -i "pyqt" requirements.txt > /tmp/reqs-docker.txt && \
    pip install --no-cache-dir -r /tmp/reqs-docker.txt && \
    rm /tmp/reqs-docker.txt

# Copy application source code
COPY analyzer.py config.py overlay.py preference_model.py stream_processor.py ./
COPY web/ ./web/
COPY test_analyzer.py test_stream.py ./

# Create directories for models and captures
RUN mkdir -p /app/models /app/captures

# Pre-download and cache ML models during image build for offline/instant startup
RUN python -c "from analyzer import PersonQualityAnalyzer; PersonQualityAnalyzer()"

# Declare persistent volumes
VOLUME ["/app/captures", "/app/models"]

EXPOSE 8000

# Run FastAPI web server
CMD ["python", "-m", "web.server", "--host", "0.0.0.0", "--port", "8000"]
