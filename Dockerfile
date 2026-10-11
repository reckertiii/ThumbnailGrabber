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

# ML weights are baked into the image at TG_MODELS_DIR. This is deliberately NOT
# /app/models: that path is a declared volume, so a bind mount from a fresh host
# directory hides the baked weights, forcing a re-download on every start - and
# failing the container outright on a host with no outbound internet.
ENV TG_MODELS_DIR=/opt/tg-models
# User ratings do need to survive restarts, so they stay on the mounted volume.
ENV TG_DATA_DIR=/app/models

# System runtime libraries for OpenCV, MediaPipe, FFMPEG and timezones.
#
# libgles2/libegl1 matter more than they look: MediaPipe's task runners
# resolve a GL context even when inference is on CPU. Without them all three
# landmarkers fail with 'libGLESv2.so.2: cannot open shared object file' and
# are skipped with only a warning, leaving scoring to the plain face box --
# no 478-point mesh, no blendshapes, no shoulder line, no hands.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libegl1 \
    ffmpeg \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Filter out desktop GUI dependencies (PyQt6) in container for minimal image size
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v -i "pyqt" requirements.txt > /tmp/reqs-docker.txt && \
    pip install --no-cache-dir -r /tmp/reqs-docker.txt && \
    rm /tmp/reqs-docker.txt

# Copy application source code. Tests are intentionally not shipped in the runtime image.
# main.py is included for the headless CLI scanner; it only imports PyQt6 inside the
# GUI branch, so it runs fine in this image without the desktop dependency.
COPY analyzer.py config.py overlay.py preference_model.py stream_processor.py video_scanner.py ai_judge.py camera_probe.py camera_store.py main.py ./
COPY web/ ./web/

# Create directories for baked models, persistent user data, and captures
RUN mkdir -p /opt/tg-models /app/models /app/captures

# Pre-download and cache ML models during image build for offline/instant startup.
# The build fails here rather than shipping an image that cannot start offline.
RUN python -c "from analyzer import PersonQualityAnalyzer; PersonQualityAnalyzer()" && \
    test -s /opt/tg-models/yunet.onnx && \
    test -s /opt/tg-models/face_landmarker.task

# Declare persistent volumes
VOLUME ["/app/captures", "/app/models"]

EXPOSE 8000

# Run FastAPI web server
CMD ["python", "-m", "web.server", "--host", "0.0.0.0", "--port", "8000"]
