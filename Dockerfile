FROM --platform=linux/amd64 python:3.10-slim-bullseye AS builder

WORKDIR /app

# Install dependencies needed for PaddleOCR and download models
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
    libssl1.1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .
RUN python download_models.py

# -------- FINAL STAGE --------
FROM --platform=linux/amd64 python:3.10-slim-bullseye

WORKDIR /app

# Install runtime dependencies (including libgomp1 here as well)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
    libssl1.1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed packages and models from the builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.paddleocr/ /root/.paddleocr/

COPY main.py .

# Create input/output folders and ensure permissions
RUN mkdir -p /app/input /app/output && chmod -R 777 /app/input /app/output

CMD ["python", "main.py"]
