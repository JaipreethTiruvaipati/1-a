FROM --platform=linux/amd64 python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script to download models
COPY download_models.py .

# Run the script to download models
RUN python download_models.py

# Final stage
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models from the builder stage
COPY --from=builder /root/.paddleocr/ /root/.paddleocr/

# Copy application code
COPY main.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Run the application
CMD ["python", "main.py"] 