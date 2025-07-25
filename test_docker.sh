#!/bin/bash

# Test script for building and running the Docker container

# Create input and output directories if they don't exist
mkdir -p input output

echo "Building Docker image..."
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .

echo "Running Docker container..."
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none pdf-outline-extractor:latest

echo "Done!"
echo "Check the output directory for JSON files." 