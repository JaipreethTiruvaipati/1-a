#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download PaddleOCR models during Docker build.
This ensures that the models are available offline when the container runs.
"""

import os
from paddleocr import PPStructure

print("Downloading PaddleOCR models...")

# Initialize PPStructure to trigger model download
# Using PP-StructureV2 for better layout analysis
engine = PPStructure(layout=True, show_log=True, structure_version='PP-StructureV2')

print("Models downloaded successfully!")

# Print the model directory for verification
model_dir = os.path.expanduser("~/.paddleocr")
print(f"Model directory: {model_dir}")

# List downloaded models
print("\nDownloaded models:")
for root, dirs, files in os.walk(model_dir):
    for file in files:
        if file.endswith(".pdmodel") or file.endswith(".pdiparams"):
            print(f" - {os.path.join(root, file)}")

print("\nModel download complete.") 