#!/bin/bash
# Inference script to run the pipeline
# This script will be called by Docker CMD

# Ensure output directory exists
mkdir -p /output

print("Running inference...")
python3 predict.py
