# Base image with CUDA and Python 3.10+
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (compatible with CUDA 12.1)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python packages
RUN pip install --no-cache-dir \
    flask \
    transformers \
    accelerate \
    bitsandbytes \
    safetensors \
    sentencepiece \
    protobuf

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY app.py .
COPY templates/ ./templates
COPY static/ ./static
COPY merged_model/ ./merged_model/

# Expose port (Flask default)
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]
