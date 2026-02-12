FROM python:3.10-slim

# System deps for OpenCV and building native extensions (psutil)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling CUDA packages
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies, skipping torch/torchvision (already installed as CPU-only)
COPY requirements.txt .
RUN grep -iv '^torch' requirements.txt > requirements-notorch.txt && \
    pip install --no-cache-dir -r requirements-notorch.txt

# Copy application code and model files
COPY . .

EXPOSE 8000 8001

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
