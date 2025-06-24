ARG CUDA_VERSION=12.9

FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget curl && \
    pip3 install --upgrade pip

WORKDIR /app

# Create volume mount directories
RUN mkdir -p /workspace/input /workspace/output

# This allows caching pip install if only code has changed
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy the source code to /app
COPY . .

CMD ["python3", "studio.py", "--input", "/workspace/input", "--output", "/workspace/output"]
