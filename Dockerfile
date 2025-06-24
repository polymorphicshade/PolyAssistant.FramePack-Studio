ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG CUDA_VERSION

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget curl && \
    pip3 install --upgrade pip

WORKDIR /app

# This allows caching pip install if only code has changed
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
RUN export CUDA_SHORT_VERSION=$(echo "${CUDA_VERSION}" | sed 's/\.//g') && \
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT_VERSION}"

# Copy the source code to /app
COPY . .

EXPOSE 7860

CMD ["python3", "studio.py"]
