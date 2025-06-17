FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget curl \
    && pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/polymorphicshade/PolyAssistant.FramePack-Studio /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

CMD ["python3", "studio.py"]