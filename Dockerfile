FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget curl \
 && pip3 install --upgrade pip

RUN git clone https://github.com/colinurbs/FramePack-Studio.git /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN mkdir -p /workspace/input /workspace/output

CMD ["python3", "app.py", "--input", "/workspace/input", "--output", "/workspace/output"]
