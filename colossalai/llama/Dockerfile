FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH "/usr/local/cuda/bin:$PATH"

RUN git clone -b example/llama https://ghproxy.com/https://github.com/hpcaitech/ColossalAI.git && \
    pip install --no-cache-dir --upgrade pip && \
    CUDA_EXT=1 pip install --no-cache-dir ./ColossalAI && \
    pip install --no-cache-dir -r ./ColossalAI/examples/language/llama/requirements.txt && \
    pip install --no-cache-dir xformers==0.0.16 tensorboard sentencepiece && \
    rm -rf ColossalAI

WORKDIR /t9k/mnt
