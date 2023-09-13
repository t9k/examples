FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
  apt-get install -yq --no-install-recommends openssh-server pdsh && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    deepspeed>=0.10.2 \
    transformers>=4.31.0 \
    datasets>=2.12.0 \
    accelerate>=0.21.0 \
    peft==0.4.0 \
    git+https://hub.nuaa.cf/huggingface/trl.git \
    scipy \
    sentencepiece \
    tiktoken \
    xformers \
    jieba \
    rouge-chinese \
    nltk \
    gradio>=3.36.0 \
    uvicorn \
    pydantic==1.10.11 \
    fastapi==0.95.1 \
    sse-starlette \
    matplotlib

WORKDIR /t9k/mnt
