FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV NB_PREFIX=/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
  apt-get install -yq --no-install-recommends git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir packaging && \
  pip install --no-build-isolation --no-cache-dir flash-attn && \
  pip install --no-cache-dir \
  fschat==0.2.35 \
  transformers \
  accelerate \
  sentencepiece \
  xformers \
  transformers_stream_generator

COPY openai.sh .
EXPOSE 80
ENTRYPOINT [ "./openai.sh" ]
