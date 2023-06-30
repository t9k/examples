FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y libibverbs1 librdmacm1 libibumad3 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -i https://pypi.douban.com/simple/ \
    transformers \
    datasets \
    tiktoken \
    wandb

RUN apt update && apt install -y gcc && rm -rf /var/lib/apt/lists/*

USER root
WORKDIR /t9k/mnt

ARG GID=1000
ARG UID=1000
RUN groupadd --gid=$GID t9kuser && mkdir /t9k && \
  useradd -rm --create-home -d /t9k/mnt --shell /bin/bash \
  --uid=$UID --gid=$GID t9kuser

USER t9kuser
