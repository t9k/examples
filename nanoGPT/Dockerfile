FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# TODO: Review if we really need all these libs.
RUN apt update && apt install -y libibverbs1 librdmacm1 libibumad3 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -i https://pypi.douban.com/simple/ \
    transformers \
    datasets \
    tiktoken \
    wandb

ARG GID=1000
ARG UID=1000
RUN mkdir /t9k && \
    groupadd --gid=$GID t9kuser && \
    useradd -rm --create-home -d /t9k/mnt --shell /bin/bash --uid=$UID --gid=$GID t9kuser
USER t9kuser
WORKDIR /t9k/mnt
