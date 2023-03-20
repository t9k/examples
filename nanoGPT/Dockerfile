FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN pip install --no-cache-dir -i https://pypi.douban.com/simple/ \
    transformers \
    datasets \
    tiktoken \
    wandb

USER root
WORKDIR /t9k/mnt
