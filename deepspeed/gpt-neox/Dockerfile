FROM nvcr.io/nvidia/pytorch:23.02-py3

# DeepSpeed basic packages
RUN apt-get update && apt-get install -yq --no-install-recommends \
    openssh-server \
    pdsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    accelerate \
    datasets \
    sentencepiece \
    transformers \
    tabulate \
    evaluate \
    deepspeed==0.10.0

# Packages for GPT NeoX
RUN pip install --no-cache-dir  \
    best_download \
    ftfy>=6.0.1 \
    huggingface_hub>=0.11.0 \
    lm_eval>=0.3.0 \
    mpi4py>=3.0.3 \
    numpy>=1.22.0 \
    pybind11>=2.6.2 \
    regex \
    sentencepiece \
    six \
    tiktoken>=0.1.2 \
    tokenizers>=0.12.1 \
    tensorboard==2.13.0 \
    git+https://github.com/EleutherAI/lm_dataformat.git@4eec05349977071bf67fc072290b95e31c8dd836

# Download and modify GPT NeoX training script.
RUN git clone https://github.com/EleutherAI/gpt-neox /t9k/workspace/gpt-neox
COPY arguments.py /t9k/workspace/gpt-neox/megatron/neox_arguments/arguments.py
COPY train.py /t9k/workspace/gpt-neox/train.py
COPY corpora.py /t9k/workspace/gpt-neox/tools/corpora.py
COPY configs/t9k /t9k/workspace/gpt-neox/configs/t9k

WORKDIR /t9k/workspace/gpt-neox

RUN mkdir /run/sshd
RUN chown root:root /usr/lib
