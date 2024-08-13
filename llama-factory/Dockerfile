FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN apt-get update && \
  apt-get install -yq --no-install-recommends openssh-server pdsh git libaio-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
  cd LLaMA-Factory && \
  pip install --no-cache-dir --upgrade pip && \
  pip install ".[torch,metrics,deepspeed,bitsandbytes]" && \
  cd .. && \
  rm -rf LLaMA-Factory

RUN mkdir /run/sshd
RUN chown root:root /usr/lib

ARG GID=1000
ARG UID=1000
RUN groupadd --gid=$GID t9kuser && mkdir /t9k && \
    useradd -rm --create-home -d /t9k/mnt --shell /bin/bash \
    --uid=$UID --gid=$GID t9kuser
USER t9kuser
