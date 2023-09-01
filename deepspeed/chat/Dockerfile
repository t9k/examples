FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && \
  apt-get install -yq --no-install-recommends openssh-server pdsh && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    accelerate>=0.15.0 \
    datasets>=2.8.0 \
    deepspeed==0.10.2 \
    evaluate \
    protobuf==3.20.3 \
    sentencepiece \
    transformers==4.31.0 \
    tabulate \
    tensorboard

RUN mkdir /run/sshd
RUN chown root:root /usr/lib
