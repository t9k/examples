FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-get update && \
  apt-get install -yq --no-install-recommends openssh-server pdsh && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    accelerate \
    datasets \
    sentencepiece \
    # require update: https://github.com/microsoft/DeepSpeed/issues/3156
    deepspeed==0.9.0 \
    transformers==4.30.2 \
    tabulate \
    evaluate

RUN mkdir /run/sshd
RUN chown root:root /usr/lib
