FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND noninteractive

RUN rm -rf /var/lib/apt/lists/* &&  rm -rf /etc/apt/sources.list.d/*\
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i s@/archive.ubuntu.com/@/mirrors.ustc.edu.cn/@g /etc/apt/sources.list && \
    apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m t9kuser
USER t9kuser
RUN conda clean --all

# Install requirements
COPY --chown=t9kuser requirements.txt /lanedet/requirements.txt
WORKDIR /lanedet
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Install LaneDet
COPY --chown=t9kuser . /lanedet
RUN pip install --no-cache-dir -e .
