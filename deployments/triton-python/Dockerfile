FROM nvcr.io/nvidia/tritonserver:23.05-py3

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --default-timeout=100 \
  torch \
  torchvision \
  transformers \
  Image

ENTRYPOINT [ "tritonserver", "--model-repository=/models" ]
