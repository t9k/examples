FROM nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3

RUN pip3 install --no-cache-dir --upgrade pip && \
  pip3 install --no-cache-dir --default-timeout=100 torch==2.1.2

COPY launch_triton_server.py .
