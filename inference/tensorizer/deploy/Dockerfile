FROM t9kpublic/coreweave-ml-containers-torch:bb02bee-base-cuda11.8.0-torch2.0.0-vision0.15.1-audio2.0.1

RUN mkdir -p /transformer/
WORKDIR /transformer

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY kserve_api.py .
COPY load_model.py .

ENTRYPOINT ["python", "kserve_api.py"]
