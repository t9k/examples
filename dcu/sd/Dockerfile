FROM image.sourcefind.cn:5000/dcu/admin/base/custom:stablediffusion

RUN pip install --no-cache-dir --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.42.3 \
    diffusers==0.29.2 \
    accelerate==0.32.1 \
    huggingface_hub==0.23.4 \
    tokenizers==0.19.1 \
    safetensors==0.4.3
