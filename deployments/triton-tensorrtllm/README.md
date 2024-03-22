# Triton（TensorRT-LLM 后端）

[Triton Inference Server](https://github.com/triton-inference-server/server) 是一个开源的推理服务软件，旨在简化 AI 推理流程。Triton 使用户能够部署多种深度学习框架的模型，包括 TensorRT、TensorFlow、PyTorch、ONNX、OpenVINO 等。Triton 为许多查询类型提供了优化的性能，包括实时、批处理、集成和音频/视频流。

本示例使用 MLService，以及 Triton 推理服务器和它的 TensorRT-LLM 后端部署一个 LLM 推理服务。

## 使用方法

在项目中创建一个名为 `triton-tensorrtllm`、大小 100 GiB 以上的 PVC，然后创建一个同样名为 `triton-tensorrtllm` 的 Notebook 挂载该 PVC，镜像选择带有 sudo 权限的 PyTorch 2.0 的类型，模板选择分配一个共享 NVIDIA GPU 的类型，并且分配的 GPU 需要有至少 16GB 的显存（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

然后从 Hugging Face Hub 或魔搭社区下载要部署的模型，这里以 <a target="_blank" rel="noopener noreferrer" href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">Llama-2-7b-chat-hf</a> 模型为例：

```bash
# 方法 1：如果可以直接访问 huggingface
# 需要登录
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir Llama-2-7b-chat-hf --local-dir-use-symlinks False

# 方法 2：对于国内用户，访问 modelscope 网络连通性更好
pip install modelscope
python -c \
  "from modelscope import snapshot_download; snapshot_download('shakechen/Llama-2-7b-chat-hf')"
mv .cache/modelscope/hub/shakechen/Llama-2-7b-chat-hf .
```

## 创建模型仓库

首先安装 TensorRT-LLM：

```bash
sudo apt-get update && sudo apt-get -y install openmpi-bin libopenmpi-dev  # password: tensorstack
sudo rm /opt/conda/compiler_compat/ld  # a workaround to build mpi4py within a conda env using an external MPI
pip install tensorrt_llm==0.8.0 --extra-index-url https://pypi.nvidia.com
```

然后克隆 <a target="_blank" rel="noopener noreferrer" href="https://github.com/NVIDIA/TensorRT-LLM">`NVIDIA/TensorRT-LLM`</a> 仓库，利用其中的 LLaMA 示例代码构建 TensorRT 引擎：

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM && git reset --hard 655524d
cd examples/llama/
python convert_checkpoint.py --model_dir ~/Llama-2-7b-chat-hf \
    --output_dir ~/Llama-2-7b-chat-hf/tllm-checkpoint-fp16-1gpu \
    --dtype float16
trtllm-build --checkpoint_dir ~/Llama-2-7b-chat-hf/tllm-checkpoint-fp16-1gpu \
    --output_dir ~/engines/1gpu \
    --gemm_plugin float16  # ~14GB GPU memory
```

克隆 <a target="_blank" rel="noopener noreferrer" href="https://github.com/triton-inference-server/tensorrtllm_backend">`triton-inference-server/tensorrtllm_backend`</a> 仓库，复制其中的 <a target="_blank" rel="noopener noreferrer" href="https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md">模型仓库（model repository）</a>模板，并修改配置：

```bash
cd ~
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend && git reset --hard da59830 && cd ~
cp -R tensorrtllm_backend/all_models/inflight_batcher_llm inflight_batcher_llm

python tensorrtllm_backend/tools/fill_template.py -i inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:Llama-2-7b-chat-hf/,triton_max_batch_size:64,preprocessing_instance_count:1
python tensorrtllm_backend/tools/fill_template.py -i inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:Llama-2-7b-chat-hf/,triton_max_batch_size:64,postprocessing_instance_count:1
python tensorrtllm_backend/tools/fill_template.py -i inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python tensorrtllm_backend/tools/fill_template.py -i inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:64
python tensorrtllm_backend/tools/fill_template.py -i inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:engines/1gpu,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600
```

## 部署

使用 `mlservice-runtime.yaml` 创建 MLServiceRuntime，再使用 `mlservice.yaml` 创建 MLService 以部署服务：

```bash
cd examples/deployments/triton-tensorrtllm
kubectl apply -f mlservice-runtime.yaml
kubectl create -f mlservice.yaml
```

对于 `mlservice-runtime.yaml` 配置文件进行如下说明：

* 每个 Predictor 最多请求 4 个 CPU（核心）、64 Gi 内存以及 1 个 GPU。
* 镜像 `t9kpublic/triton-tensorrtllm:20240307`（第 11 行）由当前目录下的 Dockerfile 定义。

对于 `mlservice.yaml` 配置文件进行如下说明：

* Predictor 数量为 1（第 13 行）。
* 如要使用队列，取消第 6-8 行的注释，并修改第 8 行的队列名称（默认为 `default`）。

监控服务是否准备就绪：

```bash
kubectl get -f mlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，便可开始使用该服务。

## 使用推理服务

继续使用 Notebook 的终端，使用 `curl` 命令发送推理请求：

```bash
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}' | sed 's#^https\?://##')
curl -X POST $address/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

返回的响应类似于：

```json
{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\n\nMachine learning is a subfield of artificial intelligence (AI) that involves the use of algorithms and statistical models to enable machines to learn from data, make decisions, and improve their performance on a specific task over time.\n\nMachine learning algorithms are designed to recognize patterns in data and learn from it, without being explicitly programmed to do so. The algorithms can be trained on large datasets, and as they process more data, they can make better predictions or decisions.\n\nMachine"}
```
