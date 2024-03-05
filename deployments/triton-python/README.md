# Triton（Python 后端）

[Triton Inference Server](https://github.com/triton-inference-server/server) 是一个开源的推理服务软件，旨在简化 AI 推理流程。Triton 使用户能够部署多种深度学习框架的模型，包括 TensorRT、TensorFlow、PyTorch、ONNX、OpenVINO 等。Triton 为许多查询类型提供了优化的性能，包括实时、批处理、集成和音频/视频流。

本示例使用 MLService，以及 Triton 推理服务器和它的 Python 后端在平台上部署一个 Hugging Face 模型的推理服务。

## 使用方法

在项目中创建一个名为 `triton-python`、大小 50 GiB 以上的 PVC（需要存储模型文件），然后创建一个同样名为 `triton-python` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

使用预制的模型仓库：

```bash
cp -R examples/deployments/triton-python/python_model_repository .
```

然后从 Hugging Face Hub 下载要部署的模型，这里以 [vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) 模型为例：

```bash
huggingface-cli download google/vit-base-patch16-224-in21k \
  --local-dir vit-base-patch16-224-in21k --local-dir-use-symlinks False
```

## 部署

使用 `mlservice-runtime.yaml` 创建 MLServiceRuntime，再使用 `mlservice.yaml` 创建 MLService 以部署服务：

```bash
cd examples/deployments/triton-python
kubectl apply -f mlservice-runtime.yaml
kubectl create -f mlservice.yaml
```

对于 `mlservice-runtime.yaml` 配置文件进行如下说明：

* 每个 Predictor 最多请求 1 个 CPU（核心）、16 Gi 内存以及 1 个 GPU。

对于 `mlservice.yaml` 配置文件进行如下说明：

* Predictor 数量为 1（第 13 行）。
* 如要使用队列，取消第 6-8 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 模型存储在 PVC `vllm` 的 `CodeLlama-7b-Instruct-hf/` 路径下（第 19 行）。

监控服务是否准备就绪：

```bash
kubectl get -f mlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，便可开始使用该服务。

## 使用推理服务

使用作为推理客户端的 Python 脚本发送推理请求：

```bash
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}' | sed 's#^https\?://##')
pip install tritonclient gevent geventhttpclient
python client.py --server_address $address --model_name python_vit
```

输出应类似于：

```
[[[ 0.2463658   0.12966464  0.13196409 ... -0.12697077  0.08220191
   -0.1261508 ]
  [ 0.10375027  0.15543337  0.14776552 ... -0.09246814  0.10163841
   -0.31893715]
  [ 0.04861938  0.15119025  0.14414431 ... -0.08075114  0.0719012
   -0.32684252]
  ...
  [ 0.2877585   0.15052384  0.17233661 ... -0.07538544  0.05114003
   -0.19613911]
  [ 0.21476139  0.17660537  0.14951637 ... -0.09027394  0.0747345
   -0.31565955]
  [ 0.2561764   0.16620857  0.13983792 ... -0.06043544  0.08778334
   -0.14347576]]]
(1, 197, 768)
```
