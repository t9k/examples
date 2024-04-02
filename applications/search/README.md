# 对话式搜索引擎

[Search with Lepton](https://github.com/leptonai/search_with_lepton) 是一个对话式搜索引擎的开源项目。[`t9k/search_with_lepton`](https://github.com/t9k/search_with_lepton) fork 了该项目并进行了一定的修改，以便于私有化部署。

本示例使用 Kubernetes 原生资源和 Tensorstack 资源在 TensorStack AI 平台上部署该修改后的对话式搜索引擎应用。

## 使用方法

创建一个名为 `search`、大小 180 GiB 以上的 PVC，然后创建一个同样名为 `search` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

### 部署 LLM 推理服务

> [!TIP]
> 以下部署 LLM 推理服务的步骤来自示例 [vLLM](../../deployments/vllm/)。本示例以 Mixtral 8x7B 模型为例。

执行以下命令以下载 Mixtral-8x7B-Instruct-v0.1 的模型文件：

```bash
# 方法1：如果可以直接访问 huggingface
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir Mixtral-8x7B-Instruct-v0.1 --local-dir-use-symlinks False

# 方法2：对于国内用户，使用 modelscope
pip install modelscope
python -c \
  "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/Mixtral-8x7B-Instruct-v0.1')"
mv .cache/modelscope/hub/AI-ModelScope/Mixtral-8x7B-Instruct-v0.1 .
```

然后使用 vLLM 部署兼容 OpenAI API 的 LLM 推理服务。使用以下 YAML 配置文件创建 MLServiceRuntime：

```bash
cd examples/applications/search
kubectl apply -f mlservice-runtime.yaml
```

再使用以下 YAML 配置文件创建 MLService 以部署服务（必要时修改 `spec.scheduler.t9kScheduler.queue` 字段指定的队列）：

```bash
kubectl create -f mlservice.yaml
```

### 部署对话式搜索服务

在 `secret.yaml` 中提供所调用搜索引擎的 API key，使用它创建 Secret：

```bash
# 修改 secret.yaml
kubectl apply -f secret.yaml
```

> [!TIP]
> 请参阅 [Setup Search Engine API](https://github.com/leptonai/search_with_lepton/tree/main?tab=readme-ov-file#setup-search-engine-api) 以获取相应搜索引擎后端的 API key 或 subscription key。

在 `deployment.yaml` 中提供环境变量，使用它创建 Deployment：

```bash
# 修改 deployment.yaml
kubectl create -f deployment.yaml
```

然后暴露 Deployment 为 Service：

```bash
kubectl create -f secret.yaml
```

### 搜索

在本地的终端中，使用 t9k-pf 命令行工具，将服务的 8080 端口转发到本地的 8080 端口：

```bash
t9k-pf service search 8080:8080 -n <PROJECT NAME>
```

然后使用浏览器访问 `127.0.0.1:8080`，搜索感兴趣的问题。
