# Search with Lepton

[Search with Lepton](https://github.com/leptonai/search_with_lepton) 是一个对话式搜索引擎的开源项目。

本示例使用 MLService 在 TensorStack AI 平台上部署 Search with Lepton 对话式搜索引擎应用。

## 使用方法

请先参阅 [vLLM](../../deployments/vllm/) 部署一个推理服务。

创建一个名为 `search`、大小 1 GiB 以上的 PVC，然后创建一个同样名为 `search` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

## 部署

使用 `mlservice-runtime.yaml` 创建 MLServiceRuntime：

```bash
cd examples/applications/search-with-lepton
kubectl apply -f mlservice-runtime.yaml
```

在 `mlservice.yaml` 中提供环境变量，再使用它创建 MLService 以部署应用：

```bash
vim mlservice.yaml
kubectl create -f mlservice.yaml
```

> [!TIP]
> 请参阅 [Setup Search Engine API](https://github.com/leptonai/search_with_lepton/tree/main?tab=readme-ov-file#setup-search-engine-api) 以获取相应搜索引擎后端的 API key 或 subscription key。

</aside>

## 搜索

在本地的终端中，使用 t9k-pf 命令行工具，将 MLService 创建的以下服务的 80 端口转发到本地的 8080 端口：

```bash
t9k-pf service search-with-lepton-vllm-predict-version1-00001-private 8080:80 -n <PROJECT NAME>
```

然后使用浏览器访问 `127.0.0.1:8080`，搜索感兴趣的问题。
