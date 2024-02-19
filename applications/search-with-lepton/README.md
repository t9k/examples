# Search with Lepton

[Search with Lepton](https://github.com/leptonai/search_with_lepton) 是一个对话式搜索引擎的开源项目。[`t9k/search_with_lepton`](https://github.com/t9k/search_with_lepton) fork 了该项目并进行了一定的修改，以便于私有化部署。

本示例使用 Kubernetes 原生资源 Deployment 在 TensorStack AI 平台上部署 Search with Lepton 对话式搜索引擎应用。

## 使用方法

请先参阅 [vLLM](../../deployments/vllm/) 部署一个推理服务。

创建一个名为 `search`、大小 1 GiB 以上的 PVC，然后创建一个同样名为 `search` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

## 部署

在 `secret.yaml` 中提供所调用搜索引擎的 API key，使用它创建 Secret：

```bash
cd examples/applications/search-with-lepton
vim secret.yaml
kubectl apply -f secret.yaml
```

> [!TIP]
> 请参阅 [Setup Search Engine API](https://github.com/leptonai/search_with_lepton/tree/main?tab=readme-ov-file#setup-search-engine-api) 以获取相应搜索引擎后端的 API key 或 subscription key。

在 `deployment.yaml` 中提供环境变量，使用它创建 Deployment 以部署应用：

```bash
vim deployment.yaml
kubectl create -f deployment.yaml
```

然后暴露 Deployment 为 Service（ClusterIP 类型）：

```bash
kubectl expose deployment search-with-lepton --name=search-with-lepton
```

## 搜索

在本地的终端中，使用 t9k-pf 命令行工具，将服务的 8080 端口转发到本地的 8080 端口：

```bash
t9k-pf service search-with-lepton 8080:8080 -n <PROJECT NAME>
```

然后使用浏览器访问 `127.0.0.1:8080`，搜索感兴趣的问题。
