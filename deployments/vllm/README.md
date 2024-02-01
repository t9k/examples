# vLLM

[vLLM](https://github.com/vllm-project/vllm) 是一个快速、灵活且易于使用的 LLM 推理和服务库，其利用 PagedAttention 注意力算法显著提高了服务吞吐量。

本示例使用 MLService 在平台上部署一个 vLLM 推理服务。

## 使用方法

创建一个名为 `vllm`、大小为 100GiB 的 PVC（需要存储模型文件），然后创建一个同样名为 `vllm` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

然后从 Hugging Face Hub（或魔搭社区）拉取要部署的模型，这里以 [CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) 为例：

```bash
git clone --depth 1 https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf
# 或
# git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/CodeLlama-7b-Instruct-hf.git
```

## 部署

使用 `mlservice-runtime.yaml` 创建 MLServiceRuntime，再使用 `mlservice.yaml` 创建 MLService 以部署服务：

```bash
cd ~/examples/deployments/vllm
kubectl apply -f mlservice-runtime.yaml
kubectl create -f mlservice.yaml
```

在命令行监控服务是否就绪：

``` bash
kubectl get -f mlservice.yaml -o wide -w
```

待 `Ready` 列变为 `True`，便可开始使用服务。

## 使用服务

使用 `curl` 命令发送聊天或生成文本的请求：

``` bash
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}')

# 聊天
curl ${address}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codellama-7b",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.5
  }'

# 生成文本
curl ${address}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codellama-7b",
    "prompt": "Long long ago, there was",
    "max_tokens": 100,
    "temperature": 0.5
  }'
```

返回的响应类似于：

```json
{
    "id": "cmpl-5915c46dc6054ecfa4d57d07225c1264",
    "object": "chat.completion",
    "created": 5101130,
    "model": "codellama-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "  Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 37,
        "completion_tokens": 27
    }
}

{
    "id": "cmpl-afbd703626c44a12ad192d0861fadd6e",
    "object": "text_completion",
    "created": 5101200,
    "model": "codellama-7b",
    "choices": [
        {
            "index": 0,
            "text": " a time when the world was dark and cold, and little light entered.\n\nA young girl named Kanna was born in this world. She was born with a burden on her back.\n\nKanna grew up in a small village, surrounded by snow and ice. The villagers were poor, and they lived in miserable huts. They were cold and hungry all the time.\n\nBut Kanna was different. She had a special gift. She could make light",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 7,
        "total_tokens": 107,
        "completion_tokens": 100
    }
}
```

我们也可以使用 [OpenAI Python 库](https://github.com/openai/openai-python)或第三方客户端（如 [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)）来与 vLLM 推理服务进行交互。有关如何使用这些客户端的详细信息，请参阅它们各自的用户文档。
