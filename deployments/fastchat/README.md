# FastChat

<a target="_blank" rel="noopener noreferrer" href="https://github.com/lm-sys/FastChat">FastChat</a> 是一个训练、伺服和评估基于 LLM 的聊天机器人的开放平台，其提供多种伺服方式，包括命令行、Web UI、兼容 OpenAI 的 RESTful API 等。

本示例使用 SimpleMLService 和 FastChat 框架在平台上部署一个 LLM 推理服务。

## 使用方法

创建一个名为 `fastchat`、大小 50 GiB 以上的 PVC（需要存储模型文件），然后创建一个同样名为 `fastchat` 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

然后从 Hugging Face Hub 或魔搭社区下载要部署的模型，这里以 [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) 模型为例：

```bash
# 方法 1：如果可以直接访问 huggingface
huggingface-cli download THUDM/chatglm3-6b \
  --local-dir chatglm3-6b --local-dir-use-symlinks False

# 方法 2：对于国内用户，访问 modelscope 网络连通性更好
pip install modelscope
python -c \
  "from modelscope import snapshot_download; snapshot_download('ZhipuAI/chatglm3-6b')"
mv .cache/modelscope/hub/ZhipuAI/chatglm3-6b .
```

## 部署

使用 `simplemlservice.yaml` 创建 SimpleMLService 以部署服务：

```bash
cd ~/examples/deployments/fastchat
kubectl create -f simplemlservice.yaml
```

对于 `simplemlservice.yaml` 配置文件进行如下说明：

* 副本数量为 1（第 6 行）。
* 如要使用队列，取消第 7-9 行的注释，并修改第 9 行的队列名称（默认为 `default`）。
* 模型存储在 PVC `fastchat` 的 `chatglm3-6b/` 路径下（第 19 行）。
* 使用的镜像 `t9kpublic/fastchat-openai`（第 24 行）由当前目录下的 Dockerfile 定义。
* 计算资源最多使用 4 个 CPU（核心）、64 GiB 内存以及 1 个 GPU（第 31-33 行）。

监控服务是否准备就绪：

``` bash
kubectl get -f simplemlservice.yaml -w
```

待其 `READY` 值变为 `true` 后，便可开始使用该服务。

## 使用推理服务

使用 `curl` 命令发送聊天或生成文本的请求：

``` bash
address=$(kubectl get -f simplemlservice.yaml -ojsonpath='{.status.address.url}')

# 聊天
curl ${address}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm3-6b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.5
  }'

# 生成文本
curl ${address}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm3-6b",
    "prompt": "很久很久以前，",
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
    "model": "chatglm3-6b",
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
    "model": "chatglm3-6b",
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

我们也可以使用 [OpenAI Python 库](https://github.com/openai/openai-python)或第三方客户端（如 [ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)）来与 LLM 推理服务进行交互。有关如何使用这些客户端的详细信息，请参阅它们各自的用户文档。
