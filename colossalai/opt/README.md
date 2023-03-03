# OPT with ColossalAI

## 数据集

使用随机数作为训练数据集（用于 benchmark）。

## 训练

使用 `job.yaml` 创建 ColossalAIJob 以启动训练，你可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 工作器数量在第 14 行定义（默认为 `4`）。
* 模型名称或路径在第 18 行定义（应为 `facebook/opt-*`，其中 `*` 取 `125m`、`350m`、`1.3b`、`2.7b`、`6.7b`、`13b`、`30b` 或 `66b`，默认为 `facebook/opt-350m`）。
* 批次大小在第 19 行定义（默认为 `16`）。
* 最大训练步数在第 20 行定义（默认为 `100`）。
* 工作器资源上限在第 31-33 行定义（默认为 `cpu: 4, memory: 8Gi, nvidia.com/gpu: "1"`，对于更大的模型应提供更大的内存，请参阅[资源需求](#资源需求)）。

```shell
kubectl create -f job.yaml
```

在训练开始之前，其中一个工作器会从 Hugging Face 下载模型检查点文件以初始化模型参数。文件会存储在 PVC `colossalai` 的 `hub/` 路径下。

## 资源需求

注意：下表仅供参考，[Gemini](https://colossalai.org/zh-Hans/docs/advanced_tutorials/meet_gemini/) 会统一管理异构内存。

| 模型                | 显存占用（每个工作器） | 内存占用（每个工作器） |
| ------------------- | ---------------------- | ---------------------- |
| `facebook/opt-125m` | 12084MiB               | 2.57GiB                |
| `facebook/opt-350m` | 13180MiB               | 3.48GiB                |
| `facebook/opt-1.3b` | 20330MiB               | 9.01GiB                |
| `facebook/opt-2.7b` | 24140MiB               | 18.7GiB                |
| `facebook/opt-6.7b` | 32422MiB               | 37.2GiB                |
| `facebook/opt-13b`  |                        |                        |
| `facebook/opt-30b`  |                        |                        |
| `facebook/opt-66b`  |                        |                        |

## 参考

* 原始论文：[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
* ColossalAI 示例：[hpcaitech/ColossalAI/examples/language/opt](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt)
