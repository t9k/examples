# OPT with ColossalAI

## 模型

从 Hugging Face 下载 `facebook/opt-*` 系列模型之一（其中 `*` 取 `125m`、`350m`、`1.3b`、`2.7b`、`6.7b`、`13b`、`30b` 或 `66b`），到 PVC `colossalai` 的 `opt-*` 路径下，例如：

```shell
cd
git clone https://huggingface.co/facebook/opt-1.3b
cd opt-1.3b/
rm pytorch_model.bin 
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/pytorch_model.bin
```

## 数据集

使用随机数作为训练数据集（用于 benchmark）。

```shell
cd
git clone https://huggingface.co/datasets/hugginglearners/netflix-shows
```

## 训练

使用 `job.yaml` 创建 ColossalAIJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 工作器数量在第 14 行定义（默认为 `4`）。
* 模型名称或路径在第 18 行定义（应为 `/t9k/mnt/opt-*`，与[模型](#模型)部分保持一致）。
* 批次大小在第 19 行定义（默认为 `16`）。
* 最大训练步数在第 20 行定义（默认为 `100`）。
* 工作器资源上限在第 31-33 行定义（默认为 `cpu: 4, memory: 8Gi, nvidia.com/gpu: "1"`，对于更大的模型应提供更大的内存，请参阅[资源需求](#资源需求)）。

```shell
kubectl create -f job.yaml
```

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
