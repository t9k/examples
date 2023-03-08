# PaLM with ColossalAI

## 数据集

使用 enwik8 作为训练数据集。enwik8 数据集是英文维基百科 2006 年 3 月 3 日的 XML 转储的前 1 亿个（100M）字节，通常用于衡量模型压缩数据的能力。

也可以使用随机数作为训练数据集（两种数据集都用于 benchmark）。

## 训练

使用 `job.yaml` 创建 ColossalAIJob 以启动训练，你可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 工作器数量在第 14 行定义（默认为 `4`）。
* 如要使用 DDP 而非 ColossalAI 作为分布式策略，修改第 18 行的 `"--distplan=colossalai"` 为 `"--distplan=pytorch"`。
* 批次数量在第 19 行定义（默认为 `100`）。
* 如要使用随机数作为训练数据集，取消第 20 行的注释。

```shell
kubectl create -f job.yaml
```

## 模型超参数和资源需求

本示例设置的模型超参数为：

```python
model = PaLM(num_tokens=20000, dim=512, depth=24, heads=8, dim_head=64)
```

训练时每个工作器占用约 11000MiB 显存和 2.63GiB 内存。

## 参考

* 原始论文：[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
* 原始项目：[lucidrains/PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)
* ColossalAI 示例：[hpcaitech/ColossalAI/examples/language/palm](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/palm)
