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

```shell
kubectl create -f job.yaml
```

注意：在训练开始前所有工作器会各自下载模型检查点文件，这一行为并不合理，会占用工作器数量倍数的下载时间和存储空间。对于较大的模型（`facebook/opt-1.3b` 及以上），应考虑将其下载到一个 PVC 中，所有工作器挂载该 PVC（第 34-40 行），再修改模型名称或路径为相应的路径（第 18 行）。

## 资源需求

对于 `facebook/opt-350m`，训练时占用约 13180MiB 显存和 3.48GiB 内存。

## 参考

* 原始论文：[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
* ColossalAI 示例：[hpcaitech/ColossalAI/examples/language/opt](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt)
