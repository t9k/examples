# Dreambooth with ColossalAI

## 数据集

使用 Teyvat 作为训练数据集。[Teyvat 数据集](https://huggingface.co/datasets/Fazzie/Teyvat)包含来自 fandom 原神维基和哔哩哔哩原神维基的角色图像以及 BLIP 生成的标题，用于训练提瓦特角色的文本到图像模型。

## 训练

使用 `job.yaml` 创建 ColossalAIJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 工作器数量在第 14 行定义（默认为 `2`）。
* 模型检查点的保存路径在第 20 行定义（默认为 `.`）。
* 训练批次大小第 23 行定义（默认为 `1`）。
* 最大训练步数在第 27 行定义（默认为 `400`）。

```shell
kubectl create -f job.yaml
```

## 资源需求

训练时每个工作器占用约 22200MiB 显存和 8.20GiB 内存。

## 参考

* 原始论文：[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
* 原始项目：[huggingface/diffusers/examples/dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
* ColossalAI 示例：[hpcaitech/ColossalAI/examples/images/dreambooth](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth)
