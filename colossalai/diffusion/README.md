# Diffusion with ColossalAI

## 数据集

使用 LAION-5B 作为训练数据集。[LAION-5B 数据集](https://laion.ai/blog/laion-5b/)包含 CLIP 过滤后的 58.5 亿个高质量图像-文本对以及一些配套数据。

使用 Teyvat 作为微调数据集。[Teyvat 数据集](https://huggingface.co/datasets/Fazzie/Teyvat)包含来自 fandom 原神维基和哔哩哔哩原神维基的角色图像以及 BLIP 生成的标题，用于训练提瓦特角色的文本到图像模型。

## 训练

由于 LAION-5B 数据集过大（224 分辨率达 80TB，384 分辨率达 240TB），暂不提供运行方法。

## 微调

使用 `job_finetune.yaml` 创建 ColossalAIJob 以启动微调，你可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 工作器数量在第 14 行定义（默认为 `2`）。
* 日志信息和模型检查点的保存路径在第 18 行定义（默认为 `/tmp/`）。

```shell
kubectl create -f job_finetune.yaml
```

## 资源需求

微调时每个工作器占用约 14500MiB 显存和最大 18GiB 内存。

## 参考

* 原始论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* 原始项目：[Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
* ColossalAI 示例：[hpcaitech/ColossalAI/examples/images/diffusion](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion)
