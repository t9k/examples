# LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个（增量）预训练、指令微调、评估和部署开源大型语言模型的项目。LLaMA-Factory 支持大部分主流模型，提供丰富的训练方法，支持多种精度，并且整合了多种先进算法和实用技术。

本示例使用 PyTorchTrainingJob 在 TensorStack AI 计算平台上完成由 LLaMA-Factory 实现的指令微调和评估。

## 使用方法

这里选取 [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) 或 [Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) 模型进行 SFT（有监督微调）和 DPO 训练，更多细节如下表所示：

| 参数量 | 步骤 | 数据集                                                                                                                                                                                                                                                                            | 配置文件            | 并行策略 | GPU 使用（参考） | 预计时间 |
| ------ | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------- | ---------------- | -------- |
| 8B     | SFT  | [identity](https://github.com/hiyouga/LLaMA-Factory/blob/625a0e32c47aeb72a6fe9c3536914996912e89d4/data/identity.json), [alpaca_gpt4_en](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM), [alpaca_gpt4_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | `sft-8b.yaml`       | -        | 1x A100 40GB     | ~12h     |
|        |      |                                                                                                                                                                                                                                                                                   | `sft-8b-2xdp.yaml`  | 数据并行 | 2x A100 40GB     | ~6h      |
|        | DPO  | [dpo_mix_en](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k), [dpo_mix_zh](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)                                                                                                                                          | `dpo-8b-4xdp.yaml`  | 数据并行 | 4x A100 40GB     | ~5h      |
| 70B    | SFT  | identity, alpaca_gpt4_en, alpaca_gpt4_zh                                                                                                                                                                                                                                          | `sft-70b-4xdp.yaml` | 数据并行 | 4x A100 80GB     |          |
|        | DPO  | dpo_mix_en, dpo_mix_zh                                                                                                                                                                                                                                                            | `dpo-70b-4xdp.yaml` | 数据并行 | 4x A100 80GB     |          |

创建一个名为 llama-factory、大小为 60 GiB（对于 8B 模型，对于 70B 模型为 500GiB）的 PVC，然后创建一个同样名为 llama-factory 的 Notebook 挂载该 PVC，镜像选择 PyTorch 2.0 的类型，模板选择 large (NVIDIA GPU) 或 large (shared NVIDIA GPU)（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆 LLaMA-Factory 以及此仓库：

```bash
cd ~
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone https://github.com/t9k/examples.git
```

安装 LLaMA-Factory 库：

```bash
pip install ./LLaMA-Factory
```

从 Hugging Face Hub（或魔搭社区）拉取预训练模型：

```bash
mkdir models && cd models
MODEL_NAME=Meta-Llama-3.1-8B  # 或 Meta-Llama-3.1-70B

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "meta-llama/$MODEL_NAME" --exclude "original/*" --local-dir "./$MODEL_NAME" --local-dir-use-symlinks False --token <HF_TOKEN>
# 或
# pip install modelscope
# modelscope download --model "LLM-Research/$MODEL_NAME" --exclude "original/*" --local_dir "./$MODEL_NAME"
```

## 训练

### 8B

#### SFT

以 `sft-8b.yaml` 为例，创建 PyTorchTrainingJob 以执行 SFT（有监督微调）训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f sft-8b.yaml  # 或 sft-8b-2xdp.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 12 行）。
* 每个副本的进程数量和 GPU 数量（第 31 和 35 行）同为 1。
* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 读取配置文件 `examples/llama-factory/training/sft-8b-config.yaml`（第 23 行）。
* 训练占用显存 ~40GB，减小配置文件的 `per_device_train_batch_size`（第 27 行）可以减小显存占用，以防止 OOM。
* 镜像 `t9kpublic/llama-factory:20240730`（第 18 行）由 [Dockerfile](./Dockerfile) 定义。

分别与基座模型以及 SFT 训练得到的模型聊天：

```bash
cd ~

# 没有聊天能力，会自言自语和无限重复
llamafactory-cli chat examples/llama-factory/inference/8b.yaml

# 有聊天能力
llamafactory-cli chat examples/llama-factory/inference/8b-sft.yaml
```

然后合并 LoRA adapter 到原模型得到新模型 Meta-Llama-3.1-8B-sft：

```bash
llamafactory-cli export examples/llama-factory/merging/8b-sft.yaml
```

#### DPO

以 `dpo-8b-4xdp.yaml` 为例，创建 PyTorchTrainingJob 以执行 DPO（直接偏好优化）训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f dpo-8b-4xdp.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 17 行）。
* 每个副本的进程数量（第 13 行）和 GPU 数量（第 35 和 39 行）同为 4。
* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 读取配置文件 `examples/llama-factory/training/dpo-8b-4xdp-config.yaml`（第 27 行）。
* 训练占用显存 ~38GB。
* 镜像 `t9kpublic/llama-factory:20240730`（第 18 行）由 [Dockerfile](./Dockerfile) 定义。

与 DPO 训练得到的模型聊天：

```bash
cd ~

# 回答更加符合人类偏好
llamafactory-cli chat examples/llama-factory/inference/8b-dpo.yaml
```

然后合并 LoRA adapter 到 Meta-Llama-3.1-8B-sft 模型得到新模型 Meta-Llama-3.1-8B-dpo：

```bash
llamafactory-cli export examples/llama-factory/merging/8b-dpo.yaml
```

### 70B

（WIP）

<!-- #### SFT

以 `sft-70b-4xdp.yaml` 为例，创建 PyTorchTrainingJob 以执行 SFT（有监督微调）训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f sft-70b-4xdp.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 17 行）。
* 每个副本的进程数量（第 13 行）和 GPU 数量（第 35 和 39 行）同为 4。
* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 读取配置文件 `examples/llama-factory/training/sft-70b-4xdp-config.yaml`（第 23 行）。
* 训练占用显存 ~GB，减小配置文件的 `per_device_train_batch_size`（第 29 行）可以减小显存占用，以防止 OOM。
* 镜像 `t9kpublic/llama-factory:20240730`（第 23 行）由 [Dockerfile](./Dockerfile) 定义。

然后合并 LoRA adapter 到原模型得到新模型 Meta-Llama-3.1-70B-sft：

```bash
llamafactory-cli export examples/llama-factory/merging/70b-sft.yaml
``` -->

## 评估

在 MMLU（或 CMMLU、C-Eval）数据集上评估模型：

```bash
cd ~
HF_ENDPOINT=https://hf-mirror.com llamafactory-cli eval examples/llama-factory/evaluation/8b.yaml  # 或 8b-sft.yaml, 8b-dpo.yaml
```

## 部署为推理服务

可以使用 vLLM 将合并后的模型部署为推理服务。
