# LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个简单易用且高效的大型语言模型（Large Language Model）训练与微调平台。LLaMA-Factory 支持大部分主流模型，提供丰富的训练方法，支持多种精度，并且整合了多种先进算法和实用技术。

本示例使用 PyTorchTrainingJob 在 TensorStack AI 计算平台上完成由 LLaMA-Factory 实现的微调和评估。

## 使用方法

这里选取 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 模型进行 LoRA SFT 和 DPO 训练，以及全量 SFT 和 DPO 训练，更多细节如下表所示：

| 微调方法 | 步骤 | 数据集                                                                                                                                                                                                                                                                          | 配置文件        | 并行策略 | GPU 使用（参考） |
| -------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | -------- | ---------------- |
| LoRA     | SFT  | [identity](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json), [alpaca_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_en_demo.json), [alpaca_zh_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_zh_demo.json) | `lora-sft.yaml` | -        | 1x A100 40GB     |
| LoRA     | DPO  | [dpo_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_en_demo.json), [dpo_zh_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_zh_demo.json)                                                                                                | `lora-dpo.yaml` | 数据并行 | 2x A100 40GB     |
| Full     | SFT  | [identity](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json), [alpaca_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_en_demo.json), [alpaca_zh_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_zh_demo.json) | `full-sft.yaml` | 数据并行 | 4x A100 40GB     |


创建一个名为 llama-factory、大小为 60 GiB 的 PVC，然后创建一个同样名为 llama-factory 的 JupyterLab/CodeServer App 挂载该 PVC。

进入 JupyterLab/CodeServer 的 UI，启动一个终端，执行以下命令以克隆 LLaMA-Factory 以及此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

从魔搭社区拉取预训练模型：

```bash
mkdir models && cd models
MODEL_NAME=Qwen2.5-7B-Instruct

modelscope download --model "Qwen/$MODEL_NAME" --local_dir "./$MODEL_NAME"
```

### 训练

#### LoRA SFT

以 `lora-sft.yaml` 为例，创建 PyTorchTrainingJob 以执行 LoRA SFT 训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f lora-sft.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 11 行）。
* 每个副本的进程数量和 GPU 数量（第 30 和 34 行）同为 1。
* 读取配置文件 `examples/llama-factory/training/lora-sft-config.yaml`（第 22 行）。
* 检查点文件保存在 `saves/qwen2-5-7b/lora/sft` 目录下。
* 训练占用显存 ~37GB，减小配置文件的 `per_device_train_batch_size`（第 30 行）可以减小显存占用，以防止 OOM。
* 镜像 `registry.cn-hangzhou.aliyuncs.com/t9k/llamafactory:20250318`（第 18 行）由 [Dockerfile](https://github.com/hiyouga/LLaMA-Factory/blob/a02a140840da08d2b0fe16adcd6de09afe732ab5/docker/docker-cuda/Dockerfile) 定义。

#### LoRA DPO

以 `lora-dpo.yaml` 为例，创建 PyTorchTrainingJob 以执行 LoRA DPO 训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f lora-dpo.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 11 行）。
* 每个副本的进程数量和 GPU 数量（第 30 和 34 行）同为 2。
* 读取配置文件 `examples/llama-factory/training/lora-dpo-config.yaml`（第 22 行）。
* 检查点文件保存在 `saves/qwen2-5-7b/lora/dpo` 目录下。
* 训练最高占用显存 ~40GB。

#### Full SFT

以 `full-sft.yaml` 为例，创建 PyTorchTrainingJob 以执行 Full SFT 训练：

```bash
cd ~/examples/llama-factory/training
kubectl create -f full-sft.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 11 行）。
* 每个副本的进程数量和 GPU 数量（第 30 和 34 行）同为 4。
* 读取配置文件 `examples/llama-factory/training/full-sft-config.yaml`（第 22 行）。
* 检查点文件保存在 `saves/qwen2-5-7b/full/sft` 目录下。
* 训练占用显存 ~GB。

### 进一步操作

要对训练完成的模型作进一步的操作，包括评估、聊天、合并模型文件等，使用 `debug.yaml` 创建一个启用调试模式的 PyTorchTrainingJob，并进入其创建的 Pod：

```bash
cd ~/examples/llama-factory/training
kubectl create -f debug.yaml
kubectl exec -it $(kubectl get pod -l tensorstack.dev/owner-name=qwen2-5-7b-debug -o jsonpath='{.items[0].metadata.name}') -- bash
```

进入 Pod 后，参照下面各部分，使用 `llamafactory-cli` 命令进行进一步操作。

操作完成后，退出 Pod 并删除 PyTorchTrainingJob：

```bash
exit
kubectl delete -f debug.yaml
```

#### 评估

在 MMLU（或 CMMLU、C-Eval）数据集上评估模型：

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
HF_ENDPOINT=https://hf-mirror.com llamafactory-cli eval examples/llama-factory/evaluation/lora-sft.yaml  # 或 lora-dpo.yaml, full-sft.yaml
```

#### 聊天

与模型聊天：

```bash
cd /workspace
llamafactory-cli chat examples/llama-factory/chat/lora-sft.yaml  # 或 lora-dpo.yaml, full-sft.yaml
```

#### 合并

合并 LoRA adapter 到 Qwen2.5-7B-Instruct，导出为新模型：

```bash
llamafactory-cli export examples/llama-factory/merging/lora-sft.yaml  # 或 lora-dpo.yaml
```

模型文件保存在 `models/Qwen2.5-7B-Instruct-sft`（或 `models/Qwen2.5-7B-Instruct-dpo`）目录下。

### 部署为推理服务

可以使用 vLLM App 将合并后的模型部署为推理服务。

## 对于其他厂商 GPU 的支持

* 沐曦（MetaX）：在当前目录下作文本替换：
    1. `registry.cn-hangzhou.aliyuncs.com/t9k/llamafactory:20250318` -> `<REGISTRY>/mximages/mxc500-deepspeed:2.29.0.8` 以替换使用的镜像。
    2. `nvidia.com/gpu` -> `metax-tech.com/gpu` 以替换使用的 GPU 类型。
    3. 同样参照[使用方法](#使用方法)进行操作。

## 参考

* 原始项目：[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
