# LLaMA-Efficient-Tuning

[LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 是一个预训练、指令微调、评估和部署开源大型语言模型的项目。

本示例使用 DeepSpeedJob 在 TensorStack AI 计算平台上完成由 LLaMA-Efficient-Tuning 实现的指令微调训练。

## 使用方法

这里选取 [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) 同时作为有监督微调的预训练模型和奖励模型的预训练模型。训练的 4 个步骤均需要 GPU 4x A100 40GB，均使用 DeepSpeed 的 ZeRO-2 数据并行（对于 PPO 为 DDP 数据并行）策略，微调方式均选择 LoRA。更多细节如下表所示：

| 步骤 | 前置条件 | 数据集                                                                           | 预计时间 |
| ---- | -------- | -------------------------------------------------------------------------------- | -------- |
| SFT  | -        | [alpaca_gpt4_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)     | ~130min  |
| RM   | -        | [comparison_gpt4_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | ~35min   |
| PPO  | SFT + RM | alpaca_gpt4_zh                                                                   | ~12h     |
| DPO  | SFT      | comparison_gpt4_zh                                                               | ~90min   |

创建一个名为 llama-efficient-tuning、大小为 250GiB 的 PVC，然后创建一个同样名为 llama-efficient-tuning 的 Notebook 挂载该 PVC，镜像选择 PyTorch 2.0 的类型，模板选择 large（如要尝试命令行聊天，模板选择 large (NVIDIA GPU) 或 large (shared NVIDIA GPU)；如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆 LLaMA-Efficient-Tuning 以及此仓库：

```shell
cd ~
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
git clone https://github.com/t9k/examples.git
```

安装 git-lfs，从 Hugging Face Hub 拉取预训练模型：

```bash
sudo apt update && sudo apt install git-lfs  # password: tensorstack
git lfs install
mkdir models && cd models
git clone https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
```

## 训练

### SFT（有监督微调）

使用 `sft.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
cd ~/examples/llama-efficient-tuning/training
kubectl create -f sft.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 35 行）。
* 每个副本的进程数量（第 11 行）和 GPU 数量（第 49 和 53 行）同为 4。
* 队列名称为 default（第 8 行）。
* 输出路径为 `/t9k/mnt/output/sft-ckpts/baichuan2/7b`（第 23 行）。
* 训练占用显存 ~35GB，减小 `--per_device_train_batch_size`（第 25 行）可以减小显存占用，以防止 OOM。
* 镜像 `t9kpublic/llama-efficient-tuning:20230918`（第 94 行）由 [Dockerfile](./docker/Dockerfile) 定义。

### RM（训练奖励模型）

使用 `rm.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
kubectl create -f rm.yaml
```

### PPO

使用 `ppo.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
kubectl create -f ppo.yaml
```

### DPO

使用 `dpo.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
kubectl create -f dpo.yaml
```

## 命令行聊天

安装必要的依赖，然后执行 `src/cli_demo.py` 脚本以开始聊天：

```shell
cd ~/LLaMA-Efficient-Tuning
pip install trl xformers

# 加载 SFT 模型
python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/

# 加载 PPO 模型
python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/,/t9k/mnt/output/ppo-ckpts/baichuan2/7b/

# 加载 DPO 模型
python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/,/t9k/mnt/output/dpo-ckpts/baichuan2/7b/
```

## 评估

## 部署为服务
