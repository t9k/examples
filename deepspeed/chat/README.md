# DeepSpeed-Chat

[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 是 DeepSpeed 推出一个快速、经济实惠、可扩展且开放的系统框架，旨在实现端到端的强化学习人类反馈（RLHF）训练，以生成高质量的 ChatGPT 风格模型。该框架能够适应各种规模的训练需求。

本项目使用 DeepSpeedJob 在 TensorStack AI 计算平台上完成 DeepSpeed-Chat 的训练。

## 使用方法

训练越大的模型，所需要的计算资源越多，对应的分布式场景也不相同。本项目针对单 GPU、单副本多 GPU 和多副本多 GPU 这三种分布式场景分别提供不同的 DeepSpeedJob YAML 配置文件，其训练不同的 Actor 模型，请求不同的资源量，如下表所示：

| YAML 配置文件        | 分布式场景     | GPU          | Actor 模型 | Reward 模型 | PVC 大小 |
| -------------------- | -------------- | ------------ | ---------- | ----------- | -------- |
| `*-single-gpu.yaml`  | 单 GPU         | 1x A100 40G  | OPT-1.3B   | OPT-350M    | 20GiB    |
| `*-single-node.yaml` | 单副本多 GPU | 4x A100 40G  | OPT-13B    | OPT-350M    | 100GiB   |
| `*-multi-node.yaml`  | 多副本多 GPU | 16x A100 80G | OPT-66B    | OPT-350M    | 500GiB   |

> DeepSpeed-Chat 即将支持 LLaMA 模型的训练。

创建一个名为 chat、相应大小的 PVC，然后创建一个同样名为 chat 的 Notebook 挂载该 PVC，镜像选择带有 sudo 权限的类型，资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
cd ~
git clone https://github.com/t9k/examples.git
```

### 准备模型和数据集

安装 git-lfs，从 Hugging Face Hub 拉取要训练的 Actor 模型，这里以 [facebook/opt-13b](https://huggingface.co/facebook/opt-13b) 为例：

```bash
sudo apt install git-lfs  # password: tensorstack
git lfs install
mkdir models && cd models
git clone https://huggingface.co/facebook/opt-13b
cd ~
```

接着使用 `download_dataset.py` 脚本下载并保存数据集，这里以 [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) 为例：

```bash
python download_dataset.py Dahoas/rm-static
```

如果模型或数据集下载失败，请过一段时间重试，或使用网络代理。

### 训练

训练分为三步进行，其中第一步和第二步可以同时进行，第三步依赖前两步输出的模型。

#### Step 1: Supervised Finetuning

使用合适的配置文件创建 DeepSpeedJob 以执行第一步训练（这里使用 `actor-single-node.yaml` 微调 LLaMA 7B 模型）：

```bash
cd examples/deepspeed/chat

# 单 GPU 训练，OPT-1.3B 模型
kubectl create -f actor/actor-single-gpu.yaml
# 单副本多 GPU 训练，OPT-13B 模型
kubectl create -f actor/actor-single-node.yaml
# 多副本多 GPU 训练，OPT-66B 模型
kubectl create -f actor/actor-multi-node.yaml
```

对于 `actor-single-node.yaml` 进行如下说明，其余配置文件也是类似的：

* 训练副本（replica）数量为 1（第 32 行）。
* 每个副本的进程数量（第 7 行）和 GPU 数量（第 46 和 50 行）同为 4。
* 使用数据集 [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)、[Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) 和 [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)（第 11 行）；可以使用一个或多个数据集；所有可用的数据集请参考[这里](./utils/data/data_utils.py#L20)。
* 模型文件会在训练完成后输出到 `output/multi-node/actor-models/13b` 路径下（第 29 行）。
* 修改以下参数可以减小显存占用，以防止 OOM：
    * 训练更小的模型（第 13 行）。
    * 在多 GPU 训练的情况下，增大 `--zero_stage`（第 25 行，可以取 0、1、2 或 3），但可能损害性能。
    * 减小批次规模（第 14 和 15 行），若不要影响收敛过程，则同时增大 `--gradient_accumulation_steps`（第 20 行）。
* 镜像 `tsz.io/t9k/deepspeed:chat-0.9.5`（第 38 行）由 [Dockerfile](./Dockerfile) 定义。

#### Step 2: Reward Model Finetuning

使用合适的配置文件创建 DeepSpeedJob 以执行第二步训练（这里使用 `actor-single-gpu.yaml` 训练 OPT 350M 模型）：

```bash
# 单 GPU 训练，OPT-1.3B 模型
kubectl create -f reward/reward-single-gpu.yaml
# 单副本多 GPU 训练，OPT-13B 模型
kubectl create -f reward/reward-single-node.yaml
# 多副本多 GPU 训练，OPT-66B 模型
kubectl create -f reward/reward-multi-node.yaml
```

配置与 Step 1 类似。

#### Step 3: RLHF finetuning

```bash
# 单 GPU 训练，OPT-1.3B 模型
kubectl create -f rlhf/rlhf-single-gpu.yaml
# 单副本多 GPU 训练，OPT-13B 模型
kubectl create -f rlhf/rlhf-single-node.yaml
# 多副本多 GPU 训练，OPT-66B 模型
kubectl create -f rlhf/rlhf-multi-node.yaml
```

配置与 Step 1 类似，除了 `--actor_model_name_or_path`（第 14 行）和 `--critic_model_name_or_path`（第 15 行）分别为 Step 1 和 2 的输出路径。