# Megatron-DeepSpeed GPT

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 用于高效训练基于 transformer 的大型语言模型（如 GPT、BERT 和 T5），支持模型并行（张量并行、序列并行和流水线并行）和多节点训练。[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/tree/main) 是 Megatron-LM 的 DeepSpeed 版本，它为几个功能添加了额外的支持，包括 MoE 模型训练、课程学习、3D 并行等。

本示例使用 DeepSpeedJob 在 TensorStack AI 计算平台上完成 Megatron-DeepSpeed 的 GPT 示例的训练。

## 使用方法

GPT-3 系列模型的参数量覆盖了从 125M 到 175B 的非常大的范围，如下图所示：

![](https://s2.loli.net/2023/07/19/Ws6STy4IdmHYRpU.png)

这里选取其中 125M、1.3B、13B 和 175B（WIP）的模型进行训练，它们的参数量大约以 10 倍的关系递增，同时训练需要的 GPU 数量、需要的数据量、花费的时间逐渐增加，采用的并行策略逐渐复杂，具有比较好的代表性。更多细节如下表所示：

| 参数量 | 训练 token 量 | 数据集 | 并行策略              | GPU 使用（参考） | 预计时间 |
| ------ | ------------- | ------ | --------------------- | ---------------- | -------- |
| 125M   | 2.5B          | enwiki | -                     | 1x A100 40GB     | ~8h      |
| 1.3B   | 26B           | enwiki | 数据并行              | 4x A100 40GB     | ~8d      |
| 13B    | 260B          | enwiki | 数据并行 + 流水线并行 | 2x 8x A100 40GB  |          |
| 175B   | 3.5T          |        |                       |                  |          |

> 要训练其他参数量的模型，可以在参数量接近的现有训练配置的基础上修改适当的参数以进行训练。

创建一个名为 megatron、大小为 250GiB 的 PVC，然后创建一个同样名为 megatron 的 Notebook 挂载该 PVC，镜像选择 PyTorch 2.0 的类型，模板选择 large（如要尝试文本生成，模板选择 large (NVIDIA GPU) 或 large (shared NVIDIA GPU)；如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆 Megatron-DeepSpeed、Megatron-LM 以及此仓库：

```shell
cd ~
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
git clone https://github.com/NVIDIA/Megatron-LM.git
git clone https://github.com/t9k/examples.git
```

## 准备数据集

暂且以 enwiki（英文维基百科）作为唯一的数据集。安装必要的 Python 库，使用 `download_wiki.py` 下载最新的 Wikipedia dump、抽取文本以及合并文件：

```shell
pip install wikiextractor
cd examples/deepspeed/megatron-gpt/dataset
python download_wiki.py en
```

> 如果下载失败，请过一段时间重试，或使用网络代理。抽取文本需要花费一些时间，请耐心等待。

使用现有数据集重新训练类 GPT-2 tokenzier：

```shell
cd ../tokenizer
python train_tokenizer.py ../dataset/wiki-en/all wiki-en-tokenizer
```

接着执行 `preprocess_wiki.sh` 脚本预处理数据集：

```shell
cd ../dataset
./preprocess_wiki.sh
```

## 训练

### 125M

使用 `gpt-125m.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
cd ~/examples/deepspeed/megatron-gpt/training
kubectl create -f gpt-125m.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 1（第 89 行）。
* 每个副本的进程数量（第 19 行）和 GPU 数量（第 106 和 110 行）同为 1。
* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 检查点的保存/加载路径同为 `output/gpt-125m/model`（第 68 和 69 行）；TensorBoard 日志的保存路径为 `output/gpt-125m/tensorboard`（第 75 行）。
* 训练占用显存 ~31GB，修改以下参数可以减小显存占用，以防止 OOM：
    * 减小 `--micro-batch-size`（第 49 行）。
    * 启用 activation checkpointing（第 54 和 85 行），但会损害性能。
* 镜像 `t9kpublic/megatron-deepspeed:23.05-py3`（第 94 行）由 [Dockerfile](./docker/Dockerfile) 定义。

### 1.3B

使用 `gpt-1-3b.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
kubectl create -f gpt-1-3b.yaml
```

除了与模型结构有关的超参数之外，与 `gpt-125m.yaml` 的不同之处在于：

* 每个副本的进程数量（第 19 行）和 GPU 数量（第 106 和 110 行）改为 4 以进行数据并行训练，同时适当增大 CPU 和内存的量（第 104、105、108、109 行）。
* 训练占用显存 ~39GB，修改以下参数可以减小显存占用，以防止 OOM：
    * 减小 `--micro-batch-size`（第 49 行）。
* ZeRO stage 改为 2（第 50 行）以减小数据并行训练中每个 GPU 的显存使用。
* 启用 activation checkpointing（第 54 和 85 行）。

### 13B

使用 `gpt-13b.yaml` 创建 DeepSpeedJob 以执行训练：

```shell
kubectl create -f gpt-13b.yaml
```

除了与模型结构有关的超参数之外，与 `gpt-1-3b.yaml` 的不同之处在于：

* 训练副本（replica）数量改为 2（第 89 行），每个副本的进程数量（第 19 行）和 GPU 数量（第 106 和 110 行）改为 8 以进行数据并行+流水线并行训练，同时适当增大 CPU 和内存的量（第 104、105、108、109 行）。
* 训练占用显存 ~70GB，修改以下参数可以减小显存占用，以防止 OOM：
    * 减小 `--micro-batch-size`（第 49 行）。
* 流水线并行度改为 4（第 53 行，同时注释第 52 行）以启用流水线并行。
* ZeRO stage 改为 1（第 50 行）以防止与流水线并行并用时损害性能。
* 启用 activation checkpointing（第 54 和 85 行）。

### 175B

### 训练日志

训练过程中会记录训练和验证损失、学习率/损失缩放倍数/运行时间等变量随步数的变化、所有超参数（命令行参数）的值等信息，并保存为 TensorBoard 日志文件。这里以 125M 为例，查看其训练日志。启动训练之后，操作 Notebook 的左侧导航目录中来到 `output/gpt-125m/tensorboard` 路径，可以看到这里存在一个刚创建的日志文件，然后创建 TensorBoard 实例以查看可视化内容：

![](./assets/create-tensorboard.png)

![](./assets/view-tensorboard.png)

关于所有 TensorBoard 日志相关的命令行参数以及它们的含义，请参考[这里](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/arguments.py#L687-L718)。

### 性能分析

PyTorch 提供的 [Profiler](https://pytorch.org/docs/stable/profiler.html) 工具允许在训练和推断过程中收集性能指标，从而帮助用户更好地理解哪些模型运算最耗费计算和内存资源，检查它们的输入形状和堆栈跟踪，研究设备内核活动并可视化执行追踪。

这里以 125M 为例，使用 PyTorch Profiler 对其训练进行性能分析。在训练之前，先 hack Megatron-DeepSpeed 的训练代码以增加 Profiler 的上下文管理器（将记录 training step 2-4 以及 7-9 的算子执行、GPU 内核时间、内存分配、调用栈、FLOPS、模块等信息，并生成 TensorBoard 日志文件）：

```shell
cd ~
cp examples/deepspeed/megatron-gpt/profiling/training.py Megatron-DeepSpeed/megatron/training.py
```

启动训练之后，操作 Notebook 的左侧导航目录中来到 `output/gpt-125m/profiling` 路径，可以看到这里存在两个刚创建的日志文件，然后创建 TensorBoard 实例以查看可视化内容：

![](./assets/create-profiling.png)

![](./assets/view-profiling-overview.png)

![](./assets/view-profiling-gpu-kernel.png)

![](./assets/view-profiling-memory.png)

## 文本生成

训练完成之后可以使用保存的检查点进行文本生成，还是以 125M 为例。首先 hack Megatron-LM 的文本生成相关的代码，并复制检查点文件到指定位置：

```shell
cd ~
cp examples/deepspeed/megatron-gpt/inference/run_text_generation_server.py Megatron-LM/tools/run_text_generation_server.py
mkdir -p output/gpt-125m/model/iter_0004769/mp_rank_00 && cp output/gpt-125m/model/global_step4769/mp_rank_00_model_states.pt output/gpt-125m/model/iter_0004769/mp_rank_00/model_optim_rng.pt
```

使用 `inference/inference-job.yaml` 创建一个启用 debug 模式的 Job，将其创建的处于空闲状态的 Pod 作为工作空间。进入该 Pod，执行 `inference/server-125m.sh` 脚本以启动服务：

```shell
kubectl create -f examples/deepspeed/megatron-gpt/inference/inference-job.yaml
kubectl exec -it gpt-inference-worker-0 -- bash

./examples/deepspeed/megatron-gpt/inference/server-125m.sh
```

打开一个新的终端，同样进入该 Pod，执行 `Megatron-LM/tools/text_generation_cli.py` 脚本以发送推理请求：

```shell
kubectl exec -it gpt-inference-worker-0 -- bash

python Megatron-LM/tools/text_generation_cli.py localhost:5000
# 根据提示输入
```

### Roadmap

* [ ] 训练 175B 模型
* [ ] 支持 13B、175B 模型的文本生成
* [ ] 增加更多数据集
* [ ] 增加错误重启机制的说明与示例
