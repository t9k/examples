# nanoGPT

[nanoGPT](https://github.com/karpathy/nanoGPT) 是一个简单、快速地训练和微调中等规模 GPT 模型的项目。代码本身非常简单和易读：`train.py` 是一个约 300 行的样板训练循环，`model.py` 是一个约 300 行的 GPT 模型定义。您可以从头开始训练新模型，或加载并微调预训练检查点。

## 使用前提

TensorStack AI 计算平台所在的集群需要：

* 拥有 1~2 个这样的节点，其拥有至少 4 个数据中心级别的 GPU。
* 连接到互联网。

## 使用方法

如果您想要完整地训练一个 GPT-2 模型，请参阅[训练 GPT-2](#训练-gpt-2)；如果您只是想要进行训练的测试，请参阅[训练 GPT-2（测试）](#训练-gpt-2测试)。

### 训练 GPT-2

创建一个名为 nanogpt、大小 200 GiB 的 PVC，然后创建一个同样名为 nanogpt 的 Notebook 挂载该 PVC，镜像选择 PyTorch 类型，资源模板选择 large（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库，然后切换到当前目录下：

```shell
cd ~
git clone https://github.com/t9k/examples.git
cd examples/nanogpt
```

#### 数据集

使用 [openwebtext](https://huggingface.co/datasets/openwebtext) 作为训练数据集。安装必要的库，然后下载和预处理数据集：

```shell
pip install tiktoken datasets -i https://pypi.douban.com/simple/
python data/openwebtext/prepare.py
```

#### 训练

使用 `job.yaml` 创建 PyTorchTrainingJob 以启动训练，您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 每个节点的进程数量在第 13 行定义（默认为 `"8"`）。
* 节点数量在第 17 行定义（默认为 `1`）。

`job.yaml` 有如下变体，您也可以使用这些配置文件来创建 PyTorchTrainingJob：

* `job_rdma.yaml`：在节点数量大于 1 的情况下，每个节点请求一张 RDMA 网卡，利用 IB 网络加速训练。
* `job_torch1.yaml`：使用 PyTorch 1.13 版本（默认为 2.0 版本）。由于无法使用 `torch.compile()` 函数编译模型，训练速度会更慢。
* `job_test_small.yaml` 和 `job_test_random.yaml`：请参阅[训练 GPT-2（测试）](#训练-gpt-2测试)

```shell
kubectl create -f job.yaml
# or
kubectl create -f job_rdma.yaml
kubectl create -f job_torch1.yaml
```

### 训练 GPT-2（测试）

创建一个名为 nanogpt-test、大小 20 GiB 的 PVC，然后创建一个同样名为 nanogpt-test 的 Notebook 挂载该 PVC，镜像选择 PyTorch 类型，资源模板选择 small（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库，然后切换到当前目录下：

```shell
cd ~
git clone https://github.com/t9k/examples.git
cd examples/nanogpt
```

#### 数据集

如要使用随机数作为数据集，请跳过这一步。

使用 [openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k) 作为训练数据集。安装必要的库，然后下载和预处理数据集：

```shell
pip install tiktoken datasets -i https://pypi.douban.com/simple/
python data/openwebtext-10k/prepare.py
```

#### 训练

使用 `job_test_small.yaml` 或 `job_test_random.yaml` 创建 PyTorchTrainingJob 以启动训练，区别在于前者使用小型数据集而后者直接使用随机数作为数据集。您可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* 每个节点的进程数量在第 13 行定义（默认为 `"8"`）。
* 节点数量在第 17 行定义（默认为 `1`）。

```shell
kubectl create -f job_test_small.yaml
# or
kubectl create -f job_test_random.yaml
```

### 模型超参数和资源需求

本示例设置的模型超参数为：

```python
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
block_size = 1024
vocab_size = 50304
```

总共 124M 个参数。

训练时每个工作器（进程）占用约 13000MiB 显存（对于 PyTorch 1.13 版本为约 31000MiB 显存）。

## 参考

* 原始论文：[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* 原始项目：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
