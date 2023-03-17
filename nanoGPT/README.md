# nanoGPT

WIP

## 使用前提

## 使用方法

如果您想要完整地训练一个 GPT-2 模型，请参阅[训练 GPT-2](#训练-gpt-2)；如果您只是想要进行训练的测试，请参阅[训练 GPT-2（测试）](#训练-gpt-2测试)。

### 训练 GPT-2

创建一个名为 nanogpt、大小 200 GiB 的 PVC，然后创建一个同样名为 nanogpt 的 Notebook 挂载该 PVC，镜像选择 PyTorch 类型，资源模板选择 large（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库，然后切换到当前目录下：

```shell
cd ~
git clone https://github.com/t9k/examples.git
cd examples/nanoGPT
```

#### 数据集

使用 openwebtext 作为训练数据集。安装必要的库，然后下载和预处理数据集：

```shell
pip install tiktoken datasets -i https://pypi.douban.com/simple/
python data/openwebtext/prepare.py
```

#### 训练

使用 `job_rdma.yaml` 创建 PyTorchTrainingJob 以启动训练，你可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* （`master` 之外的）工作器数量在第 68 行定义（默认为 `7`，加上 `master` 共 8 个工作器）。
* 如要所有工作器尽可能地分配到同一个节点上，取消第 18-26、74-82 行的注释。

```shell
kubectl create -f job_rdma.yaml
```

### 训练 GPT-2（测试）

创建一个名为 nanogpt、大小 20 GiB 的 PVC，然后创建一个同样名为 nanogpt 的 Notebook 挂载该 PVC，镜像选择 PyTorch 类型，资源模板选择 small（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库，然后切换到当前目录下：

```shell
cd ~
git clone https://github.com/t9k/examples.git
cd examples/nanoGPT
```

#### 数据集

使用 openwebtext-10k 作为训练数据集。安装必要的库，然后下载和预处理数据集：

```shell
pip install tiktoken datasets -i https://pypi.douban.com/simple/
python data/openwebtext-10k/prepare.py
```

#### 训练

使用 `job_rdma_test.yaml` 创建 PyTorchTrainingJob 以启动训练，你可以如下修改训练配置：

* 如要使用队列，取消第 6-9 行的注释，并修改第 8 行的队列名称（默认为 `default`）。
* （`master` 之外的）工作器数量在第 68 行定义（默认为 `7`，加上 `master` 共 8 个工作器）。
* 如要所有工作器尽可能地分配到同一个节点上，取消第 18-26、74-82 行的注释。

```shell
kubectl create -f job_rdma_test.yaml
```

### 模型超参数和资源需求

## 参考
