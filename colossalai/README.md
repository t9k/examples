# ColossalAI 示例

ColossalAI 是一个集成系统，为用户提供一套综合的训练方法和一系列的并行技术，旨在让大型模型的分布式训练更加高效、易用、可扩展。更多详细信息请参阅官方网站 [Colossal-AI](https://colossalai.org/) 以及论文 [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)。

本项目提供部分 [ColossalAI 官方示例](https://github.com/hpcaitech/ColossalAI/tree/main/examples)在 TensorStack AI 计算平台上的运行方法。

## 使用前提

TensorStack AI 计算平台所在的 Kubernetes 集群需要：

* 拥有一定的 AI 计算资源：拥有至少一个这样的节点，其拥有一定量的 GPU、CPU（核心数）和内存（不同的示例使用的 GPU 显存、CPU 和内存资源量不同，请参照各个示例的 README）。
* 连接到互联网。

## 使用方法

1. （可选）创建一个队列或选用一个现有队列，其应具有**充足的资源**和**适当的设置**。

    这里**充足的资源**指队列有足以运行相应示例的 CPU 额度、内存额度、扩展资源 `nvidia.com/gpu` 等资源配额。不同的示例以及不同的配置需要不同的资源配额，请参阅各个示例的 README 文档。

    **适当的设置**指队列的优先级、是否可被抢占以及通过节点筛选机制选择的 GPU 型号等。

    这一步是可选的，是否执行这一步取决于您是否需要对作业调度进行控制。例如，如果您不希望 Job 的工作器被调度到某一些节点上，那么这一步就是必要的。

1. 在您的项目中创建一个名为 colossalai、大小 500 GiB 的 PVC（可能需要存储一些模型检查点文件），然后创建一个同样名为 colossalai 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

1. 进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库。

    ```shell
    cd ~
    git clone https://github.com/t9k/examples.git
    ```

1. 继续使用 **Notebook 的终端**，参照各个示例的 README 进行操作。
