# MLPerf Training benchmark 示例

MLPerf™ Training benchmark 是一个机器学习领域的基准测试套件，旨在针对现代机器学习工作负载公正、准确、全面地评估系统性能。更多详细信息请参阅官方网站 [MLCommons v2.1 Results](https://mlcommons.org/en/training-normal-21/)以及论文 [MLPerf Training Benchmark](https://mlcommons.org/en/training-normal-21/)。

本项目提供 MLPerf Training 最新版本（当前为 v2.1）的各个 benchmark 在 TensorStack AI 计算平台上的运行方法。所有模型的实现均采用 [NVIDIA 的提交](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA)。

## 使用前提

TensorStack AI 计算平台所在的 Kubernetes 集群需要：

* 拥有足够的 AI 计算资源：拥有至少一个这样的节点，其拥有至少 4 个 A100 或同等级别的 GPU，一定量的 CPU（核心数）和内存（不同的 benchmark 使用的 CPU 和内存资源量不同，请参照各个 benchmark 的 README）。
* 拥有足够的存储资源：数个 TiB 的存储容量。
* 连接到互联网。

## 使用方法

1. （可选）创建一个队列或选用一个现有队列，其应具有**充足的资源**和**适当的设置**。

    这里**充足的资源**指队列有足以运行相应 benchmark 的 CPU 额度、内存额度、扩展资源 `nvidia.com/gpu` 等资源配额。不同的 benchmark 以及不同的配置需要不同的资源配额，请参阅各个 benchmark 的 README 文档。

    **适当的设置**指队列的优先级、是否可被抢占以及节点筛选的设置。节点筛选一般应添加适当的标签匹配规则，以限制队列可以使用的节点，例如可以匹配 `sched.tensorstack.dev/accelerator: nvidia-gpu`、`sched.tensorstack.dev/node-alloc-mode: nvidia`、`nvidia.com/gpu.product: NVIDIA-A100-SXM-80GB`、`nvidia.com/gpu.memory: 81920` 等标签。

    这一步是可选的，是否执行这一步取决于您是否需要对作业调度进行控制。例如，如果您不希望 Job 的工作器被调度到某一些节点上，那么这一步就是必要的。

1. 在您的项目中创建一个名为 mlperf、大小 2 Ti 的 PVC，然后创建一个同样名为 mlperf 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

1. 进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库。

```shell
cd ~
git clone https://github.com/t9k/mlperf-examples.git
```

1. 继续使用 **Notebook 的终端**，参照各个 benchmark 的 README 进行操作。
