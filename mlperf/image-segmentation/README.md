# 图像分割（医学）

## benchmark 概况

| Area   | Benchmark                    | Dataset | Quality Target        | Reference Implementation Model |
| ------ | ---------------------------- | ------- | --------------------- | ------------------------------ |
| Vision | Image segmentation (medical) | KiTS19  | 0.908 Mean DICE score | 3D U-Net                       |

## 运行

切换到当前目录下，使用 `workflow_<GPU_NUM>GPU<GPU_MEMORY>.yaml` 创建从下载、预处理数据集到训练的完整工作流（如果使用队列，取消对于调度器配置的注释（第 72-75 行），并修改队列名称（第 74 行，默认为 `default`））：

```shell
# cd into current directory
cd ~/examples/mlperf/image-segmentation
# vim workflow_<GPU_NUM>GPU<GPU_MEMORY>.yaml  # optionally uncomment config of scheduler (line 72-75)
                                              # and fill in name of queue (line 74) if use queue
kubectl apply -f workflow_<GPU_NUM>GPU<GPU_MEMORY>.yaml
```

创建 WorkflowRun 之后，前往工作流控制台查看其运行进度。第一次运行时，拉取镜像可能花费较长时间。

## 资源需求和指标

配置文件的名称服从格式 `workflow_<GPU_NUM>GPU<GPU_MEMORY>.yaml`，例如 `workflow_4GPU40GB.yaml` 表示训练时启动 4 个工作器，每个工作器需要 1 个至少 40GB 显存的 GPU。除此之外，每个工作器还需要 4 个 CPU（核心）以及 16Gi 内存。不同配置文件启动的训练只有 batch size 和工作器数量上的差别。

下表给出了各个配置的总资源需求量以及参考运行时间：

| 配置文件            | CPU 需求量（核心数） | 内存需求量 | GPU 需求量                                    | 参考运行时间                    |
| ------------------- | -------------------- | ---------- | --------------------------------------------- | ------------------------------- |
| `job_4GPU40GB.yaml` | 16                   | 64GiB      | 4 GPU with 40GB+ memory (e.g. A100-40GB, A40) | ~60min (A100-40GB), ~85min(A40) |
| `job_8GPU40GB.yaml` | 32                   | 128GiB     | 8 GPU with 40GB+ memory                       | ~50min(A40)                     |
| `job_4GPU80GB.yaml` | 16                   | 64GiB      | 4 GPU with 80GB+ memory (A100-80GB)           | ~50-60min                       |
| `job_8GPU80GB.yaml` | 32                   | 128GiB     | 8 GPU with 80GB+ memory                       |                                 |

需要指出的是，这里的参考运行时间仅代表最好情况。训练过程中在指标收敛到 0.900 左右之后，其在哪一个 epoch 能达到 0.908 具有相当的不确定性。

## 细节

### YAML 配置

配置文件 `workflow_<GPU_NUM>GPU<GPU_MEMORY>.yaml` 定义了如下的工作流：

* WorkflowTemplate `mlperf-segmentation-dataset`：下载并预处理数据集。
* WorkflowTemplate `mlperf-segmentation-train`：创建 MPIJob 以进行模型的分布式训练。
* WorkflowTemplate `mlperf-segmentation`：顺序执行上面两个 WorkflowTemplate。
* WorkflowRun `mlperf-segmentation`：运行 WorkflowTemplate `mlperf-segmentation`。

### 数据集

使用的数据集为 KiTS19，这是 [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) 提供的数据集。

WorkflowTemplate `mlperf-segmentation-dataset` 负责下载和预处理数据集，其执行的脚本参照步骤 [Steps to download and verify data](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/unet3d/implementations/mxnet-22.04#steps-to-download-and-verify-data) 并进行了一定的修改。

如果在下载数据集这一步卡住，请考虑使用 HTTP/HTTPS 代理服务器，通过修改配置文件的第 26-27 行。

### 训练

WorkflowTemplate `mlperf-segmentation-train` 创建一个 MPIJob 以启动训练。

当指标 `eval_accuracy`（Mean DICE score）达到 0.908 时训练结束。
