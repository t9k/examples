# 图像分割（医学）

## benchmark 概况

| Area   | Benchmark                    | Dataset | Quality Target        | Reference Implementation Model |
| ------ | ---------------------------- | ------- | --------------------- | ------------------------------ |
| Vision | Image segmentation (medical) | KiTS19  | 0.908 Mean DICE score | 3D U-Net                       |

## 运行

切换到当前目录下，使用 `workflow_DGXA100_40GB_4.yaml` 创建从下载、预处理数据集到训练的完整工作流（如果使用队列，取消对于调度器配置的注释（第 72-75 行），并修改队列名称（第 74 行，默认为 `default`））：

```shell
# cd into current directory
cd ~/mlperf-examples/image-segmentation
# vim workflow_DGXA100_40GB_4.yaml  # optionally uncomment config of scheduler (line 72-75)
                                    # and fill in name of queue (line 74) if use queue
kubectl apply -f workflow_DGXA100_40GB_4.yaml
```

文件名中的 `DGXA100_40GB` 表示配置适用于 NVIDIA A100 GPU（40GB 显存），`4` 表示 GPU 数量。如果您的硬件有所不同，请参阅[资源需求和指标](#资源需求和指标)部分以修改配置。

创建 WorkflowRun 之后，前往工作流控制台查看其运行进度。第一次运行时，拉取镜像可能花费较长时间。

## 细节

### YAML 配置

配置文件 `workflow_DGXA100_40GB_4.yaml` 定义了如下的工作流：

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

### 资源需求和指标

配置文件 `workflow_DGXA100_40GB_4.yaml` 需要 16 个 CPU（核心），32Gi 内存以及 4 个 NVIDIA A100-SXM/PCIe-40GB。若您的 GPU 显存大于（或小于）40G，则可适当增大（或减小）参数 `batch_size` 和 `val_batch_size`。请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/unet3d/implementations/mxnet-22.04)中针对不同硬件的配置。

该配置的运行时间参考值为 ~60min，接近 [v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器（NVIDIA A100-PCIe-80GB x4，花费时间 ~50-60min）。
