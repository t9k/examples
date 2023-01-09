# 图像分割（医学）

## benchmark 概况

| Area   | Benchmark                    | Dataset | Quality Target        | Reference Implementation Model |
| ------ | ---------------------------- | ------- | --------------------- | ------------------------------ |
| Vision | Image segmentation (medical) | KiTS19  | 0.908 Mean DICE score | 3D U-Net                       |

## 运行

切换到当前目录下，使用 `workflow_DGXA100_40GB_4.yaml` 创建从下载、预处理数据集到训练的完整工作流：

```shell
# cd into current directory
cd ~/mlperf-examples/image-segmentation
kubectl apply -f workflow_DGXA100_40GB_4.yaml
```

文件名中的 `DGXA100_40GB` 表示配置适用于 NVIDIA A100 GPU（40GB 显存），`4` 表示 GPU 数量。如果您的硬件有所不同，请参阅[配置和指标](#配置和指标)部分以修改配置。

创建 WorkflowRun 之后，前往工作流控制台查看其运行进度。第一次运行时，拉取镜像可能花费较长时间。

## 细节

### 数据集

使用的数据集为 KiTS19，这是 [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) 提供的数据集。

WorkflowTemplate `mlperf-segmentation-dataset` 负责下载和预处理数据集，其执行的脚本参照步骤 [Steps to download and verify data](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/unet3d/implementations/mxnet-22.04#steps-to-download-and-verify-data) 并进行了一定的修改。

如果在下载数据集这一步卡住，请考虑使用 HTTP/HTTPS 代理服务器，通过修改 YAML 配置文件的第 26-27 行。

### 训练

WorkflowTemplate `mlperf-segmentation-train` 创建一个 MPIJob 以启动训练。

当指标 `eval_accuracy`（Mean DICE score）达到 0.908 时训练结束。

### 配置和指标

默认配置 `workflow_DGXA100_40GB_4.yaml` 适用于设备 NVIDIA A100-SXM/PCIe-40GB x4 的情形。若显存大于（或小于）40G 则可适当增大（或减小）参数 `batch_size` 和 `val_batch_size`。请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/unet3d/implementations/mxnet-22.04)中针对不同硬件的配置。

默认配置的运行时间参考值为 ~60min，接近 [v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器（NVIDIA A100-PCIe-80GB x4，花费时间 ~50-60min）。
