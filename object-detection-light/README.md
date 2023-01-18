# 目标检测（轻量级）

## benchmark 概况

| Area   | Benchmark                       | Dataset     | Quality Target | Reference Implementation Model |
| ------ | ------------------------------- | ----------- | -------------- | ------------------------------ |
| Vision | Object detection (light weight) | Open Images | 34.0% mAP      | RetinaNet                      |

## 数据集

切换到当前目录下，参照步骤 [Download dataset](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/ssd/implementations/pytorch#download-dataset)，创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/mlperf-examples/object-detection-light
kubectl apply -f download_dataset_pod.yaml
```

进入该 Pod 并执行操作：

```shell
kubectl exec -it mlperf-detector-download-dataset -- bash

# set up http proxy if necessary
# export http_proxy=<proxy-server>
# export https_proxy=<proxy-server>

# download dataset
# take a very long time and take up 349GB of storage space
pip install fiftyone
mkdir /pvc/open_images
cd ./public-scripts
./download_openimages_mlperf.sh -d /pvc/open_images
```

完成后删除该 Pod：

```shell
kubectl delete pod mlperf-detector-download-dataset
```

## 训练

使用 `trainingjob.yaml` 创建 PyTorchTrainingJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 6-9 行），并修改队列名称（第 8 行，默认为 `default`））：

```shell
kubectl apply -f trainingjob.yaml
```

<!-- 当指标 `eval_accuracy` -->

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/ssd/implementations/pytorch-22.09)的 README 以及源代码。

## 资源需求和指标

配置文件 `trainingjob.yaml` 需要 32 个 CPU（核心），256Gi 内存以及 4 个 NVIDIA A100-SXM/PCIe-40GB。若您的 GPU 显存大于（或小于）40G，则可适当增大（或减小）参数 `batch_size`。请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/ssd/implementations/pytorch-22.09)中针对不同硬件的配置。

该配置的运行时间参考值为 ~6-8h，长于 [v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器（NVIDIA A100-PCIe-80GB 300W x4，花费时间 ~3-4h）。
