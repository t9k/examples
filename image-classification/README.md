# 图像分类

## benchmark 概况

| Area   | Benchmark            | Dataset  | Quality Target        | Reference Implementation Model |
| ------ | -------------------- | -------- | --------------------- | ------------------------------ |
| Vision | Image classification | ImageNet | 75.90% classification | ResNet-50 v1.5                 |

## 数据集

使用的数据集为 ImageNet（ILSVRC2012），这是一个计算机视觉领域中常用的数据集。

切换到当前目录下，参照[步骤 Steps to download and verify data](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/resnet/implementations/mxnet-22.04#steps-to-download-and-verify-data)，首先创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/mlperf-examples/image-classification
kubectl apply -f download_dataset_pod.yaml
```

进入该 Pod 并执行操作：

```shell
kubectl exec -it mlperf-image-classification-download-dataset -- bash

# **IMPORTANT**
# You need to download Training images (Task 1 & 2) and Validation images (all tasks)
# at http://image-net.org/challenges/LSVRC/2012/2012-downloads on your own, which
# requires an account. After that, move files ILSVRC2012_img_train.tar and
# ILSVRC2012_img_val.tar to /pvc/imagenet

# extract dataset
# take up 144GB of storage space
cd /pvc/imagenet
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 
tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# create script file and copy contents from prepare_imagenet.sh
vim prepare_imagenet.sh
chmod 755 prepare_imagenet.sh

# preprocess dataset
# take up another 144GB of storage space
mkdir /pvc/imagenet_processed
./prepare_imagenet.sh /pvc/imagenet /pvc/imagenet_processed
```

完成后删除该 Pod：

```shell
kubectl delete pod mlperf-image-classification-download-dataset
```

## 训练

使用 `mpijob.yaml` 创建 MPIJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 12-15 行），并修改队列名称（第 14 行，默认为 `default`））：

```shell
kubectl apply -f mpijob.yaml
```

当指标 `eval_accuracy` 达到 0.759 时训练结束。

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/resnet/implementations/mxnet-22.04)的 README 以及源代码。

## 资源需求和指标

配置文件 `mpijob.yaml` 需要 32 个 CPU（核心），128Gi 内存以及 4 个 NVIDIA A100-SXM/PCIe-40GB。若您的 GPU 显存大于（或小于）40G，则可适当增大（或减小）参数 `batch-size`。请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/resnet/implementations/mxnet-22.04)中针对不同硬件的配置。

该配置的运行时间参考值为 ~120min，长于 [v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器（NVIDIA A100-PCIe-80GB 300W x4，花费时间 ~60min）。
