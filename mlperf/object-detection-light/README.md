# 目标检测（轻量级）

## benchmark 概况

| Area   | Benchmark                       | Dataset     | Quality Target | Reference Implementation Model |
| ------ | ------------------------------- | ----------- | -------------- | ------------------------------ |
| Vision | Object detection (light weight) | Open Images | 34.0% mAP      | RetinaNet                      |

## 数据集

切换到当前目录下，参照步骤 [Download dataset](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/ssd/implementations/pytorch#download-dataset)，创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/examples/mlperf/object-detection-light
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
pip install fiftyone==0.18.0
mkdir /pvc/open_images
cd ./public-scripts
./download_openimages_mlperf.sh -d /pvc/open_images
```

完成后删除该 Pod：

```shell
kubectl delete pod mlperf-detector-download-dataset
```

## 训练

使用 `job_<WORKER_NUM>GPU<GPU_MEMORY>.yaml` 创建 PyTorchTrainingJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 6-9 行），并修改队列名称（第 8 行，默认为 `default`））：

```shell
kubectl create -f job_<WORKER_NUM>GPU<GPU_MEMORY>.yaml
```

当指标 `eval_accuracy`（mAP）达到 0.34 时训练结束。

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/ssd/implementations/pytorch-22.09)的 README 以及源代码。

## 资源需求和指标

配置文件的名称服从格式 `job_<WORKER_NUM>GPU<GPU_MEMORY>.yaml`，例如 `job_4GPU40GB.yaml` 表示启动 4 个工作器，每个工作器需要 1 个至少 40GB 显存的 GPU。除此之外，每个工作器还需要 8 个 CPU（核心）以及 80Gi 内存。不同配置文件启动的训练只有 batch size 和工作器数量上的差别。

下表给出了各个配置的总资源需求量以及参考运行时间：

| 配置文件            | CPU 需求量（核心数） | 内存需求量 | GPU 需求量                                             | 参考运行时间                   |
| ------------------- | -------------------- | ---------- | ------------------------------------------------------ | ------------------------------ |
| `job_4GPU24GB.yaml` | 32                   | 320GiB     | 4 GPU with 24GB+ memory (e.g. RTX 3090, RTX 4090, A30) | (not reach target)             |
| `job_8GPU24GB.yaml` | 64                   | 640GiB     | 8 GPU with 24GB+ memory                                | ~4.5-5.5h (A30)                |
| `job_4GPU40GB.yaml` | 32                   | 320GiB     | 4 GPU with 40GB+ memory (A100-40GB, A40)               | ~3.5h (A100-40GB), ~6-9h (A40) |
| `job_8GPU40GB.yaml` | 64                   | 640GiB     | 8 GPU with 40GB+ memory                                | ~4.5-5.5h (A40)                |
| `job_4GPU80GB.yaml` | 32                   | 320GiB     | 4 GPU with 80GB+ memory (A100-80GB)                    | ~3-3.5h                        |
| `job_8GPU80GB.yaml` | 64                   | 640GiB     | 8 GPU with 80GB+ memory                                | ~80min-2h                      |
