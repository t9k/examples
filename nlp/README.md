# 自然语言处理

## benchmark 概况

| Area     | Benchmark | Dataset              | Quality Target        | Reference Implementation Model |
| -------- | --------- | -------------------- | --------------------- | ------------------------------ |
| Language | NLP       | Wikipedia 2020/01/01 | 0.72 Mask-LM accuracy | BERT-large                     |

## 数据集

使用的数据集为截至 2020/01/01 的英文维基百科文章，这是一个自然语言处理领域中常用的数据集。

切换到当前目录下，参照步骤 [Download and prepare the data](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09#download-and-prepare-the-data)，首先创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/mlperf-examples/nlp
kubectl apply -f download_dataset_pod.yaml
```

进入该 Pod 并执行操作：

```shell
kubectl exec -it mlperf-nlp-download-dataset -- bash

# set up HTTP proxy if necessary
# export http_proxy=<proxy-server>
# export https_proxy=<proxy-server>

# download and preprocess dataset and checkpoint
# take a very long time and take up 864GB of storage space
cd /workspace/bert
./input_preprocessing/prepare_data.sh --outputdir /pvc/bert_data

# delete unnecessary files
# take up 159GB of storage space after deleting
rm -rf /pvc/bert_data/download /pvc/bert_data/hdf5/eval /pvc/bert_data/hdf5/training /pvc/bert_data/hdf5/training-4320/hdf5_4320_shards_uncompressed
```

完成后删除该 Pod：

```shell
kubectl delete pod mlperf-nlp-download-dataset
```

## 训练

使用 `trainingjob.yaml` 创建 PyTorchTrainingJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 6-9 行），并修改队列名称（第 8 行，默认为 `default`））：

```shell
kubectl apply -f trainingjob.yaml
```

当指标 `eval_accuracy` 达到 0.72 时训练结束。

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09)的 README 以及源代码。

## 配置和指标

配置文件 `trainingjob.yaml` 需要 16 个 CPU（核心），32Gi 内存以及 4 个 NVIDIA A100-SXM/PCIe-40GB。若您的 GPU 显存大于（或小于）40G，则可适当增大（或减小）参数 `train_batch_size`。请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09)中针对不同硬件的配置。

该配置的运行时间参考值为 ~120min，长于 [v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器（NVIDIA A100-PCIe-80GB 300W x4，花费时间 ~50-55min）。
