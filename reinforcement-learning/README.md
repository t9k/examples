# 强化学习

| Area     | Benchmark              | Dataset | Quality Target              | Reference Implementation Model    |
| -------- | ---------------------- | ------- | --------------------------- | --------------------------------- |
| Research | Reinforcement learning | Go      | 50% win rate vs. checkpoint | Mini Go (based on Alpha Go paper) |

## 数据集

切换到当前目录下，参照步骤 [Build docker and prepare dataset](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/minigo/implementations/tensorflow-22.09#build-docker-and-prepare-dataset)，创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/mlperf-examples/reinforcement-learning
kubectl apply -f download_dataset_pod.yaml
```

进入该 Pod 并执行操作：

```shell
kubectl exec -it mlperf-minigo-download-dataset -- bash

# set up http proxy if necessary
# export http_proxy=<proxy-server>
# export https_proxy=<proxy-server>

# install gsutil
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-408.0.1-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# download dataset
bash
cd minigo
gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
tar xfz checkpoint.tar.gz -C ml_perf/

# download the target model
mkdir -p ml_perf/target/
gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/

# freeze the target model
# comment out L331 in dual_net.py before running freeze_graph.
# L331 is: `optimizer = hvd.DistributedOptimizer(optimizer)`
# Horovod is initialized via train_loop.py and isn't needed for this step.
CUDA_VISIBLE_DEVICES=0 python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target
mv ml_perf/target/target.minigo ml_perf/target/target.minigo.tf

# take up 2.6GB of storage space
mkdir /pvc/minigo_data/
cp -a ml_perf/target /pvc/minigo_data/
cp -a ml_perf/checkpoints/mlperf07 /pvc/minigo_data/
```

完成后删除该 Pod：

```shell
kubectl delete pod mlperf-minigo-download-dataset
```

## 训练

使用 `mpijob.yaml` 创建 MPIJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 12-15 行），并修改队列名称（第 14 行，默认为 `default`））：

```shell
kubectl apply -f mpijob.yaml
```

该模型在训练指定个 epoch 之后才会开始顺序对每个 epoch 保存的模型进行评估（而不是每个 epoch 结束后立即评估），当指标 `eval_accuracy`（对于检查点的胜率）达到 0.5 时评估结束。

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/minigo/implementations/tensorflow-22.09)的 README 以及源代码。

## 资源需求和指标

配置文件 `mpijob.yaml` 需要 64 个 CPU（核心）（若 CPU 核心数不足，则减小该值，通过修改第 68 行）、384 Gi 内存以及 4 个 GPU（显存不少于 20G）。训练此模型时主要瓶颈在于 CPU，可视具体情况适当修改以下参数：

* `procs_per_gpu`：若 CPU 未满载，且 GPU 有空余的显存，则可适当增大，否则适当减小。
* `ranks_per_node`：固定为 `procs_per_gpu` + 1。

请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/minigo/implementations/tensorflow-22.09)中针对不同硬件的配置。

默认配置的运行时间主要取决于 CPU 的性能。作为参考，[v2.1 Results](https://mlcommons.org/en/training-normal-21/) 中的类似机器相比长了很多（Dell R750xax4A100-PCIE-80GB：Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz x2，NVIDIA A100-PCIe-80GB x4，花费时间 516min）。
