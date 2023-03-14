# 强化学习

## benchmark 概况

| Area     | Benchmark              | Dataset | Quality Target              | Reference Implementation Model    |
| -------- | ---------------------- | ------- | --------------------------- | --------------------------------- |
| Research | Reinforcement learning | Go      | 50% win rate vs. checkpoint | Mini Go (based on Alpha Go paper) |

## 数据集

切换到当前目录下，参照步骤 [Build docker and prepare dataset](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/minigo/implementations/tensorflow-22.09#build-docker-and-prepare-dataset)，创建一个用于下载和预处理数据集的 Pod：

```shell
# cd into current directory
cd ~/examples/mlperf/reinforcement-learning
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

使用 `job_<GPU_NUM>GPU_<CPU_NUM>CPU.yaml` 创建 MPIJob 以启动训练（如果使用队列，取消对于调度器配置的注释（第 12-15 行），并修改队列名称（第 14 行，默认为 `default`））：

```shell
kubectl apply -f mpijob.yaml
```

该模型在训练指定个 epoch 之后才会开始顺序对每个 epoch 保存的模型进行评估（而不是每个 epoch 结束后立即评估），当指标 `eval_accuracy`（对于检查点的胜率）达到 0.5 时评估结束。

更多选项设置请参照[原项目](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/minigo/implementations/tensorflow-22.09)的 README 以及源代码。

## 资源需求和指标

配置文件的名称服从格式 `job_<GPU_NUM>GPU_<CPU_NUM>CPU.yaml`，例如 `job_4GPU_32CPU.yaml` 表示启动 4 个工作器，每个工作器需要 16 个 CPU（核心）、96GiB 内存以及 1 个至少 20GB 显存的 GPU。不同配置文件启动的训练只有启动进程数量上的差别。

下表给出了各个配置的总资源需求量以及参考运行时间：

| 配置文件               | CPU 需求量（核心数） | 内存需求量 | GPU 需求量              | 参考运行时间 |
| ---------------------- | -------------------- | ---------- | ----------------------- | ------------ |
| `job_4GPU_64CPU.yaml`  | 64                   | 384GiB     | 4 GPU with 20GB+ memory | ~8h          |
| `job_8GPU_128CPU.yaml` | 128                  | 768GiB     | 8 GPU with 20GB+ memory | ~3h          |

训练此模型时的主要瓶颈在于 CPU，运行时间主要取决于 CPU 的性能，因此实际运行时间可能远长于参考运行时间。若可用的 CPU 核心数不足，可以适当减小 CPU 需求量（通过修改配置文件的第 68 行），尽管这会降低训练速度。
