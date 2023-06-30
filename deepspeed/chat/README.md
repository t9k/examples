# Chat

## 介绍

DeepSpeed Chat 是微软开发的一个通用系统框架，它可以方便地在多种预训练大型语言模型上实施 InstructGPT 风格的 3 阶段训练，生成高质量 ChatGPT 类型模型。 本文档使用 T9K 平台的 DeepSpeedJob 完成 DeepSpeed Chat。

## 准备工作

### 创建 PVC 和 Notebook

通过 Build Console 创建一个 PVC，用来存储训练脚本、实验数据和模型。
PVC 创建信息如下：
- 名称：`gpt`，用户也可以使用其他名称，但是在后续使用 PVC 的过程中需要同步修改
- PVC 大小需要根据模型的规模来决定：
  - facebook/opt-1.3b 至少 20 GiB
  - facebook/opt-13b 至少 100 GiB
  - facebook/opt-66b 至少 500 GiB

通过 Build Console 创建一个 Notebook，用来下载训练脚本、实验数据和启动训练。Notebook 需绑定上述创建的 PVC。
后续操作全部是在 Notebook 中进行。

### 下载训练脚本

进入 Notebook，启动一个 Terminal。
输入如下命令，下载训练脚本：

```
git clone -b lmh/update https://gitlab.dev.tensorstack.net/t9k/ds-operator
```

### [可选] 下载预训练模型和数据集

在训练过程中，训练脚本会自动下载预训练模型和数据集，但是由于网络问题，下载过程经常被中断。用户可以按照本节操作下载模型和数据集，避免训练过程出错。

启动一个 Debug 模式的 DeepSpeedJob：

```
kubectl create -f ./ds-operator/config/samples/chat/download/download-job.yaml
```

进入 DeepSpeedJob，并下载模型和数据集：
   
```
# 进入 DeepSpeedJob 的工作负载
kubectl exec -it download-worker-0 -- bash

# 下载预训练模型
transformers-cli download facebook/opt-1.3b
# transformers-cli download facebook/opt-13b
# transformers-cli download facebook/opt-66b

# 下载数据集
python /t9k/mnt/ds-operator/config/samples/chat/download/load_dataset.py Dahoas/rm-static [-o /t9k/mnt/hf-datasets/Dahoas_rm_static]
```

- 下载过程中可能因为网络问题报错，可以通过多次尝试或设置代理的方式解决
- 上述过程，只下载了 `Dahoas/rm-static` 数据集，如果您希望下载更多数据集，替换数据集名称 `Dahoas/rm-static` 和下载路径 `/t9k/mnt/hf-datasets/Dahoas_rm_static` 即可

#### 下载失败

如果在上述步骤中一直失败，说明用户当前环境与 Huggingface 之间存在问题。用户可以尝试以下方式：
- 访问 Huggingface 网页，找到对应文件的 URL
- 使用 wget 来下载文件，下载时可以使用合适的 proxy

以前面下载的模型和数据集为例：

```
# Dahoas/rm-static
mkdir -p /t9k/mnt/hf-datasets/download/Dahoas_rm_static
cd /t9k/mnt/hf-datasets/download/Dahoas_rm_static

wget https://huggingface.co/datasets/Dahoas/rm-static/resolve/main/dataset_infos.json
wget -P data https://huggingface.co/datasets/Dahoas/rm-static/resolve/main/data/train-00000-of-00001-2a1df75c6bce91ab.parquet
wget -P data https://huggingface.co/datasets/Dahoas/rm-static/resolve/main/data/test-00000-of-00001-8c7c51afc6d45980.parquet

# 手动下载的 dataset 文件需要使用脚本将其转换为适当的形式保存下来，才能用于后续训练
# 此处 -o 指定的路径是后续训练时填写在 --data_path 的路径
python /t9k/mnt/ds-operator/config/samples/chat/download/save_dataset.py /t9k/mnt/hf-datasets/download/Dahoas_rm_static [-o /t9k/mnt/hf-datasets/Dahoas_rm_static]

# facebook/opt-1.3b
mkdir -p /t9k/mnt/hf-models/facebook_opt_1.3b
cd /t9k/mnt/hf-models/facebook_opt_1.3b

wget https://huggingface.co/facebook/opt-1.3b/resolve/main/config.json
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/flax_model.msgpack
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/generation_config.json
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/tokenizer_config.json
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/vocab.json
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/merges.txt
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/special_tokens_map.json
wget https://huggingface.co/facebook/opt-1.3b/resolve/main/tf_model.h5
```

## 训练

训练分为三步执行，其中第一步和第二步可以同时进行。

### Step 1 - SFT

我们一共提供了三个训练部署脚本，用来应对三种不同的场景：

```
# 单 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/actor/actor-single-gpu.yaml

# 单节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/actor/actor-single-node.yaml

# 多节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/actor/actor-multi-node.yaml
```

参考 actor-multi-node.yaml 做以下说明：
- 如果希望使用已经下载下来的数据集，需要将 yaml 中的对应数据集替换成本地路径：`Dahoas/rm-static` -> `/t9k/mnt/hf-datasets/Dahoas_rm_static`。同理 `--model_name_or_path` 也可以替换成下载路径。
- 训练中可以使用多个数据集，如上例中同时使用了四个数据集：`Dahoas/rm-static、Dahoas/full-hh-rlhf、Dahoas/synthetic-instruct-gptj-pairwise、yitingxie/rlhf-reward-datasets`，您也可以使用自己的训练集进行特性化调整。
- 当前 DeepSpeedJob 会创建两个训练节点（由 spec.worker.replicas 字段指定），每个节点上有 2 个 GPU（由 nvidia.com/gpu 字段指定），训练时每个节点上会启动两个训练进程（由 spec.config.slotsPerWorker 字段指定）。这里需要注意，节点上的训练进程的数量不应大于 GPU 数量（要求：slotsPerWorker <= nvidia.com/gpu）。
- 模型文件最后会输出到 /t9k/mnt/output/multi-node/actor-models/1.3b 中
- 其他训练参数，请根据设置的资源数量自行调整。

执行以下命令，持续监测任务的状态：

```
% k get pods actor-multi-node-worker-0 -w
NAME                        READY   STATUS            RESTARTS   AGE
actor-multi-node-worker-0   0/1     Init:0/1          0          8s
actor-multi-node-worker-0   0/1     PodInitializing   0          8s
actor-multi-node-worker-0   1/1     Running           0          9s
actor-multi-node-worker-0   0/1     Completed         0          81m
```

当 Status 字段变为 Running，表示 Job 初始化完毕，开始训练
当 Status 字段变为 Completed，表示 Job 执行完毕

执行以下命令，查看训练日志：

```
% k logs actor-multi-node-worker-0 --tail=100
...
actor-multi-node-worker-0: [2023-06-21 03:45:55,424] [INFO] [timer.py:199:stop] epoch=1/micro_step=926/global_step=1880, RunningAvgSamplesPerSec=6.60372389313781, CurrSamplesPerSec=6.368302303901175, MemAllocated=7.79GB, MaxMemAllocated=10.55GB
actor-multi-node-worker-0: [2023-06-21 03:46:19,251] [INFO] [logging.py:96:log_dist] [Rank 0] step=1890, skipped=19, lr=[9.275816201087529e-08, 9.275816201087529e-08], mom=[(0.9, 0.95), (0.9, 0.95)]
actor-multi-node-worker-0: [2023-06-21 03:46:19,254] [INFO] [timer.py:199:stop] epoch=1/micro_step=936/global_step=1890, RunningAvgSamplesPerSec=6.604309943995918, CurrSamplesPerSec=6.809163050305052, MemAllocated=7.79GB, MaxMemAllocated=10.55GB
actor-multi-node-worker-0: [2023-06-21 03:46:43,211] [INFO] [logging.py:96:log_dist] [Rank 0] step=1900, skipped=19, lr=[4.940137226560615e-08, 4.940137226560615e-08], mom=[(0.9, 0.95), (0.9, 0.95)]
actor-multi-node-worker-0: [2023-06-21 03:46:43,214] [INFO] [timer.py:199:stop] epoch=1/micro_step=946/global_step=1900, RunningAvgSamplesPerSec=6.604702857325247, CurrSamplesPerSec=6.819754921642627, MemAllocated=7.79GB, MaxMemAllocated=10.55GB
actor-multi-node-worker-0: ***** Evaluating perplexity, Epoch 2/2 *****
actor-multi-node-worker-0: ppl: 2.159278392791748
actor-multi-node-worker-0: saving the final model ...
actor-multi-node-worker-1: [2023-06-21 03:47:13,571] [INFO] [launch.py:460:main] Process 123 exits successfully.
actor-multi-node-worker-0: [2023-06-21 03:47:17,522] [INFO] [launch.py:460:main] Process 259 exits successfully.
```

从日志中可以查看整个训练过程和训练指标。

### Step 2 - RW

```
# 单 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/reward/reward-single-gpu.yaml

# 单节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/reward/reward-single-node.yaml

# 多节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/reward/reward-multi-node.yaml
```

相关说明，参考 Step 1

检查训练进程和结果，参考 Step 1 。


### Step 3 - RLHF

```
# 单 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/rlhf/rlhf-single-gpu.yaml

# 单节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/rlhf/rlhf-single-node.yaml

# 多节点多 gpu 训练
kubectl create -f /t9k/mnt/ds-operator/config/samples/chat/rlhf/rlhf-multi-node.yaml
```

actor_model_name_or_path 和 critic_model_name_or_path 分别使用 Step 1 和 2 中产生的模型
其他说明，参考 Step 1

检查训练进程和结果，参考 Step 1 。

