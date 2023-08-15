# GPT-NeoX

[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 是 EleutherAI 开发的一个在 GPU 上训练大型语言模型的项目。它目前的框架基于 NVIDIA 的 Megatron 语言模型，并结合了 DeepSpeed 技术以及一些新的优化技巧。

本示例使用 DeepSpeedJob 在 TensorStack AI 计算平台上完成 GPT-NeoX 的训练。

## 镜像

使用以下命令制作镜像：

```
docker build -f Dockerfile . -t t9kpublic/deepspeed-neox:23.02-0.10.0-230814
```

该镜像以 `nvcr.io/nvidia/pytorch:23.02-py3` 为基础镜像，安装了：

- deepspeed 包以及启动 deepspeed 命令所必须的包
- GPT-NeoX 训练脚本的依赖包
- GPT-NeoX 训练脚本（[GitHub](https://github.com/EleutherAI/gpt-neox)）

并且修改了 GPT-NeoX 训练脚本中的部分文件：

- 修改 `NeoXArgs` 类：
  - 添加方法 `cosume_t9kdj_args`，用于在使用 DeepSpeedJob 启动训练时解析配置文件。（文件 [arguments.py 397 行](arguments.py#L397)）
  - 修改方法 `calculate_derived`，将其中 GPU 总数量的计算方法替换为从环境中读取 `WORLD_SIZE`。原因是 NeoXArgs 计算 GPU 的方法仅适用于 worker 0（deepspeed 的启动副本），其他 worker 在使用该方法计算的时候会只计算当前副本的 GPU，导致校验不通过。（文件 [arguments.py 910 行](arguments.py#L910)）
- 修改启动脚本 `train.py`:
  - 用 `NeoXArgs.consume_t9kdj_args()` 替换 `NeoXArgs.consume_neox_args()`。（文件 [train.py 23 行](train.py#L23)）
- 添加了更多配置文件到 `configs` 文件夹中，原项目的配置文件可以继续使用。

## 训练

用户可以使用 train 文件夹中的 YAML 文件创建 DeepSpeedJob 进行训练，这些文件的应用场景分别是：

- `single-gpu.yaml`：单 GPU 环境下训练。
- `single-debug.yaml`：单 GPU Debug 环境，该 YAML 可以用来准备训练环境，比如向 PVC 中下载数据集。
- `multi-gpu.yaml`：单节点多 GPU 环境下训练。
- `multi-nodes.yaml`：多节点多 GPU 环境下训练。
- `multi-nodes-rdma.yaml`：多节点多 GPU 环境下训练，使用 IB 网络进行通信。
- `multi-gpu-checkpoint.yaml`：单节点多 GPU 环境，添加 `--checkpoint_dir` 参数。
- `multi-elastic.yaml`：弹性训练，在训练过程中动态调节节点数量。注意：训练脚本需支持自动保存和读取 checkpoint，以免在节点重启时从头开始训练。
- `multi-nodes-autotune.yaml`：多节点多 GPU 环境下，测试 autotuning 功能。

用户可以在上述 YAML 的基础上进行修改以使用更多功能。

### 训练脚本及参数

用户可以通过 DeepSpeedJob 的 `spec.config.run.pyhon` 字段设置训练启动命令。

GPT-NeoX 的启动命令格式为：`train.py <training-config> [flags]`，支持的 flag 包括：

- `--vocab_file`：指定 vocab 文件路径。
- `--merge_file`：指定 merge 文件路径。
- `--data_path`：指定数据集路径。
- `--checkpoint_dir`：指定 checkpoint 路径，如果不设置该参数则不启用 checkpoint 功能。

### T9k Scheduler

按照以下方式添加 `spec.scheduler` 字段来使用 T9k Scheduler。

```yaml
apiVersion: batch.tensorstack.dev/v1beta1
kind: DeepSpeedJob
metadata:
  name: neox-multi
spec:
  scheduler:
    t9kScheduler:
      queue: a100-neox
      priority: 50
```

### NCCL P2P DISABLE

如果测试环境中发现 P2P 通信存在问题，可以通过添加 `NCCL_P2P_DISABLE` 环境变量禁用 P2P 通信：

```yaml
apiVersion: batch.tensorstack.dev/v1beta1
kind: DeepSpeedJob
metadata:
  name: neox-multi
spec:
  worker:
    template:
      spec:
        containers:
          - image: t9kpublic/deepspeed-neox:23.02-0.10.0-230804
            imagePullPolicy: IfNotPresent
            name: worker
            env:
              - name: NCCL_P2P_DISABLE
                value: "1"
```
