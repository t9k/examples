# NCCL Tests

[NCCL Tests](https://github.com/NVIDIA/nccl-tests) 用于检查 [NCCL](http://github.com/nvidia/nccl) 操作的正确性和性能。这里提供该测试在平台上的运行方法。

## 使用方法

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```shell
cd ~
git clone https://github.com/t9k/examples.git
```

使用 `nccl-test-*.yaml` 创建 MPIJob 以进行测试，这里以 `nccl-test-2x8.yaml` 为例，后缀的 `2x8` 表示启动 2 个副本，每个副本请求 8 个 GPU：

```shell
cd ~/examples/nccl-tests
kubectl create -f nccl-test-2x8.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 2（第 14 行）。
* 每个副本的进程数量为 1（第 15 行 `-N`），GPU 数量为 8（第 30 行）；每个进程的线程数量为 8（第 16 行 `-g`），即每个线程 1 个 GPU。
* 如要启用调试模式，取消第 8-8 行的注释。
* 如要使用队列，取消第 9-12 行的注释，并修改第 11 行的队列名称（默认为 `default`）。
* 镜像 `t9kpublic/nccl-tests:main`（第 94 行）由当前目录下的 Dockerfile 定义。
* 关于第 15 行的参数，请参阅 [mpirun man page](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php) 和 [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)；关于第 16 行的参数，请参阅 [arguments](https://github.com/NVIDIA/nccl-tests?tab=readme-ov-file#arguments)。
