# 在模型推理服务中通过 CoreWeave Tensorizer 快速加载模型

[CoreWeave Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 是一个 PyTorch 模块，用于模型和张量的序列化和反序列化。它能以更少的资源和更快的速度从 HTTP/HTTPS 和 S3 端点加载模型。

本示例通过 MLService 部署一个 [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) 大模型，并使用 [CoreWeave Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 加速模型加载。

## 使用方法

创建一个名为 `tensorizer`、大小为 50Gi 的 PVC，然后创建一个同样名为 `tensorizer` 的 Notebook 挂载该 PVC，镜像选择带有 sudo 权限的类型，资源不限（如要使用远程操作，请开启 SSH）。

进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库：

```bash
# change to mount point of PVC `tensorizer`, defaults to /t9k/mnt, and also $HOME
cd ~
git clone https://github.com/t9k/examples.git
```

### 下载模型

我们运行一个 `GenericJob` 下载并保存模型。在该 GenericJob 中，我们先使用国内模型平台 ModelScope 来下载 GPT-J-6B 模型，然后将模型转换为 CoreWeave Tensorizer 所需的格式，并保存到 PVC 中，代码细节见 [model_download.py](./download/model_download.py)。

运行 `GenericJob` 的命令如下：

```sh
# 切换到正确目录
cd ~/examples/inference/tensorizer
kubectl apply -f ./download-job.yaml
```

### 部署服务

查看下载模型任务的状态，等待 Phase 变为 `Succeeded`：

```
kubectl get -f ./download-job.yaml -w
```

待模型下载的任务结束，便可以开始部署服务。先创建 `MLServiceRuntime`：

```sh
kubectl apply -f ./runtime.yaml
```

然后部署 `MLService`：

```sh
kubectl apply -f ./mlservice.yaml
```

### 测试服务

查看 `MLService` 状态，并等待 Ready 一栏变为 `True`：

```sh
kubectl get -f ./mlservice.yaml -w
```

待其 READY 值变为 `true` 后，通过发送如下请求测试该推理服务：

```sh
address=$(kubectl get -f mlservice.yaml -ojsonpath='{.status.address.url}') && echo $address

curl ${address}/v1/models/gptj:predict -X POST \
  -H 'Content-Type: application/json' -d '{"instances": ["Once upon a time, there was"]}'
```

响应体应是一个类似于下面的 JSON：

```json
{"predictions":["Once upon a time, there was a happy prince…\n\nFaced with a constant barrage of attacks from enemies, this happy prince decided to launch a counterattack and create his own military kingdom. He created a country called Fairy Tail, where no dragon is found, but all dragon"]}
```

## 制作 Tensorizer 相关的镜像

本节将介绍如何通过自制镜像，将 Tensorizer 用于自定义模型。

### 准备 DockerConfig Secret

请参照[创建 Secret](https://github.com/t9k/tutorial-examples/blob/master/build-image/build-image-on-platform/README.md#创建-secret)，创建上传镜像所需要的 DockerConfig `Secret`。

### 模型下载镜像

切换到 `download` 文件夹下：

```sh
cd ~/examples/inference/tensorizer/download
```

打开其中的 `model_download.py` 文件，其内容分为两个部分：

1. 前半部分先使用 ModelScope 下载模型文件，然后使用 Transformers 将模型加载到内存当中。
2. 后半部分使用 Tensorizer 将模型及配置信息保存到 PVC 中。

用户可以用任意方式加载自定义模型来替换前半部分，然后沿用后半部分代码将用户自定义的模型以 Tensorizer 支持的形式保存到 PVC 中。

完成代码修改后，修改 `imagebuilder.yaml` 文件，将 `spec.dockerConfig.secret` 修改为上一步中创建的 DockerConfig `Secret` 的名称，并将 `spec.tag` 修改为目标镜像。以下是一个修改后的例子：

```yaml
apiVersion: tensorstack.dev/v1beta1
kind: ImageBuilder
metadata:
  name: tensorizer-download-image
spec:
  builder:
    kaniko: {}
  dockerConfig:
    secret: t9kpublic-docker-config
    subPath: .dockerconfigjson
  tag: t9kpublic/tensorizer-model-download:test
  workspace:
    pvc:
      contextPath: ./examples/inference/tensorizer/deploy/download
      dockerfilePath: ./examples/inference/tensorizer/deploy/download/Dockerfile
      name: tutorial
```

之后执行以下命令，创建 `ImageBuilder` ：

```sh
kubectl apply -f imagebuilder.yaml
```

查看 `ImageBuilder` 状态，等待 Phase 一栏变为 `Succeeded`：

```sh
kubectl get -f imagebuilder.yaml -w
```

成功完成后便可以将 `ImgaeBuilder` 中 `spec.tag` 指定的镜像用于替换[下载模型](#下载模型)中 `GenericJob` 的镜像。

### 模型部署镜像

切换到 `deploy` 文件夹下，

```sh
cd ~/examples/inference/tensorizer/deploy
```

打开其中的 `kserve_api.py` 文件，在该文件中会基于 [KServe](https://github.com/kserve/kserve/tree/master/python/kserve) 运行推理服务。

该文件中有一个 Model 对象，该对象会在启动时调用 `load` 方法。启动后收到推理请求时，会调用 `predict` 方法来处理请求。用户需要根据自己使用的模型，针对性的修改这两个方法。

在本示例中：

* `load` 方法调用了 `load_model.py` 中的 `load_model_based_on_type` 方法，使用 Tensorizer 从 PVC 中加载模型，用户可以复用该方法来加载自定义的模型。
* `predict` 方法基于模型的 tokenizer 处理输入、调用模型的 `generate` 方法产生推理结果。用户可以根据自己的模型，修改这里的代码，自定义处理推理请求的逻辑。

在完成上述修改后，修改 `imagebuilder.yaml` 文件，将 `spec.dockerConfig.secret` 修改为之前创建的 DockerConfig `Secret` 的名称，并将 `spec.tag` 修改为目标镜像。

之后执行以下命令，创建一个 `ImageBuilder` ：

```sh
kubectl apply -f imagebuilder.yaml
```

查看 `ImageBuilder` 状态，等待 Phase 一栏变为 `Succeeded`：

```sh
kubectl get -f imagebuilder.yaml -w
```

成功完成后便可以将 `ImgaeBuilder` 中 `spec.tag` 指定的镜像用于替换[部署服务](#部署服务)中 `MLServiceRuntime` 的镜像。

## 参考

* [Tensorizer 官方文档](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer)
* [基于 Tensorizer 和 KServe 部署 gpt-j](https://github.com/coreweave/kubernetes-cloud/tree/master/online-inference/tensorizer-isvc)
