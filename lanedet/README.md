# LaneDet

[LaneDet](https://github.com/Turoad/lanedet)  是一个基于 PyTorch 的开源车道检测工具箱。它的目标是整合各种最先进的车道检测模型，使开发者开发更先进的方法。

该项目较为久远，所以使用的工具包、技术都比较老旧。

本示例基于该项目，进行代码翻新、将其转化为分布式训练，并提供 TensorStack 训练启动方式。

## 数据集

TuSimple：https://www.kaggle.com/datasets/manideep1108/tusimple/data

该数据集需要进行身份验证（Google 账号即可），只能通过浏览器下载。

## 训练

使用 `job.yaml` 创建 PyTorchTrainingJob 以执行训练：

```shell
cd ~/examples/lanedet/t9k
kubectl create -f job.yaml
```

对于 YAML 配置文件进行如下说明：

* 训练副本（replica）数量为 2（第 13 行）。
* 每个副本的进程数量（第 9 行）和 GPU 数量（第 35 和 39 行）同为 2。
* 镜像 `t9kpublic/lanedet:230927`（第 22 行）由 [Dockerfile](./docker/Dockerfile) 定义。
* 设置共享内存（第 44 和 50 行）。
* 数据集解压并保存在 PVC lanedet 中，训练需绑定该 PVC（第 41 和 47 行）。
