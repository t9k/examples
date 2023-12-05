# DI-engine 示例

[DI-engine](https://github.com/opendilab/DI-engine) 是一个开源的决策智能平台，其为强化学习算法研究和开发工作提供完整的算法支持、友好的用户接口以及弹性的拓展能力。

## 使用前提

TensorStack AI 计算平台所在的集群需要：

* 拥有 1 个这样的节点，其拥有 1 个 GPU，以及一定量的 CPU（核心数）和内存。

## 使用方法

1. 在您的项目中创建一个名为 ding、大小 20 GiB 的 PVC（需要存储一些模型检查点文件），然后创建一个同样名为 ding 的 Notebook 挂载该 PVC，镜像和资源不限（如要使用远程操作，请开启 SSH）。

1. 进入 Notebook 或远程连接到 Notebook，启动一个终端，执行以下命令以克隆此仓库。

    ```shell
    cd ~
    git clone https://github.com/t9k/examples.git
    ```

1. 继续使用 **Notebook 的终端**，参照下面的各个环境示例进行操作。

### Lunar Lander

Lunar Lander 环境模拟了一个经典的火箭轨迹优化问题。详情请参阅[此文档](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/lunarlander_zh.html)。

![](https://gymnasium.farama.org/_images/lunar_lander.gif)

分别使用 `train.yaml`、`evaluate.yaml` 和 `deploy.yaml` 创建 PyTorchTrainingJob 以启动训练、评估和部署（演示）：

```bash
cd ~/examples/rl/di-engine/lunarlander
kubectl create -f train.yaml     # 训练
kubectl create -f evaluate.yaml  # 评估
kubectl create -f deploy.yaml    # 部署（演示）
```

部署（演示）结果示例：

https://github.com/t9k/examples/assets/64956476/b60d70b7-ad77-4221-9a1c-043abbc0600f

### 超级马里奥兄弟

[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) 环境封装了家喻户晓的电子游戏《超级马里奥兄弟》，游戏中玩家操控一个马里奥进行移动与跳跃，躲避通往终点过程中的深坑与敌人，吃到更多的金币来获取更高的分数。游戏中还会有许多的有趣的道具，来为你提供不同的效果。详情请参阅[此文档](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/gym_super_mario_bros_zh.html)。

![](https://user-images.githubusercontent.com/2184469/40948820-3d15e5c2-6830-11e8-81d4-ecfaffee0a14.png)

分别使用 `train.yaml` 和 `deploy.yaml` 创建 PyTorchTrainingJob 以启动训练和部署（演示）：

```bash
cd ~/examples/rl/di-engine/super-mario-bros
kubectl create -f train.yaml
kubectl create -f deploy.yaml
```

部署（演示）结果示例：

https://github.com/t9k/examples/assets/64956476/b817aa37-f70f-46ca-b412-71003d9d80b7

### Slime Volleyball

[SlimeVolleyballGym](https://github.com/hardmaru/slimevolleygym) 环境封装了一个简单的一对一排球游戏，智能体的目标是将球落到对手场地的地面上。详情请参阅[此文档](https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/gym_super_mario_bros_zh.html)。

![](https://otoro.net/img/slimegym/pixel.gif)

分别使用 `train.yaml`、`evaluate.yaml` 和 `deploy.yaml` 创建 PyTorchTrainingJob 以启动训练、评估和部署（演示）：

```bash
cd ~/examples/rl/di-engine/slime-volleyball
kubectl create -f train.yaml
kubectl create -f evaluate.yaml
kubectl create -f deploy.yaml
```

此示例目前不支持 rendering，参阅[此注释](https://github.com/opendilab/DI-engine/blob/main/dizoo/slime_volley/envs/slime_volley_env.py#L98)。
