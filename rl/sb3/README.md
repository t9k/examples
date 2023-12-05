# Stable Baselines3 示例

[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)（SB3）是基于 PyTorch 实现的一组可靠的强化学习算法。

## 使用前提

这些示例在多核 CPU 上进行训练和预测（演示），您可以在本地或在 TensorStack AI 计算平台上运行。

## 使用方法

1. 若在 TensorStack AI 计算平台上运行，在您的项目中创建一个名为 sb3、大小 1 GiB 的 PVC，然后创建一个同样名为 sb3 的 Notebook 挂载该 PVC，选择 PyTorch 类型的镜像，资源模板选择 large。

1. 在 Notebook 或本地的终端中，安装以下依赖：

    ```bash
    apt update && apt install python-opengl xvfb
    pip install stable_baselines3 gymnasium gymnasium[atari] gymnasium[accept-rom-license] gymnasium[mujoco] gymnasium[mujoco_py] moviepy pygame stable-retro opencv-python
    ```

1. 继续使用终端，参照下面的各个环境示例进行操作。

### Cart Pole

Cart Pole 环境是一个经典的控制环境。在该环境中，一个杆通过一个非驱动关节连接到一个沿着无摩擦轨道移动的小车上，目标是通过在小车上施加左右方向的力来保持竖直放置的杆的平衡。详情请参阅[此文档](https://gymnasium.farama.org/environments/classic_control/cart_pole/)。

![](https://gymnasium.farama.org/_images/cart_pole.gif)

先后运行 `train.py` 和 `play.py` 脚本以进行训练和预测（演示）：

```bash
cd ~/examples/rl/sb3/cart-pole
python train.py -a dqn        # 训练
python play.py -a dqn         # 预测（演示）
```

也可以使用 `train-play.yaml` 创建 PyTorchTrainingJob 以启动训练和预测（演示）：

```bash
kubectl create -f train-play.yaml  # 训练和预测（演示）
```

训练过程会产生 TensorBoard 日志，预测（演示）会产生 MP4 文件，都保存在工作目录下。

本示例支持强化学习算法 DQN、A2C 和 PPO。使用 PPO 训练的模型的一次预测（演示）如下：

### Space Invaders

Space Invaders 环境封装了最早的射击游戏之一《Space Invaders》，游戏中玩家控制一台太空飞船，通过射击外星入侵者来防止它们接近地球。详情请参阅[此文档](https://gymnasium.farama.org/environments/atari/space_invaders/)。

![](https://gymnasium.farama.org/_images/space_invaders.gif)

先后运行 `train.py` 和 `play.py` 脚本以进行训练和预测（演示）：

```bash
cd ~/examples/rl/sb3/space-invaders
python train.py -a dqn        # 训练
python play.py -a dqn         # 预测（演示）
```

也可以使用 `train-play.yaml` 创建 PyTorchTrainingJob 以启动训练和预测（演示）：

```bash
kubectl create -f train-play.yaml  # 训练和预测（演示）
```

训练过程会产生 TensorBoard 日志，预测（演示）会产生 MP4 文件，都保存在工作目录下。

本示例支持强化学习算法 DQN、A2C 和 PPO。使用 PPO 训练的模型的一次预测（演示）如下：

### Lunar Lander Continuous

Lunar Lander 环境模拟了一个经典的火箭轨迹优化问题，这里采用它的连续版本。详情请参阅[此文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/)。

![](https://gymnasium.farama.org/_images/lunar_lander.gif)

先后运行 `train.py` 和 `play.py` 脚本以进行训练和预测（演示）：

```bash
cd ~/examples/rl/sb3/lunar-lander-continuous
python train.py -a ppo        # 训练
python play.py -a ppo         # 预测（演示）
```

也可以使用 `train-play.yaml` 创建 PyTorchTrainingJob 以启动训练和预测（演示）：

```bash
kubectl create -f train-play.yaml  # 训练和预测（演示）
```

本示例支持强化学习算法 PPO、DDPG 和 SAC。使用 SAC 训练的模型的一次预测（演示）如下：

### Hopper

Hopper 环境中的“hopper”是一个二维的单腿形状，它由四个主要部分组成：顶部的躯干，中间的大腿，底部的小腿，以及支撑整个身体的单脚，目标是通过在连接这四个部分的三个铰链上施加扭矩，实现向前（右）方向的跃动。详情请参阅[此文档](https://gymnasium.farama.org/environments/mujoco/hopper/)。

![](https://gymnasium.farama.org/environments/mujoco/hopper/)

先后运行 `train.py` 和 `play.py` 脚本以进行训练和预测（演示）：

```bash
cd ~/examples/rl/sb3/hopper
python train.py -a ppo        # 训练
python play.py -a ppo         # 预测（演示）
```

也可以使用 `train-play.yaml` 创建 PyTorchTrainingJob 以启动训练和预测（演示）：

```bash
kubectl create -f train-play.yaml  # 训练和预测（演示）
```

本示例支持强化学习算法 PPO、SAC 和 TD3。使用 TD3 训练的模型的一次预测（演示）如下：

### 怀旧游戏
