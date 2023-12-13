# Stable Baselines3 示例

[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)（SB3）是基于 PyTorch 实现的一组可靠的强化学习算法。

## 使用前提

这些示例在多核 CPU 上进行训练和预测（演示），您可以在本地或在 TensorStack AI 计算平台上运行。

## 使用方法

1. 若在 TensorStack AI 计算平台上运行，在您的项目中创建一个名为 sb3、大小 1 GiB 的 PVC，然后创建一个同样名为 sb3 的 Notebook 挂载该 PVC，选择 PyTorch 类型的镜像，资源模板选择 large。

1. 在 Notebook 或本地的终端中，安装以下依赖：

    ```bash
    apt update && apt install python-opengl xvfb swig
    pip install -r requirements.txt
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

本示例支持 DQN、A2C 和 PPO 算法。使用 PPO 训练的模型的一次预测（演示）如下：

https://github.com/t9k/examples/assets/64956476/06af7b0f-6793-4eb2-aa1d-0be992bdeeac

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

本示例支持 DQN、A2C 和 PPO 算法。使用 PPO 训练的模型的一次预测（演示）如下：

https://github.com/t9k/examples/assets/64956476/05dd21e2-858a-4126-bb2e-c72f1d09ecef

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

本示例支持 PPO、DDPG 和 SAC 算法。使用 SAC 训练的模型的一次预测（演示）如下：

https://github.com/t9k/examples/assets/64956476/e1516f5d-511a-4465-8012-304f2af6464b

### Hopper

Hopper 环境中的“hopper”是一个二维的单腿形状，它由四个主要部分组成：顶部的躯干，中间的大腿，底部的小腿，以及支撑整个身体的单脚，目标是通过在连接这四个部分的三个铰链上施加扭矩，实现向前（右）方向的跃动。详情请参阅[此文档](https://gymnasium.farama.org/environments/mujoco/hopper/)。

![](https://gymnasium.farama.org/_images/hopper.gif)

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

本示例支持 PPO、SAC 和 TD3 算法。使用 SAC 训练的模型的一次预测（演示）如下：

https://github.com/t9k/examples/assets/64956476/194dd8ae-c893-4ee9-8093-31675651760d

### Stable-Retro

Stable-Retro 环境是 Gym Retro（“将经典的视频游戏转化为 Gymnasium 环境用于强化学习”）的一个 fork，增加了额外的游戏、模拟器以及支持的平台。详情请参阅[此文档](https://stable-retro.farama.org/)。

![](https://stable-retro.farama.org/_images/retro_games.png)

先后运行 `train.py` 和 `play.py` 脚本以进行训练和预测（演示）：

```bash
cd ~/examples/rl/sb3/retro
xvfb-run -a -s "-screen 0 1400x900x24" python train.py      # 训练
python play.py                                              # 预测（演示）
```

也可以使用 `train-play.yaml` 创建 PyTorchTrainingJob 以启动训练和预测（演示）：

```bash
kubectl create -f train-play.yaml  # 训练和预测（演示）
```

默认采用的环境是 Airstriker-Genesis（一款太空题材的射击游戏）。如果你恰好有正确的 ROM 文件，可以导入它们（详情请参阅[此文档](https://stable-retro.farama.org/getting_started/#importing-roms)），并创建相应的环境。以 SnowBrothers-Nes（游戏《雪人兄弟》）为例：

```bash
# 假定正确的 Snow Brothers Nes ROM 文件位于当前目录下
python -m retro.import .                                                   # 导入 ROM
xvfb-run -a -s "-screen 0 1400x900x24" python train.py --game SnowBrothers-Nes  # 训练
python play.py --game SnowBrothers-Nes                                     # 预测（演示）
```

本示例仅支持 PPO 算法。模型的一次预测（演示）如下：

https://github.com/t9k/examples/assets/64956476/55a253c4-672b-40df-97c5-adf4843674e4
