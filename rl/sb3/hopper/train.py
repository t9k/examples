import argparse
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize


def train_ppo():
    vec_env = make_vec_env("Hopper-v4")
    vec_env = VecNormalize(vec_env)
    model = PPO(policy="MlpPolicy",
                env=vec_env,
                batch_size=32,
                n_steps=512,
                gamma=0.999,
                learning_rate=9.80828e-05,
                ent_coef=0.00229519,
                clip_range=0.2,
                n_epochs=5,
                gae_lambda=0.99,
                max_grad_norm=0.7,
                vf_coef=0.835671,
                policy_kwargs=dict(log_std_init=-2,
                                   ortho_init=False,
                                   activation_fn=nn.ReLU,
                                   net_arch=dict(pi=[256, 256], vf=[256,
                                                                    256])),
                verbose=1,
                tensorboard_log="./ppo_hopper_tensorboard/")
    model.learn(total_timesteps=1e6)
    model.save("ppo_hopper")


def train_sac():
    vec_env = make_vec_env("Hopper-v4")
    model = SAC(policy="MlpPolicy",
                env=vec_env,
                learning_starts=10_000,
                verbose=1,
                tensorboard_log="./sac_hopper_tensorboard/")
    model.learn(total_timesteps=1e6, log_interval=10)
    model.save("sac_hopper")


def train_td3():
    vec_env = make_vec_env("Hopper-v4")
    vec_env = VecNormalize(vec_env)
    n_actions = vec_env.action_space.shape[0]
    model = TD3(policy="MlpPolicy",
                env=vec_env,
                learning_starts=10_000,
                action_noise=NormalActionNoise(mean=np.zeros(n_actions),
                                               sigma=0.1 * np.ones(n_actions)),
                train_freq=1,
                gradient_steps=1,
                learning_rate=3e-4,
                batch_size=256,
                verbose=1,
                tensorboard_log="./td3_hopper_tensorboard/")
    model.learn(total_timesteps=1e6, log_interval=10)
    model.save("td3_hopper")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train RL models on Hopper-v4.')
    parser.add_argument('-a',
                        '--algorithm',
                        choices=['ppo', 'sac', 'td3'],
                        required=True,
                        help='Choose the RL algorithm to train.')

    args = parser.parse_args()

    if args.algorithm == 'ppo':
        train_ppo()
    elif args.algorithm == 'sac':
        train_sac()
    elif args.algorithm == 'td3':
        train_td3()
    else:
        raise ValueError("Unsupported model: {}".format(args.algorithm))
