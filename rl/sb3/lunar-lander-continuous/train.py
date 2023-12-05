import argparse
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize


def train_ppo():
    vec_env = make_vec_env("LunarLander-v2",
                           n_envs=16,
                           env_kwargs=dict(continuous=True))
    model = PPO(policy="MlpPolicy",
                env=vec_env,
                n_steps=1024,
                batch_size=64,
                gae_lambda=0.98,
                gamma=0.999,
                n_epochs=4,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./ppo_lunar_lander_continuous_tensorboard/")
    model.learn(total_timesteps=1e6)
    model.save("ppo_lunar_lander_continuous")


def train_ddpg():
    vec_env = make_vec_env("LunarLander-v2", env_kwargs=dict(continuous=True))
    n_actions = vec_env.action_space.shape[0]
    model = DDPG(policy="MlpPolicy",
                 env=vec_env,
                 gamma=0.98,
                 buffer_size=200_000,
                 learning_starts=10_000,
                 action_noise=NormalActionNoise(mean=np.zeros(n_actions),
                                                sigma=0.1 *
                                                np.ones(n_actions)),
                 gradient_steps=-1,
                 train_freq=(1, "episode"),
                 learning_rate=1e-3,
                 policy_kwargs=dict(net_arch=[400, 300]),
                 verbose=1,
                 tensorboard_log="./ddpg_lunar_lander_continuous_tensorboard/")
    model.learn(total_timesteps=3e5)
    model.save("ddpg_lunar_lander_continuous")


def train_sac():
    vec_env = make_vec_env("LunarLander-v2", env_kwargs=dict(continuous=True))
    vec_env = VecNormalize(vec_env)
    model = SAC(policy="MlpPolicy",
                env=vec_env,
                batch_size=256,
                learning_rate=7.3e-4,
                buffer_size=1_000_000,
                ent_coef='auto',
                gamma=0.99,
                tau=0.01,
                train_freq=1,
                gradient_steps=1,
                learning_starts=10_000,
                policy_kwargs=dict(net_arch=[400, 300]),
                verbose=1,
                tensorboard_log="./sac_lunar_lander_continuous_tensorboard/")
    model.learn(total_timesteps=3e5)
    model.save("sac_lunar_lander_continuous")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train RL agents on LunarLander-v2.')
    parser.add_argument('-a',
                        '--algorithm',
                        choices=['ppo', 'ddpg', 'sac'],
                        required=True,
                        help='Choose the RL algorithm to train.')

    args = parser.parse_args()

    if args.algorithm == 'ppo':
        train_ppo()
    elif args.algorithm == 'ddpg':
        train_ddpg()
    elif args.algorithm == 'sac':
        train_sac()
    else:
        raise ValueError("Unsupported model: {}".format(args.algorithm))
