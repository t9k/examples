import argparse
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env


def train_dqn():
    # make env
    vec_env = make_vec_env("CartPole-v1")

    # train
    model = DQN(policy="MlpPolicy",
                env=vec_env,
                learning_rate=2.3e-3,
                batch_size=64,
                buffer_size=100_000,
                learning_starts=1_000,
                gamma=0.99,
                target_update_interval=10,
                train_freq=256,
                gradient_steps=128,
                exploration_fraction=0.16,
                exploration_final_eps=0.04,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=1,
                tensorboard_log="./dqn_cart_pole_tensorboard/")
    model.learn(total_timesteps=2e5, log_interval=10)

    # save model
    model.save("dqn_cart_pole")


def train_a2c():
    vec_env = make_vec_env("CartPole-v1", n_envs=8)
    model = A2C(policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log="./a2c_cart_pole_tensorboard/")
    model.learn(total_timesteps=3e5)
    model.save("a2c_cart_pole")


def train_ppo():
    vec_env = make_vec_env("CartPole-v1", n_envs=8)
    model = PPO(policy="MlpPolicy",
                env=vec_env,
                n_steps=32,
                batch_size=256,
                gae_lambda=0.8,
                gamma=0.98,
                n_epochs=20,
                ent_coef=0.0,
                learning_rate=1e-3,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./ppo_cart_pole_tensorboard/")
    model.learn(total_timesteps=1e5)
    model.save("ppo_cart_pole")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agents on CartPole-v1.")
    parser.add_argument("-a",
                        "--algorithm",
                        choices=["dqn", "a2c", "ppo"],
                        required=True,
                        help="Choose the RL algorithm to train.")

    args = parser.parse_args()

    if args.algorithm == "dqn":
        train_dqn()
    elif args.algorithm == "a2c":
        train_a2c()
    elif args.algorithm == "ppo":
        train_ppo()
    else:
        raise ValueError("Unsupported model: {}".format(args.algorithm))
