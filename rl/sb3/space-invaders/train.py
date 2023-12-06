import argparse
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecFrameStack


def train_dqn():
    # make env
    vec_env = make_vec_env("SpaceInvaders-v0", wrapper_class=AtariWrapper)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # train
    model = DQN(policy="CnnPolicy",
                env=vec_env,
                buffer_size=100_000,
                learning_rate=1e-4,
                batch_size=32,
                learning_starts=100_000,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                optimize_memory_usage=False,
                verbose=1,
                tensorboard_log="./dqn_space_invaders_tensorboard/")
    model.learn(total_timesteps=1e7, log_interval=100)

    # save model
    model.save("dqn_space_invaders")


def train_a2c():
    vec_env = make_vec_env("SpaceInvaders-v0",
                           n_envs=16,
                           wrapper_class=AtariWrapper)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = A2C(policy="CnnPolicy",
                env=vec_env,
                ent_coef=0.01,
                vf_coef=0.25,
                policy_kwargs=dict(optimizer_class=RMSpropTFLike,
                                   optimizer_kwargs=dict(eps=1e-5)),
                verbose=1,
                tensorboard_log="./a2c_space_invaders_tensorboard/")
    model.learn(total_timesteps=1e7)
    model.save("a2c_space_invaders")


def train_ppo():
    vec_env = make_vec_env("SpaceInvaders-v0",
                           n_envs=8,
                           wrapper_class=AtariWrapper)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO(policy="CnnPolicy",
                env=vec_env,
                n_steps=128,
                n_epochs=4,
                batch_size=256,
                learning_rate=2.5e-4,
                clip_range=0.1,
                vf_coef=0.5,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./ppo_space_invaders_tensorboard/")
    model.learn(total_timesteps=1e7)
    model.save("ppo_space_invaders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agents on SpaceInvaders-v0.")
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
