import argparse

import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

parser = argparse.ArgumentParser(description="Play RL agents on CartPole-v1.")
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["dqn", "a2c", "ppo"],
    required=True,
    help="Choose the model corresponding to the RL algorithm to play.")
args = parser.parse_args()

env_id = "CartPole-v1"
video_folder = "logs/videos/"
video_length = 1000

vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env,
                           video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"agent-{env_id}")

if args.algorithm == "dqn":
    model = DQN.load("dqn_cart_pole")
elif args.algorithm == "a2c":
    model = A2C.load("a2c_cart_pole")
elif args.algorithm == "ppo":
    model = PPO.load("ppo_cart_pole")
else:
    raise ValueError("Unsupported model: {}".format(args.algorithm))

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        break
# Save the video
vec_env.close()
