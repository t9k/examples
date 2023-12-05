import argparse

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack

parser = argparse.ArgumentParser(
    description="Play RL agents on SpaceInvaders-v0.")
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["dqn", "a2c", "ppo"],
    required=True,
    help="Choose the model corresponding to the RL algorithm to play.")
args = parser.parse_args()

env_id = "SpaceInvaders-v0"
video_folder = "logs/videos/"
video_length = 1000

vec_env = make_vec_env("SpaceInvaders-v0",
                       wrapper_class=AtariWrapper,
                       env_kwargs=dict(render_mode="rgb_array"))
vec_env = VecFrameStack(vec_env, n_stack=4)
obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env,
                           video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"agent-{env_id}")

if args.algorithm == "dqn":
    model = DQN.load("dqn_space_invaders")
elif args.algorithm == "a2c":
    model = A2C.load("a2c_space_invaders")
elif args.algorithm == "ppo":
    model = PPO.load("ppo_space_invaders")
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
