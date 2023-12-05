import argparse

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize

parser = argparse.ArgumentParser(description="Play RL agents on Hopper-v4.")
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["dqn", "a2c", "ppo"],
    required=True,
    help="Choose the model corresponding to the RL algorithm to play.")
args = parser.parse_args()

env_id = "Hopper-v4"
video_folder = "logs/videos/"
video_length = 2000

vec_env = make_vec_env("Hopper-v4", env_kwargs=dict(render_mode="rgb_array"))
if args.algorithm in ["ppo", "td3"]:
    vec_env = VecNormalize(vec_env)
obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env,
                           video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"agent-{env_id}")

if args.algorithm == "ppo":
    model = PPO.load("ppo_hopper")
elif args.algorithm == "sac":
    model = SAC.load("sac_hopper")
elif args.algorithm == "td3":
    model = TD3.load("td3_hopper")
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
