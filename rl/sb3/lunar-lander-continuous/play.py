import argparse

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["ppo", "ddpg", "sac"],
    required=True,
    help="Choose the model corresponding to the RL algorithm to play.")
args = parser.parse_args()

env_id = "LunarLanderContinuous-v2"
video_folder = "logs/videos/"
video_length = 1000

vec_env = make_vec_env("LunarLander-v2",
                       env_kwargs=dict(continuous=True,
                                       render_mode="rgb_array"))
obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env,
                           video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"agent-{env_id}")

if args.algorithm == "ppo":
    model = PPO.load("ppo_lunar_lander_continuous")
elif args.algorithm == "ddpg":
    model = DDPG.load("ddpg_lunar_lander_continuous")
elif args.algorithm == "sac":
    model = SAC.load("sac_lunar_lander_continuous")
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
