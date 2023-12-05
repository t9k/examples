from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
import retro

env_id = "Airstriker-Genesis"
video_folder = "logs/videos/"
video_length = 100_000

def make_env():
    env = retro.make("Airstriker-Genesis", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=4500)
    env = ClipRewardEnv(WarpFrame(env))
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecTransposeImage(VecFrameStack(vec_env, n_stack=4))

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env,
                           video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"agent-{env_id}")

model = PPO.load("ppo_retro")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        break
# Save the video
vec_env.close()
