import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import torch

from easydict import EasyDict
from ding.config import compile_config
from ding.envs import DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnWrapper, TimeLimitWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.torch_utils import to_tensor, to_ndarray, unsqueeze, squeeze
from dizoo.mario.mario_dqn_config import main_config, create_config


def single_env_forward_wrapper(forward_fn):
    def _forward(obs):
        obs = {0: to_tensor(obs)}
        action = forward_fn(obs)[0]['action']
        action = to_ndarray(action)
        return action

    return _forward


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'mario_dqn_deploy'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = DingEnvWrapper(
        JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-1-1-v0'),
                    [['right'], ['right', 'A']]),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=84),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=4),
                lambda env: TimeLimitWrapper(env, max_limit=400),
                lambda env: EvalEpisodeReturnWrapper(env),
            ]
        })
    env.enable_save_replay(replay_path='./mario_dqn_deploy/video')
    model = DQN(**cfg.policy.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    forward_fn = single_env_forward_wrapper(policy.forward)

    obs = env.reset()
    returns = 0.
    while True:
        action = forward_fn(obs)
        obs, rew, done, info = env.step(action)
        returns += rew
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')


if __name__ == '__main__':
    main(main_config=main_config,
         create_config=create_config,
         ckpt_path='mario_dqn_seed0/ckpt/iteration_1000.pth.tar')
