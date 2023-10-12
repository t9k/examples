import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnWrapper, TimeLimitWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed
from dizoo.mario.mario_dqn_config import main_config, create_config


def wrapped_mario_env():
    return DingEnvWrapper(
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


def main(cfg, seed=0):
    cfg['exp_name'] = 'mario_dqn_eval'
    cfg = compile_config(cfg,
                         BaseEnvManager,
                         DQNPolicy,
                         BaseLearner,
                         SampleSerialCollector,
                         InteractionSerialEvaluator,
                         AdvancedReplayBuffer,
                         save_cfg=True)
    cfg.policy.load_path = 'mario_dqn_seed0/ckpt/iteration_1000.pth.tar'

    # build multiple environments and use env_manager to manage them
    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_mario_env for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager)

    # switch save replay interface
    # evaluator_env.enable_save_replay(cfg.env.replay_path)
    evaluator_env.enable_save_replay(replay_path='./mario_dqn_eval/video')

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(
        torch.load(cfg.policy.load_path, map_location='cpu'))

    # Evaluate
    tb_logger = SummaryWriter(
        os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator,
                                           evaluator_env,
                                           policy.eval_mode,
                                           tb_logger,
                                           exp_name=cfg.exp_name)
    evaluator.eval()


if __name__ == '__main__':
    main(main_config)
