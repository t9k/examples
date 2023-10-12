import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config


# Get DI-engine form env class
def wrapped_lunarlander_env():
    return DingEnvWrapper(
        gym.make(main_config['env']['env_id']),
        EasyDict(env_wrapper='default'),
    )


def main(cfg, seed=0):
    cfg['exp_name'] = 'lunarlander_dqn_eval'
    cfg = compile_config(cfg,
                         BaseEnvManager,
                         DQNPolicy,
                         BaseLearner,
                         SampleSerialCollector,
                         InteractionSerialEvaluator,
                         AdvancedReplayBuffer,
                         save_cfg=True)
    cfg.policy.load_path = './final.pth.tar'

    # build multiple environments and use env_manager to manage them
    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_lunarlander_env for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager)

    # switch save replay interface
    # evaluator_env.enable_save_replay(cfg.env.replay_path)
    evaluator_env.enable_save_replay(
        replay_path='./lunarlander_dqn_eval/video')

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
