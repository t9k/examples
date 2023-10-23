import os
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, NaiveReplayBuffer, InteractionSerialEvaluator
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config.slime_volley_ppo_config import main_config


# Get DI-engine form env class
def wrapped_lunarlander_env():
    return SlimeVolleyEnv(
        EasyDict({
            'env_id': main_config['env']['env_id'],
            'agent_vs_agent': False
        }))


def main(cfg, seed=0):
    cfg['exp_name'] = 'slime_volley_ppo_eval'
    cfg = compile_config(cfg,
                         BaseEnvManager,
                         PPOPolicy,
                         BaseLearner,
                         BattleSampleSerialCollector,
                         InteractionSerialEvaluator,
                         NaiveReplayBuffer,
                         save_cfg=True)
    cfg.policy.load_path = './slime_volley_ppo_seed0/ckpt_learner1/ckpt_best.pth.tar'

    # build multiple environments and use env_manager to manage them
    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_lunarlander_env for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager)

    # switch save replay interface
    # == cannot render currently ==
    # evaluator_env.enable_save_replay(
    #     replay_path='./slime_volley_ppo_eval/video')

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
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
