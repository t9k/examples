import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnWrapper, TimeLimitWrapper
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer, termination_checker
from ding.framework.middleware.functional import online_logger
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
                lambda env: TimeLimitWrapper(env, max_limit=800),
                lambda env: EvalEpisodeReturnWrapper(env),
            ]
        })


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
        collector_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_mario_env for _ in range(collector_env_num)],
            cfg=cfg.env.manager)
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_mario_env for _ in range(evaluator_env_num)],
            cfg=cfg.env.manager)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(
            size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1e5))
        task.use(online_logger(train_show_freq=10))
        task.use(termination_checker(max_train_iter=3e6))
        task.run()


if __name__ == '__main__':
    main()
