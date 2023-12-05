import torch

from easydict import EasyDict
from ding.config import compile_config
from ding.policy import PPOPolicy, single_env_forward_wrapper
from ding.model import VAC
from dizoo.slime_volley.envs.slime_volley_env import SlimeVolleyEnv
from dizoo.slime_volley.config.slime_volley_ppo_config import main_config, create_config


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'slime_volley_ppo_deploy'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = SlimeVolleyEnv(
        EasyDict({
            'env_id': 'SlimeVolley-v0',
            'agent_vs_agent': False
        }))
    # cannot render currently
    # env.enable_save_replay(replay_path='./slime_volley_ppo_deploy/video')

    model = VAC(**cfg.policy.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    policy = PPOPolicy(cfg.policy, model=model).eval_mode
    forward_fn = single_env_forward_wrapper(policy.forward)

    obs = env.reset()
    while True:
        action = forward_fn(obs)
        timestep = env.step(action)
        obs = timestep.obs
        if timestep.done:
            print('Episode is over, eval episode return is: {}'.format(
                timestep.info['eval_episode_return']))
            break


if __name__ == '__main__':
    main(main_config=main_config,
         create_config=create_config,
         ckpt_path=
         'slime_volley_ppo_seed0/ckpt_learner1/ckpt_best.pth.tar')
