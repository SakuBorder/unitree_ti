# legged_gym/scripts/train_amp.py
import sys
import os
sys.path.append(os.getcwd())

import isaacgym
from legged_gym.envs.tiv2.ti_amp_env import TiV2AMPRobot
from legged_gym.envs.tiv2.ti_amp_config import TiV2AMPCfg, TiV2AMPCfgPPO
from legged_gym.utils import task_registry, get_args
from amp_rsl_rl.runners import AMPOnPolicyRunner

def train_amp(args):
    # 注册AMP任务
    task_registry.register("tiv2_amp", TiV2AMPRobot, TiV2AMPCfg(), TiV2AMPCfgPPO())
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name="tiv2_amp", args=args)
    
    # 创建AMP runner
    from amp_rsl_rl.utils import AMPLoader, Normalizer
    from amp_rsl_rl.networks import Discriminator
    from amp_rsl_rl.algorithms import AMP_PPO
    from rsl_rl.modules import ActorCritic
    
    # 准备AMP组件
    device = args.rl_device
    
    # 加载专家数据
    amp_data = AMPLoader(
        device=device,
        dataset_path_root=Path(env_cfg.amp.amp_data_path),
        dataset_names=env_cfg.amp.dataset_names,
        dataset_weights=env_cfg.amp.dataset_weights,
        simulation_dt=env_cfg.sim.dt * env_cfg.control.decimation,
        slow_down_factor=env_cfg.amp.slow_down_factor,
        expected_joint_names=env.dof_names[:12]  # 只使用下半身关节
    )
    
    # 创建判别器
    discriminator = Discriminator(
        input_dim=env.num_amp_obs * 2,  # 当前+下一帧
        hidden_layer_sizes=env_cfg.discriminator.hidden_dims,
        reward_scale=env_cfg.discriminator.reward_scale,
        device=device,
        loss_type=env_cfg.discriminator.loss_type
    )
    
    # 创建归一化器
    amp_normalizer = Normalizer(env.num_amp_obs, device=device)
    
    # 创建Actor-Critic网络
    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs or env.num_obs,
        num_actions=env.num_actions,
        **env_cfg.policy
    ).to(device)
    
    # 创建AMP_PPO算法
    amp_ppo = AMP_PPO(
        actor_critic=actor_critic,
        discriminator=discriminator,
        amp_data=amp_data,
        amp_normalizer=amp_normalizer,
        device=device,
        **env_cfg.algorithm
    )
    
    # 创建Runner
    runner = AMPOnPolicyRunner(
        env=env,
        cfg=env_cfg,
        alg=amp_ppo,
        log_dir=f"logs/tiv2_amp",
        device=device
    )
    
    # 开始训练
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True)

if __name__ == "__main__":
    args = get_args()
    args.task = "tiv2_amp"
    train_amp(args)