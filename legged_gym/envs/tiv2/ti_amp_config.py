# legged_gym/envs/tiv2/ti_amp_config.py
from legged_gym.envs.tiv2.tiv2_config import TiV2RoughCfg, TiV2RoughCfgPPO

class TiV2AMPCfg(TiV2RoughCfg):
    class env(TiV2RoughCfg.env):
        # 添加AMP相关配置
        num_amp_obs = 30  # 12个关节位置 + 12个关节速度 + 3个基座线速度 + 3个基座角速度
        
    class observations:
        class amp:
            # AMP观测相关的配置
            joint_pos = True
            joint_vel = True
            base_lin_vel_local = True
            base_ang_vel_local = True
    
    class amp:
        # AMP训练相关配置
        amp_data_path = "amp_datasets/tiv2"
        dataset_names = ["walk", "stand"]
        dataset_weights = [0.7, 0.3]
        slow_down_factor = 1
        reward_scale = 2.0
        replay_buffer_size = 100000
        
    class discriminator:
        hidden_dims = [1024, 512]
        reward_scale = 2.0
        loss_type = "BCEWithLogits"  # 或 "Wasserstein"

class TiV2AMPCfgPPO(TiV2RoughCfgPPO):
    class runner:
        # 使用AMP Runner
        runner_class_name = "AMPOnPolicyRunner"
        algorithm_class_name = "AMP_PPO"
        policy_class_name = "ActorCritic"  # 或使用ActorCriticMoE