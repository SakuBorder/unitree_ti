# legged_gym/envs/tiv2/ti_amp_config.py
from legged_gym.envs.tiv2.tiv2_config import TiV2RoughCfg, TiV2RoughCfgPPO

class TiV2AMPCfg(TiV2RoughCfg):
    class env(TiV2RoughCfg.env):
        num_amp_obs = 66

    class observations:
        class amp:
            joint_pos = True
            joint_vel = True
            base_lin_vel_local = True
            base_ang_vel_local = True

    class amp:
        amp_data_path = "/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/pkl"
        dataset_names = ["0-Male2Walking_c3d_B15 -  Walk turn around_poses"]
        dataset_weights = [0.7, 0.3]
        slow_down_factor = 1
        reward_scale = 2.0
        replay_buffer_size = 100000


class TiV2AMPCfgPPO(TiV2RoughCfgPPO):
    class runner:
        runner_class_name   = "AMPOnPolicyRunner"
        algorithm_class_name= "AMP_PPO"
        policy_class_name   = "ActorCriticMoE"
        experiment_name     = "tiv2_amp"
        run_name            = "tiv2_amp_run"
        resume              = False
        load_run            = ".*"
        checkpoint          = -1
        num_steps_per_env   = 24
        save_interval       = 50
        empirical_normalization = False

    class amp:
        amp_data_path   = "/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/pkl"
        dataset_names   = ["0-Male2Walking_c3d_B15 -  Walk turn around_poses"]
        dataset_weights = [0.7, 0.3]
        slow_down_factor= 1
        num_amp_obs     = 30
        replay_buffer_size = 100000
        reward_scale    = 2.0

    class discriminator:
        hidden_dims   = [1024, 512]
        reward_scale  = 2.0
        loss_type     = "BCEWithLogits"
        learning_rate = 3e-4

    class policy(TiV2RoughCfgPPO.policy):
        init_noise_std = 1.0

    # 关键：重写 algorithm，去掉父类的 symmetry_scale / use_flip 等和 AMP_PPO 无关的键
    class algorithm:
        # 下列键均出自你贴的 AMP_PPO.__init__ 签名，可按需调整
        num_learning_epochs = 1
        num_mini_batches    = 1
        clip_param          = 0.2
        gamma               = 0.998
        lam                 = 0.95
        value_loss_coef     = 1.0
        entropy_coef        = 0.0
        learning_rate       = 1e-3
        max_grad_norm       = 1.0
        use_clipped_value_loss = True
        schedule            = "fixed"   # 或 "adaptive"
        desired_kl          = 0.01
        amp_replay_buffer_size = 100000
        use_smooth_ratio_clipping = False
        # 注意：不要放 symmetry_scale / symmetry_cfg / use_flip 等不在 AMP_PPO.__init__ 里的字段
