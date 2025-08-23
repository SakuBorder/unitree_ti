# legged_gym/envs/tiv2/ti_amp_config.py
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TiV2AMPCfg(LeggedRobotCfg):
    """TiV2 AMP Environment Configuration - 继承自 LeggedRobotCfg"""
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 1.0, 0.95]  # x,y,z [m]
        default_joint_angles = {  # 与原TiV2保持一致
            'L_ANKLE_R': 0.,
            'R_ANKLE_R': 0,
            'L_ANKLE_P': 0.2,
            'L_HIP_P': -0.1,
            'L_HIP_R': 0,
            'L_HIP_Y': 0.,
            'L_KNEE_P': 0.3,
            'R_ANKLE_P': -0.2,
            'R_HIP_P': 0.1,
            'R_HIP_R': 0,
            'R_HIP_Y': 0.,
            'R_KNEE_P': -0.3,
        }

    class env(LeggedRobotCfg.env):
        num_actions = 12  # TiV2的DOF数量
        num_lower_dof = 12
        # AMP特定的观测维度
        num_amp_obs = 30  # joint_pos(12) + joint_vel(12) + base_lin_vel(3) + base_ang_vel(3)
        
        # 保持与原TiV2环境一致的观测结构
        num_one_step_observations = 2 * 12 + 11 + 12  # 47
        num_one_step_privileged_obs = num_one_step_observations + 3  # 50
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 3]
        lookat = [0., 1, 0.]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'HIP_P': 150,
                     'HIP_R': 150,
                     'HIP_Y': 150,
                     'KNEE': 200,
                     'ANKLE_R': 40,
                     'ANKLE_P': 40}
        damping = {'HIP_P': 2,
                   'HIP_R': 2,
                   'HIP_Y': 2,
                   'KNEE': 4,
                   'ANKLE_R': 2,
                   'ANKLE_P': 2}
        action_scale = 0.12
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '/home/dy/dy/code/unitree_ti/assert/ti5/tai5_12dof_no_limit.urdf'
        name = "TiV2"
        foot_name = "ANKLE_R"
        penalize_contacts_on = ["HIP", "KNEE"]
        terminate_after_contacts_on = ["base_link"]
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        self_collisions = 0
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.1, 2.]
        randomize_kp = True
        kp_factor_range = [0.8, 1.2]
        randomize_kd = True
        kd_factor_range = [0.8, 1.2]
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.92
        
        class scales(LeggedRobotCfg.rewards.scales):
            # AMP特有奖励权重
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.5
            orientation = -1.0
            base_height = -100.0
            dof_acc = -2.5e-7
            feet_air_time = 0.0
            collision = -1.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -10.0
            ankle_pos = -50.0
            feet_swing_height = -100.0
            contact = 1.0
            feet_parallel = -1.0
            feet_heading_alignment = -1.0
            lin_acc = -2.5e-5
            contact_momentum = 2.5e-2

    # AMP特定配置
    class amp:
        amp_data_path = "/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/pkl"
        dataset_names = ["0-Male2Walking_c3d_B15 -  Walk turn around_poses"]
        dataset_weights = [1.0]  # 修正：只有一个数据集，权重应该是[1.0]
        slow_down_factor = 1
        num_amp_obs = 30  # 与env.num_amp_obs保持一致
        dt = 1.0/60.0  # simulation timestep
        decimation = 4  # 与control.decimation保持一致
        replay_buffer_size = 100000
        reward_scale = 2.0
        joint_names = None  # 让AMPLoader自动推断

    class observations:
        class amp:
            joint_pos = True
            joint_vel = True
            base_lin_vel_local = True
            base_ang_vel_local = True


class TiV2AMPCfgPPO(LeggedRobotCfgPPO):
    """TiV2 AMP PPO Training Configuration"""
    
    class runner:
        runner_class_name = "AMPOnPolicyRunner"
        algorithm_class_name = "AMP_PPO"
        policy_class_name = "ActorCriticMoE"  # 或者 "ActorCritic"
        experiment_name = "tiv2_amp"
        run_name = "tiv2_amp_run"
        resume = False
        load_run = ".*"
        checkpoint = -1
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 50
        empirical_normalization = False

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        # MoE特定参数（如果使用ActorCriticMoE）
        num_experts = 4
        gate_hidden_dims = None  # 使用默认值

    # AMP_PPO算法参数 - 严格按照AMP_PPO.__init__的签名
    class algorithm:
        num_learning_epochs = 5  # 增加学习epochs
        num_mini_batches = 4     # 增加mini-batch数量
        clip_param = 0.2
        gamma = 0.998
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.01      # 增加探索
        learning_rate = 1e-3
        max_grad_norm = 1.0
        use_clipped_value_loss = True
        schedule = "adaptive"    # 使用自适应学习率
        desired_kl = 0.01
        amp_replay_buffer_size = 100000
        use_smooth_ratio_clipping = False

    # 判别器配置
    class discriminator:
        hidden_dims = [1024, 512]
        reward_scale = 2.0
        loss_type = "BCEWithLogits"
        eta_wgan = 0.3  # 如果使用Wasserstein loss
        reward_clamp_epsilon = 1e-4

    # AMP特定配置（复制到training config中）
    class amp:
        amp_data_path = "/home/dy/dy/code/unitree_ti/data/ti512/v1/singles"
        dataset_names = ["0-Male2Walking_c3d_B9 -  Walk turn left 90_poses"]
        dataset_weights = [1.0]  # 修正权重
        slow_down_factor = 1
        num_amp_obs = 30
        dt = 1.0/60.0
        decimation = 4
        replay_buffer_size = 100000
        reward_scale = 2.0
        joint_names = None