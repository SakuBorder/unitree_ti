# legged_gym/envs/tiv2/ti_amp_config.py
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TiV2AMPCfg(LeggedRobotCfg):
    """TiV2 AMP Environment Configuration - 继承自 LeggedRobotCfg"""

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 1.0, 0.95]  # x,y,z [m]
        default_joint_angles = {
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
        num_actions = 12
        num_lower_dof = 12

        # ===== 新增：AMP历史配置 =====
        num_amp_obs_steps = 2  # AMP观测历史步数（默认2步）
        
        # AMP观测维度：(12+12+3+3) × 2步 = 60维
        num_amp_obs_per_step = 30  # 单步AMP观测维度
        num_amp_obs = num_amp_obs_steps * num_amp_obs_per_step  # 总AMP观测维度

        # 保持现有观测配置不变
        num_one_step_observations = 2 * 12 + 11 + 12  # 47
        num_one_step_privileged_obs = num_one_step_observations + 3  # 50
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations  # 282
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs  # 50

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
            # —— 最小任务 + 安全/正则 ——
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -1.0
            alive = 0.05

            # —— 先关闭，避免与风格冲突 ——
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            base_height = 0.0
            dof_acc = 0.0
            feet_air_time = 0.0
            torques = 0.0
            hip_pos = 0.0
            ankle_pos = 0.0
            feet_swing_height = 0.0
            contact = 0.0
            feet_parallel = 0.0
            feet_heading_alignment = 0.0
            lin_acc = 0.0
            contact_momentum = 0.0

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
        policy_class_name = "ActorCriticMoE"   # 或 "ActorCritic"
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
        num_experts = 4

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.998
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.01
        learning_rate = 1e-3
        max_grad_norm = 1.0
        use_clipped_value_loss = True
        schedule = "adaptive"
        desired_kl = 0.01
        amp_replay_buffer_size = 100000
        use_smooth_ratio_clipping = False

    class discriminator:
        hidden_dims = [1024, 512]
        reward_scale = 2.0
        loss_type = "BCEWithLogits"
        eta_wgan = 0.3
        reward_clamp_epsilon = 1e-4

    # AMP配置
    class amp:
        amp_data_path = "/home/dy/dy/code/unitree_ti/data/ti512/v1/singles"
        dataset_names = ["walk", "maikan"]
        dataset_weights = [0.2, 0.8]
        slow_down_factor = 1
        
        # ===== 更新：AMP观测配置 =====
        num_amp_obs_steps = 2  # 历史步数
        num_amp_obs = 60  # 30 × 2步
        
        dt = 1.0/60.0
        decimation = 4
        replay_buffer_size = 100000
        reward_scale = 2.0
        joint_names = None

        # 任务/风格融合权重
        style_weight = 0.5
        task_weight = 0.5