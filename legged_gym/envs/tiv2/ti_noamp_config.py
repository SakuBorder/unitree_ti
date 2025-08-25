# legged_gym/envs/tiv2/ti_amp_config.py
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TiV2NoAMPCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 1.0, 0.95]
        default_joint_angles = {
            'L_ANKLE_R': 0.0,
            'R_ANKLE_R': 0.0,
            'L_ANKLE_P': 0.2,
            'L_HIP_P': -0.1,
            'L_HIP_R': 0.0,
            'L_HIP_Y': 0.0,
            'L_KNEE_P': 0.3,
            'R_ANKLE_P': -0.2,
            'R_HIP_P': 0.1,
            'R_HIP_R': 0.0,
            'R_HIP_Y': 0.0,
            'R_KNEE_P': -0.3,
        }

    class env(LeggedRobotCfg.env):
        num_actions = 12
        num_lower_dof = 12
        num_amp_obs_per_step = 43
        num_amp_obs_steps = 6
        num_amp_obs = num_amp_obs_steps * num_amp_obs_per_step
        num_one_step_observations = 2 * 12 + 11 + 12
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10.0, 0.0, 3.0]
        lookat = [0.0, 1.0, 0.0]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'HIP_P': 150,
            'HIP_R': 150,
            'HIP_Y': 150,
            'KNEE': 200,
            'ANKLE_R': 40,
            'ANKLE_P': 40,
        }
        damping = {
            'HIP_P': 2,
            'HIP_R': 2,
            'HIP_Y': 2,
            'KNEE': 4,
            'ANKLE_R': 2,
            'ANKLE_P': 2,
        }
        action_scale = 0.25
        decimation = 4

    class commands(LeggedRobotCfg.commands):
        resampling_time = 1e9
        lin_vel_x = [0.8, 0.8]
        lin_vel_y = [0.0, 0.0]
        ang_vel_yaw = [0.0, 0.0]
        phaseA_vx = 0.8
        phaseA_vy = 0.0
        phaseA_yaw = 0.0

    class asset(LeggedRobotCfg.asset):
        file = '/home/dy/dy/code/unitree_ti/assert/ti5/tai5_12dof_no_limit.urdf'
        name = "TiV2"
        foot_name = "ANKLE_R"
        penalize_contacts_on = ["HIP", "KNEE"]
        terminate_after_contacts_on = ["base_link"]
        left_foot_name = "L_ANKLE_P_S"
        right_foot_name = "R_ANKLE_P_S"
        self_collisions = 0
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.1, 2.0]
        randomize_kp = True
        kp_factor_range = [0.9, 1.1]
        randomize_kd = True
        kd_factor_range = [0.9, 1.1]
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 5.0
        max_push_vel_xy = 0.3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.92
        least_feet_distance_lateral = 0.2
        most_feet_distance_lateral = 0.35

        class scales(LeggedRobotCfg.rewards.scales):
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

    class observations:
        class amp:
            joint_pos = True
            joint_vel = True
            base_lin_vel_local = True
            base_ang_vel_local = True
            root_height = False
            root_rot_tannorm = True
            key_body_pos_local = True

class TiV2NoAMPCfgPPO(LeggedRobotCfgPPO):
    class runner:
        runner_class_name = "AMPOnPolicyRunner"
        algorithm_class_name = "AMP_PPO"
        policy_class_name = "ActorCritic"
        experiment_name = "tiv2_noamp"
        run_name = "tiv2_noamp"
        resume = False
        load_run = ".*"
        checkpoint = -1
        num_steps_per_env = 32
        max_iterations = 100000
        save_interval = 500
        empirical_normalization = False

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        num_experts = 1

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.998
        lam = 0.95
        value_loss_coef = 1.0
        entropy_coef = 0.02
        learning_rate = 8e-4
        max_grad_norm = 1.0
        use_clipped_value_loss = True
        schedule = "adaptive"
        desired_kl = 0.01
        amp_replay_buffer_size = 100000
        use_smooth_ratio_clipping = False

    class discriminator:
        hidden_dims = [256, 128]
        reward_scale = 1.0
        loss_type = "BCEWithLogits"
        eta_wgan = 0.3
        reward_clamp_epsilon = 1e-4

    class amp:
        amp_data_path = "/home/dy/dy/code/unitree_ti/data/ti512/v1/singles"
        dataset_names = ["walk"]
        dataset_weights = [1.0]
        slow_down_factor = 1
        num_amp_obs_steps = TiV2NoAMPCfg.env.num_amp_obs_steps
        num_amp_obs_per_step = TiV2NoAMPCfg.env.num_amp_obs_per_step
        num_amp_obs = num_amp_obs_steps * num_amp_obs_per_step
        history_steps = num_amp_obs_steps
        history_stride = 1
        dt = 1.0 / 60.0
        decimation = 4
        replay_buffer_size = 100000
        reward_scale = 2.0
        joint_names = None
        style_weight = 0.0
        task_weight = 1.0
