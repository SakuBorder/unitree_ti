from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TiV2RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 1.0, 0.95]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'L_ANKLE_R': 0.,
            'R_ANKLE_R': 0,
            'L_ANKLE_P': 0.2, #0.2
            # 'L_ELBOW_Y': 0,
            'L_HIP_P': -0.1,#-0.1
            'L_HIP_R': 0,
            'L_HIP_Y': 0.,
            'L_KNEE_P': 0.3,#0.3
            # 'L_SHOULDER_P': 0.,
            # 'L_SHOULDER_R': 0.,
            # 'L_SHOULDER_Y': 0.,
            # 'L_WRIST_P': 0,
            # "L_WRIST_R": 0.,
            # "L_WRIST_Y": 0.,
            # "NECK_Y": 0.,
            'R_ANKLE_P': -0.2,#-0.2
            # 'R_ELBOW_Y': 0,
            'R_HIP_P': 0.1,#0.1
            'R_HIP_R': 0,
            'R_HIP_Y': 0.,
            'R_KNEE_P': -0.3,#-0.3
            # 'R_SHOULDER_P': 0.,
            # 'R_SHOULDER_R': 0.,
            # 'R_SHOULDER_Y': 0.,
            # 'R_WRIST_P': 0,
            # "R_WRIST_R": 0.,
            # "R_WRIST_Y": 0.,
            # 'WAIST_P': 0,
            # "WAIST_R": 0.,
            # "WAIST_Y": 0.
        }

    class env(LeggedRobotCfg.env):
        # num_observations = 47 # 74 47
        # num_privileged_obs = 50 # 77 50
        num_actions = 12 # 21 12
        num_lower_dof = 12
        num_one_step_observations = 2 * 12 + 11 + num_actions  # 54 + 10 + 12 = 22 + 54 = 76
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 3]  # [m]
        lookat = [0., 1, 0.]  # [m]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-0.1, 2.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.5
        randomize_kp = True
        kp_factor_range = [0.8, 1.2]
        randomize_kd = True
        kd_factor_range = [0.8, 1.2]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'HIP_P': 150,
                     'HIP_R': 150,
                     'HIP_Y': 150,
                     'KNEE': 200,
                     'ANKLE_R': 40,
                     'ANKLE_P': 40,
                     # "WAIST": 300,
                     # "SHOULDER": 200,
                     # "WRIST": 20,
                     # "ELBOW": 100,
                     # "NECK": 10
                     }  # [N*m/rad]
        damping = {'HIP_P': 2,
                   'HIP_R': 2,
                   'HIP_Y': 2,
                   'KNEE': 4,
                   'ANKLE_R': 2,
                   'ANKLE_P': 2,
                   # "WAIST": 5,
                   # "SHOULDER": 4,
                   # "WRIST": 0.5,
                   # "ELBOW": 1,
                   # "NECK": 2
                   }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.12
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/assert/T170-V2.1-A0-URDF-A/urdf/Ti_noc.urdf'
        file = '/home/dy/dy/code/unitree_ti/assert/ti5/tai5_12dof_no_limit.urdf'
        name = "TiV2"
        foot_name = "ANKLE_R"
        penalize_contacts_on = ["HIP", "KNEE"]
        terminate_after_contacts_on = ["base_link"]
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.92
        least_feet_distance_lateral = 0.2
        most_feet_distance_lateral = 0.35
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
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
            hip_pos = -50.0
            ankle_pos = -50.0
            # contact_no_vel = -0.2
            feet_swing_height = -100.0
            contact = 1.0
            # standing = 0.18
            # stand_still = -0.15  
            # foot_slip = -0.25
            # foot_contact_forces = -0.00025
            # contact_momentum = 2.5e-4
            # feet_ground_parallel = -2.0
            # feet_parallel = -3.0
            feet_parallel = -1.0
            feet_heading_alignment = -1.0
            
            lin_acc = -2.5e-5
            contact_momentum = 2.5e-4
            foot_contact_forces = -0.0025



            # Bart_tracking_x_vel = 1.5
            # Bart_tracking_y_vel = 1.5
            # Bart_tracking_rot_vel  = 0.1
            # Bart_orientation = 0.2
            # Bart_feet_contact = 0.1
            # Bart_base_height = 5.0
            # # Bart_feet_airtime = 10.0
            # # Bart_feet_orientation = 0.05
            # # Bart_feet_position = 0.05
            # # Bart_base_acc = 0.1
            # # Bart_action_diff = 0.02
            # # Bart_torque = 0.02
            # Bart_hip_pos = 5.0


class TiV2RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        # actor_hidden_dims = [64,64]
        # critic_hidden_dims = [64,64]
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        use_flip = True
        entropy_coef = 0.01
        symmetry_scale = 0.1

    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = "ActorCritic"
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        max_iterations = 100000
        run_name = 'tiv2'
        save_interval = 500
        experiment_name = 'tiv2'