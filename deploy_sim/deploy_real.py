import time
import os,sys
# cur_work_path = os.getcwd()
# sys.path.append(cur_work_path)
# import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from threading import Thread
import mujoco
import mujoco.viewer
import keyboard

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()

    config_file = "ti_stand.yaml"
    with open(f"/home/speedbot/dev/t1702.1-dep-ssf_eccan_2.1/policy_deploy/configs/real/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_dt = config["control_dt"]
        # control_dt = 0.5

        # kps = np.array(config["kps"], dtype=np.float32)
        # kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    joint_names = []
    for i in range(m.njnt):
        joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
    joint_names = joint_names[1:]
    ros_joint_names = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'WAIST_P', 'L_ANKLE_R', 'R_KNEE_P', 'WAIST_R', 'WAIST_Y', 'L_HIP_P', 
                       'L_SHOULDER_P', 'L_ANKLE_P', 'R_SHOULDER_P', 'L_HIP_Y', 'L_HIP_R', 'R_ANKLE_R', 'L_KNEE_P', 'R_ANKLE_P']
    joint_indices = np.zeros(len(joint_names),dtype=np.int32)
    for index,joint_name in enumerate(joint_names):
        joint_indices[index] = ros_joint_names.index(joint_name)
    # print(joint_indices)
    # exit()


    
    


    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    # action = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # load policy
    policy = torch.jit.load(policy_path)


    low_state_data = np.zeros(7+num_actions*2,dtype=np.float32)
    init_angle = np.zeros(17,dtype=np.float32)

    # action = None
    def imu_data_listener_callback(msg):
        global low_state_data
        ang_vel = np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])
        quat = np.array([msg.orientation.w,msg.orientation.x,msg.orientation.y,msg.orientation.z])
        # import pdb
        # pdb.set_trace()
        low_state_data[0:3] = ang_vel
        low_state_data[3:7] = quat

    # ros_joint_name = None
    def joint_states_listener_callback(msg):
        global low_state_data,ros_joint_name
        joint_pos = np.array(msg.position)
        joint_vel = np.array(msg.velocity)
        # ros_joint_name = msg.name
        # print(joint_pos)
        # print(joint_vel.shape)
        # import pdb
        # pdb.set_trace()
        low_state_data[7:7+num_actions] = joint_pos[joint_indices]
        low_state_data[7+num_actions:7+num_actions*2] = joint_vel[joint_indices]
        # import pdb
        # pdb.set_trace()
        # print(1)


    rclpy.init()
    node1 = Node("test")
    imu_data_sub = node1.create_subscription(Imu,'/imu/data',imu_data_listener_callback,1)
    joint_states_sub = node1.create_subscription(JointState,'/joint_states',joint_states_listener_callback,1)
    
    
    cmd_pub = node1.create_publisher(Float64MultiArray, '/rl_motion_control_command', 1)

    # rclpy.spin(node1)

    low_state_sub_thread = Thread(target=rclpy.spin, args=(node1,), daemon=True)
    low_state_sub_thread.start()

    # print(joint_names)
    # exit()
    # while True:
    #     # print(len(low_state_data))
    #     pass
    time.sleep(1)
    time0 = time.time()
    counter = 0
    control_counter = 0

    joint_states_info = np.zeros((100,7+2*num_actions))
    target_dof_pos_info = np.zeros((100,12))
    # log_index= 0

    target_dof_pos_list = torch.load('/home/speedbot/dev/t1702.1-dep-ssf_eccan_2.1/policy_deploy/trjs/target_dof_pos_list.pt')[200:]
    cur_dof_pos_list = torch.load('/home/speedbot/dev/t1702.1-dep-ssf_eccan_2.1/policy_deploy/trjs/cur_dof_pos_list.pt')

    target_dof_pos = cur_dof_pos_list[200]
    
    while True:
        # msg = Float64MultiArray()
        # msg.data = (
        #     target_dof_pos.tolist() + [0.0,0.0,0.0,0.0,0.0]
        # )
        # cmd_pub.publish(msg)
        # node1.get_logger().warn(f'I heard: "{target_dof_pos}"')
        # print(target_dof_pos)
        if keyboard.is_pressed('a'):
            break
        
    print("exit")
    exit()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        time.sleep(3)
        while True:
            # time.sleep(3)
            # print(ros_joint_name)

            counter += 1
            control_counter += 1
            
            omega = np.array(low_state_data[0:3])
            quat = np.array(low_state_data[3:7])
            qj = np.array(low_state_data[7:7+num_actions])
            dqj = np.array(low_state_data[7+num_actions:7+2*num_actions])
            # sim_counter = np.array(low_state_data[-1:])



            # print(qj)
            d.qpos[7:7+12] = qj
            mujoco.mj_forward(m, d)
            viewer.sync()
            
            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale

            period = 0.8
            count = control_counter * control_dt
            # count = sim_counter[0] * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            # import pdb
            # pdb.set_trace() 
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)

            # policy inference
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
            target_dof_pos = target_dof_pos_list[control_counter-1]
            # target_dof_pos = cur_dof_pos_list[control_counter-1]

            # action = np.clip(action,)
            # transform action to target_dof_pos
            # if control_counter < 100:
                
            #     joint_states_info[control_counter] = np.concatenate((omega,quat,qj,dqj),axis=0)
            #     target_dof_pos_info[control_counter] = target_dof_pos
            #     np.savetxt('joint_states.txt',joint_states_info)
            #     np.savetxt('target_dof_pos.txt',target_dof_pos_info)

            # d.qpos[7:7+12] = qj
            # mujoco.mj_forward(m, d)

            msg = Float64MultiArray()
            msg.data = (
                target_dof_pos.tolist() + [0.0,0.0,0.0,0.0,0.0]
            )
            cmd_pub.publish(msg)
            node1.get_logger().warn(f'I heard: "{target_dof_pos}"')
            print(target_dof_pos)
            # exit()
            # viewer.sync()
            time.sleep(control_dt)
            
            # print(time.time() - time0)
            if time.time() - time0 >1 :
                print(counter,"hz")
                counter = 0
                time0 = time.time()
    

