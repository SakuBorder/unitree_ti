import time
import os,sys
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)
import mujoco.viewer
import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import collections

from scipy.spatial.transform import Rotation as R


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def remap_phase(phase):
    if phase < 0.7:
        # 前 80% 匀速
        return phase
    else:
        # 后 20% 拉伸 —— 用平方映射 (慢 → 快)
        normalized = (phase - 0.7) / 0.3  # 映射到 [0,1]
        mapped = normalized ** 2           # 平方: 前慢后快
        return 0.7 + 0.3 * mapped
    
if __name__ == "__main__":
    # get config file name from command line
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()


    policy_path = 'logs/ti-12dof-realpd/exported/policies/policy_1.pt'
    xml_path = 'assert/ti5/ti_no_limit.xml'

    simulation_duration = 10000
    simulation_dt = 0.005*2
    control_decimation = 2
    control_dt = simulation_dt*control_decimation

    kps = np.array([150,150,150,200,40,40,150,150,150,200,40,40], dtype=np.float32)
    kds = np.array([2,2,2,4,2,2,2,2,2,4,2,2], dtype=np.float32)

    # default_angles = np.array([0]*14, dtype=np.float32)
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, 0.2, 0.0,
                                0.1, 0.0, 0.0, -0.3, -0.2, 0.0], dtype=np.float32)

    # default_angles = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                             0.0, 0.0], dtype=np.float32)
    
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.12
    cmd_scale = np.array([2.0,2.0,1.0], dtype=np.float32)

    num_actions = 12
    num_obs = 47
    
    cmd = np.array([0.5,0.0,0.0], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    sim_counter = 0
    control_counter = 0


    def compute_observation(d, action, cmd, n_joints):
        """Compute the observation vector from current state"""
        # Get state from MuJoCo
        qj = d.qpos[7:7+n_joints].copy()
        dqj = d.qvel[6:6+n_joints].copy()
        quat = d.qpos[3:7].copy()
        # print(quat)
        # quat = np.array([1.0,0.0,0.0,0.0])
        omega = d.qvel[3:6].copy()
        
        # Scale the values
        qj_scaled = (qj - default_angles) * dof_pos_scale
        dqj_scaled = dqj * dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * ang_vel_scale

        period = 0.8 # 1.05
        # count = sim_counter * (simulation_dt)
        count = control_counter * control_dt
        phase = count % period / period
        # phase = remap_phase(phase)

        # left_phase = (phase)
        # right_phase = ((phase + 0.5) % 1)

        # left_phase = remap_phase(phase)
        # right_phase = remap_phase((phase + 0.5) % 1)
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        

        # Calculate single observation dimension
        single_obs_dim = 11 + n_joints + n_joints + 12
        
        # Create single observation
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[:3] = omega_scaled
        single_obs[3:6] = gravity_orientation
        single_obs[6:9] = cmd * cmd_scale
        single_obs[9 : 9 + num_actions] = qj_scaled
        single_obs[9 + num_actions : 9 + 2 * num_actions] = dqj_scaled
        single_obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
        single_obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

        return single_obs, single_obs_dim
    
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d,  action, cmd, num_actions)

    obs_history_len = 6
    obs_history = collections.deque(maxlen=obs_history_len)
    for _ in range(obs_history_len):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    num_obs = obs_history_len*single_obs_dim
    obs = np.zeros(num_obs, dtype=np.float32)
    # sim_counter = 0
    # control_counter = 0




    torque_limits_lower = np.array([m.actuator_ctrlrange[i][0] for i in range(m.nu)], dtype=np.float32)
    torque_limits_upper = np.array([m.actuator_ctrlrange[i][1] for i in range(m.nu)], dtype=np.float32)
    # print(torque_limits)

    log_dir = 'debug/mujoco'
    import shutil
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)  # 创建tensorboard写入器
    for i in range(m.njnt):
        name = m.joint(i).name
        addr = m.jnt_qposadr[i]  # Starting index in qpos
        dof = m.jnt_dofadr[i]    # Starting index in qvel / dof

        print(f"{i}: {name} | qpos addr = {addr} | dof addr = {dof}" )
        # force range = [{m.actuator_ctrlrange[i][0]},{m.actuator_ctrlrange[i][1]}]

        # transform action to target_dof_pos
        # import pdb
        # pdb.set_trace()
        # print(m.actuator_forcerange[i][0],m.actuator_forcerange[i][1])

    # load policy
    policy = torch.jit.load(policy_path)
    # target_dof_pos_list = []

    # cur_dof_pos_list = []
    # target_dof_pos_list = torch.load('/home/speedbot/dev/t1702.1-dep-ssf_eccan_2.1/policy_deploy/trjs/target_dof_pos_list.pt')[200:]
    # cur_dof_pos_list = torch.load("/home/speedbot/dev/t1702.1-dep-ssf_eccan_2.1/policy_deploy/trjs/cur_dof_pos_list.pt")
    # import pdb
    # pdb.set_trace()
    # d.qpos[:3] = np.array([0.0,0.0,0.92])
    # d.qpos[7:7+12] = cur_dof_pos_list[200]
    # mujoco.mj_forward(m, d)


    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        time.sleep(1)
        
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # torch.clip(torques, -self.torque_limits, self.torque_limits)
            # d.ctrl[:] = np.clip(tau,torque_limits_lower, torque_limits_upper)
            # print(tau)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # sim_counter += 1
            if sim_counter % control_decimation == 0:
                # print(sim_counter)
                # if sim_counter>250 and sim_counter<500:
                #     cmd = np.array([1.0,0.0,0.0], dtype=np.float32)
                # elif sim_counter>500 and sim_counter<750:
                #     cmd = np.array([0.0,0.0,0.0], dtype=np.float32)
                # elif sim_counter>750:
                #     cmd = np.array([1.0,0.0,0.0], dtype=np.float32)

                control_counter += 1
                # Apply control signal here.
                # Update observation
                single_obs, _ = compute_observation(d, action, cmd, num_actions)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference

                # import pdb
                # pdb.set_trace()
                # obs_tensor.fill_(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -100, 100)
                # print(action)
                # transform action to target_dof_pos
                # import pdb
                # pdb.set_trace()
                # swing_amplitude = 0.0
                # action[5] = 0.0
                # action[5+6] = 0.0
                # action_with_hand =  np.concatenate((action, np.array([swing_amplitude * cos_phase,-swing_amplitude * cos_phase])), axis=0)
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = target_dof_pos_list[control_counter-1]
                # print("target_dof_pos:",target_dof_pos)
                # print(control_counter,target_dof_pos)
                # target_dof_pos_list.append(target_dof_pos)

                
                # action[4+6] *= 1.1
                # action[5] /= 1.2
                # action[5+6] /= 1.2
                

                # 对应0-5和6-11维度成对画图
                for j in range(6):
                    writer.add_scalars(
                        f'fig_mujoco/action_{j}_vs_{j+6}',  # 不包含 '/' 的 tag
                        {
                            f'action_{j}': action[j].item(),
                            f'action_{j+6}': action[j+6].item()
                        },
                        control_counter
                    )

                for j in range(6):
                    writer.add_scalars(
                        f'fig_mujoco/dof_{j}_vs_{j+6}',  # 不包含 '/' 的 tag
                        {
                            f'dof_{j}': d.qpos[7+j].item(),
                            f'dof_{j+6}': d.qpos[7+j+6].item()
                        },
                        control_counter
                    )
                
                quat = d.qpos[3:7]
                r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy是 [x,y,z,w]
                euler = r.as_euler('xyz', degrees=True)    # (roll, pitch, yaw)             
                writer.add_scalar("euler/roll", euler[0], control_counter)
                writer.add_scalar("euler/pitch", euler[1], control_counter)
                writer.add_scalar("euler/yaw", euler[2], control_counter)
                    
                # print(d.qpos[3:7])
                # base_ang_vel_x = env.base_ang_vel[0,0]
                # base_ang_vel_y = env.base_ang_vel[0,1]
                # writer.add_scalar(
                #     f'fig/base_ang_vel_x',  # 不包含 '/' 的 tag
                #     base_ang_vel_x.cpu().numpy().item(),
                #     i
                # )
                # writer.add_scalar(
                #     f'fig/base_ang_vel_y',  # 不包含 '/' 的 tag
                #     base_ang_vel_y.cpu().numpy().item(),
                #     i
                # )

                # print(control_counter)
                # if control_counter == 500:
                #     torch.save(cur_dof_pos_list, "cur_dof_pos_list.pt")
                #     print("saved cur_dof_pos !!!")
                # target_dof_pos[5] =0
                # target_dof_pos[5+6] =0

                # print(target_dof_pos)
            
            sim_counter += 1
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # d.qpos[7:7+12] = target_dof_pos
            # mujoco.mj_forward(m, d)
            time.sleep(0.01)

            viewer.sync()
            
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # print(time_until_next_step)
            if time_until_next_step > 0:
                # print(1)
                time.sleep(time_until_next_step)