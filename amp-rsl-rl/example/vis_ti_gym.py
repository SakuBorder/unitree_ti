import os
import sys
import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from amp_rsl_rl.utils.motion_lib_taihu import MotionLibTaihu
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from amp_rsl_rl.utils.flags import flags

flags.test = True
flags.im_eval = True

def create_sim(args):
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.001
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")
    return gym, sim

def load_asset(gym, sim, urdf):
    asset_root = os.path.dirname(urdf)
    asset_file = os.path.basename(urdf)
    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = False
    asset_opts.disable_gravity = False
    asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset = gym.load_asset(sim, asset_root, asset_file, asset_opts)
    return asset

def default_pose(gym, asset, num_dofs):
    pose = np.zeros(num_dofs)
    names = [gym.get_asset_dof_name(asset, i) for i in range(num_dofs)]
    for i, n in enumerate(names):
        if 'KNEE_P' in n:
            pose[i] = 0.3
        elif 'ANKLE_P' in n:
            pose[i] = -0.15
        elif 'SHOULDER_R' in n and 'L_' in n:
            pose[i] = -1.0
        elif 'SHOULDER_R' in n and 'R_' in n:
            pose[i] = 1.0
    return pose

def build_env(gym, sim, asset):
    num_dofs = gym.get_asset_dof_count(asset)
    env = gym.create_env(sim, gymapi.Vec3(-2,-2,0), gymapi.Vec3(2,2,2), 1)
    actor = gym.create_actor(env, asset, gymapi.Transform(gymapi.Vec3(0,0,1)),
                             "robot", 0, 1)
    props = gym.get_actor_dof_properties(env, actor)
    props['driveMode'].fill(gymapi.DOF_MODE_POS)
    props['stiffness'].fill(300.0)
    props['damping'].fill(10.0)
    gym.set_actor_dof_properties(env, actor, props)
    return env, actor, num_dofs

def load_motion(motion_file, mjcf_file, device):
    sk_tree = SkeletonTree.from_mjcf(mjcf_file)
    lib = MotionLibTaihu(motion_file=motion_file, device=device,
                         masterfoot_conifg=None, fix_height=False,
                         multi_thread=False, mjcf_file=mjcf_file)
    lib.load_motions([sk_tree], [torch.zeros(17)], [np.zeros(10)])
    return lib

def main():
    custom = [
        gymutil.Argument("--asset", str, "assert/ti5/tai5_12dof_no_limit.urdf", "URDF path"),
        gymutil.Argument("--mjcf", str, "assert/ti5/ti5_12dof.xml", "MJCF path"),
        gymutil.Argument("--motion", str, "data/ti512/v1/singles/walk/B9 -  Walk turn left 90_poses.pkl", "Motion file"),
    ]
    args = gymutil.parse_arguments(description="Taihu Motion Viewer", custom_parameters=custom)
    gym, sim = create_sim(args)
    gym.add_ground(sim, gymapi.PlaneParams())
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_motion")
    asset = load_asset(gym, sim, args.asset)
    env, actor, num_dofs = build_env(gym, sim, asset)
    pose = default_pose(gym, asset, num_dofs)
    dof_state = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    dof_state['pos'] = pose
    gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
    gym.prepare_sim(sim)

    motion_lib = load_motion(args.motion, args.mjcf, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    motion_time = 0.0
    paused = False
    motion_on = True

    while not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "toggle_pause" and evt.value > 0:
                paused = not paused
            elif evt.action == "toggle_motion" and evt.value > 0:
                motion_on = not motion_on
                motion_time = 0.0

        if not paused:
            if motion_on:
                motion_len = motion_lib.get_motion_length(0).item()
                t = torch.tensor([motion_time % motion_len], device=motion_lib._device)
                state = motion_lib.get_motion_state(torch.tensor([0], device=motion_lib._device), t)
                root = torch.cat([state['root_pos'], state['root_rot'],
                                  state['root_vel'], state['root_ang_vel']], dim=-1)
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
                dof = torch.stack([state['dof_pos'], torch.zeros_like(state['dof_pos'])], dim=-1)
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof.view(-1,2)))
                motion_time += 1.0/60.0
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
