import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import quat_mul, quat_rotate, quat_conjugate

sys.path.append(os.getcwd())

from amp_rsl_rl.utils.motion_lib_taihu import MotionLibTaihu
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from amp_rsl_rl.utils.flags import flags
import torch

flags.test = True
flags.im_eval = True

args = gymutil.parse_arguments(description="Robot Motion Visualizer")

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

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

asset_root = "./"
asset_file = "assert/ti5/tai5_12dof_no_limit.urdf"
g1_xml = "assert/ti5/ti5_12dof.xml"

print(f"Loading asset: {asset_file}")

if not os.path.exists(asset_file):
    print(f"*** Asset file not found: {asset_file}")
    quit()

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = True
asset_options.vhacd_enabled = False
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

try:
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    print("Asset loaded successfully")
except Exception as e:
    print(f"*** Failed to load asset: {e}")
    quit()

num_dofs = gym.get_asset_dof_count(asset)
num_bodies = gym.get_asset_rigid_body_count(asset)
print(f"Asset info: {num_dofs} DOFs, {num_bodies} bodies")

env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)  
env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

dof_props = gym.get_actor_dof_properties(env, actor_handle)
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 300.0
    dof_props['damping'][i] = 10.0
    dof_props['velocity'][i] = 100.0
    dof_props['effort'][i] = 100.0

gym.set_actor_dof_properties(env, actor_handle, dof_props)

def get_standing_pose():
    """定义一个基本的站立姿态"""
    pose = np.zeros(num_dofs)
    
    joint_names = []
    for i in range(num_dofs):
        joint_names.append(gym.get_asset_dof_name(asset, i))
    
    for i, name in enumerate(joint_names):
        if 'HIP_P' in name:
            pose[i] = 0.0
        elif 'HIP_R' in name:
            pose[i] = 0.0
        elif 'HIP_Y' in name:
            pose[i] = 0.0
        elif 'KNEE_P' in name:
            pose[i] = 0.3
        elif 'ANKLE_P' in name:
            pose[i] = -0.15
        elif 'ANKLE_R' in name:
            pose[i] = 0.0
        elif 'SHOULDER_R' in name: 
            if 'L_' in name:
                pose[i] = -1.0 
            else:
                pose[i] = 1.0 
        else:
            pose[i] = 0.0
    
    return pose

 
standing_pose = get_standing_pose()

print("Loading motion library...")
motion_file = "data/ti512/v1/singles/walk/B9 -  Walk turn left 90_poses.pkl"

if not os.path.exists(motion_file):
    print(f"*** Motion file not found: {motion_file}")
    print("Will use standing pose only")
    motion_lib = None
    sk_tree = None
else:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sk_tree = SkeletonTree.from_mjcf(g1_xml)
        motion_lib = MotionLibTaihu(
            motion_file=motion_file, 
            device=device, 
            masterfoot_conifg=None, 
            fix_height=False, 
            multi_thread=False, 
            mjcf_file=g1_xml
        )
        
        num_motions = 1
        motion_lib.load_motions(
            skeleton_trees=[sk_tree] * num_motions,
            gender_betas=[torch.zeros(17)] * num_motions,
            limb_weights=[np.zeros(10)] * num_motions,
            random_sample=True
        )
        motion_keys = motion_lib.curr_motion_keys
        print(f"Motion library loaded successfully with {num_motions} motions")
        print(f"Motion keys: {motion_keys}")
    except Exception as e:
        print(f"Failed to load motion library: {e}")
        motion_lib = None
        sk_tree = None

 
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
for i in range(num_dofs):
    dof_states['pos'][i] = standing_pose[i]
    dof_states['vel'][i] = 0.0

gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

print("Preparing simulation...")
gym.prepare_sim(sim)

num_envs = 1
env_ids = torch.arange(num_envs, dtype=torch.int32, device=args.sim_device)

current_pose = torch.tensor(standing_pose, dtype=torch.float32, device=args.sim_device).unsqueeze(0)

dof_states_tensor = gym.acquire_dof_state_tensor(sim)
dof_states_tensor = gymtorch.wrap_tensor(dof_states_tensor)

rigidbody_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state_tensor = gymtorch.wrap_tensor(rigidbody_state_tensor)
rigidbody_state_tensor = rigidbody_state_tensor.reshape(num_envs, -1, 13)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)

try:
    contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
    contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
    contact_force_tensor = contact_force_tensor.view(num_envs, -1, 3)
    contact_available = True
    print("Contact force tensor acquired successfully")
except Exception as e:
    print(f"Warning: Could not acquire contact force tensor: {e}")
    contact_available = False

body_names = []
for i in range(num_bodies):
    body_names.append(gym.get_asset_rigid_body_name(asset, i))
print(f"Rigid bodies: {body_names}")

# CollisionColorManager 类实例化
class CollisionColorManager:
    def __init__(self, gym_instance, env, actor_handle, num_bodies):
        self.gym = gym_instance
        self.env = env
        self.actor_handle = actor_handle
        self.num_bodies = num_bodies
        self.original_colors = {}
        self.collision_bodies = set()

        self.normal_color = gymapi.Vec3(0.8, 0.8, 0.8) 
        self.collision_color = gymapi.Vec3(1.0, 0.0, 0.0) 
        
        self.reset_all_colors()
    
    def reset_all_colors(self):
        """重置所有刚体为默认颜色"""
        for i in range(self.num_bodies):
            try:
                self.gym.set_rigid_body_color(self.env, self.actor_handle, i, 
                                            gymapi.MESH_VISUAL, self.normal_color)
            except:
                pass 
    
    def update_collision_colors(self, contact_forces, force_threshold=1.0):
        """根据接触力更新刚体颜色"""
        current_collision_bodies = set()

        for body_idx in range(min(contact_forces.shape[1], self.num_bodies)):
            contact_force = contact_forces[0, body_idx] 
            force_magnitude = torch.norm(contact_force).item()

            if force_magnitude > force_threshold:
                current_collision_bodies.add(body_idx)

        internal_collision_bodies = self.detect_internal_collisions(current_collision_bodies)

        bodies_to_update = set()

        for body_idx in self.collision_bodies - internal_collision_bodies:
            bodies_to_update.add((body_idx, False)) 

        for body_idx in internal_collision_bodies - self.collision_bodies:
            bodies_to_update.add((body_idx, True)) 

        for body_idx, is_collision in bodies_to_update:
            try:
                color = self.collision_color if is_collision else self.normal_color
                self.gym.set_rigid_body_color(self.env, self.actor_handle, body_idx, 
                                            gymapi.MESH_VISUAL, color)
            except:
                pass

        self.collision_bodies = internal_collision_bodies
        
        return len(internal_collision_bodies)

    def detect_internal_collisions(self, contact_bodies):
        """检测内部碰撞（自碰撞）"""
        internal_collisions = set()

        exclusion_pairs = [
            ('L_SHOULDER_P_S', 'L_ELBOW_Y_S'),
            ('L_SHOULDER_R_S', 'L_WRIST_P_S'),
            ('L_SHOULDER_Y_S', 'L_WRIST_R_S'),
            ('R_SHOULDER_P_S', 'R_ELBOW_Y_S'),
            ('R_SHOULDER_R_S', 'R_WRIST_P_S'),
            ('R_SHOULDER_Y_S', 'R_WRIST_R_S'),
            ('L_HIP_P_S', 'L_KNEE_P_S'),
            ('L_HIP_R_S', 'L_ANKLE_P_S'),
            ('R_HIP_P_S', 'R_KNEE_P_S'),
            ('R_HIP_R_S', 'R_ANKLE_P_S'),
            ('BASE_S', 'L_WRIST_R_S'),
            ('BASE_S', 'R_WRIST_R_S'),
            ('WAIST_P_S', 'L_ANKLE_R_S'),
            ('WAIST_P_S', 'R_ANKLE_R_S'),
        ]

        ground_contact_bodies = ['L_ANKLE_R_S', 'R_ANKLE_R_S']

        for body_idx in contact_bodies:
            if body_idx < len(body_names):
                body_name = body_names[body_idx]

                if body_name not in ground_contact_bodies:
                    internal_collisions.add(body_idx)

        return internal_collisions

color_manager = CollisionColorManager(gym, env, actor_handle, num_bodies)

# 控制按键
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_pause")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "toggle_collision_vis")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "toggle_debug")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "apply_fix")

print("Simulation ready!")
print("Controls:")
print("  ESC - Exit")
print("  SPACE - Pause/Resume")
print("  M - Toggle Motion/Standing")
print("  R - Reset motion")
print("  C - Toggle collision visualization")
print("  D - Toggle debug mode")
print("  F - Apply joint mapping fix")

paused = False
use_motion = motion_lib is not None
motion_time = 0.0
motion_id = 0
dt = sim_params.dt
show_collision_vis = True
collision_count = 0
debug_mode = True 
apply_joint_fix = False 

print("Starting visualization...")
if motion_lib is not None:
    print("Motion playback enabled")
else:
    print("Standing pose only (no motion file)")

while not gym.query_viewer_has_closed(viewer):

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "toggle_pause" and evt.value > 0:
            paused = not paused
            print(f"Simulation {'PAUSED' if paused else 'RESUMED'}")
        elif evt.action == "toggle_motion" and evt.value > 0 and motion_lib is not None:
            use_motion = not use_motion
            motion_time = 0.0
            print(f"Motion {'ENABLED' if use_motion else 'DISABLED'}")
        elif evt.action == "reset" and evt.value > 0:
            motion_time = 0.0
            color_manager.reset_all_colors()
            print("Motion reset and colors restored")
        elif evt.action == "toggle_collision_vis" and evt.value > 0:
            show_collision_vis = not show_collision_vis
            if not show_collision_vis:
                color_manager.reset_all_colors()
            print(f"Collision visualization {'ON' if show_collision_vis else 'OFF'}")
        elif evt.action == "toggle_debug" and evt.value > 0:
            debug_mode = not debug_mode
            print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
        elif evt.action == "apply_fix" and evt.value > 0:
            apply_joint_fix = not apply_joint_fix
            print(f"Joint mapping fix {'APPLIED' if apply_joint_fix else 'DISABLED'}")

    if not paused:
        if use_motion and motion_lib is not None:

            try:
                motion_len = motion_lib.get_motion_length(motion_id).item()
                motion_time_wrapped = motion_time % motion_len

                motion_res = motion_lib.get_motion_state(
                    torch.tensor([motion_id]).to(args.compute_device_id),
                    torch.tensor([motion_time_wrapped]).to(args.compute_device_id)
                )
                root_pos = motion_res["root_pos"]
                root_rot = motion_res["root_rot"]
                dof_pos = motion_res["dof_pos"]

                root_vel = motion_res["root_vel"]
                root_ang_vel = motion_res["root_ang_vel"]

                # rotate from original Y-up to Z-up
                y_to_z_quat = torch.tensor(
                    [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                    device=root_rot.device,
                    dtype=root_rot.dtype,
                ).repeat(root_rot.shape[0], 1)

                rot_mat = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    device=root_pos.device,
                    dtype=root_pos.dtype,
                )

                y_to_z_conj = quat_conjugate(y_to_z_quat)
                root_rot = quat_mul(y_to_z_quat, quat_mul(root_rot, y_to_z_conj))
                root_pos = root_pos @ rot_mat.T
                root_vel = root_vel @ rot_mat.T
                root_ang_vel = root_ang_vel @ rot_mat.T

                if debug_mode and int(motion_time * 10) % 30 == 0: 
                    print(f"\n=== Motion Debug Info (t={motion_time:.2f}) ===")
                    print(f"DOF positions shape: {dof_pos.shape}")
                    print(f"DOF values: {dof_pos[0]}")

                    joint_names = []
                    for i in range(num_dofs):
                        joint_names.append(gym.get_asset_dof_name(asset, i))

                    for i, name in enumerate(joint_names):
                        if 'R_' in name and ('SHOULDER' in name or 'ELBOW' in name or 'WRIST' in name):
                            print(f"  {i:2d}: {name:20s} = {dof_pos[0][i].item():8.3f}")

                if apply_joint_fix:
                    dof_pos_clamped = create_joint_mapping_fix(dof_pos)
                else:
                    dof_pos_clamped = dof_pos.clone()
                    for i in range(num_dofs):
                        dof_pos_clamped[0][i] = torch.clamp(dof_pos_clamped[0][i], -3.14, 3.14)

                if "rg_pos" in motion_res:
                    rb_pos = motion_res["rg_pos"] @ rot_mat.T
                    gym.clear_lines(viewer)
                    gym.refresh_rigid_body_state_tensor(sim)

                    for pos_joint in rb_pos[0, 1:]:
                        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0.0, 0.0))
                        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                        gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)

                root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
                gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), 
                                                      gymtorch.unwrap_tensor(env_ids), len(env_ids))

                dof_state = torch.stack([dof_pos_clamped, torch.zeros_like(dof_pos_clamped)], dim=-1).squeeze().repeat(num_envs, 1)
                gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), 
                                               gymtorch.unwrap_tensor(env_ids), len(env_ids))

                motion_time += dt

            except Exception as e:
                print(f"Motion playback error: {e}")
                use_motion = False

        else:

            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(current_pose))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        if contact_available and show_collision_vis:
            try:
                gym.refresh_net_contact_force_tensor(sim)
                collision_count = color_manager.update_collision_colors(
                    contact_force_tensor, force_threshold=0.5
                )

                if int(motion_time * 10) % 60 == 0 and collision_count > 0: 
                    print(f"Internal collisions detected: {collision_count} bodies")

            except Exception as e:
                if int(motion_time * 10) % 300 == 0: 
                    print(f"Collision detection error: {e}")

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

print("Cleaning up...")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Done!")
