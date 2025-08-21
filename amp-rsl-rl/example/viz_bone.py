import os
import sys
import time
import argparse
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

# ========== 可视化参数配置 ==========
DT = 1.0 / 30.0           # 播放帧率
JOINT_MARK_MAX = 64       # 预分配的关节可视化"胶囊/小段"数量上限
JOINT_MARK_RADIUS = 0.03  # 关节标记半径
BONE_RADIUS = 0.02        # 骨骼连接线的半径
BONE_MAX = 64             # 预分配的骨骼连接数量上限

# SMPL 骨架连接定义（24个关节的连接关系）
SMPL_SKELETON_CONNECTIONS = [
    # 躯干主干
    (0, 1), (0, 2), (0, 3),     # 骨盆到脊椎、左髋、右髋
    (1, 4), (2, 5), (3, 6),     # 脊椎到胸部，髋到膝盖
    (4, 7), (5, 8), (6, 9),     # 胸部到颈部，膝盖到脚踝
    (7, 10), (8, 11), (9, 12),  # 颈部到头，脚踝到脚
    
    # 手臂
    (4, 13), (4, 16),           # 胸部到左右肩膀
    (13, 14), (16, 17),         # 肩膀到肘部
    (14, 15), (17, 18),         # 肘部到手腕
    
    # 手部（如果有的话）
    (15, 19), (18, 22),         # 手腕到手
    (19, 20), (20, 21),         # 左手指
    (22, 23),                   # 右手指
]

# 颜色定义
JOINT_COLOR = np.array([0.2, 0.7, 1.0, 1.0], dtype=np.float32)      # 关节颜色：蓝色
BONE_COLOR = np.array([1.0, 0.3, 0.3, 0.8], dtype=np.float32)       # 骨骼颜色：红色
SPINE_COLOR = np.array([0.3, 1.0, 0.3, 0.8], dtype=np.float32)      # 脊椎颜色：绿色
ARM_COLOR = np.array([1.0, 1.0, 0.3, 0.8], dtype=np.float32)        # 手臂颜色：黄色

def get_bone_color(connection):
    """根据骨骼连接类型返回不同颜色"""
    joint1, joint2 = connection
    
    # 脊椎相关：绿色
    if (joint1, joint2) in [(0, 1), (1, 4), (4, 7), (7, 10)]:
        return SPINE_COLOR
    
    # 手臂相关：黄色
    if joint1 >= 13 or joint2 >= 13:
        return ARM_COLOR
    
    # 其他骨骼：红色
    return BONE_COLOR

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        float(point1[0]), float(point1[1]), float(point1[2]),
        float(point2[0]), float(point2[1]), float(point2[2]),
    )

def add_visual_sphere(scene, center, radius, rgba):
    """添加一个球体到场景中"""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3),
        np.zeros(3),  
        np.zeros(9),
        rgba.astype(np.float32),
    )
    scene.geoms[scene.ngeom - 1].pos = center.astype(np.float32)
    scene.geoms[scene.ngeom - 1].size[0] = radius

# --- 键盘回调 ---
def key_call_back(keycode):
    global time_step, paused, motion_id, motion_data_keys, show_skeleton, show_joints
    try:
        ch = chr(keycode)
    except Exception:
        ch = ""
    if ch == "R":
        print("[viewer] Reset")
        time_step = 0.0
    elif ch == " ":
        paused = not paused
        print(f"[viewer] Paused = {paused}")
    elif ch == "T":
        motion_id = (motion_id + 1) % len(motion_data_keys)
        print(f"[viewer] Next motion: {motion_data_keys[motion_id]}")
    elif ch == "S":
        show_skeleton = not show_skeleton
        print(f"[viewer] Show skeleton = {show_skeleton}")
    elif ch == "J":
        show_joints = not show_joints
        print(f"[viewer] Show joints = {show_joints}")
    elif ch == "H":
        print("[viewer] Help:")
        print("  R - Reset animation")
        print("  Space - Pause/Resume")
        print("  T - Switch to next motion")
        print("  S - Toggle skeleton display")
        print("  J - Toggle joint display")
        print("  H - Show help")
    else:
        pass

def _load_one_motion_file(base_dir, humanoid_type, motion_name=None):
    singles_dir = osp.join("data", humanoid_type, "v1", "singles")
    if not osp.isabs(singles_dir):
        singles_dir = osp.abspath(singles_dir)

    if motion_name:
        # 去掉可能的 .pkl 后缀
        motion_name = motion_name[:-4] if motion_name.endswith(".pkl") else motion_name
        fpath = osp.join(singles_dir, f"{motion_name}.pkl")
        if not osp.exists(fpath):
            raise FileNotFoundError(f"Motion file not found: {fpath}")
        return fpath

    # 自动选择目录下第一个 .pkl
    all_pkls = [p for p in os.listdir(singles_dir) if p.endswith(".pkl")]
    if not all_pkls:
        raise FileNotFoundError(f"No motion .pkl in {singles_dir}")
    fpath = osp.join(singles_dir, sorted(all_pkls)[0])
    print(f"[viewer] motion_name not provided, auto pick: {osp.basename(fpath)}")
    return fpath

@hydra.main(version_base=None,
            config_path="/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/cfg",
            config_name="taihu")
def main(cfg: DictConfig) -> None:
    """
    运行方式（两种）：
    1) 在 taihu.yaml 里增加/设置： motion_name: <保存时的key名>
    2) 不设置 motion_name：自动加载 singles 目录下第一个 .pkl
    
    键盘控制：
    - R: 重置动画
    - Space: 暂停/继续
    - T: 切换到下一个动作
    - S: 显示/隐藏骨架
    - J: 显示/隐藏关节点
    - H: 显示帮助
    """

    global time_step, paused, motion_id, motion_data_keys, show_skeleton, show_joints
    device = torch.device("cpu")

    # 显示控制
    show_skeleton = True
    show_joints = True

    # 机器人 XML
    humanoid_xml = cfg.robot.asset.assetFileName
    if not osp.exists(humanoid_xml):
        raise FileNotFoundError(f"Robot XML not found: {humanoid_xml}")

    # 选择要可视化的 motion 文件
    motion_file = _load_one_motion_file(os.getcwd(), cfg.robot.humanoid_type, cfg.get("motion_name", None))
    print(f"[viewer] Motion file: {motion_file}")

    # 读取数据（结构：{data_key: {...}}）
    motion_data = joblib.load(motion_file)
    if not isinstance(motion_data, dict) or len(motion_data) == 0:
        raise ValueError(f"Bad motion data in {motion_file}")
    motion_data_keys = list(motion_data.keys())
    print(f"[viewer] Keys in file: {motion_data_keys}")

    # 检查是否有SMPL关节数据
    has_smpl_joints = False
    for key in motion_data_keys:
        if "smpl_joints" in motion_data[key]:
            has_smpl_joints = True
            break
    
    if has_smpl_joints:
        print("[viewer] Found SMPL joint data - 3D skeleton visualization enabled")
        print("[viewer] Controls: S=skeleton on/off, J=joints on/off, H=help")
    else:
        print("[viewer] No SMPL joint data found - only robot visualization")

    # MuJoCo 初始化
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = DT

    # 播放控制
    time_step = 0.0
    paused = False
    motion_id = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        print("[viewer] 3D Skeleton Visualizer Started")
        print("Press 'H' for help")
        
        while viewer.is_running():
            step_start = time.time()

            # 清空之前的可视化几何体
            viewer.user_scn.ngeom = 0

            # 当前 motion
            curr_key = motion_data_keys[motion_id]
            curr = motion_data[curr_key]

            # 取当前帧
            T = curr["dof"].shape[0]
            curr_idx = int(time_step / DT) % T

            # 赋值 root（平移 + 四元数）和 dof
            root_trans = curr["root_trans_offset"][curr_idx]           # (3,)
            root_quat_xyzw = curr["root_rot"][curr_idx]                # (4,) xyzw
            dof = curr["dof"][curr_idx]                                 # (num_dof,) 或 (num_dof,1)

            mj_data.qpos[:3] = root_trans.astype(np.float64)
            # MuJoCo 使用 wxyz 顺序
            mj_data.qpos[3:7] = np.array(
                [root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]],
                dtype=np.float64
            )
            # dof 展平
            dof_flat = dof.reshape(-1).astype(np.float64)
            mj_data.qpos[7:7 + dof_flat.shape[0]] = dof_flat

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += DT

            # 绘制3D骨架（如果存在SMPL关节数据）
            if has_smpl_joints and "smpl_joints" in curr:
                joint_positions = curr["smpl_joints"][curr_idx]  # (J, 3)
                J = joint_positions.shape[0]
                
                # 1. 绘制骨骼连接
                if show_skeleton:
                    for connection in SMPL_SKELETON_CONNECTIONS:
                        joint1_idx, joint2_idx = connection
                        if joint1_idx < J and joint2_idx < J:
                            point1 = joint_positions[joint1_idx]
                            point2 = joint_positions[joint2_idx]
                            color = get_bone_color(connection)
                            
                            add_visual_capsule(
                                viewer.user_scn,
                                point1,
                                point2,
                                BONE_RADIUS,
                                color
                            )
                
                # 2. 绘制关节点
                if show_joints:
                    for i in range(min(J, JOINT_MARK_MAX)):
                        joint_pos = joint_positions[i]
                        add_visual_sphere(
                            viewer.user_scn,
                            joint_pos,
                            JOINT_MARK_RADIUS,
                            JOINT_COLOR
                        )

            # 显示当前状态信息
            if curr_idx == 0:  # 每次循环开始时显示一次
                fps = curr.get("fps", 30)
                frame_info = f"Motion: {curr_key} | Frame: {curr_idx+1}/{T} | FPS: {fps}"
                if has_smpl_joints:
                    skeleton_info = f" | Skeleton: {'ON' if show_skeleton else 'OFF'} | Joints: {'ON' if show_joints else 'OFF'}"
                    frame_info += skeleton_info
                print(f"\r{frame_info}", end="", flush=True)

            # 同步显示
            viewer.sync()

            # 控制节拍
            remain = mj_model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

if __name__ == "__main__":
    main()