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

# ========== 可选：你可以调整这些默认显示参数 ==========
DT = 1.0 / 30.0           # 播放帧率
JOINT_MARK_MAX = 64       # 预分配的关节可视化“胶囊/小段”数量上限
JOINT_MARK_RADIUS = 0.03  # 关节标记半径
# ======================================================

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

# --- 键盘回调 ---
def key_call_back(keycode):
    global time_step, paused, motion_id, motion_data_keys
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
    else:
        pass

def _load_one_motion_file(base_dir, humanoid_type, motion_name=None):
    singles_dir = osp.join("data", humanoid_type, "v1", "singles", "maikan")
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
    """

    global time_step, paused, motion_id, motion_data_keys
    device = torch.device("cpu")

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

    # MuJoCo 初始化
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = DT

    # 播放控制
    time_step = 0.0
    paused = False
    motion_id = 0

    # 关节可视化：预先创建一些“胶囊”几何体来动态移动到关节位置
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        # 预创建 N 个关节标记（用极短的胶囊充当点）
        for _ in range(JOINT_MARK_MAX):
            add_visual_capsule(
                viewer.user_scn,
                np.zeros(3),
                np.array([1e-3, 0.0, 0.0]),
                JOINT_MARK_RADIUS,
                np.array([0.2, 0.7, 1.0, 1.0], dtype=np.float32),
            )

        while viewer.is_running():
            step_start = time.time()

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

            # 画 SMPL 关节（如果存在）
            if "smpl_joints" in curr:
                joint_gt = curr["smpl_joints"]  # (T, J, 3)
                J = joint_gt.shape[1]
                # 限制不超过预创建数量
                J_use = min(J, JOINT_MARK_MAX)
                for i in range(J_use):
                    viewer.user_scn.geoms[i].pos = joint_gt[curr_idx, i].astype(np.float32)

            # 同步显示
            viewer.sync()

            # 控制节拍
            remain = mj_model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

if __name__ == "__main__":
    main()
