import os
import sys
import time
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np
import torch
import mujoco
import mujoco.viewer
import joblib
import hydra
from omegaconf import DictConfig

# ========== 可调显示参数 ==========
DT = 1.0 / 30.0            # 播放帧率
JOINT_MARK_MAX = 64        # 预分配的关节标记数量上限
JOINT_MARK_RADIUS = 0.03   # 关节标记半径
# =================================

# ---- 全局运行态 ----
time_step = 0.0
paused = False

file_list = []             # 所有 .pkl 文件路径（已排序）
file_idx = 0               # 当前文件索引
motion_keys = []           # 当前文件里的 key 列表
motion_idx = 0             # 当前 key 索引
curr_motion = None         # 当前 motion 数据 dict
# -------------------

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene (作为关节小标记用)."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
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

def list_motion_files(humanoid_type: str, subdir: str = "walk"):
    """收集 data/<humanoid_type>/v1/singles/<subdir> 下全部 .pkl（按文件名排序）"""
    singles_dir = osp.abspath(osp.join("data", humanoid_type, "v1", "singles", subdir))
    if not osp.isdir(singles_dir):
        raise FileNotFoundError(f"Dir not found: {singles_dir}")
    pkls = [osp.join(singles_dir, p) for p in os.listdir(singles_dir) if p.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError(f"No motion .pkl in {singles_dir}")
    pkls.sort()
    return pkls

def resolve_single_motion_file(humanoid_type: str, motion_name: str, subdir: str = "walk"):
    """根据 motion_name（可带或不带 .pkl）定位单个文件"""
    singles_dir = osp.abspath(osp.join("data", humanoid_type, "v1", "singles", subdir))
    motion_name = motion_name[:-4] if motion_name.endswith(".pkl") else motion_name
    fpath = osp.join(singles_dir, f"{motion_name}.pkl")
    if not osp.exists(fpath):
        raise FileNotFoundError(f"Motion file not found: {fpath}")
    return [fpath]

def load_motion_file(path: str):
    """载入 .pkl 并返回 (keys, data)"""
    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Bad motion data in {path}")
    keys = list(data.keys())
    return keys, data

def set_current_file(new_file_idx: int):
    """切换当前文件并重置 key 与时间"""
    global file_idx, motion_keys, curr_motion, motion_idx, time_step
    file_idx = new_file_idx % len(file_list)
    fpath = file_list[file_idx]
    motion_keys, curr_motion = load_motion_file(fpath)
    motion_keys.sort()
    motion_idx = 0
    time_step = 0.0
    print(f"[viewer] File [{file_idx+1}/{len(file_list)}]: {osp.basename(fpath)}")
    print(f"[viewer] Keys: {motion_keys}")
    print(f"[viewer] Current key: {motion_keys[motion_idx]}")

def set_current_key(new_key_idx: int):
    """切换当前 key 并重置时间"""
    global motion_idx, time_step
    if not motion_keys:
        return
    motion_idx = new_key_idx % len(motion_keys)
    time_step = 0.0
    print(f"[viewer] Switch key -> {motion_keys[motion_idx]}")

# --- 键盘回调 ---
def key_call_back(keycode):
    global time_step, paused
    try:
        ch = chr(keycode)
    except Exception:
        ch = ""

    if ch == "R":
        time_step = 0.0
        print("[viewer] Reset to start")
    elif ch == " ":
        paused = not paused
        print(f"[viewer] Paused = {paused}")
    elif ch == "T":
        set_current_file(file_idx + 1)  # 下一文件
    elif ch == "K":
        set_current_key(motion_idx + 1) # 下一 key
    else:
        pass

@hydra.main(version_base=None,
            config_path="/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/cfg",
            config_name="taihu")
def main(cfg: DictConfig) -> None:
    # —— 关键修正：声明用到的全局运行态（避免 UnboundLocalError）——
    global time_step, paused, file_list, file_idx, motion_keys, motion_idx, curr_motion
    # ---------------------------------------------------------------

    device = torch.device("cpu")  # 目前未使用

    # 机器人 XML
    humanoid_xml = cfg.robot.asset.assetFileName
    if not osp.exists(humanoid_xml):
        raise FileNotFoundError(f"Robot XML not found: {humanoid_xml}")

    # 收集要可视化的 motion 文件
    motion_name = cfg.get("motion_name", None)
    if motion_name:
        file_list = resolve_single_motion_file(cfg.robot.humanoid_type, motion_name, subdir="walk")
        print(f"[viewer] Use specified motion_name: {osp.basename(file_list[0])}")
    else:
        file_list = list_motion_files(cfg.robot.humanoid_type, subdir="walk")
        print(f"[viewer] Auto collected {len(file_list)} files from directory.")

    # 初始化为第一个文件
    set_current_file(0)

    # MuJoCo 初始化
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = DT

    # Viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        # 预创建关节标记
        for _ in range(JOINT_MARK_MAX):
            add_visual_capsule(
                viewer.user_scn,
                np.zeros(3),
                np.array([1e-3, 0.0, 0.0]),  # 极短胶囊
                JOINT_MARK_RADIUS,
                np.array([0.2, 0.7, 1.0, 1.0], dtype=np.float32),
            )

        while viewer.is_running():
            step_start = time.time()

            # 当前文件与 key
            fpath = file_list[file_idx]
            curr_key = motion_keys[motion_idx]
            curr = curr_motion[curr_key]

            # === 取当前帧 ===
            dof_arr = curr["dof"]
            T = dof_arr.shape[0]
            if T <= 0:
                viewer.sync()
                continue
            curr_idx = int(time_step / DT) % T

            root_trans = curr["root_trans_offset"][curr_idx]           # (3,)
            root_quat_xyzw = curr["root_rot"][curr_idx]                # (4,) xyzw
            dof = dof_arr[curr_idx]                                    # (num_dof,) 或 (num_dof,1)

            # === 写入 qpos ===
            mj_data.qpos[:3] = np.asarray(root_trans, dtype=np.float64)
            # MuJoCo 使用 wxyz
            root_wxyz = np.array(
                [root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]],
                dtype=np.float64
            )
            mj_data.qpos[3:7] = root_wxyz

            dof_flat = np.asarray(dof, dtype=np.float64).reshape(-1)
            qpos_needed = 7 + dof_flat.shape[0]
            if qpos_needed > mj_data.qpos.shape[0]:
                # 模型/数据 DOF 不匹配时安全截断
                dof_flat = dof_flat[: max(0, mj_data.qpos.shape[0] - 7)]
            mj_data.qpos[7:7 + dof_flat.shape[0]] = dof_flat

            mujoco.mj_forward(mj_model, mj_data)

            # 时间推进
            if not paused:
                time_step += DT

            # === 画 SMPL 关节（可选）===
            if "smpl_joints" in curr:
                joint_gt = curr["smpl_joints"]  # (T, J, 3)
                if joint_gt.ndim == 3 and joint_gt.shape[0] == T:
                    J = joint_gt.shape[1]
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
