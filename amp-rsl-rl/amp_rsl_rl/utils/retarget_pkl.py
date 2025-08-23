# ===== NumPy 2.x <-> chumpy 兼容垫片（务必放在最顶部）=====
import numpy as _np
_aliases = [
    ('bool', bool),
    ('int', int),
    ('float', float),
    ('complex', complex),
    ('object', object),
    ('str', str),
    ('unicode', str),   # Py3 下 unicode ≡ str
    ('long', int),      # Py3 下 long ≡ int
]
for _name, _py in _aliases:
    if not hasattr(_np, _name):
        setattr(_np, _name, _py)
# ===========================================================

import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import math
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot

from easydict import EasyDict
import hydra
from omegaconf import DictConfig

from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPLH_Parser, SMPLX_Parser
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_MUJOCO_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch

from amp_rsl_rl.utils.torch_humanoid_batch import Humanoid_Batch


# --------------------------
# I/O helpers
# --------------------------
def load_amass_data(data_path):
    """加载 AMASS .npz 数据"""
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return None
    framerate = entry_data["mocap_framerate"]

    root_trans = entry_data["trans"]                         # (T, 3)
    pose_aa = np.concatenate([entry_data["poses"][:, :66],  # (T, 66)
                              np.zeros((root_trans.shape[0], 6))], axis=-1)  # pad到72
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


# --------------------------
# Shape fitting utilities
# --------------------------
def _build_robot_smpl_correspondence(humanoid_fk, cfg):
    """根据 cfg.robot.joint_matches 建立机器人与SMPL索引映射。"""
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    return robot_joint_pick_idx, smpl_joint_pick_idx


def fit_shape_and_scale(
    cfg,
    smpl_model_path="data/smpl",
    iters=2000,
    lr=0.1,
    device="cpu",
    save=True,
    save_path=None,
    visualize=False,
):
    """
    通过静态站姿把 SMPL 的 shape(beta) 与全局 scale 拟合到机器人的kinematic骨架上。
    返回: (shape_new[1,10], scale[1])
    """
    device = torch.device(device)
    humanoid_fk = Humanoid_Batch(cfg.robot)

    robot_joint_pick_idx, smpl_joint_pick_idx = _build_robot_smpl_correspondence(humanoid_fk, cfg)

    # 机器人"零姿态"的AA（只用到骨架位置）
    pose_aa_robot = np.repeat(
        np.repeat(sRot.identity().as_rotvec()[None, None, None,], humanoid_fk.num_bodies, axis=2),
        1,
        axis=1,
    )
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float().to(device)

    # SMPL 站姿（可通过 cfg.robot.smpl_pose_modifier 调整姿态以更接近机器人骨架）
    pose_aa_stand = np.zeros((1, 72)).reshape(-1, 24, 3)
    for modifiers in cfg.robot.smpl_pose_modifier:
        k = list(modifiers.keys())[0]
        v = list(modifiers.values())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(k)] = sRot.from_euler("xyz", eval(v), degrees=False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72)).float().to(device)

    # SMPL 解析器
    smpl_parser = SMPL_Parser(model_path=smpl_model_path, gender="neutral")

    # 初始 trans/beta
    trans = torch.zeros([1, 3], device=device)
    beta = torch.zeros([1, 10], device=device)

    # 对齐根
    with torch.no_grad():
        _, joints0 = smpl_parser.get_joints_verts(pose_aa_stand, beta, trans)
        offset = joints0[:, 0] - trans
        root_trans_offset = trans + offset

    fk_ret = humanoid_fk.fk_batch(pose_aa_robot[None,], root_trans_offset[None, 0:1])

    # 待优化变量：beta(shape) + global scale
    shape_new = torch.zeros([1, 10], device=device, requires_grad=True)
    scale = torch.ones([1], device=device, requires_grad=True)
    optim = torch.optim.Adam([shape_new, scale], lr=lr)

    print(f"[ShapeFit] start, iters={iters}, lr={lr}, device={device}")
    pbar = tqdm(range(iters))
    for it in pbar:
        verts, joints = smpl_parser.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]
        joints_n = (joints - joints[:, 0]) * scale + root_pos  # 仅做全局缩放

        if len(cfg.robot.extend_config) > 0:
            robot_j = fk_ret.global_translation_extend[:, :, robot_joint_pick_idx]
        else:
            robot_j = fk_ret.global_translation[:, :, robot_joint_pick_idx]

        diff = robot_j - joints_n[:, smpl_joint_pick_idx]
        loss = (diff.norm(dim=-1) ** 2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.set_description_str(f"[ShapeFit] it={it} loss={loss.item():.6f} | scale={scale.item():.4f}")

    print("[ShapeFit] done.")
    if save:
        if save_path is None:
            save_dir = f"data/{cfg.robot.humanoid_type}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, "shape_optimized_v1.pkl")
        else:
            save_dir = osp.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
        joblib.dump((shape_new.detach().cpu(), scale.detach().cpu()), save_path)
        print(f"[ShapeFit] saved to {save_path}")

    if visualize:
        try:
            import matplotlib.pyplot as plt

            j_robot = (
                fk_ret.global_translation_extend
                if len(cfg.robot.extend_config) > 0
                else fk_ret.global_translation
            )[0, :, robot_joint_pick_idx, :].detach().cpu().numpy()
            j_robot = j_robot - j_robot[:, 0:1]
            j_smpl = joints_n[:, smpl_joint_pick_idx].detach().cpu().numpy()
            j_smpl = j_smpl - j_smpl[:, 0:1]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(90, 0)
            ax.scatter(j_robot[0, :, 0], j_robot[0, :, 1], j_robot[0, :, 2], label="Robot", s=10)
            ax.scatter(j_smpl[0, :, 0], j_smpl[0, :, 1], j_smpl[0, :, 2], label="SMPL-Fit", s=10)
            ax.legend()
            plt.show()
        except Exception as e:
            print(f"[ShapeFit] visualization skipped: {e}")

    return shape_new.detach(), scale.detach()


def ensure_shape_file(
    cfg,
    smpl_model_path="data/smpl",
    iters=2000,
    lr=0.1,
    device="cpu",
    force_refit=False,
):
    """
    若 data/<humanoid_type>/shape_optimized_v1.pkl 存在则直接加载，否则执行拟合并保存。
    返回: (shape_new[1,10], scale[1])
    """
    save_dir = f"data/{cfg.robot.humanoid_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "shape_optimized_v1.pkl")
    
    if (not force_refit) and osp.exists(save_path):
        print(f"[ShapeFit] load existing {save_path}")
        shape_new, scale = joblib.load(save_path)
        return shape_new, scale
    else:
        return fit_shape_and_scale(
            cfg,
            smpl_model_path=smpl_model_path,
            iters=iters,
            lr=lr,
            device=device,
            save=True,
            save_path=save_path,
            visualize=False,
        )


# --------------------------
# Retarget: AMASS
# --------------------------
def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")

    humanoid_fk = Humanoid_Batch(cfg.robot)
    num_augment_joint = len(cfg.robot.extend_config)
    print(f"[DEBUG] robot body names: {humanoid_fk.body_names}")
    print(f"[DEBUG] robot parents: {humanoid_fk._parents}")
    print(f"[DEBUG] robot num_dof: {humanoid_fk.num_dof}")

    # 关节映射
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    # SMPL 解析器 + 形状/缩放
    smpl_model_path = cfg.get("smpl_model_path", "data/smpl")
    smpl_parser_n = SMPL_Parser(model_path=smpl_model_path, gender="neutral")
    try:
        shape_new, scale = joblib.load(f"data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl")
    except:
        print("Warning: shape_optimized_v1.pkl not found, using default shape and scale")
        shape_new = torch.zeros([1, 10])
        scale = torch.ones([1])

    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue
        
        # 重采样到30fps
        skip = int(amass_data["fps"] // 30)
        trans = torch.from_numpy(amass_data["trans"][::skip]).float()
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data["pose_aa"][::skip]).float()

        if N < 10:
            print("Motion too short")
            continue

        # 使用SMPL模型生成关节位置
        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        # 根部旋转处理 - 只去除pitch&roll，保留原始yaw变化
        gt_root_rot_quat = torch.from_numpy(
            (sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()
        ).float()
        gt_root_rot = torch.from_numpy(
            sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()
        ).float()
        
        gt_root_rot[:, 0] = 0.0  # x (pitch) - 固定为0
        gt_root_rot[:, 1] = 0.0  # y (roll) - 固定为0
        # z (yaw) - 保留原始转弯变化，不修改
        # first_frame_z = gt_root_rot[0, 2].clone()
        # gt_root_rot[:, 2] = first_frame_z  # 注释掉这行来允许转弯

        # 优化 - 拟合robot DOF到SMPL关节位置
        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))
        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=0.02)

        kernel_size = 5
        sigma = 0.75

        for iteration in range(cfg.get("fitting_iterations", 500)):
            # 只约束pitch和roll，不约束yaw
            with torch.no_grad():
                root_rot_new.data[:, 0] = 0.0  # 固定pitch
                root_rot_new.data[:, 1] = 0.0  # 固定roll
                # 不约束yaw，让它保持原始转弯动作

            # 构建完整pose_aa: root_rot + dof_rot + augment_joints
            pose_aa_h1_new = torch.cat([
                root_rot_new[None, :, None],
                humanoid_fk.dof_axis * dof_pos_new,
                torch.zeros((1, N, num_augment_joint, 3))
            ], axis=2)
            
            # 前向运动学
            fk_return = humanoid_fk.fk_batch(
                pose_aa_h1_new, root_trans_offset[None,] + root_pos_offset
            )

            # 计算损失 - 机器人关节位置与SMPL关节位置的差异
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

            loss_g = diff.norm(dim=-1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            loss = loss_g

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 再次只约束pitch和roll
            with torch.no_grad():
                root_rot_new.data[:, 0] = 0.0
                root_rot_new.data[:, 1] = 0.0
                # yaw保持优化结果，不固定

            # 关节限制和平滑
            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma
            ).transpose(2, 1)[..., None]

            if iteration % 50 == 0:
                tqdm.write(f"{data_key}-Iter: {iteration}\t{loss.item() * 1000:.3f}")

        # 最终约束和构建 - 只约束pitch和roll
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        with torch.no_grad():
            root_rot_new.data[:, 0] = 0.0  # 固定pitch
            root_rot_new.data[:, 1] = 0.0  # 固定roll
            # yaw保持自由，允许转弯

        pose_aa_h1_new = torch.cat([
            root_rot_new[None, :, None],
            humanoid_fk.dof_axis * dof_pos_new,
            torch.zeros((1, N, num_augment_joint, 3))
        ], axis=2)

        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        # 落地校正
        combined_mesh = humanoid_fk.mesh_fk(
            pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach()
        )
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()

        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        # 轻微下沉
        z_offset = -0.04
        root_trans_offset_dump[..., 2] += z_offset
        joints_dump[..., 2] += z_offset

        # 根部四元数 - 保留原始yaw变化
        root_rot_vec = root_rot_new.detach().clone().numpy()
        # 不再固定yaw，保持转弯动作
        root_rot_quat_clean = sRot.from_rotvec(root_rot_vec).as_quat()
        root_rot_quat_clean = root_rot_quat_clean / np.linalg.norm(root_rot_quat_clean, axis=1, keepdims=True)

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().numpy(),
            "root_rot": root_rot_quat_clean,
            "smpl_joints": joints_dump,
            "fps": 30,
            # 元信息
            "dof_names": list(cfg.robot.dof_names),
            "model": cfg.robot.asset.assetFileName,
        }
        all_data[data_key] = data_dump
    return all_data


# --------------------------
# Main
# --------------------------
@hydra.main(version_base=None, config_path="/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/cfg", config_name="taihu")
def main(cfg: DictConfig) -> None:
    # 先确保shape文件存在
    smpl_model_path = cfg.get("smpl_model_path", "data/smpl")
    if cfg.get("do_shape_fit", False):
        ensure_shape_file(
            cfg,
            smpl_model_path=smpl_model_path,
            iters=cfg.get("shape_fit_iters", 2000),
            lr=cfg.get("shape_fit_lr", 0.1),
            device=cfg.get("shape_fit_device", "cpu"),
            force_refit=cfg.get("shape_fit_force_refit", False),
        )

    # AMASS数据处理
    if "amass_root" not in cfg:
        raise ValueError("amass_root must be specified in the config")

    amass_root = cfg.amass_root
    print(f"Processing AMASS files from: {amass_root}")
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    print(f"Found {len(all_npzs)} npz files")
    
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}
    key_names = list(key_name_to_pkls.keys())

    print(f"Processing {len(key_names)} files")

    if not cfg.get("fit_all", False):
        # 调试只处理一个文件
        key_names = key_names[:1] if key_names else []

    # 多进程处理
    torch.set_num_threads(1)
    mp.set_sharing_strategy("file_descriptor")
    num_jobs = 30
    chunk = int(np.ceil(len(key_names) / num_jobs))
    jobs = [key_names[i:i + chunk] for i in range(0, len(key_names), chunk)]
    job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
    print(f"Created {len(job_args)} job chunks")

    if len(job_args) == 1:
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
        print(f"Processed {len(all_data)} motions")
    else:
        try:
            pool = mp.Pool(num_jobs)
            all_data_list = pool.starmap(process_motion, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)

    # 保存
    save_dir = f"data/{cfg.robot.humanoid_type}/v1/singles"
    os.makedirs(save_dir, exist_ok=True)
    for data_key, data_content in all_data.items():
        single_motion_data = {data_key: data_content}
        dumped_file = f"{save_dir}/{data_key}.pkl"
        joblib.dump(single_motion_data, dumped_file)
        print(f"Saved: {dumped_file}")

    print(f"Total {len(all_data)} motions saved to {save_dir}")

    # 验证
    humanoid_fk = Humanoid_Batch(cfg.robot)
    print(f"Robot has {humanoid_fk.num_dof} DOFs: {humanoid_fk.body_names}")


if __name__ == "__main__":
    main()