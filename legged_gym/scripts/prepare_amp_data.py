# scripts/prepare_amp_data.py
import numpy as np
import torch
from pathlib import Path
from amp_rsl_rl.utils import AMPLoader
from scipy.spatial.transform import Rotation as sRot

def convert_motion_to_amp_format(motion_file, joint_names, output_path):
    """将运动数据转换为AMP格式"""
    
    # 加载原始运动数据（假设是pkl格式）
    import joblib
    motion_data = joblib.load(motion_file)
    
    # 提取所需数据
    if isinstance(motion_data, dict):
        key = list(motion_data.keys())[0]
        data = motion_data[key]
        
        # 构建AMP格式数据
        amp_data = {
            "joints_list": joint_names,  # 关节名称列表
            "joint_positions": [],  # 每帧的关节位置
            "root_position": [],  # 根部位置
            "root_quaternion": [],  # 根部四元数(xyzw格式)
            "fps": data.get("fps", 30.0)
        }
        
        # 转换数据
        for i in range(len(data["dof"])):
            # 关节位置
            joint_pos = data["dof"][i].flatten()
            amp_data["joint_positions"].append(joint_pos)
            
            # 根部位置
            root_pos = data["root_trans_offset"][i]
            amp_data["root_position"].append(root_pos)
            
            # 根部旋转（确保是xyzw格式）
            root_quat = data["root_rot"][i]  # 假设已经是xyzw
            amp_data["root_quaternion"].append(root_quat)
        
        # 保存为npy文件
        np.save(output_path, amp_data)
        print(f"Saved AMP data to {output_path}")
        
    return amp_data

# 使用示例
if __name__ == "__main__":
    # TiV2的关节名称
    joint_names = [
        'L_HIP_Y', 'L_HIP_R', 'L_HIP_P', 
        'L_KNEE_P', 'L_ANKLE_P', 'L_ANKLE_R',
        'R_HIP_Y', 'R_HIP_R', 'R_HIP_P',
        'R_KNEE_P', 'R_ANKLE_P', 'R_ANKLE_R'
    ]
    
    # 转换数据
    motion_file = "data/taihu/v1/singles/walk.pkl"
    output_path = "amp_datasets/tiv2/walk.npy"
    
    convert_motion_to_amp_format(motion_file, joint_names, output_path)