import joblib
import numpy as np

p = "/home/dy/dy/code/unitree_ti/data/ti512/v1/singles/Male2Walking_c3d/B9 -  Walk turn left 90_poses.pkl"
d = joblib.load(p)

print("== keys ==", list(d.keys()))
k = list(d.keys())[0]
m = d[k]

def sh(x):
    return getattr(x, "shape", None)

# 检查并打印字段
required_fields = ['root_trans_offset', 'root_rot', 'pose_aa', 'dof', 'fps']
missing_fields = []

for field in required_fields:
    if field not in m:
        missing_fields.append(field)

if missing_fields:
    print(f"[WARNING] Missing fields: {', '.join(missing_fields)}")

# 如果字段存在，则打印它们的 shape
if 'root_trans_offset' in m:
    print("root_trans_offset:", sh(m["root_trans_offset"]))
if 'root_rot' in m:
    print("root_rot        :", sh(m["root_rot"]))
if 'pose_aa' in m:
    print("pose_aa         :", sh(m["pose_aa"]))
if 'dof' in m:
    print("dof             :", sh(m["dof"]))
if 'fps' in m:
    print("fps             :", m["fps"])

# 看第0帧的大小
if m.get("dof") is not None:
    print("dof[0] len:", m["dof"][0].reshape(-1).shape[0])
if m.get("pose_aa") is not None:
    print("pose_aa[0] flattened len:", m["pose_aa"][0].reshape(-1).shape[0])

# 如果有你之前建议的元信息
print("dof_names       :", m.get("dof_names"))
print("model           :", m.get("model"))
