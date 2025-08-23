import joblib, numpy as np

p = "data/ti512/v1/singles/0-Male2Walking_c3d_B15 -  Walk turn around_poses.pkl"
d = joblib.load(p)

print("== keys ==", list(d.keys()))
k = list(d.keys())[0]
m = d[k]

def sh(x):
    return getattr(x, "shape", None)

print("root_trans_offset:", sh(m.get("root_trans_offset")))
print("root_rot        :", sh(m.get("root_rot")))
print("pose_aa         :", sh(m.get("pose_aa")))
print("dof             :", sh(m.get("dof")))
print("fps             :", m.get("fps"))

# 看第0帧的大小
if m.get("dof") is not None:
    print("dof[0] len:", m["dof"][0].reshape(-1).shape[0])
if m.get("pose_aa") is not None:
    print("pose_aa[0] flattened len:", m["pose_aa"][0].reshape(-1).shape[0])

# 如果有你之前建议的元信息
print("dof_names       :", m.get("dof_names"))
print("model           :", m.get("model"))
