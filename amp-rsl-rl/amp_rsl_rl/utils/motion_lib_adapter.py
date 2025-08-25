from __future__ import annotations
from typing import Optional, List, Tuple, Any
import torch
from isaacgym.torch_utils import quat_rotate
from amp_rsl_rl.utils.amp_helpers import (
    build_amp_obs_per_step, calc_heading_quat_inv_xyzw
)


class MotionLibAMPAdapter:
    """
    将 MotionLibX 封装为 AMP 专家数据源（KISS 版本）：
      - feed_forward_generator -> (obs, next_obs)
      - get_state_for_reset   -> (root_rot, dof_pos, dof_vel, v_lin_local, v_ang_local)

    关键点：
      * 优先使用数据中的 rg_pos（例如左右脚），若传入的 key_body_ids 越界则自动回退为 rg_pos。
      * 对 key bodies 做长度自适应（裁剪/零填充）以满足 expect_key_bodies。
      * 简要打印 DOF/刚体名帮助对齐 13 ↔ 12。
    """

    def __init__(
        self,
        motion_lib,
        dt: float,
        history_steps: int = 2,
        history_stride: int = 1,
        expect_dof_obs_dim: int = 12,
        expect_key_bodies: int = 2,
        key_body_ids: Optional[List[int]] = None,     # 必须是 MotionLib 自己的 rg_pos 索引
        use_root_h: bool = False,
        device: torch.device = torch.device("cpu"),
        env_dof_names: Optional[List[str]] = None,    # 仅用于打印对齐（环境侧12DOF名称）
        key_body_names: Optional[List[str]] = None,   # 若提供，将解析为 MotionLib 索引
        names_print_limit: int = 32,
    ):
        self.motion_lib = motion_lib
        self.dt = float(dt)
        self.K = max(2, int(history_steps))
        self.S = max(1, int(history_stride))
        self.expect_dof_obs_dim = int(expect_dof_obs_dim)
        self.expect_key_bodies = int(expect_key_bodies)
        self.key_body_ids = key_body_ids
        self.use_root_h = bool(use_root_h)
        self.device = device
        self._names_print_limit = int(names_print_limit)

        # 若给了名字，尝试解析为 rg_pos 索引
        # import ipdb;ipdb.set_trace()
        if key_body_names:
            resolved = self._resolve_key_body_ids_from_names(key_body_names)
            if resolved:
                self.key_body_ids = resolved
            else:
                print("[WARN] key_body_names 解析失败，将不使用手动 ids（改为自动使用 rg_pos）。")
                self.key_body_ids = None

        # 简要打印名称帮助对齐
        self._debug_print_names(env_dof_names=env_dof_names)

        # 预检查 ids 是否越界；越界则回退为自动 rg_pos
        try:
            mids = self.motion_lib.sample_motions(1)
            t0 = self.motion_lib.sample_time(mids, truncate_time=0.0)
            st0 = self.motion_lib.get_motion_state(mids, t0)
            if "rg_pos" in st0 and st0["rg_pos"] is not None:
                kb_total = int(st0["rg_pos"].shape[1])
                if self.key_body_ids is not None:
                    bad = [i for i in self.key_body_ids if (i < 0 or i >= kb_total)]
                    if bad:
                        print(f"[WARN] 提供的 key_body_ids {self.key_body_ids} 超出 rg_pos 长度 {kb_total}，将回退使用 rg_pos。")
                        self.key_body_ids = None
        except Exception as e:
            print(f"[WARN] 预检查 rg_pos 失败：{e}（将继续）")

        # 探针一次，确定每步维度与总维度
        _probe = self._sample_amp_window(batch_size=1)
        self.num_amp_obs_per_step = int(_probe["obs"].shape[-1] // self.K)
        self.sample_item_dim = int(_probe["obs"].shape[-1])  # = K*D

    # ---------------- 对外 API ----------------

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            out = self._sample_amp_window(batch_size=mini_batch_size)
            yield out["obs"], out["next_obs"]

    @torch.no_grad()
    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        n = int(number_of_samples)
        if n <= 0:
            z = torch.empty
            dev = self.device
            return (
                z((0, 4), device=dev),
                z((0, 0), device=dev),
                z((0, 0), device=dev),
                z((0, 3), device=dev),
                z((0, 3), device=dev),
            )

        mids = self.motion_lib.sample_motions(n)
        t0 = self.motion_lib.sample_time(mids, truncate_time=0.0)
        st = self.motion_lib.get_motion_state(mids, t0)

        root_rot = st["root_rot"]               # [N,4] xyzw
        dof_pos = st["dof_pos"]                 # [N,J]
        dof_vel = st["dof_vel"].reshape(n, -1)  # [N,J]
        v_lin_w = st["root_vel"]                # [N,3]
        v_ang_w = st["root_ang_vel"]            # [N,3]

        heading_inv = calc_heading_quat_inv_xyzw(root_rot)
        v_lin_local = quat_rotate(heading_inv, v_lin_w)
        v_ang_local = quat_rotate(heading_inv, v_ang_w)
        return (root_rot, dof_pos, dof_vel, v_lin_local, v_ang_local)

    # ---------------- 内部实现 ----------------

    @torch.no_grad()
    def _sample_amp_window(self, batch_size: int):
        B = int(batch_size)
        K, S, dt = self.K, self.S, self.dt

        mids = self.motion_lib.sample_motions(B)
        truncate_time = (K - 1) * S * dt + 1e-8
        t0 = self.motion_lib.sample_time(mids, truncate_time=truncate_time) + truncate_time

        steps = torch.arange(0, K, device=self.device, dtype=torch.float32) * (S * dt)
        times = t0.unsqueeze(1) - steps.unsqueeze(0)   # [B,K]
        times_next = times + dt

        obs = self._build_flattened_obs(mids, times)
        next_obs = self._build_flattened_obs(mids, times_next)
        return {"obs": obs, "next_obs": next_obs}

    def _build_flattened_obs(self, mids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, K = times.shape
        per = []
        for j in range(K):
            st = self.motion_lib.get_motion_state(mids, times[:, j])

            # ---- 安全获得 key bodies ----
            key_world = None
            if "rg_pos" in st and st["rg_pos"] is not None:
                kb_total = int(st["rg_pos"].shape[1])
                if self.key_body_ids is None:
                    # 未指定 ids：直接使用 rg_pos（你的数据里就是左右脚两个关键点）
                    key_world = st["rg_pos"]
                else:
                    # 指定了 ids：越界则回退为 rg_pos
                    ids = torch.as_tensor(self.key_body_ids, device=st["rg_pos"].device, dtype=torch.long)
                    valid = (ids >= 0) & (ids < kb_total)
                    if not bool(valid.all()):
                        bad = ids[~valid].detach().cpu().tolist()
                        print(f"[WARN] key_body_ids 越界: {bad}（rg_pos K={kb_total}），回退使用 rg_pos。")
                        key_world = st["rg_pos"]
                    else:
                        key_world = st["rg_pos"].index_select(1, ids)

                # 适配至 expect_key_bodies（裁剪/零填充）
                key_world = self._fit_key_bodies(key_world)
            else:
                key_world = None  # build_amp_obs_per_step 内部会零填充

            step = build_amp_obs_per_step(
                base_pos_world=st["root_pos"],
                base_quat_xyzw=st["root_rot"],
                base_lin_vel_world=st["root_vel"],
                base_ang_vel_world=st["root_ang_vel"],
                dof_pos=st["dof_pos"],
                dof_vel=st["dof_vel"].reshape(B, -1),
                key_body_pos_world=key_world,
                use_root_h=self.use_root_h,
                expect_dof_obs_dim=self.expect_dof_obs_dim,
                expect_key_bodies=self.expect_key_bodies,
            )
            per.append(step)

        return torch.stack(per, dim=1).reshape(B, -1)

    # ---------------- 辅助 ----------------

    def _fit_key_bodies(self, key_world: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """将 [B,Kb,3] 的 key bodies 调整为 expect_key_bodies（裁剪或零填充）。"""
        if key_world is None:
            return None
        B, Kb, D = key_world.shape
        Ek = int(self.expect_key_bodies)
        if Kb == Ek:
            return key_world
        if Kb > Ek:
            return key_world[:, :Ek, :]
        # Kb < Ek -> 右侧补零
        pad = torch.zeros((B, Ek - Kb, D), dtype=key_world.dtype, device=key_world.device)
        return torch.cat([key_world, pad], dim=1)

    def _debug_print_names(self, env_dof_names: Optional[List[str]] = None):
        """简要打印 DOF/刚体名帮助对齐（不影响训练）。"""
        # DOF/Joint names（MotionLib）
        ml_dof_names = self._get_any_attr_list(self.motion_lib, ["dof_names", "joint_names", "_dof_names"], fallback_from_parsers=True)
        if ml_dof_names:
            print(f"\n[INFO] MotionLib DOF/Joint names (N={len(ml_dof_names)}), 前{min(len(ml_dof_names), self._names_print_limit)}个：")
            self._print_name_list([str(x) for x in ml_dof_names])

        # 环境期望的 12 DOF 名（若提供）
        if env_dof_names:
            print(f"\n[INFO] Expected 12-DOF from env (N={len(env_dof_names)}):")
            self._print_name_list([str(x) for x in env_dof_names])

        # 刚体名（用于核对 key body 名称/索引）
        body_names = self._get_any_attr_list(getattr(self.motion_lib, "mesh_parsers", None), ["body_names", "model_names", "rigid_body_names"])
        if body_names:
            print(f"\n[INFO] MotionLib rigid bodies (N={len(body_names)}), 前{min(len(body_names), self._names_print_limit)}个：")
            self._print_name_list([str(x) for x in body_names])
        print("")

    def _print_name_list(self, names: List[str]):
        k = min(len(names), self._names_print_limit)
        for i in range(k):
            print(f"  [{i:02d}] {names[i]}")
        if len(names) > k:
            print(f"  ... 以及 {len(names) - k} 个更多")

    def _get_any_attr_list(self, obj: Any, keys: List[str], fallback_from_parsers: bool = False) -> Optional[List[str]]:
        if obj is None:
            return None
        for k in keys:
            if hasattr(obj, k):
                v = getattr(obj, k)
                if isinstance(v, (list, tuple)):
                    return list(v)
        if fallback_from_parsers and hasattr(obj, "mesh_parsers"):
            pars = getattr(obj, "mesh_parsers")
            for k in keys:
                if hasattr(pars, k):
                    v = getattr(pars, k)
                    if isinstance(v, (list, tuple)):
                        return list(v)
        return None

    def _resolve_key_body_ids_from_names(self, wanted_names: List[str]) -> Optional[List[int]]:
        """将刚体名字解析为 rg_pos 索引（不区分大小写，子串可匹配）。"""
        parsers = getattr(self.motion_lib, "mesh_parsers", None)
        if parsers is None:
            return None
        names = None
        for key in ["body_names", "model_names", "rigid_body_names"]:
            if hasattr(parsers, key):
                v = getattr(parsers, key)
                if isinstance(v, (list, tuple)):
                    names = [str(x) for x in v]
                    break
        if not names:
            return None

        name_lc = [s.lower() for s in names]
        out: List[int] = []
        for wn in wanted_names:
            w = wn.lower()
            # 精确优先
            try:
                out.append(name_lc.index(w))
                continue
            except ValueError:
                pass
            # 子串匹配
            cands = [i for i, s in enumerate(name_lc) if w in s]
            if cands:
                out.append(cands[0])
            else:
                print(f"[WARN] 刚体名中找不到: '{wn}'")
        return out if out else None
