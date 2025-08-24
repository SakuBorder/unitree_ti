# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
import torch
from isaacgym.torch_utils import *  # quat_mul, quat_rotate, quat_rotate_inverse, etc.
from isaacgym import gymtorch
import torch.nn.functional as F
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils import torch_utils  # calc_heading_quat_inv, quat_to_tan_norm


class TiV2AMPRobot(TiV2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # ===== AMP 历史观测配置（K 步）=====
        self._num_amp_obs_steps = getattr(cfg.env, 'num_amp_obs_steps', 2)  # 历史步数 K

        # —— 选择关键刚体（用于关键点位置）——
        self._key_body_ids = self._select_key_body_ids()  # torch.long [K]

        # —— 依 NVIDIA HumanoidAMP 对齐的单步维度：13 + dof_obs + dof_vel + 3*num_key_bodies
        self._num_amp_obs_per_step = self._get_amp_obs_per_step_dim()  # 单步维度 D
        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step  # 总维度 K*D

        # 历史缓冲：[E, K, D]，其中索引 0 始终为“当前帧”
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device, dtype=torch.float32
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]   # [E, D]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]  # [E, K-1, D]

        self.amp_data = None  # 由 runner 注入

        # ------- 阶段A：命令冻结（episode 内恒定） -------
        self._phaseA_freeze_cmd = True
        # 从 cfg 读取恒定命令，默认 vx=0.8 m/s, vy=0, yaw=0
        self._phaseA_cmd_const = torch.tensor([
            getattr(cfg.commands, "phaseA_vx", 0.8),
            getattr(cfg.commands, "phaseA_vy", 0.0),
            getattr(cfg.commands, "phaseA_yaw", 0.0)
        ], dtype=torch.float32, device=self.device)  # (3,)

        # 确保 phase 属性存在
        if not hasattr(self, 'phase'):
            print("[TiV2AMPRobot] Adding missing 'phase' attribute")
            self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 调试信息
        print(f"[TiV2AMPRobot] Actor obs dim: {self.num_obs}")
        print(f"[TiV2AMPRobot] Critic obs dim: {self.num_privileged_obs}")
        print(f"[TiV2AMPRobot] AMP obs dim: {self.num_amp_obs} "
              f"({self._num_amp_obs_steps} steps × {self._num_amp_obs_per_step} per step, "
              f"key_bodies={int(self._key_body_ids.numel())})")

    # ---------------- Hook ----------------

    def set_amp_data(self, amp_data):
        """由 Runner 注入 AMPLoader 句柄"""
        self.amp_data = amp_data

    # ---------------- 阶段A：命令冻结辅助 ----------------

    def _apply_phaseA_frozen_commands(self, env_ids=None):
        """阶段A：把 self.commands 维持为恒定值，并阻止 resample 逻辑"""
        if not self._phaseA_freeze_cmd:
            return
        if not hasattr(self, "commands"):
            return  # 父类未建立命令张量时跳过

        if env_ids is None:
            self.commands[:, 0:3] = self._phaseA_cmd_const.view(1, 3)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids = env_ids.to(self.device, dtype=torch.long)
            self.commands[env_ids, 0:3] = self._phaseA_cmd_const.view(1, 3)

        # 尽力阻止父类时间驱动的重采样
        if hasattr(self, "commands_time_left"):
            self.commands_time_left[:] = 1e9
        if hasattr(self, "command_resample_time"):
            self.command_resample_time[:] = 1e9

    # ---------------- AMP 观测构造（与 NVIDIA HumanoidAMP 对齐） ----------------

    def _select_key_body_ids(self):
        """选择用于 AMP 的关键刚体 id（相对根、再旋到 heading）。至少包含左右脚。"""
        if hasattr(self, "feet_indices") and self.feet_indices is not None and len(self.feet_indices) >= 2:
            key_ids = list(self.feet_indices[:2].tolist())
        else:
            last = max(1, self.num_bodies - 1)
            key_ids = [max(1, last - 1), last]
        return torch.as_tensor(key_ids, device=self.device, dtype=torch.long)

    def _get_amp_obs_per_step_dim(self):
        """
        单步 AMP 观测维度（对齐 NVIDIA HumanoidAMP build_amp_observations）：
          = 1(root_h) + 6(root_rot_tannorm) + 3(v_lin_local) + 3(v_ang_local)
            + num_dof(dof_obs) + num_dof(dof_vel) + 3 * num_key_bodies(key_body_pos_local)
          = 13 + 2*num_dof + 3*K
        """
        num_key_bodies = int(self._key_body_ids.numel()) if hasattr(self, "_key_body_ids") else 2
        return 13 + 2 * self.num_dof + 3 * num_key_bodies

    def _dof_to_obs_identity(self, dof_pos: torch.Tensor) -> torch.Tensor:
        return dof_pos

    def _compute_amp_observations_single_step(self) -> torch.Tensor:
        """
        对齐 NVIDIA: root_h, root_rot(tan-norm, heading 对齐), 局部 v/ω, dof_obs, dof_vel, key_body_pos_local
        - 使用“前向向量水平面投影”方式计算 heading 的逆，更抗 roll/pitch 污染
        - 加入一次性 yaw 调试打印，验证 heading 去除是否正确
        """
        # 根状态（世界）
        root_pos = self.root_states[:, 0:3]       # (E,3)
        root_rot = self.root_states[:, 3:7]       # (E,4) xyzw
        root_vel = self.root_states[:, 7:10]      # (E,3)
        root_ang = self.root_states[:, 10:13]     # (E,3)

        # 高度
        root_h = root_pos[:, 2:3]                 # (E,1)

        # ---------- 稳健的 heading 逆（仅绕 z 轴） ----------
        # 从四元数取出分量 (x, y, z, w)
        x, y, z, w = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        # 机体 x 轴世界向量：R(q) * (1,0,0) = (1-2y^2-2z^2, 2xy+2wz, 2xz-2wy)
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        # 航向角（水平面投影）
        yaw = torch.atan2(fy, fx)
        half = -0.5 * yaw  # 取“逆”(-yaw)
        cy = torch.cos(half)
        sy = torch.sin(half)
        # 只绕 z 轴的四元数 (x,y,z,w) = (0,0,sy,cy)
        heading_rot = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)  # (E,4)

        # 根姿态在 heading 坐标系下，再映射到 tan-norm（6维）
        root_rot_local = quat_mul(heading_rot, root_rot)                    # (E,4)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_local)         # (E,6)

        # ---------- 一次性 yaw 调试：local yaw 应≈0 ----------
        with torch.no_grad():
            wl, xl, yl, zl = root_rot_local[:, 3], root_rot_local[:, 0], root_rot_local[:, 1], root_rot_local[:, 2]
            yaw_local = torch.atan2(2.0 * (wl * zl + xl * yl), 1.0 - 2.0 * (yl * yl + zl * zl))
            yaw_deg = torch.rad2deg(yaw_local).abs().mean().item()
            if not hasattr(self, "_dbg_once"):
                print(f"[AMP heading check] |yaw_local| mean(deg) ≈ {yaw_deg:.3f} (应当接近 0)")
                self._dbg_once = True

        # 局部线/角速度（heading 旋转）
        local_root_vel     = quat_rotate(heading_rot, root_vel)             # (E,3)
        local_root_ang_vel = quat_rotate(heading_rot, root_ang)             # (E,3)

        # dof 到 obs（若无压缩，恒等）
        dof_obs = self._dof_to_obs_identity(self.dof_pos)                   # (E,num_dof)
        dof_vel = self.dof_vel                                              # (E,num_dof)

        # 关键刚体相对根 + 旋到 heading
        key_pos_world = self.rigid_body_states_view[:, self._key_body_ids, 0:3]  # (E,K,3)
        rel_key = key_pos_world - root_pos.unsqueeze(1)                           # (E,K,3)

        # 展平成 (E*K,3) 以复用 quat_rotate；heading 复制到 (E*K,4)
        flat_rel = rel_key.reshape(-1, 3)
        heading_rep = heading_rot.unsqueeze(1).expand(-1, rel_key.shape[1], -1).reshape(-1, 4)
        local_key = quat_rotate(heading_rep, flat_rel).reshape(rel_key.shape)     # (E,K,3)
        flat_local_key = local_key.reshape(local_key.shape[0], -1)                # (E, K*3)

        # 拼接（顺序与参考一致）
        amp_obs = torch.cat(
            (root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key),
            dim=-1
        )  # (E, D)
        return amp_obs


    def _update_hist_amp_obs(self, env_ids=None):
        """将历史右移一格，使 index 0 始终是“最新帧”"""
        if self._num_amp_obs_steps <= 1:
            return

        if env_ids is None:
            src = self._amp_obs_buf[:, :-1].clone()
            self._amp_obs_buf[:, 1:] = src
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids = env_ids.to(self.device, dtype=torch.long)

            src = self._amp_obs_buf[env_ids, :-1].clone()
            self._amp_obs_buf[env_ids, 1:] = src

    def _compute_amp_observations(self, env_ids=None):
        """写入当前帧到 index=0"""
        if env_ids is None:
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_single_step()
        else:
            curr = self._compute_amp_observations_single_step()
            self._curr_amp_obs_buf[env_ids] = curr[env_ids]

    def compute_amp_observations(self):
        """对外接口：返回 [E, K*D]"""
        return self._amp_obs_buf.reshape(self.num_envs, self.num_amp_obs)

    # ---------------- Env 生命周期 ----------------

    def post_physics_step(self):
        """先执行父类物理步，再更新 AMP 历史与观测，并写入 extras。"""
        ret = super().post_physics_step()

        # 阶段A：每步都把命令钉死为常量（防父类在 step 中重采样）
        self._apply_phaseA_frozen_commands()

        # 1) 先右移历史
        self._update_hist_amp_obs()
        # 2) 写入当前帧到 index=0
        self._compute_amp_observations()

        # 3) 扁平化并写入 extras
        amp_obs_flat = self._amp_obs_buf.reshape(-1, self.num_amp_obs)
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        self.extras["amp_obs"] = amp_obs_flat
        self.extras.setdefault("observations", {})
        self.extras["observations"]["amp"] = amp_obs_flat

        return ret

    def get_observations(self):
        """兼容 Runner：返回 (actor_obs, extras)"""
        try:
            super().compute_observations()
        except AttributeError as e:
            print(f"[Warning] Parent compute_observations failed: {e}")
            print("[Warning] Using observation buffers directly")

        actor_obs = self.obs_buf
        critic_obs = self.privileged_obs_buf if self.privileged_obs_buf is not None else self.obs_buf
        amp_obs = self.compute_amp_observations()  # (E, K*D)

        if not hasattr(self, '_debug_printed'):
            print(f"[Debug] Actor obs shape:  {tuple(actor_obs.shape)}")
            print(f"[Debug] Critic obs shape: {tuple(critic_obs.shape)}")
            print(f"[Debug] AMP obs shape:    {tuple(amp_obs.shape)}")
            self._debug_printed = True

        # 对齐 critic 维度
        if critic_obs.shape[-1] != self.num_privileged_obs:
            print(f"[Warning] Critic obs dim mismatch: got {critic_obs.shape[-1]}, expected {self.num_privileged_obs}")
            if critic_obs.shape[-1] > self.num_privileged_obs:
                critic_obs = critic_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - critic_obs.shape[-1]
                critic_obs = torch.cat(
                    [critic_obs, torch.zeros(critic_obs.shape[0], padding, device=critic_obs.device)],
                    dim=-1
                )

        extras = {
            "observations": {
                "amp": amp_obs,
                "critic": critic_obs
            }
        }
        return actor_obs, extras

    def step(self, actions):
        """保持 7 元组返回"""
        ret = super().step(actions)
        if not isinstance(ret, tuple):
            raise RuntimeError(f"[TiV2AMPRobot.step] Unexpected return type from super().step: {type(ret)}")

        if len(ret) == 7:
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs = ret
        elif len(ret) == 4:
            obs, rewards, dones, infos = ret
            privileged_obs = self.get_privileged_observations()
            termination_ids = None
            termination_obs = None
            print("[TiV2AMPRobot.step][Warning] Parent step returned 4-tuple; "
                  "filled privileged_obs via get_privileged_observations(). Please fix parent to return 7-tuple.")
        else:
            raise RuntimeError(f"[TiV2AMPRobot.step] Unsupported super().step() signature with len={len(ret)}")

        # 对齐 privileged_obs 维度
        if privileged_obs is not None and privileged_obs.shape[-1] != self.num_privileged_obs:
            if privileged_obs.shape[-1] > self.num_privileged_obs:
                privileged_obs = privileged_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - privileged_obs.shape[-1]
                privileged_obs = torch.cat(
                    [privileged_obs, torch.zeros(privileged_obs.shape[0], padding, device=privileged_obs.device)],
                    dim=-1
                )

        # 调试：把 critic 放进 infos
        if isinstance(infos, dict):
            infos.setdefault("observations", {})
            infos["observations"]["critic"] = privileged_obs if privileged_obs is not None else obs

        return obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs

    def get_privileged_observations(self):
        """确保返回的 privileged 维度正确"""
        try:
            priv_obs = super().get_privileged_observations()
        except AttributeError as e:
            print(f"[Warning] Parent get_privileged_observations failed: {e}")
            priv_obs = getattr(self, 'privileged_obs_buf', None)

        if priv_obs is not None and priv_obs.shape[-1] != self.num_privileged_obs:
            if priv_obs.shape[-1] > self.num_privileged_obs:
                priv_obs = priv_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - priv_obs.shape[-1]
                priv_obs = torch.cat(
                    [priv_obs, torch.zeros(priv_obs.shape[0], padding, device=priv_obs.device)],
                    dim=-1
                )
        return priv_obs

    def compute_observations(self):
        """避免父类 phase 缺失导致异常"""
        try:
            super().compute_observations()
        except AttributeError as e:
            if "'phase'" in str(e):
                print(f"[Warning] Phase attribute missing in parent compute_observations: {e}")
                if not hasattr(self, 'phase'):
                    self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                try:
                    super().compute_observations()
                except Exception as e2:
                    print(f"[Error] Still failed after adding phase: {e2}")
                    self._manual_compute_observations()
            else:
                raise e

    # ---------------- 其它与 TiV2 保持一致的回调 ----------------

    def _post_physics_step_callback(self):
        """来自 TiV2Robot 的回调逻辑（拷贝重写）"""
        self.update_feet_state()
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_rot = self.feet_state[:, :, 3:7]
        self.feet_rpy[:, 0] = get_euler_xyz_in_tensor(self.feet_rot[:, 0])
        self.feet_rpy[:, 1] = get_euler_xyz_in_tensor(self.feet_rot[:, 1])
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

    # ---------------- Reset & 历史初始化 ----------------

    def _init_amp_obs_for_reset(self, env_ids):
        """把当前帧复制到历史，形成 [当前|当前|...|当前]"""
        if env_ids is None or len(env_ids) == 0:
            return
        self._compute_amp_observations(env_ids)

        if self._num_amp_obs_steps > 1:
            curr = self._curr_amp_obs_buf[env_ids].unsqueeze(1)  # [N,1,D]
            self._hist_amp_obs_buf[env_ids] = curr.expand(-1, self._num_amp_obs_steps - 1, -1)

    def reset_idx(self, env_ids):
        """覆盖父类 reset：先父类复位，再可选专家热启动，最后初始化历史，并锁命令"""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids)
        env_ids = env_ids.to(self.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return

        # 1) 父类复位
        super().reset_idx(env_ids)

        # 2) 阶段A：锁定这些环境的命令为恒定值
        self._apply_phaseA_frozen_commands(env_ids)

        # 3) 专家状态覆盖（如有）
        if getattr(self, "amp_data", None) is not None:
            n = env_ids.numel()
            try:
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(n)

                todev = lambda t: t.to(device=self.device, dtype=torch.float32)
                quat_xyzw  = F.normalize(todev(quat_xyzw), dim=-1)
                qpos       = todev(qpos)
                qvel       = todev(qvel)
                vlin_local = todev(vlin_local)
                vang_local = todev(vang_local)

                # DOF 对齐
                if qpos.shape[1] != self.num_dof:
                    if qpos.shape[1] > self.num_dof:
                        qpos = qpos[:, :self.num_dof]
                        qvel = qvel[:, :self.num_dof]
                    else:
                        pad = self.num_dof - qpos.shape[1]
                        qpos = F.pad(qpos, (0, pad))
                        qvel = F.pad(qvel, (0, pad))

                # 局部 -> 世界速度
                vlin_world = quat_rotate(quat_xyzw, vlin_local)
                vang_world = quat_rotate(quat_xyzw, vang_local)

                ids = env_ids.long()
                self.root_states[ids, 3:7]   = quat_xyzw
                self.root_states[ids, 7:10]  = vlin_world
                self.root_states[ids, 10:13] = vang_world

                if hasattr(self, "base_quat"):
                    self.base_quat[ids] = quat_xyzw
                if hasattr(self, "base_lin_vel"):
                    self.base_lin_vel[ids] = vlin_world
                if hasattr(self, "base_ang_vel"):
                    self.base_ang_vel[ids] = vang_world

                self.dof_pos[ids, :] = qpos
                self.dof_vel[ids, :] = qvel
                if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
                    self.dof_state[ids, :, 0] = qpos
                    self.dof_state[ids, :, 1] = qvel

                # push 到仿真
                try:
                    index_device = self.root_states.device
                    env_ids_i32 = env_ids.to(device=index_device, dtype=torch.int32)

                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.root_states),
                        gymtorch.unwrap_tensor(env_ids_i32),
                        env_ids_i32.numel(),
                    )
                    if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
                        self.gym.set_dof_state_tensor_indexed(
                            self.sim,
                            gymtorch.unwrap_tensor(self.dof_state),
                            gymtorch.unwrap_tensor(env_ids_i32),
                            env_ids_i32.numel(),
                        )
                except Exception as e:
                    print(f"[TiV2AMPRobot] Warning: failed to push AMP reset to sim: {e}")

            except Exception as e:
                print(f"[TiV2AMPRobot] Warning: get_state_for_reset failed ({e}), fallback to default reset.")

        # 4) 刷新张量并初始化历史
        try:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        except Exception:
            pass

        self._init_amp_obs_for_reset(env_ids)

    # ---------------- Demo 提取（可选） ----------------

    def fetch_amp_obs_demo(self, num_samples):
        """
        从专家数据采样 AMP 观测 demo：返回 [N, K*D]。
        注意：这里为简化示例，每步独立采样并拼接，严格时间一致的版本建议从 AMPLoader 的滑窗输出取样。
        """
        if self.amp_data is None:
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)

        try:
            demo_obs_list = []
            for _ in range(self._num_amp_obs_steps):
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(num_samples)
                demo = torch.cat([
                    qpos.to(self.device),
                    qvel.to(self.device),
                    vlin_local.to(self.device),
                    vang_local.to(self.device)
                ], dim=-1)
                demo_obs_list.append(demo)

            amp_obs_demo = torch.stack(demo_obs_list, dim=1)  # [N, K, D_env_simple]
            return amp_obs_demo.reshape(num_samples, -1)

        except Exception as e:
            print(f"[TiV2AMPRobot] Warning: fetch_amp_obs_demo failed ({e}), returning zeros")
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)
