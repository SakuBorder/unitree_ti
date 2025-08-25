# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
import torch
import torch.nn.functional as F
from isaacgym.torch_utils import *  # quat_mul, quat_rotate, quat_rotate_inverse, etc.
from isaacgym import gymtorch
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class TiV2AMPRobot(TiV2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._enable_amp = bool(getattr(cfg.env, "enable_amp", True))
        self._num_amp_obs_steps = int(getattr(cfg.env, "num_amp_obs_steps", 2))

        # ✅ 改为从 cfg.observations.amp.root_height 读取（默认 False）
        self._amp_use_root_h = bool(
            getattr(getattr(cfg, "observations", None), "amp", None)
            and getattr(cfg.observations.amp, "root_height", False)
        )

        self._key_body_ids = self._select_key_body_ids()
        self._num_amp_obs_per_step = self._get_amp_obs_per_step_dim()
        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step

        if self._enable_amp and self._num_amp_obs_steps > 0 and self._num_amp_obs_per_step > 0:
            self._amp_obs_buf = torch.zeros(
                (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                device=self.device, dtype=torch.float32
            )
            self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
            self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        else:
            self._amp_obs_buf = None
            self._curr_amp_obs_buf = None
            self._hist_amp_obs_buf = None
            self.num_amp_obs = 0

        self.amp_data = None

        self._phaseA_freeze_cmd = True
        self._phaseA_cmd_const = torch.tensor([
            getattr(cfg.commands, "phaseA_vx", 0.8),
            getattr(cfg.commands, "phaseA_vy", 0.0),
            getattr(cfg.commands, "phaseA_yaw", 0.0)
        ], dtype=torch.float32, device=self.device)

        if not hasattr(self, "phase"):
            self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        print(f"[TiV2AMPRobot] Actor obs dim: {self.num_obs}")
        print(f"[TiV2AMPRobot] Critic obs dim: {self.num_privileged_obs}")
        if self._enable_amp:
            print(f"[TiV2AMPRobot] AMP ENABLED | AMP obs dim: {self.num_amp_obs} "
                f"({self._num_amp_obs_steps}×{self._num_amp_obs_per_step}, key_bodies={int(self._key_body_ids.numel())}, "
                f"use_root_h={self._amp_use_root_h})")
        else:
            print("[TiV2AMPRobot] AMP DISABLED")

        try:
            rb_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], self.actor_handles[0])
            inv_rb_dict = {v: k for k, v in rb_dict.items()}
            key_names = [inv_rb_dict.get(int(i), f"<id:{int(i)}?>") for i in self._key_body_ids.detach().cpu().tolist()]
            print("feet_indices:", self.feet_indices)
            print("key_body_ids:", self._key_body_ids)
            print("key_body_names:", key_names)
        except Exception as e:
            print(f"[TiV2AMPRobot] Debug key body names failed: {e}")


    def set_amp_data(self, amp_data):
        self.amp_data = amp_data

    def _apply_phaseA_frozen_commands(self, env_ids=None):
        if not self._phaseA_freeze_cmd or not hasattr(self, "commands"):
            return
        if env_ids is None:
            self.commands[:, 0:3] = self._amp_use_const_cmd()
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self.commands[env_ids, 0:3] = self._amp_use_const_cmd()
        if hasattr(self, "commands_time_left"):
            self.commands_time_left[:] = 1e9
        if hasattr(self, "command_resample_time"):
            self.command_resample_time[:] = 1e9

    def _amp_use_const_cmd(self):
        return self._phaseA_cmd_const.view(1, 3)

    def _select_key_body_ids(self):
        if hasattr(self, "feet_indices") and self.feet_indices is not None and len(self.feet_indices) >= 2:
            key_ids = list(self.feet_indices[:2].tolist())
        else:
            last = max(1, self.num_bodies - 1)
            key_ids = [max(1, last - 1), last]
        return torch.as_tensor(key_ids, device=self.device, dtype=torch.long)

    def _get_amp_obs_per_step_dim(self):
        # 1(root_h) + 6(rot6) + 3(v) + 3(w) + J + J + 3*K = 13 + 2J + 3K
        num_key_bodies = int(self._key_body_ids.numel()) if hasattr(self, "_key_body_ids") else 2
        return 13 + 2 * self.num_dof + 3 * num_key_bodies

    @staticmethod
    def _quat_to_tan_norm_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
        ref_tan = torch.zeros_like(q_xyzw[..., :3]); ref_tan[..., 0] = 1.0
        ref_norm = torch.zeros_like(q_xyzw[..., :3]); ref_norm[..., 2] = 1.0
        tan = quat_rotate(q_xyzw, ref_tan)
        norm = quat_rotate(q_xyzw, ref_norm)
        return torch.cat([tan, norm], dim=-1)

    def _compute_amp_observations_single_step(self) -> torch.Tensor:
        root_pos = self.root_states[:, 0:3]
        root_rot = self.root_states[:, 3:7]   # XYZW
        root_vel = self.root_states[:, 7:10]
        root_ang = self.root_states[:, 10:13]

        if self._amp_use_root_h:
            root_h = root_pos[:, 2:3]
        else:
            root_h = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        # 去除航向（仅绕 Z），得到局部姿态
        x, y, z, w = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        yaw = torch.atan2(fy, fx)
        half = -0.5 * yaw
        cy, sy = torch.cos(half), torch.sin(half)
        heading_rot = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)

        root_rot_local = quat_mul(heading_rot, root_rot)
        root_rot_obs = self._quat_to_tan_norm_xyzw(root_rot_local)

        local_root_vel = quat_rotate(heading_rot, root_vel)
        local_root_ang_vel = quat_rotate(heading_rot, root_ang)

        dof_obs = self.dof_pos
        dof_vel = self.dof_vel

        key_pos_world = self.rigid_body_states_view[:, self._key_body_ids, 0:3]
        rel_key = key_pos_world - root_pos.unsqueeze(1)
        flat_rel = rel_key.reshape(-1, 3)
        heading_rep = heading_rot.unsqueeze(1).expand(-1, rel_key.shape[1], -1).reshape(-1, 4)
        local_key = quat_rotate(heading_rep, flat_rel).reshape(rel_key.shape)
        flat_local_key = local_key.reshape(local_key.shape[0], -1)

        amp_obs = torch.cat(
            (root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key),
            dim=-1
        )
        return amp_obs


    def _update_hist_amp_obs(self, env_ids=None):
        if self._num_amp_obs_steps <= 1:
            return
        if env_ids is None:
            self._amp_obs_buf[:, 1:] = self._amp_obs_buf[:, :-1].clone()
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._amp_obs_buf[env_ids, 1:] = self._amp_obs_buf[env_ids, :-1].clone()

    def _compute_amp_observations(self, env_ids=None):
        if env_ids is None:
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_single_step()
        else:
            curr = self._compute_amp_observations_single_step()
            self._curr_amp_obs_buf[env_ids] = curr[env_ids]

    def compute_amp_observations(self):
        if not self._enable_amp or self._amp_obs_buf is None:
            return torch.zeros((self.num_envs, 0), device=self.device, dtype=torch.float32)
        return self._amp_obs_buf.reshape(self.num_envs, self.num_amp_obs)

    def post_physics_step(self):
        ret = super().post_physics_step()
        self._apply_phaseA_frozen_commands()
        if self._enable_amp and self._amp_obs_buf is not None:
            self._update_hist_amp_obs()
            self._compute_amp_observations()
            amp_obs_flat = self._amp_obs_buf.reshape(-1, self.num_amp_obs)
            if not hasattr(self, "extras") or self.extras is None:
                self.extras = {}
            self.extras["amp_obs"] = amp_obs_flat
            self.extras.setdefault("observations", {})
            self.extras["observations"]["amp"] = amp_obs_flat
        else:
            if not hasattr(self, "extras") or self.extras is None:
                self.extras = {}
            self.extras.setdefault("observations", {})
            self.extras["observations"].pop("amp", None)
            self.extras.pop("amp_obs", None)
        return ret

    def get_observations(self):
        try:
            super().compute_observations()
        except AttributeError as e:
            print(f"[Warning] Parent compute_observations failed: {e}")
            print("[Warning] Using observation buffers directly")

        actor_obs = self.obs_buf
        critic_obs = self.privileged_obs_buf if self.privileged_obs_buf is not None else self.obs_buf

        if not hasattr(self, "_debug_printed"):
            print(f"[Debug] Actor obs shape:  {tuple(actor_obs.shape)}")
            print(f"[Debug] Critic obs shape: {tuple(critic_obs.shape)}")
            print(f"[Debug] AMP {'enabled' if self._enable_amp else 'disabled'}, per-actor AMP dim: {self.num_amp_obs}")
            self._debug_printed = True

        if critic_obs.shape[-1] != self.num_privileged_obs:
            if critic_obs.shape[-1] > self.num_privileged_obs:
                critic_obs = critic_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - critic_obs.shape[-1]
                critic_obs = torch.cat(
                    [critic_obs, torch.zeros(critic_obs.shape[0], padding, device=critic_obs.device)],
                    dim=-1
                )

        extras = {"observations": {"critic": critic_obs}}
        if self._enable_amp and self._amp_obs_buf is not None:
            extras["observations"]["amp"] = self._amp_obs_buf.reshape(self.num_envs, self.num_amp_obs)
        return actor_obs, extras

    def step(self, actions):
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

        if privileged_obs is not None and privileged_obs.shape[-1] != self.num_privileged_obs:
            if privileged_obs.shape[-1] > self.num_privileged_obs:
                privileged_obs = privileged_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - privileged_obs.shape[-1]
                privileged_obs = torch.cat(
                    [privileged_obs, torch.zeros(privileged_obs.shape[0], padding, device=privileged_obs.device)],
                    dim=-1
                )

        if isinstance(infos, dict):
            infos.setdefault("observations", {})
            infos["observations"]["critic"] = privileged_obs if privileged_obs is not None else obs

        return obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs

    def get_privileged_observations(self):
        try:
            priv_obs = super().get_privileged_observations()
        except AttributeError as e:
            print(f"[Warning] Parent get_privileged_observations failed: {e}")
            priv_obs = getattr(self, "privileged_obs_buf", None)

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
        try:
            super().compute_observations()
        except AttributeError as e:
            if "'phase'" in str(e):
                if not hasattr(self, "phase"):
                    self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                try:
                    super().compute_observations()
                except Exception as e2:
                    print(f"[Error] Still failed after adding phase: {e2}")
                    self._manual_compute_observations()
            else:
                raise e

    def _post_physics_step_callback(self):
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

    def _init_amp_obs_for_reset(self, env_ids):
        if (not self._enable_amp) or self._amp_obs_buf is None:
            return
        if env_ids is None or len(env_ids) == 0:
            return
        self._compute_amp_observations(env_ids)
        if self._num_amp_obs_steps > 1:
            curr = self._curr_amp_obs_buf[env_ids].unsqueeze(1)
            self._hist_amp_obs_buf[env_ids] = curr.expand(-1, self._num_amp_obs_steps - 1, -1)

    def reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        super().reset_idx(env_ids)
        self._apply_phaseA_frozen_commands(env_ids)

        if self._enable_amp and getattr(self, "amp_data", None) is not None:
            n = env_ids.numel()
            try:
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(n)

                todev = lambda t: t.to(device=self.device, dtype=torch.float32)
                quat_xyzw = F.normalize(todev(quat_xyzw), dim=-1)
                qpos = todev(qpos)
                qvel = todev(qvel)
                vlin_local = todev(vlin_local)
                vang_local = todev(vang_local)

                if qpos.shape[1] != self.num_dof:
                    if qpos.shape[1] > self.num_dof:
                        qpos = qpos[:, :self.num_dof]; qvel = qvel[:, :self.num_dof]
                    else:
                        pad = self.num_dof - qpos.shape[1]
                        qpos = F.pad(qpos, (0, pad)); qvel = F.pad(qvel, (0, pad))

                vlin_world = quat_rotate(quat_xyzw, vlin_local)
                vang_world = quat_rotate(quat_xyzw, vang_local)

                ids = env_ids.long()
                self.root_states[ids, 3:7] = quat_xyzw
                self.root_states[ids, 7:10] = vlin_world
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

                try:
                    index_device = self.root_states.device
                    env_ids_i32 = env_ids.to(device=index_device, dtype=torch.int32)
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim, gymtorch.unwrap_tensor(self.root_states),
                        gymtorch.unwrap_tensor(env_ids_i32), env_ids_i32.numel(),
                    )
                    if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
                        self.gym.set_dof_state_tensor_indexed(
                            self.sim, gymtorch.unwrap_tensor(self.dof_state),
                            gymtorch.unwrap_tensor(env_ids_i32), env_ids_i32.numel(),
                        )
                except Exception as e:
                    print(f"[TiV2AMPRobot] Warning: failed to push AMP reset to sim: {e}")

            except Exception as e:
                print(f"[TiV2AMPRobot] Warning: get_state_for_reset failed ({e}), fallback to default reset.")

        try:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        except Exception:
            pass

        self._init_amp_obs_for_reset(env_ids)

    def fetch_amp_obs_demo(self, num_samples):
        if (not self._enable_amp) or self._amp_obs_buf is None:
            return torch.zeros((num_samples, 0), device=self.device)
        if self.amp_data is None:
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)
        try:
            demo_obs_list = []
            for _ in range(self._num_amp_obs_steps):
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(num_samples)
                demo = torch.cat([qpos.to(self.device), qvel.to(self.device), vlin_local.to(self.device), vang_local.to(self.device)], dim=-1)
                demo_obs_list.append(demo)
            amp_obs_demo = torch.stack(demo_obs_list, dim=1)
            return amp_obs_demo.reshape(num_samples, -1)
        except Exception as e:
            print(f"[TiV2AMPRobot] Warning: fetch_amp_obs_demo failed ({e}), returning zeros")
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)
