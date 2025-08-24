# scripts/play_tiv2_amp.py
import os, sys, shutil, time
sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym
import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from amp_rsl_rl.runners import AMPOnPolicyRunner

LOG_ROOT = "./logs/tiv2_amp"   # 训练时保存目录（你的训练脚本输出在这）

def make_env_for_test(args):
    # 取配置并做轻量化设置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 8)
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
    if hasattr(env_cfg, "noise"): 
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_kp = False
        env_cfg.domain_rand.randomize_kd = False
    env_cfg.env.test = True

    # ✅ 禁用命令重采样（在构建 env 之前改 cfg）
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 1e12  # 足够大，相当于测试时不重采样
        # （可选）把采样区间收紧为常数，避免某些实现用区间初始化
        # 保留为训练区间也可以，这里设为 0 仅用于测试固定指令的场景
        env_cfg.commands.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ✅ 热修禁用：某些实现会在 step 中调用这些函数重采样
    if hasattr(env, "resample_commands"):
        env.resample_commands = lambda *a, **k: None
    if hasattr(env, "update_command"):
        env.update_command = lambda *a, **k: None
    if hasattr(env, "cfg") and hasattr(env.cfg, "commands"):
        try:
            env.cfg.commands.resampling_time = 1e12
        except Exception:
            pass

    return env, train_cfg

def load_policy_with_runner(env, train_cfg, device):
    # 用和训练一致的 AMPOnPolicyRunner 来加载权重，最稳妥
    train_cfg_dict = class_to_dict(train_cfg)
    runner = AMPOnPolicyRunner(env=env, train_cfg=train_cfg_dict, log_dir=None, device=device)

    # 选择最新 checkpoint
    ckpts = [f for f in os.listdir(LOG_ROOT) if f.startswith("model_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {LOG_ROOT}.")
    ckpts.sort(key=lambda n: int(n.split("_")[1].split(".")[0]))
    ckpt_path = os.path.join(LOG_ROOT, ckpts[-1])
    print(f"[PLAY] Loading checkpoint: {ckpt_path}")
    runner.load(ckpt_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)
    return policy

def play(args):
    args.task = "tiv2_amp"
    env, train_cfg = make_env_for_test(args)
    device = env.device

    # 可选：重置一次，拿到干净初态
    if hasattr(env, "reset"):
        env.reset()

    # AMP 环境的 get_observations() 可能返回 (actor_obs, extras)
    got = env.get_observations()
    obs = got[0] if isinstance(got, tuple) else got

    policy = load_policy_with_runner(env, train_cfg, device)

    # TensorBoard
    log_dir = 'debug/gym'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ★ 设定期望的前进速度（m/s），建议不要超过训练范围
    TARGET_VX = 0.5     # 0.3~1.0 都可以试
    TARGET_VY = 0.0
    TARGET_YAW = 0.0

    steps = 10 * int(env.max_episode_length)
    for i in range(steps):
        with torch.no_grad():
            # 先把指令钉住，再算动作（有些实现会在 act/step 前改 commands）
            if hasattr(env, "commands"):
                env.commands[:, 0] = TARGET_VX
                env.commands[:, 1] = TARGET_VY
                env.commands[:, 2] = TARGET_YAW

            actions = policy(obs.detach())

        # 再钉一次，确保进入 step 前是期望值
        if hasattr(env, "commands"):
            env.commands[:, 0] = TARGET_VX
            env.commands[:, 1] = TARGET_VY
            env.commands[:, 2] = TARGET_YAW

        # ✅ 在 step 之前打印一次（限频，避免刷屏）
        if hasattr(env, "commands") and (i < 10 or i % 50 == 0):
            print(f"[{i:06d}] before step cmd:", env.commands[0].detach().cpu().tolist())

        # 我们的 env.step 返回 7 元组
        obs, _, rews, dones, infos, _, _ = env.step(actions)

        # ✅ 在 step 之后再打印一次（观察是否被覆盖）
        if hasattr(env, "commands") and (i < 10 or i % 50 == 0):
            print(f"[{i:06d}] after  step cmd:", env.commands[0].detach().cpu().tolist())

        # ★ 保险：若环境在 step 内又改了，step 后再钉回目标
        if hasattr(env, "commands"):
            env.commands[:, 0] = TARGET_VX
            env.commands[:, 1] = TARGET_VY
            env.commands[:, 2] = TARGET_YAW

        # ★ 记录实际机体速度，确认真的在 +x 方向前进
        if hasattr(env, "base_lin_vel"):
            vx = env.base_lin_vel[0, 0].item()
            vy = env.base_lin_vel[0, 1].item()
            writer.add_scalar('eval/base_lin_vel_x', vx, i)
            writer.add_scalar('eval/base_lin_vel_y', vy, i)

        # —— 你已有的可视化保持不变 ——
        for j in range(6):
            writer.add_scalars(
                f'fig_gym/dim_{j}_vs_dim_{j+6}',
                {f'dim_{j}': actions[0, j].item(), f'dim_{j+6}': actions[0, j+6].item()},
                i
            )

        if hasattr(env, "base_ang_vel"):
            writer.add_scalar('fig/base_ang_vel_x', env.base_ang_vel[0, 0].item(), i)
            writer.add_scalar('fig/base_ang_vel_y', env.base_ang_vel[0, 1].item(), i)

        if isinstance(infos, dict) and 'feet_contact_forces' in infos:
            fcf = infos['feet_contact_forces']  # [num_env, 2 feet, 3]
            writer.add_scalars('fig_gym/feet_contact_force_z',
                               {'left': fcf[0, 0, 2].item(), 'right': fcf[0, 1, 2].item()},
                               i)

        writer.add_histogram('action_histogram', actions[0].cpu().numpy(), i)

    writer.close()
    print(f"[PLAY] Done. TensorBoard logs at: {log_dir}")

if __name__ == '__main__':
    args = get_args()
    # 单环境/小批次回放
    args.num_envs = 1
    play(args)
