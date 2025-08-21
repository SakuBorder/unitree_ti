import os
import numpy as np
from datetime import datetime
import sys
import sys
import os
# 添加本地 rsl_rl 到 Python 路径
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')
import os,sys
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # while True:
    #     env.step(torch.rand(2, 21).cuda())
    #     env.gym.simulate(env.sim)
    #     env.render()
    #     env.post_physics_step()
    #     env.compute_observations()
    # args.runner.resume = True
    # args.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    print(f'激活环境--{args.task}--')
    # print(args)
    train(args)
