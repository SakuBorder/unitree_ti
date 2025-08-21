#!/usr/bin/env python3
# benchmark_normalizer.py
# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import torch
from amp_rsl_rl.utils import Normalizer

# =============================================
# CONFIGURATION
# =============================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = 60  # total feature dimension (e.g. obs_dim)
batch_size = 4096
num_batches = 200  # how many batches to run
epsilon = 1e-4
clip_obs = 10.0


def main():
    device = torch.device(device_str)
    print(f"Running on device: {device}")

    # Initialize the normalizer
    norm = Normalizer(
        input_dim=input_dim,
        epsilon=epsilon,
        clip_obs=clip_obs,
        device=device,
    )

    # Warm up CUDA
    dummy = torch.randn(batch_size, input_dim, device=device)
    for _ in range(5):
        _ = norm.normalize(dummy)
        norm.update(dummy)

    # Benchmark normalize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_batches):
        _ = norm.normalize(dummy)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_elems = batch_size * num_batches
    norm_throughput = total_elems / (t1 - t0)
    print(f"[normalize] Throughput: {norm_throughput/1e6:.2f}M elements/s")

    # Benchmark update()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(num_batches):
        norm.update(dummy)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    update_throughput = total_elems / (t3 - t2)
    print(f"[update  ] Throughput: {update_throughput/1e6:.2f}M elements/s")

    # Combined normalize+update
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    for _ in range(num_batches):
        y = norm.normalize(dummy)
        norm.update(dummy)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    combined_throughput = total_elems / (t5 - t4)
    print(f"[total   ] Throughput: {combined_throughput/1e6:.2f}M elements/s")


if __name__ == "__main__":
    main()
