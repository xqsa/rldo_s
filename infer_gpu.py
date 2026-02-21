"""
RLDO GPU 推理脚本 (第二步：满分大脑冲击满分成绩)

使用训练好的 model_gpu.pth，以 Float64 + 300万次评估冲击极限 Fitness。

前置条件:
    1. 先用 main_gpu.py 训练出 edge_rate ≈ 1.0 的模型
    2. 确保 model_gpu.pth 存在

使用方法:
    1. 设置 config.py 中 TRAINING_MODE = False
    2. python infer_gpu.py
    
    或者直接运行（脚本自动强制推理参数）:
    python infer_gpu.py
"""

import os
import sys

# ==================== 强制推理模式 ====================
# 在导入任何模块前，先 patch config
import config as cfg

# 强制覆盖为推理模式参数，无论 config.py 中怎么设置
cfg.TRAINING_MODE = False
cfg.FAST_MODE = False
if hasattr(cfg, 'apply_mode_overrides'):
    cfg.apply_mode_overrides()

import torch
torch.set_default_dtype(torch.float64)  # 双精度：冲击极限

import torch.nn as nn
import numpy as np
import time
from tqdm.auto import tqdm
from typing import List

# GPU 模块
from gpu_population import GPUPopulation
from gpu_problem import GPUWeightedElliptic, generate_rotation_matrix, create_gpu_problem
from decc_gpu import DECC_GPU
from model import Model
from sparse_graph_utils import (
    dense_to_edge_index,
    connected_components_sparse,
    batch_edge_index,
    labels_to_groups
)

# 复用 main_gpu 中的工具函数
from main_gpu import (
    preprocess_pop_data,
    merge_groups_tensor_batched,
    run_decc_gpu_batch,
    generate_gpu_training_data,
)


def load_trained_model(model_path: str, device: torch.device) -> nn.Module:
    """
    加载训练好的模型 (Float32 权重 → Float64)
    """
    agent = Model(
        n=cfg.N_GROUPS,
        depth=cfg.DEPTH,
        hidden_size=cfg.HIDDEN_SIZE,
        num_heads=cfg.NUM_HEADS,
        pop_size=cfg.POP_SIZE,
        groupsize=cfg.GROUPSIZE,
        use_landscape=bool(getattr(cfg, 'USE_LANDSCAPE', True))
    ).to(device)
    
    # 加载 Float32 训练的权重
    # 兼容：低版本 torch 可能不支持 weights_only 参数
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    
    # 处理可能的 key 前缀 (_orig_mod. from torch.compile)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        cleaned[new_key] = v
    
    incompatible = agent.load_state_dict(cleaned, strict=False)
    missing = getattr(incompatible, 'missing_keys', []) or []
    unexpected = getattr(incompatible, 'unexpected_keys', []) or []
    if missing or unexpected:
        print("⚠ load_state_dict 提示：模型结构与权重键不完全一致（已使用 strict=False 继续）。")
        if missing:
            print(f"  missing_keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected_keys: {len(unexpected)}")
    
    # 转为 Float64
    agent = agent.double()
    agent.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  Dtype: {next(agent.parameters()).dtype}")
    
    return agent


def infer_single_batch(
    agent: nn.Module,
    device: torch.device,
    batch_id: int,
) -> dict:
    """
    单批次推理：生成问题 → 预测分组 → DECC 求解
    """
    n_groups = cfg.N_GROUPS
    groupsize = cfg.GROUPSIZE
    batch_size = cfg.PROBLEM_BATCH_SIZE
    max_fes = cfg.MAX_FES
    
    # 生成测试数据
    data = generate_gpu_training_data(n_groups, groupsize, batch_size, device)
    
    topos = data['topos']
    pop_data = data['pop_data']
    weights = data['weights']
    details = data['details']
    
    use_landscape = bool(getattr(cfg, 'USE_LANDSCAPE', True))
    
    # 预处理
    if use_landscape:
        pop_data_input = preprocess_pop_data(pop_data) if cfg.PREPROCESS_POP_DATA else pop_data
    else:
        pop_data_input = weights
    
    # [DATA LEAKAGE FIX] 创建 dummy_topos：全 1（对角线=0）
    # 推理时同样不能把真实拓扑喂给模型
    dummy_topos = torch.ones_like(topos)
    diag_fix = torch.arange(n_groups, device=device)
    dummy_topos[:, diag_fix, diag_fix] = 0.0
    
    # 推理：获取分组决策
    with torch.no_grad():
        connect_prob = agent(dummy_topos, pop_data_input)  # (B, N, N) Sigmoid 概率
    
    # 阈值过滤
    connect_threshold = getattr(cfg, 'INFER_CONNECT_THRESHOLD', 0.95)
    groups = (connect_prob > connect_threshold).to(torch.float64)
    
    # 强制对称性 + 去除自环
    groups = torch.maximum(groups, groups.transpose(1, 2))
    n = groups.shape[1]
    diag_idx = torch.arange(n, device=device)
    groups[:, diag_idx, diag_idx] = 0.0
    
    # 计算 edge_rate（诊断分组质量）
    diag_mask = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    allowed_edges = (topos > 0.5) & (~diag_mask)
    if allowed_edges.any():
        edge_rate = groups[allowed_edges].mean().item()
        p_connect = connect_prob[allowed_edges].mean().item()
    else:
        edge_rate = groups.mean().item()
        p_connect = connect_prob.mean().item()
    
    # 运行 DECC (核武器模式：Float64 + 300万次)
    t0 = time.time()
    fitness = run_decc_gpu_batch(weights, groups, details, device, max_fes, topos=topos)
    elapsed = time.time() - t0
    
    return {
        'fitness_mean': fitness.mean().item(),
        'fitness_min': fitness.min().item(),
        'fitness_max': fitness.max().item(),
        'fitness_std': fitness.std().item(),
        'edge_rate': edge_rate,
        'p_connect': p_connect,
        'elapsed': elapsed,
        'batch_size': batch_size,
    }


def run_inference():
    """
    推理主流程
    """
    print("=" * 60)
    print("RLDO GPU Inference (Float64 + Full FES)")
    print("=" * 60)
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU (will be very slow!)")
    
    # 推理参数（来自 config.py 的推理模式覆盖）
    print(f"\nInference Configuration:")
    print(f"  dtype: {torch.get_default_dtype()}")
    print(f"  MAX_FES: {cfg.MAX_FES:,}")
    print(f"  POP_SIZE: {cfg.POP_SIZE}")
    print(f"  PROBLEM_BATCH_SIZE: {cfg.PROBLEM_BATCH_SIZE}")
    print(f"  N_GROUPS: {cfg.N_GROUPS}")
    print(f"  GROUPSIZE: {cfg.GROUPSIZE}")
    print(f"  D = {cfg.N_GROUPS * cfg.GROUPSIZE}")
    
    # 查找模型文件
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, 'model_gpu.pth')
    if not os.path.exists(model_path):
        # 也检查上层目录
        alt_path = os.path.join('..', 'model_gpu.pth')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"\nERROR: model_gpu.pth not found!")
            print(f"  Searched: {os.path.abspath(model_path)}")
            print(f"  Please run 'python main_gpu.py' first to train the model.")
            sys.exit(1)
    
    # 加载模型
    agent = load_trained_model(model_path, device)
    
    # 推理批次数
    n_batches = getattr(cfg, 'N_EPOCHS', 1)  # 推理模式下 N_EPOCHS=1
    
    print(f"\n{'─' * 60}")
    print(f"Running {n_batches} batch(es) of inference...")
    print(f"{'─' * 60}\n")
    
    all_results = []
    
    for batch_id in range(n_batches):
        print(f"Batch {batch_id + 1}/{n_batches}...")
        result = infer_single_batch(agent, device, batch_id)
        all_results.append(result)
        
        print(
            f"  fitness: [{result['fitness_min']:.6e}, "
            f"{result['fitness_mean']:.6e}, "
            f"{result['fitness_max']:.6e}] "
            f"(std={result['fitness_std']:.2e})"
        )
        print(
            f"  edge_rate={result['edge_rate']:.3f}, "
            f"p_connect={result['p_connect']:.3f}, "
            f"time={result['elapsed']:.1f}s"
        )
    
    # 汇总统计
    print(f"\n{'=' * 60}")
    print("INFERENCE SUMMARY")
    print(f"{'=' * 60}")
    
    all_fitness_min = [r['fitness_min'] for r in all_results]
    all_fitness_mean = [r['fitness_mean'] for r in all_results]
    all_edge_rates = [r['edge_rate'] for r in all_results]
    total_time = sum(r['elapsed'] for r in all_results)
    
    print(f"  Best Fitness:    {min(all_fitness_min):.6e}")
    print(f"  Mean Fitness:    {np.mean(all_fitness_mean):.6e}")
    print(f"  Avg Edge Rate:   {np.mean(all_edge_rates):.3f}")
    print(f"  Total Time:      {total_time:.1f}s")
    print(f"  Problems Solved: {sum(r['batch_size'] for r in all_results)}")
    
    if np.mean(all_edge_rates) < 0.95:
        print(f"\n⚠ WARNING: edge_rate ({np.mean(all_edge_rates):.3f}) < 0.95")
        print("  The model may not have been trained enough.")
        print("  Consider re-training with more epochs until edge_rate ≈ 1.0")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    if cfg.RANDOM_SEED is not None:
        torch.manual_seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
    
    run_inference()
