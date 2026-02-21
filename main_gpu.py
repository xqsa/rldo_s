"""
RLDO GPU 训练脚本

端到端 GPU 训练，实现"数据不落地"的训练闭环。
使用 gpu_population、operators_vectorized、gpu_problem、decc_gpu 模块。

修复记录:
- [P0.1] 实现 GPU 分组合并逻辑，groups 真正影响 DECC
- [P0.2] 正确存储 action 并计算 PPO log_prob
- [P0.3] 纯 GPU 连通分量算法，消除 CPU 回退

使用方法:
    python main_gpu.py
"""

import torch
# [性能优化] 启用 TF32 Tensor Cores，充分利用 RTX 4090 加速 float32 矩阵乘法
torch.set_float32_matmul_precision('high')
# [两步走] 根据 TRAINING_MODE 动态设置精度
# 必须在其他模块导入前设置，避免张量类型不匹配
import config as _cfg_init
if getattr(_cfg_init, 'TRAINING_MODE', True):
    torch.set_default_dtype(torch.float32)  # 训练模式：快速，RTX 4090 单精度性能高 64 倍
    _USE_FLOAT64 = False
else:
    torch.set_default_dtype(torch.float64)  # 推理模式：精确，冲击极限 Fitness
    _USE_FLOAT64 = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Model
from tqdm.auto import trange, tqdm
import numpy as np
import time
import os
from typing import List, Tuple

# GPU 模块
from gpu_population import GPUPopulation
from gpu_problem import GPUWeightedElliptic, generate_rotation_matrix, create_gpu_problem
from decc_gpu import DECC_GPU

# 配置
import config as cfg

# 稀疏图工具 (PyG 风格)
from sparse_graph_utils import (
    dense_to_edge_index,
    connected_components_sparse,
    batch_edge_index,
    labels_to_groups
)


# ==================== 输入预处理 (数值稳定性) ====================

def preprocess_pop_data(pop_data: torch.Tensor) -> torch.Tensor:
    """
    预处理种群数据 — 简单 log 压缩极端值
    
    注意：核心归一化（Min-Max → [0,1]）由 LandscapeEncoder._normalize_input 负责！
    本函数只做简单的 log 变换，压缩 fitness 和 delta_fit 的极端数量级，
    避免与 Encoder 内部归一化冲突（双重归一化会破坏数据分布）。
    
    输入格式: [pos(D), fit(1), delta_pos(D), delta_fit(1)]
    处理规则:
    - pos / delta_pos: 不变（量级可控，~[-100, 100]）
    - fit:    log1p(fit)，将 1e6+ 压缩到 ~14
    - delta_fit: sign(x) * log1p(|x|)，保留符号的对数压缩
    
    Args:
        pop_data: (batch, n_groups, pop_size, input_dim)
    
    Returns:
        处理后的 pop_data
    """
    input_dim = pop_data.shape[-1]
    groupsize = getattr(cfg, 'GROUPSIZE', 100)
    
    # 动态轨迹格式: [pos(groupsize), fit(1), delta_pos(groupsize), delta_fit(1)]
    has_dynamic = (input_dim == groupsize * 2 + 2)
    
    if has_dynamic:
        pos       = pop_data[..., :groupsize]
        fit       = pop_data[..., groupsize:groupsize+1]
        delta_pos = pop_data[..., groupsize+1:groupsize*2+1]
        delta_fit = pop_data[..., groupsize*2+1:]
        
        # 只做 log 压缩极端值
        fit = torch.log1p(fit)                                # fit >= 0
        delta_fit = torch.log1p(delta_fit.abs()) * delta_fit.sign()  # 保留符号
        
        return torch.cat([pos, fit, delta_pos, delta_fit], dim=-1)
    else:
        # 兼容旧格式 [pos, fit]
        positions = pop_data[..., :-1]
        fitness = pop_data[..., -1:]
        fitness = torch.log1p(fitness)
        return torch.cat([positions, fitness], dim=-1)


# ==================== PyG 风格稀疏连通分量 ====================

def merge_groups_sparse(
    allgroups_base: List[List[int]], 
    group_matrix: torch.Tensor
) -> List[List[int]]:
    """
    稀疏版分组合并 (PyG 风格)
    
    复杂度: O(E × k) 而非 O(N² × k)
    """
    n = len(allgroups_base)
    
    # 1. 稠密 -> 稀疏 edge_index
    edge_index, _ = dense_to_edge_index(group_matrix, threshold=0.5)
    
    # 2. 稀疏连通分量
    labels = connected_components_sparse(edge_index, num_nodes=n)
    
    # 3. 标签 -> 分组
    return labels_to_groups(labels, allgroups_base)


def merge_groups_sparse_batched(
    allgroups_base: List[List[int]],
    group_matrices: torch.Tensor
) -> List[List[List[int]]]:
    """
    PyG 风格批量稀疏分组合并 (对角堆叠)
    
    将所有 batch 的图对角堆叠成一个超图，一次处理
    """
    batch_size = group_matrices.shape[0]
    n = len(allgroups_base)
    device = group_matrices.device
    
    # 1. 批量转换为 edge_index
    edge_indices = []
    for b in range(batch_size):
        ei, _ = dense_to_edge_index(group_matrices[b], threshold=0.5)
        edge_indices.append(ei)
    
    # 2. 对角堆叠成超图
    batched_edge_index, batch_vec = batch_edge_index(edge_indices, n)
    
    # 3. 一次性计算所有连通分量
    total_nodes = batch_size * n
    all_labels = connected_components_sparse(batched_edge_index, num_nodes=total_nodes)
    
    # 4. 拆分回各个 batch
    results = []
    for b in range(batch_size):
        start = b * n
        end = (b + 1) * n
        labels_b = all_labels[start:end]
        merged = labels_to_groups(labels_b, allgroups_base)
        results.append(merged)
    
    return results


def merge_groups_sparse_batched_with_weights(
    allgroups_base: List[List[int]],
    group_matrices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[List[List[List[int]]], List[torch.Tensor]]:
    """
    [旧版兼容接口] 带权重的批量稀疏分组合并
    
    返回 Python list，用于兼容旧接口
    生产环境应使用 merge_groups_tensor_batched
    """
    from sparse_graph_utils import labels_to_groups_with_weights
    
    batch_size = group_matrices.shape[0]
    n = len(allgroups_base)
    device = group_matrices.device
    
    edge_indices = []
    for b in range(batch_size):
        ei, _ = dense_to_edge_index(group_matrices[b], threshold=0.5)
        edge_indices.append(ei)
    
    batched_edge_index, batch_vec = batch_edge_index(edge_indices, n)
    total_nodes = batch_size * n
    all_labels = connected_components_sparse(batched_edge_index, num_nodes=total_nodes)
    
    merged_groups_list = []
    merged_weights_list = []
    
    for b in range(batch_size):
        start = b * n
        end = (b + 1) * n
        labels_b = all_labels[start:end]
        weights_b = weights[b]
        
        merged_groups, merged_weights = labels_to_groups_with_weights(
            labels_b, allgroups_base, weights_b, device
        )
        merged_groups_list.append(merged_groups)
        merged_weights_list.append(merged_weights)
    
    return merged_groups_list, merged_weights_list


def merge_groups_tensor_batched(
    base_indices: torch.Tensor,
    base_sizes: torch.Tensor,
    group_matrices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [全张量] 批量分组合并 + 权重累加 (向量化版本)
    
    消除内层双重 Python for-loop，使用排序+scatter 一次性处理
    
    Args:
        base_indices: (n_groups, groupsize) 原始分组索引
        base_sizes: (n_groups,) 每个原始组的大小
        group_matrices: (Batch, n_groups, n_groups) 分组邻接矩阵
        weights: (Batch, n_groups) 原始组的权重
        
    Returns:
        merged_indices: (Batch, Max_Merged, max_merged_size) 合并后的索引
        merged_valid: (Batch, Max_Merged, max_merged_size) 有效掩码
        merged_weights: (Batch, Max_Merged) 合并后的权重
        merged_counts: (Batch,) 每个 batch 的合并组数
        group_valid: (Batch, Max_Merged) 组有效掩码
    """
    Batch, n_groups, _ = group_matrices.shape
    device = group_matrices.device
    groupsize = base_indices.shape[1]

    # base 展开（与 batch 无关）
    all_flat_indices = base_indices.flatten()  # (n_groups * groupsize,)
    idx_in_group = torch.arange(groupsize, device=device).unsqueeze(0).expand(n_groups, -1)
    size_broadcast = base_sizes.unsqueeze(1).expand(-1, groupsize)
    valid_mask_flat = (idx_in_group < size_broadcast).flatten()  # (n_groups * groupsize,)
    
    # 1. [向量化] 批量构建 batched edge_index — 消除 for b 循环
    threshold = 0.5
    mask_all = group_matrices >= threshold  # (Batch, n_groups, n_groups)
    
    # 为每个 batch 添加节点偏移
    batch_offsets = torch.arange(Batch, device=device).view(Batch, 1, 1) * n_groups
    # 获取所有边 (batch_idx, src, dst) — 一次 nonzero 调用
    all_edges_raw = mask_all.nonzero()  # (total_edges, 3): [batch, src, dst]
    
    if all_edges_raw.numel() > 0:
        # 加上 batch 偏移
        src = all_edges_raw[:, 1] + all_edges_raw[:, 0] * n_groups
        dst = all_edges_raw[:, 2] + all_edges_raw[:, 0] * n_groups
        batched_edge_index = torch.stack([src, dst], dim=0)  # (2, total_edges)
    else:
        batched_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    
    # 2. 一次性计算连通分量
    total_nodes = Batch * n_groups
    all_labels = connected_components_sparse(batched_edge_index, num_nodes=total_nodes)
    
    # 3. [全向量化] 将 labels 转为合并后的分组
    max_merged = n_groups  # 最坏情况：不合并

    # -- 显存自适应估算 --
    label_counts = torch.bincount(all_labels, minlength=total_nodes)
    max_component_nodes = label_counts.max().clamp(min=1)  # 留在 GPU，不 .item()
    max_merged_size = min(int(max_component_nodes.item()) * groupsize, n_groups * groupsize)
    
    merged_indices = torch.zeros(Batch, max_merged, max_merged_size, dtype=torch.long, device=device)
    merged_valid = torch.zeros(Batch, max_merged, max_merged_size, dtype=torch.bool, device=device)
    merged_weights = torch.zeros(Batch, max_merged, device=device)
    merged_counts = torch.zeros(Batch, dtype=torch.long, device=device)
    
    # 4. [向量化] 为每个 batch 重新编号 labels 为 0..n_merged-1
    # 将全局 labels 拆分回每个 batch 并连续化
    labels_per_batch = all_labels.view(Batch, n_groups)  # (Batch, n_groups)
    
    # 将每个 batch 的 labels 映射为本地连续编号 (0, 1, 2, ...)
    # 方法：对 (batch_id * large_offset + label) 排序后 unique
    large_offset = total_nodes + 1
    global_keys = torch.arange(Batch, device=device).unsqueeze(1) * large_offset + labels_per_batch
    # 对每个 batch 独立 unique — 利用 batch 内 label 已有序的特性
    # 用排序 trick: 在每个 batch 内部做 unique_consecutive
    for b in range(Batch):
        labels_b = labels_per_batch[b]
        unique_labels_b, inverse_b = torch.unique(labels_b, return_inverse=True)
        n_merged_b = unique_labels_b.numel()
        merged_counts[b] = n_merged_b
        
        # 权重：scatter_add 累加
        merged_weights[b, :n_merged_b].scatter_add_(0, inverse_b, weights[b])
        
        # [向量化] 索引合并 — 消除内层 for m 循环
        # 为 n_groups*groupsize 个元素分配 merged_id
        idx_merged_ids = inverse_b.repeat_interleave(groupsize)  # (n_groups * groupsize,)
        
        # 生成每个 merged_id 内的局部位置编号
        # 方法: argsort(idx_merged_ids, stable=True) 得到按 merged_id 排列的顺序
        # 再利用 bincount 计算偏移
        sorted_order = torch.argsort(idx_merged_ids, stable=True)
        sorted_merged_ids = idx_merged_ids[sorted_order]
        sorted_flat_indices = all_flat_indices[sorted_order]
        sorted_valid = valid_mask_flat[sorted_order]
        
        # 计算每个 merged_id 的起始位置
        counts_per_merged = torch.bincount(idx_merged_ids, minlength=n_merged_b)
        # 有效元素（排除 padding）的 merged_ids + 局部位置
        # 生成组内局部位置
        cumcounts = torch.zeros(n_merged_b + 1, dtype=torch.long, device=device)
        cumcounts[1:] = counts_per_merged[:n_merged_b].cumsum(0)
        
        # 局部位置 = 全局排序位置 - 组起始位置
        global_positions = torch.arange(sorted_order.numel(), device=device)
        group_starts_expanded = cumcounts[sorted_merged_ids]
        local_positions = global_positions - group_starts_expanded
        
        # 过滤: 仅保留有效 & 在 max_merged_size 内的元素
        write_mask = sorted_valid & (local_positions < max_merged_size) & (sorted_merged_ids < max_merged)
        
        # scatter 写入
        write_merged_ids = sorted_merged_ids[write_mask]
        write_local_pos = local_positions[write_mask]
        write_values = sorted_flat_indices[write_mask]
        
        merged_indices[b, write_merged_ids, write_local_pos] = write_values
        merged_valid[b, write_merged_ids, write_local_pos] = True
    
    # 组有效掩码
    g_idx = torch.arange(max_merged, device=device).unsqueeze(0).expand(Batch, -1)
    group_valid = g_idx < merged_counts.unsqueeze(1)
    
    return merged_indices, merged_valid, merged_weights, merged_counts, group_valid


def generate_gpu_training_data(
    n_groups: int,
    groupsize: int,
    batch_size: int,
    device: torch.device
) -> dict:
    """
    生成 GPU 上的训练数据 (全向量化，无 Python 循环)
    
    [升级] 使用向量化 DE 探测生成动态轨迹特征
    参考 OPAL opal_design_phase.py 的 DE/rand/1/bin 循环
    
    输出格式: pop_data = [pos, fit, delta_pos, delta_fit]
    维度: (batch, n_groups, pop_size, groupsize*2+2)
    """
    D = n_groups * groupsize
    
    # [向量化] 批量生成拓扑矩阵 — 消除 for b in range(batch_size) 循环
    mask = torch.rand(batch_size, n_groups, n_groups, device=device) < cfg.G_PROB
    topos = (mask | mask.transpose(1, 2)).to(torch.get_default_dtype())
    # 批量清除对角线
    diag_idx = torch.arange(n_groups, device=device)
    topos[:, diag_idx, diag_idx] = 0.0
    
    # 生成权重
    weights = torch.rand(batch_size, n_groups, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # [向量化] 批量生成种群数据
    pop_size = cfg.POP_SIZE
    positions = torch.rand(batch_size, n_groups, pop_size, groupsize, device=device) * 200 - 100
    fitness = (positions ** 2).sum(dim=-1, keepdim=True)  # (batch, n_groups, pop_size, 1)
    
    # ==================== 动态轨迹探测 (参考 OPAL design_phase) ====================
    # 使用向量化 DE/rand/1/bin 跑 PROBE_STEPS 步
    probe_steps = getattr(cfg, 'PROBE_STEPS', 5)
    probe_F = getattr(cfg, 'PROBE_DE_F', 0.5)
    probe_CR = getattr(cfg, 'PROBE_DE_CR', 0.9)
    
    pos_init = positions.clone()
    fit_init = fitness.clone()
    
    for step in range(probe_steps):
        # 向量化 rand/1: 随机选 3 个个体索引
        idx_a = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_b = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_c = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        
        # gather 对应个体的位置
        idx_a_exp = idx_a.expand(-1, -1, -1, groupsize)
        idx_b_exp = idx_b.expand(-1, -1, -1, groupsize)
        idx_c_exp = idx_c.expand(-1, -1, -1, groupsize)
        
        xa = positions.gather(2, idx_a_exp)
        xb = positions.gather(2, idx_b_exp)
        xc = positions.gather(2, idx_c_exp)
        
        # 变异: donor = xa + F * (xb - xc)
        donor = xa + probe_F * (xb - xc)
        
        # 二项式交叉
        cr_mask = torch.rand_like(positions) < probe_CR
        j_rand = torch.randint(0, groupsize, (batch_size, n_groups, pop_size, 1), device=device)
        cr_mask.scatter_(-1, j_rand, True)
        trial = torch.where(cr_mask, donor, positions)
        trial = trial.clamp(-100, 100)
        
        # 评估 (Sphere 函数)
        trial_fit = (trial ** 2).sum(dim=-1, keepdim=True)
        
        # 贪婪选择
        improved = trial_fit < fitness
        positions = torch.where(improved, trial, positions)
        fitness = torch.where(improved, trial_fit, fitness)
    
    # 计算动态特征
    delta_pos = positions - pos_init      # 历史位移向量
    delta_fit = fitness - fit_init        # 适应度变化
    
    # 拼接: [pos, fit, delta_pos, delta_fit]
    pop_data = torch.cat([positions, fitness, delta_pos, delta_fit], dim=-1)
    # 维度: (batch, n_groups, pop_size, groupsize*2+2)
    
    # [向量化] 生成详细信息 — 用 torch 替代 numpy
    allgroups = [list(range(g * groupsize, (g + 1) * groupsize)) for g in range(n_groups)]
    
    # 使用 torch 生成旋转矩阵 (避免 numpy CPU 开销)
    R_dim = min(100, groupsize)
    H = torch.randn(R_dim, R_dim, device='cpu')  # QR 在 CPU 上更稳定
    Q, R_qr = torch.linalg.qr(H)
    R100 = Q.numpy()  # 保持兼容性，details['R100'] 期望 numpy
    
    # [向量化] 预生成分组索引 — 消除 for g 循环
    base_indices = torch.arange(D, device=device, dtype=torch.long).reshape(n_groups, groupsize)
    base_sizes = torch.full((n_groups,), groupsize, dtype=torch.long, device=device)
    
    details = {
        'allgroups_base': allgroups,
        'xopt': np.zeros(D),
        'D': D,
        'R100': R100,
        # 全张量路径数据
        'base_indices': base_indices,
        'base_sizes': base_sizes,
    }
    
    return {
        'topos': topos,
        'weights': weights,
        'pop_data': pop_data,
        'details': details,
    }


def run_decc_gpu_batch(
    weights: torch.Tensor,
    groups: torch.Tensor,
    details: dict,
    device: torch.device,
    max_fes: int,
    topos: torch.Tensor = None  # [P1 修复] Ground Truth 拓扑
) -> torch.Tensor:
    """
    批量运行 GPU DECC 优化
    
    [P1 修复] Ground Truth 评估模式
    - groups: Agent 预测的分组邻接矩阵 (用于优化决策)
    - topos: Ground Truth 拓扑邻接矩阵 (用于 Fitness 评估)
    
    当 cfg.USE_GT_EVALUATION=True 且提供 topos 时:
    - 优化过程使用 agent 预测的 groups
    - Fitness 评估使用 Ground Truth topos
    - 这样模型无法通过"作弊"（全不连接）来降低评估难度
    
    [全张量路径] 当 details 包含 base_indices/base_sizes 时自动使用
    - 消除所有 Python list/for-loop
    - weights 参数影响目标函数评估
    """
    D = details['D']
    xopt = torch.zeros(D, device=device)
    R100 = torch.tensor(details['R100'], device=device)  # 使用全局默认精度
    
    # [P1 修复] 检查是否启用 Ground Truth 评估
    use_gt = cfg.USE_GT_EVALUATION and topos is not None
    
    # 检测是否可使用全张量路径
    if 'base_indices' in details and 'base_sizes' in details:
        # [全张量路径]
        base_indices = details['base_indices']
        base_sizes = details['base_sizes']
        
        # Agent 预测的分组合并 (用于优化)
        merged_indices, merged_valid, merged_weights, merged_counts, group_valid = \
            merge_groups_tensor_batched(base_indices, base_sizes, groups, weights)
        
        # [P1 修复] Ground Truth 分组合并 (用于评估)
        gt_indices, gt_valid, gt_weights, gt_counts = None, None, None, None
        if use_gt:
            # 将 topos (邻接矩阵) 转换为二值连接矩阵
            # topos > 0 表示有连接
            gt_groups = (topos > 0).to(torch.get_default_dtype())
            # 强制对称性
            gt_groups = torch.maximum(gt_groups, gt_groups.transpose(1, 2))
            # 去除自环
            n = gt_groups.shape[1]
            diag_idx = torch.arange(n, device=gt_groups.device)
            gt_groups[:, diag_idx, diag_idx] = 0.0
            
            gt_indices, gt_valid, gt_weights, gt_counts, _ = \
                merge_groups_tensor_batched(base_indices, base_sizes, gt_groups, weights)
        
        # 直接使用张量创建 DECC
        from decc_batched import create_decc_batched_from_tensors
        
        max_groupsize = merged_indices.shape[2]
        if R100.shape[0] < max_groupsize:
            R_full = torch.eye(max_groupsize, device=device, dtype=R100.dtype)
            R_full[:R100.shape[0], :R100.shape[1]] = R100
            R100 = R_full
        else:
            R100 = R100[:max_groupsize, :max_groupsize]
        
        decc = create_decc_batched_from_tensors(
            group_indices=merged_indices,
            valid_mask=merged_valid,
            group_counts=merged_counts,
            weights=merged_weights,
            D=D,
            R=R100,
            device=device,
            NP=cfg.POP_SIZE,
            max_fes=max_fes,
            xopt=xopt,
            # [P1 修复] Ground Truth 参数
            gt_indices=gt_indices,
            gt_valid=gt_valid,
            gt_counts=gt_counts,
            gt_weights=gt_weights
        )
        results = decc.run()
    else:
        # [兼容路径] 使用 Python list
        allgroups_base = details['allgroups_base']
        merged_groups_list, merged_weights_list = merge_groups_sparse_batched_with_weights(
            allgroups_base, groups, weights
        )
        
        from decc_batched import run_decc_batched
        
        results = run_decc_batched(
            groups_list=merged_groups_list,
            weights_list=merged_weights_list,
            D=D,
            R=R100,
            device=device,
            NP=cfg.POP_SIZE,
            max_fes=max_fes,
            xopt=xopt
        )
    
    return results


# ==================== 旧版函数 (保留用于对比) ====================
def run_decc_gpu_batch_legacy(
    weights: torch.Tensor,
    groups: torch.Tensor,
    details: dict,
    device: torch.device,
    max_fes: int
) -> torch.Tensor:
    """
    [遗留版本] 批量运行 GPU DECC 优化，带有外层循环
    """
    batch_size = groups.shape[0]
    allgroups_base = details['allgroups_base']
    
    merged_groups_list = merge_groups_sparse_batched(allgroups_base, groups)
    
    results = []
    D = details['D']
    xopt = torch.zeros(D, device=device)
    R100 = torch.tensor(details['R100'], device=device)  # 使用全局默认精度
    
    for b in range(batch_size):
        merged_groups = merged_groups_list[b]
        
        n_merged = len(merged_groups)
        merged_weights = torch.ones(n_merged, device=device) / n_merged
        
        problem = GPUWeightedElliptic(
            allgroups=merged_groups,
            weights=merged_weights,
            xopt=xopt,
            R100=R100
        ).to(device)
        
        decc = DECC_GPU(
            problem=problem,
            allgroups=merged_groups,
            NP=cfg.POP_SIZE,
            D=D,
            Max_FEs=max_fes,
            device=device,
            verbose=False
        )
        
        best_fitness = decc.run()
        results.append(best_fitness)
    
    return torch.tensor(results, device=device)


def Buffer_GPU(
    agent: nn.Module,
    topos: torch.Tensor,
    pop_data: torch.Tensor,
    weights: torch.Tensor,
    details: dict,
    device: torch.device,
    repeat: int,
    max_fes: int
) -> dict:
    """
    GPU 版 Buffer — Bernoulli 分布采样
    
    [突破点二] PPO 从 Categorical 多分类 → 多重 Bernoulli 独立二元采样
    模型输出 p_connect: (B, N, N) 边连接概率 ∈ [0, 1]
    每条边独立采样: action ~ Bernoulli(p_connect)
    """
    batch_size, n, _, _ = pop_data.shape

    use_landscape = bool(getattr(cfg, 'USE_LANDSCAPE', True))
    
    # [数值稳定性] 可选预处理 pop_data，对 fitness 取对数（仅在 LandscapeEncoder 路径需要）
    if use_landscape:
        pop_data_processed = preprocess_pop_data(pop_data) if cfg.PREPROCESS_POP_DATA else pop_data
        model_input = pop_data_processed
    else:
        pop_data_processed = pop_data
        model_input = weights  # (batch, n)
    
    # [DATA LEAKAGE FIX] 创建 dummy_topos：全 1（对角线=0）
    # 含义："所有边都可能存在，请模型根据景观数据自己判断"
    # 重要：不能用全 0，否则 model.forward 的掩码会杀死所有连接概率
    dummy_topos = torch.ones_like(topos)
    diag_idx_topo = torch.arange(n, device=topos.device)
    dummy_topos[:, diag_idx_topo, diag_idx_topo] = 0.0
    
    # 获取策略输出 -- 使用 dummy_topos, 切断泄露
    with torch.no_grad():
        p_connect = agent(dummy_topos, model_input)  # (B, N, N) 连续概率
    
    # 重复
    if repeat > 1:
        p_connect = p_connect.unsqueeze(1).expand(-1, repeat, -1, -1).reshape(-1, n, n).contiguous()
        topos = topos.unsqueeze(1).expand(-1, repeat, -1, -1).reshape(-1, n, n).contiguous()
        dummy_topos = dummy_topos.unsqueeze(1).expand(-1, repeat, -1, -1).reshape(-1, n, n).contiguous()
        weights = weights.unsqueeze(1).expand(-1, repeat, -1).reshape(-1, n).contiguous()
        if use_landscape:
            pop_data_processed = pop_data_processed.unsqueeze(1).expand(
                -1, repeat, -1, -1, -1
            ).reshape(-1, n, pop_data_processed.shape[2], pop_data_processed.shape[3]).contiguous()
            model_input = pop_data_processed
        else:
            model_input = weights
    
    # [突破点二] Bernoulli 采样: 每条边独立抛硬币
    p_safe = p_connect.clamp(1e-7, 1 - 1e-7)  # 防止 log(0)
    dist = torch.distributions.Bernoulli(probs=p_safe)
    action = dist.sample()  # (batch, n, n) 0/1
    log_prob = dist.log_prob(action).sum(dim=[1, 2])  # 联合 log_prob
    
    # action 直接就是 0/1 邻接矩阵
    groups = action.to(torch.get_default_dtype())
    
    # 强制对称性
    groups = torch.maximum(groups, groups.transpose(1, 2))
    # 去除自环
    diag_idx = torch.arange(n, device=groups.device)
    groups[:, diag_idx, diag_idx] = 0.0

    # ==================== 诊断指标 ====================
    diag_mask = torch.eye(n, device=groups.device, dtype=torch.bool).unsqueeze(0)
    allowed_edges = (topos > 0.5) & (~diag_mask)

    # Bernoulli 熵: -p*log(p) - (1-p)*log(1-p)
    entropy = dist.entropy()  # (batch, n, n)
    allowed_count = allowed_edges.sum()
    entropy_allowed_sum = (entropy * allowed_edges).sum()
    p_connect_allowed_sum = (p_connect * allowed_edges).sum()
    edge_rate_allowed_sum = (groups * allowed_edges).sum()
    
    entropy_global = entropy.mean()
    p_connect_global = p_connect.mean()
    edge_rate_global = groups.mean()
    
    safe_count = allowed_count.clamp(min=1).to(entropy.dtype)
    entropy_mean = torch.where(allowed_count > 0, entropy_allowed_sum / safe_count, entropy_global)
    p_connect_mean = torch.where(allowed_count > 0, p_connect_allowed_sum / safe_count, p_connect_global)
    edge_rate_mean = torch.where(allowed_count > 0, edge_rate_allowed_sum / safe_count, edge_rate_global)
    
    # 运行 DECC 获取奖励
    fitness = run_decc_gpu_batch(weights, groups, details, device, max_fes, topos=topos)
    reward_raw = -fitness
    
    reward_mean = reward_raw.mean()
    reward_std = reward_raw.std(unbiased=False).clamp_min(1e-6)
    reward_normalized = (reward_raw - reward_mean) / reward_std
    reward_normalized = torch.clamp(reward_normalized, -5.0, 5.0)
    
    return {
        'topos': topos,
        'dummy_topos': dummy_topos,
        'pop_data': model_input,
        'action': action,  # (B, N, N) 0/1 Bernoulli 采样结果
        'log_prob': log_prob,
        'reward': reward_normalized,
        'fitness_tensor': fitness,
        'entropy_mean': entropy_mean,
        'p_connect_mean': p_connect_mean,
        'edge_rate_mean': edge_rate_mean,
    }


def train_gpu():
    """
    GPU 端到端训练
    """
    print("=" * 60)
    print("RLDO GPU Training (Pure GPU Connected Components)")
    print("=" * 60)
    
    # 设备
    device_cfg = str(getattr(cfg, 'DEVICE', 'auto')).lower()
    if device_cfg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("cfg.DEVICE='cuda' 但 CUDA 不可用，已回退到 CPU")
    elif device_cfg == 'cpu':
        device = torch.device('cpu')
        print("Using CPU (cfg.DEVICE='cpu')")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    
    # 参数
    n_groups = cfg.N_GROUPS
    groupsize = cfg.GROUPSIZE
    batch_size = cfg.PROBLEM_BATCH_SIZE  # config.py 已处理 FAST_MODE
    repeat = cfg.REPEAT
    n_epochs = cfg.N_EPOCHS
    max_fes = cfg.MAX_FES
    lr = cfg.LEARNING_RATE
    clip = cfg.PPO_CLIP
    
    print(f"\nConfiguration:")
    print(f"  dtype: {torch.get_default_dtype()}")
    print(f"  TRAINING_MODE: {getattr(cfg, 'TRAINING_MODE', True)}")
    print(f"  n_groups: {n_groups}, groupsize: {groupsize}")
    print(f"  batch_size: {batch_size}, repeat: {repeat}")
    print(f"  n_epochs: {n_epochs}, max_fes: {max_fes}")
    print(f"  POP_SIZE: {cfg.POP_SIZE}")
    print(f"  FAST_MODE: {cfg.FAST_MODE}")
    print(f"  USE_GT_EVALUATION: {cfg.USE_GT_EVALUATION}")  # [P1 修复] 显示 GT 评估状态
    print(f"  USE_SUPERVISED_LOSS: {cfg.USE_SUPERVISED_LOSS}")  # [P2 新增] 监督学习辅助
    if cfg.USE_SUPERVISED_LOSS:
        print(f"    -> coef={cfg.SUPERVISED_COEF}, decay={cfg.SUPERVISED_DECAY}, min={cfg.SUPERVISED_MIN}")
    
    # [P2 新增] 监督损失权重 (会随训练衰减)
    supervised_coef = cfg.SUPERVISED_COEF if cfg.USE_SUPERVISED_LOSS else 0.0
    print(f"  USE_LANDSCAPE: {getattr(cfg, 'USE_LANDSCAPE', True)}")
    
    # 创建模型
    agent = Model(
        n=n_groups,
        depth=cfg.DEPTH,
        hidden_size=cfg.HIDDEN_SIZE,
        num_heads=cfg.NUM_HEADS,
        pop_size=cfg.POP_SIZE,
        groupsize=groupsize,
        use_landscape=bool(getattr(cfg, 'USE_LANDSCAPE', True)),
        edge_head_dim=getattr(cfg, 'EDGE_HEAD_DIM', 64)
    ).to(device)  # 精度由 set_default_dtype 统一控制
    if _USE_FLOAT64:
        agent = agent.double()
    
    # [第三步] 可选加载预训练 Encoder 权重 + 层冻结
    if getattr(cfg, 'USE_PRETRAINED_ENCODER', False):
        pretrain_path = getattr(cfg, 'PRETRAIN_WEIGHT_PATH', 'encoder_pretrained.pth')
        if hasattr(agent, '_orig_mod'):
            target_model = agent._orig_mod  # torch.compile 包装
        else:
            target_model = agent
        if target_model.use_landscape and os.path.exists(pretrain_path):
            pretrained_state = torch.load(pretrain_path, map_location=device, weights_only=True)
            encoder_state = target_model.landscape_encoder.state_dict()
            # 只加载匹配的 key
            loaded_keys = []
            for k, v in pretrained_state.items():
                if k in encoder_state and v.shape == encoder_state[k].shape:
                    encoder_state[k] = v
                    loaded_keys.append(k)
            target_model.landscape_encoder.load_state_dict(encoder_state)
            print(f"[Pretrain] Loaded {len(loaded_keys)}/{len(encoder_state)} encoder weights from {pretrain_path}")
            
            # 冻结前 N 层 point_mlp
            freeze_layers = getattr(cfg, 'FREEZE_ENCODER_LAYERS', 0)
            if freeze_layers > 0:
                frozen_count = 0
                for i, layer in enumerate(target_model.landscape_encoder.point_mlp):
                    if i // 3 < freeze_layers:  # 每层 = Linear + BN + ReLU (3个)
                        for p in layer.parameters():
                            p.requires_grad = False
                            frozen_count += 1
                print(f"[Pretrain] Froze {frozen_count} parameters in first {freeze_layers} MLP layers")
        else:
            print(f"[Pretrain] Skipped: file not found or landscape disabled")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=lr)
    
    # [两步走] 根据精度模式决定是否使用 torch.compile
    if not _USE_FLOAT64:
        # 训练模式：Float32 可以使用 torch.compile 加速
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
            print("torch.compile enabled (Float32 training mode)")
        except Exception as e:
            print(f"torch.compile failed: {e}, using eager mode")
    else:
        # 推理模式：Float64 禁用 compile（Inductor 会注入 Float32 张量导致类型冲突）
        print("torch.compile disabled (Float64 inference mode)")
    
    print(f"\nModel parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # 训练循环
    print("\n" + "-" * 60)
    print("Training started")
    print("-" * 60)
    
    for epoch in trange(n_epochs, desc="Epochs"):
        # 生成数据
        data = generate_gpu_training_data(n_groups, groupsize, batch_size, device)
        
        # 收集经验
        buffer = Buffer_GPU(
            agent=agent,
            topos=data['topos'],
            pop_data=data['pop_data'],
            weights=data['weights'],
            details=data['details'],
            device=device,
            repeat=repeat,
            max_fes=max_fes
        )
        
        # PPO 更新 [显存优化] 使用 mini-batch 避免 OOM
        reuse_time = cfg.REUSE_TIME
        total_samples = buffer['topos'].shape[0]
        # mini-batch 大小：确保不超过 256 以适应显存
        ppo_batch_size = min(256, total_samples)
        
        for _ in range(reuse_time):
            # 随机打乱索引
            perm = torch.randperm(total_samples, device=device)
            
            # Mini-batch 更新
            total_loss = 0.0
            n_batches = 0
            for start in range(0, total_samples, ppo_batch_size):
                end = min(start + ppo_batch_size, total_samples)
                idx = perm[start:end]
                
                # 提取 mini-batch
                mb_topos = buffer['topos'][idx]           # 真实拓扑（仅用于监督损失）
                mb_dummy_topos = buffer['dummy_topos'][idx]  # [DATA LEAKAGE FIX] 模型输入
                mb_pop_data = buffer['pop_data'][idx]
                mb_action = buffer['action'][idx]
                mb_old_log_prob = buffer['log_prob'][idx]
                mb_reward = buffer['reward'][idx]
                
                # 前向传播 -- 使用 dummy_topos，输出 (B, N, N) 边概率
                p_connect = agent(mb_dummy_topos, mb_pop_data)  # (B, N, N)
                
                # [突破点二] Bernoulli 分布计算新策略下的 log_prob
                p_safe = p_connect.clamp(1e-7, 1 - 1e-7)
                dist = torch.distributions.Bernoulli(probs=p_safe)
                new_log_prob = dist.log_prob(mb_action).sum(dim=[1, 2])
                
                # PPO loss
                ratio = torch.exp(new_log_prob - mb_old_log_prob.detach())
                surr1 = ratio * mb_reward
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_reward
                ppo_loss = -torch.min(surr1, surr2).mean()

                # [突破点二] 拓扑熵正则化 (Bernoulli 熵)
                n = mb_topos.shape[1]
                diag_mask = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
                allowed_edges = (mb_topos > 0.5) & (~diag_mask)
                mb_entropy = dist.entropy()
                allowed_count = allowed_edges.sum()
                entropy_allowed = (mb_entropy * allowed_edges).sum()
                safe_count = allowed_count.clamp(min=1).to(mb_entropy.dtype)
                entropy_mean = torch.where(
                    allowed_count > 0,
                    entropy_allowed / safe_count,
                    mb_entropy.mean()
                )
                topo_entropy_coef = getattr(cfg, 'TOPO_ENTROPY_COEF', cfg.ENTROPY_COEF)
                entropy_loss = -topo_entropy_coef * entropy_mean

                # 监督损失：p_connect vs 真实拓扑
                supervised_loss = torch.zeros(1, device=device, requires_grad=False).squeeze()
                if supervised_coef > 0:
                    target = (mb_topos > 0).to(torch.get_default_dtype())
                    target[:, torch.arange(n), torch.arange(n)] = 0
                    
                    p_connect_safe_sup = p_connect.clamp(1e-7, 1 - 1e-7)
                    bce = -(target * torch.log(p_connect_safe_sup) + 
                            (1 - target) * torch.log(1 - p_connect_safe_sup))
                    
                    non_diag_mask = ~diag_mask.expand_as(bce)
                    supervised_loss = bce[non_diag_mask].mean()
                    
                    if epoch % cfg.LOG_INTERVAL == 0 and start == 0 and _ == 0:
                        target_density = target[non_diag_mask.expand(target.shape[0], -1, -1)].mean().item()
                        pred_density = p_connect[non_diag_mask.expand(p_connect.shape[0], -1, -1)].mean().item()
                        tqdm.write(
                            f"  [SUP DEBUG] target_density={target_density:.3f}, "
                            f"pred_density={pred_density:.3f}, "
                            f"sup_loss={supervised_loss.item():.4f}, "
                            f"ppo_loss={ppo_loss.item():.4f}, "
                            f"weighted_sup={supervised_coef * supervised_loss.item():.4f}"
                        )

                loss = ppo_loss + entropy_loss + supervised_coef * supervised_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # 记录平均 loss（用于最后一次迭代）
            avg_loss = total_loss / max(n_batches, 1)
        
        # [P2 新增] 衰减监督损失权重
        if cfg.USE_SUPERVISED_LOSS:
            supervised_coef = max(cfg.SUPERVISED_MIN, supervised_coef * cfg.SUPERVISED_DECAY)
        
        if epoch % cfg.LOG_INTERVAL == 0:
            # [GPU 优化] 仅在日志输出时才 .item()，批量同步一次
            fit_tensor = buffer['fitness_tensor']
            sup_info = f", sup_coef={supervised_coef:.3f}" if cfg.USE_SUPERVISED_LOSS else ""
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f}, "
                f"fitness=[{fit_tensor.min().item():.2e}, {fit_tensor.mean().item():.2e}, {fit_tensor.max().item():.2e}], "
                f"entropy={buffer['entropy_mean'].item():.3f}, "
                f"p_connect={buffer['p_connect_mean'].item():.3f}, "
                f"edge_rate={buffer['edge_rate_mean'].item():.3f}"
                f"{sup_info}"
            )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # 保存模型
    save_path = 'model_gpu.pth'
    # [torch.compile 兼容] 如果被 compile 包装，取出原始模型
    raw_agent = getattr(agent, '_orig_mod', agent)
    torch.save(raw_agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # 下一步提示
    if getattr(cfg, 'TRAINING_MODE', True):
        print(f"\n\u2501" * 50)
        print("» 下一步: 设置 config.py 中 TRAINING_MODE = False，然后运行:")
        print("  python infer_gpu.py")
        print("  这将加载 model_gpu.pth 并以 Float64 + 300万次评估冲击极限 Fitness")
        print(f"\u2501" * 50)
    
    return agent


if __name__ == "__main__":
    # 设置随机种子
    if cfg.RANDOM_SEED is not None:
        torch.manual_seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
    
    # 运行训练
    agent = train_gpu()
