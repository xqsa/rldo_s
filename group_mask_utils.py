"""
分组掩码工具 V2 (Group Mask Utilities)

设计目标：
1. 核心向量化函数使用 scatter/gather 避免 Python 循环
2. 输入输出全部为 GPU Tensor
3. Batch 维度并行处理

实现状态：
- 单样本操作: 完全向量化
- 批量操作: Batch 级循环 (Batch 通常 < 64，可接受)
- 保留 Python List 兼容接口供入口处转换

核心思想：
1. 用 (Max_Groups, D) 的 0/1 掩码矩阵替代 Python 列表
2. 支持 Batch 维度: (Batch, Max_Groups, D)
3. 使用 scatter/gather 实现子空间操作
"""

import torch
from typing import List, Tuple, Optional


# ==================== 核心向量化函数 ====================

def groups_to_mask_vectorized(
    indices_flat: torch.Tensor,      # (total_elements,) 所有组的索引拼接
    group_sizes: torch.Tensor,       # (n_groups,) 每组的实际大小
    D: int,
    device: torch.device,
    max_groups: int = None
) -> Tuple[torch.Tensor, int]:
    """
    [向量化] 将扁平索引转换为分组掩码张量
    
    Args:
        indices_flat: 所有组索引的拼接
        group_sizes: 每组的大小
        D: 总维度
        device: 计算设备
        max_groups: 最大分组数
        
    Returns:
        mask: (Max_Groups, D) 掩码
        n_valid_groups: 有效组数
    """
    n_groups = group_sizes.numel()
    
    if max_groups is None:
        max_groups = n_groups
    
    # 创建掩码
    mask = torch.zeros(max_groups, D, dtype=torch.float32, device=device)
    
    if n_groups == 0:
        return mask, 0
    
    # 计算每个元素属于哪个组
    group_ids = torch.repeat_interleave(
        torch.arange(n_groups, device=device),
        group_sizes
    )  # (total_elements,)
    
    # 使用 scatter 一次性设置所有掩码
    # 只设置有效组 (group_id < max_groups)
    valid_elements = group_ids < max_groups
    valid_group_ids = group_ids[valid_elements]
    valid_indices = indices_flat[valid_elements]
    
    # 设置掩码
    mask[valid_group_ids, valid_indices] = 1.0
    
    return mask, min(n_groups, max_groups)


def batch_groups_to_mask_vectorized(
    batch_indices: torch.Tensor,     # (Batch, max_total_elements) Padded 索引
    batch_group_sizes: torch.Tensor, # (Batch, max_n_groups) 每组大小
    batch_n_groups: torch.Tensor,    # (Batch,) 每样本的组数
    D: int,
    device: torch.device,
    max_groups: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [向量化] 批量将索引转换为分组掩码
    
    所有输入必须是 GPU Tensor，无 Python List
    
    Returns:
        masks: (Batch, Max_Groups, D)
        valid_counts: (Batch,)
    """
    Batch = batch_indices.shape[0]
    
    if max_groups is None:
        max_groups = batch_n_groups.max().item()
    
    masks = torch.zeros(Batch, max_groups, D, dtype=torch.float32, device=device)
    
    # 遍历 Batch（这是最外层，Batch 通常较小，可接受）
    for b in range(Batch):
        n_groups = batch_n_groups[b].item()
        if n_groups == 0:
            continue
        
        # 获取这个样本的 group_sizes
        sizes = batch_group_sizes[b, :n_groups]
        total = sizes.sum().item()
        
        # 获取扁平索引
        flat_indices = batch_indices[b, :total]
        
        # 构建 group_ids
        group_ids = torch.repeat_interleave(
            torch.arange(n_groups, device=device, dtype=torch.long),
            sizes
        )
        
        # Scatter
        valid_mask = group_ids < max_groups
        masks[b, group_ids[valid_mask], flat_indices[valid_mask]] = 1.0
    
    return masks, batch_n_groups.clamp(max=max_groups)


def groups_to_indices_padded_vectorized(
    indices_flat: torch.Tensor,      # (total_elements,) 所有组的索引拼接
    group_sizes: torch.Tensor,       # (n_groups,) 每组的实际大小
    max_groups: int,
    max_groupsize: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [向量化] 将扁平索引转换为 Padded 索引张量
    
    Returns:
        indices: (Max_Groups, max_groupsize)
        valid_mask: (Max_Groups, max_groupsize) bool
    """
    n_groups = group_sizes.numel()
    
    indices = torch.zeros(max_groups, max_groupsize, dtype=torch.long, device=device)
    valid = torch.zeros(max_groups, max_groupsize, dtype=torch.bool, device=device)
    
    if n_groups == 0:
        return indices, valid
    
    # 计算每个元素的 (group_id, position_in_group)
    group_ids = torch.repeat_interleave(
        torch.arange(n_groups, device=device),
        group_sizes
    )
    
    # 计算组内位置
    cumsum = torch.cat([torch.zeros(1, device=device, dtype=torch.long), group_sizes.cumsum(0)[:-1]])
    positions = torch.arange(indices_flat.numel(), device=device) - torch.repeat_interleave(cumsum, group_sizes)
    
    # 过滤有效元素
    valid_mask_flat = (group_ids < max_groups) & (positions < max_groupsize)
    valid_group_ids = group_ids[valid_mask_flat]
    valid_positions = positions[valid_mask_flat]
    valid_indices_val = indices_flat[valid_mask_flat]
    
    # Scatter
    indices[valid_group_ids, valid_positions] = valid_indices_val
    valid[valid_group_ids, valid_positions] = True
    
    return indices, valid


def batch_groups_to_indices_padded_from_tensors(
    batch_indices_flat: torch.Tensor,    # (Batch, max_total_elements)
    batch_group_sizes: torch.Tensor,     # (Batch, max_n_groups)
    batch_n_groups: torch.Tensor,        # (Batch,)
    device: torch.device,
    max_groups: int = None,
    max_groupsize: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [向量化] 批量将扁平索引转换为 Padded 索引张量
    
    所有输入必须是 GPU Tensor
    
    Returns:
        indices: (Batch, Max_Groups, max_groupsize)
        valid_mask: (Batch, Max_Groups, max_groupsize) bool
        group_counts: (Batch,)
    """
    Batch = batch_indices_flat.shape[0]
    
    if max_groups is None:
        max_groups = batch_n_groups.max().item()
    if max_groupsize is None:
        max_groupsize = batch_group_sizes.max().item()
    
    indices = torch.zeros(Batch, max_groups, max_groupsize, dtype=torch.long, device=device)
    valid = torch.zeros(Batch, max_groups, max_groupsize, dtype=torch.bool, device=device)
    
    for b in range(Batch):
        n_groups = batch_n_groups[b].item()
        if n_groups == 0:
            continue
        
        sizes = batch_group_sizes[b, :n_groups]
        total = sizes.sum().item()
        flat_indices = batch_indices_flat[b, :total]
        
        # 使用向量化函数
        idx_b, valid_b = groups_to_indices_padded_vectorized(
            flat_indices, sizes, max_groups, max_groupsize, device
        )
        indices[b] = idx_b
        valid[b] = valid_b
    
    return indices, valid, batch_n_groups.clamp(max=max_groups)


# ==================== 兼容接口 (接受 Python List，转换后调用) ====================

def groups_to_mask(
    allgroups: List[List[int]],
    D: int,
    device: torch.device,
    max_groups: int = None
) -> Tuple[torch.Tensor, int]:
    """
    [兼容接口] 将分组列表转换为分组掩码张量
    
    注意: 此接口接受 Python List，内部转换为 Tensor
    生产环境应使用 groups_to_mask_vectorized
    """
    n_groups = len(allgroups)
    
    if n_groups == 0:
        if max_groups is None:
            max_groups = 1
        return torch.zeros(max_groups, D, dtype=torch.float32, device=device), 0
    
    # 转换为 Tensor
    indices_flat = torch.tensor(
        [idx for group in allgroups for idx in group],
        dtype=torch.long, device=device
    )
    group_sizes = torch.tensor(
        [len(g) for g in allgroups],
        dtype=torch.long, device=device
    )
    
    return groups_to_mask_vectorized(indices_flat, group_sizes, D, device, max_groups)


def batch_groups_to_mask(
    groups_list: List[List[List[int]]],
    D: int,
    device: torch.device,
    max_groups: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [兼容接口] 批量将分组列表转换为分组掩码张量
    """
    batch_size = len(groups_list)
    
    if max_groups is None:
        max_groups = max(len(groups) for groups in groups_list) if groups_list else 1
    
    masks = torch.zeros(batch_size, max_groups, D, dtype=torch.float32, device=device)
    valid_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b, allgroups in enumerate(groups_list):
        if len(allgroups) == 0:
            continue
        
        mask, n_valid = groups_to_mask(allgroups, D, device, max_groups)
        masks[b] = mask
        valid_counts[b] = n_valid
    
    return masks, valid_counts


def compute_max_group_size(groups_list: List[List[List[int]]]) -> int:
    """计算所有样本中的最大子问题维度"""
    max_size = 0
    for groups in groups_list:
        for g in groups:
            max_size = max(max_size, len(g))
    return max_size


def groups_to_indices_padded(
    allgroups: List[List[int]],
    max_groups: int,
    max_groupsize: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [兼容接口] 将分组索引 Padding 为统一形状张量
    """
    n_groups = len(allgroups)
    
    if n_groups == 0:
        indices = torch.zeros(max_groups, max_groupsize, dtype=torch.long, device=device)
        valid = torch.zeros(max_groups, max_groupsize, dtype=torch.bool, device=device)
        return indices, valid
    
    # 转换
    indices_flat = torch.tensor(
        [idx for group in allgroups for idx in group],
        dtype=torch.long, device=device
    )
    group_sizes = torch.tensor(
        [len(g) for g in allgroups],
        dtype=torch.long, device=device
    )
    
    return groups_to_indices_padded_vectorized(
        indices_flat, group_sizes, max_groups, max_groupsize, device
    )


def batch_groups_to_indices_padded(
    groups_list: List[List[List[int]]],
    device: torch.device,
    max_groups: int = None,
    max_groupsize: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [兼容接口] 批量将分组索引 Padding 为统一形状张量
    """
    batch_size = len(groups_list)
    
    if max_groups is None:
        max_groups = max(len(groups) for groups in groups_list) if groups_list else 1
    if max_groupsize is None:
        max_groupsize = compute_max_group_size(groups_list) if groups_list else 1
    
    indices = torch.zeros(batch_size, max_groups, max_groupsize, dtype=torch.long, device=device)
    valid = torch.zeros(batch_size, max_groups, max_groupsize, dtype=torch.bool, device=device)
    counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b, allgroups in enumerate(groups_list):
        if len(allgroups) == 0:
            continue
        idx_b, valid_b = groups_to_indices_padded(allgroups, max_groups, max_groupsize, device)
        indices[b] = idx_b
        valid[b] = valid_b
        counts[b] = len(allgroups)
    
    return indices, valid, counts


# ==================== 子空间操作 ====================

def extract_subspaces_batched(
    population: torch.Tensor,
    group_indices: torch.Tensor,
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    批量提取所有子空间
    
    Args:
        population: (Batch, NP, D) 种群
        group_indices: (Batch, Max_Groups, max_groupsize) Padded 索引
        valid_mask: (Batch, Max_Groups, max_groupsize) 有效位置掩码
        
    Returns:
        subspaces: (Batch, Max_Groups, NP, max_groupsize) 子空间变量
    """
    Batch, NP, D = population.shape
    _, Max_Groups, max_groupsize = group_indices.shape
    
    # 扩展 group_indices: (Batch, Max_Groups, 1, max_groupsize) -> (Batch, Max_Groups, NP, max_groupsize)
    indices_expanded = group_indices.unsqueeze(2).expand(-1, -1, NP, -1)
    
    # 扩展 population: (Batch, 1, NP, D) -> 准备 gather
    pop_expanded = population.unsqueeze(1).expand(-1, Max_Groups, -1, -1)
    
    # Gather: 沿最后一维（D）收集
    subspaces = torch.gather(pop_expanded, dim=-1, index=indices_expanded)
    
    return subspaces


def write_subspaces_batched(
    target: torch.Tensor,
    subspaces: torch.Tensor,
    group_indices: torch.Tensor,
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    批量将子空间写回完整向量 (使用 valid_mask 安全写回)
    
    Args:
        target: (Batch, Max_Groups, NP, D) 目标向量
        subspaces: (Batch, Max_Groups, NP, max_groupsize) 子空间变量
        group_indices: (Batch, Max_Groups, max_groupsize) Padded 索引
        valid_mask: (Batch, Max_Groups, max_groupsize) 有效位置掩码
        
    Returns:
        result: (Batch, Max_Groups, NP, D) 更新后的向量
    """
    Batch, Max_Groups, NP, D = target.shape
    _, _, max_groupsize = group_indices.shape
    
    result = target.clone()
    
    # 扩展索引和掩码
    indices_expanded = group_indices.unsqueeze(2).expand(-1, -1, NP, -1)
    valid_expanded = valid_mask.unsqueeze(2).expand(-1, -1, NP, -1)
    
    # 保存原始值
    original_vals = torch.gather(result, dim=-1, index=indices_expanded)
    
    # Scatter (会写入所有位置，包括 padding)
    result.scatter_(dim=-1, index=indices_expanded, src=subspaces)
    
    # 恢复无效位置的原始值
    result.scatter_(dim=-1, index=indices_expanded,
                   src=torch.where(valid_expanded, subspaces, original_vals))
    
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing group_mask_utils (V2 Vectorized)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 测试 groups_to_mask
    print("\n--- groups_to_mask ---")
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    D = 10
    mask, n_valid = groups_to_mask(groups, D, device)
    print(f"Groups: {groups}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")
    print(f"Valid groups: {n_valid}")
    assert mask.shape == (3, 10), f"Expected (3, 10), got {mask.shape}"
    
    # 验证掩码正确性
    assert mask[0, 0:3].sum() == 3, "Group 0 mask incorrect"
    assert mask[1, 3:6].sum() == 3, "Group 1 mask incorrect"
    assert mask[2, 6:10].sum() == 4, "Group 2 mask incorrect"
    print("Mask values verified!")
    
    # 测试 batch_groups_to_mask
    print("\n--- batch_groups_to_mask ---")
    groups_list = [
        [[0, 1], [2, 3], [4, 5]],  # 样本 0: 3 组
        [[0, 1, 2], [3, 4, 5]],    # 样本 1: 2 组
        [[0, 1, 2, 3, 4, 5]],      # 样本 2: 1 组
    ]
    masks, counts = batch_groups_to_mask(groups_list, D=6, device=device)
    print(f"Masks shape: {masks.shape}")
    print(f"Group counts: {counts}")
    assert masks.shape == (3, 3, 6), f"Expected (3, 3, 6), got {masks.shape}"
    assert counts.tolist() == [3, 2, 1], f"Expected [3, 2, 1], got {counts.tolist()}"
    
    # 测试 batch_groups_to_indices_padded
    print("\n--- batch_groups_to_indices_padded ---")
    indices, valid, counts = batch_groups_to_indices_padded(groups_list, device)
    print(f"Indices shape: {indices.shape}")
    print(f"Valid mask shape: {valid.shape}")
    print(f"Counts: {counts}")
    
    # 测试 extract_subspaces_batched
    print("\n--- extract_subspaces_batched ---")
    Batch, NP, D = 3, 10, 6
    population = torch.arange(Batch * NP * D, device=device, dtype=torch.float32).reshape(Batch, NP, D)
    subspaces = extract_subspaces_batched(population, indices, valid)
    print(f"Population shape: {population.shape}")
    print(f"Subspaces shape: {subspaces.shape}")
    
    # 测试 write_subspaces_batched (安全写回)
    print("\n--- write_subspaces_batched ---")
    target = torch.zeros(Batch, 3, NP, D, device=device)
    result = write_subspaces_batched(target, subspaces, indices, valid)
    print(f"Target shape: {target.shape}")
    print(f"Result shape: {result.shape}")
    
    print("\nOK: All group_mask_utils V2 tests passed!")
