"""
向量化 DE 算子

参考 EvoTorch functional.py 设计，实现无循环的 DE 变异/交叉操作。
所有操作在 GPU 上批量执行，避免 Python 循环。

使用方法:
    donor = de_mutation_rand1(pop.values, F=0.5)
    trial = de_crossover_bin(pop.values, donor, CR=0.9)
"""

import torch
from typing import Optional, Tuple


def _generate_distinct_indices(
    NP: int, 
    n_indices: int, 
    device: torch.device,
    exclude_self: bool = True
) -> torch.Tensor:
    """
    为每个个体生成 n_indices 个不重复索引
    
    Args:
        NP: 种群大小
        n_indices: 每个个体需要的索引数量
        device: 计算设备
        exclude_self: 是否排除自身索引
        
    Returns:
        (NP, n_indices) 索引张量
    """
    # 生成候选池（排除自身）
    if exclude_self:
        # [GPU 优化] 使用环形索引向量化：偏移 (1 到 NP-1) + 基地址
        offsets = torch.stack([
            torch.randperm(NP - 1, device=device)[:n_indices] + 1
            for _ in range(NP)
        ])
        base = torch.arange(NP, device=device).unsqueeze(1)
        indices = (base + offsets) % NP
        return indices
    else:
        # 简单随机选择
        indices = torch.stack([
            torch.randperm(NP, device=device)[:n_indices]
            for _ in range(NP)
        ])
    
    return indices


def _generate_distinct_indices_fast(
    NP: int,
    n_indices: int,
    device: torch.device
) -> torch.Tensor:
    """
    快速生成不重复索引（使用向量化操作）
    
    [P1 改进] 小种群时使用精确方法，大种群使用快速近似
    
    Args:
        NP: 种群大小
        n_indices: 每个个体需要的索引数量
        device: 计算设备
        
    Returns:
        (NP, n_indices) 索引张量
    """
    # [P1] 小种群使用精确方法（保证排除自身和不重复）
    if NP < 10:
        return _generate_distinct_indices(NP, n_indices, device, exclude_self=True)
    
    # 大种群使用快速近似方法
    # 生成随机偏移 (1 到 NP-1，确保不选自身)
    offsets = torch.randint(1, NP, (NP, n_indices), device=device)
    # 基础索引
    base = torch.arange(NP, device=device).unsqueeze(1)
    # 环形索引
    return (base + offsets) % NP


# ==================== DE 变异算子 ====================

def de_mutation_rand1(
    values: torch.Tensor,
    F: float = 0.5
) -> torch.Tensor:
    """
    DE/rand/1 变异: v = x_r1 + F * (x_r2 - x_r3)
    
    Args:
        values: (NP, D) 种群决策变量
        F: 缩放因子
        
    Returns:
        (NP, D) 变异向量
    """
    NP = values.shape[0]
    device = values.device
    
    # 生成 3 个不重复索引
    indices = _generate_distinct_indices_fast(NP, 3, device)
    r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]
    
    return values[r1] + F * (values[r2] - values[r3])


def de_mutation_best1(
    values: torch.Tensor,
    fitness: torch.Tensor,
    F: float = 0.5
) -> torch.Tensor:
    """
    DE/best/1 变异: v = x_best + F * (x_r1 - x_r2)
    
    Args:
        values: (NP, D) 种群决策变量
        fitness: (NP,) 适应度
        F: 缩放因子
        
    Returns:
        (NP, D) 变异向量
    """
    NP = values.shape[0]
    device = values.device
    
    # 找最优个体
    best_idx = fitness.argmin()
    x_best = values[best_idx]
    
    # 生成 2 个索引
    indices = _generate_distinct_indices_fast(NP, 2, device)
    r1, r2 = indices[:, 0], indices[:, 1]
    
    return x_best + F * (values[r1] - values[r2])


def de_mutation_current_to_best1(
    values: torch.Tensor,
    fitness: torch.Tensor,
    F: float = 0.5
) -> torch.Tensor:
    """
    DE/current-to-best/1 变异: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    
    Args:
        values: (NP, D) 种群决策变量
        fitness: (NP,) 适应度
        F: 缩放因子
        
    Returns:
        (NP, D) 变异向量
    """
    NP = values.shape[0]
    device = values.device
    
    # 找最优个体
    best_idx = fitness.argmin()
    x_best = values[best_idx]
    
    # 生成 2 个索引
    indices = _generate_distinct_indices_fast(NP, 2, device)
    r1, r2 = indices[:, 0], indices[:, 1]
    
    return values + F * (x_best - values) + F * (values[r1] - values[r2])


def de_mutation_current_to_pbest1(
    values: torch.Tensor,
    fitness: torch.Tensor,
    F: float = 0.5,
    p: float = 0.1
) -> torch.Tensor:
    """
    DE/current-to-pbest/1 变异 (SHADE 风格):
    v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
    
    Args:
        values: (NP, D) 种群决策变量
        fitness: (NP,) 适应度
        F: 缩放因子
        p: top-p 比例 (选择最优 p 比例的个体)
        
    Returns:
        (NP, D) 变异向量
    """
    NP = values.shape[0]
    device = values.device
    
    # 找 top-p 个体
    k = max(1, int(NP * p))
    sorted_idx = fitness.argsort()[:k]
    
    # 为每个个体随机选择一个 pbest
    pbest_choice = torch.randint(0, k, (NP,), device=device)
    pbest_idx = sorted_idx[pbest_choice]
    x_pbest = values[pbest_idx]
    
    # 生成 2 个索引
    indices = _generate_distinct_indices_fast(NP, 2, device)
    r1, r2 = indices[:, 0], indices[:, 1]
    
    return values + F * (x_pbest - values) + F * (values[r1] - values[r2])


# ==================== DE 交叉算子 ====================

def de_crossover_bin(
    target: torch.Tensor,
    donor: torch.Tensor,
    CR: float = 0.9
) -> torch.Tensor:
    """
    二项式交叉
    
    Args:
        target: (NP, D) 目标向量 (原始个体)
        donor: (NP, D) 供体向量 (变异产生)
        CR: 交叉概率
        
    Returns:
        (NP, D) 试验向量
    """
    NP, D = target.shape
    device = target.device
    
    # 生成交叉掩码
    mask = torch.rand(NP, D, device=device) < CR
    
    # 保证至少一维来自 donor (j_rand)
    j_rand = torch.randint(D, (NP,), device=device)
    mask.scatter_(1, j_rand.unsqueeze(1), True)
    
    return torch.where(mask, donor, target)


def de_crossover_exp(
    target: torch.Tensor,
    donor: torch.Tensor,
    CR: float = 0.9
) -> torch.Tensor:
    """
    指数交叉 (向量化实现)
    
    Args:
        target: (NP, D) 目标向量
        donor: (NP, D) 供体向量
        CR: 交叉概率
        
    Returns:
        (NP, D) 试验向量
    """
    NP, D = target.shape
    device = target.device
    
    # 起始点
    L = torch.randint(D, (NP,), device=device)
    
    # [GPU 优化] 向量化几何分布长度采样，消除 Python for 循环和 .item() 同步
    # 生成 D 个随机值，用 cumprod 找连续 < CR 的运行长度
    rand_vals = torch.rand(NP, D, device=device)
    continue_mask = (rand_vals < CR).float()
    # cumprod: 连续的 1 保持为 1，第一个 0 之后全部变 0
    run_lengths = torch.cumprod(continue_mask, dim=1).sum(dim=1).long()  # (NP,)
    
    # 创建环形掩码: 从起始点 L 开始的连续 run_lengths 个位置
    positions = torch.arange(D, device=device).unsqueeze(0)  # (1, D)
    circular_pos = (positions - L.unsqueeze(1)) % D  # (NP, D)
    mask = circular_pos < run_lengths.unsqueeze(1)
    
    return torch.where(mask, donor, target)


# ==================== 选择算子 ====================

def select_survivors_greedy(
    pop_values: torch.Tensor,
    pop_fitness: torch.Tensor,
    trial_values: torch.Tensor,
    trial_fitness: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    贪婪选择: 逐个比较，保留更优者
    
    Args:
        pop_values: (NP, D) 当前种群
        pop_fitness: (NP,) 当前适应度
        trial_values: (NP, D) 试验个体
        trial_fitness: (NP,) 试验适应度
        
    Returns:
        (new_values, new_fitness)
    """
    improved = trial_fitness < pop_fitness
    
    new_values = torch.where(
        improved.unsqueeze(1).expand_as(pop_values),
        trial_values,
        pop_values
    )
    new_fitness = torch.where(improved, trial_fitness, pop_fitness)
    
    return new_values, new_fitness


def select_best_n(
    values: torch.Tensor,
    fitness: torch.Tensor,
    n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    选择最优的 n 个个体
    
    Args:
        values: (NP, D) 种群
        fitness: (NP,) 适应度
        n: 选择数量
        
    Returns:
        (selected_values, selected_fitness)
    """
    indices = fitness.argsort()[:n]
    return values[indices], fitness[indices]


# ==================== 子空间操作 (用于 CC) ====================

def apply_subspace_mutation(
    full_values: torch.Tensor,
    indices: torch.Tensor,
    F: float = 0.5,
    strategy: str = 'rand1'
) -> torch.Tensor:
    """
    在子空间上应用变异
    
    Args:
        full_values: (NP, D) 完整种群
        indices: (groupsize,) 子空间维度索引
        F: 缩放因子
        strategy: 变异策略
        
    Returns:
        (NP, D) 变异后的完整向量
    """
    NP = full_values.shape[0]
    device = full_values.device
    
    # 提取子空间
    sub_values = full_values[:, indices]
    
    # 应用变异
    if strategy == 'rand1':
        sub_donor = de_mutation_rand1(sub_values, F)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 写回
    result = full_values.clone()
    result[:, indices] = sub_donor
    
    return result


def apply_subspace_crossover(
    target_full: torch.Tensor,
    donor_full: torch.Tensor,
    indices: torch.Tensor,
    CR: float = 0.9
) -> torch.Tensor:
    """
    在子空间上应用交叉
    
    Args:
        target_full: (NP, D) 目标完整向量
        donor_full: (NP, D) 供体完整向量
        indices: (groupsize,) 子空间维度索引
        CR: 交叉概率
        
    Returns:
        (NP, D) 交叉后的完整向量
    """
    # 提取子空间
    target_sub = target_full[:, indices]
    donor_sub = donor_full[:, indices]
    
    # 应用交叉
    trial_sub = de_crossover_bin(target_sub, donor_sub, CR)
    
    # 写回
    result = target_full.clone()
    result[:, indices] = trial_sub
    
    return result


# ==================== 并行 DE 算子 (用于增强型并行 CC) ====================

def de_mutation_rand1_parallel(
    values: torch.Tensor,
    F: float = 0.5
) -> torch.Tensor:
    """
    并行 DE/rand/1 变异 (支持 N_Groups 维度)
    
    Args:
        values: (N_Groups, NP, D) 或 (NP, D) 种群决策变量
        F: 缩放因子
        
    Returns:
        同形状的变异向量
    """
    if values.dim() == 2:
        return de_mutation_rand1(values, F)
    
    # (N_Groups, NP, D)
    N_Groups, NP, D = values.shape
    device = values.device
    
    # 为每个 Group 独立生成索引 (N_Groups, NP, 3)
    offsets = torch.randint(1, NP, (N_Groups, NP, 3), device=device)
    base = torch.arange(NP, device=device).view(1, NP, 1).expand(N_Groups, NP, 3)
    indices = (base + offsets) % NP
    
    # 使用高级索引提取
    g_idx = torch.arange(N_Groups, device=device).view(-1, 1, 1).expand_as(indices)
    r1 = values[g_idx[:,:,0], indices[:,:,0]]  # (N_Groups, NP, D)
    r2 = values[g_idx[:,:,1], indices[:,:,1]]
    r3 = values[g_idx[:,:,2], indices[:,:,2]]
    
    return r1 + F * (r2 - r3)


def de_mutation_current_to_pbest1_parallel(
    values: torch.Tensor,
    fitness: torch.Tensor,
    F: float = 0.5,
    p: float = 0.1
) -> torch.Tensor:
    """
    并行 DE/current-to-pbest/1 变异 (支持 N_Groups 维度)
    
    Args:
        values: (N_Groups, NP, D) 种群决策变量
        fitness: (NP,) 全局适应度 (用于选择 pbest)
        F: 缩放因子
        p: top-p 比例
        
    Returns:
        (N_Groups, NP, D) 变异向量
    """
    if values.dim() == 2:
        return de_mutation_current_to_pbest1(values, fitness, F, p)
    
    N_Groups, NP, D = values.shape
    device = values.device
    
    # 找 top-p 个体 (基于全局 fitness)
    k = max(1, int(NP * p))
    sorted_idx = fitness.argsort()[:k]
    
    # 为每个 Group 的每个个体随机选择一个 pbest
    pbest_choice = torch.randint(0, k, (N_Groups, NP), device=device)
    pbest_idx = sorted_idx[pbest_choice]  # (N_Groups, NP)
    
    # 提取 pbest (需要从 values 的对应位置取)
    g_idx = torch.arange(N_Groups, device=device).view(-1, 1).expand(N_Groups, NP)
    x_pbest = values[g_idx, pbest_idx]  # (N_Groups, NP, D)
    
    # 生成 2 个随机索引
    offsets = torch.randint(1, NP, (N_Groups, NP, 2), device=device)
    base = torch.arange(NP, device=device).view(1, NP, 1).expand(N_Groups, NP, 2)
    indices = (base + offsets) % NP
    
    r1 = values[g_idx.unsqueeze(-1).expand(-1,-1,2)[:,:,0], indices[:,:,0]]
    r2 = values[g_idx.unsqueeze(-1).expand(-1,-1,2)[:,:,1], indices[:,:,1]]
    
    return values + F * (x_pbest - values) + F * (r1 - r2)


def de_crossover_bin_parallel(
    target: torch.Tensor,
    donor: torch.Tensor,
    CR: float = 0.9
) -> torch.Tensor:
    """
    并行二项式交叉 (支持 N_Groups 维度)
    
    Args:
        target: (N_Groups, NP, D) 目标向量
        donor: (N_Groups, NP, D) 供体向量
        CR: 交叉概率
        
    Returns:
        (N_Groups, NP, D) 试验向量
    """
    if target.dim() == 2:
        return de_crossover_bin(target, donor, CR)
    
    N_Groups, NP, D = target.shape
    device = target.device
    
    # 生成交叉掩码
    mask = torch.rand(N_Groups, NP, D, device=device) < CR
    
    # 保证每个个体至少一维来自 donor
    j_rand = torch.randint(D, (N_Groups, NP), device=device)
    # 创建 one-hot 并 scatter
    j_rand_mask = torch.zeros(N_Groups, NP, D, dtype=torch.bool, device=device)
    j_rand_mask.scatter_(2, j_rand.unsqueeze(-1), True)
    mask = mask | j_rand_mask
    
    return torch.where(mask, donor, target)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing vectorized DE operators...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    NP, D = 50, 100
    values = torch.rand(NP, D, device=device) * 200 - 100
    fitness = (values ** 2).sum(dim=-1)
    
    # 测试变异算子
    print("\n--- Mutation Operators ---")
    
    donor_rand1 = de_mutation_rand1(values, F=0.5)
    assert donor_rand1.shape == (NP, D), "rand1 shape mismatch"
    print(f"DE/rand/1: shape={donor_rand1.shape}")
    
    donor_best1 = de_mutation_best1(values, fitness, F=0.5)
    assert donor_best1.shape == (NP, D), "best1 shape mismatch"
    print(f"DE/best/1: shape={donor_best1.shape}")
    
    donor_ctb1 = de_mutation_current_to_best1(values, fitness, F=0.5)
    assert donor_ctb1.shape == (NP, D), "current-to-best/1 shape mismatch"
    print(f"DE/current-to-best/1: shape={donor_ctb1.shape}")
    
    donor_ctpb1 = de_mutation_current_to_pbest1(values, fitness, F=0.5, p=0.1)
    assert donor_ctpb1.shape == (NP, D), "current-to-pbest/1 shape mismatch"
    print(f"DE/current-to-pbest/1: shape={donor_ctpb1.shape}")
    
    # 测试交叉算子
    print("\n--- Crossover Operators ---")
    
    trial_bin = de_crossover_bin(values, donor_rand1, CR=0.9)
    assert trial_bin.shape == (NP, D), "bin crossover shape mismatch"
    # 验证至少有一个维度来自 donor
    diff = (trial_bin != values).sum(dim=-1)
    assert (diff >= 1).all(), "bin crossover should change at least 1 dim"
    print(f"Binomial crossover: shape={trial_bin.shape}, min_changes={diff.min().item()}")
    
    # 测试选择
    print("\n--- Selection Operators ---")
    
    trial_fitness = (trial_bin ** 2).sum(dim=-1)
    new_values, new_fitness = select_survivors_greedy(
        values, fitness, trial_bin, trial_fitness
    )
    improved_count = (new_fitness < fitness).sum().item()
    print(f"Greedy selection: {improved_count}/{NP} improved")
    
    # 测试子空间操作
    print("\n--- Subspace Operations ---")
    
    indices = torch.arange(10, device=device)  # 前 10 维
    sub_mutant = apply_subspace_mutation(values, indices, F=0.5)
    # 检查只有指定维度变化
    unchanged_mask = torch.ones(D, dtype=torch.bool, device=device)
    unchanged_mask[indices] = False
    assert (sub_mutant[:, unchanged_mask] == values[:, unchanged_mask]).all(), \
        "Subspace mutation should not change other dims"
    print("Subspace mutation: only specified dims changed")
    
    # 性能测试
    print("\n--- Performance Test ---")
    import time
    
    # 大规模测试
    NP_large, D_large = 1000, 1000
    values_large = torch.rand(NP_large, D_large, device=device)
    fitness_large = (values_large ** 2).sum(dim=-1)
    
    # 预热
    _ = de_mutation_current_to_pbest1(values_large, fitness_large, F=0.5, p=0.1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        donor = de_mutation_current_to_pbest1(values_large, fitness_large, F=0.5, p=0.1)
        trial = de_crossover_bin(values_large, donor, CR=0.9)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"DE cycle (mutation + crossover) on {NP_large}x{D_large}:")
    print(f"  Total: {elapsed*1000:.2f} ms for {n_iters} iterations")
    print(f"  Per iteration: {elapsed*1000/n_iters:.3f} ms")
    
    print("\nOK: All vectorized operator tests passed!")
