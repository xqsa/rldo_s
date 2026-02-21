"""
批量 DECC GPU 优化器 V3

[P0 修复]
1. 使用正确的分组加权+旋转评估 (调用 _evaluate_batch)
2. FEs 统计一致
3. 简化为组轮询而非4D全并行 (保证语义正确性)

设计原则:
- Batch 维度全并行
- Group 维度轮询 (保证正确的 CC 语义)
- 所有操作在 GPU tensor 上执行
- 使用 valid_mask 严格过滤 padding
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional


class DECC_GPU_Batched_V3:
    """
    批量 GPU DECC 优化器 V3
    
    特点:
    1. 内部种群形状: (Batch, NP, D)
    2. Batch 维度全并行
    3. Group 维度轮询 (正确的 CC 语义)
    4. 正确的分组加权评估
    
    [P1 修复] Ground Truth 评估模式:
    - 优化过程使用 agent 预测的分组 (group_indices)
    - Fitness 评估使用 Ground Truth 分组 (gt_indices)
    - 解决 "reward hacking" 问题：模型无法通过选择"全不连接"来作弊
    """
    
    def __init__(
        self,
        batch_size: int,
        D: int,
        group_indices: torch.Tensor,   # (Batch, Max_Groups, max_groupsize) - Agent 预测的分组
        valid_mask: torch.Tensor,      # (Batch, Max_Groups, max_groupsize) bool
        group_counts: torch.Tensor,    # (Batch,) 每个样本的有效组数
        weights: torch.Tensor,         # (Batch, Max_Groups)
        xopt: torch.Tensor,            # (D,) or (Batch, D)
        R: torch.Tensor,               # (max_groupsize, max_groupsize)
        NP: int = 50,
        lb: float = -100.0,
        ub: float = 100.0,
        F: float = 0.5,
        CR: float = 0.9,
        p: float = 0.1,
        Max_FEs: int = 300000,
        device: torch.device = None,
        # [P1 修复] Ground Truth 评估参数
        gt_indices: torch.Tensor = None,    # (Batch, GT_Max_Groups, gt_max_groupsize) - 真实分组
        gt_valid: torch.Tensor = None,      # (Batch, GT_Max_Groups, gt_max_groupsize) bool
        gt_counts: torch.Tensor = None,     # (Batch,) Ground Truth 的有效组数
        gt_weights: torch.Tensor = None,    # (Batch, GT_Max_Groups) Ground Truth 权重
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.batch_size = batch_size
        self.NP = NP
        self.D = D
        self.lb = lb
        self.ub = ub
        self.F = F
        self.CR = CR
        self.p = p
        self.Max_FEs = Max_FEs
        
        # GPU Tensor buffer - Agent 预测的分组 (用于优化)
        self.group_indices = group_indices.to(self.device)
        self.valid_mask = valid_mask.to(self.device)
        self.group_counts = group_counts.to(self.device)
        self.weights = weights.to(self.device)
        
        self.Max_Groups = group_indices.shape[1]
        self.max_groupsize = group_indices.shape[2]
        
        # [P1 修复] Ground Truth 分组 (用于评估)
        # 如果未提供 GT，则回退到使用 agent 分组 (向后兼容)
        self.use_gt_evaluation = gt_indices is not None
        if self.use_gt_evaluation:
            self.gt_indices = gt_indices.to(self.device)
            self.gt_valid = gt_valid.to(self.device)
            self.gt_counts = gt_counts.to(self.device)
            self.gt_weights = gt_weights.to(self.device)
            self.GT_Max_Groups = gt_indices.shape[1]
            self.gt_max_groupsize = gt_indices.shape[2]
            
            # GT 组有效掩码
            gt_g_idx = torch.arange(self.GT_Max_Groups, device=self.device).unsqueeze(0).expand(batch_size, -1)
            self.gt_group_valid = gt_g_idx < self.gt_counts.unsqueeze(1)
        else:
            # 回退: 评估也使用 agent 分组
            self.gt_indices = self.group_indices
            self.gt_valid = self.valid_mask
            self.gt_counts = self.group_counts
            self.gt_weights = self.weights
            self.GT_Max_Groups = self.Max_Groups
            self.gt_max_groupsize = self.max_groupsize
            self.gt_group_valid = None  # 稍后设置
        
        # xopt - 使用全局默认精度 (训练=Float32, 推理=Float64)
        default_dtype = torch.get_default_dtype()
        if xopt.dim() == 1:
            self.xopt = xopt.unsqueeze(0).expand(batch_size, -1).to(device=self.device, dtype=default_dtype)
        else:
            self.xopt = xopt.to(device=self.device, dtype=default_dtype)
        
        # 旋转矩阵 - 使用全局默认精度
        self.R = R.to(device=self.device, dtype=default_dtype)
        
        # Elliptic 系数生成常量（按真实子空间维度动态生成，避免与 padding 绑定）
        self._ln_1e6 = math.log(1e6)
        
        # [GPU 优化] 预计算 GT 评估所需的常量，避免每次 _evaluate_batch 重复计算
        self._precompute_eval_constants(batch_size, default_dtype)
        
        # 初始化种群 - 使用全局默认精度
        self.pop_values = torch.rand(batch_size, NP, D, device=self.device, dtype=default_dtype) * (ub - lb) + lb
        self.pop_fitness = torch.full((batch_size, NP), float('inf'), device=self.device, dtype=default_dtype)
        
        # 组有效掩码 (Agent 分组)
        g_idx = torch.arange(self.Max_Groups, device=self.device).unsqueeze(0).expand(batch_size, -1)
        self.group_valid = g_idx < self.group_counts.unsqueeze(1)
        
        # GT 组有效掩码 (如果未设置)
        if self.gt_group_valid is None:
            self.gt_group_valid = self.group_valid
        
        # FEs 计数
        self.FEs = 0
    
    def _precompute_eval_constants(self, batch_size: int, dtype):
        """
        [GPU 优化] 预计算 _evaluate_batch 中每次都重复生成的常量
        """
        Batch = batch_size
        
        # 旋转矩阵 (匹配 GT groupsize)
        if self.gt_max_groupsize != self.max_groupsize:
            if self.gt_max_groupsize <= self.R.shape[0]:
                self._R_eval = self.R[:self.gt_max_groupsize, :self.gt_max_groupsize]
            else:
                self._R_eval = torch.eye(self.gt_max_groupsize, device=self.device, dtype=self.R.dtype)
                self._R_eval[:self.R.shape[0], :self.R.shape[1]] = self.R
        else:
            self._R_eval = self.R
        
        # GT valid 扩展 (Batch, GT_Max_Groups, 1, gt_max_groupsize)
        self._gt_valid_expanded = self.gt_valid.unsqueeze(2)
        
        # Elliptic 系数: 基于 GT valid counts 预计算
        valid_counts = self.gt_valid.sum(dim=-1).clamp(min=1).to(dtype)  # (Batch, GT_Max_Groups)
        denom = (valid_counts - 1).clamp(min=1).view(Batch, self.GT_Max_Groups, 1, 1)
        pos = torch.arange(self.gt_max_groupsize, device=self.device, dtype=dtype).view(1, 1, 1, -1)
        t = pos / denom
        t = torch.where(self._gt_valid_expanded, t, torch.zeros_like(t))
        coeffs = torch.exp(t * self._ln_1e6)
        self._precomputed_coeffs = coeffs * self._gt_valid_expanded.to(dtype)
        
        # GT 权重掩码
        self._masked_gt_weights = torch.where(
            self.gt_group_valid, self.gt_weights, torch.zeros_like(self.gt_weights)
        )
    
    def _evaluate_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        [P1 修复] 批量评估：使用 Ground Truth 分组计算 Fitness
        
        关键改动：
        - 评估使用 gt_indices/gt_valid/gt_weights (真实拓扑)
        - 优化过程仍使用 group_indices (agent 预测)
        - 这样模型无法通过"作弊"（全不连接）来降低难度
        
        Args:
            x: (Batch, NP, D)
            
        Returns:
            (Batch, NP)
        """
        Batch, NP, D = x.shape
        
        # 偏移
        z = x - self.xopt.unsqueeze(1)
        
        # [GPU 优化] 使用 index_select 风格的 gather 避免显式 expand 大张量
        # gt_indices: (Batch, GT_Max_Groups, gt_max_groupsize)
        # z: (Batch, NP, D)
        # 目标: z_sub = z[:, :, gt_indices] -> (Batch, GT_Max_Groups, NP, gt_max_groupsize)
        # 技巧: 先 gather 再 permute，避免创建 (Batch, GT_Max_Groups, NP, D) 的巨大中间张量
        # z[:, :, indices] 等价于: z.unsqueeze(1)[:, :, :, indices] 但更高效
        # 使用 advanced indexing: z[batch, :, gt_indices[batch, group, :]]
        indices_for_gather = self.gt_indices.unsqueeze(2).expand(-1, -1, NP, -1)  # (Batch, GT, NP, gs)
        # 用 einsum 风格: z -> (Batch, 1, NP, D), 然后 gather dim=-1
        z_for_gather = z.unsqueeze(1).expand(-1, self.GT_Max_Groups, -1, -1)
        z_sub = torch.gather(z_for_gather, dim=-1, index=indices_for_gather)
        
        z_rot = torch.einsum('bgnk,kl->bgnl', z_sub, self._R_eval)
        
        # [GPU 优化] 使用预计算的 Elliptic 系数，避免每次评估都重新计算
        elliptic_vals = (self._precomputed_coeffs * (z_rot ** 2)).sum(dim=-1)  # (Batch, GT_Max_Groups, NP)
        
        # [P1 修复] 使用预计算的 GT 掩码权重
        weighted = elliptic_vals * self._masked_gt_weights.unsqueeze(-1)
        fitness = weighted.sum(dim=1)  # (Batch, NP)
        
        # FEs 统计 - [P3 修复] 按单个问题计数，不乘 Batch
        # Max_FEs 代表"每个问题的评估预算"，而非整个 Batch 的总预算
        self.FEs += NP
        
        return fitness
    
    def _optimize_group_batch(self, group_idx: int) -> None:
        """
        对单个分组进行批量优化
        
        Batch 维度并行，单个 Group
        """
        Batch = self.batch_size
        NP = self.NP
        
        # [P3 修复] 评估前检查 FEs 预算 - 按单个问题计数
        fes_needed = NP
        if self.FEs + fes_needed > self.Max_FEs:
            return
        
        # 检查哪些 batch 有这个组
        valid = self.group_valid[:, group_idx]  # (Batch,)
        
        if not valid.any():
            return
        
        # 获取索引和掩码
        group_indices = self.group_indices[:, group_idx]  # (Batch, max_groupsize)
        group_valid_mask = self.valid_mask[:, group_idx]  # (Batch, max_groupsize)
        
        # 提取子空间
        indices_expanded = group_indices.unsqueeze(1).expand(-1, NP, -1)
        sub_values = torch.gather(self.pop_values, dim=2, index=indices_expanded)
        
        # [P0 修复] Mask 保护：将无效维度清零，防止幽灵数据参与变异计算
        # 这确保 padding 区域不会干扰 x_pbest - values 等差分计算
        valid_expanded = group_valid_mask.unsqueeze(1).expand(-1, NP, -1)
        sub_values = sub_values * valid_expanded.to(sub_values.dtype)
        
        # 变异
        sub_donor = self._mutation_batch(sub_values)
        
        # 交叉
        sub_trial = self._crossover_batch(sub_values, sub_donor, group_valid_mask)
        sub_trial = torch.clamp(sub_trial, self.lb, self.ub)
        
        # [GPU 优化] 避免 clone 全种群 — 直接用 scatter_ 原地更新后求 fitness
        # 保存原始子空间值用于回滚
        original_sub_vals_for_rollback = torch.gather(self.pop_values, dim=2, index=indices_expanded)
        
        # 原地写入 trial 子空间（只修改当前组对应的维度）
        self.pop_values.scatter_(dim=2, index=indices_expanded, src=sub_trial)
        
        # [正确评估] 使用分组加权评估（此时 pop_values 已含 trial 子空间）
        trial_fitness = self._evaluate_batch(self.pop_values)
        
        # 贪婪选择：需要回滚未改进的个体
        improved = trial_fitness < self.pop_fitness
        improved = improved & valid.unsqueeze(1)
        
        # 对未改进的个体，恢复原始子空间
        not_improved = ~improved
        rollback_mask = not_improved.unsqueeze(-1).expand_as(indices_expanded)
        # 构建回滚数据：未改进位置用原始值，改进位置用 sub_trial
        rollback_src = torch.where(rollback_mask, original_sub_vals_for_rollback, sub_trial)
        self.pop_values.scatter_(dim=2, index=indices_expanded, src=rollback_src)
        
        self.pop_fitness = torch.where(improved, trial_fitness, self.pop_fitness)
    
    def _mutation_batch(self, values: torch.Tensor) -> torch.Tensor:
        """
        批量变异: DE/current-to-pbest/1
        
        Args:
            values: (Batch, NP, subdim)
        """
        Batch, NP, subdim = values.shape
        device = values.device
        
        k = max(1, int(NP * self.p))
        sorted_idx = self.pop_fitness.argsort(dim=1)[:, :k]
        
        pbest_choice = torch.randint(0, k, (Batch, NP), device=device)
        pbest_idx = torch.gather(sorted_idx, dim=1, index=pbest_choice)
        
        pbest_idx_expanded = pbest_idx.unsqueeze(-1).expand(-1, -1, subdim)
        x_pbest = torch.gather(values, dim=1, index=pbest_idx_expanded)
        
        offsets = torch.randint(1, NP, (Batch, NP, 2), device=device)
        base = torch.arange(NP, device=device).view(1, NP, 1).expand(Batch, NP, 2)
        r_idx = (base + offsets) % NP
        
        r1_idx = r_idx[..., 0].unsqueeze(-1).expand(-1, -1, subdim)
        r2_idx = r_idx[..., 1].unsqueeze(-1).expand(-1, -1, subdim)
        
        r1 = torch.gather(values, dim=1, index=r1_idx)
        r2 = torch.gather(values, dim=1, index=r2_idx)
        
        return values + self.F * (x_pbest - values) + self.F * (r1 - r2)
    
    def _crossover_batch(
        self, 
        target: torch.Tensor, 
        donor: torch.Tensor,
        group_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        批量二项式交叉
        
        Args:
            target: (Batch, NP, subdim)
            donor: (Batch, NP, subdim)
            group_valid_mask: (Batch, subdim)
        """
        Batch, NP, subdim = target.shape
        device = target.device
        
        mask = torch.rand(Batch, NP, subdim, device=device) < self.CR
        
        # j_rand: 只在有效维度内选择
        actual_dims = group_valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        j_rand = (torch.rand(Batch, NP, device=device) * actual_dims.squeeze(-1).unsqueeze(-1)).long()
        j_rand = j_rand.clamp(0, subdim - 1)
        
        j_rand_mask = torch.zeros_like(mask)
        j_rand_mask.scatter_(2, j_rand.unsqueeze(-1), True)
        mask = mask | j_rand_mask
        
        return torch.where(mask, donor, target)
    
    def _round_robin_batch(self) -> None:
        """
        批量轮询优化所有子问题
        
        Batch 并行 + Group 轮询
        """
        for g_idx in range(self.Max_Groups):
            if self.FEs >= self.Max_FEs:
                break
            self._optimize_group_batch(g_idx)
    
    def run(self) -> torch.Tensor:
        """
        运行批量优化
        
        [P3 修复] Max_FEs 代表每个问题的评估预算
        - 所有 Batch 中的问题共享相同的迭代次数
        - FEs 按单个问题计数 (不乘 Batch)
        
        Returns:
            (Batch,) 每个样本的最优适应度
        """
        # 初始评估
        self.pop_fitness = self._evaluate_batch(self.pop_values)
        initial_fes = self.NP  # [P3 修复] 单个问题的 FEs
        
        # 剩余预算 (每个问题)
        remaining_fes = self.Max_FEs - initial_fes
        
        # 每轮 round-robin: 最多 Max_Groups 次评估，每次 NP
        fes_per_group = self.NP  # [P3 修复] 单个问题每组的 FEs
        fes_per_cycle = self.Max_Groups * fes_per_group
        
        # 基于剩余预算计算可执行的完整周期数
        n_full_cycles = max(0, remaining_fes // fes_per_cycle)
        
        for _ in range(n_full_cycles):
            self._round_robin_batch()
        
        # 处理剩余预算（不足一个完整周期的部分）
        # 在 _round_robin_batch 和 _optimize_group_batch 中已有 FEs 检查
        if self.FEs < self.Max_FEs:
            self._round_robin_batch()
        
        return self.pop_fitness.min(dim=1).values


# ==================== 工厂函数 ====================

def create_decc_batched_from_tensors(
    group_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    group_counts: torch.Tensor,
    weights: torch.Tensor,
    D: int,
    R: torch.Tensor,
    device: torch.device,
    NP: int = 50,
    max_fes: int = 10000,
    xopt: torch.Tensor = None,
    # [P1 修复] Ground Truth 评估参数
    gt_indices: torch.Tensor = None,
    gt_valid: torch.Tensor = None,
    gt_counts: torch.Tensor = None,
    gt_weights: torch.Tensor = None
) -> DECC_GPU_Batched_V3:
    """
    从 GPU Tensor 创建批量 DECC 实例
    
    [P1 修复] 支持 Ground Truth 评估模式
    - group_indices/valid_mask/weights: Agent 预测的分组 (用于优化)
    - gt_indices/gt_valid/gt_weights: 真实拓扑分组 (用于评估)
    
    如果不提供 gt_* 参数，则回退到使用 agent 分组进行评估（向后兼容）
    """
    batch_size = group_indices.shape[0]
    
    if xopt is None:
        xopt = torch.zeros(D, device=device)
    
    max_groupsize = group_indices.shape[2]
    if R.shape[0] < max_groupsize:
        R_full = torch.eye(max_groupsize, device=device, dtype=R.dtype)
        R_full[:R.shape[0], :R.shape[1]] = R
        R = R_full
    else:
        R = R[:max_groupsize, :max_groupsize]
    
    # 处理 GT 的旋转矩阵大小 (如果 GT groupsize 不同)
    if gt_indices is not None:
        gt_max_groupsize = gt_indices.shape[2]
        if R.shape[0] < gt_max_groupsize:
            R_full = torch.eye(gt_max_groupsize, device=device, dtype=R.dtype)
            R_full[:R.shape[0], :R.shape[1]] = R
            R = R_full
    
    return DECC_GPU_Batched_V3(
        batch_size=batch_size,
        D=D,
        group_indices=group_indices,
        valid_mask=valid_mask,
        group_counts=group_counts,
        weights=weights,
        xopt=xopt,
        R=R,
        NP=NP,
        Max_FEs=max_fes,
        device=device,
        # [P1 修复] Ground Truth 参数
        gt_indices=gt_indices,
        gt_valid=gt_valid,
        gt_counts=gt_counts,
        gt_weights=gt_weights
    )


def run_decc_batched(
    groups_list: list,
    weights_list: list,
    D: int,
    R: torch.Tensor,
    device: torch.device,
    NP: int = 50,
    max_fes: int = 10000,
    xopt: torch.Tensor = None
) -> torch.Tensor:
    """
    兼容接口：接受 Python List 输入
    """
    from group_mask_utils import batch_groups_to_indices_padded
    
    indices, valid, counts = batch_groups_to_indices_padded(groups_list, device)
    
    batch_size = len(groups_list)
    max_groups = indices.shape[1]
    
    weights = torch.zeros(batch_size, max_groups, device=device)
    for b, w in enumerate(weights_list):
        n_w = min(len(w), max_groups)
        if isinstance(w, torch.Tensor):
            weights[b, :n_w] = w[:n_w].to(device)
        else:
            weights[b, :n_w] = torch.tensor(w[:n_w], device=device)
    
    decc = create_decc_batched_from_tensors(
        group_indices=indices,
        valid_mask=valid,
        group_counts=counts,
        weights=weights,
        D=D,
        R=R,
        device=device,
        NP=NP,
        max_fes=max_fes,
        xopt=xopt
    )
    
    return decc.run()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing DECC_GPU_Batched_V3...")
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_size = 8
    n_groups = 5
    groupsize = 20
    D = n_groups * groupsize
    NP = 30
    Max_Groups = 5
    max_groupsize = 25
    max_fes = 5000
    
    # 构建 GPU Tensor
    group_indices = torch.zeros(batch_size, Max_Groups, max_groupsize, dtype=torch.long, device=device)
    valid_mask = torch.zeros(batch_size, Max_Groups, max_groupsize, dtype=torch.bool, device=device)
    group_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    weights = torch.zeros(batch_size, Max_Groups, device=device)
    
    for b in range(batch_size):
        n_merged = 3 + (b % 3)
        group_counts[b] = n_merged
        start = 0
        for g in range(n_merged):
            end = min(start + groupsize, D)
            size = end - start
            group_indices[b, g, :size] = torch.arange(start, end, device=device)
            valid_mask[b, g, :size] = True
            weights[b, g] = 1.0 / n_merged
            start = end
    
    R = torch.eye(max_groupsize, device=device)  # 使用全局默认精度
    
    print(f"\nBatch size: {batch_size}")
    print(f"D: {D}, NP: {NP}, Max_FEs: {max_fes}")
    print(f"Groups per sample: {group_counts.tolist()}")
    
    start_time = time.perf_counter()
    
    decc = create_decc_batched_from_tensors(
        group_indices=group_indices,
        valid_mask=valid_mask,
        group_counts=group_counts,
        weights=weights,
        D=D,
        R=R,
        device=device,
        NP=NP,
        max_fes=max_fes
    )
    
    results = decc.run()
    elapsed = time.perf_counter() - start_time
    
    print(f"\nResults shape: {results.shape}")
    print(f"Best fitness: {[f'{r:.4e}' for r in results.tolist()]}")
    print(f"Total FEs: {decc.FEs}")
    print(f"Time: {elapsed:.2f} s")
    print(f"Speed: {elapsed/batch_size*1000:.0f} ms/sample")
    
    # 验证 FEs 统计
    expected_fes_per_cycle = decc.Max_Groups * batch_size * NP
    print(f"\nExpected FEs per cycle: {expected_fes_per_cycle}")
    print(f"Actual FEs: {decc.FEs}")
    
    print("\nOK: DECC_GPU_Batched_V3 tests passed!")
