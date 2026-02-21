"""
DECC GPU 优化器

GPU 向量化版本的 DECC (Differential Evolution with Cooperative Coevolution)。
所有计算在 GPU 上执行，无 CPU-GPU 数据迁移。

使用方法:
    decc = DECC_GPU(problem, allgroups, device='cuda')
    best_fitness = decc.run()
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple, Union

from gpu_population import GPUPopulation
from operators_vectorized import (
    de_mutation_current_to_pbest1,
    de_crossover_bin,
    select_survivors_greedy,
    # 并行算子
    de_mutation_current_to_pbest1_parallel,
    de_crossover_bin_parallel
)


# ==================== Top-k 精英档案 (用于随机上下文策略) ====================

class EliteArchive:
    """
    Top-k 精英档案
    
    用于并行 CC 的随机上下文策略，缓解评估偏差问题。
    参考: "并行协同进化算法中的震荡抑制与评估偏差修正"
    """
    
    def __init__(self, k: int, D: int, device: torch.device):
        """
        初始化精英档案
        
        Args:
            k: 精英数量
            D: 问题维度
            device: 计算设备
        """
        self.k = k
        self.D = D
        self.device = device
        
        self.solutions = torch.zeros(k, D, device=device)
        self.fitness = torch.full((k,), float('inf'), device=device)
    
    def update_from_population(self, values: torch.Tensor, fitness: torch.Tensor):
        """
        从种群中更新精英档案
        
        Args:
            values: (NP, D) 种群
            fitness: (NP,) 适应度
        """
        # 合并当前精英和新种群
        all_solutions = torch.cat([self.solutions, values], dim=0)
        all_fitness = torch.cat([self.fitness, fitness], dim=0)
        
        # 选择最优的 k 个
        _, best_indices = all_fitness.topk(self.k, largest=False)
        self.solutions = all_solutions[best_indices]
        self.fitness = all_fitness[best_indices]
    
    def sample_context(self, shape: tuple) -> torch.Tensor:
        """
        随机采样上下文向量
        
        Args:
            shape: 输出形状 (不含 D 维度)
            
        Returns:
            (*shape, D) 随机上下文向量
        """
        idx = torch.randint(0, self.k, shape, device=self.device)
        return self.solutions[idx]
    
    def get_best(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前最优解"""
        best_idx = self.fitness.argmin()
        return self.solutions[best_idx], self.fitness[best_idx]


class DECC_GPU:
    """
    GPU 向量化 DECC 优化器
    
    特点:
    1. 全程 GPU 计算，无 CPU-GPU 迁移
    2. 向量化 DE 算子，无 Python 循环
    3. 支持动态重分组回调
    """
    
    def __init__(
        self,
        problem: nn.Module,
        allgroups: List[List[int]],
        NP: int = 50,
        D: int = None,
        lb: float = -100.0,
        ub: float = 100.0,
        F: float = 0.5,
        CR: float = 0.9,
        p: float = 0.1,
        Max_FEs: int = 300000,
        device: Union[str, torch.device] = 'cuda',
        regroup_callback: Optional[Callable] = None,
        regroup_interval: int = 50000,
        verbose: bool = False
    ):
        """
        初始化 DECC 优化器
        
        Args:
            problem: 问题实例 (nn.Module，实现 evaluate 方法)
            allgroups: 分组信息 [[group1_indices], [group2_indices], ...]
            NP: 种群大小
            D: 问题维度 (自动推断)
            lb, ub: 边界
            F: DE 缩放因子
            CR: DE 交叉概率
            p: pbest 比例 (用于 current-to-pbest 变异)
            Max_FEs: 最大函数评估次数
            device: 计算设备
            regroup_callback: 重分组回调函数
            regroup_interval: 重分组间隔 (FEs)
            verbose: 是否打印进度
        """
        self.device = torch.device(device)
        self.problem = problem.to(self.device)
        
        # 分组信息
        self.allgroups = allgroups
        self.n_groups = len(allgroups)
        
        # 推断维度
        if D is None:
            D = max(max(g) for g in allgroups) + 1
        self.D = D
        
        # 将 allgroups 转换为 tensor 列表
        self.group_indices = [
            torch.tensor(g, dtype=torch.long, device=self.device)
            for g in allgroups
        ]
        
        # 参数
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.F = F
        self.CR = CR
        self.p = p
        self.Max_FEs = Max_FEs
        self.verbose = verbose
        
        # 重分组
        self.regroup_callback = regroup_callback
        self.regroup_interval = regroup_interval
        self.last_regroup_FEs = 0
        
        # 初始化种群
        self.pop = GPUPopulation(NP, D, device=self.device, lb=lb, ub=ub)
        
        # 评估计数
        self.FEs = 0
        
        # 历史记录
        self.best_history = []
        
        # [并行 CC] Top-k 精英档案
        self.elite_archive = EliteArchive(k=5, D=D, device=self.device)
        
        # [并行 CC] 准备并行数据结构
        self._prepare_parallel_structures()
    
    def _prepare_parallel_structures(self):
        """
        准备并行 CC 所需的数据结构
        
        - 统一子问题维度 (Padding)
        - 构建索引矩阵和掩码
        """
        # 找到最大子问题维度
        self.max_subdim = max(len(g) for g in self.allgroups)
        
        # 检查是否所有组大小相同 (可以用更快的批量操作)
        groupsizes = [len(g) for g in self.allgroups]
        self.uniform_groups = len(set(groupsizes)) == 1
        
        # 创建填充后的索引矩阵 (N_Groups, max_subdim)
        self.padded_indices = torch.zeros(
            self.n_groups, self.max_subdim, dtype=torch.long, device=self.device
        )
        self.valid_mask = torch.zeros(
            self.n_groups, self.max_subdim, dtype=torch.bool, device=self.device
        )
        
        for g_idx, indices in enumerate(self.group_indices):
            size = len(indices)
            self.padded_indices[g_idx, :size] = indices
            self.valid_mask[g_idx, :size] = True
    
    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        评估适应度并更新 FEs
        
        Args:
            x: (NP, D) 决策变量
            
        Returns:
            (NP,) 适应度
        """
        fitness = self.problem.evaluate(x)
        self.FEs += x.shape[0]
        return fitness
    
    def _optimize_subproblem(self, group_idx: int):
        """
        优化单个子问题
        
        Args:
            group_idx: 子问题索引
        """
        indices = self.group_indices[group_idx]
        groupsize = len(indices)
        
        # 提取子空间
        sub_values = self.pop.values[:, indices]  # (NP, groupsize)
        
        # 变异: current-to-pbest/1
        sub_donor = de_mutation_current_to_pbest1(
            sub_values, 
            self.pop.fitness,
            F=self.F,
            p=self.p
        )
        
        # 交叉
        sub_trial = de_crossover_bin(sub_values, sub_donor, CR=self.CR)
        
        # 边界裁剪
        sub_trial = torch.clamp(sub_trial, self.lb, self.ub)
        
        # 构建完整试验向量
        trial_full = self.pop.values.clone()
        trial_full[:, indices] = sub_trial
        
        # 评估
        trial_fitness = self._evaluate(trial_full)
        
        # 贪婪选择
        self.pop.update_where_improved(trial_full, trial_fitness)
    
    def _round_robin(self):
        """
        轮询优化所有子问题（一个 cycle）
        [遗留方法，用于对比]
        """
        for g_idx in range(self.n_groups):
            self._optimize_subproblem(g_idx)
            
            # 检查是否需要重分组
            if self.regroup_callback is not None:
                if self.FEs - self.last_regroup_FEs >= self.regroup_interval:
                    self._do_regroup()
            
            # 检查终止条件
            if self.FEs >= self.Max_FEs:
                break
    
    def _parallel_optimize_all_groups(self):
        """
        [增强型并行 CC] 并行优化所有子问题 (一次 CUDA kernel)
        
        特点:
        1. 所有 N_Groups 子问题同时处理
        2. 使用 Top-k 随机上下文策略缓解评估偏差
        3. 全量评估避免串行依赖
        """
        # 1. 提取所有子空间 (N_Groups, NP, max_subdim)
        # 使用 padded_indices 批量索引
        all_sub = self.pop.values[:, self.padded_indices]  # (NP, N_Groups, max_subdim)
        all_sub = all_sub.permute(1, 0, 2)  # (N_Groups, NP, max_subdim)
        
        # 2. 并行变异 (N_Groups, NP, max_subdim)
        all_donor = de_mutation_current_to_pbest1_parallel(
            all_sub,
            self.pop.fitness,
            F=self.F,
            p=self.p
        )
        
        # 3. 并行交叉
        all_trial_sub = de_crossover_bin_parallel(all_sub, all_donor, CR=self.CR)
        all_trial_sub = torch.clamp(all_trial_sub, self.lb, self.ub)
        
        # 4. 构建完整向量 (N_Groups, NP, D)
        #    使用 Top-k 随机上下文
        context = self.elite_archive.sample_context((self.n_groups, self.NP))  # (N_Groups, NP, D)
        full_trial = context.clone()
        
        # 5. [GPU 优化] 使用 scatter_ 批量写回子空间，消除 for 循环
        #    padded_indices: (N_Groups, max_subdim) → 扩展为 (N_Groups, NP, max_subdim)
        write_indices = self.padded_indices.unsqueeze(1).expand(-1, self.NP, -1)
        write_mask = self.valid_mask.unsqueeze(1).expand(-1, self.NP, -1)
        # 获取 context 在 padded_indices 位置的原值（用于保护 padding 位）
        original_at_indices = torch.gather(full_trial, dim=2, index=write_indices)
        # 有效位用 trial_sub，无效位保留 context 原值
        safe_trial_sub = torch.where(write_mask, all_trial_sub, original_at_indices)
        full_trial.scatter_(dim=2, index=write_indices, src=safe_trial_sub)
        
        # 6. 全量评估 (N_Groups * NP,)
        flat_trial = full_trial.view(-1, self.D)
        flat_fitness = self._evaluate(flat_trial)
        all_fitness = flat_fitness.view(self.n_groups, self.NP)
        
        # 7. 并行选择：找到每个个体在所有组中的最佳结果
        #    取每个 NP 位置的最小 fitness 对应的组的解
        best_group_per_individual = all_fitness.argmin(dim=0)  # (NP,)
        best_fitness_per_individual = all_fitness.min(dim=0).values  # (NP,)
        
        # 构建最终的 trial_full
        np_idx = torch.arange(self.NP, device=self.device)
        best_trial = full_trial[best_group_per_individual, np_idx]  # (NP, D)
        
        # 8. 贪婪选择
        self.pop.update_where_improved(best_trial, best_fitness_per_individual)
        
        # 9. 更新精英档案
        self.elite_archive.update_from_population(self.pop.values, self.pop.fitness)
    
    def _do_regroup(self):
        """
        执行重分组
        """
        if self.regroup_callback is None:
            return
        
        # 获取种群快照
        snapshot = self.get_population_snapshot()
        
        # 调用回调获取新分组
        new_groups = self.regroup_callback(snapshot, self.pop.values, self.pop.fitness)
        
        if new_groups is not None:
            self.allgroups = new_groups
            self.n_groups = len(new_groups)
            self.group_indices = [
                torch.tensor(g, dtype=torch.long, device=self.device)
                for g in new_groups
            ]
        
        self.last_regroup_FEs = self.FEs
    
    def run(self, parallel: bool = True) -> float:
        """
        运行优化
        
        Args:
            parallel: 是否使用并行 CC (默认 True，GPU 高效模式)
        
        Returns:
            最优适应度值
        """
        # 初始评估
        self.pop.evaluate(self.problem)
        self.FEs = self.NP
        
        # 初始化精英档案
        self.elite_archive.update_from_population(self.pop.values, self.pop.fitness)
        
        best_fit, _ = self.pop.best()
        self.best_history.append(best_fit.item())
        
        if self.verbose:
            mode = "Parallel" if parallel else "Sequential"
            print(f"[DECC_GPU] Initial: FEs={self.FEs}, Best={best_fit.item():.4e}, Mode={mode}")
        
        # 主循环
        cycle = 0
        while self.FEs < self.Max_FEs:
            if parallel:
                self._parallel_optimize_all_groups()
            else:
                self._round_robin()
            cycle += 1
            
            best_fit, _ = self.pop.best()
            self.best_history.append(best_fit.item())
            
            if self.verbose and cycle % 10 == 0:
                print(f"[DECC_GPU] Cycle {cycle}: FEs={self.FEs}, Best={best_fit.item():.4e}")
        
        best_fit, _ = self.pop.best()
        
        if self.verbose:
            print(f"[DECC_GPU] Final: FEs={self.FEs}, Best={best_fit.item():.4e}")
        
        return best_fit.item()
    
    def get_population_snapshot(self) -> torch.Tensor:
        """
        获取种群快照（用于 LandscapeEncoder）
        
        Returns:
            (n_groups, NP, groupsize+1) 张量
        """
        return self.pop.to_snapshot(self.allgroups, include_fitness=True)
    
    def get_best(self) -> Tuple[torch.Tensor, float]:
        """
        获取当前最优解
        
        Returns:
            (best_values, best_fitness)
        """
        best_fit, best_idx = self.pop.best()
        return self.pop.values[best_idx].clone(), best_fit.item()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing DECC_GPU...")
    
    import time
    from gpu_problem import GPUWeightedElliptic, generate_rotation_matrix
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    # 设置问题
    n_groups = 10
    groupsize = 100
    D = n_groups * groupsize
    NP = 50
    
    allgroups = [list(range(i * groupsize, (i + 1) * groupsize)) for i in range(n_groups)]
    weights = torch.ones(n_groups, device=device)
    xopt = torch.zeros(D, device=device)
    R100 = torch.tensor(generate_rotation_matrix(100), device=device)  # 使用全局默认精度
    
    problem = GPUWeightedElliptic(allgroups, weights, xopt, R100).to(device)
    
    # 创建优化器
    decc = DECC_GPU(
        problem=problem,
        allgroups=allgroups,
        NP=NP,
        D=D,
        Max_FEs=10000,  # 快速测试
        device=device,
        verbose=True
    )
    
    print(f"\nProblem: D={D}, NP={NP}, n_groups={n_groups}")
    
    # 运行优化
    start = time.perf_counter()
    best_fitness = decc.run()
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  Best fitness: {best_fitness:.4e}")
    print(f"  Total FEs: {decc.FEs}")
    print(f"  Time: {elapsed:.2f} s")
    print(f"  Speed: {decc.FEs / elapsed:.0f} FEs/s")
    
    # 测试快照
    snapshot = decc.get_population_snapshot()
    assert snapshot.shape == (n_groups, NP, groupsize + 1), f"Snapshot shape mismatch: {snapshot.shape}"
    print(f"  Snapshot shape: {snapshot.shape}")
    
    # 性能对比 (更多 FEs)
    print("\n--- Performance Benchmark ---")
    
    decc_bench = DECC_GPU(
        problem=problem,
        allgroups=allgroups,
        NP=NP,
        D=D,
        Max_FEs=100000,
        device=device,
        verbose=False
    )
    
    start = time.perf_counter()
    _ = decc_bench.run()
    elapsed = time.perf_counter() - start
    
    print(f"100k FEs benchmark:")
    print(f"  Time: {elapsed:.2f} s")
    print(f"  Speed: {decc_bench.FEs / elapsed:.0f} FEs/s")
    
    print("\nOK: All DECC_GPU tests passed!")
