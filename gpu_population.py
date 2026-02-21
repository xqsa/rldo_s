"""
GPU 种群容器

参考 EvoTorch SolutionBatch 设计，封装 GPU 上的种群管理。
确保数据始终留在 GPU，避免不必要的设备迁移。

使用方法:
    pop = GPUPopulation(NP=50, D=100, device='cuda')
    pop.evaluate(problem)  # 批量评估
    best_val, best_idx = pop.best()
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List


class GPUPopulation:
    """
    GPU 种群容器，参考 EvoTorch SolutionBatch 设计
    
    核心原则:
    1. 数据永远留在指定设备上，避免 CPU-GPU 迁移
    2. 所有操作向量化，无 Python 循环
    3. 提供类似 ndarray 的切片接口
    
    Attributes:
        values: (NP, D) 决策变量张量
        fitness: (NP,) 适应度张量
        device: 计算设备
    """
    
    def __init__(
        self, 
        NP: int, 
        D: int, 
        device: Union[str, torch.device] = 'cuda',
        lb: float = -100.0,
        ub: float = 100.0,
        dtype: torch.dtype = torch.float64
    ):
        """
        初始化种群
        
        Args:
            NP: 种群大小
            D: 决策变量维度
            device: 计算设备 ('cuda', 'cpu', etc.)
            lb: 下界
            ub: 上界
            dtype: 数据类型
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self._lb = lb
        self._ub = ub
        self._NP = NP
        self._D = D
        
        # 初始化决策变量 (均匀采样)
        self.values = torch.empty(NP, D, device=self.device, dtype=dtype).uniform_(lb, ub)
        
        # 初始化适应度 (未评估状态)
        self.fitness = torch.full((NP,), float('inf'), device=self.device, dtype=dtype)
        
        # 评估标志
        self._evaluated = torch.zeros(NP, dtype=torch.bool, device=self.device)
    
    @property
    def NP(self) -> int:
        """种群大小"""
        return self._NP
    
    @property
    def D(self) -> int:
        """决策变量维度"""
        return self._D
    
    @property
    def lb(self) -> float:
        """下界"""
        return self._lb
    
    @property
    def ub(self) -> float:
        """上界"""
        return self._ub
    
    def __len__(self) -> int:
        return self._NP
    
    def __getitem__(self, idx) -> 'GPUPopulation':
        """
        切片访问，返回新的 GPUPopulation 视图
        
        Args:
            idx: 索引或切片
            
        Returns:
            包含选中个体的新 GPUPopulation
        """
        new_pop = GPUPopulation.__new__(GPUPopulation)
        new_pop.device = self.device
        new_pop.dtype = self.dtype
        new_pop._lb = self._lb
        new_pop._ub = self._ub
        
        new_pop.values = self.values[idx]
        new_pop.fitness = self.fitness[idx]
        new_pop._evaluated = self._evaluated[idx]
        
        if new_pop.values.dim() == 1:
            new_pop.values = new_pop.values.unsqueeze(0)
            new_pop.fitness = new_pop.fitness.unsqueeze(0)
            new_pop._evaluated = new_pop._evaluated.unsqueeze(0)
        
        new_pop._NP = new_pop.values.shape[0]
        new_pop._D = new_pop.values.shape[1]
        
        return new_pop
    
    def clone(self) -> 'GPUPopulation':
        """
        深拷贝种群
        
        Returns:
            新的独立 GPUPopulation 实例
        """
        new_pop = GPUPopulation.__new__(GPUPopulation)
        new_pop.device = self.device
        new_pop.dtype = self.dtype
        new_pop._lb = self._lb
        new_pop._ub = self._ub
        new_pop._NP = self._NP
        new_pop._D = self._D
        
        new_pop.values = self.values.clone()
        new_pop.fitness = self.fitness.clone()
        new_pop._evaluated = self._evaluated.clone()
        
        return new_pop
    
    def to(self, device: Union[str, torch.device]) -> 'GPUPopulation':
        """
        迁移到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self (原地修改)
        """
        self.device = torch.device(device)
        self.values = self.values.to(device)
        self.fitness = self.fitness.to(device)
        self._evaluated = self._evaluated.to(device)
        return self
    
    def evaluate(self, problem: nn.Module) -> torch.Tensor:
        """
        使用问题批量评估所有个体
        
        Args:
            problem: 实现 evaluate(x) -> fitness 的问题模块
            
        Returns:
            (NP,) 适应度张量
        """
        self.fitness = problem.evaluate(self.values)
        self._evaluated.fill_(True)
        return self.fitness
    
    def set_fitness(self, fitness: torch.Tensor):
        """
        直接设置适应度值
        
        Args:
            fitness: (NP,) 适应度张量
        """
        self.fitness = fitness.to(self.device, self.dtype)
        self._evaluated.fill_(True)
    
    def best(self) -> Tuple[torch.Tensor, int]:
        """
        获取最优个体
        
        Returns:
            (best_fitness, best_index)
        """
        best_idx = self.fitness.argmin().item()
        return self.fitness[best_idx], best_idx
    
    def best_values(self) -> torch.Tensor:
        """
        获取最优个体的决策变量
        
        Returns:
            (D,) 最优解张量
        """
        _, best_idx = self.best()
        return self.values[best_idx]
    
    def argsort(self, descending: bool = False) -> torch.Tensor:
        """
        按适应度排序的索引
        
        Args:
            descending: 是否降序
            
        Returns:
            (NP,) 排序索引
        """
        return self.fitness.argsort(descending=descending)
    
    def topk(self, k: int) -> 'GPUPopulation':
        """
        获取最优的 k 个个体
        
        Args:
            k: 选取数量
            
        Returns:
            包含 top-k 个体的新 GPUPopulation
        """
        indices = self.argsort()[:k]
        return self[indices]
    
    def clamp(self, lb: Optional[float] = None, ub: Optional[float] = None):
        """
        边界裁剪 (原地操作)
        
        Args:
            lb: 下界 (默认使用初始化时的 lb)
            ub: 上界 (默认使用初始化时的 ub)
        """
        if lb is None:
            lb = self._lb
        if ub is None:
            ub = self._ub
        self.values.clamp_(lb, ub)
    
    def update_where_improved(
        self, 
        trial_values: torch.Tensor, 
        trial_fitness: torch.Tensor
    ):
        """
        贪婪更新：仅当 trial 优于当前时更新
        
        Args:
            trial_values: (NP, D) 候选解
            trial_fitness: (NP,) 候选适应度
        """
        improved = trial_fitness < self.fitness
        self.values[improved] = trial_values[improved]
        self.fitness[improved] = trial_fitness[improved]
    
    def get_subspace(self, indices: torch.Tensor) -> torch.Tensor:
        """
        获取子空间视图 (用于 CC 分组优化)
        
        Args:
            indices: 维度索引 (groupsize,)
            
        Returns:
            (NP, groupsize) 子空间张量
        """
        return self.values[:, indices]
    
    def set_subspace(self, indices: torch.Tensor, values: torch.Tensor):
        """
        设置子空间值
        
        Args:
            indices: 维度索引 (groupsize,)
            values: (NP, groupsize) 新值
        """
        self.values[:, indices] = values
    
    def to_snapshot(self, allgroups: List[List[int]], include_fitness: bool = True) -> torch.Tensor:
        """
        转换为 LandscapeEncoder 需要的快照格式
        
        Args:
            allgroups: 分组信息 [[group1_indices], [group2_indices], ...]
            include_fitness: 是否包含适应度
            
        Returns:
            (n_groups, NP, groupsize+1) 或 (n_groups, NP, groupsize)
        """
        n_groups = len(allgroups)
        groupsize = len(allgroups[0])  # 假设所有组大小相同
        
        if include_fitness:
            snapshot = torch.zeros(n_groups, self.NP, groupsize + 1, 
                                   device=self.device, dtype=self.dtype)
            for g_idx, indices in enumerate(allgroups):
                snapshot[g_idx, :, :-1] = self.values[:, indices]
                snapshot[g_idx, :, -1] = self.fitness
        else:
            snapshot = torch.zeros(n_groups, self.NP, groupsize,
                                   device=self.device, dtype=self.dtype)
            for g_idx, indices in enumerate(allgroups):
                snapshot[g_idx] = self.values[:, indices]
        
        return snapshot
    
    def __repr__(self) -> str:
        eval_status = "evaluated" if self._evaluated.all() else "unevaluated"
        best_fit = f", best={self.fitness.min().item():.4f}" if self._evaluated.any() else ""
        return f"GPUPopulation(NP={self.NP}, D={self.D}, device={self.device}, {eval_status}{best_fit})"


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing GPUPopulation...")
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # 创建种群
    pop = GPUPopulation(NP=50, D=100, device=device, lb=-100, ub=100)
    print(f"Created: {pop}")
    assert pop.values.shape == (50, 100), "Shape mismatch"
    assert pop.values.device.type == device.split(':')[0], "Device mismatch"
    
    # 测试克隆
    pop_clone = pop.clone()
    pop_clone.values[0, 0] = 999
    assert pop.values[0, 0] != 999, "Clone should be independent"
    print("Clone test passed")
    
    # 测试切片
    sub_pop = pop[:10]
    assert sub_pop.NP == 10, "Slice NP mismatch"
    print("Slice test passed")
    
    # 模拟评估
    class DummyProblem(nn.Module):
        def evaluate(self, x):
            return (x ** 2).sum(dim=-1)
    
    problem = DummyProblem().to(device)
    fitness = pop.evaluate(problem)
    assert fitness.shape == (50,), "Fitness shape mismatch"
    print(f"After evaluation: {pop}")
    
    # 测试最优解
    best_fit, best_idx = pop.best()
    print(f"Best fitness: {best_fit.item():.4f} at index {best_idx}")
    
    # 测试贪婪更新
    trial_values = pop.values.clone()
    trial_values[0] = 0  # 让第一个变为最优
    trial_fitness = problem.evaluate(trial_values)
    pop.update_where_improved(trial_values, trial_fitness)
    print(f"After greedy update: {pop}")
    
    # 测试快照
    allgroups = [list(range(i*10, (i+1)*10)) for i in range(10)]
    snapshot = pop.to_snapshot(allgroups)
    assert snapshot.shape == (10, 50, 11), f"Snapshot shape mismatch: {snapshot.shape}"
    print(f"Snapshot shape: {snapshot.shape}")
    
    print("\nOK: All GPUPopulation tests passed!")
