"""
GPU BBOB 问题定义

将 BBOB 函数重写为 PyTorch 向量化版本，使用 nn.Module 自动管理设备。
使用 register_buffer 确保旋转矩阵等参数自动跟随模型迁移到 GPU。

使用方法:
    problem = GPUElliptic(dim=100).to('cuda')
    fitness = problem.evaluate(pop.values)  # 批量评估
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


def generate_rotation_matrix(dim: int) -> np.ndarray:
    """
    生成随机正交矩阵 (Gram-Schmidt)
    
    Args:
        dim: 维度
        
    Returns:
        (dim, dim) 正交矩阵
    """
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    return Q


class GPUProblem(nn.Module):
    """
    GPU 上的优化问题基类
    
    使用 register_buffer 自动管理设备迁移。
    子类需要实现 _fitness(z) 方法。
    """
    
    def __init__(
        self,
        dim: int,
        shift: Optional[np.ndarray] = None,
        rotate: Optional[np.ndarray] = None,
        lb: float = -100.0,
        ub: float = 100.0,
        dtype: torch.dtype = torch.float64
    ):
        """
        初始化问题
        
        Args:
            dim: 问题维度
            shift: 偏移向量 (可选)
            rotate: 旋转矩阵 (可选)
            lb: 下界
            ub: 上界
            dtype: 数据类型
        """
        super().__init__()
        
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.dtype = dtype
        
        # 注册 buffer，自动跟随 .to(device) 迁移
        if shift is None:
            shift = np.zeros(dim)
        self.register_buffer('shift', torch.tensor(shift, dtype=dtype))
        
        if rotate is None:
            rotate = np.eye(dim)
        self.register_buffer('rotate', torch.tensor(rotate, dtype=dtype))
        
        # 预计算 elliptic 系数 (常用)
        self.register_buffer(
            'elliptic_coeffs',
            1e6 ** torch.linspace(0, 1, dim, dtype=dtype)
        )
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量评估适应度
        
        Args:
            x: (NP, D) 或 (batch, NP, D) 决策变量
            
        Returns:
            (NP,) 或 (batch, NP) 适应度值
        """
        # 偏移
        z = x - self.shift
        
        # 旋转
        z = z @ self.rotate.T
        
        return self._fitness(z)
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算适应度 (子类实现)
        
        Args:
            z: (NP, D) 旋转偏移后的决策变量
            
        Returns:
            (NP,) 适应度值
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, lb={self.lb}, ub={self.ub})"


class GPUSphere(GPUProblem):
    """F1: Sphere 函数 (最简单的单峰函数)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        return (z ** 2).sum(dim=-1)


class GPUElliptic(GPUProblem):
    """F2: Elliptic 函数 (高病态条件)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        return (self.elliptic_coeffs * z ** 2).sum(dim=-1)


class GPURosenbrock(GPUProblem):
    """F8: Rosenbrock 函数 (狭长山谷)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z[..., :-1]  # x_1 to x_{n-1}
        z2 = z[..., 1:]   # x_2 to x_n
        return (100 * (z2 - z1 ** 2) ** 2 + (z1 - 1) ** 2).sum(dim=-1)


class GPURastrigin(GPUProblem):
    """F3: Rastrigin 函数 (多峰)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        D = z.shape[-1]
        return 10 * D + (z ** 2 - 10 * torch.cos(2 * np.pi * z)).sum(dim=-1)


class GPUAckley(GPUProblem):
    """Ackley 函数 (多峰)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        D = z.shape[-1]
        sum_sq = (z ** 2).sum(dim=-1) / D
        sum_cos = torch.cos(self.c * z).sum(dim=-1) / D
        return -self.a * torch.exp(-self.b * torch.sqrt(sum_sq)) - torch.exp(sum_cos) + self.a + np.e


class GPUSchwefel(GPUProblem):
    """Schwefel 函数 (欺骗性多峰)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        D = z.shape[-1]
        return 418.9829 * D - (z * torch.sin(torch.sqrt(torch.abs(z)))).sum(dim=-1)


class GPUGriewank(GPUProblem):
    """Griewank 函数 (多峰)"""
    
    def _fitness(self, z: torch.Tensor) -> torch.Tensor:
        D = z.shape[-1]
        indices = torch.arange(1, D + 1, device=z.device, dtype=z.dtype)
        sum_sq = (z ** 2).sum(dim=-1) / 4000
        prod_cos = torch.prod(torch.cos(z / torch.sqrt(indices)), dim=-1)
        return sum_sq - prod_cos + 1


# ==================== 工厂函数 ====================

GPU_PROBLEMS = {
    'sphere': GPUSphere,
    'elliptic': GPUElliptic,
    'rosenbrock': GPURosenbrock,
    'rastrigin': GPURastrigin,
    'ackley': GPUAckley,
    'schwefel': GPUSchwefel,
    'griewank': GPUGriewank,
}


def create_gpu_problem(
    name: str,
    dim: int,
    shifted: bool = True,
    rotated: bool = True,
    lb: float = -100.0,
    ub: float = 100.0,
    device: Union[str, torch.device] = 'cuda'
) -> GPUProblem:
    """
    工厂函数：创建 GPU 问题实例
    
    Args:
        name: 问题名称 (sphere, elliptic, rosenbrock, etc.)
        dim: 维度
        shifted: 是否添加偏移
        rotated: 是否添加旋转
        lb: 下界
        ub: 上界
        device: 计算设备
        
    Returns:
        GPU 问题实例
    """
    if name not in GPU_PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(GPU_PROBLEMS.keys())}")
    
    # 生成偏移和旋转
    if shifted:
        shift = 0.8 * (lb + (ub - lb) * np.random.rand(dim))
    else:
        shift = None
    
    if rotated:
        rotate = generate_rotation_matrix(dim)
    else:
        rotate = None
    
    problem = GPU_PROBLEMS[name](dim=dim, shift=shift, rotate=rotate, lb=lb, ub=ub)
    return problem.to(device)


# ==================== CC 加权问题 ====================

class GPUWeightedElliptic(nn.Module):
    """
    加权 Elliptic 问题 (用于 CC 分组优化)
    
    每个子问题有独立的偏移、旋转和权重。
    
    [P1 优化] 批量向量化评估，消除 Python for-loop
    - 假设所有组大小相同 (typical CC 场景)
    - 使用 (n_groups, NP, groupsize) 批量矩阵运算
    """
    
    def __init__(
        self,
        allgroups: list,
        weights: torch.Tensor,
        xopt: torch.Tensor,
        R100: torch.Tensor,
        lb: float = -100.0,
        ub: float = 100.0
    ):
        """
        初始化加权问题
        
        Args:
            allgroups: 分组信息 [[group1_indices], ...]
            weights: (n_groups,) 权重
            xopt: (D,) 最优解
            R100: (100, 100) 旋转矩阵基础
            lb, ub: 边界
        """
        super().__init__()
        
        self.allgroups = allgroups
        self.n_groups = len(allgroups)
        self.lb = lb
        self.ub = ub
        
        # 检查是否所有组大小相同（用于批量优化）
        groupsizes = [len(g) for g in allgroups]
        self.uniform_groupsize = len(set(groupsizes)) == 1
        self.groupsize = groupsizes[0] if self.uniform_groupsize else max(groupsizes)
        self._groupsizes = groupsizes
        
        # 批量优化仅在 groupsize <= R100 尺寸时可用
        self.can_batch = self.uniform_groupsize and self.groupsize <= R100.shape[0]
        
        self.register_buffer('weights', weights)
        self.register_buffer('xopt', xopt)
        self.register_buffer('R100', R100)
        
        # [P1 优化] 构建批量索引矩阵 (n_groups, groupsize)
        if self.can_batch:
            all_indices = torch.tensor(allgroups, dtype=torch.long)  # (n_groups, groupsize)
            self.register_buffer('batch_indices', all_indices)
            
            # 预计算 Elliptic 系数
            elliptic_coeffs = 1e6 ** torch.linspace(0, 1, self.groupsize)
            self.register_buffer('elliptic_coeffs', elliptic_coeffs)
            
            # 预切片旋转矩阵
            R_slice = R100[:self.groupsize, :self.groupsize]
            self.register_buffer('R_slice', R_slice)
        else:
            # 非统一大小或超大 groupsize: 分别缓存
            for j, indices in enumerate(allgroups):
                self.register_buffer(f'indices_{j}', torch.tensor(indices, dtype=torch.long))
            
            max_groupsize = max(groupsizes)
            self.register_buffer(
                'elliptic_coeffs_max',
                1e6 ** torch.linspace(0, 1, max_groupsize)
            )
            
            # [GPU 优化] 预计算每组的 R 矩阵切片和系数，避免运行时 getattr + 条件分支
            self._precomputed_R = []
            self._precomputed_coeffs_list = []
            for j in range(self.n_groups):
                gs = groupsizes[j]
                if gs <= R100.shape[0]:
                    R_j = R100[:gs, :gs]
                else:
                    R_j = torch.eye(gs, dtype=R100.dtype)
                self.register_buffer(f'_R_{j}', R_j)
                coeffs_j = 1e6 ** torch.linspace(0, 1, gs)
                self.register_buffer(f'_coeffs_{j}', coeffs_j)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量评估适应度
        
        [P1 优化] 统一 groupsize 场景使用全批量运算，无 Python 循环
        
        Args:
            x: (NP, D) 决策变量
            
        Returns:
            (NP,) 适应度值
        """
        NP = x.shape[0]
        device = x.device
        
        # 偏移
        z = x - self.xopt  # (NP, D)
        
        if self.can_batch:
            # ========== 批量优化路径 ==========
            # 使用 index_select 批量提取所有组
            # z: (NP, D), batch_indices: (n_groups, groupsize)
            # -> z_all: (n_groups, NP, groupsize)
            z_all = z[:, self.batch_indices].permute(1, 0, 2)
            
            # 批量旋转: (n_groups, NP, groupsize) @ (groupsize, groupsize).T
            # -> (n_groups, NP, groupsize)
            z_rot = torch.einsum('gnk,kl->gnl', z_all, self.R_slice)
            
            # 批量 Elliptic: coeffs (groupsize,), z_rot² (n_groups, NP, groupsize)
            elliptic_vals = (self.elliptic_coeffs * z_rot ** 2).sum(dim=-1)  # (n_groups, NP)
            
            # 加权求和: weights (n_groups,), elliptic_vals (n_groups, NP)
            fitness = (self.weights.unsqueeze(1) * elliptic_vals).sum(dim=0)  # (NP,)
            
            return fitness
        
        else:
            # ========== 非统一 groupsize 回退路径 ==========
            # [GPU 优化] 使用预计算的 R 矩阵和系数
            fitness = torch.zeros(NP, device=device, dtype=x.dtype)
            
            for j in range(self.n_groups):
                indices_t = getattr(self, f'indices_{j}')
                z_sub = z[:, indices_t]
                R = getattr(self, f'_R_{j}')
                z_rot = z_sub @ R.T
                coeffs = getattr(self, f'_coeffs_{j}')
                elliptic_vals = (coeffs * z_rot ** 2).sum(dim=-1)
                fitness += self.weights[j] * elliptic_vals
            
            return fitness
    
    def evaluate_batched(self, x: torch.Tensor) -> torch.Tensor:
        """
        [原生张量并行] 批量评估适应度
        
        支持 (Batch, NP, D) 输入，所有 Batch 样本共享相同的问题结构。
        
        Args:
            x: (Batch, NP, D) 或 (NP, D) 决策变量
            
        Returns:
            (Batch, NP) 或 (NP,) 适应度值
        """
        # 自动处理 2D 输入
        if x.dim() == 2:
            return self.evaluate(x)
        
        Batch, NP, D = x.shape
        device = x.device
        
        # 偏移: (Batch, NP, D) - (D,) -> (Batch, NP, D)
        z = x - self.xopt
        
        if self.can_batch:
            # ========== 批量优化路径 ==========
            # 使用 batch_indices 批量提取所有组
            # z: (Batch, NP, D), batch_indices: (n_groups, groupsize)
            # -> z_all: (Batch, NP, n_groups, groupsize) -> (Batch, n_groups, NP, groupsize)
            z_all = z[:, :, self.batch_indices]  # (Batch, NP, n_groups, groupsize)
            z_all = z_all.permute(0, 2, 1, 3)    # (Batch, n_groups, NP, groupsize)
            
            # 批量旋转: (Batch, n_groups, NP, groupsize) @ (groupsize, groupsize).T
            # -> (Batch, n_groups, NP, groupsize)
            z_rot = torch.einsum('bgnk,kl->bgnl', z_all, self.R_slice)
            
            # 批量 Elliptic: coeffs (groupsize,), z_rot² (Batch, n_groups, NP, groupsize)
            elliptic_vals = (self.elliptic_coeffs * z_rot ** 2).sum(dim=-1)  # (Batch, n_groups, NP)
            
            # 加权求和: weights (n_groups,), elliptic_vals (Batch, n_groups, NP)
            # weights: (n_groups,) -> (1, n_groups, 1)
            fitness = (self.weights.view(1, -1, 1) * elliptic_vals).sum(dim=1)  # (Batch, NP)
            
            return fitness
        
        else:
            # ========== 非统一 groupsize 回退路径 ==========
            # 对于非统一情况，仍需要循环（但这种情况较少见）
            fitness = torch.zeros(Batch, NP, device=device, dtype=x.dtype)
            
            for j in range(self.n_groups):
                indices_t = getattr(self, f'indices_{j}')
                z_sub = z[:, :, indices_t]  # (Batch, NP, groupsize)
                R = getattr(self, f'_R_{j}')
                
                # 批量旋转: (Batch, NP, groupsize) @ (groupsize, groupsize)
                z_rot = torch.einsum('bnk,kl->bnl', z_sub, R)
                
                coeffs = getattr(self, f'_coeffs_{j}')
                elliptic_vals = (coeffs * z_rot ** 2).sum(dim=-1)  # (Batch, NP)
                
                fitness += self.weights[j] * elliptic_vals
            
            return fitness


# ==================== 批量问题工厂 ====================

class BatchedProblemWrapper(nn.Module):
    """
    批量问题包装器
    
    将多个不同分组结构的问题合并为一个批量问题，
    使用 Padding + Mask 策略处理分组异构性。
    """
    
    def __init__(
        self,
        batch_size: int,
        D: int,
        group_indices: torch.Tensor,  # (Batch, Max_Groups, max_groupsize)
        valid_mask: torch.Tensor,      # (Batch, Max_Groups, max_groupsize)
        group_counts: torch.Tensor,    # (Batch,)
        weights: torch.Tensor,         # (Batch, Max_Groups)
        xopt: torch.Tensor,            # (D,) 或 (Batch, D)
        R: torch.Tensor,               # (max_groupsize, max_groupsize)
        device: torch.device
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.D = D
        self.Max_Groups = group_indices.shape[1]
        self.max_groupsize = group_indices.shape[2]
        
        self.register_buffer('group_indices', group_indices)
        self.register_buffer('valid_mask', valid_mask)
        self.register_buffer('group_counts', group_counts)
        self.register_buffer('weights', weights)
        
        # 处理 xopt
        if xopt.dim() == 1:
            xopt = xopt.unsqueeze(0).expand(batch_size, -1)
        self.register_buffer('xopt', xopt)
        
        self.register_buffer('R', R)
        
        # 预计算 Elliptic 系数
        elliptic_coeffs = 1e6 ** torch.linspace(0, 1, self.max_groupsize, device=device)
        self.register_buffer('elliptic_coeffs', elliptic_coeffs)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量评估
        
        [修复] 正确使用 valid_mask 进行逐维过滤
        
        Args:
            x: (Batch, NP, D) 决策变量
            
        Returns:
            (Batch, NP) 适应度值
        """
        Batch, NP, D = x.shape
        device = x.device
        
        # 偏移: (Batch, NP, D) - (Batch, 1, D) -> (Batch, NP, D)
        z = x - self.xopt.unsqueeze(1)
        
        # 提取所有子空间
        # z: (Batch, NP, D), group_indices: (Batch, Max_Groups, max_groupsize)
        # -> z_sub: (Batch, Max_Groups, NP, max_groupsize)
        
        # 扩展索引
        indices_expanded = self.group_indices.unsqueeze(2).expand(-1, -1, NP, -1)
        z_expanded = z.unsqueeze(1).expand(-1, self.Max_Groups, -1, -1)
        z_sub = torch.gather(z_expanded, dim=-1, index=indices_expanded)
        
        # 旋转: (Batch, Max_Groups, NP, max_groupsize) @ (max_groupsize, max_groupsize)
        z_rot = torch.einsum('bgnk,kl->bgnl', z_sub, self.R)
        
        # [修复] Elliptic 计算：使用 valid_mask 逐维过滤
        # valid_mask: (Batch, Max_Groups, max_groupsize) -> (Batch, Max_Groups, 1, max_groupsize)
        valid_expanded = self.valid_mask.unsqueeze(2).expand(-1, -1, NP, -1)
        
        # 将无效维度的系数设为 0
        coeffs_expanded = self.elliptic_coeffs.view(1, 1, 1, -1).expand(Batch, self.Max_Groups, NP, -1)
        masked_coeffs = torch.where(valid_expanded, coeffs_expanded, torch.zeros_like(coeffs_expanded))
        
        # 只对有效维度求和
        elliptic_vals = (masked_coeffs * z_rot ** 2).sum(dim=-1)  # (Batch, Max_Groups, NP)
        
        # 创建组有效掩码: (Batch, Max_Groups)
        group_idx = torch.arange(self.Max_Groups, device=device).unsqueeze(0).expand(Batch, -1)
        group_valid = (group_idx < self.group_counts.unsqueeze(1)).float()
        
        # 加权求和：正确处理无效组
        masked_weights = self.weights * group_valid
        
        # (Batch, Max_Groups, NP) * (Batch, Max_Groups, 1) -> sum -> (Batch, NP)
        weighted = elliptic_vals * masked_weights.unsqueeze(-1)
        fitness = weighted.sum(dim=1)  # (Batch, NP)
        
        return fitness


def create_batched_problem(
    groups_list: list,
    weights_list: torch.Tensor,  # (Batch, Max_Groups) 或 list
    D: int,
    R: torch.Tensor,
    device: torch.device,
    xopt: torch.Tensor = None
) -> BatchedProblemWrapper:
    """
    创建批量问题实例
    
    Args:
        groups_list: 每个 Batch 样本的分组信息
        weights_list: (Batch, Max_Groups) 权重
        D: 总维度
        R: 旋转矩阵
        device: 计算设备
        xopt: 最优解位置
        
    Returns:
        BatchedProblemWrapper 实例
    """
    from group_mask_utils import batch_groups_to_indices_padded
    
    indices, valid, counts = batch_groups_to_indices_padded(groups_list, device)
    
    batch_size = len(groups_list)
    max_groups = indices.shape[1]
    max_groupsize = indices.shape[2]
    
    # 处理权重
    if isinstance(weights_list, list):
        # 将 list 转换为 Tensor，Padding 到 Max_Groups
        weights = torch.zeros(batch_size, max_groups, device=device)
        for b, w in enumerate(weights_list):
            n_w = min(len(w), max_groups)
            if isinstance(w, torch.Tensor):
                weights[b, :n_w] = w[:n_w]
            else:
                weights[b, :n_w] = torch.tensor(w[:n_w], device=device)
    else:
        weights = weights_list
    
    # 处理旋转矩阵
    if R.shape[0] < max_groupsize:
        # 需要扩展 R
        R_full = torch.eye(max_groupsize, device=device, dtype=R.dtype)
        R_full[:R.shape[0], :R.shape[1]] = R
        R = R_full
    else:
        R = R[:max_groupsize, :max_groupsize]
    
    # 处理 xopt
    if xopt is None:
        xopt = torch.zeros(D, device=device)
    
    return BatchedProblemWrapper(
        batch_size=batch_size,
        D=D,
        group_indices=indices,
        valid_mask=valid,
        group_counts=counts,
        weights=weights,
        xopt=xopt,
        R=R,
        device=device
    ).to(device)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing GPU Problems...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    dim = 100
    NP = 50
    
    # 测试各个问题
    print("\n--- Individual Problems ---")
    for name in GPU_PROBLEMS:
        problem = create_gpu_problem(name, dim, shifted=True, rotated=True, device=device)
        x = torch.rand(NP, dim, device=device) * 200 - 100
        fitness = problem.evaluate(x)
        assert fitness.shape == (NP,), f"{name} fitness shape mismatch"
        print(f"{name}: min={fitness.min().item():.4e}, max={fitness.max().item():.4e}")
    
    # 测试最优解
    print("\n--- Optimum Test ---")
    problem = create_gpu_problem('sphere', dim, shifted=True, rotated=True, device=device)
    xopt = problem.shift.clone()  # 最优解应该在 shift 处
    opt_fitness = problem.evaluate(xopt.unsqueeze(0))
    print(f"Sphere at optimum: {opt_fitness.item():.4e} (should be ~0)")
    
    # 测试加权问题
    print("\n--- Weighted Problem ---")
    n_groups = 10
    groupsize = 100
    D = n_groups * groupsize
    
    allgroups = [list(range(i * groupsize, (i + 1) * groupsize)) for i in range(n_groups)]
    weights = torch.rand(n_groups, device=device)
    xopt = torch.zeros(D, device=device)
    R100 = torch.tensor(generate_rotation_matrix(100), device=device, dtype=torch.float32)
    
    weighted_problem = GPUWeightedElliptic(allgroups, weights, xopt, R100).to(device)
    x = torch.rand(NP, D, device=device) * 200 - 100
    fitness = weighted_problem.evaluate(x)
    assert fitness.shape == (NP,), "Weighted problem fitness shape mismatch"
    print(f"Weighted Elliptic: min={fitness.min().item():.4e}")
    
    # 性能测试
    print("\n--- Performance Test ---")
    import time
    
    problem = create_gpu_problem('elliptic', 1000, device=device)
    x_large = torch.rand(1000, 1000, device=device) * 200 - 100
    
    # 预热
    _ = problem.evaluate(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = problem.evaluate(x_large)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"Elliptic evaluation on 1000x1000:")
    print(f"  Per iteration: {elapsed*1000/n_iters:.3f} ms")
    
    print("\nOK: All GPU problem tests passed!")
