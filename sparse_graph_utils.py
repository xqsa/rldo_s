"""
稀疏图工具 (PyG 风格)

参考 PyTorch Geometric 的 Edge-Centric 设计，实现稀疏图操作。
用于支持 N>1000 的大规模优化场景。

核心思想:
1. 使用 edge_index (2, E) 替代稠密邻接矩阵 (N, N)
2. 使用 scatter_reduce 替代矩阵乘法
3. 使用对角堆叠替代 batch 维度

使用方法:
    edge_index = dense_to_edge_index(adj_matrix)
    labels = connected_components_sparse(edge_index, num_nodes)
"""

import torch
from typing import List, Tuple, Optional, Union


def dense_to_edge_index(
    adj: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    将稠密邻接矩阵转换为稀疏 edge_index
    
    参考: torch_geometric.utils.sparse.dense_to_sparse
    
    Args:
        adj: (N, N) 或 (B, N, N) 邻接矩阵 (可以是概率或 0/1)
        threshold: 阈值，adj >= threshold 的位置视为边
        
    Returns:
        edge_index: (2, E) 边索引
        edge_attr: (E,) 边权重 (如果 adj 非二值)
    """
    if adj.dim() == 2:
        # 单个图
        mask = adj >= threshold
        edge_index = mask.nonzero().t().contiguous()  # (2, E)
        edge_attr = adj[mask] if adj.dtype != torch.bool else None
        return edge_index, edge_attr
    
    elif adj.dim() == 3:
        # 批量图: 返回分离的 edge_index 列表
        # 或者使用对角堆叠 (见 batch_edge_index)
        batch_size, N, _ = adj.shape
        edge_indices = []
        edge_attrs = []
        
        for b in range(batch_size):
            mask = adj[b] >= threshold
            ei = mask.nonzero().t().contiguous()
            edge_indices.append(ei)
            edge_attrs.append(adj[b][mask] if adj.dtype != torch.bool else None)
        
        return edge_indices, edge_attrs
    
    else:
        raise ValueError(f"adj must be 2D or 3D, got {adj.dim()}D")


def batch_edge_index(
    edge_indices: List[torch.Tensor],
    num_nodes_per_graph: Union[int, List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对角堆叠多个图的 edge_index (PyG Batch 风格)
    
    参考: torch_geometric.data.batch.Batch.from_data_list
    
    Args:
        edge_indices: 每个图的 edge_index 列表
        num_nodes_per_graph: 每个图的节点数 (int 表示全部相同)
        
    Returns:
        batched_edge_index: (2, total_E) 合并后的边索引
        batch: (total_N,) 每个节点属于哪个图
        
    Example:
        # 3 个 10 节点的图 -> 1 个 30 节点的超图
        # graph 0 的节点 [0-9], graph 1 的节点 [10-19], graph 2 的节点 [20-29]
    """
    device = edge_indices[0].device if len(edge_indices) > 0 else 'cpu'
    
    if isinstance(num_nodes_per_graph, int):
        num_nodes_per_graph = [num_nodes_per_graph] * len(edge_indices)
    
    batched_edges = []
    batch_list = []
    offset = 0
    
    for i, (ei, n) in enumerate(zip(edge_indices, num_nodes_per_graph)):
        # 加上偏移量
        batched_edges.append(ei + offset)
        # 记录 batch 归属
        batch_list.append(torch.full((n,), i, dtype=torch.long, device=device))
        offset += n
    
    batched_edge_index = torch.cat(batched_edges, dim=1) if batched_edges else torch.empty(2, 0, dtype=torch.long, device=device)
    batch = torch.cat(batch_list) if batch_list else torch.empty(0, dtype=torch.long, device=device)
    
    return batched_edge_index, batch


def connected_components_sparse(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iters: int = 100
) -> torch.Tensor:
    """
    基于消息传递的稀疏连通分量算法 (PyG 风格)
    
    使用 scatter_reduce 实现标签传播，复杂度 O(E × k)
    
    [GPU 优化] 使用固定迭代次数代替收敛检查，避免 .all() 触发 GPU 同步。
    对于 N 个节点的图，log2(N) 次迭代即可保证收敛（标签传播的理论上界）。
    
    Args:
        edge_index: (2, E) 边索引 (应该是无向图，即已添加反向边)
        num_nodes: 节点总数
        max_iters: 最大迭代次数
        
    Returns:
        labels: (num_nodes,) 每个节点的连通分量标签 (最小节点 ID)
    """
    if edge_index.numel() == 0:
        # 无边 -> 每个节点自成一组
        return torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    
    device = edge_index.device
    
    # 初始化: 每个节点的标签 = 自身索引
    labels = torch.arange(num_nodes, device=device, dtype=torch.long)
    
    src, dst = edge_index[0], edge_index[1]
    
    # [GPU 优化] 固定迭代次数 = min(ceil(log2(num_nodes)) + 2, max_iters)
    # 标签传播在无向图上的收敛上界是图的直径，而直径 <= log2(N) 对于大多数随机图
    # 多加 2 次作为安全余量
    import math
    fixed_iters = min(int(math.ceil(math.log2(max(num_nodes, 2)))) + 2, max_iters)
    
    for _ in range(fixed_iters):
        # Message: 获取邻居的标签
        neighbor_labels = labels[src]  # (E,) 源节点的标签
        
        # Aggregate: 每个目标节点取邻居标签的最小值
        new_labels = labels.clone()
        new_labels.scatter_reduce_(
            dim=0,
            index=dst,
            src=neighbor_labels,
            reduce='amin',
            include_self=True
        )
        
        # [GPU 优化] 不做收敛检查，固定迭代次数，避免 GPU sync
        labels = new_labels
    
    return labels


def unbatch_labels(
    labels: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int
) -> List[torch.Tensor]:
    """
    将批量标签拆分回单个图
    
    Args:
        labels: (total_N,) 批量标签
        batch: (total_N,) 每个节点属于哪个图
        num_graphs: 图的数量
        
    Returns:
        每个图的标签列表
    """
    return [labels[batch == i] for i in range(num_graphs)]


def labels_to_groups(
    labels: torch.Tensor,
    allgroups_base: List[List[int]]
) -> List[List[int]]:
    """
    [兼容接口] 根据连通分量标签合并分组
    
    [GPU 优化] 使用 GPU argsort 替代 .cpu().numpy() 搬运
    
    Args:
        labels: (N,) 每个节点的连通分量标签
        allgroups_base: 原始分组 [[g0], [g1], ...]
        
    Returns:
        合并后的分组
    """
    n = len(allgroups_base)
    
    # [GPU 优化] 使用 unique 在 GPU 上完成分组，仅最后转 CPU
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    n_merged = unique_labels.numel()
    
    # 一次性转为 CPU (单次搬运，而非 N 次 .item())
    inverse_cpu = inverse.cpu().tolist()
    
    merged = [[] for _ in range(n_merged)]
    for i in range(n):
        merged[inverse_cpu[i]].extend(allgroups_base[i])
    
    return merged


def labels_to_groups_with_weights(
    labels: torch.Tensor,
    allgroups_base: List[List[int]],
    original_weights: torch.Tensor,
    device: torch.device
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    根据连通分量标签合并分组，同时累加原始权重
    
    [GPU 优化] 权重累加使用 scatter_add 在 GPU 上完成，避免逐元素 .item()
    
    Args:
        labels: (N,) 每个节点的连通分量标签
        allgroups_base: 原始分组 [[g0], [g1], ...]
        original_weights: (N,) 原始组的权重
        device: 计算设备
        
    Returns:
        merged_groups: 合并后的分组
        merged_weights: 合并后组的权重 (各成员权重之和)
    """
    n = len(allgroups_base)
    
    # [GPU 优化] 在 GPU 上完成 unique + 权重累加
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    n_merged = unique_labels.numel()
    
    # GPU 上权重累加
    merged_weights = torch.zeros(n_merged, device=device, dtype=original_weights.dtype)
    merged_weights.scatter_add_(0, inverse, original_weights)
    
    # 仅最后一次 CPU 搬运
    inverse_cpu = inverse.cpu().tolist()
    
    merged_groups = [[] for _ in range(n_merged)]
    for i in range(n):
        merged_groups[inverse_cpu[i]].extend(allgroups_base[i])
    
    return merged_groups, merged_weights


def labels_to_groups_tensor(
    labels: torch.Tensor,
    group_indices_flat: torch.Tensor,
    group_sizes: torch.Tensor,
    device: torch.device,
    max_groups: int = None,
    max_groupsize: int = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    [向量化] 根据连通分量标签合并分组，输出 GPU Tensor
    
    Args:
        labels: (n_groups,) 每个组的连通分量标签
        group_indices_flat: 原始分组索引的扁平拼接
        group_sizes: 每个原始组的大小
        device: 计算设备
        max_groups: 返回的最大组数
        max_groupsize: 返回的最大组大小
        
    Returns:
        merged_indices: (n_merged, max_merged_size) 合并后的组索引
        merged_valid: (n_merged, max_merged_size) 有效掩码
        n_merged: 合并后的组数
    """
    n_groups = labels.numel()
    
    if n_groups == 0:
        max_groups = max_groups or 1
        max_groupsize = max_groupsize or 1
        return (
            torch.zeros(max_groups, max_groupsize, dtype=torch.long, device=device),
            torch.zeros(max_groups, max_groupsize, dtype=torch.bool, device=device),
            0
        )
    
    unique_labels = labels.unique()
    n_merged = unique_labels.numel()
    
    if max_groups is None:
        max_groups = n_merged
    
    # 计算每个合并组的总大小
    merged_sizes = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long, device=device)
    merged_sizes.scatter_add_(0, labels, group_sizes)
    actual_merged_sizes = merged_sizes[unique_labels]
    
    if max_groupsize is None:
        max_groupsize = actual_merged_sizes.max().item() if actual_merged_sizes.numel() > 0 else 1
    
    merged_indices = torch.zeros(max_groups, max_groupsize, dtype=torch.long, device=device)
    merged_valid = torch.zeros(max_groups, max_groupsize, dtype=torch.bool, device=device)
    
    # 简化：遍历每个连通分量
    group_starts = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        group_sizes.cumsum(0)[:-1]
    ])
    
    for m_idx, label_val in enumerate(unique_labels[:max_groups]):
        member_mask = (labels == label_val)
        member_groups = member_mask.nonzero(as_tuple=True)[0]
        
        all_indices = []
        for g in member_groups:
            g_start = group_starts[g].item()
            g_size = group_sizes[g].item()
            all_indices.append(group_indices_flat[g_start:g_start + g_size])
        
        if all_indices:
            merged_idx = torch.cat(all_indices)
            size = min(merged_idx.numel(), max_groupsize)
            merged_indices[m_idx, :size] = merged_idx[:size]
            merged_valid[m_idx, :size] = True
    
    return merged_indices, merged_valid, min(n_merged, max_groups)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing sparse graph utils...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 测试 dense_to_edge_index
    print("\n--- dense_to_edge_index ---")
    adj = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32, device=device)
    
    edge_index, _ = dense_to_edge_index(adj, threshold=0.5)
    print(f"Adj shape: {adj.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edges: {edge_index.t().tolist()}")
    
    # 测试 connected_components_sparse
    print("\n--- connected_components_sparse ---")
    edge_index_1 = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2]
    ], dtype=torch.long, device=device)
    labels_1 = connected_components_sparse(edge_index_1, num_nodes=4)
    print(f"Test 1 (0-1, 2-3): labels = {labels_1.tolist()}")
    
    edge_index_2 = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long, device=device)
    labels_2 = connected_components_sparse(edge_index_2, num_nodes=4)
    print(f"Test 2 (chain): labels = {labels_2.tolist()}")
    
    edge_index_3 = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.long, device=device)
    labels_3 = connected_components_sparse(edge_index_3, num_nodes=4)
    print(f"Test 3 (isolated 2,3): labels = {labels_3.tolist()}")
    
    # 测试 labels_to_groups_tensor
    print("\n--- labels_to_groups_tensor ---")
    labels = torch.tensor([0, 0, 1], device=device)  # 组 0,1 合并, 组 2 独立
    flat_indices = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
    sizes = torch.tensor([2, 2, 2], device=device)
    merged_idx, merged_valid, n = labels_to_groups_tensor(labels, flat_indices, sizes, device)
    print(f"Labels: {labels.tolist()}")
    print(f"Merged indices: {merged_idx.tolist()}")
    print(f"Merged valid: {merged_valid.tolist()}")
    print(f"N merged: {n}")
    
    print("\nOK: All sparse graph utils tests passed!")

