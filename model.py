"""
RLDO 策略网络 - 稀疏图注意力 + Bernoulli 边概率输出

[突破点一] 消灭 timm 依赖，使用 Edge-Centric 稀疏图注意力
- SparseGraphAttention: 基于 edge_index 的多头注意力: O(E) 而非 O(N²)
- 手工 scatter_softmax: 纯 PyTorch，无需 torch_scatter / PyG

[突破点二] 输出从 (B,N,N,2) Softmax → (B,N,N) Sigmoid 边概率
- 边分类头: 节点特征对 → MLP → Sigmoid → p_ij ∈ [0,1]
- 配合 Bernoulli 分布采样，实现连续型拓扑图生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from landscape_encoder import LandscapeEncoder


# ==================== 稀疏图注意力核心 ====================

def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    手工实现 scatter_softmax (纯 PyTorch，无需 torch_scatter)
    
    对属于同一个 target 节点的所有边进行 softmax 归一化。
    
    Args:
        scores: (E,) 或 (E, H) 每条边的原始注意力得分
        index: (E,) 目标节点索引 (scatter 的 index)
        num_nodes: 节点总数 (用于 scatter 的大小)
        
    Returns:
        attn_weights: 与 scores 同形状，归一化后的注意力权重
    """
    # 1) 数值稳定：减去每个 target 节点的最大值
    if scores.dim() == 1:
        # (E,) 情况
        max_vals = torch.full((num_nodes,), -1e9, device=scores.device, dtype=scores.dtype)
        max_vals.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)
        scores_shifted = scores - max_vals[index]
    else:
        # (E, H) 情况：对每个 head 独立
        H = scores.shape[1]
        max_vals = torch.full((num_nodes, H), -1e9, device=scores.device, dtype=scores.dtype)
        index_exp = index.unsqueeze(1).expand(-1, H)
        max_vals.scatter_reduce_(0, index_exp, scores, reduce='amax', include_self=True)
        scores_shifted = scores - max_vals.gather(0, index_exp)
    
    # 2) exp
    exp_scores = torch.exp(scores_shifted)
    
    # 3) scatter_add 求分母
    if scores.dim() == 1:
        sum_exp = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, index, exp_scores)
        attn_weights = exp_scores / (sum_exp[index] + 1e-12)
    else:
        sum_exp = torch.zeros(num_nodes, H, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, index_exp, exp_scores)
        attn_weights = exp_scores / (sum_exp.gather(0, index_exp) + 1e-12)
    
    return attn_weights


class SparseGraphAttention(nn.Module):
    """
    Edge-Centric 稀疏多头注意力
    
    基于 edge_index 进行稀疏消息传递，复杂度 O(E * H) 而非 O(N² * H)。
    
    张量维度流转:
        x: (B*N, H) 节点特征
        → Q, K, V 线性映射 → (B*N, num_heads, head_dim)
        → Gather by edge_index → Q_edge, K_edge, V_edge: (total_E, num_heads, head_dim)
        → 边上点积 → scores: (total_E, num_heads)
        → scatter_softmax → attn_weights: (total_E, num_heads)
        → attn_weights * V_edge → messages: (total_E, num_heads, head_dim)
        → scatter_add → out: (B*N, num_heads, head_dim) → reshape → (B*N, H)
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        total_nodes: int
    ) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, H) 扁平化的节点特征 (B 个图对角堆叠)
            edge_index: (2, total_E) 稀疏边索引 (已含 batch 偏移)
            total_nodes: B * N
            
        Returns:
            out: (total_nodes, H)
        """
        H = self.num_heads
        D = self.head_dim
        
        # Q, K, V 映射: (total_nodes, H, D)
        Q = self.q_proj(x).view(-1, H, D)
        K = self.k_proj(x).view(-1, H, D)
        V = self.v_proj(x).view(-1, H, D)
        
        src = edge_index[0]  # 源节点 (提供 K, V)
        dst = edge_index[1]  # 目标节点 (提供 Q, 接收聚合)
        
        if src.numel() == 0:
            # 无边：返回零（每个节点接收不到消息）
            return self.out_proj(x)
        
        # Gather: 抽取边上的 Q, K, V
        Q_edge = Q[dst]  # (E, H, D) — Target 节点的 Query
        K_edge = K[src]  # (E, H, D) — Source 节点的 Key
        V_edge = V[src]  # (E, H, D) — Source 节点的 Value
        
        # 边上点积: (E, H)
        scores = (Q_edge * K_edge).sum(dim=-1) * self.scale
        
        # Scatter Softmax: 对每个 target 节点归一化
        attn_weights = scatter_softmax(scores, dst, total_nodes)  # (E, H)
        
        # 加权聚合: messages = attn * V
        messages = attn_weights.unsqueeze(-1) * V_edge  # (E, H, D)
        
        # Scatter_add 回目标节点
        out = torch.zeros(total_nodes, H, D, device=x.device, dtype=x.dtype)
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand(-1, H, D)
        out.scatter_add_(0, dst_exp, messages)
        
        # reshape + 输出映射
        out = out.reshape(total_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out


class SparseGraphBlock(nn.Module):
    """
    稀疏图 Transformer Block (替代 timm Block)
    
    LayerNorm → SparseGraphAttention → residual → LayerNorm → MLP → residual
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SparseGraphAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, total_nodes: int) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, H)
            edge_index: (2, total_E)
            total_nodes: B * N
        """
        # 稀疏注意力 + 残差
        x = self.attn(self.norm1(x), edge_index, total_nodes) + x
        # MLP + 残差
        x = self.mlp(self.norm2(x)) + x
        return x


# ==================== 边分类头 ====================

class EdgeClassifier(nn.Module):
    """
    边分类头: 将节点对特征映射为边概率
    
    对于每条潜在边 (i, j):
        feature = [x_i, x_j, x_i * x_j]  (拼接 + 逐元素乘积)
        → MLP → Sigmoid → p_ij ∈ [0, 1]
    """
    def __init__(self, hidden_size: int, edge_head_dim: int = 64):
        super().__init__()
        # 输入: concat(x_i, x_j, x_i * x_j) = 3 * hidden_size
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, edge_head_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(edge_head_dim, edge_head_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(edge_head_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, H) 节点特征
            n: 节点数量 N
            
        Returns:
            p_connect: (B, N, N) 边连接概率 ∈ [0, 1]
        """
        B, N, H = x.shape
        
        # 构建所有节点对的特征
        # x_i: (B, N, 1, H) → expand → (B, N, N, H)
        # x_j: (B, 1, N, H) → expand → (B, N, N, H)
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, H)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, H)
        
        # 拼接特征: [x_i, x_j, x_i * x_j]
        edge_feat = torch.cat([x_i, x_j, x_i * x_j], dim=-1)  # (B, N, N, 3H)
        
        # MLP → logits
        logits = self.edge_mlp(edge_feat).squeeze(-1)  # (B, N, N)
        
        # Sigmoid → 概率
        p_connect = torch.sigmoid(logits)  # (B, N, N)
        
        return p_connect


# ==================== 主模型 ====================

class Model(nn.Module):
    """
    RLDO 策略网络 - 稀疏图注意力 + Bernoulli 边概率
    
    融合两种信息流:
    1. 拓扑流 (Topo Stream): 编码子问题间的结构关系 (Edge-Centric)
    2. 景观流 (Landscape Stream): 编码种群在函数景观上的分布状态
    
    输入:
        topo: (B, N, N) 拓扑邻接矩阵 → 内部转为 edge_index
        pop_data_or_weight: 种群数据或权重
        
    输出:
        p_connect: (B, N, N) 边连接概率 ∈ [0, 1]
    
    Args:
        n: 子问题数量
        depth: Transformer 层数
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层扩展比例
        groupsize: 每个子问题的维度
        pop_size: 种群大小
        use_landscape: 是否使用景观编码器
        edge_head_dim: 边分类头隐藏维度
    """
    def __init__(
        self, n, depth, hidden_size=512, num_heads=8, mlp_ratio=4.0,
        groupsize=100, pop_size=50, use_landscape=True, edge_head_dim=64,
        **block_kwargs
    ):
        super().__init__()
        
        self.n = n
        self.use_landscape = use_landscape
        self.groupsize = groupsize
        self.pop_size = pop_size
        
        # 拓扑流嵌入
        self.topo_emb = nn.Linear(n, hidden_size)
        
        # 景观流嵌入
        if use_landscape:
            self.landscape_encoder = LandscapeEncoder(
                input_dim=groupsize * 2 + 2,
                hidden_size=hidden_size,
                pop_size=pop_size,
                mlp_hidden=[64, 128, 256]
            )
        else:
            self.weight_emb = nn.Linear(1, hidden_size)
        
        # 位置编码
        self.n_emb = nn.Embedding(n, hidden_size)
        self.register_buffer('_pos_indices', torch.arange(n))
        
        # 稀疏图 Transformer blocks
        self.blocks = nn.ModuleList([
            SparseGraphBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 边分类头 (替代旧的 Linear + Softmax 输出)
        self.edge_classifier = EdgeClassifier(hidden_size, edge_head_dim)
    
    def _topo_to_edge_index(self, topo: torch.Tensor) -> torch.Tensor:
        """
        将 batch 的稠密拓扑矩阵转为对角堆叠的 edge_index
        
        Args:
            topo: (B, N, N) 邻接矩阵
            
        Returns:
            edge_index: (2, total_E) 带 batch 偏移的稀疏边索引
        """
        B, N, _ = topo.shape
        
        # 对称化 + 二值化  
        topo_sym = (topo + topo.transpose(1, 2)) > 0  # (B, N, N)
        
        # 批量提取边: nonzero 返回 (batch_idx, src, dst)
        all_edges = topo_sym.nonzero()  # (total_E, 3)
        
        if all_edges.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=topo.device)
        
        # 加上 batch 偏移
        src = all_edges[:, 1] + all_edges[:, 0] * N
        dst = all_edges[:, 2] + all_edges[:, 0] * N
        edge_index = torch.stack([src, dst], dim=0)  # (2, total_E)
        
        return edge_index
    
    def forward(self, topo, pop_data_or_weight):
        """
        前向传播
        
        Args:
            topo: (B, N, N) 拓扑邻接矩阵
            pop_data_or_weight: 
                - use_landscape=True: (B, N, pop_size, groupsize*2+2)
                - use_landscape=False: (B, N) 权重
        
        Returns:
            p_connect: (B, N, N) 边连接概率 ∈ [0, 1]
        """
        B = topo.shape[0]
        N = self.n
        total_nodes = B * N
        
        # ===== 输入端: 稠密拓扑 → 稀疏 edge_index =====
        edge_index = self._topo_to_edge_index(topo)  # (2, total_E)
        
        # ===== 特征嵌入 =====
        # 拓扑流: 对称化邻接矩阵并嵌入
        t = self.topo_emb(topo + topo.transpose(1, 2))  # (B, N, H)
        
        # 景观流或权重流
        if self.use_landscape:
            l = self.landscape_encoder(pop_data_or_weight)  # (B, N, H)
        else:
            l = self.weight_emb(pop_data_or_weight.reshape(-1, self.n, 1))
        
        # 位置编码
        n_pos = self.n_emb(
            self._pos_indices.unsqueeze(0).expand(B, -1)
        )  # (B, N, H)
        
        # 特征融合: 拓扑 + 景观/权重 + 位置
        x = t + l + n_pos  # (B, N, H)
        
        # ===== 稀疏图 Transformer =====
        # 展平为 (B*N, H) 以配合对角堆叠的 edge_index
        x = x.reshape(total_nodes, -1)
        
        for block in self.blocks:
            x = block(x, edge_index, total_nodes)
        
        # 恢复 batch 维度: (B, N, H)
        x = x.reshape(B, N, -1)
        
        # ===== 输出端: 边分类头 → Sigmoid 概率 =====
        p_connect = self.edge_classifier(x, N)  # (B, N, N)
        
        # 对称化 (无向图)
        p_connect = (p_connect + p_connect.transpose(1, 2)) / 2
        
        # 对角线置零 (自环无意义)
        diag_idx = torch.arange(N, device=topo.device)
        p_connect[:, diag_idx, diag_idx] = 0.0
        
        # 掩码: 仅在 topo 允许的位置保留概率
        # 使用对称化后的 topo 进行掩码
        topo_sym = (topo + topo.transpose(1, 2)) > 0
        p_connect = p_connect * topo_sym.to(p_connect.dtype)
        
        return p_connect


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing Model with SparseGraphAttention (Edge-Centric)...")
    
    groupsize = 100
    input_dim = groupsize * 2 + 2
    
    # 测试新版本 (use_landscape=True)
    model = Model(10, 2, hidden_size=512, use_landscape=True, groupsize=groupsize)
    topo = torch.randint(0, 2, (4, 10, 10)).float()
    pop_data = torch.randn(4, 10, 50, input_dim)
    
    output = model(topo, pop_data)
    print(f"Input shapes: topo={topo.shape}, pop_data={pop_data.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10, 10), f"Expected (4, 10, 10), got {output.shape}"
    
    # 验证输出范围 [0, 1]
    assert output.min() >= 0.0 and output.max() <= 1.0, \
        f"Output range [{output.min():.4f}, {output.max():.4f}] not in [0, 1]"
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}] ✓")
    
    # 验证对称性
    assert torch.allclose(output, output.transpose(1, 2), atol=1e-6), \
        "Output is not symmetric!"
    print("Symmetry check ✓")
    
    # 验证对角线为零
    diag_vals = output[:, range(10), range(10)]
    assert torch.all(diag_vals == 0), "Diagonal not zero!"
    print("Diagonal zero check ✓")
    
    # 测试向后兼容版本 (use_landscape=False)
    print("\nTesting backward compatibility (use_landscape=False)...")
    model_compat = Model(10, 2, hidden_size=512, use_landscape=False)
    weight = torch.randn(4, 10)
    
    output_compat = model_compat(topo, weight)
    print(f"Compat version - Output shape: {output_compat.shape}")
    assert output_compat.shape == (4, 10, 10), f"Expected (4, 10, 10), got {output_compat.shape}"
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nOK: All tests passed!")
