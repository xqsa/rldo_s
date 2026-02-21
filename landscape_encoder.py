"""
LandscapeEncoder - PointNet + Self-Attention 风格的景观编码器

将种群数据 (位置 + 适应度 + 动态特征) 编码为固定长度的景观特征向量，
用于捕捉函数景观的几何特性（如崎岖度、单峰性、是否陷入局部最优）。

核心思想：
1. Point-wise MLP: 对每个个体独立提取局部特征
2. Self-Attention: 粒子间关系建模（参考 Deep-ELA TransformerLayer Pre-LN）
3. Symmetric Aggregation: 使用 Max + Avg Pooling 聚合种群特征（保证置换不变性）
4. 输出投影: 映射到与拓扑嵌入相同的维度

归一化策略：
- 输入格式: [pos(D), fit(1), delta_pos(D), delta_fit(1)]
- 四段独立 Min-Max 归一化到 [0, 1]，避免不同量纲互相干扰
- main_gpu.py 中仅做简单 log 变换，核心归一化由本模块负责

参考: PointNet (Qi et al., 2017), Neur-ELA, Deep-ELA
"""

import torch
import torch.nn as nn


def _safe_minmax_norm(t: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    安全的 per-group Min-Max 归一化 → [0, 1]
    
    当范围极小（收敛/所有值相同）时，输出 0.5 表示"居中的未知状态"
    
    Args:
        t: (..., pop_size, feature_dim) — 在 dim=2 (pop_size) 维度上归一化
        dim: 归一化维度
    """
    t_min = t.min(dim=dim, keepdim=True).values
    t_max = t.max(dim=dim, keepdim=True).values
    t_range = t_max - t_min
    
    is_converged = t_range < 1e-6
    t_range_safe = torch.where(is_converged, torch.ones_like(t_range), t_range)
    t_norm = (t - t_min) / t_range_safe
    t_norm = torch.where(
        is_converged.expand_as(t_norm),
        torch.full_like(t_norm, 0.5),
        t_norm
    )
    return t_norm


class LandscapeEncoder(nn.Module):
    """
    PointNet + Self-Attention 风格的景观编码器
    
    Args:
        input_dim: 输入维度 (groupsize*2 + 2: pos + fit + delta_pos + delta_fit)
        hidden_size: 输出特征维度，应与 Model 的 hidden_size 一致
        pop_size: 种群大小 (仅用于文档，不影响计算)
        mlp_hidden: MLP 隐藏层维度列表
    """
    
    def __init__(
        self, 
        input_dim: int = 202,      # 100*2+2: [pos, fit, delta_pos, delta_fit]
        hidden_size: int = 512,     # 输出维度
        pop_size: int = 50,         # 种群大小
        mlp_hidden: list = None     # MLP 隐藏层
    ):
        super().__init__()
        
        if mlp_hidden is None:
            mlp_hidden = [64, 128, 256]
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.pop_size = pop_size
        
        # Point-wise MLP: 共享权重，独立处理每个个体
        layers = []
        in_channels = input_dim
        for out_channels in mlp_hidden:
            layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.point_mlp = nn.Sequential(*layers)
        self.mlp_out_dim = mlp_hidden[-1]
        
        # Self-Attention: 粒子间关系建模
        # 参考 Deep-ELA TransformerLayer (layers.py:103-146) 的 Pre-LN 设计
        # Pre-LN: 先 LayerNorm 再 MHA/FFN，训练更稳定
        self.attn_norm = nn.LayerNorm(self.mlp_out_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.mlp_out_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.ff_norm = nn.LayerNorm(self.mlp_out_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.mlp_out_dim * 2, self.mlp_out_dim)
        )
        
        # 投影层: Max + Avg 拼接后投影到 hidden_size
        # 使用两种池化可以同时捕获"最突出特征"和"平均分布特性"
        self.projector = nn.Sequential(
            nn.Linear(self.mlp_out_dim * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行四段独立归一化
        
        输入格式: [pos(D), fit(1), delta_pos(D), delta_fit(1)]
        - D = (input_dim - 2) // 2  即 groupsize
        
        四段分别做 Min-Max → [0, 1]，避免不同量纲互相干扰：
        - pos 范围 [-100, 100] → [0, 1]
        - fit 范围 [0, 1e6+]  → [0, 1]
        - delta_pos 范围不定   → [0, 1]
        - delta_fit 范围不定   → [0, 1]
        
        当某段在种群维度上完全收敛（range < 1e-6）时，输出 0.5
        """
        input_dim = x.shape[-1]
        D = (input_dim - 2) // 2  # groupsize
        
        # 四段切片
        pos       = x[..., :D]           # (batch, n_groups, pop_size, D)
        fit       = x[..., D:D+1]        # (batch, n_groups, pop_size, 1)
        delta_pos = x[..., D+1:D*2+1]    # (batch, n_groups, pop_size, D)
        delta_fit = x[..., D*2+1:]       # (batch, n_groups, pop_size, 1)
        
        # 各段独立 Min-Max 归一化
        pos_norm       = _safe_minmax_norm(pos)
        fit_norm       = _safe_minmax_norm(fit)
        delta_pos_norm = _safe_minmax_norm(delta_pos)
        delta_fit_norm = _safe_minmax_norm(delta_fit)
        
        return torch.cat([pos_norm, fit_norm, delta_pos_norm, delta_fit_norm], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, n_groups, pop_size, input_dim)
               种群数据，最后一维为 [pos, fit, delta_pos, delta_fit]
        
        Returns:
            (batch, n_groups, hidden_size) 景观特征向量
        """
        batch_size, n_groups, pop_size, input_dim = x.shape
        
        # 1. 输入归一化（四段独立 Min-Max）
        x = self._normalize_input(x)
        
        # 2. 展平为 (batch * n_groups * pop_size, input_dim) 以应用 MLP
        x_flat = x.reshape(-1, input_dim)
        
        # 3. Point-wise MLP (x_flat 已是 (N, C) 格式，BatchNorm1d 自动处理)
        point_features = self.point_mlp(x_flat)
        
        # 4. 恢复形状: (batch * n_groups, pop_size, mlp_out_dim)
        B_G = batch_size * n_groups
        point_features = point_features.reshape(B_G, pop_size, -1)
        
        # 5. Self-Attention: 粒子间关系建模
        # 参考 Deep-ELA TransformerLayer (Pre-LN: norm -> MHA -> residual -> norm -> FF -> residual)
        # MHA 块
        attn_in = self.attn_norm(point_features)
        attn_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        point_features = point_features + attn_out  # 残差连接
        
        # FF 块
        ff_in = self.ff_norm(point_features)
        point_features = point_features + self.ff(ff_in)  # 残差连接
        
        # 6. 恢复 4D 形状
        point_features = point_features.reshape(batch_size, n_groups, pop_size, -1)
        
        # 7. Symmetric Aggregation
        # Max Pooling: 捕获最显著的特征（如最优点、最陡梯度）
        max_features = point_features.max(dim=2).values  # (batch, n_groups, mlp_out_dim)
        # Avg Pooling: 捕获整体分布特性（如收敛程度、分散度）
        avg_features = point_features.mean(dim=2)         # (batch, n_groups, mlp_out_dim)
        
        # 8. 拼接并投影
        global_features = torch.cat([max_features, avg_features], dim=-1)
        output = self.projector(global_features)  # (batch, n_groups, hidden_size)
        
        return output


class LandscapeEncoderLight(nn.Module):
    """
    轻量级景观编码器（无 BatchNorm / Dropout，适合小批量）
    
    适用于推理阶段或批量较小的场景
    包含 Self-Attention 粒子间关系建模
    """
    
    def __init__(
        self,
        input_dim: int = 202,
        hidden_size: int = 512,
        mlp_hidden: list = None
    ):
        super().__init__()
        
        if mlp_hidden is None:
            mlp_hidden = [64, 128, 256]
        
        self.input_dim = input_dim
        self.mlp_out_dim = mlp_hidden[-1]
        
        # Point-wise MLP (无 BatchNorm)
        layers = []
        in_channels = input_dim
        for out_channels in mlp_hidden:
            layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.point_mlp = nn.Sequential(*layers)
        
        # Self-Attention: 粒子间关系建模 (无 dropout 版本)
        self.attn_norm = nn.LayerNorm(self.mlp_out_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.mlp_out_dim,
            num_heads=4,
            dropout=0.0,
            batch_first=True
        )
        self.ff_norm = nn.LayerNorm(self.mlp_out_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim * 2),
            nn.GELU(),
            nn.Linear(self.mlp_out_dim * 2, self.mlp_out_dim)
        )
        
        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(mlp_hidden[-1] * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """四段独立 Min-Max 归一化（与 LandscapeEncoder 逻辑一致）"""
        input_dim = x.shape[-1]
        D = (input_dim - 2) // 2
        
        pos       = x[..., :D]
        fit       = x[..., D:D+1]
        delta_pos = x[..., D+1:D*2+1]
        delta_fit = x[..., D*2+1:]
        
        return torch.cat([
            _safe_minmax_norm(pos),
            _safe_minmax_norm(fit),
            _safe_minmax_norm(delta_pos),
            _safe_minmax_norm(delta_fit),
        ], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_groups, pop_size, input_dim = x.shape
        
        # 归一化（四段独立）
        x = self._normalize_input(x)
        
        # MLP (x_flat 已是 (N, C)，Sequential 自动处理)
        x_flat = x.reshape(-1, input_dim)
        point_features = self.point_mlp(x_flat)
        
        # Self-Attention
        B_G = batch_size * n_groups
        point_features = point_features.reshape(B_G, pop_size, -1)
        
        attn_in = self.attn_norm(point_features)
        attn_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        point_features = point_features + attn_out
        
        ff_in = self.ff_norm(point_features)
        point_features = point_features + self.ff(ff_in)
        
        point_features = point_features.reshape(batch_size, n_groups, pop_size, -1)
        
        # 池化 + 投影
        max_f = point_features.max(dim=2).values
        avg_f = point_features.mean(dim=2)
        global_f = torch.cat([max_f, avg_f], dim=-1)
        
        return self.projector(global_f)


# 测试代码
if __name__ == "__main__":
    print("Testing LandscapeEncoder with Self-Attention...")
    
    # 参数
    groupsize = 100
    input_dim = groupsize * 2 + 2  # [pos, fit, delta_pos, delta_fit]
    
    # 创建编码器
    encoder = LandscapeEncoder(input_dim=input_dim, hidden_size=512, pop_size=50)
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 测试数据: batch=4, n_groups=10, pop_size=50
    x = torch.randn(4, 10, 50, input_dim)
    
    # 前向传播
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10, 512), f"Expected (4, 10, 512), got {output.shape}"
    
    # 测试四段归一化的正确性
    print("\nTesting 4-segment normalization...")
    # 构造一个有明确结构的输入: pos in [-100,100], fit >> 0, delta_pos, delta_fit
    pos = torch.rand(1, 1, 50, groupsize) * 200 - 100     # [-100, 100]
    fit = torch.rand(1, 1, 50, 1) * 1e6                   # [0, 1e6]
    dpos = torch.randn(1, 1, 50, groupsize) * 5            # small
    dfit = torch.randn(1, 1, 50, 1) * 1000                 # medium
    x_structured = torch.cat([pos, fit, dpos, dfit], dim=-1)
    
    norm_out = encoder._normalize_input(x_structured)
    # 检查每段都在 [0, 1] 范围内
    assert norm_out.min() >= -1e-6, f"Norm min is {norm_out.min()}, expected >= 0"
    assert norm_out.max() <= 1 + 1e-6, f"Norm max is {norm_out.max()}, expected <= 1"
    # 检查 pos 部分不被 fit 压缩 (验证四段独立归一化)
    pos_norm_section = norm_out[..., :groupsize]
    pos_range = (pos_norm_section.max() - pos_norm_section.min()).item()
    assert pos_range > 0.5, f"Position normalization range too small: {pos_range}, likely contaminated by fitness"
    print(f"  pos norm range: {pos_range:.3f} (should be close to 1.0) ✓")
    print(f"  all values in [0, 1] ✓")
    
    # 测试置换不变性 (Self-Attention 保持集合操作的置换等变性)
    print("\nTesting permutation invariance...")
    x1 = torch.randn(1, 1, 50, input_dim)
    perm = torch.randperm(50)
    x2 = x1[:, :, perm, :]
    
    encoder.eval()  # 切换到评估模式（禁用 BatchNorm 的运行统计）
    with torch.no_grad():
        out1 = encoder(x1)
        out2 = encoder(x2)
    
    diff = (out1 - out2).abs().max().item()
    print(f"Max difference after permutation: {diff:.6f}")
    assert diff < 1e-4, f"Encoder should be permutation invariant, but diff={diff}"
    
    # 测试 LandscapeEncoderLight
    print("\nTesting LandscapeEncoderLight...")
    encoder_light = LandscapeEncoderLight(input_dim=input_dim, hidden_size=512)
    encoder_light.eval()
    with torch.no_grad():
        out_light = encoder_light(x1)
    print(f"Light encoder output shape: {out_light.shape}")
    assert out_light.shape == (1, 1, 512), f"Expected (1, 1, 512), got {out_light.shape}"
    
    print("\nOK: All tests passed!")
