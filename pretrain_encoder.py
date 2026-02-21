"""
LandscapeEncoder 预训练脚本 — 对比学习 + 景观类型分类

独立于 RL 训练循环，参考:
- OPAL train_opal_meta.py: 辅助景观类型分类 (AUX_LAMBDA=0.3, CrossEntropy)
- Deep-ELA: Transformer Encoder 训练策略
- Neur-ELA: Feature_Extractor 独立训练后迁移到 RL agent

训练任务:
A. 对比学习 (InfoNCE/NT-Xent): 同函数不同旋转 → 近, 不同函数 → 远
B. 景观类型预测 (BCEWithLogitsLoss): 单峰 vs 多峰

使用方法:
    python pretrain_encoder.py                    # 默认配置
    python pretrain_encoder.py --epochs 50        # 自定义 epoch
    python pretrain_encoder.py --epochs 2 --batch-size 4  # 快速冒烟测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import time

from landscape_encoder import LandscapeEncoder
import config as cfg


# ==================== 函数景观生成器 ====================

def generate_sphere_landscape(batch_size, n_groups, pop_size, groupsize, device, 
                                rotation=None):
    """
    生成 Sphere 函数景观 + DE 探测轨迹（单峰）
    f(x) = sum(R @ (x - x_opt))^2
    """
    positions = torch.rand(batch_size, n_groups, pop_size, groupsize, device=device) * 200 - 100
    
    # 可选旋转
    if rotation is not None:
        # rotation: (groupsize, groupsize)
        pos_flat = positions.reshape(-1, groupsize)  # (B*G*P, D)
        pos_flat = pos_flat @ rotation.T
        positions = pos_flat.reshape(batch_size, n_groups, pop_size, groupsize)
    
    fitness = (positions ** 2).sum(dim=-1, keepdim=True)
    
    # DE 探测
    pos_init = positions.clone()
    fit_init = fitness.clone()
    for _ in range(cfg.PROBE_STEPS):
        idx_a = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_b = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_c = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        
        xa = positions.gather(2, idx_a.expand(-1, -1, -1, groupsize))
        xb = positions.gather(2, idx_b.expand(-1, -1, -1, groupsize))
        xc = positions.gather(2, idx_c.expand(-1, -1, -1, groupsize))
        
        donor = xa + cfg.PROBE_DE_F * (xb - xc)
        cr_mask = torch.rand_like(positions) < cfg.PROBE_DE_CR
        j_rand = torch.randint(0, groupsize, (batch_size, n_groups, pop_size, 1), device=device)
        cr_mask.scatter_(-1, j_rand, True)
        trial = torch.where(cr_mask, donor, positions).clamp(-100, 100)
        
        if rotation is not None:
            trial_rot = trial.reshape(-1, groupsize) @ rotation.T
            trial_fit = (trial_rot ** 2).sum(dim=-1, keepdim=True)
            trial_fit = trial_fit.reshape(batch_size, n_groups, pop_size, 1)
        else:
            trial_fit = (trial ** 2).sum(dim=-1, keepdim=True)
        
        improved = trial_fit < fitness
        positions = torch.where(improved, trial, positions)
        fitness = torch.where(improved, trial_fit, fitness)
    
    delta_pos = positions - pos_init
    delta_fit = fitness - fit_init
    
    return torch.cat([positions, fitness, delta_pos, delta_fit], dim=-1)


def generate_rastrigin_landscape(batch_size, n_groups, pop_size, groupsize, device,
                                    rotation=None):
    """
    生成 Rastrigin 函数景观 + DE 探测轨迹（多峰）
    f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))
    """
    positions = torch.rand(batch_size, n_groups, pop_size, groupsize, device=device) * 10.24 - 5.12
    
    if rotation is not None:
        pos_flat = positions.reshape(-1, groupsize)
        pos_flat = pos_flat @ rotation.T
        positions = pos_flat.reshape(batch_size, n_groups, pop_size, groupsize)
    
    def rastrigin(p):
        return (10 * groupsize + (p ** 2 - 10 * torch.cos(2 * np.pi * p)).sum(dim=-1, keepdim=True))
    
    fitness = rastrigin(positions)
    
    pos_init = positions.clone()
    fit_init = fitness.clone()
    for _ in range(cfg.PROBE_STEPS):
        idx_a = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_b = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_c = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        
        xa = positions.gather(2, idx_a.expand(-1, -1, -1, groupsize))
        xb = positions.gather(2, idx_b.expand(-1, -1, -1, groupsize))
        xc = positions.gather(2, idx_c.expand(-1, -1, -1, groupsize))
        
        donor = xa + cfg.PROBE_DE_F * (xb - xc)
        cr_mask = torch.rand_like(positions) < cfg.PROBE_DE_CR
        j_rand = torch.randint(0, groupsize, (batch_size, n_groups, pop_size, 1), device=device)
        cr_mask.scatter_(-1, j_rand, True)
        trial = torch.where(cr_mask, donor, positions).clamp(-5.12, 5.12)
        trial_fit = rastrigin(trial)
        
        improved = trial_fit < fitness
        positions = torch.where(improved, trial, positions)
        fitness = torch.where(improved, trial_fit, fitness)
    
    delta_pos = positions - pos_init
    delta_fit = fitness - fit_init
    
    return torch.cat([positions, fitness, delta_pos, delta_fit], dim=-1)


def generate_ackley_landscape(batch_size, n_groups, pop_size, groupsize, device,
                                rotation=None):
    """
    生成 Ackley 函数景观 + DE 探测轨迹（多峰）
    """
    positions = torch.rand(batch_size, n_groups, pop_size, groupsize, device=device) * 65.536 - 32.768
    
    if rotation is not None:
        pos_flat = positions.reshape(-1, groupsize)
        pos_flat = pos_flat @ rotation.T
        positions = pos_flat.reshape(batch_size, n_groups, pop_size, groupsize)
    
    def ackley(p):
        d = p.shape[-1]
        sum_sq = (p ** 2).mean(dim=-1, keepdim=True)
        sum_cos = torch.cos(2 * np.pi * p).mean(dim=-1, keepdim=True)
        return -20 * torch.exp(-0.2 * torch.sqrt(sum_sq)) - torch.exp(sum_cos) + 20 + np.e
    
    fitness = ackley(positions)
    
    pos_init = positions.clone()
    fit_init = fitness.clone()
    for _ in range(cfg.PROBE_STEPS):
        idx_a = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_b = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_c = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        
        xa = positions.gather(2, idx_a.expand(-1, -1, -1, groupsize))
        xb = positions.gather(2, idx_b.expand(-1, -1, -1, groupsize))
        xc = positions.gather(2, idx_c.expand(-1, -1, -1, groupsize))
        
        donor = xa + cfg.PROBE_DE_F * (xb - xc)
        cr_mask = torch.rand_like(positions) < cfg.PROBE_DE_CR
        j_rand = torch.randint(0, groupsize, (batch_size, n_groups, pop_size, 1), device=device)
        cr_mask.scatter_(-1, j_rand, True)
        trial = torch.where(cr_mask, donor, positions).clamp(-32.768, 32.768)
        trial_fit = ackley(trial)
        
        improved = trial_fit < fitness
        positions = torch.where(improved, trial, positions)
        fitness = torch.where(improved, trial_fit, fitness)
    
    delta_pos = positions - pos_init
    delta_fit = fitness - fit_init
    
    return torch.cat([positions, fitness, delta_pos, delta_fit], dim=-1)


def generate_elliptic_landscape(batch_size, n_groups, pop_size, groupsize, device,
                                  rotation=None):
    """
    生成 Elliptic 函数景观 + DE 探测轨迹（单峰，强条件数）
    f(x) = sum(10^(6*i/(D-1)) * x_i^2)
    """
    positions = torch.rand(batch_size, n_groups, pop_size, groupsize, device=device) * 200 - 100
    
    if rotation is not None:
        pos_flat = positions.reshape(-1, groupsize)
        pos_flat = pos_flat @ rotation.T
        positions = pos_flat.reshape(batch_size, n_groups, pop_size, groupsize)
    
    coeffs = torch.pow(torch.tensor(10.0, device=device), 
                       6.0 * torch.arange(groupsize, device=device, dtype=torch.float32) / max(groupsize - 1, 1))
    
    def elliptic(p):
        return (coeffs * p ** 2).sum(dim=-1, keepdim=True)
    
    fitness = elliptic(positions)
    
    pos_init = positions.clone()
    fit_init = fitness.clone()
    for _ in range(cfg.PROBE_STEPS):
        idx_a = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_b = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        idx_c = torch.randint(0, pop_size, (batch_size, n_groups, pop_size, 1), device=device)
        
        xa = positions.gather(2, idx_a.expand(-1, -1, -1, groupsize))
        xb = positions.gather(2, idx_b.expand(-1, -1, -1, groupsize))
        xc = positions.gather(2, idx_c.expand(-1, -1, -1, groupsize))
        
        donor = xa + cfg.PROBE_DE_F * (xb - xc)
        cr_mask = torch.rand_like(positions) < cfg.PROBE_DE_CR
        j_rand = torch.randint(0, groupsize, (batch_size, n_groups, pop_size, 1), device=device)
        cr_mask.scatter_(-1, j_rand, True)
        trial = torch.where(cr_mask, donor, positions).clamp(-100, 100)
        trial_fit = elliptic(trial)
        
        improved = trial_fit < fitness
        positions = torch.where(improved, trial, positions)
        fitness = torch.where(improved, trial_fit, fitness)
    
    delta_pos = positions - pos_init
    delta_fit = fitness - fit_init
    
    return torch.cat([positions, fitness, delta_pos, delta_fit], dim=-1)


# ==================== 随机旋转矩阵 ====================

def random_rotation_matrix(dim, device):
    """生成随机正交旋转矩阵 (QR 分解)"""
    H = torch.randn(dim, dim, device='cpu')
    Q, _ = torch.linalg.qr(H)
    return Q.to(device)


# ==================== 数据生成 ====================

# 函数注册表: (生成函数, 是否多峰)
LANDSCAPE_REGISTRY = [
    (generate_sphere_landscape,   False),  # 单峰
    (generate_elliptic_landscape, False),  # 单峰
    (generate_rastrigin_landscape, True),  # 多峰
    (generate_ackley_landscape,    True),  # 多峰
]


def generate_contrastive_batch(batch_size, n_groups, pop_size, groupsize, device):
    """
    生成对比学习训练批次
    
    Returns:
        anchor:   (batch_size, n_groups, pop_size, input_dim) — 原始景观
        positive: (batch_size, n_groups, pop_size, input_dim) — 同函数不同旋转
        negative: (batch_size, n_groups, pop_size, input_dim) — 不同函数
        labels:   (batch_size,) — 0=单峰, 1=多峰
    """
    anchors = []
    positives = []
    negatives = []
    labels = []
    
    for _ in range(batch_size):
        # 选择 anchor 函数
        func_idx = np.random.randint(0, len(LANDSCAPE_REGISTRY))
        func, is_multimodal = LANDSCAPE_REGISTRY[func_idx]
        
        # 选择不同函数作为 negative
        neg_idx = np.random.randint(0, len(LANDSCAPE_REGISTRY))
        while neg_idx == func_idx:
            neg_idx = np.random.randint(0, len(LANDSCAPE_REGISTRY))
        neg_func, _ = LANDSCAPE_REGISTRY[neg_idx]
        
        # 生成两个不同的旋转矩阵
        R1 = random_rotation_matrix(groupsize, device)
        R2 = random_rotation_matrix(groupsize, device)
        
        anchor = func(1, n_groups, pop_size, groupsize, device, rotation=R1)
        positive = func(1, n_groups, pop_size, groupsize, device, rotation=R2)
        negative = neg_func(1, n_groups, pop_size, groupsize, device, rotation=R1)
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        labels.append(1.0 if is_multimodal else 0.0)
    
    return (
        torch.cat(anchors, dim=0),
        torch.cat(positives, dim=0),
        torch.cat(negatives, dim=0),
        torch.tensor(labels, device=device, dtype=torch.float32)
    )


# ==================== 损失函数 ====================

def info_nce_loss(anchor_emb, positive_emb, negative_emb, temperature=0.07):
    """
    InfoNCE / NT-Xent 对比损失
    
    Args:
        anchor_emb:   (B, D) — anchor embedding (通过 mean pooling 所有 groups)
        positive_emb: (B, D) — positive embedding
        negative_emb: (B, D) — negative embedding
        temperature:  温度参数
    
    Returns:
        loss: scalar
    """
    # L2 归一化
    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_emb = F.normalize(negative_emb, dim=-1)
    
    # 正样本相似度
    pos_sim = (anchor_emb * positive_emb).sum(dim=-1) / temperature  # (B,)
    
    # 负样本相似度矩阵 (每个 anchor 与所有 negative 的相似度)
    neg_sim = torch.mm(anchor_emb, negative_emb.T) / temperature  # (B, B)
    
    # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+B)
    labels = torch.zeros(anchor_emb.shape[0], dtype=torch.long, device=anchor_emb.device)
    
    return F.cross_entropy(logits, labels)


# ==================== 预训练头 ====================

class PretrainHead(nn.Module):
    """预训练辅助头: 景观类型分类 (单峰 vs 多峰)"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """x: (B, hidden_size) → logit: (B,)"""
        return self.classifier(x).squeeze(-1)


# ==================== 主训练函数 ====================

def pretrain_encoder(args):
    """预训练 LandscapeEncoder"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=" * 60)
    print(f"LandscapeEncoder Contrastive Pretraining")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"AUX_LAMBDA: {args.aux_lambda}")
    
    groupsize = cfg.GROUPSIZE
    n_groups = cfg.N_GROUPS
    pop_size = cfg.POP_SIZE
    input_dim = groupsize * 2 + 2
    hidden_size = cfg.HIDDEN_SIZE
    
    print(f"\nEncoder config:")
    print(f"  groupsize={groupsize}, n_groups={n_groups}, pop_size={pop_size}")
    print(f"  input_dim={input_dim}, hidden_size={hidden_size}")
    
    # 创建模型
    encoder = LandscapeEncoder(
        input_dim=input_dim,
        hidden_size=hidden_size,
        pop_size=pop_size,
        mlp_hidden=[64, 128, 256]
    ).to(device)
    
    cls_head = PretrainHead(hidden_size).to(device)
    
    params = list(encoder.parameters()) + list(cls_head.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    total_params = sum(p.numel() for p in params)
    print(f"  total parameters: {total_params:,}")
    
    # 训练循环
    print(f"\n{'-' * 60}")
    print("Starting pretraining...")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        cls_head.train()
        
        t0 = time.time()
        
        # 生成对比学习数据
        anchor, positive, negative, labels = generate_contrastive_batch(
            args.batch_size, n_groups, pop_size, groupsize, device
        )
        
        # 前向传播: 获取 landscape embeddings
        anchor_emb = encoder(anchor).mean(dim=1)      # (B, n_groups, H) → (B, H)
        positive_emb = encoder(positive).mean(dim=1)
        negative_emb = encoder(negative).mean(dim=1)
        
        # 任务 A: 对比损失
        loss_contrastive = info_nce_loss(
            anchor_emb, positive_emb, negative_emb, 
            temperature=args.temperature
        )
        
        # 任务 B: 景观类型分类 (参考 OPAL AUX_LAMBDA)
        cls_logits = cls_head(anchor_emb)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, labels)
        
        # 总损失
        loss = loss_contrastive + args.aux_lambda * loss_cls
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        dt = time.time() - t0
        
        # 分类准确率
        with torch.no_grad():
            preds = (torch.sigmoid(cls_logits) > 0.5).float()
            accuracy = (preds == labels).float().mean().item()
        
        # 日志
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={loss.item():.4f} "
                  f"(contrastive={loss_contrastive.item():.4f}, "
                  f"cls={loss_cls.item():.4f}) | "
                  f"acc={accuracy:.3f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{dt:.1f}s")
        
        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(encoder.state_dict(), args.output)
    
    print(f"\n{'=' * 60}")
    print(f"Pretraining complete! Best loss: {best_loss:.4f}")
    print(f"Saved encoder weights to: {args.output}")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Set USE_PRETRAINED_ENCODER = True in config.py")
    print(f"  2. Set PRETRAIN_WEIGHT_PATH = '{args.output}'")
    print(f"  3. Run: python main_gpu.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LandscapeEncoder Pretraining')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs (default: {cfg.PRETRAIN_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {cfg.PRETRAIN_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {cfg.PRETRAIN_LR})')
    parser.add_argument('--temperature', type=float, default=None,
                       help=f'InfoNCE temperature (default: {cfg.PRETRAIN_TEMPERATURE})')
    parser.add_argument('--aux-lambda', type=float, default=None,
                       help=f'Classification loss weight (default: {cfg.PRETRAIN_AUX_LAMBDA})')
    parser.add_argument('--output', type=str, default=None,
                       help=f'Output path (default: {cfg.PRETRAIN_WEIGHT_PATH})')
    
    args = parser.parse_args()
    
    # 使用 config.py 中的默认值
    if args.epochs is None: args.epochs = cfg.PRETRAIN_EPOCHS
    if args.batch_size is None: args.batch_size = cfg.PRETRAIN_BATCH_SIZE
    if args.lr is None: args.lr = cfg.PRETRAIN_LR
    if args.temperature is None: args.temperature = cfg.PRETRAIN_TEMPERATURE
    if args.aux_lambda is None: args.aux_lambda = cfg.PRETRAIN_AUX_LAMBDA
    if args.output is None: args.output = cfg.PRETRAIN_WEIGHT_PATH
    
    pretrain_encoder(args)
