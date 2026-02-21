"""
RLDO 配置文件

集中管理所有可配置参数
"""

# ==================== 模型配置 ====================
USE_LANDSCAPE = True      # 是否使用 Deep ELA 景观感知
HIDDEN_SIZE = 512         # Transformer 隐藏层维度
DEPTH = 2                 # Transformer 层数
NUM_HEADS = 8             # 注意力头数

# ==================== 优化问题配置 ====================
N_GROUPS = 10             # 子问题数量
GROUPSIZE = 100           # 每个子问题的维度
G_PROB = 0.3              # 子问题间边连接概率

# ==================== 种群配置 ====================
POP_SIZE = 50             # 种群大小
POP_MODE = 'mixed'        # 种群分布模式: 'random', 'converged', 'multimodal', 'mixed'

# ==================== 训练配置 ====================
PROBLEM_BATCH_SIZE = 256   # GPU 优化: 增大到 64 以充分利用并行能力 (原: 8)
REPEAT = 7                # 每个问题的采样次数
TRAIN_BATCH_SIZE = 256     # 训练批量大小
LEARNING_RATE = 1e-5      # 学习率 [修复] 降低 10 倍，避免大梯度导致参数飞出
N_EPOCHS = 200            # 训练轮数
PPO_CLIP = 0.1            # PPO 裁剪系数
REUSE_TIME = 10           # 数据重用次数
ENTROPY_COEF = 0.05       # PPO 熵正则系数 [修复] 提高 5 倍，强迫探索避免躺平
LOG_INTERVAL = 10         # 训练日志输出间隔（epoch）
PREPROCESS_POP_DATA = True   # [修复] 必须开启！对 fitness 取 log 防止数值爆炸

# ==================== 监督学习辅助配置 ====================
# [P2 新增] "老师傅带路"：用真实拓扑监督模型输出，帮助 RL 跨越初期探索门槛
# - SUPERVISED_COEF: 监督损失权重，越大越强调"跟老师傅画图"
# - SUPERVISED_DECAY: 权重衰减率，让模型逐渐从"模仿"过渡到"自主探索"
# - SUPERVISED_MIN: 最小权重，保留一定的监督信号防止策略漂移
USE_SUPERVISED_LOSS = True    # 是否启用监督学习辅助
SUPERVISED_COEF = 1.0         # 初始监督损失权重
SUPERVISED_DECAY = 0.99       # 每 epoch 衰减率 (200 epochs 后约 0.13)
SUPERVISED_MIN = 0.1          # 最小监督权重 (保留 10% 防止遗忘)

# ==================== DECC 优化配置 ====================
MAX_FES = 50000           # 最大函数评估次数 (基础值，训练/推理模式会覆盖)
DE_F = 0.5                # 差分进化变异因子
DE_CR = 0.3               # 差分进化交叉概率

# ==================== 稀疏图注意力 + Bernoulli PPO 配置 ====================
EDGE_HEAD_DIM = 64        # 边分类头的隐藏维度
TOPO_ENTROPY_COEF = 0.05  # 拓扑熵正则化系数 (Bernoulli 分布熵, 鼓励探索)

# ==================== 动态轨迹配置 (参考 OPAL design_phase) ====================
PROBE_STEPS = 5           # DE 探测步数 (生成动态轨迹)
PROBE_DE_F = 0.5          # 探测用 DE 变异因子
PROBE_DE_CR = 0.9         # 探测用 DE 交叉概率
FITNESS_NORM_MODE = 'rank' # 适应度归一化: 'rank'(OPAL), 'zscore'(OPAL), 'minmax'(Neur-ELA), 'log'(原始)

# ==================== 预训练配置 (参考 OPAL 辅助分类设计) ====================
PRETRAIN_EPOCHS = 100     # 预训练轮数
PRETRAIN_BATCH_SIZE = 64  # 预训练批量大小
PRETRAIN_LR = 1e-3        # 预训练学习率
PRETRAIN_TEMPERATURE = 0.07  # InfoNCE 温度参数
PRETRAIN_AUX_LAMBDA = 0.3   # 辅助分类损失权重 (参考 OPAL AUX_LAMBDA=0.3)
PRETRAIN_WEIGHT_PATH = 'encoder_pretrained.pth'  # 预训练权重保存路径
USE_PRETRAINED_ENCODER = False  # 是否加载预训练权重 (首次训练设为 False)
FREEZE_ENCODER_LAYERS = 2   # 迁移学习时冻结前 N 层 point_mlp

# ==================== 系统配置 ====================
MAX_WORKERS = 8           # 最大并行进程数 (设为 None 则自动检测)
DEVICE = 'auto'           # 计算设备: 'cuda', 'cpu', 'auto'
RANDOM_SEED = None        # 随机种子 (设为整数以复现实验)

# ==================== 路径配置 ====================
MODEL_SAVE_DIR = '.'      # 模型保存目录
CHECKPOINT_INTERVAL = 5   # 保存检查点的间隔 (epochs)

# ==================== Ground Truth 评估配置 ====================
# [P1 修复] 解决奖励黑客问题：分离"优化分组"和"评估分组"
# - True: 使用真实拓扑(topos)计算fitness，模型只能学习真正有效的分组策略
# - False: 使用模型预测的分组计算fitness（存在reward hacking风险）
USE_GT_EVALUATION = True

# ==================== 快速验证模式 ====================
# 设置为 True 以使用更小的参数进行快速验证
FAST_MODE = False

# ==================== 两步走训练模式 ====================
# [核心修复] 分离"快速训练"和"高精度推理"
# - True:  训练模式 (Float32 + 小 FES)，快速迭代让 edge_rate -> 1.0
# - False: 推理模式 (Float64 + 大 FES)，冲击最低 Fitness
TRAINING_MODE = True


# ==================== 模式覆盖逻辑（可重复调用） ====================
# 说明：历史实现依赖"import 时一次性覆盖"。为了支持推理脚本在运行时强制切换 TRAINING_MODE，
# 这里提供可重复调用的覆盖函数，并在模块 import 时调用一次以保持原行为。
_BASE_DEFAULTS = {
    'MAX_FES': MAX_FES,
    'POP_SIZE': POP_SIZE,
    'PROBLEM_BATCH_SIZE': PROBLEM_BATCH_SIZE,
    'REPEAT': REPEAT,
    'REUSE_TIME': REUSE_TIME,
    'LEARNING_RATE': LEARNING_RATE,
    'N_EPOCHS': N_EPOCHS,
    'LOG_INTERVAL': LOG_INTERVAL,
}


def apply_mode_overrides() -> None:
    """
    根据 FAST_MODE / TRAINING_MODE 重新计算覆盖参数。

    重要：该函数会先将可覆盖参数恢复到"基础默认值"，再应用模式覆盖，
    因此可以在运行时多次调用而不会发生"越改越偏"的累积漂移。
    """
    global MAX_FES, POP_SIZE, PROBLEM_BATCH_SIZE, REPEAT, REUSE_TIME
    global LEARNING_RATE, N_EPOCHS, LOG_INTERVAL

    # 1) 先恢复基础默认值（避免重复调用造成累积覆盖）
    for k, v in _BASE_DEFAULTS.items():
        globals()[k] = v

    # 2) 快速模式覆盖参数
    if FAST_MODE:
        MAX_FES = 10000           # 大幅降低评估次数
        N_EPOCHS = 5              # 降低训练轮数
        PROBLEM_BATCH_SIZE = 2    # 降低批量大小
        REPEAT = 2                # 降低采样次数
        REUSE_TIME = 2            # 降低重用次数
        return

    # 3) 训练模式覆盖参数
    if TRAINING_MODE:
        MAX_FES = 50000           # 足够区分分组质量
        POP_SIZE = 50             # 轻量种群，加速迭代
        PROBLEM_BATCH_SIZE = 128  # 加大并行量
        LOG_INTERVAL = 1          # 每 epoch 输出，方便观察 edge_rate
        N_EPOCHS = 200            # 足够让 edge_rate -> 1.0
        return

    # 4) 推理模式覆盖参数（infer_gpu.py 可在运行时强制切换）
    MAX_FES = 3000000         # 300万次评估，冲击极限 Fitness
    POP_SIZE = 200            # 重兵投入
    PROBLEM_BATCH_SIZE = 16   # 推理不需要太多并行 (Float64 显存翻倍)
    N_EPOCHS = 1              # 只跑 1 轮 (无训练)
    LEARNING_RATE = 0.0       # 关闭梯度更新
    LOG_INTERVAL = 1


# 模块导入时应用一次覆盖，保持历史默认行为
apply_mode_overrides()
