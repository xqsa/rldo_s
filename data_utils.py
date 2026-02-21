import numpy as np
from collections import defaultdict, deque

# BBOB 测试套件导入（提供多样化的适应度景观）
from bbob import (
    F1, F2, F3, F8, F10, F15, F21,  # 精选 BBOB 函数
    rotate_gen, BBOB_Basic_Problem
)

# 精选的 BBOB 函数列表（覆盖不同特征）
BBOB_FUNCTIONS = {
    'sphere': F1,           # 基础：单峰
    'ellipsoidal': F2,      # 病态条件
    'rastrigin': F3,        # 多峰 + 规则网格
    'rosenbrock': F8,       # 狭长山谷
    'ellipsoidal_hc': F10,  # 高病态条件 (10^6)
    'rastrigin_rot': F15,   # 旋转多峰
    'gallagher': F21,       # 101 峰，极度多峰
}


def create_bbob_problem(func_key, dim, shifted=True, rotated=True, bounds=(-100, 100)):
    """
    创建 BBOB 问题实例
    
    Args:
        func_key: BBOB 函数键名
        dim: 问题维度
        shifted: 是否添加偏移
        rotated: 是否添加旋转
        bounds: 搜索空间边界 (lower, upper)，偏移将在此范围内生成
    
    Returns:
        BBOB 问题实例
    """
    import numpy as np
    lb, ub = bounds
    
    if shifted:
        # [FIX] 偏移在传入的 bounds 范围内生成，确保最优解在搜索空间内
        shift = 0.8 * (lb + (ub - lb) * np.random.rand(dim))
    else:
        shift = np.zeros(dim)
    
    if rotated:
        H = rotate_gen(dim)
    else:
        H = np.eye(dim)
    
    return BBOB_FUNCTIONS[func_key](dim=dim, shift=shift, rotate=H, bias=0, lb=lb, ub=ub)



def checkGraphConnectivity(adjMatrix):

    numNodes = adjMatrix.shape[0]
    visited = np.zeros(numNodes, dtype=bool)
    queue = deque([0])
    visited[0] = True

    while queue:
        currentNode = queue.popleft()
        neighbors = np.where(adjMatrix[currentNode, :] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    isConnected = np.all(visited)
    
    return isConnected, visited

def groupmake_strict(A, groupsize=100):
    group_num = A.shape[0]
    group_vars = [set() for _ in range(group_num)]
    next_var_id = 0

    # 1. 强制共享变量
    for i in range(group_num):
        for j in range(i + 1, group_num):
            k = int(A[i, j])
            if k > 0:
                shared_vars = set(range(next_var_id, next_var_id + k))
                next_var_id += k
                group_vars[i].update(shared_vars)
                group_vars[j].update(shared_vars)

    # 2. 补齐私有变量
    for i in range(group_num):
        needed = groupsize - len(group_vars[i])
        if needed < 0:
            raise ValueError(
                f"group {i} has more than {groupsize} shared variables"
            )
        private_vars = set(range(next_var_id, next_var_id + needed))
        next_var_id += needed
        group_vars[i].update(private_vars)

    # 3. 重新编号到 [0, D)
    all_vars = sorted(set().union(*group_vars))
    var_map = {v: idx for idx, v in enumerate(all_vars)}

    allgroups = []
    for i in range(group_num):
        allgroups.append([var_map[v] for v in group_vars[i]])

    return allgroups

def computeRotation(D):
    A = np.random.randn(D, D)
    Q, R = np.linalg.qr(A)
    return Q, R


def generate_mock_population(
    allgroups, 
    D, 
    pop_size=50, 
    groupsize=100,
    mode='random',
    bounds=(-100, 100),
    fitness_mode='bbob_mixed'  # 新增：使用 BBOB 适应度函数
):
    """
    生成模拟种群数据用于训练
    
    为每个子问题生成模拟的种群位置和适应度值，用于训练景观编码器。
    
    Args:
        allgroups: 分组信息，每个元素是该组包含的变量索引列表
        D: 总决策变量维度
        pop_size: 种群大小
        groupsize: 每个子问题的维度
        mode: 分布模式
            - 'random': 均匀分布 (模拟初始化阶段)
            - 'converged': 正态分布集中 (模拟收敛阶段)
            - 'multimodal': 多峰分布 (模拟多局部最优)
            - 'mixed': 随机混合以上模式
        bounds: 搜索空间边界 (lower, upper)
    
    Returns:
        pop_data: (n_groups, pop_size, groupsize+1) 
                  每组的种群位置 + 适应度
    """
    n_groups = len(allgroups)
    lower, upper = bounds
    
    pop_data = np.zeros((n_groups, pop_size, groupsize + 1))
    
    for g_idx in range(n_groups):
        if mode == 'mixed':
            # 随机选择一种模式
            current_mode = np.random.choice(['random', 'converged', 'multimodal'])
        else:
            current_mode = mode
        
        if current_mode == 'random':
            # 均匀分布 - 模拟初始化阶段
            positions = lower + (upper - lower) * np.random.rand(pop_size, groupsize)
            
        elif current_mode == 'converged':
            # 正态分布 - 模拟收敛阶段
            center = lower + (upper - lower) * np.random.rand(groupsize)
            std = (upper - lower) * 0.05  # 5% 范围的标准差
            positions = np.random.normal(center, std, size=(pop_size, groupsize))
            positions = np.clip(positions, lower, upper)
            
        elif current_mode == 'multimodal':
            # 多峰分布 - 模拟多局部最优
            n_clusters = np.random.randint(2, 5)  # 2-4 个簇
            cluster_centers = lower + (upper - lower) * np.random.rand(n_clusters, groupsize)
            cluster_std = (upper - lower) * 0.1  # 10% 范围的标准差
            
            # 为每个个体随机分配一个簇
            cluster_ids = np.random.randint(0, n_clusters, size=pop_size)
            positions = np.zeros((pop_size, groupsize))
            for i in range(pop_size):
                positions[i] = np.random.normal(
                    cluster_centers[cluster_ids[i]], 
                    cluster_std
                )
            positions = np.clip(positions, lower, upper)
        
        else:
            raise ValueError(f"Unknown mode: {current_mode}")
        
        # 生成模拟适应度值 - 使用 BBOB 函数提供多样化景观
        if fitness_mode == 'bbob_mixed':
            # 随机选择一种 BBOB 函数
            func_key = np.random.choice(list(BBOB_FUNCTIONS.keys()))
            problem = create_bbob_problem(func_key, groupsize, shifted=True, rotated=True, bounds=bounds)
            fitness = problem.eval(positions).reshape(-1, 1)
        elif fitness_mode in BBOB_FUNCTIONS:
            # 使用指定的 BBOB 函数
            problem = create_bbob_problem(fitness_mode, groupsize, bounds=bounds)
            fitness = problem.eval(positions).reshape(-1, 1)
        else:
            # 向后兼容：使用简单 Sphere
            fitness = np.sum(positions ** 2, axis=1, keepdims=True)
        
        # 组合位置和适应度
        pop_data[g_idx] = np.concatenate([positions, fitness], axis=1)
    
    return pop_data


def data_generator(problem_batch_size, G_num, G_prob, pop_size=50, pop_mode='mixed'):
    """
    数据生成器 - 生成训练数据
    
    Args:
        problem_batch_size: 批量大小
        G_num: 子问题数量
        G_prob: 边连接概率
        pop_size: 种群大小 (新增)
        pop_mode: 种群分布模式 (新增): 'random', 'converged', 'multimodal', 'mixed'
    
    Returns:
        dict: 包含 topology, weights, pop_data, details
    """
    topo_list = []
    w_list = []
    pop_data_list = []  # 新增: 种群数据
    xopt_list = []
    xopt_1_list = []
    allgroups_list = []
    D_list = []
    R100_list = []
    group_num = G_num  # 10
    G_prob = G_prob    # 0.1
    
    for func in range(1, problem_batch_size + 1):

        # [P1] 已移除 np.random.seed(None)，随机种子应由外部统一控制

        A = np.zeros((group_num, group_num))

        for i in range(group_num):
            for j in range(group_num):
                if i != j:
                    A[i, j] = np.random.rand() < G_prob  #链接概率

        for i in range(group_num):
            if np.all(A[i, :] == 0) and np.all(A[:, i] == 0):
                numbers = list(range(group_num))
                numbers.remove(i)
                selected = np.random.choice(numbers)
                A[i, selected] = 1

        for i in range(group_num):
            for j in range(group_num):
                if i != j and A[i, j] == 1:
                    A[j, i] = 1

        connect, visited = checkGraphConnectivity(A)

        while not connect:           
            # 随机选择一个未访问的节点
            unvisitedIndices = np.where(~visited)[0]
            selected_node = np.random.choice(unvisitedIndices)          
            # 随机选择一个已访问的节点
            visitedIndices = np.where(visited)[0]  # 获取所有已访问的节点
            selected_neighbor = np.random.choice(visitedIndices)  # 从已访问的节点中随机选择一个节点
            # 将未访问节点与已访问节点连接
            A[selected_node, selected_neighbor] = 1
            A[selected_neighbor, selected_node] = 1  # 无向图，双向连接
            # 重新检查图的连通性
            connect, visited = checkGraphConnectivity(A)

        A = A.astype(int)
        A = np.triu(A)
        
        for i in range(group_num):
            for j in range(group_num):
                if A[i, j] == 1:
                    A[i, j] = np.random.randint(1, 11)
                    
        topo_list.append(A)

        w = np.zeros(group_num)
        index = np.random.permutation(group_num)
        for i in range(group_num):
            w[i] = 10 ** (3 * np.random.normal(0, 1))  #CEC 2010
        w_list.append(w)

        allgroups = groupmake_strict(A, groupsize=100)
        allgroups_list.append(allgroups)
        allElements = list(set([item for sublist in allgroups for item in sublist]))
        D_list.append(len(allElements))
        
        lengh_allElements = len(allElements)
        xopt = -100 + 2 * 100 * np.random.rand(lengh_allElements)
        xopt_1 = -100 + 2 * 100 * np.random.rand(1000)
        xopt_list.append(xopt)
        xopt_1_list.append(xopt_1)

    
        R100, _ = computeRotation(100)
        R100_list.append(R100)
        
        # 生成模拟种群数据
        pop_data = generate_mock_population(
            allgroups=allgroups,
            D=lengh_allElements,
            pop_size=pop_size,
            groupsize=100,
            mode=pop_mode
        )
        pop_data_list.append(pop_data)
        
    detail = {
        "allgroups_list": allgroups_list,
        "xopt_list": xopt_list,
        "xopt_1_list": xopt_1_list,
        "D_list": D_list,
        "R100_list": R100_list,
    }

    # 使用结构化的字典格式返回，便于后续使用
    output = {
        'topology': np.array(topo_list),      # 拓扑矩阵
        'weights': np.array(w_list),          # 权重 (保留用于向后兼容)
        'pop_data': np.array(pop_data_list),  # 种群数据 (新增)
        'details': detail                     # 详细信息
    }

    return output


# 测试代码
if __name__ == "__main__":
    print("Testing data_generator with population data...")
    
    # 生成测试数据
    data = data_generator(
        problem_batch_size=2, 
        G_num=10, 
        G_prob=0.3,
        pop_size=50,
        pop_mode='mixed'
    )
    
    print(f"Topology shape: {data['topology'].shape}")
    print(f"Weights shape: {data['weights'].shape}")
    print(f"Pop data shape: {data['pop_data'].shape}")
    
    # 验证形状
    assert data['topology'].shape == (2, 10, 10), f"Expected (2, 10, 10), got {data['topology'].shape}"
    assert data['weights'].shape == (2, 10), f"Expected (2, 10), got {data['weights'].shape}"
    assert data['pop_data'].shape == (2, 10, 50, 101), f"Expected (2, 10, 50, 101), got {data['pop_data'].shape}"
    
    print("\nOK: All tests passed!")
