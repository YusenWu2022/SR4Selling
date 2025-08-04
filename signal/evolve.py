import pandas as pd
import numpy as np
import random
import operator
import math
import os, datetime
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

# 完整数据集
data = {
    'product_category': ['Electronics', 'Clothing', 'Electronics', 'Furniture', 'Clothing'],
    'brand': ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE'],
    'cost': [200, 50, 150, 300, 80],
    'profit': [50, 20, 40, 70, 30],
    'wholesale_price': [180, 40, 130, 270, 70],
    'retail_price': [250, 60, 180, 350, 100],
    'purchase_frequency': [8, 3, 12, 2, 9],
    'last_period_purchases': [150, 80, 200, 50, 130],
    'loyalty_score': [85, 60, 90, 45, 75],
    'credit_score': [780, 650, 820, 580, 710],
    'customer_type': ['Regular', 'New', 'Regular', 'New', 'Regular'],
    'region': ['North', 'South', 'East', 'West', 'Central'],
    'sales_volume': [180, 85, 220, 45, 140]
}
df = pd.DataFrame(data)

# 编码分类变量
label_encoders = {}
for col in ['product_category', 'brand', 'customer_type', 'region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 特征和目标变量
features = df.columns.drop(['sales_volume', 'last_period_purchases']).tolist()
X = df[features]
y = df['sales_volume']

# 增强的运算符定义 - 更鲁棒的处理
def protected_div(a, b):
    """安全除法，避免除以零和无效值"""
    if abs(b) < 1e-6 or np.isnan(b) or np.isinf(b):
        return 1.0
    return a / b

def protected_pow(a, b):
    """安全幂运算，处理无效值和异常"""
    try:
        # 处理负数的非整数次幂
        if a < 0 and not isinstance(b, int):
            return 1.0
        result = math.pow(a, b)
        if np.isnan(result) or np.isinf(result):
            return 1.0
        return result
    except:
        return 1.0

def protected_log(a):
    """安全对数运算，处理无效值"""
    if a <= 0 or np.isnan(a) or np.isinf(a):
        return 0.0
    return math.log(a)

def protected_exp(a):
    """安全指数运算，防止溢出"""
    try:
        result = math.exp(a)
        if np.isnan(result) or np.isinf(result):
            return 1e6 if a > 0 else 1e-6
        return result
    except:
        return 1.0

operators = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': protected_div,
    '**': protected_pow,
    'log': protected_log,
    'exp': protected_exp
}

# 表达式树节点类
class Node:
    def __init__(self, op=None, left=None, right=None, value=None, feature=None):
        self.op = op
        self.left = left
        self.right = right
        self.value = value  # 常数节点
        self.feature = feature  # 特征节点
        
    def evaluate(self, row):
        """评估节点，带有鲁棒的错误处理"""
        try:
            if self.feature is not None:
                return row[self.feature]
            elif self.value is not None:
                return self.value
            else:
                left_val = self.left.evaluate(row) if self.left else 0
                right_val = self.right.evaluate(row) if self.right else 0
                
                # 处理一元运算符
                if self.op in ['log', 'exp']:
                    return operators[self.op](left_val)
                
                return operators[self.op](left_val, right_val)
        except Exception as e:
            # 记录错误但继续执行
            # print(f"Evaluation error: {e}")
            return 0.0
    
    def complexity(self):
        """计算树的复杂度"""
        if self.feature is not None or self.value is not None:
            return 1
        left_complexity = self.left.complexity() if self.left else 0
        right_complexity = self.right.complexity() if self.right else 0
        return 2 + left_complexity + right_complexity
    
    def depth(self):
        """计算树的深度"""
        if self.feature is not None or self.value is not None:
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    def __str__(self):
        if self.feature is not None:
            return self.feature
        elif self.value is not None:
            return str(round(self.value, 2))
        elif self.op in ['log', 'exp']:  # 一元运算符
            return f"{self.op}({str(self.left)})"
        else:
            return f"({str(self.left)} {self.op} {str(self.right)})"

# 随机生成表达式树
def generate_expression_tree(depth, features, constants, max_depth=6):
    """生成表达式树，带深度控制和多样性增强"""
    if depth == 0 or (depth > 0 and random.random() < 0.3) or depth >= max_depth:
        # 叶节点
        if random.random() < 0.7:  # 70%概率选择特征
            return Node(feature=random.choice(features))
        else:  # 30%概率选择常数
            return Node(value=random.uniform(-10, 10))
    else:
        # 增加一元运算符的概率
        if random.random() < 0.2:  # 20%概率生成一元运算符
            op = random.choice(['log', 'exp'])
            left = generate_expression_tree(depth-1, features, constants, max_depth)
            return Node(op=op, left=left)
        else:
            op = random.choice(list(operators.keys())[:5])  # 只选择二元运算符
            left = generate_expression_tree(depth-1, features, constants, max_depth)
            right = generate_expression_tree(depth-1, features, constants, max_depth)
            return Node(op=op, left=left, right=right)

# 初始化种群
population_size = 50  # 增加种群大小以增强多样性
population = []
features_list = features

for _ in range(population_size):
    # 随机生成表达式树（深度2-5）
    depth = random.randint(2, 5)
    tree = generate_expression_tree(depth, features_list, None)  # 不再使用预生成常数
    population.append({
        'tree': tree,
        'fitness': -np.inf,
        'mae': np.inf,
        'mse': np.inf
    })

# 计算适应度（专注于最小化MAE）
def calculate_fitness(tree, X, y, lambda_reg=0.01, lambda_depth=0.005):
    """计算适应度，带有鲁棒的异常处理"""
    predictions = []
    valid_count = 0
    
    for i in range(len(X)):
        try:
            pred = tree.evaluate(X.iloc[i])
            # 检查预测值是否有效
            if not (np.isnan(pred) or np.isinf(pred)):
                predictions.append(pred)
                valid_count += 1
            else:
                predictions.append(0)
        except:
            predictions.append(0)
    
    # 如果没有有效预测，返回极差的适应度
    if valid_count == 0:
        return -np.inf, np.inf, np.inf
    
    predictions = np.array(predictions)
    
    # 计算MAE和MSE
    mae = mean_absolute_error(y, predictions)
    # print(predictions)
    
    mse = mean_squared_error(y, predictions)
    
    # 计算复杂度惩罚
    complexity = tree.complexity()
    depth = tree.depth()
    
    # 适应度 = -MAE - 复杂度惩罚 - 深度惩罚
    fitness = -mae - lambda_reg * complexity - lambda_depth * depth
    
    return fitness, mae, mse

# 变异操作
def mutate(tree, features, constants, mutation_rate=0.3):
    """增强的变异操作，带有更多变异类型"""
    # 随机选择变异类型
    mutation_type = random.choice([
        'node_replacement', 
        'constant_perturbation', 
        'feature_replacement',
        'subtree_replacement',
        'operator_change'
    ])
    
    # 深度复制树以避免修改原始树
    tree = copy.deepcopy(tree)
    
    # 获取所有节点
    nodes = get_all_nodes(tree)
    if not nodes:
        return tree
    
    target_node = random.choice(nodes)
    
    if mutation_type == 'node_replacement':
        # 节点替换：用新子树替换目标节点
        new_depth = random.randint(1, 3)
        target_node = generate_expression_tree(new_depth, features, constants)
    
    elif mutation_type == 'constant_perturbation' and (target_node.value is not None or random.random() < 0.5):
        # 常数扰动：修改常数值
        if target_node.value is not None:
            target_node.value += random.gauss(0, 0.5)
        else:
            target_node.value = random.uniform(-10, 10)
    
    elif mutation_type == 'feature_replacement' and (target_node.feature is not None or random.random() < 0.5):
        # 特征替换：更换特征
        if target_node.feature is not None:
            target_node.feature = random.choice(features)
        else:
            target_node.feature = random.choice(features)
            target_node.value = None
    
    elif mutation_type == 'subtree_replacement':
        # 子树替换：替换整个子树
        if random.random() < 0.7:  # 70%概率替换为叶节点
            if random.random() < 0.5:
                target_node = Node(feature=random.choice(features))
            else:
                target_node = Node(value=random.uniform(-10, 10))
        else:  # 30%概率替换为新子树
            new_depth = random.randint(1, 3)
            target_node = generate_expression_tree(new_depth, features, constants)
    
    elif mutation_type == 'operator_change' and target_node.op is not None:
        # 运算符变更：更改运算符
        if target_node.op in ['log', 'exp']:  # 一元运算符
            target_node.op = random.choice(['log', 'exp'])
        else:  # 二元运算符
            target_node.op = random.choice(list(operators.keys())[:5])
    
    return tree

def get_all_nodes(node):
    """获取树的所有节点"""
    nodes = []
    stack = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        if current.left is not None:
            stack.append(current.left)
        if current.right is not None:
            stack.append(current.right)
    return nodes

# 交叉操作
def crossover(parent1, parent2):
    """子树交叉操作"""
    # 深度复制父代
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    
    # 获取父代的所有节点
    nodes1 = get_all_nodes(parent1)
    nodes2 = get_all_nodes(parent2)
    
    if not nodes1 or not nodes2:
        return parent1, parent2
    
    # 随机选择交叉点
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)
    
    # 交换子树
    crossover_point1.left, crossover_point2.left = crossover_point2.left, crossover_point1.left
    crossover_point1.right, crossover_point2.right = crossover_point2.right, crossover_point1.right
    crossover_point1.op, crossover_point2.op = crossover_point2.op, crossover_point1.op
    crossover_point1.value, crossover_point2.value = crossover_point2.value, crossover_point1.value
    crossover_point1.feature, crossover_point2.feature = crossover_point2.feature, crossover_point1.feature
    
    return parent1, parent2

# 进化算法
def evolutionary_algorithm(population, X, y, generations=100):
    """增强的进化算法，带有交叉操作和精英保留"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"evolve_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 记录历史最佳
    best_individual = None
    best_fitness = -np.inf
    best_mae = np.inf
    best_mse = np.inf
    
    # 评估初始种群
    for individual in population:
        fitness, mae, mse = calculate_fitness(individual['tree'], X, y)
        individual['fitness'] = fitness
        individual['mae'] = mae
        individual['mse'] = mse
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_mae = mae
            best_mse = mse
            best_individual = individual
    
    # 记录初始代
    avg_mae = np.mean([ind['mae'] for ind in population])
    avg_mse = np.mean([ind['mse'] for ind in population])
    writer.add_scalar('Best_MAE', best_mae, 0)
    writer.add_scalar('Best_MSE', best_mse, 0)
    writer.add_scalar('Metrics/MAE', best_mae, 0)
    writer.add_scalar('Avg_MAE', avg_mae, 0)
    writer.add_scalar('Avg_MSE', avg_mse, 0)
    
    print(f"Initial Generation, Best Fitness: {best_fitness:.2f}, Best MAE: {best_mae:.2f}, Best MSE: {best_mse:.2f}")
    
    # 进化循环
    for generation in range(generations):
        # 创建候选集（变异当前种群 + 交叉）
        candidates = []
        
        # 精英保留：保留前10%的个体
        elite_size = max(1, int(population_size * 0.1))
        elite = sorted(population, key=lambda x: x['fitness'], reverse=True)[:elite_size]
        candidates.extend(elite)
        
        # 变异：选择剩余的90%进行变异
        for individual in population[elite_size:]:
            candidate_tree = mutate(individual['tree'], features_list, None)
            candidate = {
                'tree': candidate_tree,
                'fitness': -np.inf,
                'mae': np.inf,
                'mse': np.inf
            }
            candidates.append(candidate)
        
        # 交叉：选择部分个体进行交叉
        crossover_pool = random.sample(population, min(10, len(population)))
        for i in range(0, len(crossover_pool)-1, 2):
            parent1 = crossover_pool[i]
            parent2 = crossover_pool[i+1]
            child1_tree, child2_tree = crossover(parent1['tree'], parent2['tree'])
            child1 = {'tree': child1_tree, 'fitness': -np.inf, 'mae': np.inf, 'mse': np.inf}
            child2 = {'tree': child2_tree, 'fitness': -np.inf, 'mae': np.inf, 'mse': np.inf}
            candidates.append(child1)
            candidates.append(child2)
        
        # 评估候选集
        for candidate in candidates:
            fitness, mae, mse = calculate_fitness(candidate['tree'], X, y)
            candidate['fitness'] = fitness
            candidate['mae'] = mae
            candidate['mse'] = mse
            
            # 更新全局最佳
            if fitness > best_fitness:
                best_fitness = fitness
                best_mae = mae
                best_mse = mse
                best_individual = candidate
        
        # 环境选择：合并种群和候选集，选择前population_size个
        combined = population + candidates
        combined_sorted = sorted(combined, key=lambda x: x['fitness'], reverse=True)
        population = combined_sorted[:population_size]
        
        # 计算当前种群平均MAE和MSE
        avg_mae = np.mean([ind['mae'] for ind in population])
        avg_mse = np.mean([ind['mse'] for ind in population])
        
        # 记录到TensorBoard
        writer.add_scalar('Best_MAE', best_mae, generation+1)
        writer.add_scalar('Best_MSE', best_mse, generation+1)
        writer.add_scalar('Metrics/MAE', best_mae, generation+1)
        writer.add_scalar('Avg_MAE', avg_mae, generation+1)
        writer.add_scalar('Avg_MSE', avg_mse, generation+1)
        
        # 每10代输出一次评估结果
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"Generation {generation+1}")
            print(f"  Best Fitness: {best_fitness:.2f}, Best MAE: {best_mae:.2f}, Best MSE: {best_mse:.2f}")
            print(f"  Population Avg MAE: {avg_mae:.2f}, Avg MSE: {avg_mse:.2f}")
            print(f"  Best Expression: {str(best_individual['tree'])}")
    
    # 关闭TensorBoard写入器
    writer.close()
    return best_individual

# 执行进化算法
for _ in range(5):
    best_individual = evolutionary_algorithm(population, X, y, generations=500)

# 生成可执行函数字符串
def generate_function_string(tree):
    """生成可执行的Python函数字符串，带有鲁棒的错误处理"""
    function_str = f"def predict_sales_volume(row):\n"
    function_str += f"    try:\n"
    function_str += f"        result = {str(tree)}\n"
    function_str += f"        # 确保结果为有效数字\n"
    function_str += f"        if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):\n"
    function_str += f"            return 0\n"
    function_str += f"        return result\n"
    function_str += f"    except Exception as e:\n"
    function_str += f"        # 打印错误信息用于调试\n"
    function_str += f"        # print(f\"Prediction error:\")\n"
    function_str += f"        return 0\n"
    return function_str

# 获取最佳表达式树
final_tree = best_individual['tree']
function_string = generate_function_string(final_tree)

print("\nBest Sales Volume Prediction Function:")
print(function_string)

# 评估最终模型
predictions = []
for _, row in X.iterrows():
    try:
        pred = final_tree.evaluate(row)
        # 确保预测值是有效数字
        if not (np.isnan(pred) or np.isinf(pred)):
            predictions.append(pred)
        else:
            predictions.append(0)
    except:
        predictions.append(0)

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)

print(f"\nFinal Evaluation - MAE: {mae:.2f}, MSE: {mse:.2f}")

# 测试函数字符串
exec(function_string, globals())
df['sales_volume_predict'] = df.apply(predict_sales_volume, axis=1)

# 输出结果
print("\nDataFrame with Predictions:")
print(df[['sales_volume', 'sales_volume_predict']])