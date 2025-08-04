import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from collections import deque
import random
import math
import copy
import os
import time

class UniqueReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.unique_set = set()

    def _hash(self, item):
        """为非哈希类型生成唯一的哈希值"""
        if isinstance(item, np.ndarray):
            return hash(item.data.tobytes())
        elif isinstance(item, (list, tuple)):
            return hash(tuple(self._hash(sub_item) for sub_item in item[:3]))
        else:
            return hash(item)

    def append(self, item):
        hash_value = self._hash(item)
        
        if hash_value in self.unique_set:
            return
        
        if len(self.buffer) == self.maxlen:
            removed_item = self.buffer.popleft()
            removed_hash_value = self._hash(removed_item)
            self.unique_set.remove(removed_hash_value)
        
        self.buffer.append(item)
        self.unique_set.add(hash_value)

    def __len__(self):
        return len(self.buffer)

    def sample(self, size):
        return random.sample(self.buffer, min(size, len(self.buffer)))

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据准备
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

# 数据预处理
def preprocess_data(df):
    df_processed = df.copy()
    
    # 标签编码分类变量
    cat_cols = ['product_category', 'brand', 'customer_type', 'region']
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # 分离特征和目标
    X = df_processed.drop('sales_volume', axis=1)
    y = df_processed['sales_volume']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler, cat_cols, df_processed.columns.tolist()

X, y, scaler, cat_cols, feature_names = preprocess_data(df)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)

# 表达式树环境 - 改进版
class SymbolicRegressionEnv:
    def __init__(self, X, y, max_length=15):
        self.X = X
        self.y = y
        self.max_length = max_length
        self.feature_dim = X.shape[1]
        
        # 运算符集 - 简化
        self.operators = {
            'add': (2, lambda a, b: a + b),
            'sub': (2, lambda a, b: a - b),
            'mul': (2, lambda a, b: a * b),
            'div': (2, lambda a, b: a / (torch.abs(b) + 1e-5)),
            'feature': (0, None),
            'const': (0, None)
        }
        self.operator_names = list(self.operators.keys())
        self.operator_list = self.operator_names.copy()
        
        # 预定义特征列表
        self.feature_list = list(range(self.feature_dim))
        
        # 简单表达式的基准性能
        self.baseline_mae = self.calculate_baseline_mae()
        self.reset()
        print(f"Baseline MAE: {self.baseline_mae:.4f}")
    
    def calculate_baseline_mae(self):
        """计算使用单个特征的最佳MAE"""
        best_mae = float('inf')
        for i in range(self.feature_dim):
            prediction = self.X[:, i]
            mae = torch.mean(torch.abs(prediction - self.y)).item()
            if mae < best_mae:
                best_mae = mae
        return best_mae
    
    def reset(self):
        """重置环境状态"""
        self.expression = []  # 表达式树序列
        self.stack = []       # 计算栈
        self.complexity = 0   # 表达式复杂度
        self.valid = True     # 表达式有效性
        return self.get_state()
    
    def get_state(self):
        """获取状态表示 - 增强版本"""
        state = np.zeros(20)  # 扩展到20维状态向量
        
        # 1. 当前表达式长度 (0-1)
        state[0] = len(self.expression) / self.max_length
        
        # 2. 堆栈深度 (0-1)
        state[1] = len(self.stack) / 10
        
        # 3-8: 最近操作类型 (one-hot编码)
        if self.expression:
            last_op = self.expression[-1][0]
            op_index = self.operator_names.index(last_op)
            state[3 + op_index] = 1.0
        
        # 9. 堆栈顶部值均值 (归一化)
        # 10. 堆栈顶部值标准差 (归一化)
        if self.stack:
            top_value = self.stack[-1]
            if isinstance(top_value, torch.Tensor):
                mean_val = top_value.mean().item()
                std_val = top_value.std().item()
                state[9] = np.tanh(mean_val / 100)
                state[10] = np.tanh(std_val / 100)
            else:
                state[9] = np.tanh(top_value / 100)
                state[10] = 0.0
        
        # 11-16: 操作符使用统计 (归一化)
        op_counts = [0] * len(self.operator_names)
        for op, _ in self.expression:
            op_index = self.operator_names.index(op)
            op_counts[op_index] += 1
        total_ops = len(self.expression) if self.expression else 1
        state[11:11+len(self.operator_names)] = [c/total_ops for c in op_counts]
        
        # 17. 表达式是否完整
        state[17] = 1.0 if self.is_expression_complete() else 0.0
        
        # 18. 剩余步骤比例
        state[18] = (self.max_length - len(self.expression)) / self.max_length
        
        # 19. 常数使用比例
        const_count = sum(1 for op, _ in self.expression if op == 'const')
        state[19] = const_count / total_ops
        
        return state
    
    def step(self, action):
        """执行动作并返回新状态和奖励"""
        # 动作解码：前几位是操作符，后几位是特征/常数选择
        op_idx = action % len(self.operator_names)
        op_name = self.operator_names[op_idx]
        arity, _ = self.operators[op_name]
        value = None
        
        # 处理动作
        if op_name == 'feature':
            # 允许选择具体特征
            feature_idx = (action // len(self.operator_names)) % self.feature_dim
            value = feature_idx
            self.stack.append(self.X[:, feature_idx])
        elif op_name == 'const':
            # 生成有意义的常数 (-5到5之间)
            value = np.random.uniform(-5, 5)
            self.stack.append(torch.ones(len(self.X)) * value)
        else:
            # 检查是否有足够的参数
            if len(self.stack) < arity:
                self.valid = False
                return self.get_state(), -1, True, {}
            
            # 执行操作
            args = [self.stack.pop() for _ in range(arity)]
            try:
                # 根据操作符执行计算
                if op_name == 'add':
                    result = args[0] + args[1]
                elif op_name == 'sub':
                    result = args[0] - args[1]
                elif op_name == 'mul':
                    result = args[0] * args[1]
                elif op_name == 'div':
                    result = args[0] / (torch.abs(args[1]) + 1e-5)
                self.stack.append(result)
            except Exception as e:
                self.valid = False
                return self.get_state(), -1, True, {}
        
        # 更新表达式
        self.expression.append((op_name, value))
        self.complexity += 1
        
        # 检查终止条件
        done = self.complexity >= self.max_length
        
        # 计算中间奖励 (即使表达式未完成)
        reward = 0
        intermediate_reward = 0
        
        try:
            # 如果表达式当前有效，给予中间奖励
            if self.is_expression_complete():
                prediction = self.stack[0]
                mae = torch.mean(torch.abs(prediction - self.y)).item()
                
                # 改进的奖励函数
                improvement = max(0, self.baseline_mae - mae)
                base_reward = 10 * improvement
                
                # 复杂度奖励
                complexity_reward = 0.1 * len(self.expression)
                
                # 表达式有效性奖励
                validity_reward = 2.0
                
                intermediate_reward = base_reward + complexity_reward + validity_reward
                reward = intermediate_reward
                self.valid = True
            else:
                # 部分完成的表达式给予小奖励
                reward = 0.1
        except Exception as e:
            # 出错时给予小惩罚
            reward = -0.1
            self.valid = False
        
        # 如果是终止状态，给予最终奖励
        if done:
            if self.valid and self.is_expression_complete():
                # 最终奖励：基于MAE改进和复杂度
                final_reward = max(1.0, intermediate_reward * 1.2)  # 额外奖励
                reward = final_reward
            else:
                # 无效表达式的惩罚
                reward = -1
        else:
            # 非终止状态给予探索奖励
            reward = max(reward, 0.01)
        
        return self.get_state(), reward, done, {}
    
    def is_expression_complete(self):
        """检查表达式是否完整"""
        return len(self.stack) == 1 and len(self.expression) > 0

# 神经网络架构 - 增强版
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)

# 强化学习智能体 - 增强版
class RLAgent:
    def __init__(self, env, state_dim, action_dim):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化网络
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        
        # 同步目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 优化器
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
        # 经验回放
        self.replay_buffer = UniqueReplayBuffer(maxlen=10000)
        
        # 超参数
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 128
        self.update_freq = 10  # 每10步更新一次网络
        self.target_update_freq = 20  # 每20步更新一次目标网络
    
    def select_action(self, state, epsilon=0.5):
        """使用ε-贪婪策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
            return torch.argmax(action_probs).item()
    
    def update_networks(self):
        """使用经验回放更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从回放缓冲区采样
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # 1. 更新Q网络
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_q_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        q_loss = nn.MSELoss()(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()
        
        # 2. 更新策略网络
        action_probs = self.policy_net(states)
        with torch.no_grad():
            q_values = self.q_net(states)
        
        # 使用优势函数
        advantage = q_values - q_values.mean(1, keepdim=True)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        policy_loss = -(log_probs * advantage.gather(1, actions.unsqueeze(1))).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
    
    def update_target_network(self):
        """更新目标网络"""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def train(self, num_episodes=5000):
        """训练智能体"""
        # 创建TensorBoard写入器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/rl_{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
        
        best_reward = -float('inf')
        best_expression = None
        best_mae = float('inf')
        
        # 探索参数
        epsilon = 1.0
        min_epsilon = 0.1
        epsilon_decay = 0.995
        
        # 奖励跟踪
        reward_history = []
        mae_history = []  # 记录有效表达式的MAE
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            episode_mae = None  # 本轮的MAE（如果表达式有效）
            
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                self.replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                step_count += 1
                
                # 定期更新网络
                if step_count % self.update_freq == 0:
                    self.update_networks()
                
                # 定期更新目标网络
                if step_count % self.target_update_freq == 0:
                    self.update_target_network()
            
            # 记录当前表达式的MAE（如果有效）
            if self.env.valid and self.env.is_expression_complete():
                prediction = self.env.stack[0]
                episode_mae = torch.mean(torch.abs(prediction - self.env.y)).item()
                mae_history.append(episode_mae)
            else:
                episode_mae = None
            
            # 衰减探索率
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            # 记录最佳表达式
            if total_reward > best_reward and self.env.valid and episode_mae is not None:
                best_reward = total_reward
                best_mae = episode_mae
                best_expression = copy.deepcopy(self.env.expression)
                print(f"New best reward: {best_reward:.4f} at episode {episode+1}")
                print(f"Expression: {best_expression}")
            
            # 记录奖励历史
            reward_history.append(total_reward)
            
            # 记录到TensorBoard
            writer.add_scalar('Episode/Reward', total_reward, episode)
            writer.add_scalar('Hyperparameters/Epsilon', epsilon, episode)
            
            if episode_mae is not None:
                writer.add_scalar('Metrics/MAE', episode_mae, episode)
            
            # 每100轮记录平均指标
            if episode % 100 == 0:
                avg_reward = np.mean(reward_history[-100:]) if reward_history else 0
                avg_mae = np.mean(mae_history[-100:]) if mae_history else float('inf')
                
                writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
                writer.add_scalar('Metrics/Avg_MAE', avg_mae, episode)
            
            # 每100轮打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(reward_history[-100:]) if reward_history else 0
                avg_mae = np.mean(mae_history[-100:]) if mae_history else float('inf')
                
                print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.4f}, "
                      f"Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")
                
                if episode_mae is not None:
                    print(f"  MAE: {episode_mae:.4f}, Avg MAE: {avg_mae:.4f}")
        
        # 记录最佳结果
        writer.add_scalar('Best/Reward', best_reward)
        writer.add_scalar('Best/MAE', best_mae)
        
        # 关闭TensorBoard写入器
        writer.close()
        
        print(f"Best Reward: {best_reward:.4f}, Best MAE: {best_mae:.4f}")
        return best_expression, best_reward

# 训练智能体
env = SymbolicRegressionEnv(X_train_tensor, y_train_tensor)
state_dim = len(env.get_state())
action_dim = len(env.operator_names) * (env.feature_dim + 1)  # 操作符 * (特征数 + 常数)

print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
print("Operator names:", env.operator_names)

agent = RLAgent(env, state_dim, action_dim)
best_expression, best_reward = agent.train(num_episodes=5000)

# 将最佳表达式转换为Python函数
def expression_to_function(expression, feature_names):
    """将表达式树转换为Python函数"""
    stack = []
    operators = {
        'add': (2, lambda a, b: f"({a} + {b})"),
        'sub': (2, lambda a, b: f"({a} - {b})"),
        'mul': (2, lambda a, b: f"({a} * {b})"),
        'div': (2, lambda a, b: f"({a} / ({b} + 1e-5))"),
    }
    
    # 遍历整个表达式树
    for i, (op, val) in enumerate(expression):
        if op == 'feature':
            # 确保特征索引在有效范围内
            idx = int(val)
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                stack.append(f"x['{feature_name}']")
            else:
                stack.append("0")  # 无效索引处理
        elif op == 'const':
            # 格式化常数
            stack.append(f"{val:.4f}")
        else:
            # 处理运算符
            if op in operators:
                arity, func = operators[op]
                if len(stack) < arity:
                    # 如果参数不足，使用0代替缺失参数
                    while len(stack) < arity:
                        stack.append("0")
                args = [stack.pop() for _ in range(arity)]
                
                # 二元运算符
                if arity == 2:
                    # 注意顺序：先弹出的是右操作数
                    result = func(args[1], args[0])
                    stack.append(result)
    
    # 最终结果 - 如果栈中有多个元素，将它们连接起来
    if len(stack) > 1:
        # 处理未使用的元素
        expr = " + ".join(stack)
    elif stack:
        expr = stack[0]
    else:
        expr = "0"
    
    return f"def predict(x):\n    return {expr}"
    

# 获取特征名称
feature_names = df.drop('sales_volume', axis=1).columns.tolist()

# 生成预测函数
if best_expression:
    print("\nBest Expression:", best_expression)
    func_str = expression_to_function(best_expression, feature_names)
    print("\nGenerated prediction function:")
    print(func_str)
    
    # 测试函数
    try:
        # 动态创建预测函数
        local_vars = {}
        exec(func_str, globals(), local_vars)
        predict_func = local_vars['predict']
        
        # 在数据上测试
        predictions = []
        for _, row in df.iterrows():
            predictions.append(predict_func(row))
        
        df['predicted_sales'] = predictions
        df['abs_error'] = np.abs(df['sales_volume'] - df['predicted_sales'])
        mae = df['abs_error'].mean()
        print(f"\nMean Absolute Error: {mae:.4f}")
        print(df[['sales_volume', 'predicted_sales', 'abs_error']])
    except Exception as e:
        print(f"Error testing function: {str(e)}")
else:
    print("No valid expression found during training")