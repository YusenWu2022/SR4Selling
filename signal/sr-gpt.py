import numpy as np
import pandas as pd
import sympy as sp
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import deque
import random
import json
from llm_chat import init_ds_model, ds_14b_chat, extract_code_block
# 添加TensorBoard相关导入
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化模型
try:
    model, tokenizer = init_ds_model(device_id=3)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    raise

# 符号库配置
BINARY_OPS = ['+', '-', '*', '/', '^']
UNARY_OPS = ['sin', 'cos', 'exp', 'sqrt', 'ln']
VARIABLES = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
CONSTANT = 'c'
SYMBOL_LIBRARY = BINARY_OPS + UNARY_OPS + VARIABLES + [CONSTANT]

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前表达式序列
        self.parent = parent
        self.children = []
        self.N = 0  # 访问次数
        self.Q = 0  # 动作价值
        self.P = 0  # 先验概率

class SRGPT:
    def __init__(self, data, target_col='sales_volume', max_length=20, num_simulations=100, cpuct=1.0):
        # 创建TensorBoard写入器
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("runs", f"sr_gpt_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        try:
            self.data = self.preprocess_data(data, target_col)
            self.target_col = target_col
            self.max_length = max_length
            self.num_simulations = num_simulations
            self.cpuct = cpuct
            self.memory = deque(maxlen=1000)  # 经验回放缓冲区
            self.arity_map = self._init_arity_map()
            
            # 初始化根节点
            self.root = Node([])
            self.best_expression = None
            self.best_reward = -np.inf
            logger.info("SRGPT initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SRGPT: {e}")
            raise

    def preprocess_data(self, data, target_col):
        """预处理数据：处理类别变量并归一化"""
        try:
            df = data.copy()
            
            # 目标编码类别变量
            for col in ['product_category', 'brand', 'customer_type', 'region']:
                if col in df.columns:
                    df[col] = df.groupby(col)[target_col].transform('mean')
            
            # 归一化数值特征
            numeric_cols = ['cost', 'profit', 'wholesale_price', 'retail_price', 
                           'purchase_frequency', 'last_period_purchases', 
                           'loyalty_score', 'credit_score']
            # 只处理存在的列
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
            
            # 添加变量映射
            self.feature_cols = [c for c in df.columns if c != target_col]
            self.var_mapping = {f'x{i+1}': col for i, col in enumerate(self.feature_cols)}
            
            logger.info(f"Preprocessed data with {len(df)} rows and {len(self.feature_cols)} features")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def _init_arity_map(self):
        """初始化运算符元数映射"""
        try:
            arity = {}
            for op in BINARY_OPS:
                arity[op] = 2
            for op in UNARY_OPS:
                arity[op] = 1
            for var in VARIABLES + [CONSTANT]:
                arity[var] = 0
            return arity
        except Exception as e:
            logger.error(f"Error initializing arity map: {e}")
            return {}

    def gpt_model(self, state):
        """调用大语言模型获取概率分布和状态价值，增强解析鲁棒性"""
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                # 构造提示词（添加更严格的要求）
                prompt = self.construct_robust_prompt(state)
                
                # 调用大语言模型
                response = ds_14b_chat(prompt, model, tokenizer)
                response = extract_code_block(response)
                logger.debug(f"LLM response (attempt {attempt+1}): {response[:500]}...")  # 截断长响应
                
                # 尝试1: 直接解析JSON
                try:
                    response_data = json.loads(response)
                    return self.parse_response(response_data)
                except json.JSONDecodeError:
                    pass
                
                # 尝试2: 提取JSON块
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        response = response[json_start:json_end]
                        response_data = json.loads(response)
                        return self.parse_response(response_data)
                except:
                    pass
                
                # 尝试3: 使用宽松解析
                try:
                    response_data = self.flexible_json_parse(response)
                    return self.parse_response(response_data)
                except Exception as e:
                    logger.warning(f"Flexible parse failed: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"LLM call error (attempt {attempt+1}): {e}")
        
        # 所有尝试失败后的默认值
        logger.error("All parsing attempts failed, using uniform distribution")
        p_vector = np.ones(len(SYMBOL_LIBRARY)) / len(SYMBOL_LIBRARY)
        return p_vector, 0.5

    def parse_response(self, response_data):
        """解析响应数据，处理各种格式问题"""
        # 处理键名变体
        prob_key = next((k for k in response_data.keys() 
                        if k.lower() in ["probabilities", "probs", "p"]), None)
        value_key = next((k for k in response_data.keys() 
                        if k.lower() in ["value", "val", "v"]), None)
        
        if not prob_key or not value_key:
            raise ValueError("Missing required keys in response")
        
        probs = response_data[prob_key]
        value = response_data[value_key]
        
        # 处理概率格式
        if isinstance(probs, list) and len(probs) == len(SYMBOL_LIBRARY):
            p_vector = np.array(probs)
        elif isinstance(probs, dict):
            p_vector = np.zeros(len(SYMBOL_LIBRARY))
            for i, symbol in enumerate(SYMBOL_LIBRARY):
                # 处理大小写不一致和符号别名
                symbol_key = next((k for k in probs.keys() 
                                if k.lower() == symbol.lower()), None)
                if symbol_key:
                    p_vector[i] = probs[symbol_key]
        else:
            raise ValueError("Invalid probabilities format")
        
        # 验证和归一化概率
        if np.sum(p_vector) <= 0 or np.any(p_vector < 0):
            logger.warning("Invalid probabilities, using uniform distribution")
            p_vector = np.ones(len(SYMBOL_LIBRARY)) / len(SYMBOL_LIBRARY)
        else:
            p_vector = np.maximum(p_vector, 0)  # 确保非负
            p_vector /= np.sum(p_vector)  # 归一化
        
        # 确保值在合理范围内
        value = float(value)
        if not (0 <= value <= 1):
            value = max(0.0, min(1.0, value))
            logger.warning(f"Clamped value to [0,1]: {value}")
        
        return p_vector, value

    def flexible_json_parse(self, response_str):
        """更宽松的JSON解析方法"""
        # 尝试处理单引号
        response_str = response_str.replace("'", '"')
        
        # 尝试处理尾部逗号
        response_str = re.sub(r',\s*}', '}', response_str)
        response_str = re.sub(r',\s*]', ']', response_str)
        
        # 尝试处理缺失引号
        def add_quotes(match):
            return f'"{match.group(1)}"'
        
        response_str = re.sub(r'(\b\w+\b):', add_quotes, response_str)
        
        return json.loads(response_str)

    def construct_robust_prompt(self, state):
        """构造更严格的提示词"""
        current_expr = " ".join(state) if state else "<empty>"
        
        prompt = f"""
        You are a symbolic regression assistant. Your task is to predict the next mathematical symbol in an expression sequence and evaluate the current state's potential value.
        
        ### Requirements:
        1. Output MUST contain ONLY valid JSON
        2. Use EXACTLY the following keys: "probabilities" and "value"
        3. "probabilities" must be a dictionary with ALL symbols from the library
        4. Probabilities must sum to 1.0
        5. Value must be between 0.0 and 1.0
        
        ### Symbol Library:
        {json.dumps(SYMBOL_LIBRARY, indent=2)}
        
        ### Current Expression State:
        {current_expr}
        
        ### Response Format:
        {{
            "probabilities": {{
                "{SYMBOL_LIBRARY[0]}": 0.15,
                "{SYMBOL_LIBRARY[1]}": 0.10,
                ... (all symbols in library)
            }},
            "value": 0.75
        }}
        
        ### Example Output:
        {{
            "probabilities": {{
                "+": 0.15, 
                "-": 0.10, 
                "*": 0.20,
                ... (other symbols)
            }},
            "value": 0.68
        }}
        
        ### Important:
        - Do NOT include any additional text outside the JSON
        - Do NOT use comments in the JSON
        - Ensure all symbol keys are present
        """
        
        return prompt

    def calculate_reward(self, expression):
        """计算表达式奖励 S_NRMSE-based reward"""
        try:
            # 符号表达式计算
            df = self.data.copy()
            symbol_dict = self._create_symbol_dict()
            func = sp.lambdify(list(symbol_dict.values()), expression, 'numpy')
            
            # 提取变量数据
            var_data = [df[col].values for col in self.feature_cols]
            
            # 计算预测值
            predictions = func(*var_data)
            actual = df[self.target_col].values
            
            # 计算S_NRMSE
            rmse_y = np.sqrt(mean_squared_error(actual, predictions))
            nrmse_y = rmse_y / (actual.max() - actual.min())
            
            # 变量遗漏惩罚项
            var_penalty = 0
            for i, col in enumerate(self.feature_cols):
                if f'x{i+1}' not in expression:
                    var_penalty += np.std(df[col].values)
            
            s_nrmse = nrmse_y + 0.1 * var_penalty
            return 1 / (1 + s_nrmse)
        except Exception as e:
            logger.warning(f"Error calculating reward for expression '{expression}': {e}")
            return -1  # 无效表达式

    def _create_symbol_dict(self):
        """创建符号映射字典"""
        return {var: sp.symbols(var) for var in VARIABLES[:len(self.feature_cols)]}

    def is_valid_expression(self, tokens):
        """检查表达式有效性"""
        try:
            counter = 1
            for token in tokens:
                counter += self.arity_map[token] - 1
                if counter <= 0:
                    return False
            return counter == 0
        except KeyError as e:
            logger.warning(f"Invalid token in expression: {e}")
            return False

    def mcts_search(self):
        """蒙特卡洛树搜索主循环"""
        try:
            for _ in range(self.num_simulations):
                node = self.root
                path = []
                
                # 选择阶段
                while node.children:
                    uct_values = [
                        child.Q + self.cpuct * child.P * np.sqrt(node.N) / (1 + child.N)
                        for child in node.children
                    ]
                    node = node.children[np.argmax(uct_values)]
                    path.append(node)
                
                # 扩展阶段
                if not node.children and len(node.state) < self.max_length:
                    p, v = self.gpt_model(node.state)
                    for symbol in SYMBOL_LIBRARY:
                        new_state = node.state + [symbol]
                        child = Node(new_state, parent=node)
                        child.P = p[SYMBOL_LIBRARY.index(symbol)]
                        node.children.append(child)
                
                # 评估阶段
                if not node.children:  # 叶节点
                    reward = self.calculate_reward(" ".join(node.state))
                else:
                    reward = v
                    
                # 回溯更新
                self.backpropagate(path, reward)
            return True
        except Exception as e:
            logger.error(f"Error in MCTS search: {e}")
            return False

    def backpropagate(self, path, reward):
        """回溯更新节点统计量"""
        try:
            for node in reversed(path):
                node.N += 1
                node.Q += (reward - node.Q) / node.N
        except Exception as e:
            logger.error(f"Error in backpropagation: {e}")

    def self_play(self):
        """自搜索生成表达式"""
        try:
            current_node = self.root
            expression_tokens = []
            trajectory = []
            counter = 1
            
            while counter > 0 and len(expression_tokens) < self.max_length:
                if not self.mcts_search():
                    logger.warning("MCTS search failed, breaking self-play")
                    break
                
                # 计算选择概率
                if not current_node.children:
                    logger.warning("No children available, breaking self-play")
                    break
                
                visit_counts = [child.N for child in current_node.children]
                total_visits = sum(visit_counts)
                pi = [n / total_visits for n in visit_counts] if total_visits > 0 else [1/len(visit_counts)] * len(visit_counts)
                
                # 选择动作
                idx = np.argmax(pi)
                selected_child = current_node.children[idx]
                expression_tokens.append(selected_child.state[-1])
                trajectory.append((current_node.state, pi))
                
                # 更新计数器
                counter += self.arity_map[selected_child.state[-1]] - 1
                current_node = selected_child
            
            # 计算最终奖励
            expression = " ".join(expression_tokens)
            reward = self.calculate_reward(expression)
            
            # 存储经验
            for state, pi in trajectory:
                self.memory.append((state, pi, reward))
                
            # 更新最佳表达式
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_expression = expression
                logger.info(f"New best expression: {expression} with reward {reward:.4f}")
            
            return expression, reward
        except Exception as e:
            logger.error(f"Error in self-play: {e}")
            return "", -1

    def train_gpt(self, batch_size=32):
        """训练GPT模型（模拟实现）"""
        try:
            if len(self.memory) < batch_size:
                return
            
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))
            # 实际实现中这里会更新GPT模型参数
            logger.info(f"Training GPT with {len(batch)} samples")
            return True
        except Exception as e:
            logger.error(f"Error in GPT training: {e}")
            return False

    def evaluate(self, expression):
        """评估表达式性能"""
        try:
            if not expression:
                return float('inf'), float('inf')
                
            df = self.data.copy()
            symbol_dict = self._create_symbol_dict()
            func = sp.lambdify(list(symbol_dict.values()), expression, 'numpy')
            
            var_data = [df[col].values for col in self.feature_cols]
            predictions = func(*var_data)
            actual = df[self.target_col].values
            
            mae = mean_absolute_error(actual, predictions)
            mse = mean_squared_error(actual, predictions)
            return mae, mse
        except Exception as e:
            logger.warning(f"Error evaluating expression '{expression}': {e}")
            return float('inf'), float('inf')

    def run(self, iterations=10):
        """主运行循环"""
        logger.info(f"Starting SR-GPT optimization with {iterations} iterations")
        try:
            for i in range(iterations):
                expr, reward = self.self_play()
                self.train_gpt()
                
                # 记录到TensorBoard
                self.writer.add_scalar('Reward', reward, i)
                
                if i % 1 == 0:  # 每次迭代都评估
                    mae, mse = self.evaluate(expr)
                    self.writer.add_scalar('Metrics/MAE', mae, i)
                    self.writer.add_scalar('Metrics/MSE', mse, i)
                    self.writer.add_scalar('Metrics/RMSE', np.sqrt(mse), i)
                    
                    logger.info(f"Iteration {i}: Reward={reward:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
                    logger.info(f"Current Expression: {expr}")
        
            # 最终评估
            mae, mse = self.evaluate(self.best_expression)
            self.writer.add_text('Best_Expression', self.best_expression)
            self.writer.add_scalar('Final/MAE', mae)
            self.writer.add_scalar('Final/MSE', mse)
            self.writer.add_scalar('Final/Reward', self.best_reward)
            
            logger.info(f"\nFinal Best Expression: {self.best_expression}")
            logger.info(f"MAE={mae:.4f}, MSE={mse:.4f}, Reward={self.best_reward:.4f}")
            
            return self.best_expression
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
            return ""
        finally:
            # 确保关闭TensorBoard写入器
            self.writer.close()
            logger.info("TensorBoard writer closed.")

# 示例数据
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

# 运行SR-GPT算法
try:
    srgpt = SRGPT(df)
    best_expression = srgpt.run(iterations=10)
    logger.info(f"Optimization completed. Best expression: {best_expression}")
except Exception as e:
    logger.error(f"Fatal error in SR-GPT execution: {e}")