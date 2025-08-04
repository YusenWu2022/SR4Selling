import pandas as pd
import numpy as np
import re
import ast
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error
from llm_chat import ds_14b_chat, init_ds_model, extract_code_block
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import logging
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

use_llm = True
model, tokenizer = init_ds_model(device_id=4) if use_llm else (None, None)

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
df_example = pd.DataFrame(data)

# 1. 初始提示词
prompt_initial = """
You are now tasked with building a decision tree model to predict sales volume based on customer and SKU data.
Please generate a Python function that takes a DataFrame row as input and returns the predicted sales volume.
The function should be based on the following features:
- SKU features: product_category, brand, cost, profit, wholesale_price, retail_price
- Customer features: purchase_frequency, last_period_purchases, loyalty_score, credit_score, customer_type, region
Among them types of these features are discrete string: product_category, brand, customer_type, region; and types of these features are 
integers: cost, profit, wholesale_price, purchase_frequency, loyalty_score, credit_score, last_period_purchases.

The decision tree should have a reasonable depth and complexity (3-5 levels maximum).
Use only basic arithmetic operations and conditional logic.

Please also provide a textual visualization of the decision tree.

Example function format:
def predict_sales(row):
    if row['cost'] > 100:
        if row['purchase_frequency'] > 5:
            return row['last_period_purchases'] * 1.2
        else:
            return row['last_period_purchases'] * 0.8
    else:
        if row['loyalty_score'] > 70:
            return row['last_period_purchases'] * 1.5
        else:
            return row['last_period_purchases'] * 0.9
    return row['last_period_purchases']

After generating the decision tree function, output the model's initial predictions versus the actual values table and calculate error metrics (MAE, MSE).
"""

# 2. 后续迭代提示词模板
prompt_iteration_template = """
You are now tasked with building a decision tree model to predict sales volume based on customer and SKU data.
Please generate a Python function that takes a DataFrame row as input and returns the predicted sales volume.
The function should be based on the following features:
- SKU features: product_category, brand, cost, profit, wholesale_price, retail_price
- Customer features: purchase_frequency, last_period_purchases, loyalty_score, credit_score, customer_type, region
Among them types of these features are discrete string: product_category, brand, customer_type, region; and types of these features are 
integers: cost, profit, wholesale_price, purchase_frequency, loyalty_score, credit_score, last_period_purchases.

Based on the previous decision tree model and prediction results, please reflect and improve the model.

Previous model performance:
- MAE: {mae:.2f}
- MSE: {mse:.2f}

Previous model code:
{previous_model}

Error analysis:
{error_analysis}

Please modify the decision tree function to improve accuracy. Focus on:
1. Adjusting threshold values in conditions
2. Simplifying complex branches
3. Adding/removing features based on importance
4. Improving arithmetic expressions

Output only the new Python function code in a markdown code block.
"""

# 3. 最终总结提示词
prompt_final = """
Summarize the optimization process:
1. Initial model approach
2. Major improvements in each iteration
3. Error reduction trend
4. Final model performance
5. Key feature importance
6. Model limitations and future improvements

Output the summary in a structured format.
"""

# 增强型参数提取函数
def extract_parameters(func_str):
    """
    从函数字符串中提取所有常数参数
    返回: 参数名称列表, 初始值列表, 参数位置字典
    """
    # 生成函数内容的哈希值用于缓存
    func_hash = hashlib.md5(func_str.encode()).hexdigest()[:8]
    
    parameters = {}
    param_positions = {}
    
    try:
        # 解析函数为AST
        tree = ast.parse(func_str)
        
        # 遍历AST节点查找常数
        for node in ast.walk(tree):
            # 处理数字常量
            if isinstance(node, ast.Num):
                value = node.n
                lineno = node.lineno
                col_offset = node.col_offset
                
                # 创建唯一参数名
                param_name = f"param_{func_hash}_{len(parameters)}"
                parameters[param_name] = value
                param_positions[param_name] = (lineno, col_offset)
            
            # 处理字符串常量（条件判断中的阈值）
            elif isinstance(node, ast.Str) and node.s.replace('.', '', 1).isdigit():
                try:
                    value = float(node.s)
                    lineno = node.lineno
                    col_offset = node.col_offset
                    
                    param_name = f"param_{func_hash}_{len(parameters)}"
                    parameters[param_name] = value
                    param_positions[param_name] = (lineno, col_offset)
                except ValueError:
                    continue
            
            # 处理比较操作中的常量
            elif isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Num):
                        value = comparator.n
                        lineno = comparator.lineno
                        col_offset = comparator.col_offset
                        
                        param_name = f"param_{func_hash}_{len(parameters)}"
                        parameters[param_name] = value
                        param_positions[param_name] = (lineno, col_offset)
    
    except SyntaxError as e:
        logger.error(f"Syntax error in function: {e}")
        return [], [], {}
    
    return list(parameters.keys()), list(parameters.values()), param_positions

# 改进型参数化函数创建
def create_parameterized_function(func_str, param_names):
    """
    创建带参数占位符的函数字符串
    """
    # 使用更安全的替换方法，避免部分匹配
    paramized_func = func_str
    for param in param_names:
        # 只替换完整的单词匹配
        paramized_func = re.sub(rf"(?<!\w){re.escape(param)}(?!\w)", f"params['{param}']", paramized_func)
    return paramized_func

# 增强型参数优化
def optimize_parameters(func_str, df, target_col, optimization_method='evolutionary'):
    """
    优化函数中的常数参数
    """
    try:
        # 提取参数
        param_names, initial_values, param_positions = extract_parameters(func_str)
        
        if not param_names:
            logger.info("No parameters found for optimization.")
            return func_str, [], [], float('inf')
        
        logger.info(f"Found {len(param_names)} parameters to optimize: {param_names}")
        
        # 创建参数化函数
        paramized_func = create_parameterized_function(func_str, param_names)
        
        # 编译参数化函数
        namespace = {'pd': pd, 'np': np}
        namespace['params'] = {name: val for name, val in zip(param_names, initial_values)}
        exec(paramized_func, namespace)
        predict_func = namespace.get('predict_sales', None)
        
        if not predict_func or not callable(predict_func):
            raise ValueError("Failed to compile parameterized function")
        
        # 定义安全的损失函数
        def loss_function(params_vector):
            try:
                # 更新参数
                for i, name in enumerate(param_names):
                    namespace['params'][name] = params_vector[i]
                
                # 进行预测
                predictions = []
                for _, row in df.iterrows():
                    try:
                        pred = predict_func(row)
                        predictions.append(float(pred))
                    except Exception as e:
                        logger.warning(f"Prediction error: {e}")
                        predictions.append(0.0)
                
                y_true = df[target_col].values
                y_pred = np.array(predictions)
                
                # 处理无效值
                valid_mask = np.isfinite(y_pred)
                if not np.any(valid_mask):
                    return float('inf')
                
                return mean_absolute_error(y_true[valid_mask], y_pred[valid_mask])
            
            except Exception as e:
                logger.error(f"Error in loss function: {e}")
                return float('inf')
        
        # 设置参数边界 (0.1x to 10x 初始值)
        bounds = [(min_val, max_val) for val in initial_values
                 for min_val, max_val in [(max(0.01, 0.1 * abs(val)), 10 * abs(val))]]
        
        # 优化参数
        if optimization_method == 'gradient' and len(param_names) < 10:
            # 梯度下降优化 - 适合少量参数
            result = minimize(
                loss_function, 
                initial_values, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100}
            )
            optimized_values = result.x
            loss = result.fun
        else:
            # 进化算法优化 - 更鲁棒，适合复杂问题
            result = differential_evolution(
                loss_function, 
                bounds, 
                strategy='best1bin',
                maxiter=200,
                popsize=min(15, max(5, len(param_names)*2)),
                tol=0.01,
                polish=True
            )
            optimized_values = result.x
            loss = result.fun
        
        # 用优化后的参数替换原函数中的常数
        optimized_func = func_str
        lines = optimized_func.split('\n')
        
        # 按行号倒序替换，避免位置变化影响
        replacements = []
        for name, opt_val in zip(param_names, optimized_values):
            lineno, col_offset = param_positions[name]
            if lineno <= len(lines):
                # 格式化优化值
                if abs(opt_val) > 1000 or abs(opt_val) < 0.001:
                    formatted_val = f"{opt_val:.4e}"
                elif abs(opt_val - round(opt_val)) < 1e-6:
                    formatted_val = str(int(round(opt_val)))
                else:
                    formatted_val = f"{opt_val:.6f}".rstrip('0').rstrip('.')
                
                replacements.append((lineno, col_offset, formatted_val))
        
        # 从后往前替换
        replacements.sort(key=lambda x: (-x[0], -x[1]))
        
        for lineno, col_offset, new_val in replacements:
            line = lines[lineno-1]
            # 查找要替换的数字
            # 支持匹配各种数字格式：整数、浮点数、科学计数法
            match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', line[col_offset:])
            if match:
                start = col_offset + match.start()
                end = col_offset + match.end()
                lines[lineno-1] = line[:start] + new_val + line[end:]
        
        optimized_func = '\n'.join(lines)
        
        initial_loss = loss_function(initial_values)
        logger.info(f"Optimization completed. Loss reduced from {initial_loss:.4f} to {loss:.4f}")
        return optimized_func, initial_values, optimized_values, loss
    
    except Exception as e:
        logger.error(f"Parameter optimization failed: {e}")
        return func_str, initial_values if 'initial_values' in locals() else [], [], float('inf')

# 鲁棒的函数解析和预测
def parse_and_predict(df, model_function_str, target_col):
    original_df = df.copy()
    max_retries = 3
    retry_count = 0
    temp_col = f"predicted_temp_{np.random.randint(10000)}"
    
    while retry_count < max_retries:
        try:
            # 进行参数优化
            optimized_func, init_params, opt_params, loss = optimize_parameters(
                model_function_str, df, target_col, optimization_method='evolutionary'
            )
            
            # 创建安全的执行环境
            safe_globals = {
                'pd': pd,
                'np': np,
                '__builtins__': {
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'str': str,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round
                }
            }
            
            # 编译优化后的函数
            exec(optimized_func, safe_globals)
            predict_sales = safe_globals.get('predict_sales')
            
            if not predict_sales or not callable(predict_sales):
                raise ValueError("predict_sales function not found or not callable")
            
            # 应用函数进行预测
            df[temp_col] = df.apply(lambda row: predict_sales(row), axis=1)
            
            # 重命名列
            df = df.rename(columns={temp_col: 'predicted_sales'})
            
            return df, optimized_func, init_params, opt_params
        
        except Exception as e:
            logger.error(f"Parse and predict error (attempt {retry_count+1}): {e}")
            retry_count += 1
            df = original_df.copy()
    
    logger.error("Failed to parse and predict after maximum retries")
    return original_df, model_function_str, [], []

# 增强型错误分析报告
def generate_error_analysis(df_with_predictions):
    try:
        df = df_with_predictions.copy()
        df['absolute_error'] = np.abs(df['sales_volume'] - df['predicted_sales'])
        df['error_ratio'] = df['absolute_error'] / df['sales_volume'].clip(lower=1)
        
        # 识别高误差样本
        high_error = df[df['error_ratio'] > 0.3].copy()
        high_error_features = []
        
        # 分析特征模式
        for col in df.columns:
            if col not in ['predicted_sales', 'sales_volume', 'absolute_error', 'error_ratio']:
                # 计算高误差组特征分布
                if high_error.shape[0] > 0:
                    high_error_features.append({
                        'feature': col,
                        'high_error_mean': high_error[col].mean(),
                        'overall_mean': df[col].mean()
                    })
        
        # 特征重要性分析
        feature_importance = []
        for col in ['cost', 'profit', 'purchase_frequency', 'last_period_purchases', 
                   'loyalty_score', 'credit_score']:
            if col in df:
                corr = df[['predicted_sales', col]].corr().iloc[0, 1]
                feature_importance.append({
                    'feature': col,
                    'correlation': corr
                })
        
        # 错误模式总结
        analysis = {
            'high_error_samples': high_error.to_dict(orient='records'),
            'feature_comparison': high_error_features,
            'feature_importance': sorted(feature_importance, key=lambda x: abs(x['correlation']), reverse=True),
            'suggestions': [
                "Adjust thresholds in conditions related to cost and loyalty_score",
                "Consider adding interactions between purchase_frequency and region",
                "Simplify branching logic in electronics category predictions"
            ]
        }
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        return {"error": str(e)}

# 决策树优化主流程
def decision_tree_optimization(df, target_col, threshold=10, max_iterations=5):
    # 创建TensorBoard写入器
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"decision_tree_opt_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    iteration_history = []
    current_iteration = 0
    previous_model = None
    error_analysis = None
    mae_history = []
    mse_history = []
    
    while current_iteration < max_iterations:

        logger.info(f"Starting iteration {current_iteration}")
        
        
        # 生成决策树函数
        if current_iteration == 0:
            prompt = prompt_initial
        else:
            prompt = prompt_iteration_template.format(
                previous_model=previous_model,
                mae=mae_history[-1],
                mse=mse_history[-1],
                error_analysis=str(error_analysis)
            )
        
        while True:
            try:
        
                model_function_str = generate_decision_tree_function(prompt)
                logger.info(f"Generated model:\n{model_function_str}")
                
            
                # 解析并预测
                df_with_predictions, optimized_func, init_params, opt_params = parse_and_predict(
                    df.copy(), model_function_str, target_col
                )
                
                # 计算指标
                y_true = df_with_predictions[target_col]
                y_pred = df_with_predictions['predicted_sales']
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                
                break
            except:
                continue
            
        
        # 记录历史
        mae_history.append(mae)
        mse_history.append(mse)
        
        # TensorBoard记录
        writer.add_scalar('Metrics/MAE', mae, current_iteration)
        writer.add_scalar('Metrics/MSE', mse, current_iteration)
        
        # 生成错误分析
        error_analysis = generate_error_analysis(df_with_predictions)
        
        # 保存迭代历史
        iteration_record = {
            'iteration': current_iteration,
            'model': optimized_func,
            'mae': mae,
            'mse': mse,
            'parameters': {
                'initial': init_params,
                'optimized': opt_params
            },
            'error_analysis': error_analysis
        }
        iteration_history.append(iteration_record)
        
        # 输出结果
        logger.info(f"Iteration {current_iteration}: MAE={mae:.2f}, MSE={mse:.2f}")
        logger.info(f"Top features: {[f['feature'] for f in error_analysis.get('feature_importance', [])[:3]]}")
        
        # 检查停止条件
        if mae < threshold:
            logger.info(f"Stopping early: MAE {mae:.2f} < threshold {threshold}")
            break
        
        # 准备下一轮
        previous_model = optimized_func
        current_iteration += 1
    
    # 生成最终总结
    final_summary = generate_final_summary(iteration_history)
    iteration_history.append({'final_summary': final_summary})
    
    # 可视化优化过程
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(mae_history, 'o-', label='MAE')
        plt.plot(mse_history, 's-', label='MSE')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'optimization_progress.png'))
        writer.add_figure('Optimization Progress', plt.gcf())
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    
    # 关闭TensorBoard
    writer.close()
    logger.info("Optimization completed")
    
    return iteration_history

# 决策树函数生成
def generate_decision_tree_function(prompt):
    logger.info("Generating decision tree function...")
    
    if use_llm and model:
        response = ds_14b_chat(prompt, model, tokenizer)
        code = extract_code_block(response)
        
        # 验证函数格式
        if "def predict_sales(row):" not in code:
            logger.warning("Generated code missing function definition. Adding default.")
            code = "def predict_sales(row):\n    return row['last_period_purchases']"
        
        return code
    else:
        # 模拟生成
        if "initial" in prompt:
            return """
def predict_sales(row):
    if row['cost'] > 100:
        if row['purchase_frequency'] > 5:
            return row['last_period_purchases'] * 1.2
        else:
            return row['last_period_purchases'] * 0.8
    else:
        if row['loyalty_score'] > 70:
            return row['last_period_purchases'] * 1.5
        else:
            return row['last_period_purchases'] * 0.9
    return row['last_period_purchases']
"""
        else:
            return """
def predict_sales(row):
    if row['cost'] > 150:
        if row['purchase_frequency'] > 7 and row['region'] in ['North', 'East']:
            return row['last_period_purchases'] * 1.3
        else:
            return row['last_period_purchases'] * 0.9
    elif row['product_category'] == 'Electronics':
        return row['last_period_purchases'] * 1.1
    else:
        return row['last_period_purchases'] * 0.85
"""

# 最终总结生成
def generate_final_summary(iteration_history):
    logger.info("Generating final summary...")
    
    if use_llm and model:
        history_str = "\n".join(
            f"Iteration {i}: MAE={rec['mae']:.2f}, Changes: {rec['error_analysis'].get('suggestions', [''])}"
            for i, rec in enumerate(iteration_history)
        )
        return ds_14b_chat(prompt_final + "\nHistory:\n" + history_str, model, tokenizer)
    else:
        # 模拟总结
        return {
            "initial_mae": iteration_history[0]['mae'] if iteration_history else "N/A",
            "final_mae": iteration_history[-1]['mae'] if iteration_history else "N/A",
            "improvements": [
                "Added region-based conditions",
                "Optimized cost thresholds",
                "Simplified electronics category handling"
            ],
            "key_features": ["cost", "purchase_frequency", "region"],
            "limitations": "Model struggles with new product categories",
            "recommendations": "Incorporate seasonal factors and marketing data"
        }

# 执行优化
if __name__ == "__main__":
    for _ in range(5):
        optimization_history = decision_tree_optimization(
            df=df_example,
            target_col='sales_volume',
            threshold=-5,
            max_iterations=5000
        )
    
    # 保存结果
    import json
    with open('optimization_history.json', 'w') as f:
        json.dump(optimization_history, f, indent=4, default=str)
    
    print("Optimization completed. Results saved to optimization_history.json")