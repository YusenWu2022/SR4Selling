import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from llm_chat import ds_14b_chat, init_ds_model, extract_code_block
# 添加TensorBoard相关导入
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


use_llm = True
model, tokenizer = init_ds_model(device_id=0)

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

# 1. 初始提示词：构建初始决策树函数
prompt_initial = """
You are now tasked with building a decision tree model to predict sales volume based on customer and SKU data.
Please generate a Python function that takes a DataFrame row as input and returns the predicted sales volume.
The function should be based on the following features:
- SKU features: product_category, brand, cost, profit, wholesale_price, retail_price, last_period_purchases
- Customer features: purchase_frequency, loyalty_score, credit_score, customer_type, region
Among them types of these features are discrete string: product_category, brand, customer_type, region; and types of these features are 
integers: cost, profit, wholesale_price, purchase_frequency, loyalty_score, credit_score, last_period_purchases.

The decision tree should have a reasonable depth and complexity.
Please also provide a textual visualization of the decision tree.

Here is an example of the function format:
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

Notice the only requirements of the function is the return should be l column calculated by given columns (you can use + - * / ** and other const values, and the calculation
operation can be put between different original columns. In addition, you can add multiple if-else to the function. All the logic in the function should be added according to 
what the major loss is in history results)

After generating the decision tree function, output the model's initial predictions versus the actual values table and calculate dmetrics (MAE, MSE).
"""

# 2. 后续迭代提示词模板（基于上一轮反思结果）
prompt_iteration_template = """
You are now tasked with building a decision tree model to predict sales volume based on customer and SKU data.
Please generate a Python function that takes a DataFrame row as input and returns the predicted sales volume.
The function should be based on the following features:
- SKU features: product_category, brand, cost, profit, wholesale_price, retail_price
- Customer features: purchase_frequency, last_period_purchases, loyalty_score, credit_score, customer_type, region
Among them types of these features are discrete string: product_category, brand, customer_type, region; and types of these features are 
integers: cost, profit, wholesale_price, purchase_frequency, loyalty_score, credit_score, last_period_purchases.

Based on the previous decision tree model and prediction results, please reflect on the following points:

1. Analyze instances with large prediction errors to identify feature combinations causing deviation
2. Check for model overfitting or underfitting
3. Identify potentially overlooked important features or feature interactions
4. Evaluate the decision tree's branching logic
5. Consider adjusting the decision tree's depth or complexity

Please modify the decision tree function based on the above reflection. The modifications should include:
1. Adjusted branching conditions
2. New or removed features
3. Adjusted termination conditions
4. Improved prediction logic

After modifying, please make new predictions and output the new prediction results versus actual values, as well as the change in error metrics.

Also, record the reflections of this round and the possible causes.

Here is the previous decision tree function:
{previous_model}

Here are the previous prediction results with errors:
{prediction_results_with_errors}

Here are the previous error metrics:
MAE: {mae:.2f}
MSE: {mse:.2f}

Reflections from previous iteration:
{reflections}
"""

# 3. 最终总结提示词
prompt_final = """
Now please summarize the entire iterative process, including:
1. The initial model construction approach
2. Major improvements in each iteration
3. Analysis of the error trend
4. Final model performance evaluation
5. Key feature identification and feature importance ranking
6. Model limitations and suggestions for improvement

Please record the complete decision tree optimization process, including the reflections of each round and possible causes.
"""

# 4. 决策树函数解析与预测
def parse_and_predict(df, model_function_str):
    # 将模型函数字符串解析为可执行的函数
    namespace = {}
    exec(model_function_str, namespace)
    predict_sales = namespace.get('predict_sales', lambda row: 0)
    
    # 应用函数进行预测
    df['predicted_sales'] = df.apply(lambda row: predict_sales(row), axis=1)
    
    return df

# 5. 生成反思报告
def generate_reflection_report(mae, mse, model_function_str, df_with_predictions):
    
    reflection_prompt_template = """
    Task: Generate a reflection report for a decision tree model
    Current Model Code:
    {model_function}

    Performance Metrics:
    MAE: {mae:.2f}

    MSE: {mse:.2f}

    Current Prediction is {df_with_predictions}

    Analysis Guidelines:
    Error Patterns: Identify any patterns in high-error cases (e.g., specific categories, price ranges)
    Model Limitations: Diagnose model weaknesses (e.g., over-simplification, missing interactions)
    Feature Analysis: Evaluate feature importance and suggest underutilized features
    Complexity Assessment: Determine if the model is overfitting or underfitting
    Actionable Insights: Propose specific, measurable improvements

    Output Format:
    ## Error Analysis
    - Key observations about prediction errors
    - Patterns in high-error cases

    ## Model Limitations
    - 2-3 main weaknesses of the current model
    - Potential causes of these limitations

    ## Improvement Recommendations
    - 3-5 concrete suggestions for model refinement
    - Specific features or interactions to incorporate
    - Complexity adjustments (depth, branching, etc.)
    Note: Focus on actionable insights for the next iteration.
    """
    
    reflection_prompt = reflection_prompt_template.format(
                        mse=mse,
                        mae=mae,
                        model_function=model_function_str,
                        df_with_predictions=str(df_with_predictions)
                    )
    if use_llm:
        reflection = ds_14b_chat(reflection_prompt, model, tokenizer)
    else:
        # 模拟反思报告
        y_true = df_with_predictions['sales_volume']
        y_pred = df_with_predictions['predicted_sales']
        error = y_true - y_pred
        
        # 计算误差比例
        absolute_error = np.abs(error)
        max_error = np.max(absolute_error)
        threshold = 0.1  # 误差阈值，可根据需要调整
        
        # 分析误差较大的实例
        large_errors = df_with_predictions[absolute_error > threshold * np.mean(absolute_error)]
        
        reflection = {
            'large_error_instances': large_errors.to_dict(orient='records'),
            'possible_causes': [
                "某些特征组合可能导致预测偏差",
                "模型可能在某些区域过拟合或欠拟合",
                "可能忽略了一些重要的特征或特征交互"
            ],
            'improvement_suggestions': [
                "考虑增加决策树的深度",
                "尝试添加新的特征或特征交互",
                "调整决策树的分支条件"
            ]
        }
    
    return reflection

# 6. 决策树优化主流程
def decision_tree_optimization(df, target_col, threshold=10, max_iterations=5):
    # 创建TensorBoard写入器
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"only_llm_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    iteration_history = []
    current_iteration = 0
    previous_model = None
    prediction_results_with_errors = None
    mae = None
    mse = None
    reflections = None
    
    while current_iteration < max_iterations:
        # 保存当前迭代开始前的状态（用于错误时重置）
        reset_point = {
            'previous_model': previous_model,
            'prediction_results_with_errors': prediction_results_with_errors,
            'mae': mae,
            'mse': mse,
            'reflections': reflections
        }
        
        # 内部重试循环
        retry_count = 0
        max_retries = 3
        success = False
        
        while not success and retry_count < max_retries:
            try:
                if current_iteration == 0:
                    print("Generating initial decision tree function...")
                    model_function_str = generate_decision_tree_function(prompt_initial)
                    print(model_function_str)
                else:
                    print(f"Iteration {current_iteration}: Generating improved decision tree function...")
                    iteration_prompt = prompt_iteration_template.format(
                        previous_model=previous_model,
                        prediction_results_with_errors=prediction_results_with_errors.to_string(),
                        mae=mae,
                        mse=mse,
                        reflections=reflections
                    )
                    model_function_str = generate_decision_tree_function(iteration_prompt)
                    print(model_function_str)
                
                # 解析决策树函数并进行预测
                df_with_predictions = parse_and_predict(df, model_function_str)
                
                # 计算预测结果与真实值的对比
                y_true = df_with_predictions[target_col]
                y_pred = df_with_predictions['predicted_sales']
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                
                # 记录到TensorBoard
                writer.add_scalar('Metrics/MAE', mae, current_iteration)
                writer.add_scalar('Metrics/MSE', mse, current_iteration)
                writer.add_scalar('Metrics/RMSE', np.sqrt(mse), current_iteration)
                print(f"Recorded iteration {current_iteration}: MAE={mae:.4f}, MSE={mse:.4f} to TensorBoard")
                
                # 计算误差并添加到数据框
                df_with_predictions['absolute_error'] = np.abs(y_true - y_pred)
                mean_abs_error = np.mean(np.abs(y_true - y_pred))
                
                # 避免除以零错误
                if mean_abs_error == 0:
                    df_with_predictions['error_ratio'] = 0.0
                else:
                    df_with_predictions['error_ratio'] = df_with_predictions['absolute_error'] / mean_abs_error
                
                success = True  # 标记成功完成当前迭代
                
            except Exception as e:
                print(f"Error encountered during iteration {current_iteration}, retry {retry_count+1}: {str(e)}")
                # 重置到当前迭代开始前的状态
                previous_model = reset_point['previous_model']
                prediction_results_with_errors = reset_point['prediction_results_with_errors']
                mae = reset_point['mae']
                mse = reset_point['mse']
                reflections = reset_point['reflections']
                retry_count += 1
                
                # 最后一次重试仍然失败
                if retry_count >= max_retries:
                    print(f"Failed to complete iteration {current_iteration} after {max_retries} retries. Skipping to next iteration.")
                    break
        
        # 如果当前迭代失败，直接跳到下一轮迭代
        if not success:
            current_iteration += 1
            continue
        
        # 记录本次迭代结果
        iteration_record = {
            'iteration': current_iteration,
            'model_function': model_function_str,
            'mae': mae,
            'mse': mse,
            'predictions': y_pred.tolist(),
            'true_values': y_true.tolist()
        }
        iteration_history.append(iteration_record)
        
        # 输出本次迭代结果
        print(f"Iteration {current_iteration} Results:")
        print(f"MAE: {mae:.2f}, MSE: {mse:.2f}")
        print(f"Predicted vs Actual with Errors:")
        print(df_with_predictions[['predicted_sales', target_col, 'absolute_error']])
        
        # 判断是否达到停止条件
        if mae < threshold:
            print("Optimization complete. Error threshold achieved.")
            break
        
        # 生成反思报告
        reflections = generate_reflection_report(mae, mse, model_function_str, df_with_predictions)
        
        # 记录反思结果
        iteration_history[-1]['reflections'] = reflections
        
        # 准备下一轮迭代
        previous_model = model_function_str
        prediction_results_with_errors = df_with_predictions[['predicted_sales', target_col, 'absolute_error']]
        current_iteration += 1
    
    # 生成最终总结
    final_summary = generate_final_summary(iteration_history)
    iteration_history.append({'final_summary': final_summary})
    
    # 关闭TensorBoard写入器
    writer.close()
    print("TensorBoard writer closed.")
    
    return iteration_history

# 模拟生成决策树函数的函数
def generate_decision_tree_function(prompt):
    print("Generating decision tree function based on prompt...")
    # 在实际应用中，这里应该调用大模型API
    
    if use_llm:
        tree =  ds_14b_chat(prompt, model, tokenizer)
        tree = extract_code_block(tree)
    else:
        # 根据提示词内容返回不同的决策树函数
        if 'initial' in prompt.lower():
            tree = """
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
            tree = """
    def predict_sales(row):
        if row['cost'] > 150:
            if row['purchase_frequency'] > 10:
                if row['region'] in ['North', 'South']:
                    return row['last_period_purchases'] * 1.3
                else:
                    return row['last_period_purchases'] * 1.1
            else:
                return row['last_period_purchases'] * 0.7
        else:
            if row['loyalty_score'] > 80:
                if row['product_category'] == 'Electronics':
                    return row['last_period_purchases'] * 1.6
                else:
                    return row['last_period_purchases'] * 1.4
            else:
                return row['last_period_purchases'] * 0.85
        return row['last_period_purchases']
    """
    return tree

# 模拟生成最终总结的函数
def generate_final_summary(iteration_history):
    print("Generating final summary of the optimization process...")
    # 在实际应用中，这里应该调用大模型API
    
    # 从迭代历史中提取关键信息
    if len(iteration_history) == 0:
        return "No iterations completed."
    
    initial_mae = iteration_history[0]['mae']
    final_mae = iteration_history[-1]['mae']
    
    if use_llm:
        prompt = prompt_final+"Following is iteration_history:" + str(iteration_history)
        return ds_14b_chat(prompt, model, tokenizer)
    else:
        summary = {
            'initial_model': "Simple decision tree based on cost, purchase frequency, and loyalty score",
            'improvements': [
                "Added product category and region interactions",
                "Refined cost thresholds",
                "Incorporated more granular customer segmentation"
            ],
            'error_reduction': f"MAE reduced from {initial_mae:.2f} to {final_mae:.2f}",
            'key_features': ["product_category", "region", "cost", "loyalty_score"],
            'model_limitations': "Still struggles with predicting seasonal variations",
            'future_suggestions': "Consider adding temporal features and more complex model architecture"
        }
    
    return summary

# 执行决策树优化
for _ in range(5):
    optimization_history = decision_tree_optimization(
        df=df_example,
        target_col='sales_volume',
        threshold=-5,
        max_iterations=300
    )

# 保存优化历史
import json
with open('optimization_history.json', 'w') as f:
    json.dump(optimization_history, f, indent=4, default=str)

print("Decision tree optimization process completed and history saved.")