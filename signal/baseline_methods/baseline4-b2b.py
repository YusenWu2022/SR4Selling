import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from itertools import product
import pulp
from utils import process_material, process_customer, examine_plan
from train_selling_predict import train_prediction_model, add_to_replay_buffer, ReplayBuffer
from inference_selling_predict import predict_selling
from sklearn.linear_model import LinearRegression
import os
import pickle
from predictDNN import PD
from utils import NormalizeModule
import warnings
from tqdm import tqdm

def recommend_skus(customer_df, sku_df, selling_history_df, customer_list, N=20):
    """
    为指定客户推荐SKU及定价
    参数:
        customer_df: 客户特征表 [客户编码, 客户特征]
        sku_df: SKU特征表 [物料编码, 物料特征, 成本价, 一级品类, 二级品类]
        selling_history_df: 历史销售表 [客户编码, 物料编码, 单价, 销额, 月份]
        customer_list: 需要推荐的客户编码列表
        N: 每个客户最多推荐SKU数量 (默认20)
        
    返回:
        DataFrame: [客户编码, 物料编码, 单价] 的推荐表
    """
    # 1. 数据预处理与特征工程
    # 计算客户历史月均销售额
    cust_monthly_sales = selling_history_df.groupby(['客户编码', '月份'])['销额'].sum().reset_index()
    cust_avg_sales = cust_monthly_sales.groupby('客户编码')['销额'].mean().reset_index()
    cust_avg_sales.rename(columns={'销额': '历史月均销额'}, inplace=True)
    cust_avg_sales['销售额上限'] = cust_avg_sales['历史月均销额'] * 1.25
    
    # 合并客户特征
    customer_features = pd.merge(customer_df, cust_avg_sales, on='客户编码', how='left')
    
    # 2. 训练混合Logit模型 (简化为随机森林)
    # 准备训练数据
    history_features = selling_history_df.merge(
        customer_features[['客户编码', '客户特征']], on='客户编码'
    ).merge(
        sku_df[['物料编码', '物料特征', '成本价']], on='物料编码'
    )
    
    # 创建特征向量: 客户特征 + SKU特征 + 价格特征
    def create_feature_vector(row):
        return np.concatenate([
            row['客户特征'],
            row['物料特征'],
            [row['单价']]
        ])
    
    history_features['特征向量'] = history_features.apply(create_feature_vector, axis=1)
    
    # 训练随机森林模型 (替代混合Logit)
    X = np.stack(history_features['特征向量'].values)
    y = (history_features['销额'] > 0).astype(int)  # 购买概率
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # 3. 定义购买概率预测函数
    def predict_prob(cust_features, sku_features, price):
        """预测客户购买特定SKU的概率"""
        if isinstance(price,float):
            price = [price]
        feature_vec = np.concatenate([cust_features, sku_features, price])
        scaled_vec = scaler.transform([feature_vec])
        return rf_model.predict_proba(scaled_vec)[0][1]
    
    # 4. 为每个客户优化推荐
    results = []
    
    for cust_id in tqdm(customer_list):
        # 获取客户信息
        cust_data = customer_features[customer_features['客户编码'] == cust_id].iloc[0]
        cust_features = cust_data['客户特征']
        sales_limit = cust_data['销售额上限']
        
        # 准备SKU数据
        sku_data = sku_df.copy()
        
        # 预测函数 - 用于优化
        def purchase_probability(sku_idx, price):
            sku_row = sku_data.iloc[sku_idx]
            return predict_prob(cust_features, sku_row['物料特征'], price)
        
        # 5. 非线性优化 - 为每个SKU找到最优价格
        sku_optim_results = []
        
        for sku_idx in range(len(sku_data)):
            sku_row = sku_data.iloc[sku_idx]
            cost = sku_row['成本价']
            
            # 定义目标函数 (最大化期望利润)
            def objective(price):
                prob = purchase_probability(sku_idx, price)
                expected_profit = (price - cost) * prob
                return -expected_profit  # 最小化负利润
            
            # 约束: 价格 >= 成本价
            constraints = [{'type': 'ineq', 'fun': lambda p: p - cost}]
            
            # 初始价格 (成本价 * 1.2)
            initial_price = cost * 1.2
            
            # 执行优化
            res = minimize(
                objective,
                x0=initial_price,
                method='SLSQP',
                bounds=[(cost, cost*3)],  # 价格范围: 成本价到3倍成本价
                constraints=constraints
            )
            
            if res.success:
                opt_price = res.x[0]
                opt_prob = purchase_probability(sku_idx, opt_price)
                expected_profit = (opt_price - cost) * opt_prob
                expected_sales = opt_price * opt_prob
                
                sku_optim_results.append({
                    '物料编码': sku_row['物料编码'],
                    '推荐价格': opt_price,
                    '期望利润': expected_profit,
                    '期望销售额': expected_sales,
                    '一级品类': sku_row['一级品类'],
                    '二级品类': sku_row['二级品类']
                })
        
        # 转换为DataFrame
        sku_optim_df = pd.DataFrame(sku_optim_results)
        
        # 6. 整数规划 - 选择最优SKU组合
        # 创建问题实例
        prob = pulp.LpProblem("SKU_Selection", pulp.LpMaximize)
        
        # 创建决策变量
        sku_ids = sku_optim_df['物料编码'].tolist()
        x = pulp.LpVariable.dicts("x", sku_ids, cat='Binary')
        
        # 目标函数: 最大化总期望利润
        profit_coeff = {row['物料编码']: row['期望利润'] 
                       for _, row in sku_optim_df.iterrows()}
        prob += pulp.lpSum([profit_coeff[i] * x[i] for i in sku_ids])
        
        # 约束条件
        # 1. 推荐SKU数量不超过N
        prob += pulp.lpSum([x[i] for i in sku_ids]) <= N
        
        # 2. 销售额不超过上限
        sales_coeff = {row['物料编码']: row['期望销售额'] 
                      for _, row in sku_optim_df.iterrows()}
        prob += pulp.lpSum([sales_coeff[i] * x[i] for i in sku_ids]) <= sales_limit
        
        # 3. 品类约束
        # 获取所有一级品类
        primary_cats = sku_optim_df['一级品类'].unique()
        for cat in primary_cats:
            cat_skus = sku_optim_df[sku_optim_df['一级品类'] == cat]['物料编码']
            prob += pulp.lpSum([x[i] for i in cat_skus]) >= 2
        
        # 获取所有二级品类
        secondary_cats = sku_optim_df['二级品类'].unique()
        for cat in secondary_cats:
            cat_skus = sku_optim_df[sku_optim_df['二级品类'] == cat]['物料编码']
            prob += pulp.lpSum([x[i] for i in cat_skus]) >= 1
        
        # 求解问题
        prob.solve()
        
        # 提取结果
        selected_skus = []
        for sku_id in sku_ids:
            if x[sku_id].value() == 1:
                sku_row = sku_optim_df[sku_optim_df['物料编码'] == sku_id].iloc[0]
                selected_skus.append({
                    '客户编码': cust_id,
                    '物料编码': sku_id,
                    '单价': sku_row['推荐价格']
                })
        
        results.extend(selected_skus)
    
    # 返回最终推荐结果
    return pd.DataFrame(results)

if __name__ == "__main__":
    customer_df = pd.read_parquet('data/mock_customer_df.parquet')
    sku_df = pd.read_parquet('data/mock_sku_df.parquet')
    selling_history_df = pd.read_parquet('data/mock_selling_history_df.parquet')

    # customer_list = customer_df['客户编码'].sample(5).tolist()
    customer_list = customer_df['客户编码'].tolist()
    
    plan_df = recommend_skus(customer_df, sku_df, selling_history_df, customer_list)
    eval_result = examine_plan(plan_df, customer_df, sku_df)
    print(eval_result)
    