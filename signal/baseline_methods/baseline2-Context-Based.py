import pandas as pd
import numpy as np
from utils import process_material, process_customer, examine_plan
from train_selling_predict import train_prediction_model, add_to_replay_buffer, ReplayBuffer
from inference_selling_predict import predict_selling
from sklearn.linear_model import LinearRegression
import os
import pickle
from predictDNN import PD
from utils import NormalizeModule
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.special import expit as logistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def clustered_dynamic_pricing(customer_df, sku_df, selling_history_df, N=20):
    """
    基于在线聚类的上下文动态定价算法实现
    输入:
        customer_df: 客户特征表 [客户编码, 客户特征]
        sku_df: SKU特征表 [物料编码, 物料特征, 成本价, 一级品类, 二级品类]
        selling_history_df: 历史销售表 [客户编码, 物料编码, 单价, 销额, 月份]
        N: 每个客户最多推荐的SKU数量
    输出:
        DataFrame: 推荐表 [客户编码, 物料编码, 单价]
    """
    
    # ===== 1. 数据预处理 =====
    # 确保关键列数据类型一致
    selling_history_df['物料编码'] = selling_history_df['物料编码'].astype(str)
    sku_df['物料编码'] = sku_df['物料编码'].astype(str)
    customer_df['客户编码'] = customer_df['客户编码'].astype(str)
    
    # 合并历史数据
    history_df = pd.merge(
        selling_history_df, 
        customer_df, 
        on='客户编码'
    )
    history_df = pd.merge(
        history_df, 
        sku_df, 
        on='物料编码'
    )
    
    # 计算销量 (销额 / 单价)
    history_df['销量'] = history_df['销额'] / history_df['单价']
    history_df['销量'] = history_df['销量'].fillna(0)
    
    # 计算每个客户的历史平均月销额
    monthly_sales = selling_history_df.groupby(['客户编码', '月份'])['销额'].sum().reset_index()
    avg_monthly_sales = monthly_sales.groupby('客户编码')['销额'].mean().reset_index()
    avg_monthly_sales.rename(columns={'销额': '历史平均月销额'}, inplace=True)
    
    # ===== 2. 特征工程 =====
    # 标准化特征
    scaler = StandardScaler()
    customer_df['客户特征'] = scaler.fit_transform(np.vstack(customer_df['客户特征'])).tolist()
    sku_df['物料特征'] = scaler.fit_transform(np.vstack(sku_df['物料特征'])).tolist()
    
    # 创建上下文特征: 客户特征 + 物料特征
    def combine_features(row):
        return np.concatenate([row['客户特征'], row['物料特征']])
    
    history_df['上下文特征'] = history_df.apply(combine_features, axis=1)
    
    # ===== 3. 需求模型训练 =====
    # 使用逻辑回归作为需求模型 (购买概率)
    X_train = np.stack(history_df['上下文特征'].values)
    prices = history_df['单价'].values
    X_train = np.hstack([X_train, prices.reshape(-1, 1)])  # 添加价格特征
    
    # 创建标签: 是否购买 (销量>0)
    y_train = (history_df['销量'] > 0).astype(int).values
    
    # 训练模型
    demand_model = LogisticRegression(max_iter=1000)
    demand_model.fit(X_train, y_train)
    
    # ===== 4. 在线聚类初始化 =====
    # 动态聚类参数
    cluster_centers = []  # 存储聚类中心
    cluster_params = {}   # 存储每个聚类的需求参数
    
    # ===== 5. 为每个客户生成推荐 =====
    results = []
    
    # 获取所有客户编码列表
    customer_ids = customer_df['客户编码'].unique()
    
    for customer_id in tqdm(customer_ids, desc="Processing customers"):

        # 获取当前客户的特征
        customer_features = customer_df.loc[customer_df['客户编码'] == customer_id, '客户特征'].values[0]
        
        # 5.1 计算当前客户的月销额上限，处理空值情况
        if customer_id in avg_monthly_sales['客户编码'].values:
            cust_avg_sales = avg_monthly_sales.loc[avg_monthly_sales['客户编码'] == customer_id, 
                                                 '历史平均月销额'].values[0]
        else:
            cust_avg_sales = 999999  # 默认值
        
        # TODO
        # sales_cap = cust_avg_sales * 1.25
        sales_cap = cust_avg_sales * 100
        
        # 5.2 初始化推荐列表和约束状态
        recommended_skus = []
        recommended_prices = []
        current_sales = 0.0
        
        # 品类覆盖状态
        primary_cat_coverage = {}
        secondary_cat_coverage = {}
        
        # 5.3 遍历所有SKU进行聚类和定价
        for _, sku_data in sku_df.iterrows():
            sku_id = sku_data['物料编码']
            sku_features = sku_data['物料特征']
            
            # 组合上下文特征
            context_features = np.concatenate([customer_features, sku_features])
            
            # 在线聚类: 基于需求参数相似度
            if len(cluster_centers) == 0:
                # 初始聚类
                cluster_id = 0
                cluster_centers.append(context_features)
            else:
                # 计算与现有聚类中心的距离
                distances = pairwise_distances([context_features], cluster_centers)[0]
                min_dist = np.min(distances)
                min_idx = np.argmin(distances)
                
                # 动态创建新聚类
                if min_dist > 0.5:  # 阈值可调整
                    cluster_id = len(cluster_centers)
                    cluster_centers.append(context_features)
                else:
                    cluster_id = min_idx
            
            # 获取或初始化聚类需求参数
            if cluster_id not in cluster_params:
                cluster_params[cluster_id] = np.zeros(len(context_features) + 1)
            
            # 5.4 价格优化
            cost = sku_data['成本价']
            def profit_function(price):
                # 构建特征向量 (包括价格)
                features = np.append(context_features, price)
                
                # 预测购买概率
                prob_buy = demand_model.predict_proba([features])[0][1]
                
                # 计算期望利润
                expected_profit = (price - cost) * prob_buy
                return -expected_profit  # 负值用于最小化
            
            # 价格优化范围 [成本价, 成本价*2]
            price_bounds = (cost, cost * 2)
            res = minimize(profit_function, x0=cost * 1.5, bounds=[price_bounds])
            
            optimal_price = res.x[0]
            expected_profit = -res.fun
            
            # 5.5 候选推荐项
            candidate = {
                'sku_id': sku_id,
                'price': optimal_price,
                'profit': expected_profit,
                'primary_cat': sku_data['一级品类'],
                'secondary_cat': sku_data['二级品类'],
                'expected_sales': optimal_price * demand_model.predict_proba(
                    [np.append(context_features, optimal_price)])[0][1]
            }
            
            # 5.6 约束满足检查
            # 品类覆盖检查
            primary_needed = primary_cat_coverage.get(candidate['primary_cat'], 0) < 2
            secondary_needed = secondary_cat_coverage.get(candidate['secondary_cat'], 0) < 1
            
            # 销售额约束
            sales_ok = (current_sales + candidate['expected_sales']) <= sales_cap
            
            # 数量约束
            count_ok = len(recommended_skus) < N
            
            # 5.7 添加推荐项
            if (primary_needed or secondary_needed) and sales_ok and count_ok:
                recommended_skus.append(candidate['sku_id'])
                recommended_prices.append(candidate['price'])
                current_sales += candidate['expected_sales']
                
                # 更新品类覆盖状态
                primary_cat_coverage[candidate['primary_cat']] = primary_cat_coverage.get(
                    candidate['primary_cat'], 0) + 1
                secondary_cat_coverage[candidate['secondary_cat']] = secondary_cat_coverage.get(
                    candidate['secondary_cat'], 0) + 1
        
        # 5.8 添加剩余位置 (非必须品类)
        remaining_slots = N - len(recommended_skus)
        if remaining_slots > 0:
            # 获取所有SKU的利润排序
            all_skus = []
            for _, sku_data in sku_df.iterrows():
                sku_id = sku_data['物料编码']
                
                # 跳过已添加的SKU
                if sku_id in recommended_skus:
                    continue
                
                # 组合上下文特征
                context_features = np.concatenate([customer_features, sku_data['物料特征']])
                
                # 价格优化
                cost = sku_data['成本价']
                res = minimize(profit_function, x0=cost * 1.5, bounds=[price_bounds])
                
                optimal_price = res.x[0]
                expected_profit = -res.fun
                
                candidate = {
                    'sku_id': sku_id,
                    'price': optimal_price,
                    'profit': expected_profit,
                    'expected_sales': optimal_price * demand_model.predict_proba(
                        [np.append(context_features, optimal_price)])[0][1]
                }
                all_skus.append(candidate)
            
            # 按利润降序排序
            all_skus.sort(key=lambda x: x['profit'], reverse=True)
            
            # 添加剩余位置
            for candidate in all_skus:
                if len(recommended_skus) >= N:
                    break
                    
                if (current_sales + candidate['expected_sales']) <= sales_cap:
                    recommended_skus.append(candidate['sku_id'])
                    recommended_prices.append(candidate['price'])
                    current_sales += candidate['expected_sales']
        
        # 5.9 存储结果
        for sku_id, price in zip(recommended_skus, recommended_prices):
            results.append({
                '客户编码': customer_id,
                '物料编码': sku_id,
                '单价': price
            })
            
    # 创建最终结果DataFrame
    if results==[]:
        results.append({
                '客户编码': 'C1',
                '物料编码': 'SKU1',
                '单价': 0.0
            })
    plan_df = pd.DataFrame(results, columns=['客户编码', '物料编码', '单价'])
    
    return plan_df

if __name__ == "__main__":
    customer_df = pd.read_parquet('data/mock_customer_df.parquet')
    sku_df = pd.read_parquet('data/mock_sku_df.parquet')
    selling_history_df = pd.read_parquet('data/mock_selling_history_df.parquet')
    plan_df = clustered_dynamic_pricing(customer_df, sku_df, selling_history_df)
    print(plan_df)
    # plan_df = pd.DataFrame({
    #     '客户编码': np.random.choice(customer_df['客户编码'], 200),
    #     '物料编码': np.random.choice(sku_df['物料编码'], 200),
    #     '单价': np.random.uniform(20, 200, 200)  # 单价范围为20到200
    # })
    sku_df['物料编码'] = sku_df['物料编码'].astype(str)
    plan_df['物料编码'] = plan_df['物料编码'].astype(str)
    eval_result = examine_plan(plan_df, customer_df, sku_df)
    print(eval_result)
    # 保存结果
    # plan_df.to_parquet('data/plan_baseline1_df.parquet', index=False)