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
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def model_free_assortment_pricing(customer_df, sku_df, selling_history_df, next_month, max_skus=20):
    """
    基于激励相容约束的选品定价算法
    输入:
        customer_df: 客户特征表 [客户编码, 客户特征]
        sku_df: SKU特征表 [物料编码, 物料特征, 成本价, 一级品类, 二级品类]
        selling_history_df: 历史销售表 [客户编码, 物料编码, 单价, 销额, 月份]
        next_month: 下个月份(int)
        max_skus: 每个客户最大推荐SKU数(默认20)
    输出:
        DataFrame: [客户编码, 物料编码, 单价, 月份]
    """
    # 1. 数据预处理
    
    # 合并历史数据
    history_df = selling_history_df.merge(
        customer_df, on='客户编码'
    ).merge(
        sku_df, on='物料编码'
    )
    
    # 计算客户月均销售额
    monthly_sales = selling_history_df.groupby(
        ['客户编码', '月份']
    )['销额'].sum().reset_index()
    avg_monthly_sales = monthly_sales.groupby(
        '客户编码'
    )['销额'].mean().reset_index(name='月均销额')
    
    # 特征向量归一化
    customer_df['客户特征'] = customer_df['客户特征'].apply(lambda x: x / np.linalg.norm(x))
    sku_df['物料特征'] = sku_df['物料特征'].apply(lambda x: x / np.linalg.norm(x))
    
    # 2. 构建激励相容多面体(IC Polyhedron)
    
    # 计算客户-产品相似度矩阵
    cust_features = np.vstack(customer_df['客户特征'].values)
    sku_features = np.vstack(sku_df['物料特征'].values)
    similarity_matrix = cosine_similarity(cust_features, sku_features)
    
    # 创建相似度DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=customer_df['客户编码'],
        columns=sku_df['物料编码']
    ).stack().reset_index(name='相似度')
    similarity_df.columns = ['客户编码', '物料编码', '相似度']
    
    # 计算历史价格分位数
    price_quantiles = history_df.groupby('物料编码')['单价'].quantile(
        [0.25, 0.5, 0.75]
    ).unstack().reset_index()
    price_quantiles.columns = ['物料编码', '价格25分位', '价格中位数', '价格75分位']
    
    # 3. 截止定价(Cut-off Pricing)算法
    
    def calculate_cutoff_price(customer_id):
        """计算客户的截止价格"""
        cust_history = history_df[history_df['客户编码'] == customer_id]
        if len(cust_history) == 0:
            # 新客户使用全局中位数
            return history_df['单价'].median()
        return cust_history['单价'].median()
    
    # 4. 品类约束处理
    
    def apply_category_constraints(selected_skus, sku_info):
        """确保品类约束满足"""
        # 统计已选品类
        level1_counts = defaultdict(int)
        level2_counts = defaultdict(int)
        
        for sku in selected_skus:
            info = sku_info[sku]
            level1_counts[info['一级品类']] += 1
            level2_counts[info['二级品类']] += 1
        
        # 检查一级品类约束
        for l1, count in level1_counts.items():
            if count < 2:
                # 添加同一品类产品
                same_l1_skus = [s for s, info in sku_info.items() 
                               if info['一级品类'] == l1 and s not in selected_skus]
                if same_l1_skus:
                    # 选择最相似的产品
                    selected_skus.append(same_l1_skus[0])
                    # 更新二级品类计数
                    level2_counts[sku_info[same_l1_skus[0]]['二级品类']] += 1
        
        # 检查二级品类约束
        for l2, count in level2_counts.items():
            if count < 1:
                # 添加同一二级品类产品
                same_l2_skus = [s for s, info in sku_info.items() 
                               if info['二级品类'] == l2 and s not in selected_skus]
                if same_l2_skus:
                    selected_skus.append(same_l2_skus[0])
        
        return selected_skus
    
    # 5. 核心选品定价算法
    
    results = []
    sku_info = sku_df.set_index('物料编码').to_dict('index')
    
    for cust_id in customer_df['客户编码'].unique():
        # 计算截止价格
        cutoff_price = calculate_cutoff_price(cust_id)
        
        # 获取相似度最高的产品
        cust_similarity = similarity_df[
            similarity_df['客户编码'] == cust_id
        ].sort_values('相似度', ascending=False)
        
        # 初步选品
        selected_skus = []
        considered_skus = set()
        
        # 1. 选择历史购买过的产品
        purchased = history_df[
            history_df['客户编码'] == cust_id
        ]['物料编码'].unique()
        for sku in purchased:
            if sku in sku_info and sku not in considered_skus:
                selected_skus.append(sku)
                considered_skus.add(sku)
        
        # 2. 选择相似度高且价格合适的产品
        for _, row in cust_similarity.iterrows():
            sku = row['物料编码']
            if sku not in considered_skus:
                price_info = price_quantiles[price_quantiles['物料编码'] == sku]
                if not price_info.empty:
                    min_price = price_info['价格25分位'].values[0]
                    # 价格在可接受范围内
                    if min_price <= cutoff_price * 1.5:
                        selected_skus.append(sku)
                        considered_skus.add(sku)
            if len(selected_skus) >= max_skus:
                break
        
        # 应用品类约束
        selected_skus = apply_category_constraints(selected_skus, sku_info)[:max_skus]
        
        # 定价策略
        prices = {}
        for sku in selected_skus:
            price_info = price_quantiles[price_quantiles['物料编码'] == sku]
            if not price_info.empty:
                # 使用截止定价策略
                median_price = price_info['价格中位数'].values[0]
                prices[sku] = max(
                    sku_info[sku]['成本价'] * 1.1,  # 确保高于成本价
                    min(cutoff_price, median_price)
                )
            else:
                prices[sku] = cutoff_price
        
        # 应用销售额约束
        total_sales = sum(prices.values())
        avg_sales = avg_monthly_sales[
            avg_monthly_sales['客户编码'] == cust_id
        ]['月均销额'].values
        
        if len(avg_sales) > 0:
            max_allowed = avg_sales[0] * 1.25
            if total_sales > max_allowed:
                # 按比例降低价格
                scale_factor = max_allowed / total_sales
                for sku in prices:
                    prices[sku] = max(
                        sku_info[sku]['成本价'] * 1.05,
                        prices[sku] * scale_factor
                    )
        
        # 添加到结果
        for sku, price in prices.items():
            results.append({
                '客户编码': cust_id,
                '物料编码': sku,
                '单价': round(price, 2),
                '月份': next_month
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    customer_df = pd.read_parquet('data/mock_customer_df.parquet')
    sku_df = pd.read_parquet('data/mock_sku_df.parquet')
    selling_history_df = pd.read_parquet('data/mock_selling_history_df.parquet')
    
    next_month = selling_history_df['月份'].max() + 1
    
    plan_df = model_free_assortment_pricing(
        customer_df, 
        sku_df, 
        selling_history_df,
        next_month
    )
    eval_result = examine_plan(plan_df, customer_df, sku_df)
    print(eval_result)