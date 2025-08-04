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
from scipy.linalg import svd
from tqdm import tqdm


class LowRankBanditPricing:
    def __init__(self, rank=5, T=1000, learning_rate=0.01, exploration_param=0.1):
        """
        初始化低秩定价算法
        :param rank: 潜在特征维度
        :param T: 时间步长
        :param learning_rate: 学习率
        :param exploration_param: 探索参数
        """
        self.rank = rank
        self.T = T
        self.eta = learning_rate
        self.delta = exploration_param
        self.U_hat = None  # 产品特征估计
        self.V_hat = None  # 弹性矩阵估计
        self.z_hat = None  # 基线需求估计
        self.scaler = StandardScaler()
    
    def _kronecker_product(self, vec1, vec2):
        """计算两个向量的Kronecker积"""
        return np.outer(vec1, vec2).flatten()
    
    def _project_to_ball(self, vector, radius=20):
        """将向量投影到欧几里得球约束"""
        norm = np.linalg.norm(vector)
        if norm > radius:
            return vector * radius / norm
        return vector
    
    def _online_svd_update(self, Q_hat, q_t, t, j):
        """在线SVD更新特征空间估计"""
        if t == 0:
            Q_hat[:, j] = q_t
        else:
            Q_hat[:, j] = (t * Q_hat[:, j] + q_t) / (t + 1)
        
        # 计算SVD并更新特征估计
        U, S, Vt = svd(Q_hat, full_matrices=False)
        self.U_hat = U[:, :self.rank]
        return Q_hat
    
    def fit(self, customer_features, sku_features, historical_data, N=20):
        """
        训练低秩需求模型
        :param customer_features: 客户特征DataFrame
        :param sku_features: SKU特征DataFrame
        :param historical_data: 历史销售DataFrame
        :param N: 每个客户推荐SKU数量
        """
        # 数据预处理
        merged_data = historical_data.merge(
            customer_features, on='客户编码'
        ).merge(
            sku_features, on='物料编码'
        )
        
        # 计算实际需求（销额/单价）
        merged_data['需求'] = merged_data['销额'] / merged_data['单价']
        
        # 特征标准化
        c_features = np.vstack(merged_data['客户特征'].apply(np.array).values)
        s_features = np.vstack(merged_data['物料特征'].apply(np.array).values)
        self.scaler.fit(np.vstack([c_features, s_features]))
        
        # 初始化参数
        n_customers = len(customer_features)
        n_skus = len(sku_features)
        dim = c_features.shape[1] + s_features.shape[1]
        
        self.U_hat = np.random.randn(dim, self.rank)
        self.V_hat = np.eye(self.rank)
        self.z_hat = np.zeros(self.rank)
        
        # 在线学习过程
        Q_hat = np.zeros((dim, self.rank))
        
        for t in tqdm(range(self.T), desc="Training Low-Rank Model"):
            # 随机选择客户和SKU
            cust_idx = np.random.randint(n_customers)
            sku_idx = np.random.randint(n_skus)
            
            # 获取特征
            c_feat = self.scaler.transform([customer_features.iloc[cust_idx]['客户特征']])[0]
            s_feat = self.scaler.transform([sku_features.iloc[sku_idx]['物料特征']])[0]
            joint_feat = np.concatenate([c_feat, s_feat])
            
            # 生成随机扰动
            xi = np.random.randn(self.rank)
            xi /= np.linalg.norm(xi)
            
            # 计算低维动作 (关键修改：使用标量价格而非向量)
            x_tilde = self.z_hat + self.delta * xi
            
            # 计算标量价格 (使用特征向量的均值)
            p_t = np.mean(joint_feat) * x_tilde[0]
            
            # 预测需求 (关键修改：使用低维特征表示)
            x_low = self.U_hat.T @ joint_feat
            a = x_low @ self.z_hat
            b = x_low @ self.V_hat @ x_low
            q_t = a - b * p_t + np.random.normal(0, 1)  # 添加噪声
            
            # 在线SVD更新
            j = t % self.rank
            Q_hat = self._online_svd_update(Q_hat, joint_feat, t, j)
            
            # 参数更新 (关键修改：使用标量收益)
            revenue_estimate = -p_t * q_t  # 标量负收益
            gradient_estimate = (self.rank / self.delta) * revenue_estimate * xi
            self.z_hat -= self.eta * gradient_estimate
    
    def recommend(self, customer_df, sku_df, month, N=20):
        """
        为指定客户生成推荐
        :param customer_df: 客户特征DataFrame
        :param sku_df: SKU特征DataFrame
        :param month: 目标月份
        :param N: 每个客户推荐SKU数量
        :return: 推荐DataFrame (客户编码, 物料编码, 单价)
        """
        recommendations = []
        
        for _, customer_row in tqdm(customer_df.iterrows(), total=len(customer_df), desc="Generating Recommendations"):
            customer_id = customer_row['客户编码']
            c_feat = self.scaler.transform([customer_row['客户特征']])[0]
            
            sku_scores = []
            for _, sku_row in sku_df.iterrows():
                sku_id = sku_row['物料编码']
                cost = sku_row['成本价']
                s_feat = self.scaler.transform([sku_row['物料特征']])[0]
                joint_feat = np.concatenate([c_feat, s_feat])
                
                # 计算低维表示
                x_low = self.U_hat.T @ joint_feat
                
                # 预测需求参数
                a = x_low @ self.z_hat  # 基线需求
                b = x_low @ self.V_hat @ x_low  # 价格弹性
                
                # 计算最优价格 (考虑成本)
                if b > 1e-6:  # 防止除以零
                    p_opt = (a + b * cost) / (2 * b)
                else:
                    p_opt = cost * 1.2  # 默认加价20%
                
                # 计算预期利润
                demand = a - b * p_opt
                profit = (p_opt - cost) * demand
                
                sku_scores.append((sku_id, p_opt, profit))
            
            # 选择利润最高的N个SKU
            sku_scores.sort(key=lambda x: x[2], reverse=True)
            top_skus = sku_scores[:N]
            
            for sku_id, price, _ in top_skus:
                recommendations.append({
                    '客户编码': customer_id,
                    '物料编码': sku_id,
                    '单价': max(price, cost * 1.05)  # 确保价格高于成本
                })
        
        return pd.DataFrame(recommendations)

# 主函数
def dynamic_pricing_recommendation(customer_df, sku_df, selling_history_df, month, N=20):
    """
    动态定价推荐函数
    :param customer_df: 客户特征DataFrame
    :param sku_df: SKU特征DataFrame
    :param selling_history_df: 历史销售DataFrame
    :param month: 目标月份
    :param N: 每个客户推荐SKU数量
    :return: 推荐DataFrame (客户编码, 物料编码, 单价)
    """
    # 初始化算法
    model = LowRankBanditPricing(rank=5, T=1000, learning_rate=0.01, exploration_param=0.1)
    
    # 训练模型
    model.fit(customer_df, sku_df, selling_history_df, N)
    
    # 生成推荐
    recommendations = model.recommend(customer_df, sku_df, month, N)
    
    return recommendations

if __name__ == "__main__":
    customer_df = pd.read_parquet('data/mock_customer_df.parquet')
    sku_df = pd.read_parquet('data/mock_sku_df.parquet')
    selling_history_df = pd.read_parquet('data/mock_selling_history_df.parquet')
    
    # plan_df = pd.DataFrame({
    #     '客户编码': np.random.choice(customer_df['客户编码'], 200),
    #     '物料编码': np.random.choice(sku_df['物料编码'], 200),
    #     '单价': np.random.uniform(20, 200, 200)  # 单价范围为20到200
    # })
    plan_df = dynamic_pricing_recommendation(
        customer_df, sku_df, selling_history_df, month=202306, N=20
    )
    eval_result = examine_plan(plan_df, customer_df, sku_df)
    print(eval_result)
    # 保存结果
    # plan_df.to_parquet('data/plan_baseline1_df.parquet', index=False)



