import pandas as pd
import numpy as np
from utils import process_material, process_customer
from train_selling_predict import train_prediction_model, add_to_replay_buffer, ReplayBuffer
from inference_selling_predict import predict_selling
from sklearn.linear_model import LinearRegression
import os
import pickle
from predictDNN import PD
from utils import NormalizeModule
import warnings
# 设置随机种子以保证结果可复现
np.random.seed(42)

# 定义参数
N1 = 5  # 客户特征维度
N2 = 5  # 物料特征维度
num_customers = 300  # 客户数量
num_skus = 100  # 物料数量
num_sales = 20000  # 历史销售记录数量
# 生成客户特征表
customer_df = pd.DataFrame({
    '客户编码': [f'C{i}' for i in range(num_customers)],
    '客户特征': [np.random.rand(N1) for _ in range(num_customers)]
})
# 生成SKU特征表
sku_df = pd.DataFrame({
    '物料编码': [f'SKU{i}' for i in range(num_skus)],
    '物料特征': [np.random.rand(N2) for _ in range(num_skus)],
    '成本价': np.random.uniform(10, 100, num_skus),  # 成本价范围为10到100
    '一级品类': np.random.randint(1, 5, num_skus),
    '二级品类': np.random.randint(1, 10, num_skus)
    
})
# 为每个SKU设置不同的单价阈值
sku_df['单价阈值'] = np.random.uniform(100, 200, num_skus)  # 阈值范围为100到200
# 生成历史销售表
selling_history_df = pd.DataFrame({
    '客户编码': np.random.choice(customer_df['客户编码'], num_sales),
    '物料编码': np.random.choice(sku_df['物料编码'], num_sales),
    '单价': np.random.uniform(20, 200, num_sales),  # 单价范围为20到200
    '月份': np.random.randint(1, 13, num_sales),
})
# 构造销额的基底函数
def generate_sales_base(customer_features, sku_features, price, price_threshold):
    # 假设销额基底是客户特征和物料特征的多项式函数
    base_sales = (
        np.sum(customer_features * np.random.rand(N1)) +  # 客户特征影响
        np.sum(sku_features * np.random.rand(N2))  # 物料特征影响
    )
    # 单价与销额成线性负相关，且不同SKU的单价阈值不同
    if price > price_threshold:
        return 0  # 如果单价超过阈值，销额为0
    else:
        # 单价的影响：单价越高，销额越低，线性负相关
        base_sales *= (price_threshold - price) / price_threshold
    return base_sales
# 计算销额
selling_history_df['销额'] = selling_history_df.apply(lambda row: generate_sales_base(
    customer_df.loc[customer_df['客户编码'] == row['客户编码'], '客户特征'].iloc[0],
    sku_df.loc[sku_df['物料编码'] == row['物料编码'], '物料特征'].iloc[0],
    row['单价'],
    sku_df.loc[sku_df['物料编码'] == row['物料编码'], '单价阈值'].iloc[0]
), axis=1)
# 添加40%左右的噪声扰动
noise_factor = 0.4
selling_history_df['销额'] += np.random.normal(0, noise_factor * selling_history_df['销额'].std(), num_sales)
# 确保销额为非负数
selling_history_df['销额'] = selling_history_df['销额'].apply(lambda x: max(x, 0))
# 输出结果
print("客户特征表：")
print(customer_df.head())
customer_df.to_parquet("data/mock_customer_df.parquet")
print("\nSKU特征表：")
print(sku_df.head())
sku_df.to_parquet("data/mock_sku_df.parquet")
print("\n历史销售表：")
print(selling_history_df.head())
selling_history_df.to_parquet("data/mock_selling_history_df.parquet")




# 训练市场模型

# 客户特征表 [客户编码 客户特征]
customer_df = pd.read_parquet('data/mock_customer_df.parquet')
# sku特征表 [物料编码，物料特征，成本价]
sku_df = pd.read_parquet('data/mock_sku_df.parquet')
# 历史销售表 [客户编码，物料编码，单价，销额]
selling_history_df = pd.read_parquet('data/mock_selling_history_df.parquet')

# 市场模型 f:[客户编码, 物料编码，单价]->销额
N1 = len(customer_df['客户特征'].iloc[0])
# 物料特征的维度
N2 = len(sku_df['物料特征'].iloc[0])
# obs的长度
obs_length = N1 + N2
def extract_features(row, customer_df, sku_df):
    # 提取客户特征
    customer_features = customer_df.loc[customer_df['客户编码'] == row['客户编码'], '客户特征'].iloc[0]
    # 提取物料特征
    sku_features = sku_df.loc[sku_df['物料编码'] == row['物料编码'], '物料特征'].iloc[0]
    # 拼接成大向量
    obs = np.concatenate((customer_features, sku_features))
    return obs
# 应用函数，生成obs字段
selling_history_df['obs'] = selling_history_df.apply(lambda row: extract_features(row, customer_df, sku_df), axis=1)
# 新建output字段，值与销额字段相同
selling_history_df['output'] = selling_history_df['销额']
train_df = selling_history_df
train_data_path = "data/mock_train_data.parquet"
save_path = "models/mock_model_save_path"
train_df.to_parquet(train_data_path, index=False)
train_prediction_model(train_data_path, save_path, step_num=5000, batch_size=32, state_dim=obs_length, dim_action=1)