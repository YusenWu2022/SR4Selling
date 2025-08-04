import pandas as pd
import pandas as pd
import numpy as np
import os
import pickle
from predictDNN import PD
from train_selling_predict import train_prediction_model, add_to_replay_buffer, ReplayBuffer
from inference_selling_predict import predict_selling
from sklearn.linear_model import LinearRegression

def process_material(df):
    grouped_material = df.groupby('物料编码')
    mode_columns = ['SKU价格', '标准重量', '含税单价', '箱规']
    def get_mode(x):
        try:
            return x.mode()[0]
        except:
            return None
    material_mode = grouped_material[mode_columns].agg(get_mode)
    total_sales = grouped_material['物料总销额'].sum()
    transaction_months = grouped_material['月份'].nunique()
    total_customers_online = grouped_material['客户编码'].nunique() 
    total_online_months = grouped_material['月份'].count() 
    average_online_months = total_online_months / total_customers_online
    material_result = pd.concat([material_mode, total_sales, transaction_months, average_online_months], axis=1)
    material_result.columns = ['SKU价格', '标准重量', '含税单价', '箱规', '物料总销额总和', '物料总交易月份个数', '平均上线月份']
    material_result = material_result.reset_index()
    material_result.to_csv("data/物料编码特征.csv")
    # print(total_sales)
    return material_result

# Step2 简单统计客户特征，可扩充
def process_customer(df):
    current_month = df['月份'].max()
    grouped_customer = df.groupby('客户编码')
    def calculate_rolling_sums(group):
        group = group.sort_values('月份', ascending=False)
        monthly_sums = group.groupby('月份')[['折后未税收入金额', '重量', '利润']].sum().reset_index()
        unique_months = len(monthly_sums)
        sums_12m = monthly_sums.head(12)[['折后未税收入金额', '重量', '利润']].sum()
        sums_6m = monthly_sums.head(6)[['折后未税收入金额', '重量', '利润']].sum()
        sums_3m = monthly_sums.head(3)[['折后未税收入金额', '重量', '利润']].sum()
        avg_12m = sums_12m / min(12, unique_months)
        avg_6m = sums_6m / min(6, unique_months)
        avg_3m = sums_3m / min(3, unique_months)
        avg_monthly_revenue = monthly_sums['折后未税收入金额'].sum() / unique_months
        result = pd.Series({
            '折后未税收入金额_12m_avg': avg_12m['折后未税收入金额'],
            '重量_12m_avg': avg_12m['重量'],
            '利润_12m_avg': avg_12m['利润'],
            '折后未税收入金额_6m_avg': avg_6m['折后未税收入金额'],
            '重量_6m_avg': avg_6m['重量'],
            '利润_6m_avg': avg_6m['利润'],
            '折后未税收入金额_3m_avg': avg_3m['折后未税收入金额'],
            '重量_3m_avg': avg_3m['重量'],
            '利润_3m_avg': avg_3m['利润'],
            '平均单次进货总销额': avg_monthly_revenue
        })
        return result
    customer_rolling_sums = grouped_customer.apply(calculate_rolling_sums).reset_index()
    def create_brand_columns(group):
        brands = group['预算品牌'].unique()
        result = {}
        for brand in brands:
            brand_data = group[group['预算品牌'] == brand]
            result[f'{brand}预算品牌月总销额'] = brand_data['客户-品牌月总销额'].sum()
            result[f'{brand}预算品牌历史总销额'] = brand_data['客户-品牌历史总销额'].sum()
            result[f'{brand}预算品牌有效月份个数'] = brand_data['品牌有效月份个数'].sum()
            total_weight = brand_data['标准重量'].sum()
            if total_weight > 0:
                weighted_tax_price = (brand_data['标准重量'] * brand_data['含税单价']).sum() / total_weight
            else:
                weighted_tax_price = 0
            result[f'{brand}加权含税单价'] = weighted_tax_price
        return pd.Series(result)
    customer_brand_columns = grouped_customer.apply(create_brand_columns).reset_index()
    customer_result = pd.merge(customer_rolling_sums, customer_brand_columns, on='客户编码', how='left')
    customer_result = customer_result.reset_index()
    customer_result.to_csv("data/客户编码特征.csv")
    # print(customer_result.columns)
    return customer_result.fillna(0.0)





class NormalizeModule:
    def __init__(self,save_path):
        self.min_values = None
        self.max_values = None
        self.save_path = save_path

    def fit_normalize(self, df_column):
        data = np.array(df_column.tolist())
        self.min_values = np.min(data, axis=0)
        self.max_values = np.max(data, axis=0)
        normalized_data = (data - self.min_values) / (self.max_values - self.min_values)
        self.save_parameters()
        return normalized_data.tolist()

    def transform_normalize(self, df_column):
        if self.min_values is None or self.max_values is None:
            self.load_parameters()
        data = np.array(df_column.tolist())
        normalized_data = (data - self.min_values) / (self.max_values - self.min_values)
        return normalized_data.tolist() 
    
    def save_parameters(self):
        parameters = {
            'min_values': self.min_values,
            'max_values': self.max_values
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(parameters, f)

    def load_parameters(self):
        try:
            with open(self.save_path, 'rb') as f:
                parameters = pickle.load(f)
                self.min_values = parameters['min_values']
                self.max_values = parameters['max_values']
        except FileNotFoundError:
            raise ValueError("尚未保存normalize参数，请先调用fit_normalize方法")

def examine_plan(plan_df, customer_df, sku_df):
    N1 = len(customer_df['客户特征'].iloc[0])
    # 物料特征的维度
    N2 = len(sku_df['物料特征'].iloc[0])
    # obs的长度
    obs_length = N1 + N2
    # 定义函数：根据客户编码和物料编码提取特征并拼接成大向量
    def extract_features(row, customer_df, sku_df):
        # 提取客户特征
        customer_features = customer_df.loc[customer_df['客户编码'] == row['客户编码'], '客户特征'].iloc[0]
        # 提取物料特征
        sku_features = sku_df.loc[sku_df['物料编码'] == row['物料编码'], '物料特征'].iloc[0]
        # 拼接成大向量
        obs = np.concatenate((customer_features, sku_features))
        return obs
    # 应用函数，生成obs字段
    plan_df = pd.merge(plan_df, sku_df[['物料编码','成本价']], on='物料编码',how='left')
    plan_df['obs'] = plan_df.apply(lambda row: extract_features(row, customer_df, sku_df), axis=1)
    # 添加output字段，值全为0
    plan_df['output'] = [[0] for _ in range(len(plan_df))]
    save_path = "models/mock_model_save_path"
    test_data_path = "data/mock_test_data.parquet"  
    test_df = plan_df
    test_df.to_parquet(test_data_path, index=False)
    output_path = "data/mock_predictions.parquet"
    predict_selling(test_data_path, save_path, inference_num=len(test_df), output_path=output_path, state_dim=obs_length, dim_action=1)
    real_selling = pd.read_parquet(output_path)
    real_selling['实际销额'] = real_selling['predict_output_0']
    
    # 评估，包含约束违反程度，总利润，总销额
    real_selling['销售收入'] = real_selling['实际销额'] * real_selling['单价']
    real_selling['利润'] = real_selling['实际销额'] * (real_selling['单价'] - real_selling['成本价'])
    # 计算总销额和总利润
    total_sales = real_selling['销售收入'].sum()
    total_profit = real_selling['利润'].sum()
    eval_result = {
        "total_sales":total_sales,
        "total_profit":total_profit
    }
    return eval_result
