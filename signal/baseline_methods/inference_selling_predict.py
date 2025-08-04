import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from predictDNN import PD

def add_to_replay_buffer(replay_buffer, training_data):
    for row in training_data.itertuples():
        obs = np.array(row.obs)
        output = np.array(row.output)
        replay_buffer.push(obs, output)

def load_model(load_path, state_dim, dim_action):
    model = PD(dim_obs=state_dim, dim_actions=dim_action)
    model.load_net(load_path)
    return model   

def predict_selling(test_data_path, model_path, inference_num, output_path, state_dim, dim_action):
    # 加载模型
    model = load_model(model_path, state_dim, dim_action)
    # 读取测试数据
    test_data = pd.read_parquet(test_data_path)
    # 初始化回放池
    test_buffer = ReplayBuffer()
    # 将测试数据添加到回放池
    add_to_replay_buffer(test_buffer, test_data)
    # 从回放池中提取obs和output
    obs, output = test_buffer.sample_from_head(inference_num)
    # 进行预测
    pred_output = model.make_prediction(obs)
    pred_output = np.array(pred_output)
    # 将output转换为numpy数组（确保先移动到CPU）
    output = np.array(output.cpu())  # 修改这里
    # 计算MSE
    concatenated = np.hstack((output[:20], pred_output[:20]))
    non_zero_indices = output != 0
    a_filtered = pred_output[non_zero_indices]
    b_filtered = output[non_zero_indices]
    mse = np.mean(((a_filtered - b_filtered) / b_filtered))
    # print("MSE:", mse)
    # print("Concatenated output and predictions (first 20):")
    # print(concatenated)
    # 将预测结果添加到DataFrame中
    for i in range(pred_output.shape[1]):
        test_data[f"predict_output_{i}"] = pred_output[:, i]
    # 保存预测结果到Parquet文件
    test_data.to_parquet(output_path, index=False)

if __name__ == "__main__":
    file_name = "new"
    data_path = "/root/pku/item_choice/data/"+file_name
    output_path = data_path+"/selling_predict.parquet"
    test_data_path = data_path+'/predict_table.parquet'
    model_path = data_path+"/pd-10270-model/ckpt/0/"
    
    inference_num = 10812
    state_dim = 3  # '客户-品牌月总销额', '客户-品牌历史总销额', '品牌有效月份个数'
    dim_action = 2  # '月平均销额', '物料进货月份个数'
    predict_selling(test_data_path=test_data_path, model_path=model_path, inference_num=inference_num, output_path=output_path, state_dim=state_dim, dim_action=dim_action)