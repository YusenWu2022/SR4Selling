import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from predictDNN import PD

def train_prediction_model(train_data_path, save_path, step_num, batch_size, state_dim, dim_action):
    # 读取训练数据（包含obs和output两列）
    training_data = pd.read_parquet(train_data_path)
    writer = SummaryWriter(save_path+'/log')
    
    # 初始化经验回放池
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data)
    
    print("Total training samples: "+str(len(replay_buffer)))

    def pd_train_steps(model, replay_buffer, step_num=100000, batch_size=100):
        for i in tqdm(range(step_num)):
            if i%2000 == 0:
                model.save_net(save_path+"/ckpt/"+str(i))
                
            # 从回放池采样并拆分特征
            obs_batch, output_batch = replay_buffer.sample(batch_size)
            
            # 模型训练步骤
            loss = model.step(obs_batch, output_batch)
            
            if i%1000 == 0:
                writer.add_scalar('Prediction loss', np.mean(loss), i)

    # 初始化预测模型（保持原始输入输出维度不变）
    model = PD(
        dim_obs=state_dim,   # 客户特征维度
        dim_actions=dim_action   # 输出维度
    )
    
    pd_train_steps(model, replay_buffer, step_num, batch_size)
    model.save_net(save_path)
    writer.close()

def add_to_replay_buffer(replay_buffer, training_data):
    """将原始数据转换为新的存储格式"""
    for row in training_data.itertuples():
        # 存储到回放池
        replay_buffer.push(
            obs=row.obs,   # 客户特征
            output=row.output     # 销售结果（支持多维）
        )

if __name__ == "__main__":
    file_name = "selling_predict_1103"
    data_path = "/root/pku/item_choice/data/"+file_name
    train_data_path = data_path+'/predict_table.parquet'
    save_path = data_path+"/pd-10270-model"
    
    # 参数设置（需要根据实际特征维度调整）
    step_num = 10000
    batch_size = 256
    original_obs_dim = 27    # 客户特征维度
    state_dim = original_obs_dim   # 客户特征维度
    dim_action = 1           # 输出维度
    
    train_prediction_model(
        train_data_path,
        save_path,
        step_num,
        batch_size,
        state_dim,
        dim_action
    )