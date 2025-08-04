import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["obs", "output"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self):
        self.memory = []

    def push(self, obs, output):
        """Save an experience tuple"""
        experience = Experience(obs, output)
        self.memory.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        batch = random.sample(self.memory, batch_size)
        obs, output = zip(*batch)
        
        obs = np.stack(obs)
        output = np.stack(output)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return (torch.tensor(obs, dtype=torch.float, device=device),
                torch.tensor(output, dtype=torch.float, device=device))

    def sample_from_head(self, batch_size):
        """Sample from oldest experiences"""
        batch = self.memory[:batch_size]
        obs, output = zip(*batch)
        
        obs = np.stack(obs)
        output = np.stack(output)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return (torch.tensor(obs, dtype=torch.float, device=device),
                torch.tensor(output, dtype=torch.float, device=device))

    def __len__(self):
        """Return the current size of the replay buffer"""
        return len(self.memory)


if __name__ == '__main__':
    # 测试示例
    buffer = ReplayBuffer()
    
    # 添加不同维度的输出测试
    for i in range(1000):
        # obs为3维特征，output为2维动作
        buffer.push(
            obs=np.random.rand(3),          # 3维观察值
            output=np.random.rand(2)       # 2维输出
        )
    
    # 测试采样
    obs_batch, output_batch = buffer.sample(20)
    print("Observation batch shape:", obs_batch.shape)     # 应输出 torch.Size([20, 3])
    print("Output batch shape:", output_batch.shape)       # 应输出 torch.Size([20, 2])