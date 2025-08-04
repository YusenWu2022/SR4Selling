import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np

class V(nn.Module):
    '''
    prediction net
    '''

    def __init__(self, dim_observation, dim_actions):
        super(V, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 1024)
        self.FC2 = nn.Linear(1024, 4096)
        self.dropout = nn.Dropout(p=0.5) 
        self.FC3 = nn.Linear(4096, 2048)
        self.FC4 = nn.Linear(2048, 256)
        self.FC5 = nn.Linear(256, dim_actions)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.dropout(result)
        result = F.relu(self.FC3(result))
        result = F.relu(self.FC4(result))
        return self.FC5(result)

class PD(nn.Module):
    """
    Usage:
    pd = PD(dim_obs=28, dim_action=2)
    pd.load_net(load_path="path_to_saved_model")
    pred_next_obs = pd.take_actions([states, actions])
    """

    def __init__(self, dim_obs, dim_actions, value_model_lr=0.0001, network_random_seed=1, train_iter=3):
        super().__init__()
        self.dim_obs = dim_obs
        self.dim_actions = dim_actions
        self.value_model_lr = value_model_lr
        self.network_random_seed = network_random_seed
        torch.manual_seed(self.network_random_seed)
        self.value_model = V(dim_obs, dim_actions)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.value_model.to(self.device)
        self.train_episode = 0
        self.train_iter=train_iter
        self.optimizer = Adam(self.value_model.parameters(), lr=self.value_model_lr)

    def step(self, obs, output):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        output = torch.tensor(output, dtype=torch.float32, device=self.device)
        loss_list = []
        for _ in range(self.train_iter):
            predicted_output = self.value_model(obs)
            loss = nn.MSELoss()(predicted_output, output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        return np.array(loss_list)

    def make_prediction(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.value_model = self.value_model.to(self.device) 
        with torch.no_grad():
            pred_output = self.value_model(obs)
        pred_output = pred_output.clamp(min=0).cpu().numpy()
        return pred_output

    def save_net_pkl(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pkl")
        torch.save(self.value_model, file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/bc_model.pth')

    def forward(self, states):
        with torch.no_grad():
            actions = self.value_model(states)
        actions = torch.clamp(actions, min=0)
        return actions

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pt")
        torch.save(self.value_model.state_dict(), file_path)

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        file_path = os.path.join(load_path, "bc.pt")
        self.value_model.load_state_dict(torch.load(file_path, map_location=device))
        self.value_model.to(self.device)
        print(f"Model loaded from {self.device}.")

    def load_net_pkl(self, load_path="saved_model/fixed_initial_budget"):
        file_path = os.path.join(load_path, "bc.pkl")
        self.value_model = torch.load(file_path, map_location=self.device)
        self.value_model_optimizer = Adam(self.value_model.parameters(), lr=self.value_model_lr)
        self.value_model.to(self.device)
        print(f"Model loaded from {self.device}.")