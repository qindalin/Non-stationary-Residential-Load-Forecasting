import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.hidden_size = configs.hidden_dims
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.fc1 = torch.nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(
            nn.Linear(self.pred_len, self.hidden_size),
            torch.nn.ReLU())
        self.fc3 = torch.nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            torch.nn.ReLU())
        self.fc4 = torch.nn.Sequential(
            nn.Linear(self.hidden_size, int(0.5 * self.hidden_size)),
            torch.nn.ReLU())
        self.linear = nn.Linear(32, self.pred_len)
    
    def forward(self, x, x_temp):
        h1 = self.fc1(x)
        h2 = self.fc2(x_temp)
        h3 = self.fc3(torch.concat([h1, h2], axis=-1))
        h4 = self.fc4(h3)
        output = self.linear(h4)
        return output
