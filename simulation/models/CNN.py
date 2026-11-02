import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.hidden_size = configs.hidden_dims
        self.kernel_size = 5
        self.conv1d = nn.Conv1d(1, self.hidden_size, kernel_size=self.kernel_size, padding='same')
        self.fc = torch.nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            torch.nn.ReLU())
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x, x_temp):
        h1 = self.conv1d(x.unsqueeze(-1).permute(0, 2, 1))
        h1 = h1.permute(0, 2, 1)
        x_temp = x_temp.unsqueeze(-1)
        assert x_temp.shape[1] <= h1.shape[1]
        fc_input = torch.cat([h1[:, -x_temp.shape[1]:, :], x_temp], axis=-1)
        h2 = self.fc(fc_input)
        output = self.linear(h2)
        return output[:,:,-1]