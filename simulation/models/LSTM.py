import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_size = configs.hidden_dims

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = torch.nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            torch.nn.ReLU())
        self.linear = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x, x_temp):
        h1, _ = self.lstm(x.unsqueeze(-1))
        x_temp = x_temp.unsqueeze(-1)
        assert x_temp.shape[1] <= h1.shape[1]
        fc_input = torch.cat([h1[:, -x_temp.shape[1]:, :], x_temp], axis=-1)
        h2 = self.fc(fc_input)
        output = self.linear(h2)
        return output[:,:,-1]
    