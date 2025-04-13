import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn()

        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, out_dim)

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(batch, length, channel)
        '''
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.bn1(self.fc1(x)))
        x = self.activation_fn(self.bn2(self.fc2(x)))
        x = self.activation_fn(self.bn3(self.fc3(x)))
        out = self.fc4(x)  
        return out

