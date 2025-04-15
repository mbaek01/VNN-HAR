import torch.nn as nn

# class Baseline(nn.Module):
#     def __init__(self, in_dim, out_dim, activation_fn):
#         super().__init__()
#         self.activation_fn = activation_fn()

#         self.fc1 = nn.Linear(in_dim, 1024)
#         self.bn1 = nn.LayerNorm(1024)

#         self.fc2 = nn.Linear(1024, 512)
#         self.bn2 = nn.LayerNorm(512)

#         self.fc3 = nn.Linear(512, 64)
#         self.bn3 = nn.LayerNorm(64)

#         self.fc4 = nn.Linear(64, out_dim)

#     def forward(self, x):
#         '''
#         :param x: torch.Tensor
#                 Shape(batch, length, channel)
#         '''
#         # x = x.view(x.size(0), -1)
#         x = self.activation_fn(self.bn1(self.fc1(x)))
#         x = self.activation_fn(self.bn2(self.fc2(x)))
#         x = self.activation_fn(self.bn3(self.fc3(x)))
#         out = self.fc4(x)  
#         return out

class Baseline(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn()

        self.fc1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, out_dim)

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(batch, length, channel)
        '''
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.bn1(self.fc1(x)))
        x = self.activation_fn(self.bn2(self.fc2(x)))
        x = self.activation_fn(self.bn3(self.fc3(x)))
        x = self.activation_fn(self.bn4(self.fc4(x)))
        out = self.fc5(x)  
        return out

# temporal, spatial attention
# TInyHar - temporal dimension

