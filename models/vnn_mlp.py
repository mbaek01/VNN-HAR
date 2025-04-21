import torch
import torch.nn as nn
import torch.nn.functional as F
from .vn_layers import VNStdFeature, VNMaxPool, mean_pool, VNLinearLeakyReLU
from utils import vn_c_reshape

class VNN_MLP(nn.Module):
    def __init__(self, batch_size, time_length, channel, num_classes, pooling='mean'):
        super().__init__()
        self.batch_size = batch_size
        self.time_length = time_length
        self.channel = channel
        self.num_classes = num_classes

        self.vn1 = VNLinearLeakyReLU(3, 64//3, dim=4, negative_slope=0.0) # negative_slope=0.2, 0.0
        self.vn2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.vn3 = VNLinearLeakyReLU(128//3, 256//3, dim=4, negative_slope=0.0)

        self.std_feature = VNStdFeature(256//3 * 2, dim=3, normalize_frame=False) # dim=4 if Pooling not used

        self.fc1 = nn.Linear(256//3 * 6, 128)
        self.bn1 = nn.BatchNorm1d(128)
        # self.dp1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, num_classes)

        if pooling == 'max':
            self.pool = VNMaxPool(64//3)

        elif pooling == 'mean':
            self.pool = mean_pool

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(batch, length, channel)
        '''
        # Reshape: (batch, length, channel) -> (batch, channel//3, 3, length)
        x = vn_c_reshape(x, self.batch_size, self.time_length)
        x = x.permute(0,2,3,1)

        # VN layers
        x = self.vn1(x)
        x = self.pool(x)
        x = self.vn2(x)
        x = self.vn3(x)

        # Invariant layer
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(self.batch_size, -1, num_points)

        # FC layers
        x = x.view(256, 256//3 * 6) # shape: (batch, hidden_dim, 3) -> batch, hidden_dim*3 ? 
        x = F.relu(self.bn1(self.fc1(x))) 
        x = F.relu(self.bn2(self.fc2(x)))

        out = self.fc3(x)
        
        return out


## pooling 사용 유무 및 위치
## baseline 과 비교시 layer-wise or parameter-wise? 
##