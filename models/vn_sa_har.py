import torch
import torch.nn as nn

from .vn_attn import VNEncoderLayer, VNAttentionWithContext
from .vn_layers import VNLeakyReLU, VNStdFeature
from utils import vn_c_reshape

class VN_SA_HAR(nn.Module):
    def __init__(self, input, nb_classes, nb_units, attn_act_fn = "leaky_relu"):
        super().__init__()
        self.batch_size = input[0]
        self.time_length = input[2]
        self.channel = input[3]
        self.nb_units = nb_units // 3

        self.fc1 = nn.Linear(self.channel // 3, self.nb_units, bias=False) # equivalent to VNLinear
        self.vn_act = VNLeakyReLU(self.nb_units, share_nonlinearity=False, negative_slope=0.0)

        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

        self.VNAttentionWithContext = VNAttentionWithContext(self.nb_units, attn_act_fn)

        self.std_feature = VNStdFeature(self.nb_units*2, dim=3, normalize_frame=False)

        self.fc2 = nn.Linear(self.nb_units*6, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):
        x = vn_c_reshape(x, self.batch_size, self.time_length)

        x = self.vn_act(self.fc1(x))
        
        x = self.VNEncoderLayer1(x)
        x = self.VNEncoderLayer2(x)

        x = self.VNAttentionWithContext(x)

        N = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 2)
        x, trans = self.std_feature(x)
        x = x.view(self.batch_size, -1) # x = x.view(self.batch_size, -1, N)
        

        x = self.dropout(self.relu(self.fc2(x)))
        out = self.fc_out(x)

        return out