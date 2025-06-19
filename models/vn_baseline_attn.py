import torch
import torch.nn as nn

from .vn_attn import VNEncoderLayer, VNAttentionWithContext, VNAttentionWithContext2
from .vn_inv_attn import VNInvEncoderLayer, VNInvAttentionWithContext
from .vn_layers import VNLinear, VNLeakyReLU, VNStdFeature
from utils import vn_c_reshape

class VN_Inv_Baseline_Attn(nn.Module):
    def __init__(self, input, nb_classes, nb_units):
        super().__init__()
        self.batch_size = input[0]
        self.time_length = input[2]
        self.channel = input[3] // 3    # Dimension C
        self.nb_units = nb_units // 3   # Dimension D

        # Initial FC Layer
        self.fc1 = VNLinear(self.channel, self.nb_units)
        self.vn_act = VNLeakyReLU(self.nb_units, share_nonlinearity=False, negative_slope=0.0)

        # Self-Attention Encoder
        self.VNInvEncoderLayer1 = VNInvEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4*2)
        self.VNInvEncoderLayer2 = VNInvEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4*2)

        # Global Temporal Attention
        self.VNInvAttentionWithContext = VNInvAttentionWithContext(self.nb_units)

        # Final Invariant Layer
        self.std_feature = VNStdFeature(self.nb_units, dim=3, normalize_frame=False)

        # MLP Classifier
        self.fc2 = nn.Linear(3*self.nb_units, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)
    
    def forward(self, x):                                                # (B, 1, L, C)

        # reshape C -> 3, C//3
        x = vn_c_reshape(x, self.time_length).squeeze(1).transpose(1,-1) # (B, C//3, 3, L)

        # Initial fc layer
        x = self.vn_act(self.fc1(x))                                     # (B, D, 3, L)

        # Self-Attention Encoder
        x = self.VNInvEncoderLayer1(x)                                   # (B, D, 3, L)
        x = self.VNInvEncoderLayer2(x)                                   # (B, D, 3, L)

        # Global Temporal Attention
        x = self.VNInvAttentionWithContext(x)                            # (B, D, 3)

        # Final Invariant Layer
        batch_temp = x.size(0)
        # x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # x = torch.cat((x, x_mean), 2)
        x, _ = self.std_feature(x)                                       # (B, D, 3)

        # MLP Classifier
        x = x.view(batch_temp, -1)                                       # (B, D*3)
        
        x = self.dropout(self.relu(self.fc2(x)))                         # (B, 4*N); N = num classes

        out = self.fc_out(x)                                             # (B, N)

        return out


class VN_Baseline_Attn(nn.Module):
    def __init__(self, input, nb_classes, nb_units, attn_act_fn = "leaky_relu"):
        super().__init__()
        self.batch_size = input[0]
        self.time_length = input[2]
        self.channel = input[3] // 3
        self.nb_units = nb_units // 3

        # bias=False for VNLinear
        self.fc1 = VNLinear(self.channel, self.nb_units)
        self.vn_act = VNLeakyReLU(self.nb_units, share_nonlinearity=False, negative_slope=0.0)

        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

        self.VNAttentionWithContext = VNAttentionWithContext2(self.nb_units, self.time_length, attn_act_fn)

        self.std_feature = VNStdFeature(self.nb_units, dim=3, normalize_frame=False)

        # classifier - VN not used from here on
        self.fc2 = nn.Linear(self.nb_units*3, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):                                                 # (B, 1, L, C)
        x = vn_c_reshape(x, self.time_length).squeeze(1).transpose(1,-1)  # (B, C//3, 3, L)

        x = self.vn_act(self.fc1(x))                                      # (B, D, 3, L)
        
        x = self.VNEncoderLayer1(x)                                       # (B, D, 3, L)
        x = self.VNEncoderLayer2(x)                                       # (B, D, 3, L)

        x = self.VNAttentionWithContext(x)                                # (B, D, 3)

        # N = x.size(-1)
        batch_temp = x.size(0)
        # x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # x = torch.cat((x, x_mean), 2)
        
        x, trans = self.std_feature(x)                                    # (B, D, 3)

        x = x.view(batch_temp, -1)                                        # (B, D*3)

        x = self.dropout(self.relu(self.fc2(x)))                          # (B, 4*N); N = num classes

        out = self.fc_out(x)                                              # (B, N)

        return out