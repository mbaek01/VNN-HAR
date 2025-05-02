import torch
import torch.nn as nn

from .vn_attn import VNSensorAttention, VNEncoderLayer, VNAttentionWithContext
from .vn_layers import VNLinear, VNLeakyReLU, VNLinearLeakyReLU, VNStdFeature
from utils import vn_c_reshape

class VNConvBlock(nn.Module):
    """
    Vector-Neuron Replacement for ConvBlock
    """
    def __init__(self, filter_width, input_filters, nb_units, batch_norm):
        super(VNConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_units = nb_units
        self.batch_norm = batch_norm

        # TODO: test with share_nonlinearity 
        self.conv1 = VNLinearLeakyReLU(self.input_filters, self.nb_units, batch_norm=self.batch_norm, dim=5)
        self.conv2 = VNLinearLeakyReLU(self.nb_units, 1, batch_norm=self.batch_norm, dim=5)
        # if self.batch_norm:
        #     self.norm1 = nn.BatchNorm2d(self.nb_units)
        #     self.norm2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x: (B, 1, L, 3, C)
        out = self.conv1(x)       # (B, self.nb_units, L, 3, C)

        out = self.conv2(out)     # (B, 1, L, 3, C)

        return out                # (B, 1, L, 3, C)


class VN_SA_HAR(nn.Module):
    def __init__(self, input_shape, nb_classes, nb_units, attn_act_fn = "vn_leaky_relu"):
        super().__init__()
        self.batch_size = input_shape[0]  # B   
        self.time_length = input_shape[2] # L
        self.nb_units = nb_units // 3

        self.first_conv = VNConvBlock(filter_width=5, 
                            input_filters=input_shape[1], # f_in
                            nb_units=self.nb_units, 
                            batch_norm=True).double()
        
        self.VNSensorAttention = VNSensorAttention(input_shape, self.nb_units)
        self.conv1d = VNLinearLeakyReLU(input_shape[3]//3, self.nb_units, negative_slope=0.0)

        self.vn_act = VNLeakyReLU(self.nb_units, share_nonlinearity=False, negative_slope=0.0)

        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

        self.VNAttentionWithContext = VNAttentionWithContext(self.nb_units, attn_act_fn)

        self.std_feature = VNStdFeature(self.nb_units*2, dim=3, normalize_frame=False)

        # classifier - VN not used from here on
        self.fc2 = nn.Linear(self.nb_units*6, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):
        '''
        x: [B, 1, L, 3(xyz), C] ; C = C//3
        '''
        x = vn_c_reshape(x, self.time_length)                 # (B, 1, L, 3, C);
      
        x = self.first_conv(x) # conv on f_in                 # (B, 1, L, 3, C)
        x = x.squeeze(1)                                      # (B, L, 3, C)

        si, _ = self.VNSensorAttention(x)                     # (B, L, 3, C)

        x = self.conv1d(si.transpose(1, -1)).transpose(1, -1) # conv on C
        
        x = self.VNEncoderLayer1(x)
        # Shape: (B, L, 3, C); C = nb_units // 3
        x = self.VNEncoderLayer2(x)
        # Shape: (B, L, 3, C); C = nb_units // 3
        x = self.VNAttentionWithContext(x)
        # Shape: (B, L, 3, C); C = nb_units // 3

        # N = x.size(-1)
        batch_temp = x.size(0)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 2)
        # Shape: (B, 3, 2*C); C = nb_units // 3

        x, trans = self.std_feature(x)
        # Shape: (B, 2*C, 3)

        x = x.view(batch_temp, -1) # x = x.view(self.batch_size, -1, N)
        # Shape: (B, 2*C*3)

        x = self.dropout(self.relu(self.fc2(x)))
        # Shape: (B, 4*nb_classes)

        out = self.fc_out(x)
        # Shape: (B, nb_classes)

        return out