import torch
import torch.nn as nn

from .vn_attn import VNSensorAttention, VNEncoderLayer, VNAttentionWithContext
from .vn_layers import VNLinearLeakyReLU, VNStdFeature
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
        self.conv1 = VNLinearLeakyReLU(self.input_filters, self.nb_units, batch_norm=self.batch_norm, dim=5, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(self.nb_units, 1, batch_norm=self.batch_norm, dim=5)

    def forward(self, x):
        # x: (B, 1, L, 3, C)
                                              # (B, C, L, 3, 1)
        out = self.conv1(x.transpose(1, -1))  # (B, C, L, 3, nb_units)

        out = self.conv2(out)                 # (B, C, L, 3, 1)

        return out.transpose(1, -1)           # (B, 1, L, 3, C)


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
        self.conv1d = VNLinearLeakyReLU(input_shape[3]//3, self.nb_units, share_nonlinearity=True, negative_slope=0.0)  # share_nonlinearity for Conv1D

        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

        self.VNAttentionWithContext = VNAttentionWithContext(self.nb_units, attn_act_fn)

        self.std_feature = VNStdFeature(self.nb_units, dim=3, normalize_frame=False)

        # classifier - VN not used from here on
        self.fc = nn.Linear(self.nb_units*3, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):
        '''
        x: [B, 1, L, 3(xyz), C]     where   C = self.nb_units
        '''
        x = vn_c_reshape(x, self.time_length)                 # (B, 1, L, 3, C);
      
        # conv on f_in  
        x = self.first_conv(x)                                # (B, 1, L, 3, C)
        x = x.squeeze(1)                                      # (B, L, 3, C)

        si, _ = self.VNSensorAttention(x)                     # (B, L, 3, C)

        # conv on C
        x = self.conv1d(si)                                   # (B, L, 3, C)
        
        x = self.VNEncoderLayer1(x)                           # (B, L, 3, C)
        x = self.VNEncoderLayer2(x)                           # (B, L, 3, C)
        
        x = self.VNAttentionWithContext(x)                    # (B, 3, C)
       
        batch_temp = x.size(0)
        # x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # x = torch.cat((x, x_mean), 2)                       # if using, (B, 3, 2*C)
       
        # VN Invariant Layer
        x, trans = self.std_feature(x)                        # (B, C, 3) 
       
        # x = x.view(self.batch_size, -1, N)
        x = x.view(batch_temp, -1)                            # (B, C*3)
        
        x = self.dropout(self.relu(self.fc(x)))               # (B, 4*nb_classes)
        
        out = self.fc_out(x)                                  # (B, nb_classes) 
       
        return out