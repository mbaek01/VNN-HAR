import torch
import torch.nn as nn

from .model import SA_HAR
from .vn_attn import VNSensorAttention, VNEncoderLayer, VNAttentionWithContext
from .vn_layers import VNLinearLeakyReLU, VNStdFeature, get_graph_feature_cross
from .vn_dgcnn import VNDGCNNEncoder
from utils import vn_c_reshape

# TODO: input shape to [B, C, 3, L, ...]
class VNConvBlock(nn.Module):
    """
    Vector-Neuron Replacement for ConvBlock
    """
    def __init__(self, input_filters, nb_units, batch_norm):
        super(VNConvBlock, self).__init__()
        # self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_units = nb_units
        self.batch_norm = batch_norm

        self.conv1 = VNLinearLeakyReLU(self.input_filters, self.nb_units, batch_norm=self.batch_norm, dim=5, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(self.nb_units, 1, batch_norm=self.batch_norm, dim=5)

    def forward(self, x):
        # x: (B, C, L, 3, D)
                                              # (B, C, L, 3, 1)
        out = self.conv1(x.transpose(1, -1))  # (B, C, L, 3, nb_units)

        out = self.conv2(out)                 # (B, C, L, 3, 1)

        return out.transpose(1, -1)           # (B, 1, L, 3, C)


class VN_SA_HAR(nn.Module):
    def __init__(self, input_shape, nb_classes, nb_units, num_neighbors):
        super().__init__()
        self.batch_size = input_shape[0]  # B   
        self.time_length = input_shape[2] # L
        self.nb_units = nb_units // 3
        self.num_neighbors = num_neighbors

        self.first_conv = VNConvBlock(# filter_width=5, 
                                      input_filters=3, # f_in of invariant feature from get_graph_feature_cross
                                      nb_units=self.nb_units, 
                                      batch_norm=True).double()
        
        self.VNSensorAttention = VNSensorAttention(input_shape, self.nb_units)
        self.conv1d = VNLinearLeakyReLU(input_shape[3]//3, self.nb_units, share_nonlinearity=True, negative_slope=0.0)  # share_nonlinearity for Conv1D

        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

        self.VNAttentionWithContext = VNAttentionWithContext(self.nb_units)
        self.std_feature = VNStdFeature(self.nb_units, dim=3, normalize_frame=False)

        # classifier - VN not used from here on
        self.fc = nn.Linear(self.nb_units*3, 4*nb_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):
        '''
        x: [B, 1, L, D] -> [B, 1, L, 3(xyz), D]    
        and C -> self.nb_units
        '''
        x = vn_c_reshape(x, self.time_length)                 # (B, 1, L, 3, D)
        x = get_graph_feature_cross(x, k=self.num_neighbors)  # (B, 3, L, 3, D, num_neighbors)

      
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
    

class VN_With_SA_HAR(nn.Module):
    def __init__(self, input_shape, nb_classes, num_neighbors, f_in, f_out, config):
        super().__init__()
        self.VNDGCNNEncoder = VNDGCNNEncoder(num_neighbors, dims=[f_in, 21, 21, 42, 85, f_out], use_bn=True)

        self.std_feature = VNStdFeature(2*f_out, dim=4, normalize_frame=False)

        self.time_length = input_shape[2]
        input_shape[1] = 2*f_out  # C 
        # input_shape[3] //= 3    # (B, f_out, L, D)

        self.SA_HAR_Classifier = SA_HAR(input_shape, nb_classes, config)

    def forward(self, x):
        x = vn_c_reshape(x, self.time_length)                                     # (B, 1, L, 3, D)
            
        '''
        TODO: D = P*S where 
            P = 3 (position: hand, chest, ankle)
            S = 2 (sensor: acc, gyro)

            Test Variations
            (B, 1, 3, L*D)
            (B, P, 3, L*S)
            (B, S, 3, L*P)
            
        Need to apply changes also in apply_rotation()'s reshaping methods in rotation.py    
        '''

        # Equivariant Lifting Layer
        B, _, L, _, D = x.shape
        x = x.reshape(B, 1, 3, L*D) # TODO: change L*D by the variation

        x = self.VNDGCNNEncoder(x)                              # (B, C, 3, L*D)

        # VN Invariant Layer
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)                           # (B, 2*C, 3, L*D)
    
        x, _ = self.std_feature(x)                              # (B, 2*C, 3, L*D)

        # Classifier
        x = x.reshape(B, -1, L, D*3)
        x = self.SA_HAR_Classifier(x)                       

        return x