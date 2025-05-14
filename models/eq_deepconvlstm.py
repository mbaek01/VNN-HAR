import torch
import torch.nn as nn

from escnn import gspaces, nn as escnn_nn

class EqConvBlock(nn.Module):
    """
    SO(3)-equivariant convolution block using escnn
    """
    def __init__(self, filter_width, input_fields, nb_fields, dilation, batch_norm):
        super().__init__()
        self.filter_width = filter_width
        self.input_fields = input_fields  # Number of input vector fields (e.g., 3 sensors)
        self.nb_fields = nb_fields        # Number of output vector fields ("filters")
        self.dilation = dilation
        self.batch_norm = batch_norm

        # 1. Define SO(3) group action
        self.gspace = gspaces.rot3dOnR3()

        # 2. Define feature types
        self.feat_type_in = escnn_nn.FieldType(self.gspace, self.input_fields * [self.gspace.irrep(1)])
        self.feat_type_out = escnn_nn.FieldType(self.gspace, self.nb_fields * [self.gspace.irrep(1)])

        # 3. First equivariant convolution
        self.conv1 = escnn_nn.R2Conv(
            self.feat_type_in,
            self.feat_type_out,
            kernel_size=self.filter_width,      # only square filters available
            dilation=self.dilation,
            padding="replicate",
        )

        # 4. Equivariant BatchNorm and nonlinearity
        if self.batch_norm:
            self.norm1 = escnn_nn.InnerBatchNorm(self.feat_type_out)
        self.relu = escnn_nn.NormNonLinearity(self.feat_type_out)

        # 5. Second equivariant convolution with stride
        self.conv2 = escnn_nn.R2Conv(
            self.feat_type_out,
            self.feat_type_out,
            kernel_size=(self.filter_width, 1),
            dilation=(self.dilation, 1),
            padding=(self.dilation * (self.filter_width - 1) // 2, 0),
            stride=(2, 1)  # Reduces temporal dimension by half
        )

        if self.batch_norm:
            self.norm2 = escnn_nn.InnerBatchNorm(self.feat_type_out)

    def forward(self, x: torch.Tensor):
        # First conv block
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)
        
        return out.tensor  # Return standard tensor for compatibility


class EqDeepConvLSTM(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                #  filter_scaling_factor,
                 config):
                 #nb_conv_blocks         = 2,
                 #nb_fields              = 64,
                 #dilation               = 1,
                 #batch_norm             = False,
                 #filter_width           = 5,
                 #nb_layers_lstm         = 1,
                 #drop_prob              = 0.5,
                 #nb_units_lstm          = 128):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
        
        """
        super(EqDeepConvLSTM, self).__init__()
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_fields     = int(config["nb_fields"])
        self.dilation       = 1 # config["dilation"]
        self.batch_norm     = bool(config["batch_norm"])
        self.filter_width   = config["filter_width"]
        self.nb_layers_lstm = config["nb_layers_lstm"]
        self.drop_prob      = config["drop_prob"]
        self.nb_units_lstm  = int(config["nb_units_lstm"])
        
        
        self.nb_channels    = input_shape[3] // 3
        self.nb_classes     = nb_classes

    
        self.conv_blocks = []

        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = input_shape[1]
            else:
                input_filters = self.nb_fields
    
            self.conv_blocks.append(EqConvBlock(self.filter_width, input_filters, self.nb_fields, self.dilation, self.batch_norm))

        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        
        # define lstm layers
        self.lstm_layers = []
        for i in range(self.nb_layers_lstm):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_fields, self.nb_units_lstm, batch_first =True))
            else:
                self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm, batch_first =True))
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define classifier
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        '''
        x: [B, 1, L, 3(xyz), C]     where   C = self.nb_units
        '''
        # reshape data for convolutions
        x = x.squeeze(1).transpose(1,2)                                           # (B, 3, L, C)

        # Wrap input as GeometricTensor
        x = escnn_nn.GeometricTensor(x, self.feat_type_in)
        
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        final_seq_len = x.shape[2]

        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_fields * self.nb_channels * 3)

        x = self.dropout(x)
        

        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            

        x = x[:, -1, :]
    
        x = self.fc(x)


        return x

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)