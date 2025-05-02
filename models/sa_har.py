import torch
import torch.nn as nn
from .attn import AttentionWithContext, EncoderLayer

class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_units, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_units = nb_units
        self.dilation = dilation
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_units, (self.filter_width, 1), dilation=(self.dilation, 1),padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_units, 1, (self.filter_width, 1), dilation=(self.dilation, 1), stride=(1,1),padding='same')
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_units)
            self.norm2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x: (B, 1, L, C)
        out = self.conv1(x)       # (B, self.nb_units, L, C)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out) # (B, self.nb_units, L, C)

        out = self.conv2(out)     # (B, 1, L, C)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)

        return out                # (B, 1, L, C)

class SensorAttention(nn.Module):
    def __init__(self, input_shape, nb_units ):
        super(SensorAttention, self).__init__()
        self.ln = nn.LayerNorm(input_shape[3])        # input_shape[3] = c_in
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_units, kernel_size=3, dilation=2, padding='same')
        self.conv_f = nn.Conv2d(in_channels=nb_units, out_channels=1, kernel_size=1, padding='same') #  1x1 conv
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=3)

        
    def forward(self, inputs):
        '''
        input: [batch  * length * channel]
        output: [batch, 1, length, d]
        '''
        inputs = self.ln(inputs)        # B, L, C      
             
        x = inputs.unsqueeze(1)         # B, 1, L, C           
        
        x = self.conv_1(x)              
        x = self.relu(x)                # B, nb_units, L, C;  nb_units = 128 
                                    
        x = self.conv_f(x)              # B, 1, L, C     
        
        x = self.softmax(x) # dim=3; C
        x = x.squeeze(1)                # B, L, C        

        return torch.mul(inputs, x), x  # (B, L, C), (B, L ,C)
        # batch * channel * len, batch * channel * len 


class SA_HAR(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,  
                 config):
        super(SA_HAR, self).__init__()

        self.nb_units     = int(config["nb_units"])

        self.first_conv = ConvBlock(filter_width=5, 
                                    input_filters=input_shape[1], # f_in
                                    nb_units=self.nb_units, 
                                    dilation=1, 
                                    batch_norm=True).double()
        
        self.SensorAttention = SensorAttention(input_shape,self.nb_units)
        self.conv1d = nn.Conv1d(in_channels=input_shape[3], out_channels=self.nb_units, kernel_size=1) # input_shape[3] = c_in
        
        
        #self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(input_shape[2], self.nb_units), requires_grad=False)
        #self.pos_dropout = nn.Dropout(p=0.2) 
        
        self.EncoderLayer1 = EncoderLayer( d_model = self.nb_units, n_heads =4 , d_ff = self.nb_units*4)
        self.EncoderLayer2 = EncoderLayer( d_model = self.nb_units, n_heads =4 , d_ff = self.nb_units*4)


        self.AttentionWithContext = AttentionWithContext(self.nb_units)

        self.fc1 = nn.Linear(self.nb_units, 4*nb_classes)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.2)

        self.fc_out = nn.Linear(4*nb_classes, nb_classes)      

    
    def forward(self,x): 
        # x -- > (B, fin, L, C)
        x = self.first_conv(x)
        x = x.squeeze(1) 
        # x -- > (B, L, C)
	
        # B L C
        si, _ = self.SensorAttention(x) 
        
        # B L C
        x = self.conv1d(si.permute(0,2,1)).permute(0,2,1) 
        x = self.relu(x)            
        # B L C
        #x = x + self.pos_embedding
        #x = self.pos_dropout(x)

        x = self.EncoderLayer1(x)            # batch * len * d_dim
        x = self.EncoderLayer2(x)            # batch * len * d_dim
        
        # Global Temporal Attention
        x = self.AttentionWithContext(x)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc_out(x)
        
        return x
    
    @staticmethod
    def sinusoidal_embedding(length, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(length)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
