import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vn_layers import VNLeakyReLU, VNLinearLeakyReLU, VNLinear, VNLayerNorm

class VNSensorAttention(nn.Module):
    def __init__(self, input_shape, nb_units):
        super(VNSensorAttention, self).__init__()
        self.ln = VNLayerNorm(input_shape[3] // 3)        # input_shape[3] = c_in
        
        # TODO: test share_nonlinearlity
        # self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_units, kernel_size=3, dilation=2, padding='same')
        self.conv_1 = VNLinearLeakyReLU(1, nb_units, dim=5, share_nonlinearity=False, negative_slope=0.2) # kernel_size, dilation ?
        # self.conv_f = nn.Conv2d(in_channels=nb_units, out_channels=1, kernel_size=1, padding='same') #  1x1 conv
        self.conv_f = VNLinearLeakyReLU(nb_units, 1, dim=5, share_nonlinearity=False, negative_slope=0.0)
        self.softmax = nn.Softmax(dim=3)

        
    def forward(self, inputs):
        '''
        input: [batch  * length * channel]
        output: [batch, 1, length, d]
        '''
        inputs = self.ln(inputs)        # B, L, 3, C      
             
        x = inputs.unsqueeze(1)         # B, 1, L, 3, C           
        
        x = self.conv_1(x)              # B, nb_units, L, 3, C;  nb_units = 128 // 3
                                    
        x = self.conv_f(x)              # B, 1, L, 3, C     
        
        x = self.softmax(x) # dim=3; C
        x = x.squeeze(1)                # B, L, 3, C        

        return torch.mul(inputs, x), x  # (B, L, 3, C), (B, L , 3, C)
        # batch * channel * len, batch * channel * len 


class VNAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(VNAttentionLayer, self).__init__()

        # bias=False for VNLinear
        self.query = VNLinear(d_model, d_model) 
        self.key = VNLinear(d_model, d_model)
        self.value = VNLinear(d_model, d_model)
        self.out = VNLinear(d_model, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _, _ = queries.shape 
        H = self.n_heads

        queries = self.query(queries).view(B, L, 3, H, -1)
        keys = self.key(keys).view(B, L, 3, H, -1)
        values = self.value(values).view(B, L, 3, H, -1)

        score = torch.einsum("blvhd, bsvhd->bhls", queries, keys)
        _, _, _, _, E = queries.shape
        scale = 1./math.sqrt(E)
        Attn = torch.softmax(scale * score, dim=-1)
        V = torch.einsum("bhls,bsvhd->blvhd", Attn, values).contiguous()

        # Merge heads: (B, L, 3, d_model)
        out = V.view(B, L, 3, -1)
        out = self.out(out)
        return out, Attn
    

class VNEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(VNEncoderLayer, self).__init__()

        self.attention = VNAttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = VNLayerNorm(d_model)    
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = VNLinear(d_model, d_ff)
        self.vn_relu = VNLeakyReLU(d_ff, share_nonlinearity=False, negative_slope=0.0)
        self.ffn2 = VNLinear(d_ff, d_model)
         
        self.layernorm2 = VNLayerNorm(d_model)               
        
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (B, L, 3, C); C = d_model

        attn_output, attn = self.attention( x, x, x )
        attn_output = self.dropout1(attn_output)
        # Shape: (B, L, 3, C); C = d_model

        out1  = self.layernorm1(x + attn_output)

        ffn_output = self.ffn2(self.vn_relu(self.ffn1(out1)))
        ffn_output = self.dropout2(ffn_output)
        # Shape: (B, L, 3, C); C = d_model

        out2 =  self.layernorm2(out1 + ffn_output)
        # (B, L, 3, C)

        return out2 


class VNAttentionWithContext(nn.Module):
    def __init__(self, hidden_dim, act_fn="vn_leaky_relu"):
        super(VNAttentionWithContext, self).__init__()

        self.fc1 = VNLinear(hidden_dim, hidden_dim)  
        
        if act_fn == "vn_leaky_relu":
            self.activation = VNLeakyReLU(hidden_dim, share_nonlinearity=False, negative_slope=0.2)
        elif act_fn == "vn_relu":
            self.activation = VNLeakyReLU(hidden_dim, share_nonlinearity=False, negative_slope=0.0)
        else:
            raise NotImplementedError
        
        self.fc2 = VNLinear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # context = x[:, :-1, :]
        # last = x[:, -1, :]

        # x shape: (B, L, 3, C); C = nb_units // 3

        uit = self.activation(self.fc1(x))
        # uit shape: (B, L, 3, H); H = nb_units // 3

        ait = self.fc2(uit) 
        # ait shape: (B, L, 3, 1)

        attn_weights = F.softmax(ait, dim=1).transpose(-1, -3)
        # shape: (B, 1, 3, L)

        out = torch.einsum('bsik,bkin->bin', attn_weights, x).squeeze(-2) # + last 
        # shape: (B, 3, C); C = nb_units // 3

        # can also do fc(out) here
        return out


