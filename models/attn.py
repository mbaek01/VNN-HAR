import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection   = nn.Linear(d_model, d_model, bias=True)
        self.value_projection = nn.Linear(d_model, d_model, bias=True)
        self.out_projection   = nn.Linear(d_model, d_model, bias=True)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape

        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        _, _, _, E = queries.shape
        scale = 1./math.sqrt(E)
        Attn = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", Attn, values).contiguous()

        out = V.view(B, L, -1)
        out = self.out_projection(out)
        return out, Attn
    

class EncoderLayer(nn.Module):
    def __init__(self,  d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = AttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)    
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
         
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)               
        
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):

        attn_output, attn = self.attention( x, x, x )
        attn_output = self.dropout1(attn_output)
        out1  = self.layernorm1(x + attn_output)
        # Shape: (B, L, C); C = nb_units = d_model

        ffn_output = self.ffn2(self.relu(self.ffn1(out1)))
        # Shape: (B, L, d_ff) -> (B, L, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 =  self.layernorm2(out1 + ffn_output)
        # Shape: (B, L, d_model)

        return out2

class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim, act_fn="tanh"):
        super(AttentionWithContext, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim) # hidden_dim = nb_units
        
        if act_fn == "tanh":
            self.activation = nn.Tanh() 
        elif act_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError
        
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # context = x[:, :-1, :]
        # last = x[:, -1, :]
        '''
        x : (B, L, C)   where  C = nb_units
        '''
        uit = self.activation(self.fc1(x))                         # (B, L, C) 
                                    
        ait = self.fc2(uit)                                        # (B, L, 1) 
        
        attn_weights = F.softmax(ait, dim=1).transpose(-1, -2)     # (B, 1, L)
       
        out = torch.matmul(attn_weights, x).squeeze(-2) # + last   # (B, C)
        # can also do fc(out) here
        return out


