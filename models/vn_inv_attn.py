import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vn_layers import VNLeakyReLU, VNStdFeature, VNLinear

class VNInvAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(VNInvAttentionLayer, self).__init__()

        # bias=False for VNLinear
        self.query = VNLinear(d_model, d_model)                                                     
        self.key = VNLinear(d_model, d_model)                                    
        self.value = VNLinear(d_model, d_model)                                 
        self.out = VNLinear(2*d_model, d_model)

        self.n_heads = n_heads

        self.q_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)
        self.k_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)
        self.v_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)

    def forward(self, queries, keys, values):
        B, L, _, _ = queries.shape 
        H = self.n_heads

        queries = self.query(queries).view(B, L, 3, H, -1)                      # (B, L, 3, H, D)
        keys = self.key(keys).view(B, L, 3, H, -1)                              # (B, L, 3, H, D) 
        values = self.value(values).view(B, L, 3, H, -1)                        # (B, L, 3, H, D) 

        # Invariant query, key, values
        queries_mean = queries.mean(dim=1, keepdim=True).expand(queries.size())
        queries = torch.cat((queries, queries_mean), -1)

        keys_mean = keys.mean(dim=1, keepdim=True).expand(keys.size())
        keys = torch.cat((keys, keys_mean), -1)         

        values_mean = values.mean(dim=1, keepdim=True).expand(values.size())
        values = torch.cat((values, values_mean), -1)                                                   

        queries_inv, _ = self.q_std_feature(queries)                            # (B, L, 3, H, 2*D)
        keys_inv, _ = self.k_std_feature(keys)                                  # (B, L, 3, H, 2*D)
        values_inv, _ = self.v_std_feature(values)                              # (B, L, 3, H, 2*D)

        # Attention Score Applied
        score = torch.einsum("blvhd, bsvhd->bvhls", queries_inv, keys_inv)      # (B, 3, H, L, L)
        _, _, _, _, D = queries_inv.shape
        scale = 1./math.sqrt(D)
        Attn = torch.softmax(scale * score, dim=-1)
        V = torch.einsum("bvhls,bsvhd->blvhd", Attn, values_inv).contiguous()   # (B, L, 3, H, 2*D)

        # Merge heads
        out = V.view(B, L, 3, -1)                                               # (B, L, 3, 2*D)
        out = self.out(out)
        return out, Attn
    

class VNInvEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(VNInvEncoderLayer, self).__init__()

        self.attention = VNInvAttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)    
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = VNLinear(d_model, d_ff)
        self.vn_relu = VNLeakyReLU(d_ff, share_nonlinearity=False, negative_slope=0.0)
        self.ffn2 = VNLinear(d_ff, d_model)
         
        self.layernorm2 = nn.LayerNorm(d_model)               
        
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        x shape: (B, L, 3, D)   where   D = d_model =nb_units // 3
        '''
        attn_output, attn = self.attention( x, x, x )
        attn_output = self.dropout1(attn_output)             # (B, L, 3, D)

        # Residual Connection with LayerNorm
        out1  = self.layernorm1(x + attn_output)

        ffn_output = self.ffn2(self.vn_relu(self.ffn1(out1)))
        ffn_output = self.dropout2(ffn_output)               # (B, L, 3, D)

        # Residual Connection with LayerNorm
        out2 = self.layernorm2(out1 + ffn_output)            # (B, L, 3, D)

        return out2 


class VNInvAttentionWithContext(nn.Module):
    def __init__(self, hidden_dim, act_fn="vn_leaky_relu"):
        super(VNInvAttentionWithContext, self).__init__()

        self.fc1 = VNLinear(hidden_dim, hidden_dim)  
        
        if act_fn == "vn_leaky_relu":
            self.activation = VNLeakyReLU(hidden_dim, share_nonlinearity=False, negative_slope=0.2)
        elif act_fn == "vn_relu":
            self.activation = VNLeakyReLU(hidden_dim, share_nonlinearity=False, negative_slope=0.0)
        else:
            raise NotImplementedError
        
        # Invariant Layer for attn_weights
        self.attn_std_feature = VNStdFeature(2*hidden_dim, dim=4, normalize_frame=False)
        
        self.fc2 = VNLinear(2*hidden_dim, 1)

    def forward(self, x):
        # context = x[:, :-1, :]
        # last = x[:, -1, :]
        '''
        x shape: (B, L, 3, D)   where   D = d_model =nb_units // 3
        '''
        uit = self.activation(self.fc1(x))                                          # (B, L, 3, D)

        # Invariant Layer 
        uit_mean = uit.mean(dim=1, keepdim=True).expand(uit.size())
        uit = torch.cat((uit, uit_mean), -1) 

        inv_uit, _ = self.attn_std_feature(uit)                                     # (B, L, 3, 2*D)
       
        ait = self.fc2(inv_uit)                                                     # (B, L, 3, 1)
       
        # Applying Attention Weight 
        attn_weights = F.softmax(ait, dim=1).transpose(-1, -3)                      # (B, 1, 3, L)

        out = torch.einsum('bsik,bkin->bin', attn_weights, x).squeeze(-2) # + last  # (B, 3, D)

        return out
