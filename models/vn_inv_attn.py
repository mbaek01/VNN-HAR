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
        self.out = VNLinear(d_model, d_model) # 2*d_model

        self.n_heads = n_heads

        # self.q_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)
        # self.k_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)
        # self.v_std_feature = VNStdFeature(2*d_model//n_heads, dim=5, normalize_frame=False)

    def forward(self, queries, keys, values):
        '''
        x: [B, D, 3, L]
        '''
        B, _, _, L = queries.shape 
        H = self.n_heads

        # Embeddings and Reshaping them 
        queries = self.query(queries).permute(0,3,1,2).view(B, L, H, -1, 3)     # (B, L, H, D, 3)                  
        keys = self.key(keys).permute(0,3,1,2).view(B, L, H, -1, 3)
        values = self.value(values).permute(0,3,1,2).view(B, L, H, -1, 3)

        # # Invariant query, key, values with mean concatenation
        # queries_mean = queries.mean(dim=1, keepdim=True).expand(queries.size())    
        # queries = torch.cat((queries, queries_mean), -2)                        # (B, L, H, 2*D, 3)

        # keys_mean = keys.mean(dim=1, keepdim=True).expand(keys.size())
        # keys = torch.cat((keys, keys_mean), -2)         

        # values_mean = values.mean(dim=1, keepdim=True).expand(values.size())
        # values = torch.cat((values, values_mean), -2)                                                   

        # queries_inv, _ = self.q_std_feature(queries.permute(0,3,4,1,2))         # (B, 2*D, 3, L, H)                     
        # keys_inv, _ = self.k_std_feature(keys.permute(0,3,4,1,2))                                  
        # values_inv, _ = self.v_std_feature(values.permute(0,3,4,1,2))                           

        # queries_inv = queries_inv.permute(0,3,4,1,2)                            # (B, L, H, 2*D, 3) 
        # keys_inv = keys_inv.permute(0,3,4,1,2)
        # values_inv = values_inv.permute(0,3,4,1,2)

        # Attention Score Applied
        score = torch.einsum("blhdv, bshdv->bhvls", queries, keys)      # (B, H, 3, L, L)
        _, _, _, _, D = queries.shape
        scale = 1./math.sqrt(D)
        Attn = torch.softmax(scale * score, dim=-1)
        V = torch.einsum("bhvls, bshdv->blvhd", Attn, values).contiguous()  # (B, L, 3, H, D)

        # Merge heads
        out = V.view(B, L, 3, -1)                                               # (B, L, 3, D)
        out = self.out(out.transpose(1,-1))                                     # (B, D, 3, L)
        return out, Attn
    

class VNInvEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(VNInvEncoderLayer, self).__init__()

        self.attention = VNInvAttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm((d_model,3))    
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = VNLinear(d_model, d_ff)
        self.vn_relu = VNLeakyReLU(d_ff, share_nonlinearity=False, negative_slope=0.0)
        self.ffn2 = VNLinear(d_ff, d_model)
         
        self.layernorm2 = nn.LayerNorm((d_model,3))               
        
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        x shape: (B, D, 3, L)   where   D = d_model =nb_units // 3
        '''
        attn_output, attn = self.attention( x, x, x )                                 # (B, D, 3, L)
        attn_output = self.dropout1(attn_output)                                                 

        # Residual Connection with LayerNorm
        out1  = self.layernorm1((x + attn_output).permute(0,3,1,2)).permute(0,2,3,1)  # (B, L, D, 3) -> (B, D, 3, L) permute

        ffn_output = self.ffn2(self.vn_relu(self.ffn1(out1)))                         # (B, D, 3, L)
        ffn_output = self.dropout2(ffn_output)                                                                 
 
        # Residual Connection with LayerNorm
        out2 = self.layernorm2((out1 + ffn_output).permute(0,3,1,2)).permute(0,2,3,1) # (B, L, D, 3) -> (B, D, 3, L) permute

        return out2                                                                   # (B, D, 3, L)


class VNInvAttentionWithContext(nn.Module):
    '''
    Global Temporal Attention from "Human Activity Recognition from Wearable Sensor Data Using Self-Attention" by Mahmud et al.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    '''
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
        
        # TODO: nn.Linear?
        self.fc2 = VNLinear(2*hidden_dim, 1)        

    def forward(self, x):
        # context = x[:, :-1, :]
        # last = x[:, -1, :]
        '''
        x: (B, D, 3, L)   where   D = d_model =nb_units // 3
        '''
        # First FC
        uit = self.activation(self.fc1(x))                                          # (B, D, 3, L)

        # Invariant Layer 
        uit_mean = uit.mean(dim=-1, keepdim=True).expand(uit.size())
        uit = torch.cat((uit, uit_mean), 1)                                         # (B, 2*D, 3, L)

        inv_uit, _ = self.attn_std_feature(uit)                                     # (B, 2*D, 3, L)                                     
       
        # Second FC
        ait = self.fc2(inv_uit)                                                     # (B, 1, 3, L)
       
        # Applying Attention Weight 
        attn_weights = F.softmax(ait, dim=3)                                      

        out = torch.einsum("bsvl, bdvl->bsdv", attn_weights, x).squeeze(1) # + last # (B, D, 3)

        return out
