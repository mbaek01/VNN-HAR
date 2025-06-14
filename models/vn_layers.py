import torch
import torch.nn as nn
import torch.nn.functional as F 

EPS = 1e-6

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False) # bias interferes with the equivariance
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # x_out = self.map_to_feat(x)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        # d = self.map_to_dir(x)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, dim=4, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.batch_norm = batch_norm
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

        if self.batch_norm: 
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # p = self.map_to_feat(x)

        # BatchNorm
        if self.batch_norm:
            p = self.batchnorm(p)

        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1) # k
        # d = self.map_to_dir(x)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...] 
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    '''
    TODO: Dimensions need to be adjusted
    '''
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        # d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        d = self.map_to_dir(x)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, batch_norm=True, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, batch_norm=True, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x                                                  
        z0 = self.vn1(z0)                                      
        z0 = self.vn2(z0)                                       
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1) 
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)  # z0 : [B, 3(hidden), 3, ...] to [B, 3, 3(hidden), ...]
            
        '''
            j = Initial x,y,z vector dimension of 3 
            k = Dimension D linearly transformed to 3
        '''
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
            # Shape:
            # x:  (B, D, 3, L)
            # z0: (B, 3, 3, L)
            # ->  (B, D, 3, L)
            
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
            # Shape: 
            # x:  (B, D, 3),  
            # z0: (B, 3, 3) 
            # ->  (B, D, 3)

        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
            # Shape:
            # x:  (B, 2*D, 3, L, H) 
            # z0: (B, 3, 3, L, H) 
            # ->  (B, 2*D, 3, L, H)
        
        return x_std, z0
    

class VNLayerNorm(nn.Module):
    def __init__(self, d_features, eps=1e-6, affine=True, bias=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.bias = bias
        if self.affine:
            # gamma: learnable scale for each feature (C)
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, d_features))  # shape (1,1,1,C)
        if self.bias:
            # Not used for equivariance, but included for completeness
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, d_features))  # shape (1,1,1,C)

    def forward(self, x):
        # x: (B, L, 3, C)
        # Compute the norm of each 3D vector (over dim=2)
        norms = torch.norm(x, dim=2, keepdim=True)  # (B, L, 1, C)
        
        # Compute mean and std over C (feature dimension)
        mean = norms.mean(dim=3, keepdim=True)      # (B, L, 1, 1)
        std = norms.std(dim=3, keepdim=True)        # (B, L, 1, 1)
        
        # Normalize norms
        normed_norms = (norms - mean) / (std + self.eps)  # (B, L, 1, C)

        # Apply scale (and optional bias)
        if self.affine:
            normed_norms = normed_norms * self.gamma
        if self.bias:
            normed_norms = normed_norms + self.beta

        # Rescale original vectors to have normalized norms
        safe_norms = norms + self.eps  # avoid division by zero
        x_normed = x / safe_norms * normed_norms  # (B, L, 3, C)
        return x_normed